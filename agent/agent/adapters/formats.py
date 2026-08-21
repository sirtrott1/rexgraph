"""
agent.adapters.formats: readers for file types auto_rex could not open.

Each format is read as the structure it already is, so nothing here is domain
framing. These are containers, the way .csv is a container:

    .sdf .mol      atoms and bonds        -> labeled graph, bond order as edge type
    .pdb           atoms and CONECT       -> labeled graph
    .fasta .fa     sequences              -> k-mer overlap graph
    .vcf           samples vs variants    -> bipartite incidence
    .gff .gtf .bed intervals on a coord   -> overlap graph
    .h5ad .loom    a matrix and its axes  -> the existing feature-matrix path

Parsers use h5py and the standard library. anndata, rdkit and biopython are not
required: these layouts are documented and stable, and taking a hard dependency on
a domain toolkit in order to read a file would be the wrong trade for a library
that is otherwise domain-agnostic.

Readers are registered, so a format is added by registering one rather than by
editing this module or auto_rex's dispatch.
"""

from __future__ import annotations

from fractions import Fraction

#: MDL atom-block charge codes. 0 and 4 are not charges (4 is a doublet radical).
_MDL_CHARGE = {1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}

import gzip
from pathlib import Path

import numpy as np

from rexgraph.registry import Registry

from . import EdgeConstruction

_READERS = Registry("file reader")


def register_reader(name: str, fn, *, extensions=()) -> None:
    """Register a reader. `fn(path, **kw)` returns whatever that format yields."""
    _READERS.register(name, fn, extensions=tuple(e.lower() for e in extensions))


def unregister_reader(name: str):
    return _READERS.unregister(name)


def available_readers() -> list[str]:
    return _READERS.available()


def available_extensions() -> dict[str, str]:
    """extension -> reader name."""
    return {e: name for name in _READERS
            for e in _READERS.meta(name).get("extensions", ())}


def reader_fn(name: str):
    """The registered function for a reader name, so a caller can ask what
    parameters it takes rather than guess which ones to pass through."""
    return _READERS.get(name)


def reader_for(path) -> str | None:
    """The reader registered for `path`'s extension, or None.

    A .gz suffix is transparent: variant and annotation files are usually shipped
    compressed and the inner extension is what names the format.
    """
    p = Path(str(path))
    ext = p.suffix.lower()
    if ext == ".gz":
        ext = Path(p.stem).suffix.lower()
    return available_extensions().get(ext)


def read(path, **kw):
    """Read `path` with whatever reader claims its extension."""
    name = reader_for(path)
    if name is None:
        raise ValueError(
            f"no reader for {Path(str(path)).suffix!r}. Supported: "
            f"{', '.join(sorted(available_extensions()))}")
    return _READERS.require(name)(path, **kw)


def _open_text(path):
    """Open plain or gzipped text without the caller caring which it is."""
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, encoding="utf-8", errors="replace")


def _ec(sources, targets, labels, *, types=None, type_names=None, weights=None,
        aliases=None, origin=""):
    """Assemble an EdgeConstruction from parallel edge arrays.

    `aliases` maps a vertex label to the other identifiers naming the same entity.
    Passing them through is what lets a file be joined to another that knows the same
    entity by a different name.
    """
    src = np.asarray(sources, dtype=np.int32)
    tgt = np.asarray(targets, dtype=np.int32)
    n = src.shape[0]
    types = np.zeros(n, np.int32) if types is None else np.asarray(types, np.int32)
    names = list(type_names or ["default"])
    return EdgeConstruction(
        sources=src, targets=tgt,
        weights=(np.ones(n, np.float64) if weights is None
                 else np.asarray(weights, np.float64)),
        signs=np.ones(n, np.float64),
        type_labels=types,
        vertex_labels=list(labels),
        n_types=max(1, len(names)),
        type_names=names,
        vertex_aliases={k: list(v) for k, v in (aliases or {}).items() if v},
        origin=origin,
    )


#### bonded structure
def _sdf_record(lines: list[str], base: int, tag: str):
    """Parse one MOL record into atom labels and bonds as (src, tgt, order).

    Atom indices are offset by `base` and labels are prefixed with `tag`, which places
    the record on its own vertex block. Returns (None, None) when the record has no
    readable counts line, the shape of the empty chunk after a trailing `$$$$`.
    """
    if len(lines) < 4:
        return None, None, None, None
    counts = lines[3]
    try:
        n_atoms, n_bonds = int(counts[0:3]), int(counts[3:6])
    except ValueError:
        return None, None, None, None

    # The counts line is a claim about the record. Trusting it past the end of the
    # block invented atoms out of the bond table and the M END line, and the real
    # bonds were then read from beyond the file and lost.
    need = 4 + n_atoms + n_bonds
    if need > len(lines):
        raise ValueError(
            f"malformed MOL record {tag.rstrip(':') or '1'}: the counts line declares "
            f"{n_atoms} atoms and {n_bonds} bonds, which needs {need} lines, "
            f"but the record has {len(lines)}")

    labels, coordinates, atom_attrs = [], [], []
    for i in range(n_atoms):
        parts = lines[4 + i].split()
        if len(parts) < 4:
            raise ValueError(
                f"malformed MOL record {tag.rstrip(':') or '1'}: atom line {i + 1} "
                f"is not an atom: {lines[4 + i]!r}")
        labels.append(f"{tag}{parts[3]}{i + 1}")
        # the element is an ATTRIBUTE, not a name. Putting it only in the label means
        # selecting the carbons requires parsing "C12" back apart, and a label is not a
        # schema: "CL1" is chlorine atom 1 or carbon atom L1 depending on who wrote it.
        atom = {"element": parts[3]}
        if len(parts) > 4:
            # MDL charge codes: 1..7 map to +3,+2,+1,0,-1,-2,-3 (4 is a radical)
            code = _MDL_CHARGE.get(parts[4] if not parts[4].isdigit() else int(parts[4]))
            if code is not None:
                atom["formal_charge"] = code
        atom_attrs.append(atom)
        # the atom block carries the EMBEDDING, exactly: an SDF writes four decimal
        # places, so each coordinate is a Fraction over 10^4 and the geometry taken
        # against it stays on the exact tower. Discarding it left the complex with only
        # its intrinsic lengths, which are a function of arity and say nothing about the
        # conformation the file recorded.
        coordinates.append([Fraction(parts[0]), Fraction(parts[1]), Fraction(parts[2])])

    bonds = []
    for j in range(n_bonds):
        k = 4 + n_atoms + j
        if k >= len(lines):
            break
        row = lines[k]
        try:
            a, b, order = int(row[0:3]), int(row[3:6]), int(row[6:9])
        except ValueError:
            parts = row.split()
            if len(parts) < 3:
                continue
            a, b, order = int(parts[0]), int(parts[1]), int(parts[2])
        bonds.append((base + a - 1, base + b - 1, order))   # SDF atom indices are 1-based
    return labels, bonds, coordinates, atom_attrs


def _aromatic_systems(bonds):
    """The connected components of the aromatic bonds, as vertex lists.

    MDL bond order 4 is "aromatic", which is the file saying the electrons are
    DELOCALISED over the ring rather than sitting in alternating pairs. That is a k-way
    relation among the ring atoms, and splitting it into k separate 2-ary bonds is the
    same loss as expanding a hyperedge into a clique: it invents bonds the chemistry does
    not have and dissolves the system's identity as one object.

    Components rather than rings: a fused system like naphthalene is ONE delocalised
    system over ten atoms, not two rings sharing an edge, and the file does not say
    otherwise.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, order in bonds:
        if order == 4:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    systems = {}
    for a, b, order in bonds:
        if order == 4:
            systems.setdefault(find(a), set()).update((a, b))
    return [sorted(vs) for vs in systems.values() if len(vs) > 2]


def load_sdf(path, *, aromatic: str = "branching", **kw) -> EdgeConstruction:
    """MDL SDF/MOL V2000. Atoms are vertices, bonds are relations, bond order is the
    relation type, which is typed-edge information the complex already carries.

    `aromatic="branching"` (the default) reads a delocalised system as ONE relation over
    its atoms, which is what the file means by bond order 4: the electrons are shared
    across the ring, not held in alternating pairs. Benzene comes out as one 6-ary
    relation plus six C-H bonds rather than twelve 2-ary bonds, so the ring is a single
    cell, its boundary column sums to zero over six atoms, and `auto_hyperface` can close
    it into a 2-cell. Drawing it the other way is the crude pairwise picture the complex
    exists to replace.

    `aromatic="pairwise"` keeps every aromatic bond as its own 2-ary relation, which is
    what every reader did before and what a caller comparing against one will want.

    An SDF holds several records separated by `$$$$`. All of them are read into one
    complex. Atom indices are per-record in the format, so each record is offset onto
    its own vertex block and labelled with the record number, leaving the molecules as
    separate components with beta_0 counting them.
    """
    if aromatic not in ("branching", "pairwise"):
        raise ValueError(
            f"aromatic must be 'branching' or 'pairwise', got {aromatic!r}")
    with _open_text(path) as fh:
        text = fh.read()

    labels: list[str] = []
    coordinates: list[list] = []
    src: list[int] = []
    tgt: list[int] = []
    orders: list[int] = []
    branching: list[list[int]] = []
    attributes: dict = {}
    n_rec = 0
    for chunk in text.split("$$$$"):
        lines = chunk.splitlines()
        while lines and not lines[0].strip():          # leading blank after a delimiter
            lines.pop(0)
        tag = "" if n_rec == 0 else f"m{n_rec}:"
        rec_labels, bonds, rec_xyz, rec_attrs = _sdf_record(lines, len(labels), tag)
        if rec_labels is None:
            continue
        for j, values in enumerate(rec_attrs):
            attributes.setdefault(0, {})[len(labels) + j] = values
        labels.extend(rec_labels)
        coordinates.extend(rec_xyz)
        # BOTH grades: the sigma framework stays 2-ary and the delocalised system is
        # added as one k-ary relation over the same atoms. That is the chemistry (a ring
        # has both a bonded framework and a shared pi system) and it is also what makes
        # the ring closable: the wide relation alone bounds nothing, since nothing is
        # enclosed by one cell, so a complex built from the systems alone is a forest of
        # stars with beta_1 = 0 and no face to attach.
        for a, b, o in bonds:
            src.append(a)
            tgt.append(b)
            orders.append(o)
        if aromatic == "branching":
            branching.extend(_aromatic_systems(bonds))
        n_rec += 1

    if not n_rec:
        raise ValueError(f"{path}: no readable SDF/MOL record")

    uniq = sorted(set(orders))
    idx = {o: i for i, o in enumerate(uniq)}
    # Bond order is a magnitude, so it is the edge weight and enters the complex
    # through W. It is also carried as a type, which names the relation. It is not
    # part of the face condition: a ring is a ring whichever bonds close it.
    construction = _ec(src, tgt, labels,
                       types=[idx[o] for o in orders],
                       type_names=[f"bond_order_{o}" for o in uniq] or ["bond"],
                       weights=[float(o) for o in orders] or None)
    # bond order on the relation it belongs to. It is already the relation TYPE, but a
    # type is an index into a name list and an attribute is a value you can compare.
    #
    # The delocalised system gets `relation_kind` and NO bond order, rather than a
    # sentinel under the same key. It is not a bond, so it has no order; and a key whose
    # column holds both numbers and a string is packed as strings by `_pack_cell_metadata`,
    # which coerces every number in it, so `bond_order` came back as "4" after a round
    # trip. Mixing types under one key is what costs the type, not the round trip.
    for e, order in enumerate(orders):
        cell = attributes.setdefault(1, {}).setdefault(e, {})
        cell["bond_order"] = int(order)
        cell["relation_kind"] = "bond"
    for e in range(len(orders), len(orders) + len(branching)):
        attributes.setdefault(1, {}).setdefault(e, {})["relation_kind"] = "delocalised"
    construction.branching = branching
    construction.embedding = coordinates
    construction.attributes = attributes
    return construction


def load_pdb(path, *, backbone: bool = True, **kw) -> EdgeConstruction:
    """PDB ATOM records, with CONECT bonds and the residue chain.

    CONECT alone is not enough. Real PDB files omit it for standard residues, whose
    bonds follow from the residue templates, so 1CA2 gives four bonds for 2207 atoms
    and a complex of isolated points. `backbone` adds the covalent
    structure that is definitional rather than inferred: atoms within a residue,
    and the peptide bond between consecutive residues of a chain.

    Distance-based bond inference is still refused. That is a modelling decision
    with a cutoff in it, and it is not what reading a file means.
    """
    serial_to_idx: dict[int, int] = {}
    labels: list[str] = []
    conect: list[tuple[int, int]] = []
    residues: list[tuple[str, str, int]] = []      # (chain, resseq, atom index)
    atom_attrs: list[dict] = []
    coordinates: list[list] = []
    with _open_text(path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                serial = int(line[6:11])
                name = line[12:16].strip()
                res = line[17:20].strip()
                chain = line[21:22].strip() or "_"
                resseq = line[22:27].strip()
                serial_to_idx.setdefault(serial, len(labels))
                residues.append((chain, resseq, len(labels), rec == "ATOM"))
                # everything the line says, on the atom it says it about. The label keeps
                # three of these joined by colons and the rest were read and dropped: the
                # chain and residue number were parsed for the backbone pass and then
                # discarded, so a file could not be filtered to one chain afterwards.
                atom = {"residue": res, "atom_name": name, "serial": serial,
                        "chain": chain, "resseq": resseq,
                        "record": "ATOM" if rec == "ATOM" else "HETATM"}
                element = line[76:78].strip()
                if element:
                    atom["element"] = element
                for key, lo, hi in (("occupancy", 54, 60), ("b_factor", 60, 66)):
                    text = line[lo:hi].strip()
                    if text:
                        try:
                            atom[key] = float(text)
                        except ValueError:
                            pass
                atom_attrs.append(atom)
                # a PDB carries coordinates in fixed columns, three decimal places, so
                # each is a Fraction over 10^3 and the geometry against it is exact
                try:
                    coordinates.append([Fraction(line[30:38].strip()),
                                        Fraction(line[38:46].strip()),
                                        Fraction(line[46:54].strip())])
                except (ValueError, ZeroDivisionError):
                    coordinates.append([Fraction(0), Fraction(0), Fraction(0)])
                labels.append(f"{res}:{name}:{serial}")
            elif rec == "CONECT":
                nums = [int(line[i:i + 5]) for i in range(6, len(line.rstrip()), 5)
                        if line[i:i + 5].strip().isdigit()]
                for other in nums[1:]:
                    conect.append((nums[0], other))
    bonds_idx: list[tuple[int, int]] = []
    if backbone:
        # Atoms of one residue are bonded to each other through it, and successive
        # residues of a chain through the peptide bond. Both follow from the file,
        # not from a distance cutoff. Built directly in index space: converting to
        # serials and back lost every atom of every model but the last, because an
        # NMR file repeats its serials per MODEL.
        by_res: dict[tuple[str, str], list[int]] = {}
        polymer: dict[tuple[str, str], bool] = {}
        order: list[tuple[str, str]] = []
        for chain, resseq, idx, is_atom in residues:
            key = (chain, resseq)
            if key not in by_res:
                by_res[key] = []
                polymer[key] = is_atom
                order.append(key)
            by_res[key].append(idx)
        for key in order:
            members = by_res[key]
            bonds_idx.extend(zip(members, members[1:], strict=False))
        # A peptide bond joins consecutive POLYMER residues. Waters, ions, ligands
        # and sugars arrive as HETATM and are bonded to nothing by adjacency; the
        # unguarded version chained the waters of any ordinary structure together.
        for prev_key, next_key in zip(order, order[1:], strict=False):
            if prev_key[0] != next_key[0]:
                continue                       # a different chain is not bonded
            if not (polymer[prev_key] and polymer[next_key]):
                continue
            bonds_idx.append((by_res[prev_key][-1], by_res[next_key][0]))

    src, tgt, seen = [], [], set()
    pairs = [(serial_to_idx[a], serial_to_idx[b]) for a, b in conect
             if a in serial_to_idx and b in serial_to_idx] + bonds_idx
    for i, j in pairs:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        src.append(i)
        tgt.append(j)
    if not labels:
        raise ValueError(f"{path}: no ATOM records")
    construction = _ec(src, tgt, labels, type_names=["bond"])
    construction.attributes = {0: dict(enumerate(atom_attrs))}
    construction.embedding = coordinates
    return construction


#### sequences
def load_fasta(path, *, k: int = 5, **kw) -> EdgeConstruction:
    """FASTA as a k-mer overlap graph: consecutive k-mers share k-1 characters, and
    that overlap is the edge. The de Bruijn reading, which is the structure a
    sequence already has rather than one imposed on it."""
    records: list[tuple[str, str]] = []
    name, buf = None, []
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buf)))
                name, buf = line[1:].split()[0] if len(line) > 1 else "seq", []
            else:
                buf.append(line.upper())
    if name is not None:
        records.append((name, "".join(buf)))
    if not records:
        raise ValueError(f"{path}: no FASTA records")

    index: dict[str, int] = {}
    src, tgt, weight = [], [], {}
    for _, seq in records:
        prev = None
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer not in index:
                index[kmer] = len(index)
            cur = index[kmer]
            if prev is not None and prev != cur:
                key = (prev, cur)
                weight[key] = weight.get(key, 0.0) + 1.0
            prev = cur
    for (a, b), _w in weight.items():
        src.append(a)
        tgt.append(b)
    if not index:
        raise ValueError(f"{path}: sequences shorter than k={k}")
    labels = [""] * len(index)
    for kmer, i in index.items():
        labels[i] = kmer
    return _ec(src, tgt, labels,
               weights=[weight[(a, b)] for a, b in zip(src, tgt, strict=False)],
               type_names=["overlap"])


#### incidence
def load_vcf(path, **kw) -> EdgeConstruction:
    """VCF as a bipartite incidence between samples and variants.

    An edge exists where a sample carries a non-reference allele. A 0/0 genotype is
    the ABSENCE of an edge, not an edge weighted zero. Existence is a condition of
    the complex, and encoding "no variant" as a present edge would put it in the
    wrong one.
    """
    samples: list[str] = []
    variants: list[str] = []
    variant_records: dict = {}
    sample_records: dict = {}
    src, tgt = [], []
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            cols = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = cols[9:]
                for j, name in enumerate(samples):
                    sample_records[j] = {"sample": name, "role": "sample"}
                continue
            if len(cols) < 10 or not samples:
                continue
            vid = cols[2] if cols[2] not in (".", "") else f"{cols[0]}:{cols[1]}"
            # the variant's own columns, on the variant. All of these were read past, so
            # a caller could not select the SNVs or the calls passing the filter.
            record = {"chrom": cols[0], "pos": int(cols[1]),
                      "ref": cols[3] if len(cols) > 3 else "",
                      "alt": cols[4] if len(cols) > 4 else ""}
            if len(cols) > 5 and cols[5] not in (".", ""):
                try:
                    record["qual"] = float(cols[5])
                except ValueError:
                    pass
            if len(cols) > 6 and cols[6] not in (".", ""):
                record["filter"] = cols[6]
            ref, alt = record["ref"], record["alt"].split(",")[0]
            record["variant_type"] = (
                "snv" if len(ref) == 1 and len(alt) == 1 and alt not in (".", "")
                else "indel" if ref and alt else "other")
            variant_records[vid] = record
            v_idx = len(samples) + len(variants)
            variants.append(vid)
            # GT is wherever FORMAT says it is, and it is not always first. Reading
            # sub-field 0 blind turned a DP-only record's read depth into "carries".
            keys = cols[8].split(":") if len(cols) > 8 else []
            try:
                gt_at = keys.index("GT")
            except ValueError:
                continue                       # no genotype, so no carrier to record
            for s_i, cell in enumerate(cols[9:]):
                parts = cell.split(":")
                if gt_at >= len(parts):
                    continue
                gt = parts[gt_at].replace("|", "/")
                alleles = [a for a in gt.split("/") if a.isdigit()]
                if any(int(a) > 0 for a in alleles):
                    src.append(s_i)
                    tgt.append(v_idx)
    if not samples:
        raise ValueError(f"{path}: no #CHROM header, so no sample columns")
    # the vertex block is samples first, then variants, so the variant records offset
    construction = _ec(src, tgt, samples + variants, type_names=["carries"])
    attributes = dict(sample_records)
    for j, vid in enumerate(variants):
        record = variant_records.get(vid)
        if record:
            attributes[len(samples) + j] = record
    if attributes:
        construction.attributes = {0: attributes}
    return construction


#### intervals
def _interval_overlap_ec(rows: list[tuple[str, int, int, str]], *,
                         aliases=None, origin="") -> EdgeConstruction:
    """Intervals sharing a coordinate axis and overlapping become an edge.

    Sorted sweep rather than the O(n^2) pair scan, so a whole annotation file is
    tractable. Intervals on different sequences never overlap however close their
    coordinates look.
    """
    labels = [name for _, _, _, name in rows]
    order = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][1]))
    src, tgt = [], []
    active: list[int] = []
    for i in order:
        seq, start, end, _ = rows[i]
        active = [j for j in active if rows[j][0] == seq and rows[j][2] > start]
        for j in active:
            src.append(j)
            tgt.append(i)
        active.append(i)
    return _ec(src, tgt, labels, type_names=["overlap"],
               aliases=aliases, origin=origin)


def _dbxrefs(attrs: str) -> set:
    """Cross-references a GFF row declares.

    `Dbxref=GeneID:672,HGNC:HGNC:1100` is the row saying, explicitly, which other
    databases name this feature. That is a join key stated by the file itself, which
    is the only kind worth trusting.
    """
    out = set()
    for field in attrs.replace('"', "").split(";"):
        field = field.strip()
        for key in ("Dbxref=", "db_xref ", "Dbxref "):
            if field.startswith(key):
                for ref in field[len(key):].split(","):
                    ref = ref.strip()
                    if ref:
                        out.add(ref)
    return out


#: GFF column 3 values grouped by the level they sit at. A row's own identifier and
#: its parent's are different columns at each level, and conflating them is what makes
#: a transcript inherit its gene's name as if it were an alias.
_GFF_LEVELS = {
    "gene": ("gene", ("ID", "gene_id"), ("gene_name", "Name", "gene_symbol"), ()),
    "transcript": ("transcript", ("ID", "transcript_id"),
                   ("transcript_name", "Name"), ("Parent", "gene_id")),
    "exon": ("exon", ("ID", "exon_id"), ("Name",), ("Parent", "transcript_id",
                                                    "gene_id")),
}
_GFF_LEVEL_OF = {
    "gene": "gene", "pseudogene": "gene", "ncrna_gene": "gene",
    "transcript": "transcript", "mrna": "transcript", "ncrna": "transcript",
    "lnc_rna": "transcript", "rrna": "transcript", "trna": "transcript",
    "exon": "exon", "cds": "exon", "five_prime_utr": "exon",
    "three_prime_utr": "exon", "utr": "exon", "start_codon": "exon",
    "stop_codon": "exon",
}


def _gff_attrs(attrs: str) -> dict:
    """A GFF/GTF attribute column as a dict.

    GFF3 writes `key=value;`, GTF writes `key "value";`. Both appear in files named
    either way, so both are read.
    """
    out: dict[str, str] = {}
    for field in attrs.replace('"', "").split(";"):
        field = field.strip()
        if not field:
            continue
        if "=" in field:
            k, _, v = field.partition("=")
        else:
            k, _, v = field.partition(" ")
        k, v = k.strip(), v.strip()
        if k and v and k not in out:
            out[k] = v
    return out


def _attrs_by_label(construction, by_label):
    """Attach per-label attribute records onto the vertices they landed on.

    Readers that build intervals do not know a feature's vertex index until the overlap
    pass has run, so they collect by label and this maps them across afterwards.
    """
    index_of = {name: i for i, name in enumerate(construction.vertex_labels)}
    attributes = {}
    for label, record in by_label.items():
        index = index_of.get(label)
        if index is not None:
            attributes[index] = record
    if attributes:
        construction.attributes = {0: attributes}
    return construction


def load_gff(path, **kw) -> EdgeConstruction:
    """GFF3/GTF features as an overlap graph plus the containment they declare.

    Two kinds of relation come out of one file. Features that overlap on a sequence
    are related by position, which is what the interval sweep finds. Features also
    declare what contains them, through `Parent=` in GFF3 or `gene_id`/
    `transcript_id` in GTF, and that hierarchy is stated rather than computed.

    Which identifiers name a row is decided by its level, because they are not
    interchangeable: `gene_name` on a transcript row names the transcript's GENE. As
    an alias it would merge the transcript into its own gene; as a parent it is the
    containment edge it actually is. This is the join that bioinformatics pipelines
    most often get wrong, and it is wrong in a direction that silently loses
    resolution rather than failing.
    """
    rows: list[tuple[str, int, int, str]] = []
    aliases: dict[str, set] = {}
    parent_of: list[tuple[str, str]] = []
    id_index: dict[str, str] = {}
    by_label: dict[str, dict] = {}

    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 5:
                continue
            attrs = _gff_attrs(c[8] if len(c) > 8 else "")
            ftype = (c[2] if len(c) > 2 else "").strip().lower()
            level = _GFF_LEVEL_OF.get(ftype)
            _name, own_keys, alias_keys, parent_keys = _GFF_LEVELS.get(
                level, (None, ("ID", "gene_id", "transcript_id"), ("Name",),
                        ("Parent",)))

            label = next((attrs[k] for k in own_keys if attrs.get(k)), None)
            if level == "exon" and not label and attrs.get("exon_number"):
                base = attrs.get("transcript_id") or attrs.get("Parent") or c[0]
                label = f"{base}:exon{attrs['exon_number']}"
            if not label:
                label = f"{c[0]}:{c[3]}-{c[4]}"

            own = {attrs[k] for k in own_keys if attrs.get(k)}
            own |= {attrs[k] for k in alias_keys if attrs.get(k)}
            own |= _dbxrefs(c[8] if len(c) > 8 else "")
            own.discard(label)
            if own:
                aliases.setdefault(label, set()).update(own)
            for ident in {label, *own}:
                id_index.setdefault(ident, label)

            parent = next((attrs[k] for k in parent_keys if attrs.get(k)), None)
            if parent and parent != label:
                parent_of.append((label, parent))

            # GFF coordinates are 1-based and inclusive; the sweep is half-open, as
            # BED is. Passing them through unconverted made features that share a
            # single base read as disjoint.
            rows.append((c[0], int(c[3]) - 1, int(c[4]), label))
            # column 9 is already parsed into a dict and only two keys were read from
            # it. The rest of the line is standard and was read past: a caller could not
            # select the minus strand or the features on one contig without re-reading
            # the file. Keyed by LABEL here because vertex indices are assigned later.
            record = {"seqid": c[0], "start": int(c[3]), "end": int(c[4]),
                      "feature_type": ftype or "feature"}
            if len(c) > 1 and c[1].strip() and c[1].strip() != ".":
                record["source"] = c[1].strip()
            if len(c) > 5 and c[5].strip() not in ("", "."):
                try:
                    record["score"] = float(c[5])
                except ValueError:
                    pass
            if len(c) > 6 and c[6].strip() in ("+", "-"):
                record["strand"] = c[6].strip()
            if len(c) > 7 and c[7].strip() not in ("", "."):
                record["phase"] = c[7].strip()
            for key, value in attrs.items():
                record.setdefault(str(key), value)
            by_label.setdefault(label, record)

    if not rows:
        raise ValueError(f"{path}: no features")

    ec = _interval_overlap_ec(rows, aliases=aliases, origin=str(path))
    label_at = {name: i for i, name in enumerate(ec.vertex_labels)}
    contained = []
    for child, declared in parent_of:
        parent = id_index.get(declared, declared)
        if parent == child:
            continue
        ci, pi = label_at.get(child), label_at.get(parent)
        if ci is not None and pi is not None:
            contained.append((ci, pi))
    if not contained:
        return _attrs_by_label(ec, by_label)

    src = np.concatenate([ec.sources,
                          np.asarray([a for a, _ in contained], np.int32)])
    tgt = np.concatenate([ec.targets,
                          np.asarray([b for _, b in contained], np.int32)])
    types = np.concatenate([np.zeros(ec.nE, np.int32),
                            np.ones(len(contained), np.int32)])
    return _attrs_by_label(
        _ec(src, tgt, ec.vertex_labels, types=types,
            type_names=["overlap", "part_of"],
            aliases=aliases, origin=str(path)),
        by_label)


def load_bed(path, **kw) -> EdgeConstruction:
    """BED intervals as an overlap graph. BED starts are 0-based and ends
    exclusive, which is already the half-open convention the sweep assumes."""
    rows = []
    by_label: dict = {}
    with _open_text(path) as fh:
        for line in fh:
            first = line.split()[0] if line.split() else ""
            if line.startswith("#") or first in ("track", "browser") or not line.strip():
                continue                       # a contig may legitimately be named track1
            c = line.rstrip("\n").split("\t")
            if len(c) < 3:
                continue
            name = c[3] if len(c) > 3 else f"{c[0]}:{c[1]}-{c[2]}"
            rows.append((c[0], int(c[1]), int(c[2]), name))
            record = {"chrom": c[0], "start": int(c[1]), "end": int(c[2])}
            if len(c) > 4 and c[4].strip() not in ("", "."):
                try:
                    record["score"] = float(c[4])
                except ValueError:
                    pass
            if len(c) > 5 and c[5].strip() in ("+", "-"):
                record["strand"] = c[5].strip()
            by_label.setdefault(name, record)
    if not rows:
        raise ValueError(f"{path}: no intervals")
    return _attrs_by_label(_interval_overlap_ec(rows, origin=str(path)), by_label)


#### matrix containers
def _h5_index(group) -> list[str]:
    """The axis labels of an AnnData-style dataframe group."""
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    if key not in group:
        # Falling through to the first key by HDF5 order picked whatever sorted
        # first, so a boolean flag column became the gene names. Prefer the columns
        # that hold names, then any text column, and label positionally otherwise.
        named = ("gene_symbol", "gene_symbols", "gene_name", "gene_names", "symbol",
                 "feature_name", "gene_ids", "index", "name", "Gene", "CellID")
        key = next((k for k in named if k in group), None)
        if key is None:
            key = next((k for k in group
                        if getattr(group[k], "dtype", None) is not None
                        and group[k].dtype.kind in ("S", "U", "O")), None)
    if key is None:
        return []
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v)
            for v in group[key][:]]


def load_h5ad(path, **kw):
    """AnnData: the X matrix plus its two axis label sets.

    Returns (matrix, obs_names, var_names). The HDF5 layout is read directly, so
    anndata is not required for what is a matrix and two label vectors.
    """
    import h5py
    with h5py.File(str(path), "r") as f:
        if "X" not in f:
            raise ValueError(f"{path}: no X matrix")
        node = f["X"]
        if isinstance(node, h5py.Group):          # sparse CSR/CSC layout
            import scipy.sparse as sp
            enc = node.attrs.get("encoding-type", "csr_matrix")
            if isinstance(enc, bytes):
                enc = enc.decode("utf-8")
            shape = tuple(int(x) for x in node.attrs["shape"])
            cls = sp.csc_matrix if "csc" in str(enc) else sp.csr_matrix
            Xs = cls((node["data"][:], node["indices"][:], node["indptr"][:]),
                     shape=shape)
            # this reader returns a dense array, so a stored-sparse X is materialised
            # here. Ask the library's own guard first: a cells x genes matrix is
            # routinely large enough to exhaust memory, and failing with the limit named
            # is worth more than an OOM from inside h5py.
            from rexgraph.core._common import check_dense_allocation
            check_dense_allocation("load_h5ad X", int(shape[0]), int(shape[1]))
            X = Xs.toarray()
        else:
            X = np.asarray(node[:], dtype=np.float64)
        obs = _h5_index(f["obs"]) if "obs" in f else []
        var = _h5_index(f["var"]) if "var" in f else []
    return np.asarray(X, dtype=np.float64), obs, var


def load_loom(path, **kw):
    """Loom: the main matrix plus row and column attributes. Loom stores
    genes x cells, the transpose of AnnData, so it is transposed to match."""
    import h5py
    with h5py.File(str(path), "r") as f:
        if "matrix" not in f:
            raise ValueError(f"{path}: no matrix")
        X = np.asarray(f["matrix"][:], dtype=np.float64).T
        def _attr(group, *names):
            if group not in f:
                return []
            for n in names:
                if n in f[group]:
                    return [v.decode("utf-8") if isinstance(v, bytes) else str(v)
                            for v in f[group][n][:]]
            return []
        # loompy writes Name/Accession as often as Gene, and a missing lookup left
        # the genes unnamed rather than falling back to something meaningful.
        obs = _attr("col_attrs", "CellID", "cell_id", "CellName", "cell_name", "obs_names")
        var = _attr("row_attrs", "Gene", "gene", "Name", "name", "gene_name",
                    "GeneName", "Symbol", "Accession", "var_names")
    return X, obs, var


def load_bcf(path, **kw):
    """BCF is the binary encoding of VCF and needs a different reader.

    Registered so the extension gives a straight answer rather than failing inside
    the text parser with "no #CHROM header", which describes the wrong problem.
    """
    raise ValueError(
        f"{path}: BCF is binary (BGZF), and this reader parses text VCF. "
        f"Convert it first: `bcftools view -Ov -o out.vcf {path}`")


register_reader("sdf", load_sdf, extensions=[".sdf", ".mol"])
register_reader("pdb", load_pdb, extensions=[".pdb", ".ent"])
register_reader("fasta", load_fasta, extensions=[".fasta", ".fa", ".fna", ".faa"])
register_reader("vcf", load_vcf, extensions=[".vcf"])
register_reader("bcf", load_bcf, extensions=[".bcf"])
register_reader("gff", load_gff, extensions=[".gff", ".gff3", ".gtf"])
register_reader("bed", load_bed, extensions=[".bed"])
register_reader("h5ad", load_h5ad, extensions=[".h5ad"])
register_reader("loom", load_loom, extensions=[".loom"])

# Ontology files are containers too: .obo, .owl, .ttl, .nt and the GO annotation
# formats read through the same registry, so `read(path)` opens an ontology the way
# it opens a .pdb.
from .ontology_formats import register as _register_ontology_readers  # noqa: E402

_register_ontology_readers(register_reader)
