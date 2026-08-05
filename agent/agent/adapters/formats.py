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


def _ec(sources, targets, labels, *, types=None, type_names=None, weights=None):
    """Assemble an EdgeConstruction from parallel edge arrays."""
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
    )


#### bonded structure
def _sdf_record(lines: list[str], base: int, tag: str):
    """Parse one MOL record into atom labels and bonds as (src, tgt, order).

    Atom indices are offset by `base` and labels are prefixed with `tag`, which places
    the record on its own vertex block. Returns (None, None) when the record has no
    readable counts line, the shape of the empty chunk after a trailing `$$$$`.
    """
    if len(lines) < 4:
        return None, None
    counts = lines[3]
    try:
        n_atoms, n_bonds = int(counts[0:3]), int(counts[3:6])
    except ValueError:
        return None, None

    # The counts line is a claim about the record. Trusting it past the end of the
    # block invented atoms out of the bond table and the M END line, and the real
    # bonds were then read from beyond the file and lost.
    need = 4 + n_atoms + n_bonds
    if need > len(lines):
        raise ValueError(
            f"malformed MOL record {tag.rstrip(':') or '1'}: the counts line declares "
            f"{n_atoms} atoms and {n_bonds} bonds, which needs {need} lines, "
            f"but the record has {len(lines)}")

    labels = []
    for i in range(n_atoms):
        parts = lines[4 + i].split()
        if len(parts) < 4:
            raise ValueError(
                f"malformed MOL record {tag.rstrip(':') or '1'}: atom line {i + 1} "
                f"is not an atom: {lines[4 + i]!r}")
        labels.append(f"{tag}{parts[3]}{i + 1}")

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
    return labels, bonds


def load_sdf(path, **kw) -> EdgeConstruction:
    """MDL SDF/MOL V2000. Atoms are vertices, bonds are edges, bond order is the edge
    type, which is typed-edge information the complex already carries.

    An SDF holds several records separated by `$$$$`. All of them are read into one
    complex. Atom indices are per-record in the format, so each record is offset onto
    its own vertex block and labelled with the record number, leaving the molecules as
    separate components with beta_0 counting them.
    """
    with _open_text(path) as fh:
        text = fh.read()

    labels: list[str] = []
    src: list[int] = []
    tgt: list[int] = []
    orders: list[int] = []
    n_rec = 0
    for chunk in text.split("$$$$"):
        lines = chunk.splitlines()
        while lines and not lines[0].strip():          # leading blank after a delimiter
            lines.pop(0)
        tag = "" if n_rec == 0 else f"m{n_rec}:"
        rec_labels, bonds = _sdf_record(lines, len(labels), tag)
        if rec_labels is None:
            continue
        labels.extend(rec_labels)
        for a, b, o in bonds:
            src.append(a)
            tgt.append(b)
            orders.append(o)
        n_rec += 1

    if not n_rec:
        raise ValueError(f"{path}: no readable SDF/MOL record")

    uniq = sorted(set(orders))
    idx = {o: i for i, o in enumerate(uniq)}
    # Bond order is a magnitude, so it is the edge weight and enters the complex
    # through W. It is also carried as a type, which names the relation. It is not
    # part of the face condition: a ring is a ring whichever bonds close it.
    return _ec(src, tgt, labels,
               types=[idx[o] for o in orders],
               type_names=[f"bond_order_{o}" for o in uniq] or ["bond"],
               weights=[float(o) for o in orders] or None)


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
    return _ec(src, tgt, labels, type_names=["bond"])


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
    src, tgt = [], []
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            cols = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = cols[9:]
                continue
            if len(cols) < 10 or not samples:
                continue
            vid = cols[2] if cols[2] not in (".", "") else f"{cols[0]}:{cols[1]}"
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
    return _ec(src, tgt, samples + variants, type_names=["carries"])


#### intervals
def _interval_overlap_ec(rows: list[tuple[str, int, int, str]]) -> EdgeConstruction:
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
    return _ec(src, tgt, labels, type_names=["overlap"])


def load_gff(path, **kw) -> EdgeConstruction:
    """GFF3/GTF features as an interval overlap graph."""
    rows = []
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 5:
                continue
            attrs = c[8] if len(c) > 8 else ""
            name = None
            # Most specific identifier first. GTF has no ID=, so gene_id won for a
            # gene, its transcripts and its exons alike and every row of one gene
            # got the same label.
            found = {}
            for field in attrs.replace('"', "").split(";"):
                field = field.strip()
                for key in ("ID=", "ID ", "exon_id ", "exon_number ",
                            "transcript_id ", "gene_id ", "Name="):
                    if field.startswith(key):
                        found.setdefault(key.strip(" ="), field[len(key):].strip())
                        break
            for key in ("ID", "exon_id", "transcript_id", "gene_id", "Name"):
                if found.get(key):
                    name = found[key]
                    break
            if found.get("exon_number") and name:
                name = f"{name}:exon{found['exon_number']}"
            # GFF coordinates are 1-based and inclusive; the sweep is half-open, as
            # BED is. Passing them through unconverted made features that share a
            # single base read as disjoint.
            rows.append((c[0], int(c[3]) - 1, int(c[4]),
                         name or f"{c[0]}:{c[3]}-{c[4]}"))
    if not rows:
        raise ValueError(f"{path}: no features")
    return _interval_overlap_ec(rows)


def load_bed(path, **kw) -> EdgeConstruction:
    """BED intervals as an overlap graph. BED starts are 0-based and ends
    exclusive, which is already the half-open convention the sweep assumes."""
    rows = []
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
    if not rows:
        raise ValueError(f"{path}: no intervals")
    return _interval_overlap_ec(rows)


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
            X = cls((node["data"][:], node["indices"][:], node["indptr"][:]),
                    shape=shape).toarray()
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
