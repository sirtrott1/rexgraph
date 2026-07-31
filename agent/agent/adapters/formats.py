"""
agent.adapters.formats: readers for file types auto_rex could not open.

Each format is read as the structure it already is, so nothing here is domain
framing -- these are containers, the way .csv is a container:

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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rexgraph.registry import Registry

from . import EdgeConstruction

_READERS = Registry("file reader")


def register_reader(name: str, fn, *, extensions=()) -> None:
    """Register a reader. `fn(path, **kw)` returns whatever that format yields."""
    _READERS.register(name, fn, extensions=tuple(e.lower() for e in extensions))


def unregister_reader(name: str):
    return _READERS.unregister(name)


def available_readers() -> List[str]:
    return _READERS.available()


def available_extensions() -> Dict[str, str]:
    """extension -> reader name."""
    return {e: name for name in _READERS
            for e in _READERS.meta(name).get("extensions", ())}


def reader_for(path) -> Optional[str]:
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
    return open(p, "r", encoding="utf-8", errors="replace")


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


# --- bonded structure ---------------------------------------------------------

def load_sdf(path, **kw) -> EdgeConstruction:
    """MDL SDF/MOL V2000. Atoms are vertices, bonds are edges, bond order is the
    edge type -- which is the typed-edge information the complex already carries,
    so it needs no encoding of its own."""
    with _open_text(path) as fh:
        lines = fh.read().splitlines()
    if len(lines) < 4:
        raise ValueError(f"{path}: too short to be an SDF/MOL record")
    counts = lines[3]
    try:
        n_atoms, n_bonds = int(counts[0:3]), int(counts[3:6])
    except ValueError:
        raise ValueError(f"{path}: unreadable counts line {counts!r}")

    labels = []
    for i in range(n_atoms):
        parts = lines[4 + i].split()
        element = parts[3] if len(parts) > 3 else "X"
        labels.append(f"{element}{i + 1}")

    src, tgt, orders = [], [], []
    for j in range(n_bonds):
        row = lines[4 + n_atoms + j]
        try:
            a, b, order = int(row[0:3]), int(row[3:6]), int(row[6:9])
        except ValueError:
            parts = row.split()
            if len(parts) < 3:
                continue
            a, b, order = int(parts[0]), int(parts[1]), int(parts[2])
        src.append(a - 1)                      # SDF atom indices are 1-based
        tgt.append(b - 1)
        orders.append(order)

    uniq = sorted(set(orders))
    idx = {o: i for i, o in enumerate(uniq)}
    return _ec(src, tgt, labels,
               types=[idx[o] for o in orders],
               type_names=[f"bond_order_{o}" for o in uniq] or ["bond"])


def load_pdb(path, **kw) -> EdgeConstruction:
    """PDB ATOM records with explicit CONECT bonds. Only CONECT is used: inferring
    bonds from distance is a modelling decision, not a file read."""
    serial_to_idx: Dict[int, int] = {}
    labels: List[str] = []
    conect: List[Tuple[int, int]] = []
    with _open_text(path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                serial = int(line[6:11])
                name = line[12:16].strip()
                res = line[17:20].strip()
                serial_to_idx[serial] = len(labels)
                labels.append(f"{res}:{name}:{serial}")
            elif rec == "CONECT":
                nums = [int(line[i:i + 5]) for i in range(6, len(line.rstrip()), 5)
                        if line[i:i + 5].strip().isdigit()]
                for other in nums[1:]:
                    conect.append((nums[0], other))
    src, tgt, seen = [], [], set()
    for a, b in conect:
        if a not in serial_to_idx or b not in serial_to_idx:
            continue
        i, j = serial_to_idx[a], serial_to_idx[b]
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        src.append(i)
        tgt.append(j)
    if not labels:
        raise ValueError(f"{path}: no ATOM records")
    return _ec(src, tgt, labels, type_names=["bond"])


# --- sequences ----------------------------------------------------------------

def load_fasta(path, *, k: int = 5, **kw) -> EdgeConstruction:
    """FASTA as a k-mer overlap graph: consecutive k-mers share k-1 characters, and
    that overlap is the edge. The de Bruijn reading, which is the structure a
    sequence already has rather than one imposed on it."""
    records: List[Tuple[str, str]] = []
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

    index: Dict[str, int] = {}
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
    for (a, b), w in weight.items():
        src.append(a)
        tgt.append(b)
    if not index:
        raise ValueError(f"{path}: sequences shorter than k={k}")
    labels = [""] * len(index)
    for kmer, i in index.items():
        labels[i] = kmer
    return _ec(src, tgt, labels,
               weights=[weight[(a, b)] for a, b in zip(src, tgt)],
               type_names=["overlap"])


# --- incidence ----------------------------------------------------------------

def load_vcf(path, **kw) -> EdgeConstruction:
    """VCF as a bipartite incidence between samples and variants.

    An edge exists where a sample carries a non-reference allele. A 0/0 genotype is
    the ABSENCE of an edge, not an edge weighted zero -- existence is a condition of
    the complex, and encoding "no variant" as a present edge would put it in the
    wrong one.
    """
    samples: List[str] = []
    variants: List[str] = []
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
            for s_i, cell in enumerate(cols[9:]):
                gt = cell.split(":")[0].replace("|", "/")
                alleles = [a for a in gt.split("/") if a.isdigit()]
                if any(int(a) > 0 for a in alleles):
                    src.append(s_i)
                    tgt.append(v_idx)
    if not samples:
        raise ValueError(f"{path}: no #CHROM header, so no sample columns")
    return _ec(src, tgt, samples + variants, type_names=["carries"])


# --- intervals ----------------------------------------------------------------

def _interval_overlap_ec(rows: List[Tuple[str, int, int, str]]) -> EdgeConstruction:
    """Intervals sharing a coordinate axis and overlapping become an edge.

    Sorted sweep rather than the O(n^2) pair scan, so a whole annotation file is
    tractable. Intervals on different sequences never overlap however close their
    coordinates look.
    """
    labels = [name for _, _, _, name in rows]
    order = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][1]))
    src, tgt = [], []
    active: List[int] = []
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
            for field in attrs.replace('"', "").split(";"):
                field = field.strip()
                for key in ("ID=", "gene_id ", "ID ", "Name="):
                    if field.startswith(key):
                        name = field[len(key):].strip()
                        break
                if name:
                    break
            rows.append((c[0], int(c[3]), int(c[4]), name or f"{c[0]}:{c[3]}-{c[4]}"))
    if not rows:
        raise ValueError(f"{path}: no features")
    return _interval_overlap_ec(rows)


def load_bed(path, **kw) -> EdgeConstruction:
    """BED intervals as an overlap graph. BED starts are 0-based and ends
    exclusive, which is already the half-open convention the sweep assumes."""
    rows = []
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")) or not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 3:
                continue
            name = c[3] if len(c) > 3 else f"{c[0]}:{c[1]}-{c[2]}"
            rows.append((c[0], int(c[1]), int(c[2]), name))
    if not rows:
        raise ValueError(f"{path}: no intervals")
    return _interval_overlap_ec(rows)


# --- matrix containers --------------------------------------------------------

def _h5_index(group) -> List[str]:
    """The axis labels of an AnnData-style dataframe group."""
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    if key not in group:
        key = next((k for k in group), None)
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
        obs = _attr("col_attrs", "CellID", "cell_id")
        var = _attr("row_attrs", "Gene", "gene")
    return X, obs, var


register_reader("sdf", load_sdf, extensions=[".sdf", ".mol"])
register_reader("pdb", load_pdb, extensions=[".pdb", ".ent"])
register_reader("fasta", load_fasta, extensions=[".fasta", ".fa", ".fna", ".faa"])
register_reader("vcf", load_vcf, extensions=[".vcf", ".bcf"])
register_reader("gff", load_gff, extensions=[".gff", ".gff3", ".gtf"])
register_reader("bed", load_bed, extensions=[".bed"])
register_reader("h5ad", load_h5ad, extensions=[".h5ad"])
register_reader("loom", load_loom, extensions=[".loom"])
