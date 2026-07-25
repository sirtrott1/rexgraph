"""
Single-cell / 10X Genomics adapter.

Handles the sparse Matrix Market format the core biology workflow used
(``matrix.mtx(.gz)`` + ``barcodes.tsv(.gz)`` + ``features.tsv(.gz)`` /
``genes.tsv(.gz)``), assigns cell types by marker-gene scoring, and
builds a cell-cell interaction network via
:class:`~agent.adapters.lr_interaction.LRInteractionAdapter`.

Pipeline
--------
1. ``load_10x(dir)``            -> sparse cells x genes, barcodes, genes
2. marker scoring              -> per-cell type label (argmax of scores)
3. mean expression per type    -> cell-type x gene table
4. L-R scoring between types   -> EdgeConstruction (via LRInteractionAdapter)

If no markers are supplied, cells are grouped by a light k-means on the
top principal components so the adapter still produces a usable
cell-type table instead of failing.
"""

from __future__ import annotations

import gzip
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import DomainAdapter, EdgeConstruction

logger = logging.getLogger(__name__)

# Compact default marker panel (broad lineages). Callers with a proper
# panel should pass ``markers=...``.
DEFAULT_MARKERS: Dict[str, List[str]] = {
    "T_cell": ["CD3D", "CD3E", "CD2", "TRAC"],
    "B_cell": ["CD19", "MS4A1", "CD79A", "CD79B"],
    "Myeloid": ["LYZ", "CD68", "ITGAM", "CSF1R"],
    "NK": ["NKG7", "GNLY", "KLRD1"],
    "Endothelial": ["PECAM1", "VWF", "CDH5"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19"],
}

_MTX_NAMES = ["matrix.mtx.gz", "matrix.mtx"]
_BARCODE_NAMES = ["barcodes.tsv.gz", "barcodes.tsv"]
_FEATURE_NAMES = [
    "features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv",
]


def _find(dir_path: Path, names: Sequence[str]) -> Optional[Path]:
    for n in names:
        p = dir_path / n
        if p.exists():
            return p
    return None


def is_10x_dir(path) -> bool:
    """True if ``path`` is a directory holding 10X triplet files."""
    try:
        p = Path(path)
    except (TypeError, ValueError):
        return False
    if not p.is_dir():
        return False
    return _find(p, _MTX_NAMES) is not None and (
        _find(p, _FEATURE_NAMES) is not None
    )


def _open_maybe_gz(p: Path):
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt")
    return open(p, "r")


def _read_tsv_column(p: Path, col: int = 0) -> List[str]:
    out: List[str] = []
    with _open_maybe_gz(p) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or parts == [""]:
                continue
            out.append(parts[col] if len(parts) > col else parts[0])
    return out


def load_10x(path) -> Tuple["scipy.sparse.csr_matrix", List[str], List[str]]:
    """Load a 10X directory into (cells x genes CSR, barcodes, gene names).

    Matrix Market from 10X is genes x cells; this returns the transpose
    (cells x genes) as CSR for row-wise (per-cell) operations.
    """
    from scipy import sparse
    from scipy.io import mmread

    p = Path(path)
    mtx = _find(p, _MTX_NAMES)
    feat = _find(p, _FEATURE_NAMES)
    bc = _find(p, _BARCODE_NAMES)
    if mtx is None or feat is None:
        raise FileNotFoundError(
            f"{p} is not a 10X directory (need matrix.mtx[.gz] and "
            "features/genes.tsv[.gz])."
        )

    if str(mtx).endswith(".gz"):
        with gzip.open(mtx, "rb") as fh:
            m = mmread(fh)
    else:
        m = mmread(str(mtx))
    m = sparse.csr_matrix(m)  # genes x cells

    # 10X features.tsv: col 0 = id, col 1 = symbol. Prefer symbol.
    with _open_maybe_gz(feat) as fh:
        first = fh.readline().rstrip("\n").split("\t")
    symbol_col = 1 if len(first) > 1 else 0
    genes = _read_tsv_column(feat, col=symbol_col)
    barcodes = _read_tsv_column(bc, col=0) if bc else [
        f"cell_{i}" for i in range(m.shape[1])
    ]

    # Orient to cells x genes.
    if m.shape[0] == len(genes):
        cells_x_genes = m.T.tocsr()
    elif m.shape[1] == len(genes):
        cells_x_genes = m.tocsr()
    else:
        # Fall back to the 10X convention (genes x cells).
        cells_x_genes = m.T.tocsr()

    return cells_x_genes, barcodes, genes


def _normalize_log(cxg):
    """Library-size normalise to 10k counts then log1p (standard scRNA)."""
    from scipy import sparse

    cxg = cxg.astype(np.float64)
    lib = np.asarray(cxg.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    scaling = (1e4 / lib)
    D = sparse.diags(scaling)
    normed = D @ cxg
    normed.data = np.log1p(normed.data)
    return normed.tocsr()


def score_marker_types(cxg, genes: List[str], markers: Dict[str, List[str]]):
    """Assign each cell the marker set with the highest mean expression.

    Returns (labels, type_names). Cells with no marker signal get the
    label ``"Unassigned"``.
    """
    gidx = {g: i for i, g in enumerate(genes)}
    type_names = list(markers.keys())
    n_cells = cxg.shape[0]
    scores = np.zeros((n_cells, len(type_names)), dtype=np.float64)

    dense_cols: Dict[int, np.ndarray] = {}

    def col(j):
        if j not in dense_cols:
            dense_cols[j] = np.asarray(cxg[:, j].todense()).ravel()
        return dense_cols[j]

    for ti, tname in enumerate(type_names):
        present = [gidx[g] for g in markers[tname] if g in gidx]
        if not present:
            continue
        acc = np.zeros(n_cells, dtype=np.float64)
        for j in present:
            acc += col(j)
        scores[:, ti] = acc / len(present)

    # z-score across types so panels of different sizes are comparable.
    mu = scores.mean(axis=0, keepdims=True)
    sd = scores.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    z = (scores - mu) / sd

    best = np.argmax(z, axis=1)
    # Cells whose best score is still ~0 (no marker expression) -> Unassigned.
    row_has_signal = scores[np.arange(n_cells), best] > 0
    labels = np.array(
        [type_names[b] if row_has_signal[i] else "Unassigned"
         for i, b in enumerate(best)],
        dtype=object,
    )
    return labels, type_names


def _kmeans_types(cxg, k: int = 6):
    """Fallback clustering when no markers are given (top-PC k-means)."""
    normed = _normalize_log(cxg)
    # Use the densest genes to keep the embedding cheap.
    gene_tot = np.asarray(normed.sum(axis=0)).ravel()
    top = np.argsort(gene_tot)[-min(200, normed.shape[1]):]
    Xs = normed[:, top].tocsr()                    # n_cells × m (m ≤ 200), SPARSE
    n_cells, m = Xs.shape
    mu = np.asarray(Xs.mean(axis=0)).ravel()       # column means, length m

    # Top-k PCA embedding U[:, :k]·S[:k] = X_centered · V[:, :k], computed WITHOUT
    # densifying the n_cells×m matrix or running a full SVD. The right singular
    # vectors V and S² come from the tiny m×m Gram matrix of the CENTERED data:
    #   G = X_centeredᵀ X_centered = Xsᵀ Xs - n_cells·μμᵀ   (m×m ≤ 200², exact).
    # Same result as the old dense SVD (identical up to per-component sign, to which
    # Euclidean k-means is invariant); cost is O(nnz + m²), scaling in nnz not cells.
    try:
        G = np.asarray((Xs.T @ Xs).todense()) - n_cells * np.outer(mu, mu)
        G = 0.5 * (G + G.T)
        kk = int(min(k, m))
        evals, evecs = np.linalg.eigh(G)
        V = evecs[:, ::-1][:, :kk]                 # top-kk singular directions
        emb = np.asarray(Xs @ V) - (mu @ V)        # X_centered · V, (n_cells × kk)
    except Exception:
        emb = np.asarray(Xs[:, :min(k, m)].todense()) - mu[:min(k, m)]

    # Tiny k-means.
    rng = np.random.default_rng(0)
    k = min(k, emb.shape[0])
    centers = emb[rng.choice(emb.shape[0], size=k, replace=False)]
    labels = np.zeros(emb.shape[0], dtype=int)
    for _ in range(25):
        d = ((emb[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = emb[m].mean(axis=0)
    names = [f"cluster_{c}" for c in range(k)]
    return np.array([names[l] for l in labels], dtype=object), names


def mean_expression_by_type(cxg, genes, labels):
    """Return (types, type x gene mean-expression matrix) on log-norm data."""
    normed = _normalize_log(cxg)
    types = sorted(set(labels.tolist()))
    mat = np.zeros((len(types), len(genes)), dtype=np.float64)
    for ti, t in enumerate(types):
        mask = labels == t
        if not mask.any():
            continue
        rows = normed[np.where(mask)[0], :]
        mat[ti, :] = np.asarray(rows.mean(axis=0)).ravel()
    return types, mat


class SingleCellAdapter(DomainAdapter):
    """Turn a 10X directory into a cell-cell interaction complex."""

    name = "single_cell"

    def build(
        self,
        data,
        markers: Optional[Dict[str, List[str]]] = None,
        lr_pairs: Optional[Sequence[Tuple[str, str]]] = None,
        *,
        n_clusters: int = 6,
        min_score: float = 0.0,
        **kwargs,
    ) -> EdgeConstruction:
        """Build an L-R interaction complex from a 10X directory.

        Parameters
        ----------
        data : str | Path
            Path to a directory containing the 10X triplet.
        markers : dict, optional
            ``{cell_type: [marker genes]}``.  Defaults to
            :data:`DEFAULT_MARKERS`.  If none of the markers match the
            data, falls back to unsupervised clustering.
        lr_pairs : list of (ligand, receptor), optional
            Passed to :class:`LRInteractionAdapter`.
        n_clusters : int
            Cluster count for the marker-free fallback.
        """
        cxg, barcodes, genes = load_10x(data)
        logger.info(
            "Loaded 10X: %d cells x %d genes", cxg.shape[0], cxg.shape[1]
        )

        panel = markers if markers is not None else DEFAULT_MARKERS
        normed = _normalize_log(cxg)
        labels, _ = score_marker_types(normed, genes, panel)

        # If marker scoring assigned almost nothing, fall back to k-means.
        n_assigned = int(np.sum(labels != "Unassigned"))
        if n_assigned < max(3, 0.05 * cxg.shape[0]):
            logger.warning(
                "Marker scoring assigned only %d cells; using unsupervised "
                "clustering fallback.", n_assigned
            )
            labels, _ = _kmeans_types(cxg, k=n_clusters)

        types, type_gene = mean_expression_by_type(cxg, genes, labels)

        from .lr_interaction import LRInteractionAdapter
        lr = LRInteractionAdapter()
        ec = lr.build(
            type_gene,
            lr_pairs=lr_pairs,
            gene_names=genes,
            cell_types=types,
            min_score=min_score,
        )
        ec.input_type = "single_cell"
        # Stash the intermediate table for callers that want it.
        ec.cell_type_expression = {
            "cell_types": types,
            "genes": genes,
            "matrix": type_gene,
            "n_cells": int(cxg.shape[0]),
            "label_counts": {
                t: int(np.sum(labels == t)) for t in sorted(set(labels.tolist()))
            },
        }
        return ec

    def interpret(self, results: dict) -> dict:
        out = dict(results)
        out.setdefault("domain", "single_cell")
        return out
