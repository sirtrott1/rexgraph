"""
Ligand-receptor interaction adapter.

Reconstructs the core biology workflow's cell-cell communication step:
given a *cell-type x gene* mean-expression table and a set of curated
ligand-receptor (L-R) pairs, score directed interactions

    score(A -> B) = sum over (L, R) pairs of  mean_expr[A, L] * mean_expr[B, R]

and turn cell types into vertices and interactions into typed, directed
edges.  Each L-R pair becomes an edge *type*, so ``typed_face_selection``
can separate same-pathway triangles (faces) from cross-pathway ones
(voids) exactly as the manual workflow did.

The output is an :class:`EdgeConstruction`, so it flows through
``build_rex_from_edges`` / ``auto_rex`` like every other adapter.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from . import DomainAdapter, EdgeConstruction

logger = logging.getLogger(__name__)

# A small default panel so the adapter is usable without external files.
# These are widely-used, well-characterised human L-R pairs; callers with
# a curated panel (CellPhoneDB, etc.) should pass their own.
DEFAULT_LR_PAIRS: list[tuple[str, str]] = [
    ("TGFB1", "TGFBR1"),
    ("TGFB1", "TGFBR2"),
    ("VEGFA", "FLT1"),
    ("VEGFA", "KDR"),
    ("PDGFB", "PDGFRB"),
    ("IL6", "IL6R"),
    ("TNF", "TNFRSF1A"),
    ("CXCL12", "CXCR4"),
    ("CCL2", "CCR2"),
    ("EGF", "EGFR"),
    ("WNT5A", "FZD1"),
    ("DLL4", "NOTCH1"),
    ("CD274", "PDCD1"),
    ("LGALS9", "HAVCR2"),
]


def _as_type_gene_frame(expression, gene_names=None, cell_types=None):
    """Normalise the expression input to (matrix, cell_types, gene_names).

    Accepts a pandas DataFrame (index = cell type, columns = gene),
    a dict {cell_type: {gene: value}}, or a 2-D array with explicit
    ``cell_types`` and ``gene_names``.
    """
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and isinstance(expression, pd.DataFrame):
        mat = expression.to_numpy(dtype=np.float64)
        return mat, list(expression.index.astype(str)), list(
            expression.columns.astype(str)
        )

    if isinstance(expression, dict):
        types = list(expression.keys())
        gene_order: list[str] = []
        for row in expression.values():
            for g in row:
                if g not in gene_order:
                    gene_order.append(g)
        mat = np.zeros((len(types), len(gene_order)), dtype=np.float64)
        gidx = {g: i for i, g in enumerate(gene_order)}
        for ti, t in enumerate(types):
            for g, v in expression[t].items():
                mat[ti, gidx[g]] = float(v)
        return mat, [str(t) for t in types], gene_order

    mat = np.asarray(expression, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("expression must be a 2-D cell-type x gene table")
    if cell_types is None:
        cell_types = [f"type_{i}" for i in range(mat.shape[0])]
    if gene_names is None:
        gene_names = [f"gene_{j}" for j in range(mat.shape[1])]
    return mat, [str(c) for c in cell_types], [str(g) for g in gene_names]


class LRInteractionAdapter(DomainAdapter):
    """Score ligand-receptor interactions between cell types.

    Example
    -------
    >>> adapter = LRInteractionAdapter()
    >>> ec = adapter.build(
    ...     expression_df,               # index = cell type, cols = gene
    ...     lr_pairs=[("TGFB1", "TGFBR1")],
    ... )
    >>> rex = build_rex_from_edges(ec)
    """

    name = "lr_interaction"

    def build(
        self,
        expression,
        lr_pairs: Sequence[tuple[str, str]] | None = None,
        gene_names: list[str] | None = None,
        cell_types: list[str] | None = None,
        *,
        min_score: float = 0.0,
        self_interactions: bool = False,
        expressed_frac: float = 0.0,
        **kwargs,
    ) -> EdgeConstruction:
        """Build directed L-R interaction edges between cell types.

        Parameters
        ----------
        expression : DataFrame | dict | ndarray
            Cell-type x gene mean expression.
        lr_pairs : list of (ligand, receptor)
            Curated L-R pairs.  Defaults to :data:`DEFAULT_LR_PAIRS`.
        gene_names, cell_types : list of str, optional
            Required only when ``expression`` is a bare ndarray.
        min_score : float
            Drop interactions whose total score is <= this value.
        self_interactions : bool
            If False (default), skip A -> A edges (autocrine).
        expressed_frac : float
            A gene must exceed ``expressed_frac`` of its own max across
            cell types to count as "expressed" in a type; guards against
            noise driving spurious interactions.  0 disables the gate.
        """
        mat, types, genes = _as_type_gene_frame(
            expression, gene_names=gene_names, cell_types=cell_types
        )
        pairs = list(lr_pairs) if lr_pairs is not None else list(DEFAULT_LR_PAIRS)
        gidx = {g: i for i, g in enumerate(genes)}

        # Keep only pairs whose ligand and receptor are both present.
        usable = [(l, r) for (l, r) in pairs if l in gidx and r in gidx]
        if not usable:
            logger.warning(
                "No L-R pairs matched the expression genes "
                "(%d pairs requested, 0 usable). Check gene symbols.",
                len(pairs),
            )
            return self._empty()

        # Per-gene "expressed" mask (optional gate).
        if expressed_frac > 0:
            col_max = np.maximum(mat.max(axis=0), 1e-12)
            expressed = mat >= (expressed_frac * col_max)
        else:
            expressed = np.ones_like(mat, dtype=bool)

        nT = len(types)
        # Accumulate edges keyed by (src, dst, pair_index).
        src_l: list[int] = []
        dst_l: list[int] = []
        w_l: list[float] = []
        type_l: list[int] = []
        type_names: list[str] = [f"{l}->{r}" for (l, r) in usable]

        for pi, (lig, rec) in enumerate(usable):
            li, ri = gidx[lig], gidx[rec]
            for a in range(nT):          # ligand-producing type
                if not expressed[a, li]:
                    continue
                la = mat[a, li]
                if la <= 0:
                    continue
                for b in range(nT):      # receptor-bearing type
                    if a == b and not self_interactions:
                        continue
                    if not expressed[b, ri]:
                        continue
                    score = la * mat[b, ri]
                    if score <= min_score:
                        continue
                    src_l.append(a)
                    dst_l.append(b)
                    w_l.append(float(score))
                    type_l.append(pi)

        if not src_l:
            logger.warning(
                "L-R scoring produced no interactions above min_score=%s.",
                min_score,
            )
            return self._empty(vertex_labels=types)

        sources = np.asarray(src_l, dtype=np.int32)
        targets = np.asarray(dst_l, dtype=np.int32)
        weights = np.asarray(w_l, dtype=np.float64)
        # Normalise weights to [0, 1] so downstream thresholds are stable.
        wmax = float(weights.max())
        if wmax > 0:
            weights = weights / wmax
        signs = np.ones(len(sources), dtype=np.float64)
        type_labels = np.asarray(type_l, dtype=np.int32)

        ec = EdgeConstruction(
            sources=sources,
            targets=targets,
            weights=weights,
            signs=signs,
            type_labels=type_labels,
            vertex_labels=list(types),
            n_types=len(type_names),
            type_names=type_names,
        )
        # Tag so auto_rex records the right provenance.
        ec.input_type = "lr_interaction"
        return ec

    @staticmethod
    def _empty(vertex_labels: list[str] | None = None) -> EdgeConstruction:
        return EdgeConstruction(
            sources=np.array([], dtype=np.int32),
            targets=np.array([], dtype=np.int32),
            weights=np.array([], dtype=np.float64),
            signs=np.array([], dtype=np.float64),
            type_labels=np.array([], dtype=np.int32),
            vertex_labels=list(vertex_labels or []),
            n_types=0,
            type_names=[],
        )

    def interpret(self, results: dict) -> dict:
        """Label results with cell-communication semantics."""
        out = dict(results)
        out.setdefault("domain", "cell_communication")
        out.setdefault(
            "domain_note",
            "Vertices are cell types; directed edges are ligand->receptor "
            "signalling; edge types are L-R pairs. Faces are same-pathway "
            "signalling triangles; voids are cross-pathway gaps.",
        )
        return out
