"""
Correlation and adjacency matrix adapters.

Handle square symmetric matrices - correlation matrices, similarity
matrices, adjacency matrices - and turn them into typed edges.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import DomainAdapter, EdgeConstruction
from .feature_matrix import _auto_threshold, _spectral_cluster_features


class CorrelationAdapter(DomainAdapter):
    """Construct a typed relational complex from a correlation/similarity matrix."""

    name = "correlation"

    def build(
        self,
        R: NDArray,
        labels: list[str] | None = None,
        threshold: str | float = "auto",
        typing: str = "spectral",
        sign: str = "matrix",
    ) -> EdgeConstruction:
        """Build typed edges from a symmetric matrix.

        Parameters
        ----------
        R : ndarray (n, n), symmetric
        labels : list of str, optional
            Vertex names. Defaults to v0, v1, ...
        threshold : 'auto' or float
        typing : 'spectral' or 'none'
        sign : 'matrix' (use sign of R[i,j]) or 'positive'
        """
        R = np.asarray(R, dtype=np.float64)
        n = R.shape[0]

        if labels is None:
            labels = [f"v{i}" for i in range(n)]

        R_work = R.copy()
        np.fill_diagonal(R_work, 0.0)

        threshold_val = _auto_threshold(R_work) if threshold == "auto" else float(threshold)

        # Build edges - vectorized over the upper triangle (was an O(n²) Python
        # double loop); identical result and edge order (i<j).
        iu, ju = np.triu_indices(n, k=1)
        r_up = R_work[iu, ju]
        mask = np.abs(r_up) > threshold_val
        sources = iu[mask].astype(np.int32)
        targets = ju[mask].astype(np.int32)
        corr_arr = r_up[mask].astype(np.float64)
        weights = np.abs(corr_arr)

        if sign == "matrix":
            signs = np.sign(corr_arr)
            signs[signs == 0] = 1.0
        else:
            signs = np.ones(len(sources), dtype=np.float64)

        # Typing
        if typing == "spectral" and n >= 4:
            vertex_types = _spectral_cluster_features(np.abs(R_work))
            n_vt = int(vertex_types.max()) + 1
            type_names = [f"cluster_{i}" for i in range(n_vt)] + ["cross"]
            cross_label = n_vt
            type_labels = np.empty(len(sources), dtype=np.int32)
            for k in range(len(sources)):
                if vertex_types[sources[k]] == vertex_types[targets[k]]:
                    type_labels[k] = vertex_types[sources[k]]
                else:
                    type_labels[k] = cross_label
        else:
            type_labels = np.zeros(len(sources), dtype=np.int32)
            type_names = ["edge"]

        return EdgeConstruction(
            sources=sources,
            targets=targets,
            weights=weights,
            signs=signs,
            type_labels=type_labels,
            vertex_labels=labels,
            n_types=len(type_names),
            type_names=type_names,
        )


class AdjacencyAdapter(DomainAdapter):
    """Construct a relational complex from an adjacency matrix."""

    name = "adjacency"

    def build(
        self,
        A: NDArray,
        labels: list[str] | None = None,
        directed: bool = False,
    ) -> EdgeConstruction:
        """Build edges from an adjacency matrix. No thresholding - every
        nonzero entry becomes an edge."""
        A = np.asarray(A, dtype=np.float64)
        n = A.shape[0]

        if labels is None:
            labels = [f"v{i}" for i in range(n)]

        # Vectorized edge extraction (was an O(n²) Python double loop); identical
        # result and row-major (directed) / i<j (undirected) edge order.
        if directed:
            M = np.abs(A) > 1e-15
            np.fill_diagonal(M, False)
            src, tgt = np.nonzero(M)             # C-order: i outer, j inner
            wt_arr = A[src, tgt]
        else:
            iu, ju = np.triu_indices(n, k=1)
            val = A[iu, ju] + A[ju, iu]
            mask = np.abs(val) > 1e-15
            src, tgt = iu[mask], ju[mask]
            wt_arr = val[mask]

        sources = src.astype(np.int32)
        targets = tgt.astype(np.int32)
        wt_arr = wt_arr.astype(np.float64)
        weights = np.abs(wt_arr)
        signs = np.sign(wt_arr)
        signs[signs == 0] = 1.0

        return EdgeConstruction(
            sources=sources,
            targets=targets,
            weights=weights,
            signs=signs,
            type_labels=np.zeros(len(sources), dtype=np.int32),
            vertex_labels=labels,
            n_types=1,
            type_names=["edge"],
        )
