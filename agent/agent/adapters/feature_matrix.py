"""
Feature matrix adapter: tabular data -> typed relational complex.

Takes a (samples × features) matrix and constructs typed, signed edges
between features based on correlation structure. Same-type triangles
become faces, cross-type become voids, ∂²=0 guaranteed.

Handles: cancer imaging, gene expression, financial features, sensor
data, survey responses - anything where rows are observations and
columns are measurements.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from . import DomainAdapter, EdgeConstruction


def _compute_correlation(X: NDArray) -> NDArray:
    """Feature-feature Pearson correlation with NaN safety."""
    R = np.corrcoef(X.T)
    R = np.nan_to_num(R, nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(R, 0.0)
    return R


def _auto_threshold(R: NDArray, target_density: float = 0.08) -> float:
    """Find correlation threshold that gives target edge density.

    Aims for ~5-10 edges per vertex on average. This gives enough
    structure for faces and voids without drowning in noise.

    Parameters
    ----------
    R : (n, n) absolute correlation matrix (diagonal zeroed)
    target_density : fraction of possible edges to keep (default 8%)

    Returns
    -------
    threshold : float in [0, 1]
    """
    n = R.shape[0]
    max_edges = n * (n - 1) // 2
    target_edges = max(int(max_edges * target_density), n)  # at least n edges
    upper = np.abs(R[np.triu_indices(n, k=1)])
    if len(upper) == 0:
        return 0.5
    sorted_vals = np.sort(upper)[::-1]
    idx = min(target_edges, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _spectral_cluster_features(R: NDArray, n_clusters: str | int = "auto") -> NDArray:
    """Cluster features by correlation structure using the Fiedler partition.

    Uses the eigenvectors of the correlation Laplacian to find natural
    feature groupings. This is itself a relational complex computation -
    we use a simple spectral decomposition to configure the full rex.

    Parameters
    ----------
    R : (n, n) absolute correlation matrix
    n_clusters : 'auto' or int. If 'auto', uses eigengap heuristic.

    Returns
    -------
    labels : int32 array of cluster assignments
    """
    # Rex-native LAPACK wrapper. Same dsyev_ call as scipy.linalg.eigh,
    # with the identical (evals sorted ascending, evecs in columns)
    # return convention. Keeping the eigendecomposition inside the
    # compiled kernel removes one of the remaining scipy hot paths in
    # the agent adapters.
    from rexgraph.core._linalg import eigh

    n = R.shape[0]
    if n < 3:
        return np.zeros(n, dtype=np.int32)

    # Build Laplacian from absolute correlations
    A = np.abs(R)
    np.fill_diagonal(A, 0.0)
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Symmetrize to guard against round-off (the correlation-derived
    # Laplacian should be symmetric by construction, but the rex eigh
    # expects a clean symmetric matrix). Cast to contiguous f64 so the
    # Cython-level typed memoryview accepts the buffer.
    L_sym = 0.5 * (L + L.T)
    L_sym = np.ascontiguousarray(L_sym, dtype=np.float64)
    # Only the smallest ~9 eigenpairs are ever used (the eigengap heuristic over
    # evals[1:max_k+1], max_k ≤ 8, and the Fiedler/low eigenvectors evecs[:,1:k]).
    # So compute just the low end via a partial solver instead of the full O(n³)
    # dense decomposition - the difference is a solver-capability boundary (ARPACK
    # needs k < n-1), NOT an accuracy trade: both return the exact smallest pairs.
    n_want = min(9, n - 1)
    if n_want >= n - 1:
        evals, evecs = eigh(L_sym)          # tiny n: ARPACK inapplicable, dense
    else:
        try:
            from scipy.sparse.linalg import eigsh
            evals, evecs = eigsh(L_sym, k=n_want, which="SA")
            order = np.argsort(evals)
            evals, evecs = evals[order], evecs[:, order]
        except Exception:
            evals, evecs = eigh(L_sym)      # robustness fallback
    evals = np.maximum(evals, 0.0)

    # Eigengap heuristic for number of clusters
    if n_clusters == "auto":
        max_k = min(8, n // 2)
        if max_k < 2:
            return np.zeros(n, dtype=np.int32)
        gaps = np.diff(evals[1:max_k + 1])
        n_clusters = int(np.argmax(gaps) + 2)
        n_clusters = max(2, min(n_clusters, max_k))

    # k-means on the first k eigenvectors
    V = evecs[:, 1:n_clusters]
    # Normalize rows
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    V = V / norms

    # Simple k-means (avoid sklearn dependency)
    labels = _kmeans(V, n_clusters, max_iter=50)
    return labels.astype(np.int32)


def _kmeans(X: NDArray, k: int, max_iter: int = 50) -> NDArray:
    """Minimal k-means. No sklearn needed."""
    n = X.shape[0]
    rng = np.random.RandomState(42)
    centers = X[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=np.int32)

    for _ in range(max_iter):
        # Assign
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in centers])
        new_labels = dists.argmin(axis=0).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centers
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)

    return labels


def _detect_column_families(names: List[str]) -> Optional[NDArray]:
    """Try to detect feature families from column name prefixes.

    Looks for common naming patterns:
    - PyRadiomics: original_shape_Sphericity, wavelet_glcm_Contrast
    - Underscore-separated: shape_volume, texture_entropy
    - Dot-separated: shape.volume, texture.entropy

    Returns None if no clear family structure is detected.
    """
    if not names or len(names) < 4:
        return None

    # Try underscore prefix
    prefixes = []
    for name in names:
        parts = name.replace(".", "_").split("_")
        if len(parts) >= 2:
            prefixes.append(parts[0].lower())
        else:
            prefixes.append(name.lower())

    unique = sorted(set(prefixes))
    if 2 <= len(unique) <= 10 and len(unique) < len(names):
        prefix_map = {p: i for i, p in enumerate(unique)}
        labels = np.array([prefix_map[p] for p in prefixes], dtype=np.int32)
        # Check that no single group dominates > 80%
        counts = np.bincount(labels)
        if counts.max() / len(labels) < 0.8:
            return labels

    return None


class FeatureMatrixAdapter(DomainAdapter):
    """Construct a typed relational complex from a feature matrix.

    The features become vertices. Correlated feature pairs become edges.
    Edge types come from feature families (detected or spectral-clustered).
    Same-type triangles become faces. Cross-type triangles become voids.
    ∂²=0 is guaranteed by typed_face_selection.
    """

    name = "feature_matrix"

    def build(
        self,
        X: NDArray,
        feature_names: Optional[List[str]] = None,
        threshold: str | float = "auto",
        typing: str = "auto",
        sign: str = "correlation",
        n_clusters: str | int = "auto",
        **kwargs,
    ) -> EdgeConstruction:
        """Build typed edges from a feature matrix.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
            Rows are observations, columns are features. Should be numeric.
            NaN values are replaced with column means.
        feature_names : list of str, optional
            Names for each feature column. If None, uses f0, f1, ...
        threshold : 'auto' or float
            Correlation threshold for edge creation.
            'auto': adaptive based on density target (~8% of possible edges).
            float in [0, 1]: use directly.
        typing : str
            How to assign edge types.
            'auto': try column family detection first, fall back to spectral.
            'column_family': parse feature names for prefix families.
            'spectral': spectral cluster the correlation matrix.
            'none': all edges get the same type (promotes all cycles).
        sign : str
            'correlation': edge sign = sign of Pearson correlation.
            'positive': all edges +1.
        n_clusters : 'auto' or int
            Number of clusters for spectral typing. 'auto' uses eigengap.

        Returns
        -------
        EdgeConstruction
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Handle NaN: replace with column means
        col_means = np.nanmean(X, axis=0)
        for j in range(n_features):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = col_means[j]

        # Feature names
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]
        feature_names = list(feature_names)[:n_features]

        # Correlation matrix
        R = _compute_correlation(X)

        # Threshold
        if threshold == "auto":
            threshold_val = _auto_threshold(R)
        else:
            threshold_val = float(threshold)

        # Build edges from thresholded correlations. Vectorized over the upper
        # triangle (was an O(n_features²) Python double loop) - identical result,
        # but the edge test is a single boolean mask over the triu entries.
        iu, ju = np.triu_indices(n_features, k=1)
        r_upper = R[iu, ju]
        mask = np.abs(r_upper) > threshold_val

        if not np.any(mask) and iu.size > 0:
            # No edges above threshold - lower it to get ~n_features edges
            # (one per vertex on average), same fallback as before.
            fallback = np.sort(np.abs(r_upper))[::-1]
            n_keep = min(n_features * 2, len(fallback))
            threshold_val = float(fallback[n_keep - 1]) * 0.99
            mask = np.abs(r_upper) > threshold_val

        sources = iu[mask].astype(np.int32)
        targets = ju[mask].astype(np.int32)
        corr_arr = r_upper[mask].astype(np.float64)

        # Weights and signs
        weights = np.abs(corr_arr)
        if sign == "correlation":
            signs = np.sign(corr_arr)
            signs[signs == 0] = 1.0
        else:
            signs = np.ones(len(sources), dtype=np.float64)

        # Edge typing
        type_labels, type_names = self._assign_types(
            typing, feature_names, R, n_features, sources, targets, n_clusters
        )

        return EdgeConstruction(
            sources=sources,
            targets=targets,
            weights=weights,
            signs=signs,
            type_labels=type_labels,
            vertex_labels=feature_names,
            n_types=len(type_names),
            type_names=type_names,
        )

    def _assign_types(
        self,
        typing: str,
        feature_names: List[str],
        R: NDArray,
        n_features: int,
        sources: NDArray,
        targets: NDArray,
        n_clusters,
    ) -> Tuple[NDArray, List[str]]:
        """Assign edge types based on the chosen strategy."""

        # Get vertex-level type labels
        if typing == "auto":
            vertex_types = _detect_column_families(feature_names)
            if vertex_types is None:
                vertex_types = _spectral_cluster_features(R, n_clusters)
                type_names = [f"cluster_{i}" for i in range(vertex_types.max() + 1)]
            else:
                prefixes = []
                for name in feature_names:
                    parts = name.replace(".", "_").split("_")
                    prefixes.append(parts[0].lower() if len(parts) >= 2 else name.lower())
                type_names = sorted(set(prefixes))
        elif typing == "column_family":
            vertex_types = _detect_column_families(feature_names)
            if vertex_types is None:
                vertex_types = np.zeros(n_features, dtype=np.int32)
                type_names = ["all"]
            else:
                prefixes = []
                for name in feature_names:
                    parts = name.replace(".", "_").split("_")
                    prefixes.append(parts[0].lower() if len(parts) >= 2 else name.lower())
                type_names = sorted(set(prefixes))
        elif typing == "spectral":
            vertex_types = _spectral_cluster_features(R, n_clusters)
            type_names = [f"cluster_{i}" for i in range(vertex_types.max() + 1)]
        else:  # 'none'
            vertex_types = np.zeros(n_features, dtype=np.int32)
            type_names = ["all"]

        # Convert vertex types to edge types
        # Same-type edge: both endpoints in the same cluster -> that cluster's label
        # Cross-type edge: endpoints in different clusters -> cross label
        n_vertex_types = len(type_names)
        cross_label = n_vertex_types
        type_names_full = type_names + ["cross"]

        edge_type_labels = np.empty(len(sources), dtype=np.int32)
        for k in range(len(sources)):
            ts = vertex_types[sources[k]]
            tt = vertex_types[targets[k]]
            if ts == tt:
                edge_type_labels[k] = ts
            else:
                edge_type_labels[k] = cross_label

        return edge_type_labels, type_names_full
