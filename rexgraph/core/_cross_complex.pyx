# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._cross_complex - Cross-complex structural comparison.

Aligns two relational complexes by shared vertex labels and compares
structural invariants (coherence kappa, void fraction, spectral
channel scores) across them. All data is passed as arrays; this
module does not import or depend on RexGraph.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    get_EPSILON_NORM,
)

np.import_array()


# Label alignment

def align_by_labels(list labels_A, list labels_B):
    """Find shared vertices between two complexes by label matching.

    Parameters
    ----------
    labels_A : list of str
        Vertex labels for complex A. labels_A[i] is the label of vertex i.
    labels_B : list of str
        Vertex labels for complex B.

    Returns
    -------
    tuple
        shared_labels : list of str
            Labels present in both complexes.
        idx_A : i32 array
            idx_A[k] is the vertex index in A for shared_labels[k].
        idx_B : i32 array
            idx_B[k] is the vertex index in B for shared_labels[k].
    """
    # Build label -> index map for B
    cdef dict b_map = {}
    cdef Py_ssize_t i
    for i in range(len(labels_B)):
        b_map[labels_B[i]] = i

    shared = []
    a_idx = []
    b_idx = []

    for i in range(len(labels_A)):
        label = labels_A[i]
        if label in b_map:
            shared.append(label)
            a_idx.append(i)
            b_idx.append(b_map[label])

    return (
        shared,
        np.array(a_idx, dtype=np.int32),
        np.array(b_idx, dtype=np.int32),
    )


# Pearson correlation (single-pass, same pattern as _faces._pearson_corr)

cdef f64 _pearson(const f64* x, const f64* y, int n) noexcept nogil:
    """Single-pass Pearson correlation. Returns 0.0 if n < 2 or zero variance."""
    if n < 2:
        return 0.0

    cdef f64 sx = 0.0, sy = 0.0
    cdef f64 sx2 = 0.0, sy2 = 0.0, sxy = 0.0
    cdef f64 xi, yi
    cdef int i
    cdef f64 nn = <f64>n
    cdef f64 dx, dy, denom

    for i in range(n):
        xi = x[i]
        yi = y[i]
        sx += xi
        sy += yi
        sx2 += xi * xi
        sy2 += yi * yi
        sxy += xi * yi

    dx = nn * sx2 - sx * sx
    dy = nn * sy2 - sy * sy

    if dx < 1e-30 or dy < 1e-30:
        return 0.0

    denom = sqrt(dx * dy)
    return (nn * sxy - sx * sy) / denom


# Kappa correlation

def cross_complex_kappa(np.ndarray[f64, ndim=1] kappa_A,
                         np.ndarray[f64, ndim=1] kappa_B,
                         np.ndarray[i32, ndim=1] idx_A,
                         np.ndarray[i32, ndim=1] idx_B):
    """Correlate coherence kappa across two complexes at shared vertices.

    Parameters
    ----------
    kappa_A : f64[nV_A]
        Coherence from complex A.
    kappa_B : f64[nV_B]
        Coherence from complex B.
    idx_A : i32[n_shared]
        Shared vertex indices in A (from align_by_labels).
    idx_B : i32[n_shared]
        Shared vertex indices in B (from align_by_labels).

    Returns
    -------
    dict
        correlation : float
            Pearson correlation of kappa at shared vertices.
        n_shared : int
        kappa_A_shared : f64[n_shared]
        kappa_B_shared : f64[n_shared]
        mean_A : float
        mean_B : float
    """
    cdef int n_shared = idx_A.shape[0]

    if n_shared < 2:
        return {
            'correlation': 0.0,
            'n_shared': n_shared,
            'kappa_A_shared': np.zeros(n_shared, dtype=np.float64),
            'kappa_B_shared': np.zeros(n_shared, dtype=np.float64),
            'mean_A': 0.0,
            'mean_B': 0.0,
        }

    cdef np.ndarray[f64, ndim=1] ka = np.empty(n_shared, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] kb = np.empty(n_shared, dtype=np.float64)
    cdef f64[::1] kav = ka, kbv = kb
    cdef f64[::1] ka_full = kappa_A, kb_full = kappa_B
    cdef i32[::1] ia = idx_A, ib = idx_B
    cdef int k
    cdef f64 sum_a = 0.0, sum_b = 0.0

    for k in range(n_shared):
        kav[k] = ka_full[ia[k]]
        kbv[k] = kb_full[ib[k]]
        sum_a += kav[k]
        sum_b += kbv[k]

    cdef f64 corr = _pearson(&kav[0], &kbv[0], n_shared)

    return {
        'correlation': float(corr),
        'n_shared': n_shared,
        'kappa_A_shared': ka,
        'kappa_B_shared': kb,
        'mean_A': float(sum_a / n_shared),
        'mean_B': float(sum_b / n_shared),
    }


# Void fraction comparison

def cross_complex_void_fraction(int n_voids_A, int n_potential_A,
                                 int n_voids_B, int n_potential_B):
    """Compare void fractions between two complexes.

    void_fraction = n_voids / n_potential where n_potential is the
    total number of triangles (realized + void).

    Parameters
    ----------
    n_voids_A : int
    n_potential_A : int
        Total triangles in complex A.
    n_voids_B : int
    n_potential_B : int
        Total triangles in complex B.

    Returns
    -------
    dict
        void_fraction_A : float
        void_fraction_B : float
        difference : float
            A - B.
    """
    cdef f64 vf_a = 0.0, vf_b = 0.0

    if n_potential_A > 0:
        vf_a = <f64>n_voids_A / <f64>n_potential_A
    if n_potential_B > 0:
        vf_b = <f64>n_voids_B / <f64>n_potential_B

    return {
        'void_fraction_A': float(vf_a),
        'void_fraction_B': float(vf_b),
        'difference': float(vf_a - vf_b),
    }


# Channel score correlation

def cross_complex_channel_scores(np.ndarray[f64, ndim=1] scores_A,
                                   np.ndarray[f64, ndim=1] scores_B):
    """Correlate spectral channel scores between two complexes.

    Scores are per-group (e.g. per treatment arm) vectors from
    group_channel_scores. Correlation measures whether the two
    complexes rank groups similarly.

    Parameters
    ----------
    scores_A : f64[n_groups]
        Per-group scores from complex A.
    scores_B : f64[n_groups]
        Per-group scores from complex B.

    Returns
    -------
    dict
        correlation : float
        n_groups : int
    """
    cdef int n = scores_A.shape[0]

    if n < 2:
        return {'correlation': 0.0, 'n_groups': n}

    cdef f64 corr = _pearson(&scores_A[0], &scores_B[0], n)

    return {
        'correlation': float(corr),
        'n_groups': n,
    }


# Full bridge analysis

def cross_complex_bridge(np.ndarray[f64, ndim=1] kappa_A,
                          np.ndarray[f64, ndim=1] kappa_B,
                          np.ndarray[i32, ndim=1] idx_A,
                          np.ndarray[i32, ndim=1] idx_B,
                          int n_voids_A, int n_potential_A,
                          int n_voids_B, int n_potential_B,
                          channel_scores_A=None,
                          channel_scores_B=None):
    """Full cross-complex bridge analysis.

    Combines kappa correlation, void fraction comparison, and
    (optionally) channel score correlation into a single result.
    All data is passed as arrays extracted from the two complexes
    by the caller.

    Parameters
    ----------
    kappa_A : f64[nV_A]
    kappa_B : f64[nV_B]
    idx_A : i32[n_shared]
    idx_B : i32[n_shared]
    n_voids_A, n_potential_A : int
    n_voids_B, n_potential_B : int
    channel_scores_A : f64[n_groups] or None
    channel_scores_B : f64[n_groups] or None

    Returns
    -------
    dict
        kappa : dict from cross_complex_kappa
        void : dict from cross_complex_void_fraction
        channel : dict from cross_complex_channel_scores (if provided)
        n_shared : int
    """
    kappa_result = cross_complex_kappa(kappa_A, kappa_B, idx_A, idx_B)
    void_result = cross_complex_void_fraction(n_voids_A, n_potential_A,
                                               n_voids_B, n_potential_B)

    result = {
        'kappa': kappa_result,
        'void': void_result,
        'n_shared': int(idx_A.shape[0]),
    }

    if channel_scores_A is not None and channel_scores_B is not None:
        result['channel'] = cross_complex_channel_scores(
            np.ascontiguousarray(channel_scores_A, dtype=np.float64),
            np.ascontiguousarray(channel_scores_B, dtype=np.float64),
        )

    return result
