# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._rcfe - RCFE curvature, strain, and conservation laws.

Curvature C(sigma) measures how concentrated face structure is at each edge.
Strain S = sum C(e) * RL[e,e] is the total structural stress.
Bianchi identity: B1 @ diag(C) @ B2 = 0 (conservation law).
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


# Curvature

def compute_curvature(B2, Py_ssize_t nE, Py_ssize_t nF):
    """RCFE curvature per edge.

    C(e) = sum_f B2[e,f]^2 / ||B2[:,f]||^2.
    Measures how concentrated face structure is at edge e.
    """

    if nF == 0:
        return np.zeros(nE, dtype=np.float64)

    B2_d = np.asarray(B2, dtype=np.float64)

    cdef np.ndarray[f64, ndim=1] curv = np.zeros(nE, dtype=np.float64)
    cdef f64[::1] cv = curv
    cdef Py_ssize_t e, f
    cdef f64 col_norm_sq, val

    B2_arr = np.asarray(B2_d, dtype=np.float64)

    for f in range(nF):
        col = B2_arr[:, f]
        col_norm_sq = float(np.dot(col, col))
        if col_norm_sq < 1e-15:
            continue
        for e in range(nE):
            val = col[e]
            cv[e] += val * val / col_norm_sq

    return curv


# Strain

def compute_strain(np.ndarray[f64, ndim=1] curvature,
                    np.ndarray[f64, ndim=1] rl_diag,
                    Py_ssize_t nE):
    """Total RCFE strain: S = sum_e C(e) * RL[e,e]."""
    cdef f64[::1] cv = curvature, rd = rl_diag
    cdef f64 total = 0.0
    cdef Py_ssize_t e
    for e in range(nE):
        total += cv[e] * rd[e]
    return total


def compute_strain_per_face(B2, curvature, Py_ssize_t nE, Py_ssize_t nF):
    """Strain contribution per face."""

    if nF == 0:
        return np.zeros(0, dtype=np.float64)

    B2_d = np.asarray(B2, dtype=np.float64)

    strain_f = np.zeros(nF, dtype=np.float64)
    curv = np.asarray(curvature, dtype=np.float64)

    for f in range(nF):
        col = B2_d[:, f]
        for e in range(nE):
            if abs(col[e]) > 0.5:
                strain_f[f] += curv[e]

    return strain_f


# Bianchi identity

def verify_bianchi(B1, B2, curvature, Py_ssize_t nE, Py_ssize_t nF,
                    f64 tol=1e-10):
    """Verify B1 @ diag(C) @ B2 ~ 0.

    The RCFE Bianchi identity: curvature is a cocycle.
    """

    if nF == 0:
        return True, 0.0

    B1_d = np.asarray(B1, dtype=np.float64)

    B2_d = np.asarray(B2, dtype=np.float64)

    C_diag = np.diag(np.asarray(curvature, dtype=np.float64))
    product = B1_d @ C_diag @ B2_d
    max_err = float(np.max(np.abs(product)))

    return max_err < tol, max_err


def bianchi_residual(B1, B2, curvature, Py_ssize_t nE, Py_ssize_t nF):
    """Per-face Bianchi residual: ||B1 diag(C) B2[:,f]||."""

    if nF == 0:
        return np.zeros(0, dtype=np.float64)

    B1_d = np.asarray(B1, dtype=np.float64)

    B2_d = np.asarray(B2, dtype=np.float64)

    curv = np.asarray(curvature, dtype=np.float64)
    residuals = np.zeros(nF, dtype=np.float64)

    for f in range(nF):
        col = B2_d[:, f] * curv
        r = B1_d @ col
        residuals[f] = float(np.sqrt(np.dot(r, r)))

    return residuals


# Face realization rates

def face_realization_rates(B2, tri_edges, Py_ssize_t nT,
                            Py_ssize_t nV, Py_ssize_t nE,
                            sources, targets):
    """Fraction of potential triangles realized at each vertex/edge."""

    if nT == 0:
        return np.zeros(nV, dtype=np.float64), np.zeros(nE, dtype=np.float64)

    _, _, n_voids = (
        __import__('rexgraph.core._void', fromlist=['classify_triangles'])
        .classify_triangles(B2, tri_edges, nT, nE)
    )
    n_realized = nT - n_voids
    rate = float(n_realized) / nT if nT > 0 else 0.0

    # Per-edge: count triangles containing each edge, fraction realized
    per_edge_total = np.zeros(nE, dtype=np.float64)
    per_edge_real = np.zeros(nE, dtype=np.float64)

    te = np.asarray(tri_edges, dtype=np.int32)
    for k in range(nT):
        for j in range(3):
            per_edge_total[te[k, j]] += 1.0

    # Realized triangles
    realized, _, _ = (
        __import__('rexgraph.core._void', fromlist=['classify_triangles'])
        .classify_triangles(B2, tri_edges, nT, nE)
    )
    for k in range(nT):
        if realized[k]:
            for j in range(3):
                per_edge_real[te[k, j]] += 1.0

    per_edge_rate = np.where(per_edge_total > 0,
                              per_edge_real / per_edge_total, 0.0)

    # Per-vertex: aggregate from incident edges
    src = np.asarray(sources, dtype=np.int32)
    tgt = np.asarray(targets, dtype=np.int32)
    per_vertex = np.zeros(nV, dtype=np.float64)
    per_vertex_count = np.zeros(nV, dtype=np.float64)

    for e in range(nE):
        per_vertex[src[e]] += per_edge_rate[e]
        per_vertex_count[src[e]] += 1.0
        per_vertex[tgt[e]] += per_edge_rate[e]
        per_vertex_count[tgt[e]] += 1.0

    per_vertex = np.where(per_vertex_count > 0,
                           per_vertex / per_vertex_count, 0.0)

    return per_vertex, per_edge_rate


# Coupling tensor

def coupling_tensor(B2, RL, hats, Py_ssize_t nhats,
                     Py_ssize_t nE, Py_ssize_t nF):
    """Per-face energy decomposition by operator channel.

    tensor[f, k] = sum_{e in boundary(f)} hat_k[e,e] / RL[e,e].
    """
    from rexgraph.core._character import extract_diag

    if nF == 0:
        return np.zeros((0, nhats), dtype=np.float64)

    B2_d = np.asarray(B2, dtype=np.float64)

    rl_diag = extract_diag(RL)
    hat_diags = [extract_diag(hats[k]) for k in range(nhats)]

    tensor = np.zeros((nF, nhats), dtype=np.float64)

    for f in range(nF):
        for e in range(nE):
            if abs(B2_d[e, f]) > 0.5 and rl_diag[e] > 1e-15:
                for k in range(nhats):
                    tensor[f, k] += hat_diags[k][e] / rl_diag[e]

    return tensor


# ═══ Derived RCFE quantities ═══

def relational_integrity(np.ndarray[f64, ndim=1] curvature,
                          np.ndarray[f64, ndim=1] rl_diag,
                          Py_ssize_t nE,
                          np.ndarray[f64, ndim=2] B2=None,
                          Py_ssize_t nF=0):
    """Relational integrity RI = 1 / (1 + kappa_total).

    Also computes per-face RI if B2 is provided.
    kappa_total = sum_e C(e) * RL[e,e] (= strain).

    Returns dict with RI, kappa_total, per_face_RI (if B2 given).
    """
    cdef f64 kappa_total = 0
    cdef f64[::1] cv = curvature, rd = rl_diag
    cdef int e, f
    for e in range(nE):
        kappa_total += cv[e] * rd[e]

    result = {
        'RI': 1.0 / (1.0 + kappa_total),
        'kappa_total': float(kappa_total),
    }

    if B2 is not None and nF > 0:
        B2_arr = np.asarray(B2, dtype=np.float64)
        pf_curv = compute_strain_per_face(B2_arr, curvature, nE, nF)
        pf_ri = np.zeros(nF, dtype=np.float64)
        for f in range(nF):
            pf_ri[f] = 1.0 / (1.0 + pf_curv[f])
        result['per_face_RI'] = pf_ri
        result['per_face_curvature'] = pf_curv

    return result


def face_overlap_K2(np.ndarray[f64, ndim=2] B2, Py_ssize_t nE, Py_ssize_t nF):
    """Face overlap matrix K2 = |B2|^T |B2|.

    K2[f,f'] = number of shared boundary edges between faces f and f'.
    """
    cdef np.ndarray[f64, ndim=2] absB2 = np.abs(B2)
    cdef np.ndarray[f64, ndim=2] K2 = np.zeros((nF, nF), dtype=np.float64)
    cdef f64[:, ::1] av = absB2, kv = K2
    cdef int f1, f2, e

    # K2 = absB2^T @ absB2
    for f1 in range(nF):
        for f2 in range(f1, nF):
            for e in range(nE):
                if av[e, f1] > 0.5 and av[e, f2] > 0.5:
                    kv[f1, f2] += 1
            kv[f2, f1] = kv[f1, f2]

    return K2


def edge_weight_conjugation(np.ndarray[f64, ndim=2] L,
                              np.ndarray[f64, ndim=1] sqw,
                              Py_ssize_t nE):
    """Weighted Laplacian: L_w = sqrt(W) * L * sqrt(W).

    For dynamic edge weighting where w(e) depends on vertex amplitudes.
    sqw[e] = sqrt(w(e)).
    """
    cdef np.ndarray[f64, ndim=2] Lw = np.empty((nE, nE), dtype=np.float64)
    cdef f64[:, ::1] lv = L, lwv = Lw
    cdef f64[::1] sv = sqw
    cdef int i, j

    for i in range(nE):
        for j in range(nE):
            lwv[i, j] = sv[i] * lv[i, j] * sv[j]

    return Lw
