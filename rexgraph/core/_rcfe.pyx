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

    if nF == 0:
        return np.zeros((0, nhats), dtype=np.float64)

    B2_d = np.asarray(B2, dtype=np.float64)

    rl_diag = np.diag(np.asarray(RL, dtype=np.float64))
    hat_diags = [np.diag(np.asarray(hats[k], dtype=np.float64)) for k in range(nhats)]

    tensor = np.zeros((nF, nhats), dtype=np.float64)

    for f in range(nF):
        for e in range(nE):
            if abs(B2_d[e, f]) > 0.5 and rl_diag[e] > 1e-15:
                for k in range(nhats):
                    tensor[f, k] += hat_diags[k][e] / rl_diag[e]

    return tensor


# Derived RCFE quantities

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


# ═══ Phase 3: Dynamic RCFE strain (Sections 3, 5) ═══

def attributed_curvature(np.ndarray[f64, ndim=2] B1,
                          np.ndarray[f64, ndim=2] B2,
                          np.ndarray[f64, ndim=1] w_e,
                          np.ndarray[f64, ndim=1] a_v,
                          Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF):
    """Attributed boundary curvature (Def 3.1-3.2).

    Build attributed boundary operators:
        B1^w[v,e] = a_v * B1[v,e] * sqrt(w_e)
        B2^w[e,f] = sqrt(w_e) * B2[e,f]

    Then curvature residual R = B1^w @ B2^w, and
        kappa_f = ||R[:,f]||_2

    Parameters
    ----------
    B1 : f64[nV, nE]
    B2 : f64[nE, nF]
    w_e : f64[nE] - edge weights (> 0)
    a_v : f64[nV] - vertex amplitudes (>= 0)
    nV, nE, nF : dimensions

    Returns
    -------
    dict with kappa_f, R, B1w, B2w
    """
    if nF == 0:
        return {
            'kappa_f': np.zeros(0, dtype=np.float64),
            'R': np.zeros((nV, 0), dtype=np.float64),
            'B1w': np.zeros((nV, nE), dtype=np.float64),
            'B2w': np.zeros((nE, 0), dtype=np.float64),
        }

    cdef np.ndarray[f64, ndim=1] sqw = np.sqrt(np.maximum(w_e, 0))
    cdef np.ndarray[f64, ndim=2] B1w = np.empty((nV, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] B2w = np.empty((nE, nF), dtype=np.float64)
    cdef f64[:, ::1] b1v = B1, b2v = B2, b1wv = B1w, b2wv = B2w
    cdef f64[::1] sv = sqw, av = a_v
    cdef int v, e, f

    # B1^w[v,e] = a_v * B1[v,e] * sqrt(w_e)
    for v in range(nV):
        for e in range(nE):
            b1wv[v, e] = av[v] * b1v[v, e] * sv[e]

    # B2^w[e,f] = sqrt(w_e) * B2[e,f]
    for e in range(nE):
        for f in range(nF):
            b2wv[e, f] = sv[e] * b2v[e, f]

    # R = B1^w @ B2^w
    cdef np.ndarray[f64, ndim=2] R = np.asarray(B1w) @ np.asarray(B2w)

    # kappa_f = ||R[:,f]||_2
    cdef np.ndarray[f64, ndim=1] kappa_f = np.empty(nF, dtype=np.float64)
    cdef f64[::1] kv = kappa_f
    cdef f64 norm_sq
    for f in range(nF):
        norm_sq = 0
        for v in range(nV):
            norm_sq += R[v, f] * R[v, f]
        kv[f] = sqrt(norm_sq)

    return {
        'kappa_f': kappa_f,
        'R': R,
        'B1w': B1w,
        'B2w': B2w,
    }


def face_deficit(np.ndarray[f64, ndim=1] kappa_f,
                  f64 alpha,
                  np.ndarray[f64, ndim=1] born_face,
                  Py_ssize_t nF):
    """Face deficit: delta_f = kappa_f - alpha * |Psi_f|^2 (Def 5.1).

    Parameters
    ----------
    kappa_f : f64[nF] - attributed curvature per face
    alpha : float - coupling constant
    born_face : f64[nF] - Born probability per face from Dirac state

    Returns
    -------
    delta : f64[nF] - deficit per face
    """
    cdef np.ndarray[f64, ndim=1] delta = np.empty(nF, dtype=np.float64)
    cdef f64[::1] dv = delta, kv = kappa_f, bv = born_face
    cdef int f
    for f in range(nF):
        dv[f] = kv[f] - alpha * bv[f]
    return delta


def relational_strain_dynamic(np.ndarray[f64, ndim=2] B2,
                                np.ndarray[f64, ndim=1] delta,
                                Py_ssize_t nE, Py_ssize_t nF):
    """Relational strain: sigma = B2 @ delta (Def 5.2).

    sigma(e) measures the net face deficit across edge e.
    B1 @ sigma = 0 by the chain condition (Bianchi conservation).

    Returns f64[nE].
    """
    if nF == 0:
        return np.zeros(nE, dtype=np.float64)

    cdef np.ndarray[f64, ndim=1] sigma = np.zeros(nE, dtype=np.float64)
    cdef f64[:, ::1] b2v = np.asarray(B2, dtype=np.float64)
    cdef f64[::1] sv = sigma, dv = delta
    cdef int e, f

    for e in range(nE):
        for f in range(nF):
            sv[e] += b2v[e, f] * dv[f]

    return sigma


def optimal_alpha(np.ndarray[f64, ndim=2] B2,
                   np.ndarray[f64, ndim=1] kappa_f,
                   np.ndarray[f64, ndim=1] born_face,
                   Py_ssize_t nE, Py_ssize_t nF):
    """Optimal coupling: alpha = <B2 kappa, B2 pF> / ||B2 pF||^2 (Def 5.3).

    Minimizes ||sigma||^2 = ||B2 (kappa - alpha pF)||^2.

    Returns alpha (float). Returns 0 if denominator is zero.
    """
    if nF == 0:
        return 0.0

    B2_d = np.asarray(B2, dtype=np.float64)
    B2_kappa = B2_d @ np.asarray(kappa_f, dtype=np.float64)
    B2_pF = B2_d @ np.asarray(born_face, dtype=np.float64)

    cdef f64 numer = 0, denom = 0
    cdef f64[::1] bkv = B2_kappa, bpv = B2_pF
    cdef int e
    for e in range(nE):
        numer += bkv[e] * bpv[e]
        denom += bpv[e] * bpv[e]

    if denom < 1e-15:
        return 0.0
    return float(numer / denom)


def verify_bianchi_strain(np.ndarray[f64, ndim=2] B1,
                           np.ndarray[f64, ndim=1] sigma,
                           Py_ssize_t nV, Py_ssize_t nE,
                           f64 tol=1e-10):
    """Verify Bianchi conservation: B1 @ sigma = 0 (Theorem 6.1).

    Since sigma = B2 @ delta and B1 @ B2 = 0, this must hold exactly.

    Returns (is_valid, max_residual).
    """
    B1_d = np.asarray(B1, dtype=np.float64)
    residual = B1_d @ np.asarray(sigma, dtype=np.float64)
    cdef f64 max_res = float(np.max(np.abs(residual)))
    return max_res < tol, max_res


def strain_equilibrium(np.ndarray[f64, ndim=2] B1,
                        np.ndarray[f64, ndim=2] B2,
                        np.ndarray[f64, ndim=1] kappa_f,
                        np.ndarray[f64, ndim=1] born_face,
                        Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF):
    """Full strain equilibrium analysis.

    Computes optimal alpha, face deficit, relational strain, Bianchi check.

    Returns dict with alpha, delta, sigma, bianchi_ok, bianchi_residual,
    strain_norm.
    """
    alpha_opt = optimal_alpha(B2, kappa_f, born_face, nE, nF)
    delta = face_deficit(kappa_f, alpha_opt, born_face, nF)
    sigma = relational_strain_dynamic(B2, delta, nE, nF)
    bianchi_ok, bianchi_res = verify_bianchi_strain(B1, sigma, nV, nE)

    return {
        'alpha': float(alpha_opt),
        'delta': delta,
        'sigma': sigma,
        'bianchi_ok': bianchi_ok,
        'bianchi_residual': float(bianchi_res),
        'strain_norm': float(np.sqrt(np.dot(sigma, sigma))),
    }
