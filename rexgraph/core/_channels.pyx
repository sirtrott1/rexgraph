# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._channels - Per-channel signal decomposition and group scoring.

Primal signal character decomposes an edge signal's energy across
typed Laplacian channels via hat^+ quadratic forms. Spectral channel
scores propagate a source signal through RL eigenmodes and project
onto a target vector. Group scores aggregate spectral scores across
entity groups defined by vertex masks.

All eigendata is pre-computed and passed in. No eigensolves here.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs
from libc.string cimport memset

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    get_EPSILON_NORM,
)

from rexgraph.core._linalg cimport (
    bl_dot, bl_nrm2, bl_gemv_n, bl_gemv_t,
    spectral_pinv_matvec,
)

np.import_array()


# Primal signal character

cdef f64 _quadratic_pinv(const f64* psi,
                          const f64* evals, const f64* evecs,
                          int nE) noexcept nogil:
    """psi^T @ M^+ @ psi via spectral decomposition.

    M^+ = sum_{lam>tol} (1/lam) v_j v_j^T, so
    psi^T M^+ psi = sum_{lam>tol} (1/lam) (v_j . psi)^2.
    """
    cdef int j, i
    cdef f64 cj, result = 0.0

    for j in range(nE):
        if evals[j] > 1e-10:
            cj = 0.0
            for i in range(nE):
                cj += evecs[i * nE + j] * psi[i]
            result += cj * cj / evals[j]

    return result


def primal_signal_character(np.ndarray[f64, ndim=1] psi,
                             list hat_evals_list,
                             list hat_evecs_list,
                             int nhats, int nE):
    """Energy decomposition of an edge signal across typed channels.

    E_X = psi^T @ hat_X^+ @ psi for each channel X.
    Returned as fractions summing to 1.

    Parameters
    ----------
    psi : f64[nE]
        Edge signal.
    hat_evals_list : list of f64[nE]
        Eigenvalues per hat, from hat_eigen_all.
    hat_evecs_list : list of f64[nE, nE]
        Eigenvectors per hat, from hat_eigen_all.
    nhats : int
    nE : int

    Returns
    -------
    f64[nhats]
        Energy fractions per channel, summing to 1.
        Zero vector if total energy is zero.
    """
    cdef np.ndarray[f64, ndim=1] char = np.empty(nhats, dtype=np.float64)
    cdef f64[::1] cv = char
    cdef f64 total = 0.0
    cdef int k
    cdef np.ndarray[f64, ndim=1] ev_k
    cdef np.ndarray[f64, ndim=2] ec_k

    for k in range(nhats):
        ev_k = hat_evals_list[k]
        ec_k = hat_evecs_list[k]
        cv[k] = _quadratic_pinv(&psi[0], &ev_k[0], &ec_k[0, 0], nE)
        total += cv[k]

    if total > 1e-30:
        for k in range(nhats):
            cv[k] /= total
    else:
        for k in range(nhats):
            cv[k] = 1.0 / nhats if nhats > 0 else 0.0

    return char


# Spectral channel score

cdef f64 _spectral_channel_score(const f64* source, const f64* target,
                                   const f64* evals_RL, const f64* evecs_RL,
                                   int nE) noexcept nogil:
    """s = sum_j (c_j^src * c_j^tgt) / lambda_j for lambda_j > 0.

    c_j^src = <v_j, source>, c_j^tgt = <v_j, target>.
    Propagates source through RL eigenmodes and projects onto target.
    """
    cdef int j, i
    cdef f64 cj_src, cj_tgt, result = 0.0

    for j in range(nE):
        if evals_RL[j] < 1e-10:
            continue

        cj_src = 0.0
        cj_tgt = 0.0
        for i in range(nE):
            cj_src += evecs_RL[i * nE + j] * source[i]
            cj_tgt += evecs_RL[i * nE + j] * target[i]

        result += cj_src * cj_tgt / evals_RL[j]

    return result


def spectral_channel_score(np.ndarray[f64, ndim=1] source,
                            np.ndarray[f64, ndim=1] target,
                            np.ndarray[f64, ndim=1] evals_RL,
                            np.ndarray[f64, ndim=2] evecs_RL,
                            int nE):
    """Spectral propagation score: source through RL eigenmodes onto target.

    s = sum_j (c_j^src * c_j^tgt) / lambda_j where lambda_j > 0,
    c_j^src = <v_j, source>, c_j^tgt = <v_j, target>.

    Parameters
    ----------
    source : f64[nE]
        Source edge signal.
    target : f64[nE]
        Target edge signal.
    evals_RL : f64[nE]
        RL eigenvalues.
    evecs_RL : f64[nE, nE]
        RL eigenvectors.
    nE : int

    Returns
    -------
    float
    """
    return float(_spectral_channel_score(&source[0], &target[0],
                                          &evals_RL[0], &evecs_RL[0, 0], nE))


# Group channel scores

def group_channel_scores(np.ndarray[np.uint8_t, ndim=2] group_masks,
                          np.ndarray[f64, ndim=1] target,
                          np.ndarray[f64, ndim=1] evals_RL,
                          np.ndarray[f64, ndim=2] evecs_RL,
                          np.ndarray[f64, ndim=2] B1,
                          int nV, int nE, int n_groups):
    """Per-group spectral channel scores.

    For each group, builds an edge source from vertex membership
    via B1^T @ mask, normalizes, then computes the spectral
    channel score against the target vector.

    Parameters
    ----------
    group_masks : uint8[n_groups, nV]
        Binary vertex membership per group.
    target : f64[nE]
        Target edge signal.
    evals_RL : f64[nE]
    evecs_RL : f64[nE, nE]
    B1 : f64[nV, nE]
    nV, nE, n_groups : int

    Returns
    -------
    f64[n_groups]
    """
    cdef np.ndarray[f64, ndim=1] scores = np.empty(n_groups, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] v_mask_f = np.empty(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] e_source = np.empty(nE, dtype=np.float64)
    cdef np.uint8_t[:, ::1] gm = group_masks
    cdef f64[::1] vmf = v_mask_f, esv = e_source, sv = scores
    cdef int g, v
    cdef f64 norm

    for g in range(n_groups):
        # Convert uint8 mask to float64
        for v in range(nV):
            vmf[v] = <f64>gm[g, v]

        # Edge source = B1^T @ vertex_mask
        bl_gemv_t(&B1[0, 0], &vmf[0], &esv[0], nV, nE)

        # Normalize
        norm = bl_nrm2(&esv[0], nE)
        if norm > 1e-15:
            for v in range(nE):
                esv[v] /= norm

        sv[g] = _spectral_channel_score(&esv[0], &target[0],
                                          &evals_RL[0], &evecs_RL[0, 0], nE)

    return scores


# Multi-channel profile

def multi_channel_profile(np.ndarray[f64, ndim=1] iv,
                           np.ndarray[f64, ndim=1] primal_char,
                           f64 coverage_val,
                           f64 kappa_mean,
                           f64 efficiency):
    """Assemble a multi-dimensional profile for visualization.

    Combines interfacing vector components, primal signal character,
    spectral coverage, mean coherence, and source efficiency into
    a single dict.

    Parameters
    ----------
    iv : f64[n_channels]
        Interfacing vector.
    primal_char : f64[nhats]
        Primal signal character fractions.
    coverage_val : float
    kappa_mean : float
        Mean vertex coherence kappa.
    efficiency : float

    Returns
    -------
    dict with named fields for each dimension.
    """
    cdef int n_iv = iv.shape[0]
    cdef int n_pc = primal_char.shape[0]

    result = {
        'coverage': float(coverage_val),
        'kappa_mean': float(kappa_mean),
        'efficiency': float(efficiency),
    }

    # Interfacing vector channels
    if n_iv >= 1:
        result['iv_T'] = float(iv[0])
    if n_iv >= 2:
        result['iv_G'] = float(iv[1])
    if n_iv >= 3:
        result['iv_F'] = float(iv[2])
    if n_iv >= 4:
        result['iv_Sch'] = float(iv[3])

    # Primal character channels
    if n_pc >= 1:
        result['pc_T'] = float(primal_char[0])
    if n_pc >= 2:
        result['pc_G'] = float(primal_char[1])
    if n_pc >= 3:
        result['pc_F'] = float(primal_char[2])
    if n_pc >= 4:
        result['pc_C'] = float(primal_char[3])

    return result
