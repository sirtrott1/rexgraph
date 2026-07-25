# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._interfacing - Interfacing vector and channel scoring.

Maps a set of source vertices through typed response operators and
projects onto a target edge vector to produce per-channel scores.
The interfacing vector I lives on S^{n-1} after normalization and
classifies entities by their structural mechanism.

Response operators:
    S_T = B1 @ L0^+ @ B1^T    (gradient flow through vertex space)
    S_G = L_O                  (overlap co-membership)
    S_F = L_SG                 (frustration sign coherence)
    I_Sch = sum_j |c_j|^2 |<target, v_j>|^2  (Born probability)

Quality gate: q(x) = x / (x + median(|x|)) per channel.

Confidence diagnostics: spectral coverage, source efficiency,
channel conflict detection.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs, log, sqrt, exp
from libc.string cimport memset

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    get_EPSILON_NORM,
    get_EPSILON_DIV,
)

from rexgraph.core._linalg cimport (
    lp_eigh,
    bl_gemm_nn, bl_gemm_nt, bl_gemm_tn,
    bl_gemv_n, bl_gemv_t, bl_symv,
    bl_dot, bl_nrm2, bl_scal,
    spectral_pinv, spectral_pinv_matvec,
)

np.import_array()


# Vertex source

def build_vertex_source(np.ndarray[i32, ndim=1] target_indices,
                         np.ndarray[f64, ndim=1] target_weights,
                         np.ndarray[f64, ndim=1] vertex_weights,
                         int nV):
    """Weighted vertex source vector.

    rho[v] = sum_{t in targets} a(t) * w(t) where a(t) is the
    target weight and w(t) is the vertex weight (e.g. IDF).

    Parameters
    ----------
    target_indices : i32[n_targets]
    target_weights : f64[n_targets]
    vertex_weights : f64[nV]
    nV : int

    Returns
    -------
    f64[nV]
    """
    cdef np.ndarray[f64, ndim=1] rho = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] rv = rho, tw = target_weights, vw = vertex_weights
    cdef i32[::1] ti = target_indices
    cdef int k, n_targets = target_indices.shape[0]
    cdef i32 v

    for k in range(n_targets):
        v = ti[k]
        if v >= 0 and v < nV:
            rv[v] += tw[k] * vw[v]

    return rho


# Edge signal from vertex source

cdef void _build_edge_signal(const f64* rho,
                              const f64* B1,
                              const f64* evals_L0, const f64* evecs_L0,
                              f64* psi_out, f64* L0p_rho_buf,
                              int nV, int nE) noexcept nogil:
    """psi = B1^T @ L0^+ @ rho. All BLAS/LAPACK, no Python.

    L0p_rho_buf: pre-allocated nV workspace for L0^+ @ rho.
    """
    # L0^+ @ rho via spectral pseudoinverse matvec
    spectral_pinv_matvec(evals_L0, evecs_L0, rho, L0p_rho_buf, nV, 1e-10)

    # psi = B1^T @ (L0^+ @ rho)
    bl_gemv_t(B1, L0p_rho_buf, psi_out, nV, nE)


def build_edge_signal(np.ndarray[f64, ndim=1] rho,
                       np.ndarray[f64, ndim=2] B1,
                       np.ndarray[f64, ndim=1] evals_L0,
                       np.ndarray[f64, ndim=2] evecs_L0,
                       int nV, int nE):
    """Edge gradient of vertex Poisson solution: psi = B1^T @ L0^+ @ rho.

    Parameters
    ----------
    rho : f64[nV]
        Vertex source vector.
    B1 : f64[nV, nE]
    evals_L0 : f64[nV]
        L0 eigenvalues from spectral_bundle.
    evecs_L0 : f64[nV, nV]
        L0 eigenvectors from spectral_bundle.
    nV, nE : int

    Returns
    -------
    f64[nE]
    """
    cdef np.ndarray[f64, ndim=1] psi = np.empty(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] buf = np.empty(nV, dtype=np.float64)

    _build_edge_signal(&rho[0], &B1[0, 0],
                        &evals_L0[0], &evecs_L0[0, 0],
                        &psi[0], &buf[0], nV, nE)
    return psi


# Response operators

def build_response_operators(np.ndarray[f64, ndim=2] B1,
                              np.ndarray[f64, ndim=1] evals_L0,
                              np.ndarray[f64, ndim=2] evecs_L0,
                              np.ndarray[f64, ndim=2] L_O,
                              np.ndarray[f64, ndim=2] L_SG,
                              int nV, int nE):
    """Typed response operators for the three structural channels.

    S_T = B1 @ L0^+ @ B1^T    (nE x nE)
    S_G = L_O                  (nE x nE, passed through)
    S_F = L_SG                 (nE x nE, passed through)

    Parameters
    ----------
    B1 : f64[nV, nE]
    evals_L0 : f64[nV]
    evecs_L0 : f64[nV, nV]
    L_O : f64[nE, nE]
    L_SG : f64[nE, nE]
    nV, nE : int

    Returns
    -------
    dict with S_T, S_G, S_F, L0_pinv
    """
    # L0^+ via spectral decomposition
    cdef np.ndarray[f64, ndim=2] L0p = np.zeros((nV, nV), dtype=np.float64)
    spectral_pinv(&evals_L0[0], &evecs_L0[0, 0], &L0p[0, 0], nV, 1e-10)

    # S_T = B1 @ L0^+ @ B1^T
    # First: tmp = B1 @ L0^+ (nV x nE)^T @ (nV x nV) -- wait, we need (nE x nE)
    # S_T = (B1^T)^T @ L0^+ @ B1^T = B1 @ L0^+ @ B1^T but that's nV x nV
    # Actually: S_T operates on edge space. We need B1^T @ L0^+ @ B1 ... no.
    # From the math: S_T maps edge signals to edge signals via gradient flow.
    # S_T = B1^T @ L0^+ @ B1? No, that's (nE x nV)(nV x nV)(nV x nE) = nE x nE. 
    # But the definition says S_T = B1 @ L0^+ @ B1^T which is nV x nV.
    # The interfacing score is I_T = <target | S_T | psi> where target and psi
    # are edge vectors. So S_T must be nE x nE.
    # Correct form: S_T = B1^T @ L0^+ @ B1, giving nE x nE.

    # tmp = L0^+ @ B1 (nV x nE)
    cdef np.ndarray[f64, ndim=2] L0p_B1 = np.empty((nV, nE), dtype=np.float64)
    bl_gemm_nn(&L0p[0, 0], &B1[0, 0], &L0p_B1[0, 0], nV, nE, nV)

    # S_T = B1^T @ L0p_B1 (nE x nE)
    cdef np.ndarray[f64, ndim=2] S_T = np.empty((nE, nE), dtype=np.float64)
    bl_gemm_tn(&B1[0, 0], &L0p_B1[0, 0], &S_T[0, 0], nE, nE, nV)

    return {
        'S_T': S_T,
        'S_G': L_O,
        'S_F': L_SG,
        'L0_pinv': L0p,
    }


# Per-channel scores

cdef void _channel_scores_3(const f64* psi, const f64* target,
                              const f64* S_T, const f64* S_G, const f64* S_F,
                              f64* scores_out, f64* buf,
                              int nE) noexcept nogil:
    """I_X = target^T @ S_X @ psi for X in {T, G, F}. buf is nE workspace."""
    # Channel 0: topological
    bl_gemv_n(S_T, psi, buf, nE, nE)
    scores_out[0] = bl_dot(target, buf, nE)

    # Channel 1: geometric
    bl_gemv_n(S_G, psi, buf, nE, nE)
    scores_out[1] = bl_dot(target, buf, nE)

    # Channel 2: frustration
    bl_gemv_n(S_F, psi, buf, nE, nE)
    scores_out[2] = bl_dot(target, buf, nE)


def channel_scores(np.ndarray[f64, ndim=1] psi,
                    np.ndarray[f64, ndim=2] S_T,
                    np.ndarray[f64, ndim=2] S_G,
                    np.ndarray[f64, ndim=2] S_F,
                    np.ndarray[f64, ndim=1] target,
                    int nE):
    """Per-channel interfacing scores I_X = target^T @ S_X @ psi.

    Parameters
    ----------
    psi : f64[nE]
        Edge signal from build_edge_signal.
    S_T, S_G, S_F : f64[nE, nE]
        Response operators from build_response_operators.
    target : f64[nE]
        Target/phenotype edge vector.
    nE : int

    Returns
    -------
    f64[3]
        Scores for topological, geometric, frustration channels.
    """
    cdef np.ndarray[f64, ndim=1] scores = np.empty(3, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] buf = np.empty(nE, dtype=np.float64)

    _channel_scores_3(&psi[0], &target[0],
                        &S_T[0, 0], &S_G[0, 0], &S_F[0, 0],
                        &scores[0], &buf[0], nE)
    return scores


# Schrodinger channel score

cdef f64 _schrodinger_score(const f64* psi, const f64* target,
                              const f64* evals_RL, const f64* evecs_RL,
                              int nE) noexcept nogil:
    """I_Sch = sum_j |c_j|^2 * |<target, v_j>|^2.

    c_j = <v_j, psi>, time-averaged Born probability projected onto target.
    """
    cdef int j, i
    cdef f64 cj, tj, result = 0.0

    for j in range(nE):
        if evals_RL[j] < 1e-10:
            continue
        # c_j = <v_j, psi>
        cj = 0.0
        for i in range(nE):
            cj += evecs_RL[i * nE + j] * psi[i]
        # t_j = <v_j, target>
        tj = 0.0
        for i in range(nE):
            tj += evecs_RL[i * nE + j] * target[i]
        result += cj * cj * tj * tj

    return result


def schrodinger_score(np.ndarray[f64, ndim=1] psi,
                       np.ndarray[f64, ndim=1] evals_RL,
                       np.ndarray[f64, ndim=2] evecs_RL,
                       np.ndarray[f64, ndim=1] target,
                       int nE):
    """Schrodinger channel: time-averaged Born probability on target.

    I_Sch = sum_j |c_j|^2 * |<target, v_j>|^2 where c_j = <v_j, psi>.
    Only eigenmodes with evals > 0 contribute.

    Parameters
    ----------
    psi : f64[nE]
    evals_RL : f64[nE]
    evecs_RL : f64[nE, nE]
    target : f64[nE]
    nE : int

    Returns
    -------
    float
    """
    return float(_schrodinger_score(&psi[0], &target[0],
                                     &evals_RL[0], &evecs_RL[0, 0], nE))


# Quality gate

def quality_gate(np.ndarray[f64, ndim=2] scores):
    """Bayesian quality gate: q(x) = x / (x + median(|x|)).

    Applied per-channel (column) across all entities (rows).

    Parameters
    ----------
    scores : f64[n_entities, n_channels]

    Returns
    -------
    f64[n_entities, n_channels]
    """
    cdef int n_ent = scores.shape[0]
    cdef int n_ch = scores.shape[1]
    cdef np.ndarray[f64, ndim=2] out = np.empty((n_ent, n_ch), dtype=np.float64)
    cdef f64[:, ::1] sv = scores, ov = out
    cdef int d, c
    cdef f64 med, x

    # Per-channel median of absolute values
    cdef np.ndarray[f64, ndim=1] abs_col
    for c in range(n_ch):
        abs_col = np.abs(scores[:, c])
        med = float(np.median(abs_col))
        if med < 1e-30:
            med = 1e-30
        for d in range(n_ent):
            x = sv[d, c]
            ov[d, c] = x / (x + med) if (x + med) > 1e-30 else 0.0

    return out


# Interfacing vector assembly

def interfacing_vector(np.ndarray[f64, ndim=1] scores,
                        np.ndarray[f64, ndim=1] quality):
    """I = scores * quality elementwise.

    Parameters
    ----------
    scores : f64[n_channels]
    quality : f64[n_channels]

    Returns
    -------
    f64[n_channels]
    """
    cdef int n = scores.shape[0]
    cdef np.ndarray[f64, ndim=1] iv = np.empty(n, dtype=np.float64)
    cdef f64[::1] sv = scores, qv = quality, rv = iv
    cdef int k

    for k in range(n):
        rv[k] = sv[k] * qv[k]

    return iv


def sphere_position(np.ndarray[f64, ndim=1] iv):
    """Project to unit sphere: iv / ||iv||.

    Parameters
    ----------
    iv : f64[n_channels]

    Returns
    -------
    f64[n_channels]
    """
    cdef int n = iv.shape[0]
    cdef f64 norm = bl_nrm2(&iv[0], n)
    cdef np.ndarray[f64, ndim=1] pos = iv.copy()

    if norm > 1e-30:
        bl_scal(1.0 / norm, &pos[0], n)

    return pos


# Spectral coverage

cdef f64 _coverage(const f64* psi, const f64* evals_RL,
                    const f64* evecs_RL,
                    int nE, f64 floor_val) noexcept nogil:
    """Fraction of RL eigenmodes with |c_j| > floor_val."""
    cdef int j, i, count = 0, total = 0
    cdef f64 cj

    for j in range(nE):
        if evals_RL[j] < 1e-10:
            continue
        total += 1
        cj = 0.0
        for i in range(nE):
            cj += evecs_RL[i * nE + j] * psi[i]
        if fabs(cj) > floor_val:
            count += 1

    if total == 0:
        return 0.0
    return <f64>count / <f64>total


def coverage(np.ndarray[f64, ndim=1] psi,
              np.ndarray[f64, ndim=1] evals_RL,
              np.ndarray[f64, ndim=2] evecs_RL,
              int nE, f64 probe_floor):
    """Spectral coverage: fraction of RL eigenmodes activated by psi.

    C = count(|c_j| > probe_floor) / count(lambda_j > 0).

    Parameters
    ----------
    psi : f64[nE]
    evals_RL : f64[nE]
    evecs_RL : f64[nE, nE]
    nE : int
    probe_floor : float
        Minimum projection magnitude. Typically 1 / nV^3.

    Returns
    -------
    float in [0, 1]
    """
    return float(_coverage(&psi[0], &evals_RL[0], &evecs_RL[0, 0],
                            nE, probe_floor))


def poisson_floor():
    """Minimum acceptable coverage: 1 - 1/e = 0.6321."""
    return 1.0 - exp(-1.0)


# Source efficiency

def source_efficiency(np.ndarray[i32, ndim=1] target_indices,
                       np.ndarray[f64, ndim=2] B1,
                       int nV, int nE):
    """Fraction of boundary entries that are activating (positive).

    eta = n_positive / (n_positive + n_negative) across all edges
    incident to target vertices.

    Parameters
    ----------
    target_indices : i32[n_targets]
    B1 : f64[nV, nE]
    nV, nE : int

    Returns
    -------
    float in [0, 1]
    """
    cdef i32[::1] ti = target_indices
    cdef f64[:, ::1] b1v = B1
    cdef int k, e, n_targets = target_indices.shape[0]
    cdef i32 v
    cdef int n_pos = 0, n_neg = 0
    cdef f64 val

    for k in range(n_targets):
        v = ti[k]
        if v < 0 or v >= nV:
            continue
        for e in range(nE):
            val = b1v[v, e]
            if val > 1e-15:
                n_pos += 1
            elif val < -1e-15:
                n_neg += 1

    cdef int total = n_pos + n_neg
    if total == 0:
        return 0.5
    return <f64>n_pos / <f64>total


# Confidence flags

def confidence_flags(f64 coverage_val, f64 efficiency, f64 phi_T):
    """Confidence diagnostics from coverage and efficiency.

    Flags:
        CONFIDENT - coverage >= Poisson floor and no channel conflict
        LOW_SIGNAL - coverage < 1 - 1/e
        CHANNEL_CONFLICT - efficiency < 0.5 and phi_T < 2/3

    Parameters
    ----------
    coverage_val : float
    efficiency : float
    phi_T : float
        Topological channel fraction of vertex character.

    Returns
    -------
    dict with flag, reasons list
    """
    cdef f64 pf = 1.0 - exp(-1.0)
    reasons = []

    if coverage_val < pf:
        reasons.append('LOW_SIGNAL')
    if efficiency < 0.5 and phi_T < 2.0 / 3.0:
        reasons.append('CHANNEL_CONFLICT')

    if len(reasons) == 0:
        return {'flag': 'CONFIDENT', 'reasons': []}
    return {'flag': reasons[0], 'reasons': reasons}


# One-call entry point

def build_interfacing_bundle(np.ndarray[i32, ndim=1] target_indices,
                              np.ndarray[f64, ndim=1] target_weights,
                              np.ndarray[f64, ndim=1] vertex_weights,
                              np.ndarray[f64, ndim=2] B1,
                              np.ndarray[f64, ndim=1] evals_L0,
                              np.ndarray[f64, ndim=2] evecs_L0,
                              np.ndarray[f64, ndim=2] L_O,
                              np.ndarray[f64, ndim=2] L_SG,
                              np.ndarray[f64, ndim=1] evals_RL,
                              np.ndarray[f64, ndim=2] evecs_RL,
                              np.ndarray[f64, ndim=1] target,
                              int nV, int nE):
    """Full interfacing vector computation.

    Chains: vertex source -> edge signal -> response operators ->
    per-channel scores -> Schrodinger score -> sphere position ->
    coverage -> efficiency -> confidence.

    Parameters
    ----------
    target_indices : i32[n_targets]
        Vertex indices of source entity targets.
    target_weights : f64[n_targets]
        Per-target weights (e.g. binding affinity).
    vertex_weights : f64[nV]
        Per-vertex weights (e.g. IDF).
    B1 : f64[nV, nE]
    evals_L0 : f64[nV]
        L0 eigenvalues from spectral_bundle.
    evecs_L0 : f64[nV, nV]
        L0 eigenvectors from spectral_bundle.
    L_O : f64[nE, nE]
        Overlap Laplacian.
    L_SG : f64[nE, nE]
        Frustration Laplacian.
    evals_RL : f64[nE]
        RL eigenvalues.
    evecs_RL : f64[nE, nE]
        RL eigenvectors.
    target : f64[nE]
        Target/phenotype edge vector.
    nV, nE : int

    Returns
    -------
    dict
        rho : f64[nV], vertex source
        psi : f64[nE], edge signal
        scores : f64[3], per-channel (T, G, F)
        schrodinger : float, Schrodinger channel score
        iv : f64[4], interfacing vector (T, G, F, Sch)
        sphere_pos : f64[4], unit sphere position
        signal_magnitude : float, ||psi||
        coverage : float
        efficiency : float
        confidence : dict
    """
    # Source
    rho = build_vertex_source(target_indices, target_weights,
                               vertex_weights, nV)

    # Edge signal
    cdef np.ndarray[f64, ndim=1] psi = build_edge_signal(
        rho, B1, evals_L0, evecs_L0, nV, nE)

    cdef f64 sig_mag = bl_nrm2(&psi[0], nE)

    # Response operators
    ops = build_response_operators(B1, evals_L0, evecs_L0, L_O, L_SG, nV, nE)

    # Channel scores
    scores_3 = channel_scores(psi, ops['S_T'], ops['S_G'], ops['S_F'],
                               target, nE)

    # Schrodinger score
    sch = schrodinger_score(psi, evals_RL, evecs_RL, target, nE)

    # Assemble 4-channel vector (T, G, F, Sch)
    cdef np.ndarray[f64, ndim=1] iv_raw = np.empty(4, dtype=np.float64)
    iv_raw[0] = scores_3[0]
    iv_raw[1] = scores_3[1]
    iv_raw[2] = scores_3[2]
    iv_raw[3] = sch

    # Sphere position
    sp = sphere_position(iv_raw)

    # Coverage
    cdef f64 pf_val = 1.0 / (<f64>nV * <f64>nV * <f64>nV) if nV > 0 else 1e-10
    cov = coverage(psi, evals_RL, evecs_RL, nE, pf_val)

    # Efficiency
    eff = source_efficiency(target_indices, B1, nV, nE)

    # Confidence
    conf = confidence_flags(cov, eff, float(sp[0]) if sp.shape[0] > 0 else 0.0)

    return {
        'rho': rho,
        'psi': psi,
        'scores': scores_3,
        'schrodinger': float(sch),
        'iv': iv_raw,
        'sphere_pos': sp,
        'signal_magnitude': float(sig_mag),
        'coverage': float(cov),
        'efficiency': float(eff),
        'confidence': conf,
    }
