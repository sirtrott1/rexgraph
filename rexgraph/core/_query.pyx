# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._query - Relational complex query engine.

Predicate masking (SELECT WHERE), signal imputation (INSERT missing),
spectral propagation (AGGREGATE), cell explanation (EXPLAIN).
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    should_use_dense_eigen,
    get_EPSILON_NORM,
    get_EPSILON_DIV,
)

np.import_array()

# Predicate ops
cdef enum:
    _PRED_GT = 0
    _PRED_GE = 1
    _PRED_LT = 2
    _PRED_LE = 3
    _PRED_EQ = 4
    _PRED_NE = 5
    _PRED_BETWEEN = 6

PRED_GT = _PRED_GT
PRED_GE = _PRED_GE
PRED_LT = _PRED_LT
PRED_LE = _PRED_LE
PRED_EQ = _PRED_EQ
PRED_NE = _PRED_NE
PRED_BETWEEN = _PRED_BETWEEN


# Predicate masking

@cython.boundscheck(False)
@cython.wraparound(False)
def predicate_mask(np.ndarray[f64, ndim=1] values, Py_ssize_t n,
                    int op, f64 threshold, f64 threshold_high=0.0):
    """Apply predicate to values array. Returns (mask, count)."""
    cdef np.ndarray[np.uint8_t, ndim=1] mask = np.zeros(n, dtype=np.uint8)
    cdef np.uint8_t[::1] mv = mask
    cdef f64[::1] vv = values
    cdef Py_ssize_t i
    cdef int count = 0
    cdef f64 v

    for i in range(n):
        v = vv[i]
        if op == _PRED_GT and v > threshold:
            mv[i] = 1; count += 1
        elif op == _PRED_GE and v >= threshold:
            mv[i] = 1; count += 1
        elif op == _PRED_LT and v < threshold:
            mv[i] = 1; count += 1
        elif op == _PRED_LE and v <= threshold:
            mv[i] = 1; count += 1
        elif op == _PRED_EQ and fabs(v - threshold) < 1e-10:
            mv[i] = 1; count += 1
        elif op == _PRED_NE and fabs(v - threshold) > 1e-10:
            mv[i] = 1; count += 1
        elif op == _PRED_BETWEEN and v >= threshold and v <= threshold_high:
            mv[i] = 1; count += 1

    return mask, count


def chi_mask(chi, Py_ssize_t nE, Py_ssize_t nhats,
              Py_ssize_t channel, int op, f64 threshold):
    """Predicate on chi[:, channel]."""
    vals = np.asarray(chi[:, channel], dtype=np.float64)
    return predicate_mask(vals, nE, op, threshold)


def phi_mask(phi, Py_ssize_t nV, Py_ssize_t nhats,
              Py_ssize_t channel, int op, f64 threshold):
    """Predicate on phi[:, channel]."""
    vals = np.asarray(phi[:, channel], dtype=np.float64)
    return predicate_mask(vals, nV, op, threshold)


def kappa_mask(kappa, Py_ssize_t nV, int op, f64 threshold):
    """Predicate on kappa."""
    return predicate_mask(np.asarray(kappa, dtype=np.float64), nV, op, threshold)


def mask_and(np.ndarray[np.uint8_t, ndim=1] a,
              np.ndarray[np.uint8_t, ndim=1] b, Py_ssize_t n):
    return np.logical_and(a, b).astype(np.uint8)

def mask_or(np.ndarray[np.uint8_t, ndim=1] a,
             np.ndarray[np.uint8_t, ndim=1] b, Py_ssize_t n):
    return np.logical_or(a, b).astype(np.uint8)

def mask_not(np.ndarray[np.uint8_t, ndim=1] a, Py_ssize_t n):
    return np.logical_not(a).astype(np.uint8)


# Signal imputation

def signal_impute(RL, np.ndarray[f64, ndim=1] observed_signal,
                   np.ndarray[np.uint8_t, ndim=1] observed_mask,
                   Py_ssize_t nE):
    """Impute missing signal values via harmonic interpolation.

    g_missing = -RL_mm^+ RL_mo g_observed.
    Adaptive: dense pinv if n_missing <= dense limit, CG otherwise.
    """

    obs_idx = np.where(observed_mask)[0]
    mis_idx = np.where(~observed_mask.astype(bool))[0]
    n_obs = len(obs_idx)
    n_mis = len(mis_idx)

    imputed = observed_signal.copy()
    confidence = np.ones(nE, dtype=np.float64)

    if n_mis == 0:
        return {'imputed': imputed, 'confidence': confidence,
                'residual': 0.0, 'n_observed': n_obs, 'n_imputed': 0}

    if n_obs == 0:
        return {'imputed': imputed, 'confidence': np.zeros(nE, dtype=np.float64),
                'residual': 0.0, 'n_observed': 0, 'n_imputed': n_mis}

    # Extract RL as dense for subblock extraction
    RL_d = np.asarray(RL, dtype=np.float64)

    RL_mm = RL_d[np.ix_(mis_idx, mis_idx)]
    RL_mo = RL_d[np.ix_(mis_idx, obs_idx)]
    g_obs = observed_signal[obs_idx]

    if should_use_dense_eigen(n_mis):
        from rexgraph.core._linalg import pinv_spectral, eigh as lp_eigh
        ev_mm, evec_mm = lp_eigh(RL_mm)
        RL_mm_pinv = pinv_spectral(ev_mm, evec_mm)
        g_imputed = -RL_mm_pinv @ RL_mo @ g_obs
    else:
        from scipy.sparse.linalg import cg
        rhs = -RL_mo @ g_obs
        g_imputed, _ = cg(RL_mm, rhs, tol=1e-10, maxiter=1000)

    imputed[mis_idx] = g_imputed

    # Confidence from RL_mm diagonal
    diag_mm = np.diag(RL_mm)
    max_diag = np.max(diag_mm) if len(diag_mm) > 0 else 1.0
    for i, mi in enumerate(mis_idx):
        confidence[mi] = diag_mm[i] / max_diag if max_diag > 1e-15 else 0.0

    # Residual at observed positions
    residual = float(np.sqrt(np.sum((RL_d[np.ix_(obs_idx, range(nE))] @ imputed) ** 2)))

    return {'imputed': imputed, 'confidence': confidence,
            'residual': residual, 'n_observed': n_obs, 'n_imputed': n_mis}


# Spectral propagation

def spectral_propagate(RL, hats, Py_ssize_t nhats,
                        np.ndarray[f64, ndim=1] source,
                        np.ndarray[f64, ndim=1] target,
                        Py_ssize_t nE):
    """score = source^T RL^+ target / (||source|| ||target||)."""
    from rexgraph.core._relational import rl_eigen, rl_pinv_matvec

    evals, evecs = rl_eigen(RL)
    propagated = rl_pinv_matvec(evals, evecs, source)

    ns = float(np.sqrt(np.dot(source, source)))
    nt = float(np.sqrt(np.dot(target, target)))
    score = float(np.dot(propagated, target)) / (ns * nt) if ns > 1e-15 and nt > 1e-15 else 0.0

    # Per-channel scores
    typed_scores = np.zeros(nhats, dtype=np.float64)
    for k in range(nhats):
        hat_k = hats[k]
        if False:
            ch_prop = hat_k.dot(propagated)
        else:
            ch_prop = np.asarray(hat_k, dtype=np.float64) @ propagated
        typed_scores[k] = float(np.dot(source, ch_prop))

    # Coverage
    n_modes = np.sum(evals > 1e-10)
    coeffs = evecs[:, evals > 1e-10].T @ source
    n_covered = np.sum(np.abs(coeffs) > 1e-10)
    coverage = float(n_covered) / float(n_modes) if n_modes > 0 else 0.0

    return {
        'score': score,
        'typed_scores': typed_scores,
        'energy': float(source @ np.asarray(RL, dtype=np.float64) @ source),
        'coverage': coverage,
    }


# Explain edge

def explain_edge(B1, B2, K1, RL, hats, Py_ssize_t nhats,
                  Py_ssize_t edge_idx,
                  Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF):
    """Full diagnostic for a single edge."""

    # Dense versions for simple access
    B1_d = np.asarray(B1, dtype=np.float64)

    # Below: boundary vertices
    below = list(np.where(np.abs(B1_d[:, edge_idx]) > 0.5)[0])

    # Above: co-boundary faces
    above = []
    if nF > 0:
        B2_d = np.asarray(B2, dtype=np.float64)
        above = list(np.where(np.abs(B2_d[edge_idx, :]) > 0.5)[0])

    # Lateral: K1 neighbors
    if hasattr(K1, 'toarray'):
        K1_d = np.asarray(K1.toarray(), dtype=np.float64)
    else:
        K1_d = np.asarray(K1, dtype=np.float64)
    lateral = [int(e) for e in range(nE) if e != edge_idx and K1_d[edge_idx, e] > 0.5]

    # Chi
    rl_diag = np.diag(np.asarray(RL, dtype=np.float64))
    hat_diags = [np.diag(np.asarray(hats[k], dtype=np.float64)) for k in range(nhats)]
    chi = np.zeros(nhats, dtype=np.float64)
    if rl_diag[edge_idx] > 1e-15:
        for k in range(nhats):
            chi[k] = hat_diags[k][edge_idx] / rl_diag[edge_idx]

    dominant = int(np.argmax(chi))

    # R_self
    from rexgraph.core._relational import rl_eigen, rl_pinv_dense
    evals, evecs = rl_eigen(RL)
    if should_use_dense_eigen(nE):
        RLp = rl_pinv_dense(evals, evecs)
        r_self = float(RLp[edge_idx, edge_idx])
    else:
        r_self = float('nan')

    return {
        'below': np.array(below, dtype=np.int32),
        'above': np.array(above, dtype=np.int32),
        'lateral': np.array(lateral, dtype=np.int32),
        'chi': chi,
        'dominant_channel': dominant,
        'effective_resistance': r_self,
        'n_incident_faces': len(above),
        'degree': len(below),
    }


# Explain vertex

def explain_vertex(B1, RL, hats, Py_ssize_t nhats,
                    phi, kappa, chi_edges,
                    v2e_ptr, v2e_idx,
                    Py_ssize_t vertex_idx,
                    Py_ssize_t nV, Py_ssize_t nE):
    """Full diagnostic for a single vertex."""

    phi_v = np.asarray(phi[vertex_idx], dtype=np.float64)

    # Chi-star for this vertex
    vp = np.asarray(v2e_ptr, dtype=np.int32)
    vi = np.asarray(v2e_idx, dtype=np.int32)
    lo = int(vp[vertex_idx])
    hi = int(vp[vertex_idx + 1])
    chi_arr = np.asarray(chi_edges, dtype=np.float64)

    chi_star_v = np.zeros(nhats, dtype=np.float64)
    cnt = hi - lo
    if cnt > 0:
        for j in range(lo, hi):
            e = int(vi[j])
            chi_star_v += chi_arr[e]
        chi_star_v /= cnt

    kappa_v = float(kappa[vertex_idx])

    # Discrepancy analysis
    gaps = np.abs(phi_v - chi_star_v)
    disc_ch = int(np.argmax(gaps))
    dom_ch = int(np.argmax(phi_v))

    # Incident edges
    incident = [int(vi[j]) for j in range(lo, hi)]

    # Neighbor vertices
    B1_d = np.asarray(B1, dtype=np.float64)

    neighbors = set()
    for e in incident:
        for v in range(nV):
            if v != vertex_idx and abs(B1_d[v, e]) > 0.5:
                neighbors.add(v)

    return {
        'phi': phi_v,
        'chi_star': chi_star_v,
        'kappa': kappa_v,
        'discrepant_channel': disc_ch,
        'channel_gap': float(gaps[disc_ch]),
        'dominant_channel': dom_ch,
        'degree': cnt,
        'incident_edges': np.array(incident, dtype=np.int32),
        'neighbor_vertices': np.array(sorted(neighbors), dtype=np.int32),
    }
