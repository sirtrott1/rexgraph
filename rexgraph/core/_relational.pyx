# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._relational - Relational Laplacian and Green function.

All computation via LAPACK/BLAS through _linalg cimport. Zero Python
in hot paths.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    should_use_dense_eigen,
    can_allocate_dense_f64,
)

from rexgraph.core._linalg cimport (
    lp_eigh, lp_lstsq,
    bl_gemm_nn, bl_gemm_nt, bl_gemm_tn,
    bl_dot,
    spectral_pinv, spectral_pinv_matvec,
    mat_trace,
)

np.import_array()


cdef void _trace_normalize_inplace(f64* L, f64* tr_out, int n) noexcept nogil:
    cdef f64 tr = mat_trace(L, n)
    cdef int i
    cdef f64 inv_tr
    tr_out[0] = tr
    if tr > 1e-15:
        inv_tr = 1.0 / tr
        for i in range(n * n):
            L[i] *= inv_tr
    else:
        memset(L, 0, n * n * sizeof(f64))


def trace_normalize(np.ndarray[f64, ndim=2] L):
    cdef int n = L.shape[0]
    cdef np.ndarray[f64, ndim=2] out = L.copy()
    cdef f64 tr_val = 0
    _trace_normalize_inplace(&out[0, 0], &tr_val, n)
    return out, float(tr_val)


cdef void _build_RL_3(const f64* L1, const f64* L_O, const f64* L_SG,
                       f64* RL, f64* h1, f64* hO, f64* hSG,
                       f64* traces, int n) noexcept nogil:
    cdef int nn = n * n
    cdef int i
    memcpy(h1, L1, nn * sizeof(f64))
    _trace_normalize_inplace(h1, &traces[0], n)
    memcpy(hO, L_O, nn * sizeof(f64))
    _trace_normalize_inplace(hO, &traces[1], n)
    memcpy(hSG, L_SG, nn * sizeof(f64))
    _trace_normalize_inplace(hSG, &traces[2], n)
    for i in range(nn):
        RL[i] = h1[i] + hO[i] + hSG[i]


def build_RL_from_laplacians(np.ndarray[f64, ndim=2] L1,
                               np.ndarray[f64, ndim=2] L_O,
                               np.ndarray[f64, ndim=2] L_SG):
    cdef int nE = L1.shape[0]
    cdef np.ndarray[f64, ndim=2] RL = np.empty((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] h1 = np.empty((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] hO = np.empty((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] hSG = np.empty((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] traces = np.empty(3, dtype=np.float64)
    _build_RL_3(&L1[0, 0], &L_O[0, 0], &L_SG[0, 0],
                &RL[0, 0], &h1[0, 0], &hO[0, 0], &hSG[0, 0],
                &traces[0], nE)
    return {
        'RL': RL, 'hats': [h1, hO, hSG], 'nhats': 3,
        'trace_values': traces, 'hat_names': ['L1_down', 'L_O', 'L_SG'],
    }


def rl_eigen(np.ndarray[f64, ndim=2] RL):
    cdef int n = RL.shape[0]
    cdef np.ndarray[f64, ndim=2] RL_F = np.asfortranarray(RL.copy())
    cdef np.ndarray[f64, ndim=1] evals = np.empty(n, dtype=np.float64)
    cdef int i
    lp_eigh(&RL_F[0, 0], &evals[0], n)
    for i in range(n):
        if evals[i] < 0 and fabs(evals[i]) < 1e-10: evals[i] = 0.0
        if fabs(evals[i]) < 1e-12: evals[i] = 0.0
    cdef np.ndarray[f64, ndim=2] evecs = np.ascontiguousarray(RL_F)
    return evals, evecs


def rl_pinv_dense(np.ndarray[f64, ndim=1] evals, np.ndarray[f64, ndim=2] evecs):
    cdef int n = evals.shape[0]
    cdef np.ndarray[f64, ndim=2] out = np.zeros((n, n), dtype=np.float64)
    spectral_pinv(&evals[0], &evecs[0, 0], &out[0, 0], n, 1e-10)
    return out


def rl_pinv_matvec(np.ndarray[f64, ndim=1] evals,
                    np.ndarray[f64, ndim=2] evecs,
                    np.ndarray[f64, ndim=1] x):
    cdef int n = evals.shape[0]
    cdef np.ndarray[f64, ndim=1] out = np.empty(n, dtype=np.float64)
    spectral_pinv_matvec(&evals[0], &evecs[0, 0], &x[0], &out[0], n, 1e-10)
    return out


def build_green_cache(np.ndarray[f64, ndim=2] RL,
                       np.ndarray[f64, ndim=2] B1,
                       np.ndarray[f64, ndim=1] evals,
                       np.ndarray[f64, ndim=2] evecs):
    cdef int nV = B1.shape[0]
    cdef int nE = B1.shape[1]
    cdef np.ndarray[f64, ndim=2] RLp = np.zeros((nE, nE), dtype=np.float64)
    spectral_pinv(&evals[0], &evecs[0, 0], &RLp[0, 0], nE, 1e-10)
    cdef np.ndarray[f64, ndim=2] B1_RLp = np.empty((nV, nE), dtype=np.float64)
    bl_gemm_nn(&B1[0, 0], &RLp[0, 0], &B1_RLp[0, 0], nV, nE, nE)
    cdef np.ndarray[f64, ndim=2] S0 = np.empty((nV, nV), dtype=np.float64)
    bl_gemm_nt(&B1_RLp[0, 0], &B1[0, 0], &S0[0, 0], nV, nV, nE)
    return {
        'RL_pinv': RLp, 'B1_RLp': B1_RLp, 'S0': S0,
        'evals': evals, 'evecs': evecs, 'nV': nV, 'nE': nE, 'dense': True,
    }


def rl_cg_solve(np.ndarray[f64, ndim=2] RL, np.ndarray[f64, ndim=1] b):
    cdef int n = RL.shape[0]
    cdef np.ndarray[f64, ndim=2] A_F = np.asfortranarray(RL.copy())
    cdef np.ndarray[f64, ndim=1] B = b.copy()
    cdef np.ndarray[f64, ndim=1] S = np.empty(n, dtype=np.float64)
    cdef int rank = 0
    lp_lstsq(&A_F[0, 0], &B[0], n, n, 1, &S[0], &rank)
    return B


def rl_solve_column(np.ndarray[f64, ndim=2] RL, np.ndarray[f64, ndim=2] B1,
                     int vertex_idx):
    cdef int nV = B1.shape[0], nE = B1.shape[1]
    cdef np.ndarray[f64, ndim=1] rhs = np.zeros(nE, dtype=np.float64)
    cdef f64[:, ::1] b1v = B1
    cdef f64[::1] rv = rhs
    cdef int e
    for e in range(nE):
        rv[e] = b1v[vertex_idx, e]
    return rl_cg_solve(RL, rhs)


def build_line_graph(np.ndarray[f64, ndim=2] K1, int nE):
    cdef f64[:, ::1] kv = K1
    cdef int i, j, count = 0
    for i in range(nE):
        for j in range(i + 1, nE):
            if kv[i, j] > 1e-15: count += 1
    cdef np.ndarray[i32, ndim=1] src = np.empty(count, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] tgt = np.empty(count, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] wt = np.empty(count, dtype=np.float64)
    cdef i32[::1] sv = src, tv = tgt
    cdef f64[::1] wv = wt
    cdef int idx = 0
    for i in range(nE):
        for j in range(i + 1, nE):
            if kv[i, j] > 1e-15:
                sv[idx] = i; tv[idx] = j; wv[idx] = kv[i, j]; idx += 1
    return {'src': src, 'tgt': tgt, 'weights': wt, 'nV_L': nE, 'nE_L': count}


def build_L_coPC(line_graph_info):
    cdef int nV_L = line_graph_info['nV_L']
    cdef int nE_L = line_graph_info['nE_L']
    if nE_L == 0:
        return np.zeros((nV_L, nV_L), dtype=np.float64)
    cdef np.ndarray[i32, ndim=1] src = line_graph_info['src']
    cdef np.ndarray[i32, ndim=1] tgt = line_graph_info['tgt']
    cdef np.ndarray[f64, ndim=2] B1_L = np.zeros((nV_L, nE_L), dtype=np.float64)
    cdef f64[:, ::1] blv = B1_L
    cdef i32[::1] sv = src, tv = tgt
    cdef int j
    for j in range(nE_L): blv[sv[j], j] = -1.0; blv[tv[j], j] = 1.0
    cdef np.ndarray[f64, ndim=2] L = np.empty((nV_L, nV_L), dtype=np.float64)
    bl_gemm_tn(&blv[0, 0], &blv[0, 0], &L[0, 0], nV_L, nV_L, nE_L)
    return L
