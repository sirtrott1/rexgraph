# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._linalg - LAPACK/BLAS runtime and workspace.

Allocates the static workspace buffer used by all LAPACK calls.
Provides Python-callable wrappers for testing.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.math cimport fabs

from rexgraph.core._linalg cimport *

np.import_array()

# Static workspace buffer for dsyev_ (allocated once at module load)
cdef double _work_static[WORK_SIZE]
_lp_work = _work_static


# Python-callable eigensolve

def eigh(np.ndarray[f64, ndim=2] A_in):
    """Symmetric eigendecomposition via LAPACK dsyev_.

    Parameters
    ----------
    A_in : f64[n, n], symmetric.

    Returns
    -------
    (evals f64[n], evecs f64[n, n]) sorted ascending.
    Eigenvectors in columns of evecs (row-major: evecs[:, k] is eigenvector k).
    """
    cdef int n = A_in.shape[0]

    # dsyev_ needs column-major (Fortran order)
    cdef np.ndarray[f64, ndim=2] A_F = np.asfortranarray(A_in.copy())
    cdef np.ndarray[f64, ndim=1] evals = np.empty(n, dtype=np.float64)

    lp_eigh(&A_F[0, 0], &evals[0], n)

    # Clean eigenvalues
    cdef int i
    for i in range(n):
        if evals[i] < 0 and fabs(evals[i]) < 1e-10:
            evals[i] = 0.0

    # Convert to row-major (eigenvectors in columns)
    cdef np.ndarray[f64, ndim=2] evecs = np.ascontiguousarray(A_F)
    return evals, evecs


# Python-callable SVD

def svd(np.ndarray[f64, ndim=2] A_in):
    """General SVD via LAPACK dgesvd_.

    Returns (U, S, Vt) where A = U @ diag(S) @ Vt.
    """
    cdef int m = A_in.shape[0]
    cdef int n = A_in.shape[1]
    cdef int mn = m if m < n else n

    cdef np.ndarray[f64, ndim=2] A_F = np.asfortranarray(A_in.copy())
    cdef np.ndarray[f64, ndim=1] S = np.empty(mn, dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] U = np.empty((m, m), dtype=np.float64, order='F')
    cdef np.ndarray[f64, ndim=2] Vt = np.empty((n, n), dtype=np.float64, order='F')

    lp_svd(&A_F[0, 0], &S[0], &U[0, 0], &Vt[0, 0], m, n)

    return np.ascontiguousarray(U), S, np.ascontiguousarray(Vt)


# Python-callable least squares

def lstsq(np.ndarray[f64, ndim=2] A_in, np.ndarray[f64, ndim=1] b_in):
    """Least squares via LAPACK dgelsd_.

    Solves min ||A @ x - b||_2.
    Returns (x, rank).
    """
    cdef int m = A_in.shape[0]
    cdef int n = A_in.shape[1]
    cdef int nrhs = 1
    cdef int mn = m if m < n else n

    cdef np.ndarray[f64, ndim=2] A_F = np.asfortranarray(A_in.copy())
    # B must be max(m, n) for dgelsd
    cdef int ldb = m if m > n else n
    cdef np.ndarray[f64, ndim=1] B = np.zeros(ldb, dtype=np.float64)
    B[:m] = b_in

    cdef np.ndarray[f64, ndim=1] S = np.empty(mn, dtype=np.float64)
    cdef int rank = 0

    cdef int info = lp_lstsq(&A_F[0, 0], &B[0], m, n, nrhs, &S[0], &rank)

    return B[:n].copy(), rank


# Python-callable matrix rank

def matrix_rank(np.ndarray[f64, ndim=2] A_in, double tol=1e-10):
    """Matrix rank via SVD."""
    cdef int m = A_in.shape[0]
    cdef int n = A_in.shape[1]
    cdef np.ndarray[f64, ndim=2] A_F = np.asfortranarray(A_in.copy())
    return compute_rank_svd(&A_F[0, 0], m, n, tol)


# Python-callable matrix multiply

def gemm_nn(np.ndarray[f64, ndim=2] A, np.ndarray[f64, ndim=2] B):
    """C = A @ B via BLAS dgemm."""
    cdef int M = A.shape[0]
    cdef int K = A.shape[1]
    cdef int N = B.shape[1]
    cdef np.ndarray[f64, ndim=2] C = np.empty((M, N), dtype=np.float64)
    bl_gemm_nn(&A[0, 0], &B[0, 0], &C[0, 0], M, N, K)
    return C


def gemm_nt(np.ndarray[f64, ndim=2] A, np.ndarray[f64, ndim=2] B):
    """C = A @ B^T via BLAS dgemm."""
    cdef int M = A.shape[0]
    cdef int K = A.shape[1]
    cdef int N = B.shape[0]  # B^T is K x N, so B is N x K
    cdef np.ndarray[f64, ndim=2] C = np.empty((M, N), dtype=np.float64)
    bl_gemm_nt(&A[0, 0], &B[0, 0], &C[0, 0], M, N, K)
    return C


def gemm_tn(np.ndarray[f64, ndim=2] A, np.ndarray[f64, ndim=2] B):
    """C = A^T @ B via BLAS dgemm."""
    cdef int K = A.shape[0]
    cdef int M = A.shape[1]  # A^T is M x K
    cdef int N = B.shape[1]
    cdef np.ndarray[f64, ndim=2] C = np.empty((M, N), dtype=np.float64)
    bl_gemm_tn(&A[0, 0], &B[0, 0], &C[0, 0], M, N, K)
    return C


# Python-callable spectral pseudoinverse

def pinv_spectral(np.ndarray[f64, ndim=1] evals,
                   np.ndarray[f64, ndim=2] evecs,
                   double tol=1e-10):
    """RL^+ from eigendecomposition. evecs[:, k] = eigenvector k."""
    cdef int n = evals.shape[0]
    cdef np.ndarray[f64, ndim=2] out = np.zeros((n, n), dtype=np.float64)
    spectral_pinv(&evals[0], &evecs[0, 0], &out[0, 0], n, tol)
    return out


def pinv_matvec(np.ndarray[f64, ndim=1] evals,
                 np.ndarray[f64, ndim=2] evecs,
                 np.ndarray[f64, ndim=1] x,
                 double tol=1e-10):
    """RL^+ @ x without forming RL^+."""
    cdef int n = evals.shape[0]
    cdef np.ndarray[f64, ndim=1] out = np.empty(n, dtype=np.float64)
    spectral_pinv_matvec(&evals[0], &evecs[0, 0], &x[0], &out[0], n, tol)
    return out


# Full RL pipeline (C-level, no Python in hot path)

def rl_pipeline(np.ndarray[f64, ndim=2] B1,
                np.ndarray[f64, ndim=2] L1,
                np.ndarray[f64, ndim=2] L_O,
                np.ndarray[f64, ndim=2] L_SG):
    """Full RL -> chi -> phi -> kappa pipeline. All LAPACK/BLAS, zero Python.

    Parameters
    ----------
    B1 : f64[nV, nE]
    L1, L_O, L_SG : f64[nE, nE]

    Returns
    -------
    dict with RL, chi, phi, kappa, evals, evecs, RLp, B1_RLp, S0_diag, hats
    """
    cdef int nV = B1.shape[0]
    cdef int nE = B1.shape[1]
    cdef int nhats = 3
    cdef int i, j, k, v, e
    cdef f64 tr_val, inv_lam, rl_ee, s0_vv

    # Allocate all output arrays
    cdef np.ndarray[f64, ndim=2] h1 = np.empty((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] hO = np.empty((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] hSG = np.empty((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] RL = np.empty((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] evals = np.empty(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] evecs
    cdef np.ndarray[f64, ndim=2] RLp = np.zeros((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] chi = np.zeros((nE, nhats), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] B1_RLp = np.empty((nV, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] S0_diag = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] phi = np.zeros((nV, nhats), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] chi_star = np.zeros((nV, nhats), dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] kappa = np.empty(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] tmp = np.empty((nV, nE), dtype=np.float64)

    # Typed memoryviews
    cdef f64[:, ::1] h1v=h1, hOv=hO, hSGv=hSG, rlv=RL
    cdef f64[:, ::1] b1v=B1, l1v=L1, lov=L_O, lsgv=L_SG
    cdef f64[::1] ev=evals
    cdef f64[:, ::1] rlpv=RLp, chiv=chi, brv=B1_RLp, phiv=phi, csv=chi_star, tv=tmp
    cdef f64[::1] s0d=S0_diag, kv=kappa

    # Trace normalize L1
    tr_val = 0
    for i in range(nE): tr_val += l1v[i, i]
    if tr_val > 1e-15:
        for i in range(nE):
            for j in range(nE):
                h1v[i, j] = l1v[i, j] / tr_val
    else:
        memset(&h1v[0, 0], 0, nE * nE * sizeof(f64))

    # Trace normalize L_O
    tr_val = 0
    for i in range(nE): tr_val += lov[i, i]
    if tr_val > 1e-15:
        for i in range(nE):
            for j in range(nE):
                hOv[i, j] = lov[i, j] / tr_val
    else:
        memset(&hOv[0, 0], 0, nE * nE * sizeof(f64))

    # Trace normalize L_SG
    tr_val = 0
    for i in range(nE): tr_val += lsgv[i, i]
    if tr_val > 1e-15:
        for i in range(nE):
            for j in range(nE):
                hSGv[i, j] = lsgv[i, j] / tr_val
    else:
        memset(&hSGv[0, 0], 0, nE * nE * sizeof(f64))

    # RL = h1 + hO + hSG
    for i in range(nE):
        for j in range(nE):
            rlv[i, j] = h1v[i, j] + hOv[i, j] + hSGv[i, j]

    # Eigendecompose RL via LAPACK dsyev_
    cdef np.ndarray[f64, ndim=2] RL_F = np.asfortranarray(RL.copy())
    lp_eigh(&RL_F[0, 0], &ev[0], nE)
    for i in range(nE):
        if ev[i] < 0: ev[i] = 0
        if fabs(ev[i]) < 1e-12: ev[i] = 0
    evecs = np.ascontiguousarray(RL_F)
    cdef f64[:, ::1] ecv = evecs

    # RL^+ via spectral decomposition
    spectral_pinv(&ev[0], &ecv[0, 0], &rlpv[0, 0], nE, 1e-10)

    # Chi (diagonal extraction)
    for e in range(nE):
        rl_ee = rlv[e, e]
        if rl_ee > 1e-15:
            chiv[e, 0] = h1v[e, e] / rl_ee
            chiv[e, 1] = hOv[e, e] / rl_ee
            chiv[e, 2] = hSGv[e, e] / rl_ee

    # B1 @ RLp via BLAS
    bl_gemm_nn(&b1v[0, 0], &rlpv[0, 0], &brv[0, 0], nV, nE, nE)

    # S0 diagonal = einsum('ve,ve->v', B1_RLp, B1)
    for v in range(nV):
        for e in range(nE):
            s0d[v] += brv[v, e] * b1v[v, e]

    # Phi: for each hat k, diag(B1_RLp @ hat_k @ B1_RLp^T)
    cdef f64* hat_ptrs[3]
    hat_ptrs[0] = &h1v[0, 0]
    hat_ptrs[1] = &hOv[0, 0]
    hat_ptrs[2] = &hSGv[0, 0]

    for k in range(nhats):
        # tmp = B1_RLp @ hat_k via BLAS
        bl_gemm_nn(&brv[0, 0], hat_ptrs[k], &tv[0, 0], nV, nE, nE)
        # phi[:, k] = diag(tmp @ B1_RLp^T) = einsum('ve,ve->v', tmp, B1_RLp)
        for v in range(nV):
            s0_vv = s0d[v]
            if fabs(s0_vv) > 1e-15:
                phiv[v, k] = 0
                for e in range(nE):
                    phiv[v, k] += tv[v, e] * brv[v, e]
                phiv[v, k] /= s0_vv

    # Chi-star: mean of chi over incident edges (via |B1|)
    cdef f64 deg_v
    for v in range(nV):
        deg_v = 0
        for e in range(nE):
            if fabs(b1v[v, e]) > 0.5:
                deg_v += 1
                for k in range(nhats):
                    csv[v, k] += chiv[e, k]
        if deg_v > 0:
            for k in range(nhats):
                csv[v, k] /= deg_v

    # Kappa = 1 - 0.5 * ||phi - chi_star||_1
    cdef f64 l1_norm
    for v in range(nV):
        l1_norm = 0
        for k in range(nhats):
            l1_norm += fabs(phiv[v, k] - csv[v, k])
        kv[v] = 1.0 - 0.5 * l1_norm

    return {
        'RL': RL, 'evals': evals, 'evecs': evecs, 'RLp': RLp,
        'chi': chi, 'phi': phi, 'chi_star': chi_star, 'kappa': kappa,
        'B1_RLp': B1_RLp, 'S0_diag': S0_diag,
        'hats': [h1, hO, hSG], 'nhats': nhats,
    }
