# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._laplacians - Hodge Laplacians and spectral decomposition.

Builds L0, L1_down, L1_up, L1_full, L2 from boundary operators B1, B2.
Eigendecomposition via LAPACK dsyev_ through _linalg cimport.
All matrix products via BLAS dgemm.

Trace normalization and coupling constant computation for the
relational Laplacian pipeline.

This module is the spectral foundation: graph.py's spectral_bundle
calls build_all_laplacians() once and caches the result dict.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt, log, isinf, isnan
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    should_use_dense_eigen,
    can_allocate_dense_f64,
    get_EPSILON_NORM,
    get_EPSILON_DIV,
)

from rexgraph.core._linalg cimport (
    lp_eigh,
    bl_gemm_nn, bl_gemm_nt, bl_gemm_tn,
    bl_dot, bl_nrm2,
    mat_trace, mat_diag,
    spectral_pinv,
)

np.import_array()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cdef np.ndarray _to_dense_f64(obj):
    """Convert any matrix-like to dense float64 ndarray."""
    if hasattr(obj, 'toarray'):
        return np.asarray(obj.toarray(), dtype=np.float64)
    if hasattr(obj, 'row_ptr'):
        # DualCSR from _sparse
        try:
            from rexgraph.core._sparse import to_dense_f64
            return to_dense_f64(obj)
        except ImportError:
            pass
    return np.ascontiguousarray(obj, dtype=np.float64)


cdef inline f64 _safe_fiedler(f64[::1] evals, int n) noexcept nogil:
    """Return second-smallest eigenvalue (Fiedler value), or 0.0."""
    cdef int i
    cdef f64 val
    for i in range(n):
        val = evals[i]
        if val > 1e-10:
            return val
    return 0.0


# ---------------------------------------------------------------------------
# Laplacian construction — all via BLAS
# ---------------------------------------------------------------------------

def build_L0(B1_in):
    """L_0 = B_1 B_1^T (vertex Laplacian).

    Parameters
    ----------
    B1_in : array-like, shape (nV, nE)

    Returns
    -------
    L0 : ndarray, shape (nV, nV)
    """
    cdef np.ndarray[f64, ndim=2] B1 = _to_dense_f64(B1_in)
    cdef int nV = B1.shape[0]
    cdef int nE = B1.shape[1]
    cdef np.ndarray[f64, ndim=2] L0 = np.empty((nV, nV), dtype=np.float64)
    bl_gemm_nt(&B1[0, 0], &B1[0, 0], &L0[0, 0], nV, nV, nE)
    return L0


def build_L1_down(B1_in):
    """L_1^{down} = B_1^T B_1 (edge downward Laplacian).

    Parameters
    ----------
    B1_in : array-like, shape (nV, nE)

    Returns
    -------
    L1_down : ndarray, shape (nE, nE)
    """
    cdef np.ndarray[f64, ndim=2] B1 = _to_dense_f64(B1_in)
    cdef int nV = B1.shape[0]
    cdef int nE = B1.shape[1]
    cdef np.ndarray[f64, ndim=2] L1d = np.empty((nE, nE), dtype=np.float64)
    bl_gemm_tn(&B1[0, 0], &B1[0, 0], &L1d[0, 0], nE, nE, nV)
    return L1d


def build_L1_up(B2_in):
    """L_1^{up} = B_2 B_2^T (edge upward Laplacian).

    Parameters
    ----------
    B2_in : array-like, shape (nE, nF)

    Returns
    -------
    L1_up : ndarray, shape (nE, nE)
    """
    cdef np.ndarray[f64, ndim=2] B2 = _to_dense_f64(B2_in)
    cdef int nE = B2.shape[0]
    cdef int nF = B2.shape[1]
    cdef np.ndarray[f64, ndim=2] L1u = np.empty((nE, nE), dtype=np.float64)
    if nF == 0:
        memset(&L1u[0, 0], 0, nE * nE * sizeof(f64))
        return L1u
    bl_gemm_nt(&B2[0, 0], &B2[0, 0], &L1u[0, 0], nE, nE, nF)
    return L1u


def build_L1_full(L1_down_in, L1_up_in):
    """L_1 = L_1^{down} + L_1^{up} (full edge Hodge Laplacian).

    Parameters
    ----------
    L1_down_in : ndarray, shape (nE, nE)
    L1_up_in : ndarray, shape (nE, nE)

    Returns
    -------
    L1 : ndarray, shape (nE, nE)
    """
    cdef np.ndarray[f64, ndim=2] Ld = np.ascontiguousarray(L1_down_in, dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] Lu = np.ascontiguousarray(L1_up_in, dtype=np.float64)
    cdef int nE = Ld.shape[0]
    cdef np.ndarray[f64, ndim=2] L1 = np.empty((nE, nE), dtype=np.float64)
    cdef int i
    cdef int nn = nE * nE
    cdef f64* dp = &Ld[0, 0]
    cdef f64* up = &Lu[0, 0]
    cdef f64* op = &L1[0, 0]

    for i in range(nn):
        op[i] = dp[i] + up[i]

    return L1


def build_L2(B2_in):
    """L_2 = B_2^T B_2 (face Laplacian).

    Parameters
    ----------
    B2_in : array-like, shape (nE, nF)

    Returns
    -------
    L2 : ndarray, shape (nF, nF)
    """
    cdef np.ndarray[f64, ndim=2] B2 = _to_dense_f64(B2_in)
    cdef int nE = B2.shape[0]
    cdef int nF = B2.shape[1]
    if nF == 0:
        return np.zeros((0, 0), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] L2 = np.empty((nF, nF), dtype=np.float64)
    bl_gemm_tn(&B2[0, 0], &B2[0, 0], &L2[0, 0], nF, nF, nE)
    return L2


# ---------------------------------------------------------------------------
# Eigensolve via LAPACK dsyev_
# ---------------------------------------------------------------------------

def eigen_symmetric(np.ndarray[f64, ndim=2] L_in):
    """Eigendecompose a symmetric matrix via LAPACK dsyev_.

    Parameters
    ----------
    L_in : ndarray, shape (n, n), symmetric

    Returns
    -------
    evals : ndarray, shape (n,), ascending
    evecs : ndarray, shape (n, n), row-major, columns are eigenvectors
    """
    cdef int n = L_in.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    # dsyev_ needs Fortran-order input; overwrites with eigenvectors
    cdef np.ndarray[f64, ndim=2] A_F = np.asfortranarray(L_in.copy())
    cdef np.ndarray[f64, ndim=1] evals = np.empty(n, dtype=np.float64)

    lp_eigh(&A_F[0, 0], &evals[0], n)

    # Clean near-zero eigenvalues
    cdef int i
    for i in range(n):
        if fabs(evals[i]) < 1e-12:
            evals[i] = 0.0
        elif evals[i] < 0.0 and fabs(evals[i]) < 1e-9:
            evals[i] = 0.0

    # Convert to row-major (C-contiguous)
    cdef np.ndarray[f64, ndim=2] evecs = np.ascontiguousarray(A_F)
    return evals, evecs


def clean_eigenvalues(np.ndarray[f64, ndim=1] evals, f64 tol=1e-10):
    """Zero out eigenvalues below tolerance."""
    cdef int n = evals.shape[0]
    cdef np.ndarray[f64, ndim=1] out = evals.copy()
    cdef f64[::1] ov = out
    cdef int i
    for i in range(n):
        if fabs(ov[i]) < tol:
            ov[i] = 0.0
    return out


def fiedler_value(np.ndarray[f64, ndim=1] evals):
    """Second-smallest eigenvalue (algebraic connectivity)."""
    cdef int n = evals.shape[0]
    cdef f64[::1] ev = evals
    return float(_safe_fiedler(ev, n))


def fiedler_vector(np.ndarray[f64, ndim=2] evecs,
                    np.ndarray[f64, ndim=1] evals):
    """Eigenvector corresponding to the Fiedler value."""
    cdef int n = evals.shape[0]
    cdef int i
    for i in range(n):
        if evals[i] > 1e-10:
            return evecs[:, i].copy()
    return np.zeros(n, dtype=np.float64)


# ---------------------------------------------------------------------------
# Diagonal extraction
# ---------------------------------------------------------------------------

def extract_diag_L1(B1_in, B2_in):
    """Extract diagonals of L1_down and L1_up without forming full matrices.

    diag(B1^T B1)[e] = sum_v B1[v,e]^2  (column sum of squares)
    diag(B2 B2^T)[e] = sum_f B2[e,f]^2  (row sum of squares)

    Returns
    -------
    (diag_down, diag_up) : tuple of ndarray
    """
    cdef np.ndarray[f64, ndim=2] B1 = _to_dense_f64(B1_in)
    cdef np.ndarray[f64, ndim=2] B2 = _to_dense_f64(B2_in)
    cdef int nV = B1.shape[0], nE = B1.shape[1], nF = B2.shape[1]
    cdef np.ndarray[f64, ndim=1] dd = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] du = np.zeros(nE, dtype=np.float64)
    cdef f64[::1] ddv = dd, duv = du
    cdef f64[:, ::1] b1v = B1, b2v = B2
    cdef int v, e, f

    for e in range(nE):
        for v in range(nV):
            ddv[e] += b1v[v, e] * b1v[v, e]

    for e in range(nE):
        for f in range(nF):
            duv[e] += b2v[e, f] * b2v[e, f]

    return dd, du


# ---------------------------------------------------------------------------
# Composite operators
# ---------------------------------------------------------------------------

def build_L1_alpha(L1_in, L_O_in, f64 alpha):
    """L_1(alpha) = L_1 + alpha * L_O.

    Parameters
    ----------
    L1_in : ndarray (nE, nE)
    L_O_in : ndarray (nE, nE)
    alpha : float

    Returns
    -------
    ndarray (nE, nE)
    """
    cdef np.ndarray[f64, ndim=2] L1 = np.ascontiguousarray(L1_in, dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] LO = np.ascontiguousarray(L_O_in, dtype=np.float64)
    cdef int nE = L1.shape[0]
    cdef np.ndarray[f64, ndim=2] out = np.empty((nE, nE), dtype=np.float64)
    cdef int nn = nE * nE, i
    cdef f64* l1p = &L1[0, 0]
    cdef f64* lop = &LO[0, 0]
    cdef f64* op = &out[0, 0]

    for i in range(nn):
        op[i] = l1p[i] + alpha * lop[i]

    return out


def build_Lambda(B1_in, L_O_in):
    """Lambda = B_1 L_O B_1^T.

    Parameters
    ----------
    B1_in : array-like (nV, nE)
    L_O_in : ndarray (nE, nE)

    Returns
    -------
    ndarray (nV, nV)
    """
    cdef np.ndarray[f64, ndim=2] B1 = _to_dense_f64(B1_in)
    cdef np.ndarray[f64, ndim=2] LO = np.ascontiguousarray(L_O_in, dtype=np.float64)
    cdef int nV = B1.shape[0], nE = B1.shape[1]

    # tmp = B1 @ L_O  (nV x nE)
    cdef np.ndarray[f64, ndim=2] tmp = np.empty((nV, nE), dtype=np.float64)
    bl_gemm_nn(&B1[0, 0], &LO[0, 0], &tmp[0, 0], nV, nE, nE)

    # Lambda = tmp @ B1^T  (nV x nV)
    cdef np.ndarray[f64, ndim=2] out = np.empty((nV, nV), dtype=np.float64)
    bl_gemm_nt(&tmp[0, 0], &B1[0, 0], &out[0, 0], nV, nV, nE)
    return out


def compute_coupling_constants(np.ndarray[f64, ndim=1] evals_L1,
                                np.ndarray[f64, ndim=1] evals_L_O,
                                int beta1, int nE):
    """Coupling constants alpha_G and alpha_T.

    alpha_G = fiedler(L1) / fiedler(L_O)   (geometric coupling)
    alpha_T = beta_1 / nE                    (topological coupling)

    Returns
    -------
    (alpha_G, alpha_T)
    """
    cdef f64 fiedler_L1 = 0.0, fiedler_LO = 0.0
    cdef f64[::1] e1v = evals_L1, eov = evals_L_O
    cdef int n1 = evals_L1.shape[0], no = evals_L_O.shape[0]

    fiedler_L1 = _safe_fiedler(e1v, n1)
    fiedler_LO = _safe_fiedler(eov, no)

    cdef f64 alpha_G, alpha_T
    if fiedler_LO > 1e-15:
        alpha_G = fiedler_L1 / fiedler_LO
    else:
        alpha_G = float('nan')

    alpha_T = <f64>beta1 / <f64>nE if nE > 0 else 0.0

    return float(alpha_G), float(alpha_T)


# ---------------------------------------------------------------------------
# Trace normalization (for RL pipeline)
# ---------------------------------------------------------------------------

def trace_normalize(L_in):
    """Trace-normalize: L_hat = L / tr(L).

    Returns (L_hat, trace_value). Returns (zero, 0.0) if tr(L) < epsilon.
    Works on dense ndarray.
    """
    cdef np.ndarray[f64, ndim=2] L = np.ascontiguousarray(L_in, dtype=np.float64)
    cdef int n = L.shape[0]
    cdef f64 tr_val = mat_trace(&L[0, 0], n)

    if fabs(tr_val) < 1e-15:
        return np.zeros((n, n), dtype=np.float64), 0.0

    cdef np.ndarray[f64, ndim=2] out = L / tr_val
    return out, float(tr_val)


# ---------------------------------------------------------------------------
# build_all_laplacians — the main bundle builder
# ---------------------------------------------------------------------------

def build_all_laplacians(B1_in, B2_in, L_O_in,
                          bint auto_alpha=True,
                          int k=-1):
    """Build all Hodge Laplacians, eigendecompositions, and coupling constants.

    This is the single call that graph.py's spectral_bundle invokes.
    Everything routes through LAPACK/BLAS; zero np.linalg calls.

    Parameters
    ----------
    B1_in : array-like, shape (nV, nE)
        Vertex-edge boundary operator.
    B2_in : array-like or None, shape (nE, nF)
        Edge-face boundary operator. None if no faces.
    L_O_in : ndarray or None, shape (nE, nE)
        Overlap Laplacian. None if not computed.
    auto_alpha : bool
        If True, compute coupling constants and build RL_1.
    k : int
        Number of eigenvalues for sparse path (-1 = all, dense).

    Returns
    -------
    dict with keys:
        L0, L1_down, L1_up, L1_full, L2,
        evals_L0, evecs_L0, fiedler_val_L0, fiedler_vec_L0,
        evals_L1, evecs_L1, fiedler_val_L1,
        evals_L2,
        evals_L_O, fiedler_L_O, fiedler_vec_L_O,
        beta0, beta1, beta2,
        alpha_G, alpha_T,
        RL_1, evals_RL_1, evecs_RL_1,
        hats, trace_values, nhats, hat_names,
    """
    # Dense boundary operators
    cdef np.ndarray[f64, ndim=2] B1 = _to_dense_f64(B1_in)
    cdef int nV = B1.shape[0]
    cdef int nE = B1.shape[1]

    cdef np.ndarray[f64, ndim=2] B2
    cdef int nF = 0
    cdef bint has_faces = (B2_in is not None)

    if has_faces:
        B2 = _to_dense_f64(B2_in)
        nF = B2.shape[1]
        if nF == 0:
            has_faces = False

    result = {}

    # ---- L0 = B1 B1^T and eigendecompose ----
    cdef np.ndarray[f64, ndim=2] L0 = np.empty((nV, nV), dtype=np.float64)
    bl_gemm_nt(&B1[0, 0], &B1[0, 0], &L0[0, 0], nV, nV, nE)
    result['L0'] = L0

    cdef np.ndarray[f64, ndim=1] evals_L0
    cdef np.ndarray[f64, ndim=2] evecs_L0
    if nV > 0:
        evals_L0, evecs_L0 = eigen_symmetric(L0)
    else:
        evals_L0 = np.empty(0, dtype=np.float64)
        evecs_L0 = np.empty((0, 0), dtype=np.float64)

    result['evals_L0'] = evals_L0
    result['evecs_L0'] = evecs_L0

    # Fiedler for L0
    cdef f64 fiedler_val_L0 = 0.0
    if nV > 1:
        fiedler_val_L0 = fiedler_value(evals_L0)
    result['fiedler_val_L0'] = float(fiedler_val_L0)
    if nV > 1:
        result['fiedler_vec_L0'] = fiedler_vector(evecs_L0, evals_L0)
    else:
        result['fiedler_vec_L0'] = np.zeros(nV, dtype=np.float64)

    # ---- L1_down = B1^T B1 ----
    cdef np.ndarray[f64, ndim=2] L1d = np.empty((nE, nE), dtype=np.float64)
    bl_gemm_tn(&B1[0, 0], &B1[0, 0], &L1d[0, 0], nE, nE, nV)
    result['L1_down'] = L1d

    # ---- L1_up = B2 B2^T ----
    cdef np.ndarray[f64, ndim=2] L1u
    if has_faces:
        L1u = np.empty((nE, nE), dtype=np.float64)
        bl_gemm_nt(&B2[0, 0], &B2[0, 0], &L1u[0, 0], nE, nE, nF)
    else:
        L1u = np.zeros((nE, nE), dtype=np.float64)
    result['L1_up'] = L1u

    # ---- L1_full = L1_down + L1_up ----
    cdef np.ndarray[f64, ndim=2] L1f = np.empty((nE, nE), dtype=np.float64)
    cdef int nn = nE * nE, i
    cdef f64* l1dp = &L1d[0, 0]
    cdef f64* l1up = &L1u[0, 0]
    cdef f64* l1fp = &L1f[0, 0]
    for i in range(nn):
        l1fp[i] = l1dp[i] + l1up[i]
    result['L1_full'] = L1f

    # ---- Eigendecompose L1 ----
    cdef np.ndarray[f64, ndim=1] evals_L1
    cdef np.ndarray[f64, ndim=2] evecs_L1
    if nE > 0:
        evals_L1, evecs_L1 = eigen_symmetric(L1f)
    else:
        evals_L1 = np.empty(0, dtype=np.float64)
        evecs_L1 = np.empty((0, 0), dtype=np.float64)

    result['evals_L1'] = evals_L1
    result['evecs_L1'] = evecs_L1

    cdef f64 fiedler_val_L1 = 0.0
    if nE > 1:
        fiedler_val_L1 = fiedler_value(evals_L1)
    result['fiedler_val_L1'] = float(fiedler_val_L1)

    # ---- L2 = B2^T B2 ----
    cdef np.ndarray[f64, ndim=2] L2
    cdef np.ndarray[f64, ndim=1] evals_L2
    if has_faces and nF > 0:
        L2 = np.empty((nF, nF), dtype=np.float64)
        bl_gemm_tn(&B2[0, 0], &B2[0, 0], &L2[0, 0], nF, nF, nE)
        result['L2'] = L2
        evals_L2, _ = eigen_symmetric(L2)
        result['evals_L2'] = evals_L2
    else:
        result['L2'] = np.zeros((0, 0), dtype=np.float64)
        result['evals_L2'] = np.empty(0, dtype=np.float64)
        evals_L2 = result['evals_L2']

    # ---- Betti numbers from eigenvalue nullities ----
    cdef int beta0 = 0, beta1 = 0, beta2 = 0
    cdef int j
    for j in range(evals_L0.shape[0]):
        if fabs(evals_L0[j]) < 1e-8:
            beta0 += 1
    for j in range(evals_L1.shape[0]):
        if fabs(evals_L1[j]) < 1e-8:
            beta1 += 1
    for j in range(evals_L2.shape[0]):
        if fabs(evals_L2[j]) < 1e-8:
            beta2 += 1

    result['beta0'] = beta0
    result['beta1'] = beta1
    result['beta2'] = beta2

    # ---- Overlap Laplacian eigenanalysis ----
    cdef np.ndarray[f64, ndim=2] L_O
    cdef np.ndarray[f64, ndim=1] evals_LO
    cdef np.ndarray[f64, ndim=2] evecs_LO
    cdef bint has_LO = (L_O_in is not None)
    cdef f64 fiedler_LO = 0.0
    cdef f64 alpha_G = float('nan')
    cdef f64 alpha_T = 0.0

    if has_LO:
        L_O = np.ascontiguousarray(L_O_in, dtype=np.float64)
        if nE > 0:
            evals_LO, evecs_LO = eigen_symmetric(L_O)
        else:
            evals_LO = np.empty(0, dtype=np.float64)
            evecs_LO = np.empty((0, 0), dtype=np.float64)
        result['evals_L_O'] = evals_LO

        fiedler_LO = 0.0
        if nE > 1:
            fiedler_LO = fiedler_value(evals_LO)
        result['fiedler_L_O'] = float(fiedler_LO)
        if nE > 1:
            result['fiedler_vec_L_O'] = fiedler_vector(evecs_LO, evals_LO)
        else:
            result['fiedler_vec_L_O'] = np.zeros(nE, dtype=np.float64)
    else:
        result['evals_L_O'] = np.empty(0, dtype=np.float64)
        result['fiedler_L_O'] = 0.0
        result['fiedler_vec_L_O'] = np.zeros(nE, dtype=np.float64)

    # ---- Coupling constants ----

    if auto_alpha and has_LO and nE > 0:
        alpha_G_py, alpha_T_py = compute_coupling_constants(
            evals_L1, result['evals_L_O'], beta1, nE
        )
        alpha_G = alpha_G_py
        alpha_T = alpha_T_py

    result['alpha_G'] = float(alpha_G)
    result['alpha_T'] = float(alpha_T)

    # ---- RL_1 = L1 + alpha_G * L_O ----
    if auto_alpha and has_LO and nE > 0 and not (isnan(alpha_G) or isinf(alpha_G)):
        RL_1 = build_L1_alpha(L1f, L_O, alpha_G)
        result['RL_1'] = RL_1

        # Eigendecompose RL_1
        if nE > 0:
            evals_RL, evecs_RL = eigen_symmetric(RL_1)
            result['evals_RL_1'] = evals_RL
            result['evecs_RL_1'] = evecs_RL
        else:
            result['evals_RL_1'] = np.empty(0, dtype=np.float64)
            result['evecs_RL_1'] = np.empty((0, 0), dtype=np.float64)
    else:
        result['RL_1'] = None
        result['evals_RL_1'] = None
        result['evecs_RL_1'] = None

    # ---- Trace-normalized hats for RCF pipeline ----
    hats = []
    trace_values = []
    hat_names = []

    if nE > 0:
        # hat(L1_down)
        hat_L1d, tr_L1d = trace_normalize(L1d)
        if tr_L1d > 1e-15:
            hats.append(hat_L1d)
            trace_values.append(tr_L1d)
            hat_names.append('L1_down')

        # hat(L_O)
        if has_LO:
            hat_LO, tr_LO = trace_normalize(L_O)
            if tr_LO > 1e-15:
                hats.append(hat_LO)
                trace_values.append(tr_LO)
                hat_names.append('L_O')

    result['hats'] = hats
    result['trace_values'] = np.array(trace_values, dtype=np.float64) if trace_values else np.empty(0, dtype=np.float64)
    result['nhats'] = len(hats)
    result['hat_names'] = hat_names

    return result
