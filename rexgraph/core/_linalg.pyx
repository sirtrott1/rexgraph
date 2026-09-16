# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._linalg: LAPACK/BLAS runtime and workspace.

Allocates the static workspace buffer used by all LAPACK calls.
Provides Python callable wrappers for testing.
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


# Python callable eigensolve

def eigh(np.ndarray[f64, ndim=2] A_in, *, bint clip_negative_roundoff=True):
    """Symmetric eigendecomposition via LAPACK dsyev_.

    Parameters

    A_in : f64[n, n], symmetric.
    clip_negative_roundoff : bool
        Legacy PSD-oriented cleanup of negative values smaller than 1e-10.
        Set False for general symmetric/indefinite spectral reference work.

    Returns

    (evals f64[n], evecs f64[n, n]) sorted ascending.
    Eigenvectors in columns of evecs (row major: evecs[:, k] is eigenvector k).
    """
    cdef int n = A_in.shape[0]
    if A_in.shape[1] != n or not np.all(np.isfinite(A_in)):
        raise ValueError("eigh requires a finite square matrix")
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)

    # dsyev_ needs column major (Fortran order)
    cdef np.ndarray[f64, ndim=2] A_F = np.asfortranarray(A_in.copy())
    cdef np.ndarray[f64, ndim=1] evals = np.empty(n, dtype=np.float64)

    lp_eigh(&A_F[0, 0], &evals[0], n)

    # Clean eigenvalues
    cdef int i
    for i in range(n):
        if clip_negative_roundoff and evals[i] < 0 and fabs(evals[i]) < 1e-10:
            evals[i] = 0.0

    # Convert to row major (eigenvectors in columns)
    cdef np.ndarray[f64, ndim=2] evecs = np.ascontiguousarray(A_F)
    return evals, evecs


# Python callable SVD

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


# Python callable least squares

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


# Python callable matrix rank

def matrix_rank(np.ndarray[f64, ndim=2] A_in, double tol=1e-10):
    """Matrix rank via SVD."""
    cdef int m = A_in.shape[0]
    cdef int n = A_in.shape[1]
    cdef np.ndarray[f64, ndim=2] A_F = np.asfortranarray(A_in.copy())
    return compute_rank_svd(&A_F[0, 0], m, n, tol)


# Python callable matrix multiply

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


# Python callable spectral pseudoinverse

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

    B1 : f64[nV, nE]
    L1, L_O, L_SG : f64[nE, nE]

    Returns

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

    # Chi star: mean of chi over incident edges (via |B1|)
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


# Nonspectral solve contracts.
#
# Everything above this line needs the whole matrix in memory: dsyev_ on an nE x nE
# operator, dgelsd_ on a dense system. The field calculus never requires that. A
# boundary action is a pass over the incidence, a metric adjoint is one solve, and
# the sector projectors follow from image and kernel frames with small Gram solves.
# Nothing below forms an operator, an inverse, or an eigendecomposition; each takes
# the operator as an action and the frame as the narrow thing it actually is.


def _as_action(operator):
    """Anything that applies to a vector, as a callable. A NativeSparse, a CSR, a
    dense array and a plain function all arrive here as the same one argument map."""
    if callable(operator):
        return operator
    apply_ = getattr(operator, "apply", None)
    if apply_ is not None and callable(apply_):
        return lambda v: np.asarray(apply_(v), dtype=np.float64).ravel()
    if hasattr(operator, "dot"):
        return lambda v: np.asarray(operator.dot(v), dtype=np.float64).ravel()
    dense = np.asarray(operator, dtype=np.float64)
    return lambda v: dense @ v


def _metric_action(metric, Py_ssize_t n):
    """A grade metric as an action, with whether it is the identity and whether it is
    diagonal. An absent metric is the identity because that is what was declared, never
    because one was guessed."""
    if metric is None:
        return (lambda v: v), True, True
    values = np.asarray(metric, dtype=np.float64) if not callable(metric) else None
    if values is not None and values.ndim == 1:
        if values.shape[0] != n:
            raise ValueError("diagonal grade metric does not match the field length")
        return (lambda v: values * v), False, True
    return _as_action(metric), False, False


def _cho_factor(np.ndarray[f64, ndim=2] gram):
    """Cholesky of a small SPD Gram via dpotrf_, Fortran ordered for dpotrs_.

    The factorization is also the independence test: for `F* M F` with positive `M`,
    `dpotrf_` fails exactly when `F` has dependent columns.
    """
    cdef int n = gram.shape[0]
    cdef int info = 0
    cdef char uplo = b'L'
    cdef np.ndarray[f64, ndim=2] A_F = np.asfortranarray(gram.copy())
    dpotrf_(&uplo, &n, &A_F[0, 0], &n, &info)
    if info != 0:
        raise ValueError(
            "the frame Gram is not positive definite: the frame is redundant, so take "
            "an image frame rather than loosen a tolerance")
    return A_F


def _cho_solve(np.ndarray[f64, ndim=2] factor, rhs):
    """One `dpotrs_` against a prepared factor: two triangular solves, no refactoring.

    A vector, or a block of columns in one call.
    """
    cdef int n = factor.shape[0]
    cdef int info = 0
    cdef char uplo = b'L'
    flat = np.asarray(rhs, dtype=np.float64)
    cdef bint vector = flat.ndim == 1
    # `dpotrs_` solves IN PLACE, so this must own its buffer. `asfortranarray` alone
    # does not guarantee that: a single row or single column array is already both C-
    # and F-contiguous and is returned unchanged, so the solve would overwrite the caller's
    # array. The copy is unconditional.
    cdef np.ndarray[f64, ndim=2] B = np.array(
        flat.reshape(n, 1) if vector else flat,
        dtype=np.float64, order="F", copy=True)
    cdef int nrhs = B.shape[1]
    if nrhs == 0:
        return B.ravel() if vector else B
    dpotrs_(&uplo, &n, &nrhs, &factor[0, 0], &n, &B[0, 0], &n, &info)
    if info != 0:
        raise ArithmeticError("the prepared Cholesky solve failed")
    # A C-contiguous copy on the way out. `dpotrs_` needs column major, and handing the
    # Fortran ordered buffer straight back leaves callers reading it through strides that
    # numpy is free to reinterpret on a transpose.
    return B[:, 0].copy() if vector else np.ascontiguousarray(B)


def frame_projector(frame, metric=None):
    """`Pi_F = F (F* M F)^-1 F* M` as a prepared ACTION, not a matrix.

    The Gram is factored once, so each application is one pass through `F` in either
    direction plus a `dpotrs_`. Prepare it when the projector sits inside an iteration.

    `F` must be an INDEPENDENT frame; a redundant one makes the Gram singular, and the
    answer is an image frame, not a tolerance. No columns gives the zero projector.

    A callable passes through unchanged, so a subspace with structure cheaper than a
    dense frame supplies its own action and its own `.diagonal()`.

    The returned action takes a vector or a block of columns and carries `.diagonal()`,
    the per coordinate self response, and `.frame`.
    """
    if callable(frame):
        return frame
    F = np.asarray(frame, dtype=np.float64)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    n = int(F.shape[0])
    r = int(F.shape[1])
    if r == 0:
        def empty(x):
            block = np.asarray(x, dtype=np.float64)
            return np.zeros_like(block, dtype=np.float64)
        empty.diagonal = lambda: np.zeros(n, dtype=np.float64)
        empty.frame = F
        return empty
    M, identity, _ = _metric_action(metric, n)
    MF = F if identity else np.column_stack([M(F[:, j]) for j in range(r)])
    factor = _cho_factor(np.ascontiguousarray(F.T @ MF))

    def project(x):
        block = np.asarray(x, dtype=np.float64)
        if block.ndim == 1:
            return F @ _cho_solve(factor, MF.T @ np.ascontiguousarray(block.ravel()))
        return F @ _cho_solve(factor, np.ascontiguousarray(MF.T @ block))

    # `diag(Pi_F)[i] = F[i,:] (F* M F)^-1 (M F)[i,:]*`: the per coordinate self response,
    # which a Green diagonal needs and which applying the projector n times would not give.
    def diagonal():
        solved = np.ascontiguousarray(
            _cho_solve(factor, np.ascontiguousarray(MF.T)).T)       # n x r
        return np.sum(F * solved, axis=1)

    project.diagonal = diagonal
    project.frame = F
    return project


def frame_project(frame, x, metric=None):
    """`Pi_F x`, for one application. Repeated use wants `frame_projector`."""
    return frame_projector(frame, metric)(x)


def _jacobi(operator, Py_ssize_t n):
    """The diagonal of a sparse operator as an inverse diagonal action, or None.

    A missing or nonpositive diagonal gives None rather than a guessed
    preconditioner.
    """
    diag = getattr(operator, "diagonal", None)
    if diag is None or not callable(diag):
        return None
    try:
        d = np.asarray(diag(), dtype=np.float64).ravel()
    except Exception:
        return None
    if d.shape[0] != n or not np.all(np.isfinite(d)) or np.any(d <= 0.0):
        return None
    inv = 1.0 / d
    return lambda v: inv * v


def metric_cg(operator, b, metric=None, x0=None, precond=None,
              double tol=1e-12, int maxiter=0):
    """Solve `A y = b` for an operator that is self adjoint and positive in `M`.

    Conjugate gradients in the `M` inner product, because that is the pairing the
    operator is symmetric in. `L_k` is self adjoint in its grade metric, not in
    Euclidean coordinates -- `M L` is the symmetric object -- so running ordinary CG
    on `L` under a nonidentity metric is solving with a nonsymmetric operator and its
    convergence means nothing. Returns (y, iterations, relative residual).
    """
    cdef Py_ssize_t n
    cdef int k, limit
    cdef double rs, rs_new, alpha, pAp, bnorm

    b = np.ascontiguousarray(b, dtype=np.float64).ravel()
    n = b.shape[0]
    A = _as_action(operator)
    M, identity, diagonal = _metric_action(metric, n)
    # Preconditioned CG needs the preconditioner self adjoint in the SAME pairing as the
    # operator. A diagonal commutes with a diagonal metric, so Jacobi is self adjoint in
    # M there; against a general metric it need not be, and an unproven preconditioner
    # would silently break the contract CG is being used for. Run unpreconditioned.
    if precond is None and diagonal:
        precond = _jacobi(operator, n)
    apply_p = (lambda v: v) if precond is None else precond

    y = np.zeros(n, dtype=np.float64) if x0 is None else \
        np.ascontiguousarray(x0, dtype=np.float64).ravel().copy()
    r = b - A(y) if x0 is not None else b.copy()
    Mb = b if identity else M(b)
    bnorm = float(b @ Mb)
    if bnorm <= 0.0:
        return np.zeros(n, dtype=np.float64), 0, 0.0
    cdef bint plain = precond is None
    z = r if plain else apply_p(r)
    p = z.copy()
    rs = float(r @ (z if identity else M(z)))
    # Without a preconditioner z is r, so the search direction product IS the residual
    # norm and a second inner product per step would buy nothing.
    cdef double rMr = rs if plain else float(r @ (r if identity else M(r)))
    limit = maxiter if maxiter > 0 else <int>(4 * n + 64)
    for k in range(limit):
        if rMr <= tol * tol * bnorm:
            return y, k, (rMr / bnorm) ** 0.5
        Ap = A(p)
        pAp = float(p @ (Ap if identity else M(Ap)))
        if pAp <= 0.0:
            raise ArithmeticError(
                "the operator is not positive in this metric; CG has no contract here")
        alpha = rs / pAp
        y += alpha * p
        r -= alpha * Ap
        z = r if plain else apply_p(r)
        rs_new = float(r @ (z if identity else M(z)))
        p = z + (rs_new / rs) * p
        rs = rs_new
        rMr = rs_new if plain else float(r @ (r if identity else M(r)))
    return y, limit, (rMr / bnorm) ** 0.5


def harmonic_pinv_matvec(operator, harmonic, x, metric=None, precond=None,
                         double tol=1e-12, int maxiter=0):
    """`L^# x`, the metric harmonic complement inverse, without an eigensolve.

    `L^# = (L + Pi_h)^-1 - Pi_h`, where `Pi_h` projects onto the harmonic frame. On
    the harmonic space `L + Pi_h` is the identity; on its complement `L` is positive
    and `Pi_h` vanishes. So the sum is positive definite, one CG solves it, and
    `L L^# = L^# L = I - Pi_h` exactly as the pseudoinverse requires.

    `L` is only ever applied, and `harmonic` is a frame of width `beta_k`.

    In Euclidean coordinates this agrees with Moore Penrose. Under a nonidentity grade
    metric it does not, and `L^#` is the one the field calculus means: the coordinate
    pseudoinverse is a different operator, not a rounding of this one.
    """
    x = np.ascontiguousarray(x, dtype=np.float64).ravel()
    trivial = False
    if not callable(harmonic):
        H = np.asarray(harmonic, dtype=np.float64)
        if H.size == 0:
            H = np.zeros((x.shape[0], 0), dtype=np.float64)
        elif H.ndim == 1:
            H = H.reshape(-1, 1)
        trivial = H.shape[1] == 0
        harmonic = H
    project = frame_projector(harmonic, metric)
    L = _as_action(operator)
    if precond is None and _metric_action(metric, x.shape[0])[2]:
        precond = _jacobi(operator, x.shape[0])
    shifted = L if trivial else (lambda v: L(v) + project(v))
    y, iters, resid = metric_cg(shifted, x, metric=metric, precond=precond,
                                tol=tol, maxiter=maxiter)
    return y - project(x)


def least_quadrance(p0, kernel, metric=None):
    """The least quadrance realization among those with the same boundary.

    `p_* = p_0 - Z(Z* M Z)^-1 Z* M p_0` for `Z` spanning `ker B`: every feasible field
    is `p_0 + Z a`, and this is the unique `a` minimising `p* M p`. Exact for rational
    inputs, and one small Gram solve rather than an iterative least squares.

    It minimises QUADRANCE, not support. The minimum can be spread across every route
    where a shortest path uses one, and it can carry a cycle its support contains; a
    combinatorial objective is a separate request, not this one rounded.
    """
    p0 = np.ascontiguousarray(p0, dtype=np.float64).ravel()
    Z = np.asarray(kernel, dtype=np.float64)
    if Z.size == 0:
        return p0.copy()
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    return p0 - frame_project(Z, p0, metric)


def spd_solve(A, b, *, strict=False):
    """`A^-1 b` for a dense symmetric positive definite `A`, by Cholesky.

    Returns None when `A` is not numerically positive definite, so a caller can fall
    back rather than be handed a wrong answer. `strict=True` raises instead.

    A positive definite system has an inverse, so `A^+ b` and `A^-1 b` are the same
    vector and a pseudoinverse would add only a rank threshold. `dpotrf_` settles
    positivity and the solve together.

    `dpotrf_` reads one triangle, so a not symmetric matrix would factor without
    complaint and return the solution to a different system. Symmetry is structural,
    so it is tested exactly and a violation raises: it is a caller error, not a
    numerical condition. Whether the matrix is positive definite is a question, and
    the answer to it is None.
    """
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("spd_solve needs a square matrix")
    if A.shape[0] == 0:
        return np.zeros_like(np.asarray(b, dtype=np.float64))
    if not np.array_equal(A, A.T):
        raise ValueError(
            "spd_solve needs a symmetric matrix; dpotrf_ reads one triangle and would "
            "otherwise solve a different system")
    try:
        factor = _cho_factor(A)
    except ValueError:
        if strict:
            raise ArithmeticError("the matrix is not positive definite") from None
        return None
    return _cho_solve(factor, np.ascontiguousarray(np.asarray(b, dtype=np.float64)))


def spd_inverse_diagonal(A):
    """`diag(A^-1)` for a dense SPD `A`, from one factorization and `n` solves.

    None when `A` is not positive definite. For a single entry use `spd_solve` against
    that basis vector and pay `O(n^2)`.
    """
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
    cdef Py_ssize_t n = A.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    try:
        factor = _cho_factor(A)
    except ValueError:
        return None
    solved = _cho_solve(factor, np.ascontiguousarray(np.eye(n, dtype=np.float64)))
    return np.ascontiguousarray(solved).diagonal().copy()
