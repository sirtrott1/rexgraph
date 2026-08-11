"""The L0 Fiedler pair, with the kernel deflated rather than thresholded.

L0 = B1 B1^T is PSD and its kernel is not something to discover with a tolerance:
zero column-sum of B1 propagates to zero row-sum of L0, so the indicator of each
connected component is exactly a kernel vector and there are exactly that many. The
Fiedler value is the smallest eigenvalue in the orthogonal complement of those, which
is what LOBPCG computes when they are handed to it as constraints.

This is scipy orchestration, so it lives in Python: the same code in the Cython
extension is compiled with boundscheck/wraparound off and segfaulted on type
inference it had no reason to be doing.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lobpcg, splu

__all__ = ["fiedler_L0", "kernel_basis"]

_TOL = 1e-10
_DENSE_MAX = 2000          # below this an exact dense eigh is simply cheaper

# For a symmetric eigenproblem the residual bounds the error outright:
#   |lambda_computed - lambda_true| <= ||r||
# so ||r||/lambda is a genuine RELATIVE accuracy statement and the only scale-free way
# to ask whether a Fiedler value is resolved. An absolute test cannot do it: 1.6e-02 is
# a converged answer for lambda = 5.54 and a meaningless one for lambda = 1.6e-06.
# Measured, the two regimes sit four orders apart on this ratio (2.8e-03 against
# 5.4e+01), so this is the discriminator and not a tuned constant.
_REL_RESID_OK = 1e-2       # the value is resolved to better than a percent


def kernel_basis(L0):
    """Orthonormal basis of ker(L0): the component indicators that actually annihilate it.

    A component indicator is in ker(L0) because zero column-sum of B1 gives zero row-sum
    of L0. That premise fails for the one column in the model that does NOT sum to zero,
    the witness (arity 1, column `(+1)`), and there the indicator is not a kernel vector
    at all: a lone witness has `L0 @ u = u`. So the candidates are CHECKED rather than
    assumed, which costs one sparse apply and keeps the guarantee the caller relies on.

    A component that fails the check contributes nothing: L0 is positive definite there,
    so it needs no deflation and `L0 + P_H` stays positive definite either way.

    Exact and combinatorial, not a spectral estimate. Returns (U, ncols)."""
    nV = L0.shape[0]
    ncomp, labels = sp.csgraph.connected_components(L0, directed=False)
    sizes = np.bincount(labels, minlength=ncomp).astype(np.float64)
    U = np.zeros((nV, ncomp), dtype=np.float64)
    U[np.arange(nV), labels] = 1.0 / np.sqrt(sizes[labels])
    if ncomp:
        resid = np.asarray(np.abs(L0 @ U).max(axis=0)).ravel()
        keep = resid <= 1e-12
        if not keep.all():
            U = np.ascontiguousarray(U[:, keep])
    return U, int(U.shape[1])


def _dense(L0, nV, k):
    evals, evecs = np.linalg.eigh(np.asarray(L0.todense(), dtype=np.float64))
    evals[np.abs(evals) < 1e-10] = 0.0
    evals[evals < 0] = 0.0
    fval, fvec = 0.0, np.zeros(nV, dtype=np.float64)
    for i in range(len(evals)):
        if evals[i] > 1e-10:
            fval, fvec = float(evals[i]), evecs[:, i].copy()
            break
    return fval, fvec, evals, evecs


def fiedler_L0(L0, k: int = 6):
    """(fiedler_val, fiedler_vec, evals, evecs) for a sparse L0.

    The preconditioner is chosen by the solver's own residual, not by a size rule,
    because the two regimes are complementary:

      well-connected graphs   converge under Jacobi, and are exactly the ones a
                              complete factorization chokes on: on a 10000-vertex
                              random graph splu turns 209796 nonzeros into 82 million
                              and takes 54 s, while Jacobi lands on the same value to
                              5e-15 in half a second.
      path-like graphs        do not converge under Jacobi (a 2500-path reports
                              8.97e-06 for 1.58e-06), and are exactly the ones that
                              factorize in milliseconds.

    So Jacobi runs first and the RELATIVE residual decides whether to redo the solve
    against the exact factorization. See _REL_RESID_OK.
    """
    L0 = sp.csr_matrix(L0)
    nV = L0.shape[0]
    if nV <= 1:
        return (0.0, np.zeros(nV, dtype=np.float64),
                np.zeros(nV, dtype=np.float64), np.eye(nV, dtype=np.float64))
    if nV <= _DENSE_MAX:
        return _dense(L0, nV, k)

    U, ncomp = kernel_basis(L0)
    kk = max(1, min(int(k), nV - ncomp))
    X0 = np.random.default_rng(0).standard_normal((nV, kk))   # deterministic start

    d0 = np.asarray(L0.diagonal(), dtype=np.float64)
    Mjac = sp.diags(1.0 / np.where(np.abs(d0) > 1e-300, d0, 1.0))
    with warnings.catch_warnings():
        # Not reaching tol here is the probe's ANSWER, not a problem to report: the
        # relative-residual test below reads it and re-solves against a factorization.
        warnings.simplefilter("ignore")
        lam, vecs, hist = lobpcg(L0, X0.copy(), Y=U, M=Mjac, largest=False,
                                 tol=_TOL, maxiter=300, retResidualNormsHistory=True)
    lam0 = float(np.min(np.atleast_1d(np.asarray(lam, dtype=np.float64))))
    resid = float(np.max(np.atleast_1d(hist[-1])))
    if not (lam0 > 0.0 and resid <= _REL_RESID_OK * lam0):
        lu = splu((L0 + 1e-8 * sp.eye(nV, format="csc")).tocsc())
        Mexact = LinearOperator((nV, nV), matvec=lu.solve, dtype=np.float64)
        lam, vecs = lobpcg(L0, X0.copy(), Y=U, M=Mexact,
                           largest=False, tol=1e-12, maxiter=200)

    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    vecs = np.atleast_2d(np.asarray(vecs, dtype=np.float64))
    if vecs.shape[0] != nV:
        vecs = vecs.T
    lam[np.abs(lam) < 1e-14] = 0.0
    lam[lam < 0] = 0.0
    order = np.argsort(lam)
    lam, vecs = lam[order], vecs[:, order]

    # The kernel was constrained out, so lam[0] IS the smallest nonzero eigenvalue.
    # It is not recoverable by scanning `evals`: a complex with more components than
    # k has an eigenvalue prefix that is all zeros.
    fval = float(lam[0]) if lam.size else 0.0
    fvec = vecs[:, 0].copy() if vecs.shape[1] else np.zeros(nV, dtype=np.float64)
    evals = np.concatenate([np.zeros(ncomp, dtype=np.float64), lam])[:max(k, 1)]
    evecs = np.hstack([U, vecs])[:, :max(k, 1)]
    return fval, fvec, evals, evecs
