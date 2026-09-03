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

__all__ = ["fiedler_L0", "kernel_basis", "kernel_from_boundary", "deflated_operator",
           "minimum_norm_gram_solve", "solve_block_width", "leverage_diagonal",
           "leverage_sketch"]

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

    Exact and combinatorial, not a spectral estimate. Returns (U, ncols).

    U IS SPARSE, and has to be: it is a component-indicator matrix, so every row carries
    exactly ONE nonzero and the nnz is nV whatever the component count. As dense it is
    nV x ncomp, which on a real lexical complex of 721,649 vertices and 23,313 components
    asks for 0.13 TB to hold 721,649 numbers. Dense is never the better representation
    here, so it is not offered; callers do `U @ (U.T @ P)` and `U.multiply(U).sum(1)`,
    both of which scipy supports directly.
    """
    nV = L0.shape[0]
    ncomp, labels = sp.csgraph.connected_components(L0, directed=False)
    sizes = np.bincount(labels, minlength=ncomp).astype(np.float64)
    U = sp.csr_matrix(
        (1.0 / np.sqrt(sizes[labels]), (np.arange(nV), labels)),
        shape=(nV, max(ncomp, 0)), dtype=np.float64)
    if ncomp:
        # a witness column (arity 1) does not sum to zero, so its component's indicator
        # is not a kernel vector; check rather than assume, one sparse apply
        resid = np.asarray(np.abs(L0 @ U).max(axis=0).todense()).ravel() \
            if sp.issparse(L0 @ U) else np.abs(np.asarray(L0 @ U)).max(axis=0)
        keep = np.flatnonzero(resid <= 1e-12)
        if keep.size != ncomp:
            U = U[:, keep].tocsr()
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
    # lobpcg's constraint block Y must be dense: scipy factorizes Y^T B Y with cho_factor,
    # which rejects a sparse argument. This is the ONE place the indicator matrix has to
    # be materialised, and it is bounded by the eigensolver's own path rather than being
    # a representation choice; every other consumer keeps it sparse.
    U_dense = np.asarray(U.todense(), dtype=np.float64) if sp.issparse(U) else U
    kk = max(1, min(int(k), nV - ncomp))
    X0 = np.random.default_rng(0).standard_normal((nV, kk))   # deterministic start

    d0 = np.asarray(L0.diagonal(), dtype=np.float64)
    Mjac = sp.diags(1.0 / np.where(np.abs(d0) > 1e-300, d0, 1.0))
    with warnings.catch_warnings():
        # Not reaching tol here is the probe's ANSWER, not a problem to report: the
        # relative-residual test below reads it and re-solves against a factorization.
        # Narrowed to the two non-convergence notices lobpcg raises for that, so anything
        # else it has to say still reaches the caller.
        warnings.filterwarnings("ignore", message=r"Exited at iteration.*",
                                category=UserWarning)
        warnings.filterwarnings("ignore", message=r"Exited postprocessing.*",
                                category=UserWarning)
        lam, vecs, hist = lobpcg(L0, X0.copy(), Y=U_dense, M=Mjac, largest=False,
                                 tol=_TOL, maxiter=300, retResidualNormsHistory=True)
    lam0 = float(np.min(np.atleast_1d(np.asarray(lam, dtype=np.float64))))
    resid = float(np.max(np.atleast_1d(hist[-1])))
    if not (lam0 > 0.0 and resid <= _REL_RESID_OK * lam0):
        lu = splu((L0 + 1e-8 * sp.eye(nV, format="csc")).tocsc())
        Mexact = LinearOperator((nV, nV), matvec=lu.solve, dtype=np.float64)
        lam, vecs = lobpcg(L0, X0.copy(), Y=U_dense, M=Mexact,
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
    evecs = np.hstack([U_dense, vecs])[:, :max(k, 1)]
    return fval, fvec, evals, evecs


def kernel_from_boundary(B1):
    """`ker(L_0)` from the BOUNDARY alone, without ever forming `L_0 = B_1 B_1^T`.

    Same object `kernel_basis` returns, reached without the product. Components come from
    the bipartite incidence, which has the same vertex partition as `L_0`'s graph, and the
    witness check is one matrix-free apply. This component construction is exact only for
    zero-sum pairwise C1 columns (plus arity-one witnesses, whose components are checked).
    A branching relation can leave multiple independent directions inside one support
    component, so it must be handled by the general minimum-norm Green action or an
    explicitly supplied exact ``ker(B1.T)`` basis; this function refuses rather than
    manufacture an incomplete deflation basis.
    """
    B = sp.csr_matrix(B1)
    nV = B.shape[0]
    if nV == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64), 0
    C = B.tocsc(copy=True)
    C.sum_duplicates()
    C.eliminate_zeros()
    counts = np.diff(C.indptr)
    for relation in np.flatnonzero(counts == 2):
        lo = int(C.indptr[relation])
        if C.data[lo] + C.data[lo + 1] != 0.0:
            raise ValueError(
                "component kernel deflation requires zero-sum pairwise C1 columns; "
                "supply an exact ker(B1.T) basis or use the general Green operator"
            )
    if np.any(counts > 2):
        raise ValueError(
            "component kernel deflation is not defined for branching C1 relations; "
            "supply an exact ker(B1.T) basis or use the general Green operator"
        )
    big = sp.bmat([[None, B], [B.T, None]], format="csr")
    _n, labels = sp.csgraph.connected_components(big, directed=False)
    labels = labels[:nV]
    uniq, labels = np.unique(labels, return_inverse=True)
    ncomp = int(uniq.size)
    sizes = np.bincount(labels, minlength=ncomp).astype(np.float64)
    U = sp.csr_matrix((1.0 / np.sqrt(sizes[labels]), (np.arange(nV), labels)),
                      shape=(nV, ncomp), dtype=np.float64)
    if ncomp:
        resid = np.abs(B @ (B.T @ U))                     # L_0 U, matrix-free
        resid = np.asarray(resid.max(axis=0).todense()).ravel() if sp.issparse(resid) \
            else np.abs(np.asarray(resid)).max(axis=0)
        keep = np.flatnonzero(resid <= 1e-12)
        if keep.size != ncomp:
            U = U[:, keep].tocsr()
    return U, int(U.shape[1])


def deflated_operator(B1, *, kernel=None):
    """The regularised Laplacian `L_0 + P_H` as an OPERATOR, plus its Jacobi diagonal.

    `L_0` is singular, so §6e's Theorem 15 solves through `(L_0 + P_H)^{-1} - P_H`. Every
    caller that wants it was building `L_0 = B_1 B_1^T` first and handing the product to
    `_block_cg`, which takes a callable and never needed the matrix. That product is the
    expensive object: on a real lexical complex `B_1` is 4,725,208 nnz and 60 MB while the
    formed `L_0` is 108,744,551 nnz and 1.31 GB, and because a matvec costs what it reads,
    forming it makes every apply 6.8x SLOWER (84.2 ms against 12.5 ms) as well as 22x
    larger.

    So nothing is formed here. `L_0 x` is `B_1 (B_1^T x)`, `diag(L_0)` is the row sums of
    `B_1 * B_1`, and the kernel comes from the boundary. Without an explicit ``kernel``
    this is a pairwise/witness-only helper; branching C1 must use the general Green
    action because support components do not span its full kernel. Returns
    `(apply_A, dinv, U, ncomp)` ready for `sparse_character._block_cg`.
    """
    B = sp.csr_matrix(B1)
    Bt = B.T.tocsr()
    U, ncomp = kernel_from_boundary(B) if kernel is None else kernel
    d = np.asarray(B.multiply(B).sum(axis=1), dtype=np.float64).ravel()
    if ncomp:
        d = d + np.asarray(U.multiply(U).sum(axis=1)).ravel()
    dinv = np.where(d > 1e-30, 1.0 / d, 1.0)

    def apply_A(P):
        out = B @ (Bt @ P)
        if ncomp:
            out = out + U @ (U.T @ P)
        return out

    return apply_A, dinv, U, ncomp


def minimum_norm_gram_solve(B, values, *, tol=1e-12, maxit=500):
    """Numerical Moore--Penrose action ``(B B.T)^+ values`` for arbitrary Ck.

    This is the general Green lane for a boundary whose kernel is not represented by
    pairwise component indicators.  It never invents a partial deflation basis and
    never forms the Gram matrix: LSMR acts on ``B @ (B.T @ x)`` and returns its
    minimum-norm solution.  It is deliberately labelled numerical; the exact C1/C2
    topology and rank paths remain in the integer/rational stack.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator, lsmr

    boundary = sp.csr_matrix(B)
    block = np.asarray(values, dtype=np.float64)
    one = block.ndim == 1
    if one:
        block = block[:, None]
    elif block.ndim != 2:
        raise ValueError("minimum_norm_gram_solve expects a vector or a two-dimensional block")
    if block.shape[0] != boundary.shape[0]:
        raise ValueError(
            f"minimum_norm_gram_solve RHS has {block.shape[0]} rows for a "
            f"{boundary.shape[0]}-row boundary"
        )

    operator = LinearOperator(
        (boundary.shape[0], boundary.shape[0]),
        matvec=lambda value: boundary @ (boundary.T @ value),
        rmatvec=lambda value: boundary @ (boundary.T @ value),
        dtype=np.float64,
    )
    out = np.empty_like(block)
    for column in range(block.shape[1]):
        solution = lsmr(
            operator, block[:, column], atol=tol, btol=tol, conlim=0.0, maxiter=maxit,
        )
        if solution[1] not in (0, 1, 2):
            raise RuntimeError(
                "minimum-norm Gram solve did not converge, "
                f"istop={solution[1]}"
            )
        out[:, column] = solution[0]
    return out[:, 0] if one else out


def solve_block_width(nV, nE, *, panels=6):
    """How many right-hand sides fit under the configured dense ceiling.

    The tall dimension is `nV + nE`, not `nV`, and that distinction IS the bug this
    exists to close. Block CG holds `panels` dense nV-tall panels (X, R, Z, P, AP and the
    right-hand side), but `deflated_operator`'s apply is `B @ (B^T P)`, so it forms an
    nE-tall transient of the same width inside the operator. Blocking on nV alone still
    allocated nE x ncols out of sight of the caller: on one 331 KB book that transient was
    468,291 x 466,489, or 1.59 TiB.

    The budget is whatever `check_dense_allocation` is configured to allow, probed rather
    than read, so the width follows the caller's own `configure_memory` ceiling instead of
    a constant chosen here.
    """
    from rexgraph.core._common import CoreMemoryLimitError, check_dense_allocation

    tall = int(panels) * max(int(nV), 1) + max(int(nE), 1)
    width = 512
    while width > 1:
        try:
            check_dense_allocation("leverage_diagonal block", tall, width)
            return width
        except CoreMemoryLimitError:
            # the limit is the only thing being probed. Catching wider would turn a
            # fault inside the check into a silent halving to width 1, which reads as a
            # small memory budget and costs 512 times the solves.
            width //= 2
    return 1


def leverage_diagonal(B, *, columns=None, kernel=None, block=None,
                      tol=1e-12, maxit=500):
    """`diag(B^T (B B^T)^+ B)` over `columns`, matrix-free and blocked.

    This is one object under two names. As a projector it is the LEVERAGE of each column
    on the row space of `B`; as a solve it is the effective resistance `b^T L_0^+ b` of
    each relation. `_effective_resistance_batch` and `partition._leverage_of` were running
    the same solve with different blocking (none, and 512), which is why one of them fell
    over. They both come here now.

    `L_0` is never formed and neither is the projector. Returns `(values, rank)` with the
    rank read off Foster's identity, the sum of the leverage over ALL columns, so it is
    only the complex's rank when `columns` is every column.
    """
    import scipy.sparse as sp

    from rexgraph.sparse_character import _block_cg

    Bc = sp.csc_matrix(B)
    n = Bc.shape[1]
    cols = np.arange(n) if columns is None else np.asarray(columns, dtype=int).ravel()
    out = np.zeros(cols.size, dtype=np.float64)
    if cols.size == 0 or n == 0:
        return out, 0

    try:
        apply_A, dinv, _U, _nc = deflated_operator(Bc, kernel=kernel)
    except ValueError:
        if kernel is not None:
            raise
        # A component indicator does not span ker(B.T) at arbitrary arity.  Use
        # the full minimum-norm Green action rather than an incomplete deflation.
        width = int(block) if block else solve_block_width(Bc.shape[0], Bc.shape[1])
        for lo in range(0, cols.size, width):
            hi = min(lo + width, cols.size)
            P = np.ascontiguousarray(
                np.asarray(Bc[:, cols[lo:hi]].todense(), dtype=np.float64))
            X = minimum_norm_gram_solve(Bc, P, tol=tol, maxit=maxit)
            out[lo:hi] = np.einsum("ve,ve->e", P, X)
        from rexgraph.graded_boundary import _exact_rank_reduction
        exact_rank = _exact_rank_reduction(Bc)
        return out, int(round(float(out.sum()))) if exact_rank is None else int(exact_rank)
    width = int(block) if block else solve_block_width(Bc.shape[0], Bc.shape[1])
    for lo in range(0, cols.size, width):
        hi = min(lo + width, cols.size)
        P = np.ascontiguousarray(
            np.asarray(Bc[:, cols[lo:hi]].todense(), dtype=np.float64))
        X = _block_cg(apply_A, P, dinv, tol=tol, maxit=maxit)
        out[lo:hi] = np.einsum("ve,ve->e", P, X)
    return out, int(round(float(out.sum())))


def leverage_sketch(B, *, dim=None, epsilon=0.1, seed=0, tol=1e-10, maxit=500,
                    kernel=None):
    """The leverage field in O(log nE) solves instead of one per relation.

    The exact reading is a projector DIAGONAL, and a diagonal costs a solve per column
    because each entry asks about a different direction. But the same field is a set of
    squared LENGTHS,

        R_eff(c) = b_c^T L^+ b_c = || B^T L^+ b_c ||^2   (since L^+ L L^+ = L^+)

    and squared lengths survive a random projection. So project the nE-dimensional side
    down to `dim` rows once, solve for THOSE, and every relation's share is read off as a
    column norm. The solve count stops depending on the number of relations, which is the
    part that made the exact form unusable on a corpus: 27,192 solves become ~100.

    What is given up is exactness, and it is given up in a stated way rather than
    silently: Johnson-Lindenstrauss puts every entry within a factor `1 +/- epsilon` with
    high probability, and `dim` follows from `epsilon` rather than being tuned. Foster's
    identity still holds in expectation but no longer to the last bit, so a caller that
    needs the sum to close exactly wants `leverage_diagonal`.

    Returns `(values, rank_estimate)`. `seed` fixes the projection so a reading is
    reproducible; two different seeds are two samples of the same field, not two fields.
    """
    import scipy.sparse as sp

    from rexgraph.sparse_character import _block_cg

    Bc = sp.csc_matrix(B)
    nV, nE = Bc.shape
    if nE == 0 or nV == 0:
        return np.zeros(nE, dtype=np.float64), 0
    k = int(dim) if dim else max(1, min(nE, int(np.ceil(8.0 * np.log(max(nE, 2))
                                                        / float(epsilon) ** 2))))
    if k >= nE:                      # the projection would cost more than the truth
        return leverage_diagonal(Bc, kernel=kernel, tol=tol, maxit=maxit)

    rng = np.random.default_rng(int(seed))
    # +/-1 Rademacher, which is the cheap JL family and needs no dense normal draw
    Q = rng.integers(0, 2, size=(nE, k)).astype(np.float64) * 2.0 - 1.0
    Q /= np.sqrt(k)
    Y = np.ascontiguousarray(Bc @ Q)                      # nV x k, the only dense panel

    try:
        apply_A, dinv, _U, _nc = deflated_operator(Bc, kernel=kernel)
    except ValueError:
        if kernel is not None:
            raise
        X = minimum_norm_gram_solve(Bc, Y, tol=tol, maxit=maxit)
        Z = Bc.T @ X
        out = np.einsum("ek,ek->e", Z, Z)
        return out, int(round(float(out.sum())))
    width = solve_block_width(nV, nE)
    X = np.zeros_like(Y)
    for lo in range(0, k, width):
        hi = min(lo + width, k)
        X[:, lo:hi] = _block_cg(apply_A, Y[:, lo:hi], dinv, tol=tol, maxit=maxit)
    Z = Bc.T @ X                                          # nE x k
    out = np.einsum("ek,ek->e", Z, Z)
    return out, int(round(float(out.sum())))
