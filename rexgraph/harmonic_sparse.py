"""rexgraph.harmonic_sparse: the harmonic plane, combinatorial and low-rank.

Per the math reference Part V (scripts 09, 10): the harmonic space
`ker(L1) = ker(B1) ∩ ker(B2ᵀ)` is a **combinatorial** object - a basis of the cycle
space `ker(B1)` is the set of spanning-tree fundamental cycles (integer ±1 vectors),
projected onto `ker(B2ᵀ)` to remove the face (curl) directions. The harmonic
projector is applied **low-rank**:

    P_harm · x = H (HᵀH)⁻¹ Hᵀ x          (H is nE × dim_H, dim_H = β₁ - rank(B2))

so it never forms the dense nE×nE projector and never calls an eigensolver: the
correct, scale-free replacement for `_harmonic.harmonic_projectors` (which builds
`hb@hbᵀ`, `B1ᵀ pinv(B1B1ᵀ) B1`, and `eye(nE)` - three dense nE×nE matrices) whenever
only the harmonic component of a flow is needed.
"""
from __future__ import annotations

import numpy as np

_f64 = np.float64


def _cycle_basis_from_edges(nV, nE, src, tgt):
    """Spanning-tree fundamental cycle basis of `ker(B1)` given the edge endpoint
    arrays directly (the rex-free core of `cycle_basis`). Returns a sparse integer
    matrix C (nE × β₁): columns are the fundamental cycles (±1), each a non-tree
    edge closed by the tree path between its endpoints. Combinatorial (union-find +
    BFS tree paths), no eigensolve, `B1 @ C = 0` by construction."""
    from collections import deque

    import scipy.sparse as sp

    nV, nE = int(nV), int(nE)
    src = np.asarray(src, dtype=np.int64)
    tgt = np.asarray(tgt, dtype=np.int64)

    parent = list(range(nV))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:            # path compression
            parent[x], x = root, parent[x]
        return root

    tree_adj = [[] for _ in range(nV)]       # v -> list of (neighbor, edge)
    nontree = []
    for e in range(nE):
        i, j = int(src[e]), int(tgt[e])
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree_adj[i].append((j, e))
            tree_adj[j].append((i, e))
        else:
            nontree.append(e)

    beta1 = len(nontree)
    if beta1 == 0:
        return sp.csc_matrix((nE, 0), dtype=_f64)

    def tree_path(a, b):
        """Edges (with traversal direction) on the tree path a -> b."""
        prev = {a: None}
        q = deque([a])
        while q:
            u = q.popleft()
            if u == b:
                break
            for v, e in tree_adj[u]:
                if v not in prev:
                    prev[v] = (u, e)
                    q.append(v)
        out = []
        cur = b
        while prev[cur] is not None:
            u, e = prev[cur]
            out.append((u, cur, e))          # traversed u -> cur
            cur = u
        return out

    rows, cols, vals = [], [], []
    for c, e in enumerate(nontree):
        i, j = int(src[e]), int(tgt[e])
        rows.append(e); cols.append(c); vals.append(1.0)     # closing edge, forward
        for (u, v, te) in tree_path(j, i):                   # close the loop j -> i
            ti, tj = int(src[te]), int(tgt[te])
            rows.append(te); cols.append(c)
            vals.append(1.0 if (ti, tj) == (u, v) else -1.0)
    return sp.csc_matrix((vals, (rows, cols)), shape=(nE, beta1), dtype=_f64)


def _rational_nullspace(rex, nE):
    """ker(B1) over the rationals, via the complex's own exact cycle basis.

    `faces.cycle_basis` already dispatches this correctly: a pure pairwise complex takes
    the spanning-forest traversal, and ANY arity above two takes exact elimination on
    ker(B1), because `rank(B1) = n0 - c` is a graph identity that a branching relation
    breaks. Reaching for it here means the branching path is exact rather than a second
    answer to the same question.

    Returned as a sparse float matrix: the basis vectors have their denominators cleared
    to integers, so the conversion is lossless.
    """
    import scipy.sparse as sp

    from rexgraph.faces import cycle_basis as _exact_cycles
    cols = _exact_cycles(rex)
    if not cols:
        return sp.csc_matrix((nE, 0), dtype=_f64)
    M = np.zeros((nE, len(cols)), dtype=_f64)
    for j, c in enumerate(cols):
        for e, v in enumerate(c):
            if v != 0:
                M[e, j] = float(v)
    return sp.csc_matrix(M)


def _exact_nullspace(B1, nE):
    """ker(B1) from the boundary matrix alone, by dense SVD.

    The DENSE ORACLE, not the path. It exists for the callers that hold only a boundary
    matrix (`_void`'s harmonic content, `_quotient`'s relative cycle basis) and have no
    complex to ask, and for checking the exact path against something independent.
    Anything holding a rex goes through `_rational_nullspace`, which is exact and never
    densifies. Returns a sparse nE × (nE−rank B1) matrix."""
    import scipy.sparse as sp
    from scipy.linalg import null_space
    if nE == 0:
        return sp.csc_matrix((0, 0), dtype=_f64)
    B1d = np.asarray(B1.todense() if sp.issparse(B1) else B1, dtype=_f64)
    ns = null_space(B1d)                                # nE × k, columns span ker(B1)
    return sp.csc_matrix(np.ascontiguousarray(ns))


def _validated_cycle_basis(B1, nE, src=None, tgt=None, rex=None):
    """The combinatorial cycle basis of `ker(B1)`, validated against the true boundary.

    Fast path: the spanning-tree fundamental-cycle basis (integer, combinatorial). For
    BRANCHING hyperedges (arity != 2) the endpoint reduction can invent "cycles" that are
    NOT in ker(B1), so the basis is checked (‖B1·C‖ = 0 and correct dimension) and, when
    invalid, replaced by the exact nullspace of B1. Simple graphs always take the fast
    path unchanged.

    `src`/`tgt` may be supplied when the caller holds authoritative endpoints; otherwise
    they are derived from B1. Every entry point that needs a cycle basis goes through
    here, so none of them can silently use the unvalidated form.
    """
    import scipy.sparse as sp
    B1 = B1.tocsr() if sp.issparse(B1) else sp.csr_matrix(np.asarray(B1, dtype=_f64))
    B1 = B1.astype(_f64)
    nV = B1.shape[0]
    if src is None or tgt is None:
        src, tgt = _endpoints_from_b1(B1)
    C = _cycle_basis_from_edges(nV, nE, src, tgt)
    if nE == 0:
        return C
    try:
        from rexgraph.graded_boundary import _sparse_rank
        expected = nE - int(_sparse_rank(B1))           # exact dim ker(B1)
    except Exception:
        expected = None
    Cd = C.toarray() if sp.issparse(C) else np.asarray(C)
    valid = (expected is None or Cd.shape[1] == expected) and \
            (Cd.shape[1] == 0 or float(np.linalg.norm(B1 @ Cd)) < 1e-9)
    if valid:
        return C
    # branching: exact ker(B1). Through the complex's own rational elimination when
    # there is a complex to ask; the dense SVD oracle only when there is not.
    if rex is not None:
        return _rational_nullspace(rex, nE)
    return _exact_nullspace(B1, nE)


def cycle_basis(rex):
    """Basis of `ker(B1)` (the cycle space, dim = nE − rank B1) as a sparse matrix.

    Combinatorial where that is correct, exact nullspace where branching arity makes the
    endpoint reduction unsound. See `_validated_cycle_basis`."""
    from rexgraph.core._sparse import to_scipy_csr
    nE = int(rex.nE)
    src, tgt = rex._ensure_src_tgt()
    B1 = to_scipy_csr(rex._B1_dual).tocsr().astype(_f64)
    return _validated_cycle_basis(B1, nE, src, tgt, rex=rex)


def _endpoints_from_b1(B1):
    """Derive (src, tgt) per edge from a signed boundary B1 (nV × nE, -1 source /
    +1 target). Source = the most-negative signed row, target = the most-positive.
    For a pairwise edge this is exactly the (−1, +1) endpoints; for a witness or
    branching column (arity ≠ 2) it picks the extreme-signed endpoints, which is the
    right reduction for a spanning-tree cycle basis over the 1-skeleton."""
    import scipy.sparse as sp
    B1 = B1.tocsc() if sp.issparse(B1) else sp.csc_matrix(np.asarray(B1, dtype=_f64))
    nV, nE = B1.shape
    src = np.zeros(nE, dtype=np.int64)
    tgt = np.zeros(nE, dtype=np.int64)
    indptr, indices, data = B1.indptr, B1.indices, B1.data
    for e in range(nE):
        lo, hi = int(indptr[e]), int(indptr[e + 1])
        if hi <= lo:
            continue
        rows = indices[lo:hi]
        vals = data[lo:hi]
        src[e] = int(rows[np.argmin(vals)])
        tgt[e] = int(rows[np.argmax(vals)])
    return src, tgt


def harmonic_basis_from_boundaries(B1, B2):
    """Basis of the harmonic plane `ker(B1) ∩ ker(B2ᵀ)` from the sparse boundary
    matrices directly (the rex-free core of `harmonic_basis`, reused by `_void`'s
    harmonic-content and `_quotient`'s relative cycle basis). Same combinatorial
    cycle basis C = null(B1) projected onto `ker(B2ᵀ)` via H = C · null(B2ᵀC),
    applied low-rank downstream. Never forms a dense nE×nE projector, never
    eigendecomposes. Returns a sparse nE × dim_H matrix (dim_H = β₁ − rank(B2)).

    Routes through `_validated_cycle_basis`, so a branching complex gets the exact
    nullspace instead of invented cycles outside ker(B1)."""
    import scipy.sparse as sp
    B1 = B1.tocsc() if sp.issparse(B1) else sp.csc_matrix(np.asarray(B1, dtype=_f64))
    nV, nE = B1.shape
    C = _validated_cycle_basis(B1, nE)
    if C.shape[1] == 0:
        return C
    if B2 is None:
        return C
    B2 = B2.tocsr() if sp.issparse(B2) else sp.csr_matrix(np.asarray(B2, dtype=_f64))
    if B2.shape[1] == 0 or B2.nnz == 0:
        return C
    M = (B2.T @ C)                                     # nF × β₁ (face flux of cycles)
    if M.nnz == 0:
        return C                                       # cycles already flux-free
    from scipy.linalg import null_space
    ns = null_space(np.asarray(M.todense()))           # β₁ × dim_H (reduced problem)
    if ns.shape[1] == 0:
        return sp.csc_matrix((C.shape[0], 0), dtype=_f64)
    return sp.csc_matrix(np.asarray(C @ ns))           # nE × dim_H


def _b2_csr(rex):
    import scipy.sparse as sp
    B2 = getattr(rex, "_B2_hodge_dual", None)
    if B2 is None or int(getattr(rex, "nF_hodge", 0)) == 0:
        return None
    from rexgraph.core._sparse import to_scipy_csr
    try:
        return to_scipy_csr(B2).tocsr()
    except Exception:
        return sp.csr_matrix(np.asarray(rex.B2_hodge, dtype=_f64))


def harmonic_basis(rex):
    """Basis of the harmonic plane `ker(B1) ∩ ker(B2ᵀ)` as a sparse nE × dim_H
    matrix: the cycle basis C projected onto `ker(B2ᵀ)` (H = C · null(B2ᵀC)).
    Spans exactly `ker(L1)` (the same space the dense eigendecomposition returns) but
    combinatorially and low-rank. dim_H = β₁ - rank(B2) is the oscillatory-mode count."""
    import scipy.sparse as sp
    C = cycle_basis(rex)
    if C.shape[1] == 0:
        return C
    B2 = _b2_csr(rex)
    if B2 is None:
        return C                              # no faces -> harmonic = cycle space
    M = (B2.T @ C)                            # nF × β₁ (face flux of each cycle)
    if M.nnz == 0:
        return C                              # cycles already flux-free
    from scipy.linalg import null_space
    ns = null_space(np.asarray(M.todense()))  # β₁ × dim_H  (reduced problem)
    if ns.shape[1] == 0:
        return sp.csc_matrix((C.shape[0], 0), dtype=_f64)
    # C is sparse, ns is a dense (small) null-space basis -> C @ ns is dense;
    # wrap it (nE × dim_H, dim_H small) rather than calling .tocsc() on an ndarray.
    return sp.csc_matrix(np.asarray(C @ ns))  # nE × dim_H


def harmonic_projection(H, flow):
    """Apply the harmonic projector to `flow` LOW-RANK: `P_harm·flow =
    H (HᵀH)⁻¹ Hᵀ flow`, never forming the dense nE×nE projector. HᵀH is kept SPARSE
    (cycles share few edges, so it is a sparse SPD dim_H×dim_H Gram) and solved with
    a sparse factorization, so this scales even when dim_H is large. H =
    `harmonic_basis` (sparse nE × dim_H). Returns f64[nE]."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    flow = np.asarray(flow, dtype=_f64).ravel()
    k = H.shape[1]
    if k == 0:
        return np.zeros(H.shape[0], dtype=_f64)
    Hs = H.tocsr() if sp.issparse(H) else sp.csr_matrix(np.asarray(H, dtype=_f64))
    Htf = np.asarray(Hs.T @ flow).ravel()             # dim_H
    HtH = (Hs.T @ Hs).tocsc()                         # SPARSE SPD dim_H × dim_H
    try:
        coords = sla.spsolve(HtH, Htf)                # sparse LU/Cholesky solve
    except Exception:
        coords = sla.cg(HtH, Htf, rtol=1e-10, maxiter=2000)[0]
    return np.asarray(Hs @ np.asarray(coords).ravel()).ravel()   # nE
