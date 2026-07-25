"""rexgraph.harmonic_sparse - the harmonic plane, combinatorial and low-rank.

Per the math reference Part V (scripts 09, 10): the harmonic space
`ker(L1) = ker(B1) ∩ ker(B2ᵀ)` is a **combinatorial** object - a basis of the cycle
space `ker(B1)` is the set of spanning-tree fundamental cycles (integer ±1 vectors),
projected onto `ker(B2ᵀ)` to remove the face (curl) directions. The harmonic
projector is applied **low-rank**:

    P_harm · x = H (HᵀH)⁻¹ Hᵀ x          (H is nE × dim_H, dim_H = β₁ - rank(B2))

so it never forms the dense nE×nE projector and never calls an eigensolver - the
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
    import scipy.sparse as sp
    from collections import deque

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


def _exact_nullspace(B1, nE):
    """Exact basis of ker(B1) (the cycle space) for an arbitrary boundary, including
    BRANCHING hyperedges where the spanning-tree fundamental-cycle basis is invalid.
    Uses the SVD null space of B1 - not combinatorial, but this is the correctness
    fallback taken ONLY when the fast combinatorial basis fails its validation (branching
    inputs); simple graphs never reach it. Returns a sparse nE × (nE−rank B1) matrix."""
    import scipy.sparse as sp
    from scipy.linalg import null_space
    if nE == 0:
        return sp.csc_matrix((0, 0), dtype=_f64)
    B1d = np.asarray(B1.todense() if sp.issparse(B1) else B1, dtype=_f64)
    ns = null_space(B1d)                                # nE × k, columns span ker(B1)
    return sp.csc_matrix(np.ascontiguousarray(ns))


def cycle_basis(rex):
    """Basis of `ker(B1)` (the cycle space, dim = nE − rank B1) as a sparse matrix.

    Fast path: the spanning-tree fundamental-cycle basis (integer, combinatorial). For
    BRANCHING hyperedges (arity != 2) the endpoint reduction can invent "cycles" that
    are NOT in ker(B1), so the combinatorial basis is VALIDATED against the true boundary
    (‖B1·C‖ = 0 and correct dimension) and, when invalid, replaced by the exact nullspace
    of B1. Simple graphs always take the fast path unchanged."""
    import scipy.sparse as sp
    from rexgraph.core._sparse import to_scipy_csr
    nV, nE = int(rex.nV), int(rex.nE)
    src, tgt = rex._ensure_src_tgt()
    C = _cycle_basis_from_edges(nV, nE, src, tgt)
    if nE == 0:
        return C
    B1 = to_scipy_csr(rex._B1_dual).tocsr().astype(_f64)
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
    return _exact_nullspace(B1, nE)                     # branching: exact ker(B1)


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
    eigendecomposes. Returns a sparse nE × dim_H matrix (dim_H = β₁ − rank(B2))."""
    import scipy.sparse as sp
    B1 = B1.tocsc() if sp.issparse(B1) else sp.csc_matrix(np.asarray(B1, dtype=_f64))
    nV, nE = B1.shape
    src, tgt = _endpoints_from_b1(B1)
    C = _cycle_basis_from_edges(nV, nE, src, tgt)
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
    Spans exactly `ker(L1)` - same space the dense eigendecomposition returns - but
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
    a sparse factorization - so this scales even when dim_H is large. H =
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
