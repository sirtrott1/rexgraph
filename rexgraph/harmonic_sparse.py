"""rexgraph.harmonic_sparse: the harmonic plane, combinatorial and low-rank.

Per the math reference Part V: the harmonic space
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
    # The check stays SPARSE. `C.toarray()` here undid the whole point of building a
    # sparse basis: on the Gene Ontology joined with its annotations C is 151331 x
    # 110681, which is 134 GB dense, and the validation is the only thing that wanted
    # it. Neither half of the test needs a dense array: the dimension is a shape, and
    # B1 @ C is a sparse product whose norm scipy takes directly.
    n_cols = C.shape[1]
    if expected is not None and n_cols != expected:
        valid = False
    elif n_cols == 0:
        valid = True
    else:
        img = B1 @ C
        residual = (sp.linalg.norm(img) if sp.issparse(img)
                    else float(np.linalg.norm(img)))
        valid = float(residual) < 1e-9
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
    # A primary relation at any other arity has no endpoint representation.
    # Its kernel is the exact rational nullspace of its declared boundary, not
    # a traversal over a chosen two-participant projection.
    if not rex._is_standard_only:
        return _rational_nullspace(rex, nE)
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
    return _face_reduced_frame(C, M)                    # nE × dim_H, integer when M is


def _integer_nullspace(M):
    """ker(M) for an INTEGER matrix, exactly, as integer columns.

    `M` here is `B2^T C`: the face flux of each cycle. Both factors have entries in
    {-1, 0, +1}, so M is integer and its kernel has an integer basis. Taking it by
    dense SVD instead returns float columns, and the frame built from them stops
    being integer, which is the property every exact reading downstream rests on
    (the coordinates, the closure, the Gram determinant).

    Fraction row reduction, then denominators cleared by the column gcd. Returns a
    dense array of shape (M.shape[1], dim_ker) holding exact integers, or None when
    the caller should take the float path: the entries are not integral to begin
    with, or clearing denominators pushed a coordinate past what a float64 holds
    exactly. The second case is real, not defensive: elimination on a dense random
    integer matrix overflows by 40x80. A face-flux matrix is sparse and structured
    and does not, but the guard is what makes that a measurement rather than a hope.
    """
    from fractions import Fraction
    from math import gcd

    A = np.asarray(M.todense() if hasattr(M, "todense") else M)
    if not np.array_equal(A, np.round(A)):
        return None
    rows, cols = A.shape
    R = [[Fraction(int(round(A[i][j]))) for j in range(cols)] for i in range(rows)]
    pivots = []
    r = 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if R[i][c] != 0), None)
        if p is None:
            continue
        R[r], R[p] = R[p], R[r]
        d = R[r][c]
        R[r] = [x / d for x in R[r]]
        for i in range(rows):
            if i != r and R[i][c] != 0:
                f = R[i][c]
                R[i] = [a - f * b for a, b in zip(R[i], R[r], strict=False)]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in set(pivots)]
    if not free:
        return np.zeros((cols, 0), dtype=np.int64)
    basis = []
    for fc in free:
        v = [Fraction(0)] * cols
        v[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -R[i][fc]
        den = 1
        for x in v:
            den = den * x.denominator // gcd(den, x.denominator)
        iv = [int(x * den) for x in v]
        g = 0
        for x in iv:
            g = gcd(g, abs(x))
        if g > 1:
            iv = [x // g for x in iv]
        basis.append(iv)
    limit = 1 << 53                       # past this a float64 no longer holds n exactly
    if any(abs(x) >= limit for col in basis for x in col):
        return None
    return np.array([[float(x) for x in col] for col in basis], dtype=np.float64).T


def _face_reduced_frame(C, M):
    """`C @ ker(M)`, exactly when M is integer and by dense SVD when it is not."""
    import scipy.sparse as sp

    ns = _integer_nullspace(M)
    if ns is None:
        from scipy.linalg import null_space
        ns = null_space(np.asarray(M.todense() if hasattr(M, "todense") else M))
    if ns.shape[1] == 0:
        return sp.csc_matrix((C.shape[0], 0), dtype=_f64)
    return sp.csc_matrix(np.asarray(C @ ns, dtype=_f64))


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
    combinatorially and low-rank. dim_H = β₁ - rank(B2) is the oscillatory-mode count.

    COST, because it is a column per hole and holes are not rare. Building the basis
    is a spanning tree plus one tree path per non-tree relation, so it scales with
    dim_H times the path length, and dim_H is a fact about the data rather than a
    tuning knob: one Gutenberg book runs nE 1,991,070 with β₁ 1,769,648, where this
    had not returned after 12 minutes and the result would not have been a feature
    vector if it had.

    `rex.betti[1]` is the exact column count and costs about 5s at that size, so ask
    it first when the complex is not known to be small; there is no threshold here
    deciding for you. When β₁ is large, do not build the basis: pass the cycles you
    actually care about to `harmonic_winding(H, flow)` or
    `RexGraph.harmonic_winding(flow, cycles=...)` and pay one matvec against those
    alone (1 to 4 ms for 16 to 256 cycles at that same size).

    Note also that most of β₁ on such a complex is repetition rather than shape:
    `multiplicity_dimension` measured 37% to 85% across the Gutenberg store, and
    `simple_cycle_dimension` is the part that is not."""
    C = cycle_basis(rex)
    if C.shape[1] == 0:
        return C
    B2 = _b2_csr(rex)
    if B2 is None:
        return C                              # no faces -> harmonic = cycle space
    M = (B2.T @ C)                            # nF × β₁ (face flux of each cycle)
    if M.nnz == 0:
        return C                              # cycles already flux-free
    # ker(B2^T C) exactly: M is integer, so the frame stays integer and every exact
    # reading downstream (coordinates, closure, Gram determinant) keeps its footing
    return _face_reduced_frame(C, M)          # nE × dim_H


def _b1_csc(rex):
    """B1 as sparse CSC from the dual. `rex.B1` is a dense cached_property and
    materialises nV x nE: 3,317 GB on a Gutenberg complex, which raises rather than
    swaps, so nothing here goes near it."""
    from rexgraph.core._sparse import to_scipy_csr
    rex._ensure_clean()
    return to_scipy_csr(rex._B1_dual).tocsc()


def multiplicity_groups(rex, min_size=2):
    """Relations sharing an identical boundary column, grouped.

    Two relations can be distinct and still have the same boundary: two occurrences
    of one token are two spans, so two witnesses on one vertex, and the corpus is
    full of them. Their difference has zero boundary, so it is a CYCLE, but it is
    not a hole: it records that the relation occurred twice.

    Grouped up to overall sign, since a column and its negative also cancel. Returns
    `(indices, signs)` pairs, largest group first: `signs[i] * column(indices[i])` is
    the same vector for every member, so the sign is what says whether two members
    cancel by difference or by sum. Dropping it silently emits non-cycles.
    """
    B1 = _b1_csc(rex)
    B1.sort_indices()                        # the key is positional, so order matters
    arity = np.diff(B1.indptr)
    out = []
    for k in np.unique(arity):
        k = int(k)
        if k == 0:
            continue
        cols = np.where(arity == k)[0]
        if cols.size < min_size:
            continue
        span = B1.indptr[cols][:, None] + np.arange(k)[None, :]
        idx = np.ascontiguousarray(B1.indices[span].astype(np.int64))   # m x k
        dat = np.ascontiguousarray(B1.data[span].astype(_f64))          # m x k
        sign = np.where(dat[:, 0] < 0, -1.0, 1.0)    # canonicalise the overall sign
        dat *= sign[:, None]
        np.round(dat, 12, out=dat)
        order, bounds = _identical_runs(idx, dat)
        size = np.diff(bounds)
        for a, b in zip(bounds[:-1][size >= min_size], bounds[1:][size >= min_size], strict=False):
            members = np.sort(order[a:b])
            out.append((cols[members], sign[members]))
    out.sort(key=lambda t: -t[0].size)
    return out


def _identical_runs(idx, dat):
    """Group identical rows of (idx | dat). Returns `(order, bounds)` such that
    `order[bounds[i]:bounds[i+1]]` is one group of identical rows.

    `np.unique(axis=0)` lexsorts a 2k-wide float view and costs 21s on a 2e6-relation
    book. This hashes each row to one uint64, sorts THAT (a single O(m log m) pass
    over a 1-D array), and then cuts runs wherever the FULL rows differ, so a
    collision cannot merge two different relations. It can only split one group into
    several runs, which is detected by a hash appearing in more than one run and
    repaired exactly for those hashes alone. In practice that never fires.
    """
    m = idx.shape[0]
    h = np.zeros(m, dtype=np.uint64)
    bits = dat.view(np.uint64)
    for c in range(idx.shape[1]):
        h *= np.uint64(0x100000001B3)
        h ^= idx[:, c].view(np.uint64)
        h *= np.uint64(0x100000001B3)
        h ^= bits[:, c]
    order = np.argsort(h, kind="stable")
    same = h[order][1:] == h[order][:-1]
    if idx.shape[1]:
        same &= (idx[order][1:] == idx[order][:-1]).all(axis=1)
        same &= (dat[order][1:] == dat[order][:-1]).all(axis=1)
    bounds = np.flatnonzero(np.r_[True, ~same, True])

    run_h = h[order][bounds[:-1]]
    if run_h.size and np.unique(run_h).size != run_h.size:   # a genuine collision
        key = np.concatenate([idx.astype(_f64), dat], axis=1)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        cuts = inv[order][1:] != inv[order][:-1]
        bounds = np.flatnonzero(np.r_[True, cuts, True])
    return order, bounds


def multiplicity_dimension(rex, groups=None):
    """How much of the cycle space is multiplicity rather than topology: the exact
    integer sum of (group size - 1).

    Within a group of m identical columns the differences span {x on the group with
    sum(x) = 0}, dimension m - 1; groups have disjoint support, so they are
    independent and the total is exact.

    This is a CHAIN-level quantity: W is a subspace of Z1, not of H1. A face can
    fill part of it: put a face on a bigon and W still has dimension 1 while beta_1
    is 0, so subtracting this from `rex.betti[1]` is only valid with no faces. For
    the split of H1 itself use `simple_cycle_dimension`, which is exact either way.

    MEASURED on the Gutenberg store, where this is not a rounding effect: 39 to 85
    percent of beta_1 across five documents, the largest single group holding
    157,674 identical relations. Any shortest-cycle method returns these first,
    every one of them 2-sparse, so a cycle reading that does not separate them is
    reading occurrence counts and calling them topology.
    """
    if groups is not None:
        return int(sum(int(idx.size) - 1 for idx, _ in groups))
    # sum over groups of (size - 1) is exactly (columns - runs), so the dimension
    # never needs the groups materialised: no Python loop over the 193,422 buckets
    # a full book produces.
    B1 = _b1_csc(rex)
    B1.sort_indices()
    arity = np.diff(B1.indptr)
    total = 0
    for k in np.unique(arity):
        k = int(k)
        if k == 0:
            continue
        cols = np.where(arity == k)[0]
        if cols.size < 2:
            continue
        span = B1.indptr[cols][:, None] + np.arange(k)[None, :]
        idx = np.ascontiguousarray(B1.indices[span].astype(np.int64))
        dat = np.ascontiguousarray(B1.data[span].astype(_f64))
        dat *= np.where(dat[:, 0] < 0, -1.0, 1.0)[:, None]
        np.round(dat, 12, out=dat)
        _, bounds = _identical_runs(idx, dat)
        total += int(cols.size) - (int(bounds.size) - 1)
    return int(total)


def collapse_map(rex, groups=None):
    """`pi`: C1(X) -> C1(X'), identifying relations that share a boundary column.

    Sends `e` to `sign_e * r` for its group representative r, the sign being what
    makes `canonical(e) = sign_e * column(e)` the same vector across a group. Then
    `B1' pi = B1` and `B2' = pi B2` keeps `B1' B2' = 0`, so X' is a complex and not
    just a relabelling. Returns `(pi, keep)` with `keep` the representative columns.
    """
    import scipy.sparse as sp

    nE = int(rex.nE)
    rep = np.arange(nE, dtype=np.int64)
    sgn = np.ones(nE, dtype=_f64)
    for idx, sign in (multiplicity_groups(rex) if groups is None else groups):
        rep[idx] = int(idx[0])
        sgn[idx] = sign * sign[0]        # relative to the representative
    keep = np.flatnonzero(rep == np.arange(nE))
    newid = np.full(nE, -1, dtype=np.int64)
    newid[keep] = np.arange(keep.size)
    rows = newid[rep]
    pi = sp.csr_matrix((sgn, (rows, np.arange(nE))), shape=(keep.size, nE))
    return pi, keep


def simple_cycle_dimension(rex, groups=None):
    """beta_1 of the complex with identical-boundary relations identified.

    The exact complement of the multiplicity part IN HOMOLOGY:

        beta_1(X) = dim H1_multiplicity + simple_cycle_dimension(X)

    holds with faces or without, because it is a quotient rather than a subtraction:
    Z1(X)/(W + B1) is isomorphic to Z1(X')/B1(X') = H1(X'). Subtracting
    `multiplicity_dimension` instead is only right when nothing fills a multiplicity
    cycle, which is why that shortcut is taken only at nF = 0.
    """
    b1 = int(rex.betti[1])
    if int(rex.nF) == 0:
        # nothing can fill a multiplicity cycle, so W injects into H1 and the
        # dimension shortcut applies, with no groups to materialise.
        return b1 - multiplicity_dimension(rex, groups=groups)
    g = multiplicity_groups(rex) if groups is None else groups
    if not g:
        return b1

    from rexgraph.graded_boundary import betti_numbers
    pi, keep = collapse_map(rex, groups=g)
    B1p = _b1_csc(rex)[:, keep].tocsr()
    B2 = _b2_csr(rex)
    tower = [B1p] + ([(pi @ B2).tocsr()] if B2 is not None else [])
    return int(betti_numbers(tower)[1])


def multiplicity_homology_dimension(rex, groups=None):
    """How much of beta_1 the repeated relations carry: the exact difference
    `beta_1(X) - simple_cycle_dimension(X)`. Non-negative by construction, and it
    sums with the simple part to beta_1 whether or not the complex has faces."""
    return int(rex.betti[1]) - int(simple_cycle_dimension(rex, groups=groups))


def multiplicity_cycles(rex, groups=None, limit=None):
    """The multiplicity cycles themselves: sparse nE x d, entries in {-1, 0, +1}.

    Consecutive differences within each group, which is one basis of the subspace.
    `limit` caps the column count, since d runs to 1.5e6 on a full book.
    """
    import scipy.sparse as sp

    g = multiplicity_groups(rex) if groups is None else groups
    rows, cols, vals, j = [], [], [], 0
    for idx, sign in g:
        # canonical(e) = sign[e] * column(e) is equal across the group, so
        # sign[a]*col(a) - sign[b]*col(b) = 0. A sign-flipped pair therefore
        # cancels by SUM, not by difference, which is why the signs are carried.
        for (a, sa), (b, sb) in zip(zip(idx[:-1], sign[:-1], strict=False), zip(idx[1:], sign[1:], strict=False), strict=False):
            if limit is not None and j >= limit:
                break
            rows += [int(a), int(b)]
            cols += [j, j]
            vals += [float(sa), -float(sb)]
            j += 1
        if limit is not None and j >= limit:
            break
    if j == 0:
        return sp.csc_matrix((int(rex.nE), 0), dtype=_f64)
    return sp.csc_matrix((vals, (rows, cols)), shape=(int(rex.nE), j), dtype=_f64)


def as_edge_signal(values, nE, *, what="signal"):
    """An edge signal as f64[nE], from an array, a list, or a torch tensor.

    Two things a caller building a layer or an agent hits immediately, neither of
    which said anything useful before:

    A TORCH TENSOR is accepted and DETACHED. The harmonic readings are exact
    integer/rational counts, not differentiable functions of the signal, so there is
    no gradient to carry through a winding and detaching is the honest behaviour
    rather than a limitation to work around. Passing one with requires_grad used to
    surface torch's own "Can't call numpy() on Tensor that requires grad", which
    reads as a bug in the caller's code. Use these to build FEATURES; do not expect
    to backpropagate through them.

    A WRONG LENGTH is refused by naming what was expected, in the same shape as
    `RexGraph.signal`, instead of surfacing a scipy matmul dimension mismatch that
    mentions neither nE nor which reading was being taken.
    """
    if hasattr(values, "detach"):                 # torch, jax-like, anything tracing
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy") and not isinstance(values, np.ndarray):
        values = values.numpy()
    arr = np.asarray(values, dtype=_f64).ravel()
    if arr.shape[0] != int(nE):
        raise ValueError(
            f"Expected {int(nE)} values for the edge {what}, got {arr.shape[0]}.")
    return arr


def harmonic_winding(H, flow):
    """The winding of `flow` around each independent cycle: `Hᵀ flow`, dim_H long.

    This is the whole of what a cycle can see. Pairing a cycle z against the three
    Hodge sectors gives

        gradient   <B₁ᵀφ, z> = <φ, B₁z> = 0      a potential has no holonomy
        curl       <B₂ψ,  z> = <ψ, B₂ᵀz> = 0     what bounds is invisible to z
        harmonic   nonzero                        the holonomy

    both exactly, so the winding of the FULL cochain equals the winding of its
    harmonic part and nothing is being discarded by not projecting first.

    It is one sparse matvec, no solve and no metric. `harmonic_basis` keeps the
    frame integer, so integer data in gives integer windings out and the result is
    a COUNT rather than a measurement: a number of turns, with no angle, no arc
    length and no transcendental in it. The dtype reflects that: integer in,
    integer out.

    This is the exact, pre-metric reading. `harmonic_coordinates` is the metric one:
    it applies the inverse Gram (the harmonic metric is HᵀH, not the identity) and
    is necessarily float. Use the winding for anything counted, compared or stored;
    use the coordinates when the answer has to live in the metric.
    """
    import scipy.sparse as sp
    if H.shape[1] == 0:
        return np.zeros(0, dtype=_f64)
    f = as_edge_signal(flow, H.shape[0], what="flow")
    Hs = H.tocsr() if sp.issparse(H) else sp.csr_matrix(H)
    w = np.asarray(Hs.T @ f).ravel()
    exact = np.array_equal(f, np.round(f)) and _frame_is_integer(H)
    if exact and np.array_equal(w, np.round(w)):
        return np.atleast_1d(np.round(w).astype(np.int64))
    return np.atleast_1d(w)


def _frame_is_integer(H):
    import scipy.sparse as sp
    d = H.data if sp.issparse(H) else np.asarray(H)
    return d.size == 0 or np.array_equal(d, np.round(d))


def harmonic_coordinates(H, flow):
    """Where `flow` sits on the harmonic frame `H`: f64[dim_H].

    `(HᵀH)⁻¹ Hᵀ flow`, the small side of the harmonic projector. HᵀH is kept
    SPARSE (cycles share few edges, so it is a sparse SPD dim_H×dim_H Gram) and
    solved with a sparse factorization, so this scales even when dim_H is large.
    One coordinate per independent hole.

    The numerator `Hᵀ flow` is `harmonic_winding`, and it is the exact half: the
    Gram solve is what turns an integer count into a float coordinate.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    if H.shape[1] == 0:
        return np.zeros(0, dtype=_f64)
    Hs = H.tocsr() if sp.issparse(H) else sp.csr_matrix(np.asarray(H, dtype=_f64))
    Htf = np.asarray(harmonic_winding(H, flow), dtype=_f64)   # dim_H
    HtH = (Hs.T @ Hs).tocsc()                         # SPARSE SPD dim_H × dim_H
    try:
        coords = sla.spsolve(HtH, Htf)                # sparse LU/Cholesky solve
    except Exception:
        coords = sla.cg(HtH, Htf, rtol=1e-10, maxiter=2000)[0]
    return np.atleast_1d(np.asarray(coords, dtype=_f64).ravel())


def harmonic_projection(H, flow):
    """Apply the harmonic projector to `flow` LOW-RANK: `P_harm·flow =
    H (HᵀH)⁻¹ Hᵀ flow`, never forming the dense nE×nE projector. H =
    `harmonic_basis` (sparse nE × dim_H). Returns f64[nE].

    The coordinates it goes through are `harmonic_coordinates`, which callers
    working in the harmonic plane rather than the edge space read directly.
    """
    import scipy.sparse as sp
    if H.shape[1] == 0:
        return np.zeros(H.shape[0], dtype=_f64)
    coords = harmonic_coordinates(H, as_edge_signal(flow, H.shape[0], what="flow"))
    Hs = H.tocsr() if sp.issparse(H) else sp.csr_matrix(np.asarray(H, dtype=_f64))
    return np.asarray(Hs @ coords).ravel()            # nE
