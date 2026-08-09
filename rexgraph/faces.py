"""Face columns, solved from the chain condition.

A grade-1 column is DECLARED: Definition 2.1 fixes its shape, -1 at the distinguished
vertex and 1/(k-1) on the rest, so arity is legible from two entries. A grade-2 column is
not declared. Nothing imposes a shape on it. It is whatever satisfies

    B1 c_f = 0

on the edges it spans, and what the solution owes is cancellation, which is exactly the
chain condition. The consequences of that asymmetry are visible in the answers: a cycle
face comes back with uniform moduli (so the grade-0 arity ratio returns 2 at every k_f,
and the gon is |supp(c_f)| instead), while a branching fan comes back with the shape that
cancels the fan.

So the primitive here is a solver, not a template. It works over Fraction, which keeps
this on the integer/exact tower: the chain condition holds at 0 rather than at a
tolerance, there is no eigensolve, and there is no brute force over 2^k sign patterns.

Refusal is a result. A set of edges that bounds nothing returns None: a path encloses no
area, a lone relation encloses nothing, and a partial overlap does not cancel. Forcing a
face there would assert a topology the complex does not have.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

__all__ = ["solve_face_column", "solve_face_basis", "face_reading",
           "orientation_holonomy", "face_support", "cycle_basis", "find_cycles",
           "find_hyperface_groups", "autoface", "auto_hyperface"]


def _exact_b1_block(rex, edge_ids):
    """The B1 columns for `edge_ids` as exact rationals, keyed by vertex.

    Rebuilt from the boundary structure rather than read back from the assembled float
    B1: the coefficients are -1 and 1/(k-1), and recovering those from a float would put
    the solve on the approximation tower for no reason.
    """
    rex._ensure_clean()
    bp, bi = rex._boundary_ptr, rex._boundary_idx
    cols = []
    if bp is None:                                   # standard-only: every column is (-1,+1)
        src, tgt = rex._ensure_src_tgt()
        for e in edge_ids:
            s, t = int(src[e]), int(tgt[e])
            cols.append({} if s == t else {s: Fraction(-1), t: Fraction(1)})
        return cols
    for e in edge_ids:
        start, end = int(bp[e]), int(bp[e + 1])
        k = end - start
        col: dict[int, Fraction] = {}
        if k == 1:
            col[int(bi[start])] = Fraction(1)
        elif k >= 2:
            share = Fraction(1, k - 1)
            for j in range(start, end):
                v = int(bi[j])
                col[v] = col.get(v, Fraction(0)) + (Fraction(-1) if j == start else share)
        cols.append({v: c for v, c in col.items() if c != 0})
    return cols


def solve_face_column(rex, edge_ids):
    """Solve ``B1[:, edge_ids] c = 0`` for a face column over the rationals.

    Returns a list of ``Fraction`` of the same length as ``edge_ids``, normalised so the
    first nonzero entry is +1 and the entries clear to integers where they can. Returns
    ``None`` when the only solution is the trivial one, which is the honest answer for a
    set of edges that bounds nothing.

    The nullspace is taken by exact fraction-free elimination on the (vertices x edges)
    block, so the cost is set by the block, not by the complex, and the result is exact.
    When the nullspace has dimension > 1 the first basis vector is returned; that happens
    when the given edges already contain more than one independent cycle, and the caller
    is expected to pass one face's worth of edges.
    """
    edge_ids = np.asarray(edge_ids, dtype=np.int64).ravel()
    m = int(edge_ids.shape[0])
    if m == 0:
        return None
    cols = _exact_b1_block(rex, edge_ids)
    verts = sorted({v for col in cols for v in col})
    if not verts:
        return None
    row_of = {v: i for i, v in enumerate(verts)}
    A = [[Fraction(0)] * m for _ in verts]
    for j, col in enumerate(cols):
        for v, c in col.items():
            A[row_of[v]][j] = c

    # Gauss-Jordan over Q, tracking which columns are pivots.
    n_rows = len(verts)
    pivot_col_of_row: list[int] = []
    r = 0
    for c in range(m):
        piv = next((i for i in range(r, n_rows) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = Fraction(1) / A[r][c]
        A[r] = [x * inv for x in A[r]]
        for i in range(n_rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r], strict=True)]
        pivot_col_of_row.append(c)
        r += 1
        if r == n_rows:
            break

    pivots = set(pivot_col_of_row)
    free = [c for c in range(m) if c not in pivots]
    if not free:
        return None                                  # trivial nullspace: nothing is bounded

    f0 = free[0]
    x = [Fraction(0)] * m
    x[f0] = Fraction(1)
    for i, pc in enumerate(pivot_col_of_row):
        x[pc] = -A[i][f0]

    lead = next((v for v in x if v != 0), None)
    if lead is None:
        return None
    x = [v / lead for v in x]
    # clear denominators so a cycle face comes back as +/-1 rather than a scaled copy
    den = 1
    for v in x:
        den = den * v.denominator // np.gcd(den, v.denominator)
    return [v * den for v in x]


def solve_face_basis(rex, edge_ids) -> list:
    """Every independent face the given relations bound, as exact columns.

    `solve_face_column` returns the first basis vector, which is the whole answer when the
    nullspace is one-dimensional and an arbitrary choice when it is not. A group carrying
    two independent cycles bounds a two-dimensional space of faces, and picking one vector
    out of it attaches a cell that depends on the elimination order rather than on the
    complex: on a 4-ary relation with its induced 4-cycle the pick came back
    `[3, -3, -2, -1, 0]`, a face that claims five relations and uses four.

    Returns a list of columns, one per independent cycle, so a caller can attach all of
    them and leave nothing half-filled. Empty when the relations bound nothing.
    """
    edge_ids = np.asarray(edge_ids, dtype=np.int64).ravel()
    m = int(edge_ids.shape[0])
    if m == 0:
        return []
    cols = _exact_b1_block(rex, edge_ids)
    verts = sorted({v for col in cols for v in col})
    if not verts:
        return []
    row_of = {v: i for i, v in enumerate(verts)}
    A = [[Fraction(0)] * m for _ in verts]
    for j, col in enumerate(cols):
        for v, c in col.items():
            A[row_of[v]][j] = c

    pivot_col_of_row: list[int] = []
    r = 0
    for c in range(m):
        piv = next((i for i in range(r, len(verts)) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = Fraction(1) / A[r][c]
        A[r] = [x * inv for x in A[r]]
        for i in range(len(verts)):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r], strict=True)]
        pivot_col_of_row.append(c)
        r += 1
        if r == len(verts):
            break

    pivots = set(pivot_col_of_row)
    basis = []
    for fc in (c for c in range(m) if c not in pivots):
        x = [Fraction(0)] * m
        x[fc] = Fraction(1)
        for i, pc in enumerate(pivot_col_of_row):
            x[pc] = -A[i][fc]
        lead = next((v for v in x if v != 0), None)
        if lead is None:
            continue
        x = [v / lead for v in x]
        den = 1
        for v in x:
            den = den * v.denominator // np.gcd(den, v.denominator)
        basis.append([v * den for v in x])
    return basis


def face_reading(rex, edge_ids, column=None) -> dict:
    """Whether a set of relations bounds a face, what its signs are, and why not.

    `solve_face_column` answers with a column or with None, and None covers two different
    situations that a caller has to tell apart. This reports which::

        bounds       nullity 1: exactly one cycle, so the column is determined up to an
                     overall sign and the face is real
        open         nullity 0: the boundary columns are independent, nothing is bounded,
                     and attaching a face here would invent a cell
        degenerate   nullity > 1: the set carries several independent cycles, so it is not
                     one face and the caller passed too many relations

    The SIGN CONTEXT is the rest of it. Solving gives the coefficient each relation is
    traversed with, so `reversed_relations` is the ones running against their stored
    orientation and `holonomy` is the product of the signs around the cycle. Two complexes
    with the same support and different orientations are the same cell to homology and
    different cells here, which is the whole content of the grade-2 boundary: existence
    says there is a cycle, the sign says how it closes.

    Pass `column` to check signs the caller already has instead of solving for them. A
    wrong column is otherwise invisible: `_B2_hodge_dual` filters chain-invalid faces
    silently, so nF_hodge stays 0, the cycle stays open, and nothing says why. Here it
    comes back as `chain_valid: False` with the exact residual and the column that would
    have worked.
    """
    edge_ids = [int(e) for e in np.asarray(edge_ids, dtype=np.int64).ravel()]
    m = len(edge_ids)
    out = {"relations": edge_ids, "gon": m}
    if m == 0:
        return {**out, "state": "open", "nullity": 0, "column": None,
                "reason": "no relations were given"}

    cols = _exact_b1_block(rex, edge_ids)
    verts = sorted({v for col in cols for v in col})
    nullity = m - _block_rank(cols, verts)
    solved = solve_face_column(rex, edge_ids)
    state = "bounds" if nullity == 1 else ("open" if nullity == 0 else "degenerate")
    out.update({"state": state, "nullity": nullity})

    if column is None:
        chosen = solved
    else:
        chosen = [x if isinstance(x, Fraction) else Fraction(x).limit_denominator()
                  for x in column]
        if len(chosen) != m:
            raise ValueError("column must have one coefficient per relation")

    if chosen is None:
        out["column"] = None
        out["reason"] = ("these relations are independent, so they bound nothing"
                         if state == "open" else
                         "more than one independent cycle here: this is not one face")
        return out

    residual = _chain_residual(cols, verts, chosen)
    out["column"] = [str(x) for x in chosen]
    if state == "degenerate" and column is None:
        out["reason"] = (f"these relations carry {nullity} independent cycles, so they "
                         f"bound a space of faces and this is one basis vector of it, "
                         f"fixed by the elimination order rather than by the complex. "
                         f"Use solve_face_basis to attach all {nullity}.")
    out["chain_valid"] = all(r == 0 for r in residual)
    out["residual"] = [str(r) for r in residual if r != 0]
    signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in chosen]
    out["reversed_relations"] = [edge_ids[i] for i, sg in enumerate(signs) if sg < 0]
    out["holonomy"] = int(np.prod([sg for sg in signs if sg != 0])) if any(signs) else 0
    out["support"] = face_support(chosen)
    if column is not None and not out["chain_valid"]:
        out["solved_column"] = [str(x) for x in solved] if solved else None
        out["reason"] = ("the given signs do not satisfy B1 c = 0, so this face would be "
                         "dropped by the chain filter with no other symptom")
    return out


def _block_rank(cols, verts) -> int:
    """Rank of the (vertices x relations) boundary block, exactly."""
    m = len(cols)
    row_of = {v: i for i, v in enumerate(verts)}
    A = [[Fraction(0)] * m for _ in verts]
    for j, col in enumerate(cols):
        for v, c in col.items():
            A[row_of[v]][j] = c
    rank = 0
    for c in range(m):
        piv = next((i for i in range(rank, len(verts)) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        inv = Fraction(1) / A[rank][c]
        A[rank] = [x * inv for x in A[rank]]
        for i in range(len(verts)):
            if i != rank and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[rank], strict=True)]
        rank += 1
    return rank


def _chain_residual(cols, verts, column) -> list:
    """`B1 c` per vertex, exactly. Zero throughout is the chain condition."""
    total = dict.fromkeys(verts, Fraction(0))
    for coeff, col in zip(column, cols, strict=True):
        for v, c in col.items():
            total[v] += coeff * c
    return [total[v] for v in verts]


def orientation_holonomy(rex, *, grade: int = 2) -> dict:
    """Whether the cells at a grade can be coherently oriented, gauge-invariantly.

    A solved face column is determined only up to an overall sign: a face and its reverse
    are the same cell, and `solve_face_column` returns the leading-positive representative
    because something has to be returned. So any per-cell sign reading measures the
    REPRESENTATIVE, not the complex. Negating one column of a two-triangle complex moves a
    raw sign product from +1 to -1 without changing a single cell.

    The invariant is the holonomy, exactly as at grade 1, where frustration is the product
    of signs around a cycle rather than a count of negative edges. One grade up the loop
    runs through the cells: two faces meeting on a relation agree when they induce
    OPPOSITE coefficients on it, which is what coherent orientation means, so the pairwise
    agreement is `-sign(c_a[e] c_b[e])` and the holonomy is its product around a closed
    loop of faces. Each face appears exactly twice in a closed loop, so its sign cancels
    and the product is invariant. Measured: a tetrahedron boundary reads +1 on every loop
    under any subset of flips, a five-triangle Moebius band reads -1 under any.

    Balanced (+1 everywhere) is exactly coherent orientability, so this MEASURES what
    orienting face-by-face attempts, and does not need the attempt to succeed in order to
    report the obstruction.

    Returns the per-loop holonomies, the frustrated fraction, and `orientable`.
    """
    grade = int(grade)
    if grade < 2:
        return {"grade": grade, "loops": [], "orientable": None,
                "reason": ("orientation is a relation BETWEEN cells, and at grade 1 the "
                           "column is canonical (-1, +share, ...) with no freedom to "
                           "gauge; use frustration_exact for the grade-1 holonomy")}
    B = _grade_boundary(rex, grade)
    if B is None or B.shape[1] < 2:
        return {"grade": grade, "loops": [], "n_loops": 0, "orientable": True,
                "frustrated": 0, "reason": "fewer than two cells: no loop to close"}

    B = B.tocsc()
    n = B.shape[1]
    support = [dict(zip(B.indices[B.indptr[c]:B.indptr[c + 1]],
                        B.data[B.indptr[c]:B.indptr[c + 1]], strict=False)) for c in range(n)]

    # the dual graph: cells adjacent when they share a bounding cell one grade down
    adjacency, agreement = {c: [] for c in range(n)}, {}
    for a in range(n):
        for b in range(a + 1, n):
            shared = [e for e in support[a] if e in support[b]
                      and support[a][e] != 0 and support[b][e] != 0]
            if not shared:
                continue
            e = min(shared)
            adjacency[a].append(b)
            adjacency[b].append(a)
            agreement[(a, b)] = agreement[(b, a)] = (
                -1 if support[a][e] * support[b][e] > 0 else 1)

    # a spanning forest fixes a gauge; every non-tree edge closes one independent loop,
    # and the cycle space is spanned by those, so no eigensolver and no enumeration
    parent, orient, seen = {}, {}, set()
    order = []
    for root in range(n):
        if root in seen:
            continue
        seen.add(root)
        orient[root] = 1
        parent[root] = None
        stack = [root]
        while stack:
            a = stack.pop()
            order.append(a)
            for b in adjacency[a]:
                if b not in seen:
                    seen.add(b)
                    parent[b] = a
                    orient[b] = orient[a] * agreement[(a, b)]
                    stack.append(b)

    loops = []
    tree = {(a, parent[a]) for a in parent if parent[a] is not None}
    for a in range(n):
        for b in adjacency[a]:
            if a < b and (a, b) not in tree and (b, a) not in tree:
                loops.append({"cells": [a, b],
                              "holonomy": int(orient[a] * agreement[(a, b)] * orient[b])})
    frustrated = sum(1 for loop in loops if loop["holonomy"] < 0)
    return {
        "grade": grade, "n_cells": n, "n_loops": len(loops), "loops": loops,
        "frustrated": frustrated,
        "rate": (Fraction(frustrated, len(loops)) if loops else Fraction(0)),
        "orientable": frustrated == 0,
        "reading": ("+1 on every independent loop is coherent orientability; a -1 is the "
                    "obstruction, and it survives any per-cell sign flip"),
    }


def _grade_boundary(rex, grade: int):
    """That grade's boundary operator as scipy sparse, or None when the grade is absent."""
    from rexgraph.graded_boundary import graded_boundaries_from_rex
    boundaries = graded_boundaries_from_rex(rex)
    if grade < 1 or grade > len(boundaries):
        return None
    B = boundaries[grade - 1]
    return None if B.shape[1] == 0 else B


def face_support(column) -> int:
    """|supp(c_f)|, which IS the gon. The shape of a face is how many relations bound it,
    and that is independent of the grade: grade 2 says a cell is a face, not how many
    sides it has."""
    return sum(1 for x in column if x != 0)


####
# Detection: find candidate faces, then solve them
####
def _edge_supports(rex):
    """supp(e) as a frozenset of vertices, per relation. Arity-general: reads the whole
    boundary, not {src, tgt}."""
    rex._ensure_clean()
    bp, bi = rex._boundary_ptr, rex._boundary_idx
    if bp is None:
        src, tgt = rex._ensure_src_tgt()
        return [frozenset((int(s), int(t))) for s, t in zip(src, tgt, strict=True)]
    return [frozenset(int(bi[j]) for j in range(int(bp[e]), int(bp[e + 1])))
            for e in range(int(rex.nE))]


def cycle_basis(rex, *, traversal="bfs"):
    """A basis of ker(B1), the cycle space, exact over the rationals. Arity-general.

    Dispatches, the way the definition demands rather than the way a graph library would.

    PURE PAIRWISE -> a spanning FOREST traversal. A traversal is a GRADIENT: it assigns a
    potential outward from a root, and the tree it leaves spans im(B1^T). The cycle space
    is that gradient's complement. BFS is that gradient path taken at WIDTH and DFS the
    same path taken at LENGTH; both span im(B1^T), so both give a valid basis, and the
    choice changes WHICH fundamental cycles come back, not whether they are cycles. A
    forest, not a tree: one root per component, or every edge of every other component
    reads as a back edge.

    ANY ARITY ABOVE TWO -> the kernel outright. rank(B1) = n0 - c is a GRAPH identity,
    and an arity-k relation touches k vertices while contributing rank one, so reaching a
    new VERTEX stops meaning reaching a new DIRECTION. Only the second is what a cycle is
    the absence of, and a traversal cannot see it: the double-T has no cycle, but a walk
    meets the shared pair twice and reports one.

    Returns a list of exact-rational vectors of length nE, of length nE - rank(B1).
    """
    rex._ensure_clean()
    nE = int(rex.nE)
    if nE == 0:
        return []
    sup = _edge_supports(rex)
    if any(len(s) != 2 for s in sup):
        return _cycle_basis_kernel(rex)
    return _cycle_basis_traversal(rex, sup, traversal=traversal)


def _cycle_basis_kernel(rex):
    """ker(B1) by exact elimination. The definition, so arity cannot be assumed."""
    nE = int(rex.nE)
    cols = _exact_b1_block(rex, np.arange(nE, dtype=np.int64))
    verts = sorted({v for col in cols for v in col})
    if not verts:
        return []
    row_of = {v: i for i, v in enumerate(verts)}
    A = [[Fraction(0)] * nE for _ in verts]
    for j, col in enumerate(cols):
        for v, c in col.items():
            A[row_of[v]][j] = c

    n_rows = len(verts)
    pivot_col_of_row: list[int] = []
    r = 0
    for c in range(nE):
        piv = next((i for i in range(r, n_rows) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = Fraction(1) / A[r][c]
        A[r] = [x * inv for x in A[r]]
        for i in range(n_rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r], strict=True)]
        pivot_col_of_row.append(c)
        r += 1
        if r == n_rows:
            break

    pivots = set(pivot_col_of_row)
    out = []
    for f in (c for c in range(nE) if c not in pivots):
        x = [Fraction(0)] * nE
        x[f] = Fraction(1)
        for i, pc in enumerate(pivot_col_of_row):
            x[pc] = -A[i][f]
        out.append(_clear_denominators(x))
    return out


def _cycle_basis_traversal(rex, sup, *, traversal="bfs"):
    """Fundamental cycles of a spanning forest, valid where every relation is 2-ary.

    The traversal's job is to find the SUPPORT of each fundamental cycle: one non-tree
    relation plus the tree path closing it. The coefficients are then solved on that
    support by `solve_face_column`, which is the same exact primitive faces use. Deriving
    path signs by hand here would be a second implementation of the one thing that is
    already solved exactly, and it is where this first went wrong.

    `traversal` picks the gradient path: "bfs" takes it at width, "dfs" at length. Both
    span im(B1^T), so both yield a basis; they differ in which fundamental cycles come
    back, not in whether the vectors are cycles.
    """
    from collections import deque

    nV, nE = int(rex.nV), int(rex.nE)
    adj: dict[int, list] = {}
    for e, sset in enumerate(sup):
        if len(sset) != 2:
            continue
        a, b = tuple(sset)
        adj.setdefault(a, []).append((b, e))
        adj.setdefault(b, []).append((a, e))

    parent_edge = [-1] * nV
    parent_of = [-1] * nV
    seen = [False] * nV
    tree = [False] * nE
    for root in range(nV):
        if seen[root] or root not in adj:
            continue
        seen[root] = True
        frontier = deque([root])
        pop = frontier.popleft if traversal == "bfs" else frontier.pop
        while frontier:
            v = pop()
            for w, e in adj.get(v, ()):
                if not seen[w]:
                    seen[w] = True
                    parent_of[w] = v
                    parent_edge[w] = e
                    tree[e] = True
                    frontier.append(w)

    def to_root(v):
        chain = []
        while parent_edge[v] != -1:
            chain.append(parent_edge[v])
            v = parent_of[v]
        return v, chain

    out = []
    for e in range(nE):
        if tree[e] or len(sup[e]) != 2:
            continue
        a, b = tuple(sup[e])
        ra, pa = to_root(a)
        rb, pb = to_root(b)
        if ra != rb:
            continue                                   # different components: no cycle
        support = sorted({e, *pa, *pb})
        col = solve_face_column(rex, np.asarray(support, dtype=np.int64))
        if col is None:
            continue
        x = [Fraction(0)] * nE
        for idx, ed in enumerate(support):
            x[ed] = col[idx]
        out.append(x)
    return out


def _clear_denominators(x):
    lead = next((v for v in x if v != 0), None)
    if lead is None:
        return x
    x = [v / lead for v in x]
    den = 1
    for v in x:
        den = den * v.denominator // np.gcd(den, v.denominator)
    return [v * den for v in x]


def find_cycles(rex, k):
    """Candidate k-gons: cycle-space basis vectors whose SUPPORT has size k.

    The gon is |supp(c)|, and that is the only place it lives, so this reads it off the
    basis rather than walking vertices. Arity-general as a result: a branching relation
    participates in a cycle exactly when the kernel says it does, and a relation that IS
    the mean of two others forms one.
    """
    return [np.asarray([e for e, v in enumerate(c) if v != 0], dtype=np.int32)
            for c in cycle_basis(rex) if face_support(c) == k]


def find_hyperface_groups(rex):
    """Candidate hyperfaces: each branching relation together with the relations whose
    support lies inside its own.

    This is the boundary INTERSECTION, and it invents nothing. No pairwise relations are
    added, which would be clique expansion, and no hub vertex is added, which would be
    star expansion. The branching relation stays the single column it already is; what is
    being asked is whether the surrounding relations span enough of its boundary to cancel
    it, and `solve_face_column` answers by solving rather than assuming.
    """
    sup = _edge_supports(rex)
    groups = []
    for h, sh in enumerate(sup):
        if len(sh) < 3:
            continue                                   # not branching: nothing to close
        legs = [e for e, se in enumerate(sup)
                if e != h and len(se) >= 1 and se <= sh]
        if legs:
            groups.append(np.asarray([h] + sorted(legs), dtype=np.int32))
    return groups


def autoface(rex, k=3):
    """Attach every k-gon the connectivity allows, coefficients solved. Returns the count.

    `k` may be an int or an iterable of ints. Geometry FROM topology: the choice condition
    on pure topology, filling what the connectivity allows. That is a different source of
    geometry from weighting, which forces a metric whatever the face choice.
    """
    ks = [k] if isinstance(k, int) else list(k)
    cand = []
    for kk in ks:
        if kk >= 3:
            cand.extend(find_cycles(rex, kk))
    if not cand:
        return 0
    before = int(rex.nF_hodge)
    rex.add_faces(cand, signs=None)
    return int(rex.nF_hodge) - before


def auto_hyperface(rex):
    """Close each branching relation against the relations spanning its boundary.

    A relation alone bounds nothing, correctly, since nothing is enclosed; a partial
    overlap does not cancel and is refused. Returns the number attached.
    """
    groups = find_hyperface_groups(rex)
    if not groups:
        return 0
    before = int(rex.nF_hodge)
    rex.add_faces(groups, signs=None)
    return int(rex.nF_hodge) - before
