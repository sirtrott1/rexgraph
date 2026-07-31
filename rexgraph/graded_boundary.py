"""rexgraph.graded_boundary: a general, graded, mixed-arity boundary builder.

A relational complex (rex) is a finite sequence of signed integer boundary maps

    C_G --B_G--> C_{G-1} --> ... --> C_1 --B_1--> C_0

with entries in {-1, 0, +1}, satisfying the chain condition B_{d-1} B_d = 0 at every
consecutive pair. Edges are primitive and vertices are derived: a vertex exists exactly
when some column of B_1 is nonzero in its row. Nothing is assumed beyond the integers:
no metric, no manifold, no continuity.

Each ``B_d : C_d -> C_{d-1}`` is a *signed* sparse matrix whose COLUMNS carry any number
of nonzeros. The number of nonzeros in a column is the cell's *arity* and is INDEPENDENT
of its *grade* (dimension):

    nnz = 1  -> a "witness" cell (single face),
    nnz = 2  -> an ordinary cell (pairwise edge / bigon),
    nnz = k  -> a "branching" cell (k-ary edge, n-gon face, ...).

Pairwise and branching edges coexist; triangle, pentagon, hexagon and n-gon faces
coexist; and grades stack arbitrarily high. The only law is the chain condition

    d o d = 0   <=>   B_d @ B_{d+1} = 0   for every consecutive pair,

which is a *structural* (sparse) zero, never a densified product.

This module is the single, kernel-free source of truth for building, verifying and
reading graded boundaries. It is pure Python + scipy.sparse; it does not touch the
Cython core, and it is the generalization of
``rexgraph.dirac_propagator._boundaries_from_rex``.

Sign convention (positional, matching the existing B1 storage in graph.py):
a d-cell given as a plain list of (d-1)-cell indices ``[i0, i1, ...]`` has the FIRST
index signed ``-1`` and every remaining index signed ``+1`` (source ``-1``, targets
``+1``). Cells may instead be given in explicit ``[(index, sign), ...]`` form for
arbitrary orientations. Both forms are accepted per cell.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

__all__ = [
    "build_graded_boundaries",
    "verify_chain",
    "graded_laplacians",
    "betti_numbers",
    "graded_boundaries_from_rex",
    "truncated_icosahedron_3rex",
    "solid_octahedron_3rex",
    "square_pyramid_3rex",
]

_f64 = np.float64


# ---------------------------------------------------------------------------
# Cell parsing
# ---------------------------------------------------------------------------

def _is_signed_cell(cell) -> bool:
    """True if ``cell`` is in explicit ``[(index, sign), ...]`` form rather than a
    plain list of indices. A signed cell's every element is a length-2 pair whose
    second entry is a +/-1 sign; a plain cell's elements are bare integers."""
    if len(cell) == 0:
        return False
    for x in cell:
        if not isinstance(x, (tuple, list, np.ndarray)):
            return False
        if len(x) != 2:
            return False
        s = x[1]
        if s not in (1, -1, 1.0, -1.0):
            return False
    return True


def _cell_entries(cell) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, signs)`` for one cell.

    Plain form ``[i0, i1, ...]`` -> first index ``-1``, the rest ``+1`` (positional).
    Signed form ``[(i, s), ...]`` -> exactly as given.
    """
    if _is_signed_cell(cell):
        idx = np.fromiter((int(x[0]) for x in cell), dtype=np.int64, count=len(cell))
        sgn = np.fromiter((float(x[1]) for x in cell), dtype=_f64, count=len(cell))
        return idx, sgn
    idx = np.asarray(cell, dtype=np.int64).ravel()
    sgn = np.ones(idx.shape[0], dtype=_f64)
    if idx.shape[0] >= 1:
        sgn[0] = -1.0
    return idx, sgn


def build_graded_boundaries(cells_by_grade) -> List[sp.csr_matrix]:
    """Build the signed boundary maps ``[B_1, B_2, ..., B_G]`` of a graded complex.

    Parameters
    ----------
    cells_by_grade : sequence
        ``cells_by_grade[0]`` is the vertex count ``n_V`` (an int). For ``d >= 1``,
        ``cells_by_grade[d]`` is a list of d-cells; each d-cell is either a plain
        list of ``(d-1)``-cell indices (positional signs: first ``-1``, rest ``+1``)
        or an explicit ``[(index, sign), ...]`` list. Mixed arity within a grade is
        allowed and expected.

    Returns
    -------
    list of scipy.sparse.csr_matrix
        ``B_d`` has shape ``(n_{d-1}, n_d)`` and is signed with arbitrary column
        arity. Length ``G`` where ``G`` is the top grade present.
    """
    if len(cells_by_grade) == 0:
        raise ValueError("cells_by_grade must at least declare the vertex count")
    n_prev = int(cells_by_grade[0])
    boundaries: List[sp.csr_matrix] = []

    for d in range(1, len(cells_by_grade)):
        cells = cells_by_grade[d]
        n_cells = len(cells)
        rows: List[np.ndarray] = []
        cols: List[np.ndarray] = []
        vals: List[np.ndarray] = []
        for j, cell in enumerate(cells):
            idx, sgn = _cell_entries(cell)
            if idx.shape[0] == 0:
                continue
            rows.append(idx)
            cols.append(np.full(idx.shape[0], j, dtype=np.int64))
            vals.append(sgn)
        if rows:
            r = np.concatenate(rows)
            c = np.concatenate(cols)
            v = np.concatenate(vals)
        else:
            r = np.zeros(0, dtype=np.int64)
            c = np.zeros(0, dtype=np.int64)
            v = np.zeros(0, dtype=_f64)
        B = sp.coo_matrix((v, (r, c)), shape=(n_prev, n_cells)).tocsr()
        boundaries.append(B)
        n_prev = n_cells

    return boundaries


# ---------------------------------------------------------------------------
# Verification, Laplacians, homology
# ---------------------------------------------------------------------------

def verify_chain(boundaries: Sequence[sp.spmatrix], tol: float = 1e-9) -> Tuple[bool, float]:
    """Sparsely check ``B_d @ B_{d+1} == 0`` for every consecutive pair.

    Never densifies: each product is a sparse matmul and only its stored nonzeros
    are inspected.

    Returns
    -------
    (ok, max_residual)
        ``ok`` is True iff ``max_residual <= tol``.
    """
    max_res = 0.0
    for d in range(len(boundaries) - 1):
        prod = (boundaries[d].tocsr() @ boundaries[d + 1].tocsr())
        if prod.nnz:
            max_res = max(max_res, float(np.abs(prod.data).max()))
    return (max_res <= tol), max_res


def graded_laplacians(boundaries: Sequence[sp.spmatrix]) -> List[sp.csr_matrix]:
    """The Hodge Laplacian ``L_d`` per grade, sparse.

    ``L_d = B_d^T B_d + B_{d+1} B_{d+1}^T`` with the boundary terms dropped where
    they do not exist, i.e. ``L_0 = B_1 B_1^T`` and the top-grade Laplacian is
    ``L_G = B_G^T B_G``.

    Returns a list of length ``G + 1`` (one operator per grade ``0..G``).
    """
    B = [b.tocsr() for b in boundaries]
    G = len(B)                          # top grade index; grades run 0..G
    sizes = [B[0].shape[0]] + [b.shape[1] for b in B] if B else [0]
    out: List[sp.csr_matrix] = []
    for g in range(G + 1):
        n_g = sizes[g]
        L = sp.csr_matrix((n_g, n_g), dtype=_f64)
        if g >= 1:                       # down: B_g^T B_g,  B_g = B[g-1]
            L = L + (B[g - 1].T @ B[g - 1])
        if g <= G - 1:                   # up: B_{g+1} B_{g+1}^T,  B_{g+1} = B[g]
            L = L + (B[g] @ B[g].T)
        out.append(L.tocsr())
    return out


def _is_integer_matrix(M: sp.spmatrix) -> bool:
    """True if every stored entry is an integer (the unweighted boundary maps are)."""
    d = M.data
    return d.size == 0 or bool(np.all(d == np.round(d)))


from collections import OrderedDict as _OrderedDict

# Content-addressed memo for the exact integer rank. The rational column reduction below
# is the dominant cost on a large monitor step, and the SAME integer boundary map is
# reduced more than once per step (e.g. the pairwise interaction complex and the faced
# coordination complex share an identical B1). The key is the matrix's exact canonical
# content (shape + CSC structure + rounded integer data), so a hit returns a value that is
# byte-for-byte the same matrix - zero collision/staleness risk (dict compares keys
# exactly). Bounded so it never grows without limit; a race only ever costs a redundant
# (correct) recompute, so it is safe under the coordinator's thread lane too.
_RANK_MEMO: "_OrderedDict[tuple, int]" = _OrderedDict()
_RANK_MEMO_MAX = 64


def _exact_rank_reduction(M: sp.spmatrix) -> int:
    """EXACT rank of an INTEGER sparse matrix via column reduction over Q (Fraction) -
    eigen-free, NO SVD, no eigendecomposition, no dense operator. Each column is
    reduced against the registered pivots (lowest-nonzero-row 'low' convention, as in
    persistence reduction); rank = number of columns that keep a pivot. This is the
    canon's `rank(B_k) via Z/Q elimination` (Part III) and is exact for integer /
    rational entries. Columns are sparse dicts, so cost tracks fill, not n^3.

    Memoized on exact matrix content (see :data:`_RANK_MEMO`)."""
    from fractions import Fraction as Fr
    A = M.tocsc()
    A.sort_indices()                        # canonical CSC for a stable content key
    indptr, indices, data = A.indptr, A.indices, A.data
    idata = np.round(data).astype(np.int64)
    key = (A.shape, indptr.tobytes(), indices.tobytes(), idata.tobytes())
    hit = _RANK_MEMO.get(key)
    if hit is not None:
        _RANK_MEMO.move_to_end(key)
        return hit

    pivots: dict = {}                       # pivot_row -> reduced column {row: Fraction}
    rank = 0
    for j in range(A.shape[1]):
        col = {int(indices[k]): Fr(int(idata[k]))
               for k in range(indptr[j], indptr[j + 1])}
        while col:
            low = max(col)                  # 'low' pivot = highest row index present
            piv = pivots.get(low)
            if piv is None:
                pivots[low] = col
                rank += 1
                break
            factor = col[low] / piv[low]
            for r, val in piv.items():
                nv = col.get(r, Fr(0)) - factor * val
                if nv == 0:
                    col.pop(r, None)
                else:
                    col[r] = nv

    _RANK_MEMO[key] = rank
    _RANK_MEMO.move_to_end(key)
    if len(_RANK_MEMO) > _RANK_MEMO_MAX:
        _RANK_MEMO.popitem(last=False)
    return rank


def _sparse_rank(M: sp.spmatrix, tol: float = 1e-9) -> int:
    """Rank of a sparse matrix. Betti comes from RANKS (the canon), not spectra.

    For INTEGER boundary maps (the unweighted topology) rank is computed EXACTLY and
    EIGEN-FREE by rational column reduction (:func:`_exact_rank_reduction`), the
    canon's Z/Q-elimination path - no SVD, no dense operator. Only genuinely
    non-integer (float-weighted) matrices fall back to the dense/truncated SVD.
    """
    if M.nnz == 0 or min(M.shape) == 0:
        return 0
    if _is_integer_matrix(M):
        return _exact_rank_reduction(M)
    m, n = M.shape
    # Densify only when the matrix is small enough to be harmless; boundary maps of
    # the complexes this module builds are far below this bound.
    if min(m, n) <= 1500:
        s = np.linalg.svd(M.toarray(), compute_uv=False)
        if s.size == 0:
            return 0
        thresh = tol * s[0] * max(m, n)
        return int(np.sum(s > max(thresh, tol)))
    # Large: estimate via truncated SVD (rank cannot exceed k here; the complexes
    # exercised in this library never reach this branch).
    k = min(min(m, n) - 1, 400)
    s = sp.linalg.svds(M.asfptype(), k=k, return_singular_vectors=False)
    thresh = tol * s.max() * max(m, n)
    return int(np.sum(s > max(thresh, tol)))


def _beta0_components(B1: sp.spmatrix) -> int:
    """beta_0 = number of connected components over the vertices, from the 0/1
    incidence pattern of ``B_1`` (combinatorial, via union-find on the graph whose
    cliques are the edge supports). Isolated vertices count as components."""
    nV = B1.shape[0]
    if nV == 0:
        return 0
    Bc = B1.tocsc()
    parent = list(range(nV))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    indptr, indices = Bc.indptr, Bc.indices
    for e in range(Bc.shape[1]):
        support = indices[indptr[e]:indptr[e + 1]]
        for k in range(1, len(support)):
            union(int(support[0]), int(support[k]))
    return len({find(v) for v in range(nV)})


def betti_numbers(boundaries: Sequence[sp.spmatrix], tol: float = 1e-9) -> List[int]:
    """Betti numbers ``[beta_0, ..., beta_G]`` from ranks.

    ``beta_g = dim ker(B_g) - rank(B_{g+1}) = n_g - rank(B_g) - rank(B_{g+1})`` with
    ``rank(B_0) = rank(B_{G+1}) = 0``. ``beta_0`` is taken combinatorially as the
    number of connected components (equivalently ``n_0 - rank(B_1)``), which is exact
    and cheap.
    """
    B = [b.tocsr() for b in boundaries]
    G = len(B)
    if G == 0:
        return []
    sizes = [B[0].shape[0]] + [b.shape[1] for b in B]
    ranks = [_sparse_rank(b, tol) for b in B]        # ranks[d] = rank(B_{d+1})

    betti: List[int] = []
    for g in range(G + 1):
        n_g = sizes[g]
        rank_down = ranks[g - 1] if g >= 1 else 0    # rank(B_g)
        rank_up = ranks[g] if g <= G - 1 else 0      # rank(B_{g+1})
        if g == 0:
            betti.append(_beta0_components(B[0]))
        else:
            betti.append(int(n_g - rank_down - rank_up))
    return betti


# ---------------------------------------------------------------------------
# Reading graded boundaries off a RexGraph (single source of truth)
# ---------------------------------------------------------------------------

def graded_boundaries_from_rex(rex) -> List[sp.csr_matrix]:
    """The full sparse boundary list ``[B_1, B_2, B_3, ...]`` of a RexGraph.

    This is the generalization of ``dirac_propagator._boundaries_from_rex`` and the
    single source of truth for reading a rex's graded structure:

      * ``B_1`` always, from the rex's own signed vertex-edge incidence;
      * ``B_2`` when ``nF > 0``, from the chain-consistent Hodge slice
        (``_B2_hodge_dual``), so whatever face arity the complex carries is kept;
      * ``B_3, B_4, ...`` when the rex additionally stores higher boundaries in the
        optional ``_graded_duals`` attribute (populated by ``RexGraph.from_cells``).

    Every returned matrix is scipy CSR; nothing is densified.
    """
    from rexgraph.core._sparse import to_scipy_csr

    B1 = _rex_b1_csr(rex)
    boundaries: List[sp.csr_matrix] = [B1]

    if int(getattr(rex, "nF", 0)) > 0 and getattr(rex, "_B2_hodge_dual", None) is not None:
        boundaries.append(to_scipy_csr(rex._B2_hodge_dual).tocsr())

    duals = getattr(rex, "_graded_duals", None)
    if duals:
        for Bd in duals:
            boundaries.append(sp.csr_matrix(Bd))
    return boundaries


def _rex_b1_csr(rex) -> sp.csr_matrix:
    """B_1 (nV x nE, signed) of a rex as scipy CSR, via the rex's own DualCSR."""
    from rexgraph.core._sparse import to_scipy_csr
    return to_scipy_csr(rex._B1_dual).tocsr()


# ---------------------------------------------------------------------------
# Constructor helpers: genuine grade-3 complexes (d^2 = 0)
# ---------------------------------------------------------------------------

def _order_face_ccw(points: np.ndarray, face_idx: Sequence[int],
                    center: np.ndarray) -> List[int]:
    """Order a convex, planar face's vertices CCW as seen from OUTSIDE the solid.

    The outward normal is the direction from the solid's centroid to the face
    centroid; sorting the (coplanar, convex) face vertices by their polar angle in
    the plane orthogonal to that normal yields the boundary loop with a globally
    consistent (outward) orientation - which is exactly what makes the closed
    surface orientable, hence ``B_2 @ 1 = 0`` and ``B_2 B_3 = 0``.
    """
    fi = list(face_idx)
    pts = points[fi]
    fc = pts.mean(axis=0)
    normal = fc - center
    nrm = np.linalg.norm(normal)
    if nrm < 1e-12:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / nrm
    # An in-plane basis (e1, e2) with e2 = normal x e1, so angle increases CCW
    # about the outward normal.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, normal) * normal
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    ang = []
    for p in pts:
        d = p - fc
        ang.append(np.arctan2(np.dot(d, e2), np.dot(d, e1)))
    order = np.argsort(ang)
    return [fi[k] for k in order]


def _polyhedron_3rex(points: np.ndarray, face_vertex_sets: Sequence[Sequence[int]]):
    """Assemble a SOLID convex polyhedron as a 3-rex ``cells_by_grade``.

    ``points`` are the vertex coordinates; ``face_vertex_sets`` lists, per face, the
    (unordered) vertex indices bounding it. Faces are oriented outward, edges are
    derived from the oriented face loops, and a single volume (grade-3) cell is added
    bounded by all faces with ``+1`` signs - which closes as ``B_2 B_3 = 0`` because
    the outward orientation makes every edge cancel between its two faces.

    Returns ``cells_by_grade = [nV, edges, faces_signed, [volume]]``.
    """
    points = np.asarray(points, dtype=_f64)
    nV = points.shape[0]
    center = points.mean(axis=0)

    # Order each face CCW outward.
    ordered_faces = [_order_face_ccw(points, fs, center) for fs in face_vertex_sets]

    # Derive edges from the oriented face loops; store each with a fixed orientation
    # (first-seen direction) so face signs are relative to that stored direction.
    edge_index = {}
    edges: List[List[int]] = []
    for loop in ordered_faces:
        L = len(loop)
        for k in range(L):
            a, b = loop[k], loop[(k + 1) % L]
            key = frozenset((a, b))
            if key not in edge_index:
                edge_index[key] = len(edges)
                edges.append([a, b])

    # Signed grade-2 faces in edge space.
    faces_signed: List[List[Tuple[int, float]]] = []
    for loop in ordered_faces:
        L = len(loop)
        col: List[Tuple[int, float]] = []
        for k in range(L):
            a, b = loop[k], loop[(k + 1) % L]
            eidx = edge_index[frozenset((a, b))]
            stored = edges[eidx]
            sign = 1.0 if (stored[0] == a and stored[1] == b) else -1.0
            col.append((eidx, sign))
        faces_signed.append(col)

    # Single volume bounded by every face (+1); outward orientation => B2 @ 1 = 0.
    volume = [[(f, 1.0) for f in range(len(faces_signed))]]

    return [nV, edges, faces_signed, volume]


_PHI = (1.0 + 5.0 ** 0.5) / 2.0


def _icosahedron():
    """Icosahedron combinatorics from golden-ratio coordinates: returns
    ``(points[12x3], neighbors: list[set], faces: list[(a,b,c)])``.

    Edges are vertex pairs at the (minimal) squared distance; triangular faces are
    triples that are pairwise adjacent. Coordinate-driven, so exact and orientation-
    agnostic - the truncation and orientation are handled downstream.
    """
    p = _PHI
    verts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            verts.append((0.0, s1 * 1.0, s2 * p))
            verts.append((s1 * 1.0, s2 * p, 0.0))
            verts.append((s1 * p, 0.0, s2 * 1.0))
    P = np.array(verts, dtype=_f64)
    # Deduplicate should not be needed (12 distinct), but guard against ordering.
    nV = P.shape[0]
    # Pairwise squared distances; edge length^2 == 4 for a unit icosahedron here.
    d2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2)
    off = d2 + np.eye(nV) * 1e9
    emin = off.min()
    adj = np.abs(d2 - emin) < 1e-6
    neighbors = [set(np.nonzero(adj[i])[0].tolist()) for i in range(nV)]
    # Triangular faces: mutually adjacent triples.
    faces = []
    for a in range(nV):
        for b in neighbors[a]:
            if b <= a:
                continue
            for c in neighbors[a] & neighbors[b]:
                if c <= b:
                    continue
                faces.append((a, b, c))
    return P, neighbors, faces


def truncated_icosahedron_3rex():
    """The SOLID truncated icosahedron (soccer ball) as a 3-rex ``cells_by_grade``.

    60 vertices, 90 edges, 32 faces (12 pentagons + 20 hexagons = mixed grade-2
    arity), 1 volume. Built programmatically by truncating the icosahedron: each
    icosahedron vertex ``v`` with an incident edge to neighbor ``n`` becomes a
    "corner" point ``v + (n - v)/3``; the 5 corners around ``v`` form a pentagon and
    the 6 corners of each icosahedron triangle form a hexagon. Faces are oriented
    outward so the shell is orientable and the single enclosed volume closes with
    ``B_2 B_3 = 0``.

    This is exactly "a topological 2-sphere with its boundary encoded as a
    5-6-gon-3-rex", promoted to a solid by the enclosing 3-cell.
    """
    P, neighbors, faces = _icosahedron()

    # Corner points, indexed by the ordered pair (vertex, neighbor).
    corner_index = {}
    corner_pts: List[np.ndarray] = []
    for v in range(P.shape[0]):
        for n in neighbors[v]:
            corner_index[(v, n)] = len(corner_pts)
            corner_pts.append(P[v] + (P[n] - P[v]) / 3.0)
    pts = np.array(corner_pts, dtype=_f64)

    face_vertex_sets: List[List[int]] = []
    # Pentagons: the 5 corners around each icosahedron vertex.
    for v in range(P.shape[0]):
        face_vertex_sets.append([corner_index[(v, n)] for n in neighbors[v]])
    # Hexagons: the 6 corners of each icosahedron triangle {a,b,c}.
    for (a, b, c) in faces:
        face_vertex_sets.append([
            corner_index[(a, b)], corner_index[(b, a)],
            corner_index[(b, c)], corner_index[(c, b)],
            corner_index[(c, a)], corner_index[(a, c)],
        ])

    return _polyhedron_3rex(pts, face_vertex_sets)


def solid_octahedron_3rex():
    """The SOLID octahedron as a 3-rex ``cells_by_grade``: 6 vertices, 12 edges,
    8 triangular faces (arity-3), 1 volume. A simple, fully triangulated grade-3
    complex with ``d^2 = 0``."""
    pts = np.array([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ], dtype=_f64)
    # 8 faces, one per (x-sign, y-sign, z-sign) octant.
    faces = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ]
    return _polyhedron_3rex(pts, faces)


def square_pyramid_3rex():
    """A SOLID square pyramid as a small MIXED-ARITY 3-rex ``cells_by_grade``:
    5 vertices, 8 edges, 5 faces (4 triangles of arity 3 + 1 square base of arity 4),
    1 volume. Demonstrates mixed grade-2 arity in a genuine grade-3 complex."""
    pts = np.array([
        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],   # square base
        [0.0, 0.0, 1.5],                        # apex
    ], dtype=_f64)
    faces = [
        [0, 1, 2, 3],       # square base (arity 4)
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],   # 4 triangles (arity 3)
    ]
    return _polyhedron_3rex(pts, faces)
