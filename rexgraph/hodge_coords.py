"""Coordinates for the three Hodge spaces.

`hodge` splits an edge signal into gradient, curl and harmonic parts and hands
back three vectors of length nE. Each of those is already the image of something
smaller, and the smaller thing is what was solved for:

    grad = B1^T phi      phi on the vertices        nV
    curl = B2 psi        psi on the faces           nF
    harm = H c           c on the harmonic frame    dim_H

So a flow that needs nE numbers in the edge space needs nV + nF + dim_H in these,
and the coordinates are what the components are built from rather than a second
reading of them. Both solvers computed phi and psi and discarded them; the
harmonic projector computed c and returned H c. This module returns the small
side.

The harmonic frame is the interesting one. Its axes are cycles, `dim_H` of them,
and a flow's position along them is the whole of its harmonic content. That is a
coordinate system on the harmonic plane with one axis per independent hole.

Redundancy is exact and worth stating. phi is fixed up to a constant on each
connected component and psi up to ker(B2), so the chart carries beta_0 +
(nF - rank(B2)) more numbers than the space has dimensions. Quotient those out
and what is left is rank(B1) + rank(B2) + dim_H, which is nE exactly. Use
`coordinate_dims` to read both counts.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from rexgraph.harmonic_sparse import harmonic_basis, harmonic_coordinates

__all__ = [
    "HodgeCoords",
    "coordinate_dims",
    "from_harmonic_coords",
    "from_hodge_coords",
    "harmonic_coords",
    "harmonic_frame",
    "harmonic_gram_det",
    "complex_structure",
    "harmonic_metric",
    "harmonic_closure",
    "harmonic_spread",
    "harmonic_structure_constants",
    "hodge_coords",
    "structure_alignment",
]

_f64 = np.float64

#: phi on the vertices, psi on the faces, c on the harmonic frame
HodgeCoords = namedtuple("HodgeCoords", "phi psi harmonic")


def harmonic_frame(rex):
    """The harmonic plane's axes as a sparse nE x dim_H matrix.

    One column per independent hole, each a cycle carrying no face flux. This is
    `harmonic_basis`, named for what it is used as here.
    """
    return harmonic_basis(rex)


def harmonic_coords(rex, flow, *, frame=None):
    """Where `flow` sits on the harmonic frame: f64[dim_H].

    Solves the normal equations on the frame's Gram, which is a sparse SPD
    dim_H x dim_H because cycles share few edges. `H @ harmonic_coords(...)` is
    the harmonic projection, so this is the projector's small side.
    """
    H = harmonic_frame(rex) if frame is None else frame
    from rexgraph.harmonic_sparse import as_edge_signal
    return harmonic_coordinates(H, as_edge_signal(flow, rex.nE, what="flow"))


def harmonic_metric(rex, *, frame=None):
    """The harmonic plane's metric: the frame Gram `HᵀH`, sparse dim_H x dim_H.

    The frame's axes are cycles and cycles share edges, so the axes are not
    orthogonal and the coordinates are not an isometric embedding. Lengths and
    angles in the plane are taken through this form, never as plain dot products.
    Measured on K6, reading the coordinates as Euclidean puts the angle off by up
    to 0.36 in spread and moves half of 200 random pairs by more than 0.05.
    """
    import scipy.sparse as sp

    H = harmonic_frame(rex) if frame is None else frame
    Hs = H.tocsr() if sp.issparse(H) else sp.csr_matrix(np.asarray(H, dtype=_f64))
    return (Hs.T @ Hs).tocsc()


def harmonic_spread(rex, u, v, *, frame=None):
    """Spread between two flows' harmonic parts, computed in the plane.

    Spread is sin^2 of the angle, so it stays rational and needs no square root.
    Taken through `harmonic_metric`, this is the spread of the ambient harmonic
    projections to 3.3e-16, at dim_H terms rather than nE.

    Returns 0.0 when either flow has no harmonic part, and when the complex has
    no holes at all, since there is then no angle to speak of.
    """
    H = harmonic_frame(rex) if frame is None else frame
    if H.shape[1] == 0:
        return 0.0
    G = harmonic_metric(rex, frame=H)
    a = harmonic_coords(rex, u, frame=H)
    b = harmonic_coords(rex, v, frame=H)
    qa = float(a @ (G @ a))
    qb = float(b @ (G @ b))
    if qa <= 0.0 or qb <= 0.0:
        return 0.0
    ab = float(a @ (G @ b))
    return 1.0 - (ab * ab) / (qa * qb)


def from_harmonic_coords(rex, c, *, frame=None):
    """The edge signal a set of harmonic coordinates names: f64[nE]."""
    import scipy.sparse as sp

    H = harmonic_frame(rex) if frame is None else frame
    c = np.atleast_1d(np.asarray(c, dtype=_f64).ravel())
    if H.shape[1] == 0:
        return np.zeros(H.shape[0], dtype=_f64)
    Hs = H.tocsr() if sp.issparse(H) else sp.csr_matrix(np.asarray(H, dtype=_f64))
    return np.asarray(Hs @ c).ravel()


def hodge_coords(rex, flow, *, frame=None):
    """`flow` in all three Hodge spaces at once: HodgeCoords(phi, psi, harmonic).

    phi and psi come from the same solve `hodge` runs, so this costs one
    decomposition and not two. The harmonic coordinates are a separate solve on
    the frame's Gram, which is small.
    """
    from rexgraph.core import _hodge
    from rexgraph.harmonic_sparse import as_edge_signal

    rex._ensure_clean()
    g = np.ascontiguousarray(as_edge_signal(flow, rex.nE, what="flow"))
    _, _, _, phi, psi = _hodge.hodge_decomposition(
        rex._B1_dual, rex._B2_hodge_dual, g, potentials=True)
    return HodgeCoords(phi=np.asarray(phi, dtype=_f64),
                       psi=np.asarray(psi, dtype=_f64),
                       harmonic=harmonic_coords(rex, g, frame=frame))


def from_hodge_coords(rex, coords, *, frame=None):
    """Rebuild the edge signal from its coordinates: f64[nE].

    `B1^T phi + B2 psi + H c`. Inverts `hodge_coords` on the signal, not on the
    coordinates: phi shifted by a constant on a component names the same flow.
    """
    from rexgraph.core._sparse import matvec, rmatvec

    rex._ensure_clean()
    phi, psi, c = coords
    out = np.zeros(rex.nE, dtype=_f64)
    phi = np.asarray(phi, dtype=_f64).ravel()
    if phi.size:
        out += np.asarray(rmatvec(rex._B1_dual, phi)).ravel()
    psi = np.asarray(psi, dtype=_f64).ravel()
    B2 = rex._B2_hodge_dual
    if psi.size and B2 is not None and B2.ncol > 0:
        out += np.asarray(matvec(B2, psi)).ravel()
    c = np.atleast_1d(np.asarray(c, dtype=_f64).ravel())
    if c.size:
        out += from_harmonic_coords(rex, c, frame=frame)
    return out


def coordinate_dims(rex, *, frame=None):
    """How many coordinates the chart carries, and how many the spaces have.

    `chart` is nV + nF + dim_H, what `hodge_coords` returns. `independent` is
    rank(B1) + rank(B2) + dim_H, which equals nE: the three Hodge spaces are
    orthogonal and together span the edge space. The difference is the gauge
    freedom in phi and psi.
    """
    rex._ensure_clean()
    H = harmonic_frame(rex) if frame is None else frame
    dim_h = int(H.shape[1])
    nV, nE = int(rex.nV), int(rex.nE)
    n_faces = int(rex._B2_hodge_dual.ncol) if rex._B2_hodge_dual is not None else 0
    # rank(B1) = nV - b0 is exact: b0 counts connected components, which is the
    # dimension of ker(B1 B1^T). rank(B2) then follows from the Hodge theorem
    # rather than a second rank computation.
    rank_b1 = nV - int(rex.betti[0])
    rank_b2 = nE - dim_h - rank_b1
    return {
        "nV": nV, "nE": nE, "nF": n_faces,
        "dim_H": dim_h,
        "rank_B1": rank_b1, "rank_B2": rank_b2,
        "chart": nV + n_faces + dim_h,
        "independent": rank_b1 + rank_b2 + dim_h,
    }


def _exact_ints(values, what):
    """`Fraction` of each value, refusing to round one that is not already integral.

    Every exact reading here rests on the frame being integer, and the frame is
    integer only while `harmonic_sparse._integer_nullspace` can carry it: past 2**53
    it declines and `_face_reduced_frame` falls back to a float SVD. Rounding that
    float silently returns a plausible integer for a quantity that is not one --
    a half-integer frame on K5 gave det 1 for a true determinant of 0.0305, so it
    is refused loudly instead. Convert deliberately, or read the float path.
    """
    from fractions import Fraction

    arr = np.asarray(values, dtype=_f64)
    if not np.array_equal(arr, np.round(arr)):
        worst = float(np.abs(arr - np.round(arr)).max())
        raise ValueError(
            f"{what} is not integral (off by {worst:.3g}), so an exact reading of it "
            "would be a rounded guess. The usual cause is a non-integer harmonic "
            "frame: _integer_nullspace declines past 2**53 and _face_reduced_frame "
            "then falls back to a float SVD.")
    flat = [Fraction(int(round(x))) for x in arr.ravel()]
    if arr.ndim == 0:
        return flat[0]
    out, k = [], 0
    for _ in range(arr.shape[0]):
        row = flat[k:k + (arr.shape[1] if arr.ndim > 1 else 1)]
        k += len(row)
        out.append(row if arr.ndim > 1 else row[0])
    return out


def _solve_gram(G, rhs, exact):
    """`G^-1 rhs`, exactly over Fraction when asked."""
    import scipy.sparse.linalg as sla
    if not exact:
        return sla.spsolve(G.tocsc(), rhs)

    n = G.shape[0]
    M = np.asarray(G.todense() if hasattr(G, "todense") else G)
    _rows = _exact_ints(np.column_stack([np.asarray(M, dtype=_f64),
                                         np.asarray(rhs, dtype=_f64).reshape(-1, 1)]),
                        "the harmonic Gram and right-hand side")
    A = [list(_rows[i])
         for i in range(n)]
    for i in range(n):
        pv = next(k for k in range(i, n) if A[k][i] != 0)
        A[i], A[pv] = A[pv], A[i]
        d = A[i][i]
        A[i] = [x / d for x in A[i]]
        for k in range(n):
            if k != i and A[k][i] != 0:
                f = A[k][i]
                A[k] = [a - f * b for a, b in zip(A[k], A[i], strict=False)]
    return [A[i][n] for i in range(n)]


def harmonic_structure_constants(rex, i, j, *, frame=None, exact=False):
    """Where the Hadamard product of two frame axes lands in the plane.

    `h_i * h_j` entrywise, projected back and read in coordinates: `G^-1 H^T p`.
    The frame axes are cycles with entries in {0, +1, -1}, so the product is
    supported exactly on the edges the two cycles share, and `H^T p` is an
    integer vector. Only the Gram solve makes it rational, which is why `exact`
    is available at all.

    One pair at a time. The full table is dim_H cubed, so it is the caller's
    choice to build.
    """
    import scipy.sparse as sp

    H = harmonic_frame(rex) if frame is None else frame
    Hs = H.tocsr() if sp.issparse(H) else sp.csr_matrix(np.asarray(H, dtype=_f64))
    col_i = np.asarray(Hs[:, i].todense()).ravel()
    col_j = np.asarray(Hs[:, j].todense()).ravel()
    q = np.asarray(Hs.T @ (col_i * col_j)).ravel()
    return _solve_gram(harmonic_metric(rex, frame=H), q, exact)


def harmonic_closure(rex, *, frame=None, exact=False):
    """How much of each Hadamard product stays in the harmonic plane.

    `closure[i,j] = ||P (h_i * h_j)||^2 / ||h_i * h_j||^2`, in [0, 1]. It is 1
    exactly when the product is itself harmonic, so the matrix says whether the
    plane is an algebra under the entrywise product. Zero marks a pair of axes
    that share no edge, where the product vanishes and there is nothing to place.

    With `q = H^T p`, `||P p||^2` is `q^T G^-1 q`, so this needs the small Gram
    and never the nE x nE projector, and never an eigendecomposition. Under
    `exact` it is a ratio of integers over det(G).

    Measured on complete graphs, whose fundamental cycle basis is all triangles:
    the diagonal is 1 - 8/(3n) and an overlapping pair is 1 - 2/n, exactly, for
    n = 4 through 9. Both approach 1, so the plane is closed only in the limit.
    Closure is read against a chosen frame, so those numbers belong to the
    triangle basis and not to K_n on its own.

    `exact` runs a rational Gram solve per column and costs orders of magnitude
    more as dim_H grows (measured 158x float at dim_H 58, 20000x at 398).
    """
    import scipy.sparse as sp

    H = harmonic_frame(rex) if frame is None else frame
    k = int(H.shape[1])
    if k == 0:
        return np.zeros((0, 0), dtype=_f64)
    Hs = H.tocsr() if sp.issparse(H) else sp.csr_matrix(np.asarray(H, dtype=_f64))
    Hd = np.asarray(Hs.todense())
    G = harmonic_metric(rex, frame=Hs)
    out = [[None] * k for _ in range(k)] if exact else np.zeros((k, k), dtype=_f64)
    for i in range(k):
        for j in range(k):
            prod = Hd[:, i] * Hd[:, j]
            denom = float(prod @ prod)
            if denom == 0.0:
                if exact:
                    from fractions import Fraction
                    out[i][j] = Fraction(0)
                continue
            q = np.asarray(Hs.T @ prod).ravel()
            x = _solve_gram(G, q, exact)
            if exact:
                qf = _exact_ints(q, "a harmonic frame product's coordinates")
                df = _exact_ints(np.asarray(denom), "a harmonic product's norm")
                out[i][j] = sum(qf[a] * x[a] for a in range(k)) / df
            else:
                out[i, j] = float(np.dot(q, x)) / denom
    return out


def harmonic_gram_det(rex, *, frame=None):
    """Exact determinant of the frame Gram, by fraction-free elimination.

    With no faces and every relation 2-ary, the frame is the full cycle space and
    this is the number of spanning FORESTS: the product over connected components
    of each component's spanning-tree count. That reduces to the spanning trees of
    the graph when it is connected, which is the case the claim was first read on.
    Verified against the Matrix-Tree cofactor of L0 on six random graphs, against
    Cayley's n^(n-2) on K4 through K19, and against the component product on four
    disconnected complexes (two and three triangles, triangle plus K4, triangle
    plus C4). Multigraphs are fine: a doubled relation reads 8 on the triangle.

    Outside that scope it is still the exact Gram determinant but it is NOT a tree
    count, and the earlier wording implied otherwise. Measured counterexamples: a
    disconnected pair of triangles reads 9 against 0 spanning trees, K4 with one
    face reads 432 against 16, and Matrix-Tree does not apply at all once a
    relation is 1-ary or branching (witness reads 3 against 2).

    It is where the harmonic readings get their denominators. A reading is
    `q^T G^-1 q` shaped, which is `q^T adj(G) q / det(G)`, so a coordinate's
    denominator divides det(G) and a closure entry's divides det(G) * ||p||^2,
    the extra factor being the size of the product's support. On K5 that is
    125 and 3, and the diagonal closure is 7/15.

    Float LU on the same matrix drifts (160 absolute at K16, 9.4e6 at K19),
    which is why this path is fraction-free.
    """

    from rexgraph.rational_trig import bareiss_determinant

    H = harmonic_frame(rex) if frame is None else frame
    if H.shape[1] == 0:
        return 1
    G = np.asarray(harmonic_metric(rex, frame=H).todense())
    d = bareiss_determinant(_exact_ints(G, "the harmonic frame Gram"))
    return int(d) if d.denominator == 1 else d


def complex_structure(A, *, tol=1e-12):
    """Read an antisymmetric operator as a complex structure.

    A real antisymmetric operator is a rotation generator: it has even rank, its
    nonzero spectrum is conjugate pairs on the imaginary axis, and on each 2-plane
    it spans it normalises to J with J^2 = -I. So it splits the space into complex
    lines, one per pair, and a real kernel it cannot reach.

    This is what to read instead of a norm. ``||A||`` is one number and throws away
    which directions rotate, how fast, and which stay real; the rates and the rank
    are what carry that, and they are what distinguish two structures that happen
    to have the same magnitude.

    Returns a dict with ``dim``, ``rank``, ``pairs``, ``real_dim`` and ``rates``,
    the per-plane rotation rates in decreasing order. Raises if `A` is not
    antisymmetric, since every statement here depends on it.
    """
    A = np.asarray(A, dtype=_f64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("complex_structure needs a square operator")
    n = A.shape[0]
    if n == 0:
        return {"dim": 0, "rank": 0, "pairs": 0, "real_dim": 0, "rates": []}
    scale = float(np.abs(A).max()) or 1.0
    if np.abs(A + A.T).max() > tol * scale:
        raise ValueError("complex_structure needs an ANTISYMMETRIC operator; "
                         "a commutator of two symmetric operators is one")
    sv = np.linalg.svd(A, compute_uv=False)
    keep = sv > tol * (sv[0] if sv.size else 1.0)
    rank = int(keep.sum())
    rates = [float(x) for x in sv[:rank:2]]      # singular values come in pairs
    return {"dim": n, "rank": rank, "pairs": rank // 2,
            "real_dim": n - rank, "rates": rates}


def structure_alignment(A, B):
    """Frobenius alignment of two antisymmetric operators, in [-1, 1].

    Both are rotation generators on the same space, so this asks whether they turn
    the same planes. Near zero means they are independent directions in the space
    of complex structures; near +-1 means one is essentially the other.

    Reading this instead of comparing ``||A||`` with ``||B||`` is the point: two
    generators of equal magnitude can be orthogonal or identical, and the magnitude
    cannot tell them apart.
    """
    A = np.asarray(A, dtype=_f64)
    B = np.asarray(B, dtype=_f64)
    na = float(np.linalg.norm(A, "fro"))
    nb = float(np.linalg.norm(B, "fro"))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.sum((A / na) * (B / nb)))
