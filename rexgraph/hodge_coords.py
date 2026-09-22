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

Redundancy is exact and worth stating. phi is fixed up to ker(B1^T) and psi
up to ker(B2), so the chart carries beta_0 +
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


def harmonic_frame(rex, *, native=False):
    """The harmonic plane's axes as a sparse nE x dim_H matrix.

    One column per independent hole, each a cycle carrying no face flux. This is
    `harmonic_basis`, named for what it is used as here.
    """
    return harmonic_basis(rex, native=native)


def _exact_flow(flow, nE):
    """The flow as exact rationals, a float coordinate read at its exact binary value."""
    from rexgraph.exact_green import exact_field
    from rexgraph.harmonic_sparse import as_edge_signal
    if isinstance(flow, (list, tuple)) or (isinstance(flow, np.ndarray) and flow.dtype == object):
        values = np.asarray(flow, dtype=object).ravel()
        if values.shape[0] != nE:
            raise ValueError(f"flow has {values.shape[0]} entries; the complex has {nE} relations")
        return exact_field(values.tolist())
    return exact_field(as_edge_signal(flow, nE, what="flow").tolist())


def _exact_frame_or_none(rex, frame, exact):
    """The integer frame as an exact matrix when the exact path answers, else None.

    exact=None takes the exact path within `exact_field_limit` and when the frame is
    integral; exact=True requires both of it by refusing a frame that is not integral.
    """
    from rexgraph.exact_green import exact_path_available
    if exact is False or (exact is None and not exact_path_available(rex, 1)):
        return None
    H = harmonic_frame(rex, native=True) if frame is None else frame
    try:
        return _exact_frame(H)
    except ValueError:
        if exact:
            raise
        return None


def harmonic_coords(rex, flow, *, frame=None, exact=None):
    """Where `flow` sits on the harmonic frame: its dim_H coordinates c.

    c solves G c = H^T flow on the frame Gram G = H^T H. The frame is integer, so
    H^T flow is exact for an exact flow and the only division is the Gram solve,
    whose denominators divide det(G). `H @ c` is the harmonic projection, so this is
    the projector's small side. G is the Euclidean form; see `harmonic_metric`.

    exact=None (default) solves over Q when the complex is within
    `configure_algorithms(exact_field_limit=...)` and the frame is integral, and
    returns f64[dim_H] values of that exact answer; otherwise the float least
    squares solve on H answers. exact=True returns the Fractions and refuses a frame
    that is not integral. exact=False asks for the float solve. A float coordinate
    of the flow is read at its exact binary value, as `RexGraph.hodge` reads one.
    """
    H = _exact_frame_or_none(rex, frame, exact)
    if H is not None:
        if H.ncols == 0:
            return [] if exact else np.zeros(0, dtype=_f64)
        c = _frame_gram(H).solve(H.T.apply(tuple(_exact_flow(flow, H.nrows))))
        return list(c) if exact else np.asarray([float(v) for v in c], dtype=_f64)
    H = harmonic_frame(rex, native=True) if frame is None else frame
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


def harmonic_spread(rex, u, v, *, frame=None, exact=None):
    """Spread between two flows' harmonic parts, computed in the plane.

    Spread is sin^2 of the angle, so it stays rational and needs no square root.
    With a = G^-1 H^T u and b = G^-1 H^T v it is 1 - (a.H^T v)^2 / ((a.H^T u)(b.H^T v)),
    since G a = H^T u; no product with G is formed. This is the spread of the ambient
    harmonic projections, at dim_H terms rather than nE.

    exact follows `harmonic_coords`: exact=None returns the float value of the exact
    spread when the exact path answers, exact=True returns the Fraction, and
    exact=False reads it through the float coordinates and `harmonic_metric`.

    Returns 0 when either flow has no harmonic part, and when the complex has no
    holes at all, since there is then no angle to speak of.
    """
    from fractions import Fraction
    Hx = _exact_frame_or_none(rex, frame, exact)
    if Hx is not None:
        if Hx.ncols == 0:
            return Fraction(0) if exact else 0.0
        G = _frame_gram(Hx)
        hu = Hx.T.apply(tuple(_exact_flow(u, Hx.nrows)))
        hv = Hx.T.apply(tuple(_exact_flow(v, Hx.nrows)))
        a, b = G.solve(hu), G.solve(hv)
        qa = sum((x * y for x, y in zip(a, hu, strict=True)), Fraction(0))
        qb = sum((x * y for x, y in zip(b, hv, strict=True)), Fraction(0))
        ab = sum((x * y for x, y in zip(a, hv, strict=True)), Fraction(0))
        result = Fraction(0) if qa == 0 or qb == 0 else 1 - ab * ab / (qa * qb)
        return result if exact else float(result)
    H = harmonic_frame(rex) if frame is None else frame
    if H.shape[1] == 0:
        return 0.0
    G = harmonic_metric(rex, frame=H)
    a = harmonic_coords(rex, u, frame=H, exact=False)
    b = harmonic_coords(rex, v, frame=H, exact=False)
    qa = float(a @ (G @ a))
    qb = float(b @ (G @ b))
    if qa <= 0.0 or qb <= 0.0:
        return 0.0
    ab = float(a @ (G @ b))
    return 1.0 - (ab * ab) / (qa * qb)


def from_harmonic_coords(rex, c, *, frame=None, exact=False):
    """The edge signal a set of harmonic coordinates names: `H c`, f64[nE].

    exact=True returns `H c` as Fractions, computed from the integer frame and the
    coordinates as given, which is how exact coordinates come back to the edge space.
    """
    if exact:
        from fractions import Fraction
        H = _exact_frame_or_none(rex, frame, True)
        values = [v if isinstance(v, Fraction) else Fraction(v) for v in np.asarray(c, dtype=object).ravel()]
        if len(values) != H.ncols:
            raise ValueError(f"{len(values)} coordinates for a frame with {H.ncols} axes")
        return list(H.apply(tuple(values))) if H.ncols else [Fraction(0)] * H.nrows
    from rexgraph.native_sparse import as_native

    H = harmonic_frame(rex, native=True) if frame is None else frame
    c = np.atleast_1d(np.asarray(c, dtype=_f64).ravel())
    if H.shape[1] == 0:
        return np.zeros(H.shape[0], dtype=_f64)
    return as_native(H).apply(c)


def hodge_coords(rex, flow, *, frame=None):
    """`flow` in all three Hodge spaces at once: HodgeCoords(phi, psi, harmonic).

    phi and psi come from the float least squares decomposition, which takes no
    metric, so they answer the unweighted question. The harmonic coordinates are a
    separate solve on the frame's Gram, which is small, and follow the default
    policy of `harmonic_coords`: exact within `exact_field_limit`.
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
    # rank(B1) = nV - b0 is exact: b0 is the dimension of ker(B1^T).
    # It counts support components only in the pairwise case. rank(B2) follows
    # from the Hodge dimension identity
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
    """Preserve supplied integers and reject nonintegral frame coefficients."""
    from fractions import Fraction
    from numbers import Integral, Real
    from math import isfinite

    arr = np.asarray(values, dtype=object)
    converted = []
    for value in arr.flat:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{what} requires integer coefficients")
        if isinstance(value, Integral):
            result = Fraction(int(value))
        elif isinstance(value, Fraction) and value.denominator == 1:
            result = value
        elif isinstance(value, Real) and isfinite(value) and abs(value) <= 2**53 and value == int(value):
            result = Fraction(int(value))
        else:
            raise ValueError(f"{what} is not integral or has no certified integer representation")
        converted.append(result)
    result = np.asarray(converted, dtype=object).reshape(arr.shape)
    return result.item() if arr.ndim == 0 else result.tolist()


def _exact_frame(frame):
    """Read integer frame entries before any product can round or overflow.

    A native frame is read from its stored columns, so the exact path does not need SciPy.
    """
    from rexgraph.exact_green import ExactSparse
    from rexgraph.native_sparse import NativeSparse
    if isinstance(frame, NativeSparse):
        dual = frame.dual
        ptr, rows = np.asarray(dual.col_ptr), np.asarray(dual.row_idx)
        coefficients = _exact_ints(np.asarray(dual.vals_csc), "the harmonic frame")
        entries = {}
        for j in range(frame.shape[1]):
            for k in range(int(ptr[j]), int(ptr[j + 1])):
                key = int(rows[k]), j
                entries[key] = entries.get(key, 0) + coefficients[k]
        return ExactSparse(*frame.shape, entries)
    if hasattr(frame, "tocoo"):
        coo = frame.tocoo()
        coefficients = _exact_ints(coo.data, "the harmonic frame")
        entries = {}
        for i, j, value in zip(coo.row, coo.col, coefficients, strict=True):
            key = int(i), int(j)
            entries[key] = entries.get(key, 0) + value
        return ExactSparse(*frame.shape, entries)
    original = np.asarray(frame, dtype=object)
    values = np.asarray(_exact_ints(frame, "the harmonic frame"), dtype=object).reshape(original.shape)
    if values.ndim != 2:
        raise ValueError("the harmonic frame must have two axes")
    return ExactSparse(*values.shape, {index: v for index, v in np.ndenumerate(values) if v})


def _frame_gram(frame):
    from rexgraph.exact_green import ExactSparse
    rows = {}
    for (i, j), value in frame.entries.items():
        rows.setdefault(i, []).append((j, value))
    entries = {}
    for row in rows.values():
        for i, left in row:
            for j, right in row:
                key = i, j
                entries[key] = entries.get(key, 0) + left * right
    return ExactSparse(frame.ncols, frame.ncols, entries)


def _solve_gram(G, rhs, exact):
    """Apply a Gram solve under the selected arithmetic contract."""
    if not exact:
        import scipy.sparse.linalg as sla
        return sla.spsolve(G.tocsc(), rhs)
    matrix = _exact_frame(G)
    values = np.asarray(_exact_ints(rhs, "the harmonic Gram right hand side"), dtype=object).reshape(-1)
    return list(matrix.solve(values.tolist()))


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
    if exact:
        from fractions import Fraction
        H = _exact_frame(harmonic_frame(rex) if frame is None else frame)
        if not 0 <= i < H.ncols or not 0 <= j < H.ncols:
            raise IndexError("harmonic frame column is outside the declared space")
        product = tuple(H.entries.get((row, i), Fraction(0)) * H.entries.get((row, j), Fraction(0))
                        for row in range(H.nrows))
        return list(_frame_gram(H).solve(H.T.apply(product)))

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
    if exact:
        from fractions import Fraction
        H = _exact_frame(harmonic_frame(rex) if frame is None else frame)
        G = _frame_gram(H)
        result = []
        for i in range(H.ncols):
            row_values = []
            for j in range(H.ncols):
                product = tuple(H.entries.get((row, i), Fraction(0)) * H.entries.get((row, j), Fraction(0))
                                for row in range(H.nrows))
                norm = sum((v*v for v in product), Fraction(0))
                if not norm:
                    row_values.append(Fraction(0))
                    continue
                q = H.T.apply(product)
                x = G.solve(q)
                row_values.append(sum((a*b for a, b in zip(q, x, strict=True)), Fraction(0)) / norm)
            result.append(row_values)
        return result if H.ncols else np.zeros((0, 0), dtype=object)

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
    """Exact determinant of the frame Gram, by fraction free elimination.

    With no faces and every relation 2 ary, the frame is the full cycle space and
    this is the number of spanning FORESTS: the product over connected components
    of each component's spanning tree count. That reduces to the spanning trees of
    the graph when it is connected, which is the case the claim was first read on.
    Verified against the Matrix Tree cofactor of L0 on six random graphs, against
    Cayley's n^(n-2) on K4 through K19, and against the component product on four
    disconnected complexes (two and three triangles, triangle plus K4, triangle
    plus C4). Multigraphs are fine: a doubled relation reads 8 on the triangle.

    Outside that scope it is still the exact Gram determinant but it is NOT a tree
    count, and the earlier wording implied otherwise. Measured counterexamples: a
    disconnected pair of triangles reads 9 against 0 spanning trees, K4 with one
    face reads 432 against 16, and Matrix Tree does not apply at all once a
    relation is 1 ary or branching (witness reads 3 against 2).

    It is where the harmonic readings get their denominators. A reading is
    `q^T G^-1 q` shaped, which is `q^T adj(G) q / det(G)`, so a coordinate's
    denominator divides det(G) and a closure entry's divides det(G) * ||p||^2,
    the extra factor being the size of the product's support. On K5 that is
    125 and 3, and the diagonal closure is 7/15.

    Float LU on the same matrix drifts (160 absolute at K16, 9.4e6 at K19),
    which is why this path is fraction free.
    """

    from rexgraph.rational_trig import bareiss_determinant

    H = harmonic_frame(rex) if frame is None else frame
    if H.shape[1] == 0:
        return 1
    G = _frame_gram(_exact_frame(H))
    rows = [[G.entries.get((i, j), 0) for j in range(G.ncols)] for i in range(G.nrows)]
    d = bareiss_determinant(rows)
    return int(d) if d.denominator == 1 else d


def complex_structure(A, *, tol=1e-12):
    """Read an antisymmetric operator as a complex structure.

    A real antisymmetric operator is a rotation generator: it has even rank, its
    nonzero spectrum is conjugate pairs on the imaginary axis, and on each 2 plane
    it spans it normalises to J with J^2 = -I. So it splits the space into complex
    lines, one per pair, and a real kernel it cannot reach.

    This is what to read instead of a norm. ``||A||`` is one number and throws away
    which directions rotate, how fast, and which stay real; the rates and the rank
    are what carry that, and they are what distinguish two structures that happen
    to have the same magnitude.

    Returns a dict with ``dim``, ``rank``, ``pairs``, ``real_dim`` and ``rates``,
    the per plane rotation rates in decreasing order. Raises if `A` is not
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
