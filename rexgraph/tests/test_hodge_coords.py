"""Coordinates for the three Hodge spaces.

`hodge` returns three vectors of length nE, and each is the image of something
smaller that the solver already computed and discarded: phi on the vertices, psi
on the faces, the harmonic coordinates on the frame. These read the small side.
"""

import numpy as np
import pytest

from rexgraph.core._sparse import matvec, rmatvec
from rexgraph.graph import RexGraph
from rexgraph.harmonic_sparse import harmonic_basis, harmonic_projection
from rexgraph.hodge_coords import (
    HodgeCoords,
    coordinate_dims,
    from_harmonic_coords,
    from_hodge_coords,
    harmonic_closure,
    harmonic_coords,
    harmonic_frame,
    harmonic_gram_det,
    harmonic_metric,
    harmonic_spread,
    harmonic_structure_constants,
    hodge_coords,
)


def _two_rings():
    """A 4-ring and a 3-ring, disjoint: two independent holes, no faces."""
    src = np.array([0, 1, 2, 3, 4, 5, 6], np.int32)
    tgt = np.array([1, 2, 3, 0, 5, 6, 4], np.int32)
    return RexGraph(sources=src, targets=tgt)


def _filled_square():
    """A square with a diagonal, both triangles filled: no holes, faces live."""
    r = RexGraph(sources=np.array([0, 1, 2, 3, 0], np.int32),
                 targets=np.array([1, 2, 3, 0, 2], np.int32))
    r.add_faces([[0, 1, 4], [2, 3, 4]], signs=[[1, 1, -1], [1, 1, 1]])
    r._ensure_clean()
    return r


def _flow(rex, seed=0):
    return np.random.default_rng(seed).normal(size=rex.nE)


#### the harmonic plane
def test_the_harmonic_frame_has_one_axis_per_independent_hole():
    r = _two_rings()
    assert harmonic_frame(r).shape == (r.nE, 2)
    assert r.betti[1] == 2


def test_harmonic_coordinates_are_the_projector_small_side():
    """The projector already solved for these and returned H c instead."""
    r = _two_rings()
    f = _flow(r)
    c = harmonic_coords(r, f)
    assert c.shape == (2,)
    assert np.allclose(from_harmonic_coords(r, c), harmonic_projection(
        harmonic_basis(r), f), atol=1e-10)


def test_the_harmonic_coordinates_agree_with_the_decomposition():
    r = _two_rings()
    f = _flow(r)
    assert np.allclose(from_harmonic_coords(r, harmonic_coords(r, f)),
                       r.hodge(f)[2], atol=1e-8)


def test_a_flow_already_on_one_axis_reads_as_that_axis():
    """A cycle is a harmonic vector, so its coordinates are the axis it is."""
    r = _two_rings()
    H = harmonic_frame(r)
    axis = np.asarray(H[:, 0].todense()).ravel()
    c = harmonic_coords(r, axis, frame=H)
    assert np.allclose(c, [1.0, 0.0], atol=1e-8)


def test_a_complex_with_no_holes_has_no_harmonic_coordinates():
    r = _filled_square()
    assert harmonic_coords(r, _flow(r)).shape == (0,)
    assert np.allclose(from_harmonic_coords(r, np.zeros(0)), 0.0)


#### all three at once
def test_each_coordinate_block_builds_its_component():
    """phi builds the gradient, psi the curl. That is what they are."""
    r = _filled_square()
    f = _flow(r, 1)
    c = hodge_coords(r, f)
    grad, curl, _ = r.hodge(f)
    assert np.allclose(np.asarray(rmatvec(r._B1_dual, c.phi)).ravel(), grad, atol=1e-7)
    assert np.allclose(np.asarray(matvec(r._B2_hodge_dual, c.psi)).ravel(), curl,
                       atol=1e-7)


@pytest.mark.parametrize("build", [_two_rings, _filled_square])
def test_the_coordinates_reconstruct_the_flow(build):
    """B1^T phi + B2 psi + H c is the flow it came from."""
    r = build()
    f = _flow(r, 2)
    assert np.allclose(from_hodge_coords(r, hodge_coords(r, f)), f, atol=1e-7)


def test_the_blocks_are_sized_by_their_own_space():
    r = _filled_square()
    c = hodge_coords(r, _flow(r))
    assert c.phi.shape == (r.nV,)
    assert c.psi.shape == (r._B2_hodge_dual.ncol,)
    assert c.harmonic.shape == (harmonic_frame(r).shape[1],)


def test_the_method_on_the_complex_is_the_module_function():
    r = _two_rings()
    f = _flow(r)
    a, b = r.hodge_coords(f), hodge_coords(r, f)
    assert isinstance(a, HodgeCoords)
    for x, y in zip(a, b, strict=True):
        assert np.allclose(x, y)


#### the dimension count
@pytest.mark.parametrize("build", [_two_rings, _filled_square])
def test_the_three_spaces_span_the_edge_space_exactly(build):
    """rank(B1) + rank(B2) + dim_H = nE. Nothing in the edge space is missed and
    nothing is counted twice."""
    d = coordinate_dims(build())
    assert d["independent"] == d["nE"]


def test_the_chart_carries_the_gauge_freedom_and_says_so():
    """phi is fixed up to a constant per component, so the chart is larger than
    the space. Two disjoint rings means two constants."""
    d = coordinate_dims(_two_rings())
    assert d["chart"] - d["independent"] == 2
    assert d["rank_B1"] == d["nV"] - 2


#### the potentials come from the decomposition, not a second solve
def test_the_kernel_returns_the_potentials_it_already_computed():
    from rexgraph.core import _hodge
    r = _filled_square()
    f = _flow(r, 3)
    r._ensure_clean()
    three = _hodge.hodge_decomposition(r._B1_dual, r._B2_hodge_dual, f)
    five = _hodge.hodge_decomposition(r._B1_dual, r._B2_hodge_dual, f,
                                      potentials=True)
    assert len(three) == 3 and len(five) == 5
    for a, b in zip(three, five[:3], strict=True):
        assert np.allclose(a, b)
    assert five[3].shape == (r.nV,)
    assert five[4].shape == (r._B2_hodge_dual.ncol,)


#### the plane has a metric, and it is not the identity
def test_the_frame_is_not_orthogonal_so_the_plane_carries_a_metric():
    """Axes are cycles and cycles share edges."""
    import itertools

    e = list(itertools.combinations(range(6), 2))
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    G = np.asarray(harmonic_metric(r).todense())
    assert G.shape == (10, 10)
    assert not np.allclose(G, np.eye(10)), "the frame would need to be orthonormal"


def test_the_spread_in_the_plane_is_the_spread_of_the_projections():
    """The whole claim: the angle can be read on dim_H coordinates instead of nE
    edges, and it is the same angle."""
    import itertools

    from rexgraph.rational_trig import spread
    e = list(itertools.combinations(range(6), 2))
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    H = harmonic_frame(r)
    rng = np.random.default_rng(0)
    for _ in range(25):
        u, v = rng.normal(size=r.nE), rng.normal(size=r.nE)
        ambient = spread(from_harmonic_coords(r, harmonic_coords(r, u, frame=H), frame=H),
                         from_harmonic_coords(r, harmonic_coords(r, v, frame=H), frame=H))
        assert np.isclose(harmonic_spread(r, u, v, frame=H), ambient, atol=1e-10)


def test_reading_the_coordinates_as_euclidean_is_wrong():
    """Guards the reason `harmonic_metric` exists. Without the form the angle is
    off by enough to reorder pairs."""
    import itertools

    from rexgraph.rational_trig import spread
    e = list(itertools.combinations(range(6), 2))
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    H = harmonic_frame(r)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        u, v = rng.normal(size=r.nE), rng.normal(size=r.nE)
        naive = spread(harmonic_coords(r, u, frame=H), harmonic_coords(r, v, frame=H))
        worst = max(worst, abs(harmonic_spread(r, u, v, frame=H) - naive))
    assert worst > 0.05, "the frame was orthogonal after all, so the test is void"


def test_a_flow_against_itself_has_no_spread():
    r = _two_rings()
    f = _flow(r)
    assert np.isclose(harmonic_spread(r, f, f), 0.0, atol=1e-12)


def test_a_complex_with_no_holes_has_no_angle_to_report():
    r = _filled_square()
    assert harmonic_spread(r, _flow(r, 1), _flow(r, 2)) == 0.0


#### the Hadamard product on the harmonic plane
def _complete(n):
    import itertools
    e = list(itertools.combinations(range(n), 2))
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    r._ensure_clean()
    return r


def test_closure_is_read_without_an_eigendecomposition_or_the_big_projector():
    """The dense path builds nE x nE L1 and runs eigh. This reads the same
    numbers off the small Gram."""
    from rexgraph.core._harmonic import harmonic_basis as dense_basis
    r = _complete(5)
    H = np.asarray(harmonic_frame(r).todense())
    U, _ = dense_basis(np.asarray(r.B1_dense, float), np.asarray(r.B2_dense, float))
    P = U @ U.T
    C = harmonic_closure(r)
    for i in range(H.shape[1]):
        for j in range(H.shape[1]):
            p = H[:, i] * H[:, j]
            if p @ p == 0:
                continue
            assert np.isclose(C[i, j], (p @ P @ p) / (p @ p), atol=1e-9)


def test_closure_is_exactly_rational_on_an_integer_frame():
    """Cycle axes are {0, +1, -1}, so H^T p is integer and only the Gram solve
    makes it a ratio."""
    from fractions import Fraction
    C = harmonic_closure(_complete(5), exact=True)
    assert all(isinstance(x, Fraction) for row in C for x in row)
    assert C[0][0] == Fraction(7, 15)


def test_the_complete_graph_closure_law():
    """Exact, and it held as a prediction at n = 8 and 9. The fundamental cycle
    basis of K_n is all triangles, so these belong to that basis."""
    from fractions import Fraction
    for n in (4, 5, 6, 7, 8):
        r = _complete(n)
        H = np.asarray(harmonic_frame(r).todense())
        assert {int((H[:, i] != 0).sum()) for i in range(H.shape[1])} == {3}
        C = harmonic_closure(r, exact=True)
        assert C[0][0] == Fraction(3 * n - 8, 3 * n), n
        off = next(C[0][j] for j in range(1, H.shape[1]) if C[0][j] != 0)
        assert off == Fraction(n - 2, n), n


def test_the_plane_is_not_closed_under_the_product():
    """Closure below 1 says the product leaves the plane. It approaches 1 with n
    and does not reach it."""
    C = harmonic_closure(_complete(6), exact=True)
    assert all(x <= 1 for row in C for x in row)
    assert any(0 < x < 1 for row in C for x in row)


def test_axes_that_share_no_edge_have_no_product_to_place():
    r = _complete(5)
    H = np.asarray(harmonic_frame(r).todense())
    C = harmonic_closure(r)
    for i in range(H.shape[1]):
        for j in range(H.shape[1]):
            if not np.any((H[:, i] != 0) & (H[:, j] != 0)):
                assert C[i, j] == 0.0


def test_structure_constants_place_the_product_in_coordinates():
    """closure is the length of what lands; the constants are where it lands."""
    r = _complete(5)
    H = harmonic_frame(r)
    c = harmonic_structure_constants(r, 0, 1)
    assert c.shape == (H.shape[1],)
    Hd = np.asarray(H.todense())
    p = Hd[:, 0] * Hd[:, 1]
    assert np.allclose(from_harmonic_coords(r, c),
                       harmonic_projection(harmonic_basis(r), p), atol=1e-9)


#### the Gram determinant counts spanning trees
def _spanning_trees(nV, pairs):
    """Matrix-Tree: any cofactor of L0, computed exactly and independently."""
    from fractions import Fraction

    from rexgraph.rational_trig import bareiss_determinant
    L = [[0] * nV for _ in range(nV)]
    for a, b in pairs:
        L[a][b] -= 1
        L[b][a] -= 1
    for i in range(nV):
        L[i][i] = -sum(L[i][j] for j in range(nV) if j != i)
    minor = [[Fraction(L[i][j]) for j in range(1, nV)] for i in range(1, nV)]
    return int(bareiss_determinant(minor))


def test_the_frame_gram_determinant_counts_spanning_trees():
    """With no faces the frame is the whole cycle space, and its Gram determinant
    is the spanning-tree count. Checked against the Matrix-Tree cofactor, which
    shares no code with the frame."""
    import itertools

    rng = np.random.default_rng(7)
    for nV, m in ((6, 10), (8, 14), (10, 18)):
        pairs = set()
        while len(pairs) < m:
            a, b = rng.integers(0, nV, 2)
            if a != b:
                pairs.add((min(a, b), max(a, b)))
        r = RexGraph(sources=np.array([a for a, b in pairs], np.int32),
                     targets=np.array([b for a, b in pairs], np.int32))
        r._ensure_clean()
        if r.betti[0] != 1:
            continue
        assert harmonic_gram_det(r) == _spanning_trees(nV, pairs), (nV, m)
    # and the complete graphs, where the count is Cayley's n^(n-2)
    for n in (4, 5, 6, 7, 8):
        e = list(itertools.combinations(range(n), 2))
        r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                     targets=np.array([b for a, b in e], np.int32))
        assert harmonic_gram_det(r) == n ** (n - 2), n


def test_the_determinant_is_where_the_denominators_come_from():
    """A reading is q^T adj(G) q / det(G), so a closure entry's denominator
    divides det(G) times the size of the product's support, not det(G) alone.
    On K5 the diagonal is 7/15 against det 125: the extra 3 is the triangle."""
    import itertools
    from fractions import Fraction

    n = 5
    e = list(itertools.combinations(range(n), 2))
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    det = harmonic_gram_det(r)
    H = np.asarray(harmonic_frame(r).todense())
    C = harmonic_closure(r, exact=True)
    assert C[0][0] == Fraction(7, 15) and det == 125
    for i in range(H.shape[1]):
        for j in range(H.shape[1]):
            x = C[i][j]
            assert isinstance(x, Fraction)
            p = H[:, i] * H[:, j]
            pp = int(round(float(p @ p)))
            if pp == 0:
                continue
            assert (det * pp) % x.denominator == 0, (i, j, x, det, pp)


def test_a_complex_with_no_holes_has_an_empty_gram():
    assert harmonic_gram_det(_filled_square()) == 1


#### the shape the frame makes
def test_the_determinant_is_a_squared_volume_and_the_frame_is_equiangular():
    """det(Gram) is the squared volume of the parallelepiped the axes span. On a
    complete graph every axis is a triangle, so the shape is rigid: one quadrance
    and two angles for the whole configuration."""
    from fractions import Fraction

    from rexgraph.rational_trig import quadrance, spread
    for n in (5, 6, 7):
        r = _complete(n)
        H = np.asarray(harmonic_frame(r).todense()).astype(int)
        k = H.shape[1]
        assert {quadrance(H[:, i].tolist(), exact=True) for i in range(k)} == {3}
        spreads = {spread(H[:, i].tolist(), H[:, j].tolist(), exact=True)
                   for i in range(k) for j in range(k) if i != j}
        # sharing one edge puts the dot at +-1 against quadrance 3 twice
        assert spreads <= {Fraction(8, 9), Fraction(1)}
        assert Fraction(8, 9) in spreads
        assert harmonic_gram_det(r) == n ** (n - 2)


def test_a_subset_determinant_counts_the_edge_sets_it_is_independent_on():
    """Cauchy-Binet against a unimodular cycle matrix: every maximal minor is 0 or
    +-1, so the determinant is a count, not only a volume."""
    import itertools

    from rexgraph.rational_trig import gram_determinant
    r = _complete(5)
    H = np.asarray(harmonic_frame(r).todense()).astype(int)
    for s in (1, 2, 3, 4):
        idx = list(range(s))
        det = int(gram_determinant([H[:, i].tolist() for i in idx], exact=True))
        counted = sum(1 for T in itertools.combinations(range(r.nE), s)
                      if round(np.linalg.det(H[np.ix_(list(T), idx)])) != 0)
        assert det == counted, s


def test_the_sign_forms_in_the_plane_are_exactly_the_cycles():
    """A harmonic class written out on edges is {s, s, ..., -s}. Those vectors are
    the {0,+1,-1} points of the plane, and there is one pair per cycle."""
    import itertools

    from rexgraph.rational_trig import quadrance
    r = _complete(4)
    H = np.asarray(harmonic_frame(r).todense()).astype(float)
    P = H @ np.linalg.solve(H.T @ H, H.T)
    forms = [np.array(v, float) for v in itertools.product([-1, 0, 1], repeat=r.nE)
             if any(v) and np.allclose(P @ np.array(v, float), np.array(v, float),
                                       atol=1e-9)]
    # K4 has 4 triangles and 3 quadrilaterals, each in two polarities
    assert len(forms) == 14
    supports = sorted({int((f != 0).sum()) for f in forms})
    assert supports == [3, 4]
    # support size is the quadrance, every entry being +-1
    for f in forms:
        assert quadrance(f.tolist(), exact=True) == int((f != 0).sum())
    # and their coordinates are integer too, so the forms are the lattice points
    for f in forms:
        c = harmonic_coords(r, f)
        assert np.allclose(c, np.round(c), atol=1e-9)


def test_a_rotation_is_g_orthogonal_in_coordinates_not_orthogonal():
    """The same motion is an ordinary rotation in the edge space. The frame is not
    orthonormal, so in coordinates the matrix has to carry the metric."""
    r = _complete(4)
    H = np.asarray(harmonic_frame(r).todense()).astype(float)
    G = np.asarray(harmonic_metric(r).todense())
    u = H[:, 0] / np.sqrt(H[:, 0] @ H[:, 0])
    w = H[:, 1] - (H[:, 1] @ u) * u
    w /= np.sqrt(w @ w)
    cs, sn = 0.8, 0.6                       # exact 3-4-5, spread 9/25

    def rot(x):
        a, b = x @ u, x @ w
        return x - a * u - b * w + (cs * a - sn * b) * u + (sn * a + cs * b) * w

    R = np.array([harmonic_coords(r, rot(H[:, i])) for i in range(H.shape[1])]).T
    assert not np.allclose(R.T @ R, np.eye(R.shape[0]))
    assert np.allclose(R.T @ G @ R, G)


def test_rotation_carries_a_sign_form_off_the_lattice():
    """Rotation is continuous and the sign forms are isolated, so the two are
    different structures. Quadrance survives; the {0,+1,-1} entries do not."""
    import itertools
    r = _complete(4)
    H = np.asarray(harmonic_frame(r).todense()).astype(float)
    P = H @ np.linalg.solve(H.T @ H, H.T)
    forms = [np.array(v, float) for v in itertools.product([-1, 0, 1], repeat=r.nE)
             if any(v) and np.allclose(P @ np.array(v, float), np.array(v, float),
                                       atol=1e-9)]
    u = H[:, 0] / np.sqrt(H[:, 0] @ H[:, 0])
    w = H[:, 1] - (H[:, 1] @ u) * u
    w /= np.sqrt(w @ w)
    x = forms[0]
    a, b = x @ u, x @ w
    y = x - a * u - b * w + (0.8 * a - 0.6 * b) * u + (0.6 * a + 0.8 * b) * w
    assert np.isclose(y @ y, x @ x), "a rotation preserves quadrance"
    assert not np.allclose(y, np.round(y), atol=1e-9), "and leaves the lattice"
    assert not any(np.allclose(y, f, atol=1e-9) for f in forms)


#### the lattice is a lattice in the crystallographic sense
def _lattice_shells(H, G, radius=2, upto=8):
    """How many lattice points at each quadrance."""
    import itertools
    from collections import Counter
    k = H.shape[1]
    out = Counter()
    for c in itertools.product(range(-radius, radius + 1), repeat=k):
        c = np.array(c)
        if not c.any():
            continue
        q = int(c @ G @ c)
        if q <= upto:
            out[q] += 1
    return out


def test_the_cycle_lattice_of_k4_is_the_body_centred_cubic_lattice():
    """Not a resemblance: an explicit unimodular change of basis. K4 is the
    tetrahedron graph, and its space of circulations is BCC."""
    import itertools
    r = _complete(4)
    H = np.asarray(harmonic_frame(r).todense()).astype(int)
    G = H.T @ H
    shells = _lattice_shells(H, G)
    assert shells[3] == 8 and shells[4] == 6      # 8 nearest, then 6
    assert harmonic_gram_det(r) == 16
    # BCC = {x in Z^3 : x = y = z mod 2}, basis (1,1,1), (1,-1,1), (1,1,-1)
    Bc = np.array([[1, 1, 1], [1, -1, 1], [1, 1, -1]])
    Gb = Bc @ Bc.T
    found = None
    for cols in itertools.product(itertools.product(range(-2, 3), repeat=3), repeat=3):
        U = np.array(cols).T
        if abs(round(np.linalg.det(U))) != 1:
            continue
        if np.array_equal(U.T @ G @ U, Gb):
            found = U
            break
    assert found is not None, "no unimodular U with U^T G U = Gb"


def test_the_lattice_automorphisms_are_the_graph_automorphisms():
    """The integer G-orthogonal maps. For K4 there are 48 of them, which is |S4|
    times the polarity and also the order of the octahedral point group. The
    complex does not carry a symmetry group and a geometry separately."""
    import itertools
    from math import factorial
    r = _complete(4)
    H = np.asarray(harmonic_frame(r).todense()).astype(int)
    G = H.T @ H
    auts = []
    for cols in itertools.product(itertools.product(range(-2, 3), repeat=3), repeat=3):
        U = np.array(cols).T
        if abs(round(np.linalg.det(U))) != 1:
            continue
        if np.array_equal(U.T @ G @ U, G):
            auts.append(U)
    assert len(auts) == 48 == 2 * factorial(4)
    dets = [round(np.linalg.det(U)) for U in auts]
    assert dets.count(1) == dets.count(-1) == 24


def test_the_lattice_keeps_every_ring_where_a_basis_must_drop_one():
    """Cubane, the textbook ambiguity in ring perception. The 3-cube has six square
    faces and beta_1 is five, so any cycle basis leaves one out and no rule says
    which. The lattice's minimal vectors keep all six."""
    import itertools
    V = list(itertools.product([0, 1], repeat=3))
    idx = {v: i for i, v in enumerate(V)}
    E = [(idx[a], idx[b]) for a in V for b in V
         if sum(x != y for x, y in zip(a, b, strict=False)) == 1 and idx[a] < idx[b]]
    r = RexGraph(sources=np.array([a for a, _ in E], np.int32),
                 targets=np.array([b for _, b in E], np.int32))
    r._ensure_clean()
    assert (r.nV, r.nE) == (8, 12)
    H = np.asarray(harmonic_frame(r).todense()).astype(int)
    assert H.shape[1] == 5, "a basis holds five"
    shells = _lattice_shells(H, H.T @ H, radius=2, upto=4)
    assert shells[4] == 12, "the lattice holds six, in two polarities"


def test_holonomy_is_a_z2_functional_on_the_lattice():
    """A sign per edge is a gauge field; its product around a lattice vector is the
    Wilson loop. Vertex re-signing is a gauge transformation and cannot move it;
    flipping one edge is physical and does."""
    import itertools
    e = list(itertools.combinations(range(4), 2))
    src = np.array([a for a, b in e], np.int32)
    tgt = np.array([b for a, b in e], np.int32)
    r = RexGraph(sources=src, targets=tgt)
    r._ensure_clean()
    H = np.asarray(harmonic_frame(r).todense()).astype(int)

    def holonomy(sig):
        return [int(np.prod([sig[i] for i in np.nonzero(H[:, c])[0]]))
                for c in range(H.shape[1])]

    sig = np.ones(r.nE, int)
    base = holonomy(sig)
    t = np.array([1, -1, -1, 1])                      # a vertex pattern
    gauged = np.array([t[src[i]] * t[tgt[i]] * sig[i] for i in range(r.nE)])
    assert (gauged != sig).sum() > 0, "the gauge actually moved edge signs"
    assert holonomy(gauged) == base, "a gauge transformation cannot move holonomy"
    flipped = sig.copy()
    flipped[0] = -1
    assert holonomy(flipped) != base, "flipping one edge is physical"


#### the graded tower stays integer, and C60 is where that shows
def test_the_boundary_tower_is_integer_at_every_grade_and_closes_exactly():
    """C60 as a solid: a real molecule with mixed-arity faces. The chain condition
    closes at exactly zero, not to a tolerance, because nothing left the integers."""
    from rexgraph.graded_boundary import (
        betti_numbers,
        build_graded_boundaries,
        truncated_icosahedron_3rex,
        verify_chain,
    )
    Bs = build_graded_boundaries(truncated_icosahedron_3rex())
    assert len(Bs) == 3
    for B in Bs:
        A = B.toarray()
        assert set(float(x) for x in A[A != 0]) <= {-1.0, 1.0}
    ok, residual = verify_chain(Bs)
    assert ok and residual == 0.0
    assert betti_numbers(Bs) == [1, 0, 0, 0]


def test_grade_two_carries_mixed_arity_which_a_simplicial_complex_cannot():
    """Twelve pentagons and twenty hexagons in one B2. Triangulating to fit a
    simplicial complex would invent cells the molecule does not have."""
    from rexgraph.graded_boundary import build_graded_boundaries, truncated_icosahedron_3rex
    B2 = build_graded_boundaries(truncated_icosahedron_3rex())[1].toarray()
    arities = np.abs(B2).sum(axis=0)
    assert sorted({int(a) for a in arities}) == [5, 6]
    assert int((arities == 5).sum()) == 12
    assert int((arities == 6).sum()) == 20


def test_the_intrinsic_angle_of_a_simple_graph_carries_no_valence():
    """Worth pinning so it is not over-read. Two 2-ary relations meeting at a
    vertex have spread 3/4 whatever the degree, because that is the shape of a
    k=2 column. Arity is where geometry enters, and it moves there."""
    from fractions import Fraction

    from rexgraph.geometry import relation_quadrance, relation_spread
    for deg in (2, 3, 4):
        r = RexGraph(sources=np.zeros(deg, np.int32),
                     targets=np.arange(1, deg + 1, dtype=np.int32))
        r._ensure_clean()
        assert relation_spread(r, 0, 1, exact=True) == Fraction(3, 4)
    # a branching relation puts 1/(k-1) in the column and the quadrance follows
    seen = set()
    for k in (2, 3, 4, 5):
        ptr = np.array([0, k, 2 * k], np.int64)
        idx = np.array(list(range(k)) + [0] + list(range(k, 2 * k - 1)), np.int64)
        r = RexGraph.from_hypergraph(ptr, idx)
        r._ensure_clean()
        assert relation_quadrance(r, 0, exact=True) == 1 + Fraction(1, k - 1)
        seen.add(relation_spread(r, 0, 1, exact=True))
    assert len(seen) > 1, "arity has to move the angle"


#### the frame stays integer once faces exist
def _faced(n, n_tris):
    import itertools
    e = list(itertools.combinations(range(n), 2))
    pos = {p: i for i, p in enumerate(e)}
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    tris = list(itertools.combinations(range(n), 3))[:n_tris]
    faces = [[pos[(min(a, b), max(a, b))]
              for a, b in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2]))] for t in tris]
    r.add_faces(faces, signs=[[1, 1, -1]] * len(faces))
    r._ensure_clean()
    return r


@pytest.mark.parametrize("n,tris", [(4, 1), (5, 3), (6, 5), (7, 6)])
def test_the_frame_is_integer_with_faces_present(n, tris):
    """`ker(B2^T C)` is the kernel of an integer matrix, so it has an integer basis.
    Taking it by dense SVD returned normalized float columns, and the frame built
    from them was not integer: every exact reading downstream (coordinates,
    closure, the Gram determinant) only held on face-free complexes."""
    r = _faced(n, tris)
    H = np.asarray(harmonic_frame(r).todense())
    assert H.shape[1] > 0, "fixture has no harmonic content to test"
    assert np.array_equal(H, np.round(H)), "the frame left the integers"


@pytest.mark.parametrize("n,tris", [(5, 3), (6, 5)])
def test_the_faced_frame_still_spans_ker_l1(n, tris):
    """Exactness must not have moved the space. Checked against the dense
    eigendecomposition, which is the oracle for this and not the path."""
    r = _faced(n, tris)
    H = np.asarray(harmonic_frame(r).todense())
    P = H @ np.linalg.solve(H.T @ H, H.T)
    B1 = np.asarray(r.B1_dense, float)
    B2 = np.asarray(r.B2_dense, float)
    w, V = np.linalg.eigh(B1.T @ B1 + B2 @ B2.T)
    U = V[:, w < 1e-9]
    assert np.allclose(P, U @ U.T, atol=1e-8)


def test_the_exact_kernel_declines_rather_than_overflowing():
    """Clearing denominators can push a coordinate past what a float64 holds
    exactly. A dense random integer matrix does it at 40x80. The routine returns
    None there so the caller takes the float path, rather than wrapping to a wrong
    integer."""
    import scipy.sparse as sp

    from rexgraph.harmonic_sparse import _integer_nullspace
    rng = np.random.default_rng(0)
    A = rng.integers(-1, 2, size=(40, 80))
    assert _integer_nullspace(sp.csr_matrix(A.astype(float))) is None
    # and a structured one does not
    small = _integer_nullspace(sp.csr_matrix(np.array([[1., -1., 0.], [0., 1., -1.]])))
    assert small is not None and np.array_equal(small, np.round(small))
