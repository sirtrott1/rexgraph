"""One integer sequence fixes the whole tower, and it is additive where the character is not.

`L_k = B_k^T B_k + B_{k+1} B_{k+1}^T` and `tr(X^T X) = ||X||_F^2`, so
`tr(L_k) = ||B_k||^2 + ||B_{k+1}||^2`. That makes every trace and every moment a
consequence of the mass sequence alone. It is an identity, so these check it rather than
sample it.

The mass is EXTENSIVE (a sum over stored entries, additive over disjoint components)
where the trace-normalised character is not. That difference is the whole reason to
prefer it: it carries the structural content without the global coupling.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.faces import auto_hyperface, autoface
from rexgraph.graded_boundary import (
    solid_octahedron_3rex,
    square_pyramid_3rex,
    truncated_icosahedron_3rex,
)
from rexgraph.graph import RexGraph
from rexgraph.tower import (
    boundary_mass,
    closure_at,
    incidence_degrees,
    mass_tower,
    moments,
    tower_law,
    trace_tower,
)


def _path():
    return RexGraph(sources=np.array([0, 1], dtype=np.int32),
                    targets=np.array([1, 2], dtype=np.int32))


def _filled(k: int):
    g = RexGraph(sources=np.arange(k, dtype=np.int32),
                 targets=np.roll(np.arange(k, dtype=np.int32), -1))
    autoface(g, k)
    return g


def _branching():
    g = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10, 12], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32))
    auto_hyperface(g)
    return g


#### the law


@pytest.mark.parametrize("build", [_path, lambda: _filled(3), lambda: _filled(6),
                                   _branching],
                         ids=["path", "triangle", "C6", "branching"])
def test_the_tower_law_holds(build):
    out = tower_law(build())
    assert out["holds"], f"residual {out['residual']}"
    assert out["residual"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("cells,name", [
    (solid_octahedron_3rex(), "octahedron"),
    (square_pyramid_3rex(), "square pyramid"),
    (truncated_icosahedron_3rex(), "truncated icosahedron"),
], ids=["octahedron", "pyramid", "trunc-icosa"])
def test_the_law_holds_at_grade_three(cells, name):
    """Genuine 3-complexes, where there is a B_3 for the law to reach."""
    import scipy.sparse as sp

    from rexgraph.graded_boundary import build_graded_boundaries, graded_laplacians
    B = build_graded_boundaries(cells)
    fro = [float(sp.csr_matrix(b).multiply(sp.csr_matrix(b)).sum()) for b in B]
    tr = [float(sp.csr_matrix(L).diagonal().sum()) for L in graded_laplacians(B)]
    for k, t in enumerate(tr):
        down = fro[k - 1] if k >= 1 else 0.0
        up = fro[k] if k < len(fro) else 0.0
        assert t == pytest.approx(down + up, abs=1e-9), f"{name} grade {k}"


def test_the_moments_follow_from_the_mass():
    """d tr(0->1) = ||B_2||^2 exactly, which is why a face-free complex reads zero."""
    g = _filled(3)
    assert moments(g)[0] == pytest.approx(float(boundary_mass(g, 2)), abs=1e-9)
    assert moments(_path())[0] == pytest.approx(0.0, abs=1e-12)


#### the mass is extensive


def test_the_mass_is_additive_over_disjoint_components():
    """The property the normalised character does not have, and the reason to use this."""
    a = RexGraph(sources=np.array([0, 1], dtype=np.int32),
                 targets=np.array([1, 2], dtype=np.int32))
    b = RexGraph(sources=np.array([0, 0, 0], dtype=np.int32),
                 targets=np.array([1, 2, 3], dtype=np.int32))
    union = RexGraph(sources=np.array([0, 1, 3, 3, 3], dtype=np.int32),
                     targets=np.array([1, 2, 4, 5, 6], dtype=np.int32))
    assert boundary_mass(a, 1) + boundary_mass(b, 1) == boundary_mass(union, 1)


def test_the_gradient_mass_is_the_total_quadrance():
    """`||B_1||^2 = sum_e Q(e)`, so the tower and the rendering geometry are one sum."""
    from rexgraph.geometry import relation_quadrance
    g = _branching()
    total = sum((relation_quadrance(g, e) for e in range(int(g.nE))), Fraction(0))
    assert boundary_mass(g, 1) == total


def test_the_mass_is_exact_at_branching_arity():
    """A 4-ary relation contributes Q = 4/3, so the mass is rational and stays so."""
    g = _branching()
    m = boundary_mass(g, 1)
    assert isinstance(m, Fraction)
    assert m == Fraction(28, 3)                  # one 4-ary at 4/3 plus four pairwise at 2


def test_grades_count_from_one():
    with pytest.raises(ValueError, match="grades count from 1"):
        boundary_mass(_path(), 0)


#### closure


@pytest.mark.parametrize("cells", [solid_octahedron_3rex(), square_pyramid_3rex(),
                                   truncated_icosahedron_3rex()],
                         ids=["octahedron", "pyramid", "trunc-icosa"])
def test_a_closed_solid_has_every_edge_in_two_faces(cells):
    import scipy.sparse as sp

    from rexgraph.graded_boundary import build_graded_boundaries
    B = build_graded_boundaries(cells)
    deg = np.asarray((abs(sp.csr_matrix(B[1])) > 0).sum(axis=1)).ravel()
    assert (deg == 2).all()


def test_an_open_complex_is_not_closed():
    g = _filled(3)
    c = closure_at(g, 2)
    assert c["closed"] is False
    assert c["every_two"] is False


def test_mass_equality_is_necessary_but_not_sufficient():
    """Recorded because it is the trap: the equality is a statement about the MEAN
    incidence degree being two. Degrees (1, 2, 2, 3) satisfy it and are not a closure."""
    import scipy.sparse as sp

    from rexgraph.graded_boundary import build_graded_boundaries
    cells = [4, [[0, 1], [1, 2], [2, 0], [0, 3]],
             [[(0, 1), (1, 1), (3, 1)], [(1, 1), (2, 1), (3, 1)], [(2, 1), (3, 1)]]]
    B = build_graded_boundaries(cells)
    fro = [float(sp.csr_matrix(b).multiply(sp.csr_matrix(b)).sum()) for b in B]
    deg = np.asarray((abs(sp.csr_matrix(B[1])) > 0).sum(axis=1)).ravel()
    assert fro[0] == fro[1], "the adversary should satisfy the cheap test"
    assert not (deg == 2).all(), "and fail the real one"


def test_closure_reports_the_degree_census():
    g = _filled(3)
    assert closure_at(g, 2)["degrees"] == {"1": 3}


def test_incidence_degrees_are_per_lower_cell():
    g = _filled(3)
    assert incidence_degrees(g, 2).shape[0] == g.nE


#### shape


def test_the_towers_line_up():
    g = _filled(3)
    assert len(trace_tower(g)) == len(mass_tower(g)) + 1
    assert len(moments(g)) == len(trace_tower(g)) - 1


#### latent, filled, closed: three states, not two


def _tetra(n_faces: int):
    e = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    f = [[(0, 1), (3, 1), (1, -1)], [(0, 1), (4, 1), (2, -1)],
         [(1, 1), (5, 1), (2, -1)], [(3, 1), (5, 1), (4, -1)]][:n_faces]
    g = RexGraph(sources=np.array([x[0] for x in e], dtype=np.int32),
                 targets=np.array([x[1] for x in e], dtype=np.int32))
    if f:
        g.add_faces([[i for i, _ in col] for col in f],
                    [[s for _, s in col] for col in f])
    return g


def test_an_unfilled_complex_is_latent():
    from rexgraph.tower import manifold_state
    st = manifold_state(_tetra(0))
    assert st["state"] == "latent"
    assert st["curl"] == 0 and st["harmonic"] == st["cycles"] == 3


def test_a_fully_filled_complex_can_still_be_open():
    """The separating case, and the reason homology cannot answer this. Three faces
    span the tetrahedron's three cycles, so nothing is harmonic and there is no hole
    anywhere, yet every edge does not lie in two faces."""
    from rexgraph.tower import manifold_state
    st = manifold_state(_tetra(3))
    assert st["harmonic"] == 0, "no holes remain"
    assert st["closed"] is False, "and it is still not a closed surface"
    assert st["state"] == "filled"


def test_the_closed_case_is_reached_by_the_redundant_face():
    """The fourth face adds no rank: homologically redundant, geometrically
    necessary."""
    from rexgraph.tower import manifold_state
    three, four = manifold_state(_tetra(3)), manifold_state(_tetra(4))
    assert three["curl"] == four["curl"] == 3, "the fourth face adds no rank"
    assert four["state"] == "closed"


def test_filling_never_changes_the_cycle_count():
    """ker(B_1) is fixed by the 1-skeleton; filling moves cycles from harmonic to
    curl, which is what harmonic_shadow counts."""
    from rexgraph.tower import manifold_state
    counts = {manifold_state(_tetra(n))["cycles"] for n in (0, 1, 2, 3, 4)}
    assert counts == {3}


def test_an_acyclic_complex_says_so():
    from rexgraph.tower import manifold_state
    g = RexGraph(sources=np.array([0, 1], dtype=np.int32),
                 targets=np.array([1, 2], dtype=np.int32))
    assert manifold_state(g)["state"] == "acyclic"


def test_mass_equality_is_not_sufficient_even_when_the_chain_condition_holds():
    """The question this closes. K5 with every triangle filled has mean edge-face
    degree 3; a five-edge path has degree 0. Their disjoint union is chain-valid and
    satisfies the mass equality, and nothing in it has degree two."""
    import itertools

    import scipy.sparse as sp

    from rexgraph.graded_boundary import build_graded_boundaries, verify_chain
    edges = [list(p) for p in itertools.combinations(range(5), 2)]
    index = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    faces = []
    for a, b, c in itertools.combinations(range(5), 3):
        col = []
        for x, y in ((a, b), (b, c), (c, a)):
            i = index[tuple(sorted((x, y)))]
            col.append((i, 1.0 if (x, y) == tuple(sorted((x, y))) else -1.0))
        faces.append(col)
    edges += [[5 + i, 6 + i] for i in range(5)]
    B = build_graded_boundaries([11, edges, faces])
    ok, _res = verify_chain(B)
    fro = [float(sp.csr_matrix(b).multiply(sp.csr_matrix(b)).sum()) for b in B]
    deg = np.asarray((abs(sp.csr_matrix(B[1])) > 0).sum(axis=1)).ravel()
    assert ok, "the counterexample must itself be a valid complex"
    assert fro[0] == fro[1], "mass equality holds"
    assert not (deg == 2).all(), "and it is not a closure"


#### the surface identity: 2/d + 2/k = 1 + chi/E


@pytest.mark.parametrize("V,E,F,name", [
    (4, 6, 4, "tetrahedron"), (6, 12, 8, "octahedron"), (8, 12, 6, "cube"),
    (12, 30, 20, "icosahedron"), (20, 30, 12, "dodecahedron"),
    (60, 90, 32, "truncated icosahedron"), (5, 8, 5, "square pyramid"),
    (16, 32, 16, "torus, quadrangulated"),
], ids=lambda x: str(x))
def test_the_surface_identity_is_exact(V, E, F, name):
    """Euler divided through by E, so it is an equality over the rationals rather than
    an approximation. Holds at any genus: the torus reads chi = 0 and the identity
    becomes exactly 1."""
    chi = V - E + F
    d, k = Fraction(2 * E, V), Fraction(2 * E, F)
    assert Fraction(2) / d + Fraction(2) / k == Fraction(1) + Fraction(chi, E)


@pytest.mark.parametrize("k,d", [(3, 6), (4, 4), (6, 3)])
def test_the_continuum_limit_is_the_three_regular_tilings(k, d):
    """chi/E -> 0 leaves 2/d + 2/k = 1, whose integer solutions are exactly the three
    tilings of the plane. That is the topological ideal a refinement approaches, and
    the k-gon structure decides which one."""
    assert Fraction(2, k) + Fraction(2, d) == 1


def test_homology_cannot_separate_what_the_pair_does():
    """The point of the identity. Tetrahedron and octahedron are both chi = 2, so no
    Betti reading distinguishes them; (k, d) is (3,3) against (3,4)."""
    tet = (4, 6, 4)
    octa = (6, 12, 8)
    assert tet[0] - tet[1] + tet[2] == octa[0] - octa[1] + octa[2] == 2
    def pair(V, E, F):
        return Fraction(2 * E, F), Fraction(2 * E, V)
    assert pair(*tet) != pair(*octa)


def test_the_deviation_is_exactly_chi_over_E():
    """Not an error term: at every finite stage the gap from the ideal IS chi/E."""
    for V, E, F in [(4, 6, 4), (6, 12, 8), (12, 30, 20)]:
        chi = V - E + F
        d, k = Fraction(2 * E, V), Fraction(2 * E, F)
        assert (Fraction(2) / d + Fraction(2) / k) - 1 == Fraction(chi, E)


def test_a_triangulated_sphere_has_degree_deficit_twelve_over_V():
    """6 - d = 6 chi / V, so on a sphere the deficit is exactly 12/V at every level of
    refinement. Measured on the subdivision sequence 4, 8, 20, 56."""
    for V, E in [(4, 6), (8, 18), (20, 54), (56, 162)]:
        assert 6 - Fraction(2 * E, V) == Fraction(12, V)


def test_the_identity_reports_scope():
    from rexgraph.tower import surface_identity
    out = surface_identity(_tetra(4))
    assert out["applicable"] is True and out["holds"] is True
    open_one = surface_identity(_tetra(3))
    assert open_one["applicable"] is False, "an open complex is out of scope"


#### the time/space Lagrangian pair


def test_the_cr_violation_cannot_see_a_face():
    """Forced, not incidental: the RL_4 channels are strictly 1-skeleton, so the
    Cauchy-Riemann violation is identical at every stage of filling. It separates
    1-skeletons, not manifold states."""
    readings = {_tetra(n).cr_violation() for n in (0, 1, 2, 3, 4)}
    assert len(readings) == 1, f"filling moved it: {readings}"


def test_the_cr_violation_separates_different_one_skeletons():
    c4 = RexGraph(sources=np.array([0, 1, 2, 3], dtype=np.int32),
                  targets=np.array([1, 2, 3, 0], dtype=np.int32))
    assert _tetra(0).cr_violation() != c4.cr_violation()


@pytest.mark.parametrize("build,name", [
    (lambda: RexGraph(sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
                      targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)), "tetra"),
    (lambda: RexGraph(sources=np.array([0, 1, 2, 3], dtype=np.int32),
                      targets=np.array([1, 2, 3, 0], dtype=np.int32)), "C4"),
    (lambda: RexGraph(sources=np.array([0, 1, 2, 3], dtype=np.int32),
                      targets=np.array([1, 2, 3, 4], dtype=np.int32)), "path"),
    (lambda: RexGraph(sources=np.array([0, 0, 1, 1, 2, 5], dtype=np.int32),
                      targets=np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)), "tree"),
], ids=["tetra", "C4", "path", "tree"])
def test_mean_c_squared_is_three_on_a_pairwise_complex(build, name):
    """A measured pattern over pairwise complexes, not a proof. Per-relation c2 ranges
    over 2 to 4 while the mean sits exactly on 3."""
    lf = build().lagrangian_fields()
    assert float(np.mean(lf["c2"])) == pytest.approx(3.0, abs=1e-9)


def test_branching_moves_the_mean_off_three():
    """Which is what makes the deviation an arity signature rather than noise."""
    g = RexGraph.from_hypergraph(np.array([0, 4, 6, 8], dtype=np.int32),
                                 np.array([0, 1, 2, 3, 0, 1, 1, 2], dtype=np.int32))
    lf = g.lagrangian_fields()
    assert float(np.mean(lf["c2"])) != pytest.approx(3.0, abs=1e-6)


def test_zero_frustration_is_read_rather_than_returning_nothing():
    """A consistently oriented star has zero frustration. This used to drop the F
    channel, leaving three, and these accessors returned None rather than risk the
    remaining three being read as if they were T,G,F,C.

    F is carried at zero now, so the position of every channel is fixed and that
    misreading cannot happen. The accessors answer instead of declining, which is
    the point: orientation conflict measuring zero is a measurement about the
    complex, not a reason to stop reporting."""
    star = RexGraph(sources=np.array([0, 0, 0], dtype=np.int32),
                    targets=np.array([1, 2, 3], dtype=np.int32))
    fields = star.lagrangian_fields()
    assert fields is not None
    assert list(fields["channels"]) == ["L1_down", "L_O", "L_SG", "L_C"]
    # L_t is the T channel's hat DIAGONAL and L_s the other three summed, so the pair
    # adds to RL[e,e] and not to 1. It is 1 on this fixture only because a 3-edge star
    # normalises that way; a 4-edge star reads 0.75.
    rl_diag = np.diagonal(np.asarray(star.RL))
    assert np.allclose(fields["Lt"] + fields["Ls"], rl_diag)
    # F contributes exactly nothing to L_s here, which is the whole point of the fixture
    hats = [np.asarray(h) for h in star._rcf_bundle["hats"]]
    assert np.allclose(np.diagonal(hats[2]), 0.0)
    assert star.cr_violation() is not None


#### the arity- and degree-general form


@pytest.mark.parametrize("cells,name", [
    ([3, [[0, 1], [1, 2], [2, 0]], [[(0, 1), (1, 1), (2, 1)]]], "triangle, c=1"),
    ([4, [[0, 1, 2, 3], [0, 1], [1, 2], [2, 3], [3, 0]],
      [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]]], "4-ary, a!=2"),
    ([4, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
      [[(0, 1), (3, 1), (1, -1)], [(0, 1), (4, 1), (2, -1)],
       [(1, 1), (5, 1), (2, -1)]]], "tetra, 3 faces"),
], ids=["open", "branching", "partial"])
def test_the_general_identity_holds_where_the_twos_do_not(cells, name):
    """`a/d + c/k = 1 + chi/E` for the declared incidence profile.

    This is an arity-count identity, not a chain-complex construction test. In
    particular, its branching fixture deliberately has an all-positive five-relation
    C2 declaration so that ``c=5``; canonical C1 shares correctly reject that as a
    face boundary. Count the declared supports directly rather than asking the exact
    relational-complex importer to materialize a non-closing C2 column.
    """
    nV = int(cells[0])
    nE = len(cells[1])
    nF = len(cells[2])
    i1 = sum(len(cell) for cell in cells[1])
    i2 = sum(len(cell) for cell in cells[2])
    a, d = Fraction(i1, nE), Fraction(i1, nV)
    c, k = Fraction(i2, nE), Fraction(i2, nF)
    chi = nV - nE + nF
    assert a / d + c / k == Fraction(1) + Fraction(chi, nE)


def test_the_general_form_reduces_to_the_surface_one():
    """At a = c = 2 it is the surface identity, which is why that one is a special
    case rather than a different statement."""
    a = c = Fraction(2)
    d, k = Fraction(4), Fraction(3)              # octahedron
    assert a / d + c / k == Fraction(2) / d + Fraction(2) / k


@pytest.mark.parametrize("a,c,expected", [
    (2, 2, [(3, 6), (4, 4), (6, 3)]),
    (3, 2, [(4, 8), (5, 5), (6, 4), (9, 3)]),
    (3, 3, [(4, 12), (6, 6), (12, 4)]),
])
def test_each_arity_profile_has_its_own_ideal_family(a, c, expected):
    """The three regular tilings are the a = c = 2 row of a family. A branching complex
    refines toward a different ideal set."""
    found = []
    for d in range(1, 60):
        rest = Fraction(1) - Fraction(a, d)
        if rest <= 0:
            continue
        k = Fraction(c) / rest
        if k.denominator == 1 and k >= 2:
            found.append((d, int(k)))
    assert found == expected


@pytest.mark.parametrize("a,c", [(2, 2), (2, 3), (3, 2), (3, 3), (4, 3), (5, 2)])
def test_the_self_dual_ideal_is_a_plus_c(a, c):
    """d = k forces (a + c)/d = 1, so the balanced member of every family is a + c."""
    d = a + c
    assert Fraction(a, d) + Fraction(c, d) == 1


def test_the_graded_form_holds_at_grade_three():
    """Euler divided by n_1 reaches any number of grades; the solids carry a volume
    cell, so chi is 1 rather than 2."""
    import scipy.sparse as sp

    from rexgraph.graded_boundary import build_graded_boundaries
    for cells in (solid_octahedron_3rex(), square_pyramid_3rex(),
                  truncated_icosahedron_3rex()):
        B = [sp.csr_matrix(b) for b in build_graded_boundaries(cells)]
        n = [B[0].shape[0]] + [b.shape[1] for b in B]
        chi = sum((-1) ** k * n[k] for k in range(len(n)))
        lhs = sum(Fraction((-1) ** k * n[k], n[1]) for k in range(len(n)))
        assert lhs == Fraction(chi, n[1])
        assert chi == 1, "a solid ball, not a surface"
