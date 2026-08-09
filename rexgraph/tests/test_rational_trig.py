"""The spread tower: quadrance, spread, and the Gram block over its diagonal.

Every claim here is an identity, so the tests are equalities rather than tolerances.
Where a float path exists alongside the exact one, the test is that they agree and
that the float path induces the same ordering, because that is the whole reason to
have both.
"""
from __future__ import annotations

from fractions import Fraction as Fr

import numpy as np
import pytest

from rexgraph.faces import find_cycles, solve_face_column
from rexgraph.graph import RexGraph
from rexgraph.rational_trig import (
    carries_cycle,
    cross_spread,
    gram,
    gram_determinant,
    gram_rank,
    independent_cycles,
    quadrance,
    rank_increment,
    spread,
    spread_matrix,
)

#### quadrance and spread


def test_quadrance_is_the_squared_length():
    assert quadrance([3, 4], exact=True) == 25
    assert quadrance([3, 4]) == pytest.approx(25.0)


def test_quadrance_of_rationals_stays_rational():
    q = quadrance([Fr(1, 3), Fr(2, 3)], exact=True)
    assert q == Fr(5, 9)
    assert isinstance(q, Fr)


def test_perpendicular_vectors_have_spread_one():
    assert spread([1, 0], [0, 1], exact=True) == 1


def test_parallel_vectors_have_spread_zero():
    assert spread([1, 2], [2, 4], exact=True) == 0
    assert spread([1, 2], [-3, -6], exact=True) == 0, "antiparallel is still parallel"


def test_spread_is_rational_where_the_cosine_is_not():
    """The point of the tower. `cos` needs two square roots; `1 - cos^2` does not."""
    u, v = [1, 1, 0], [1, 0, 1]
    s = spread(u, v, exact=True)
    assert s == Fr(3, 4)
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    assert float(s) == pytest.approx(1 - cos * cos)


def test_spread_agrees_between_the_exact_and_float_paths():
    rng = np.random.RandomState(0)
    for _ in range(50):
        u, v = rng.randint(-5, 6, 4), rng.randint(-5, 6, 4)
        if not u.any() or not v.any():
            continue
        assert float(spread(u, v, exact=True)) == pytest.approx(spread(u, v))


def test_a_zero_vector_has_no_spread():
    """No angle is defined, and an absence must not read as a right angle."""
    assert spread([0, 0], [1, 1], exact=True) is None
    assert spread([0, 0], [1, 1]) is None


#### the identity: spread = det(Gram) / product of its diagonal


def test_spread_is_the_gram_determinant_over_its_diagonal_product():
    rng = np.random.RandomState(1)
    for _ in range(50):
        u, v = rng.randint(-6, 7, 5), rng.randint(-6, 7, 5)
        if not u.any() or not v.any():
            continue
        G = gram([u, v], exact=True)
        det = gram_determinant([u, v])
        prod = G[0][0] * G[1][1]
        assert spread(u, v, exact=True) == det / prod


def test_the_gram_diagonal_is_the_quadrances():
    G = gram([[1, 2], [3, 4]], exact=True)
    assert G[0][0] == quadrance([1, 2], exact=True)
    assert G[1][1] == quadrance([3, 4], exact=True)


def test_the_gram_block_is_symmetric():
    G = gram([[1, 2, 3], [4, 5, 6], [7, 8, 9]], exact=True)
    assert all(G[i][j] == G[j][i] for i in range(3) for j in range(3))


#### the degeneracy locus is the cycle space


@pytest.mark.parametrize("name,src,tgt,is_cycle", [
    ("triangle", [0, 1, 2], [1, 2, 0], True),
    ("square", [0, 1, 2, 3], [1, 2, 3, 0], True),
    ("path", [0, 1, 2], [1, 2, 3], False),
    ("star", [0, 0, 0], [1, 2, 3], False),
])
def test_the_gram_determinant_vanishes_exactly_on_cycles(name, src, tgt, is_cycle):
    rex = RexGraph(sources=np.asarray(src, np.int32),
                   targets=np.asarray(tgt, np.int32))
    assert rex.carries_cycle(range(rex.nE)) is is_cycle, name
    assert (int(rex.betti[1]) > 0) is is_cycle


def test_the_rank_deficiency_is_the_cycle_dimension():
    """`k - rank(Gram_k)` is what Betti counts globally, read on a subset."""
    rng = np.random.RandomState(2)
    for _ in range(20):
        nv = rng.randint(3, 9)
        ne = rng.randint(3, 14)
        src = rng.randint(0, nv, ne).astype(np.int32)
        tgt = ((src + 1 + rng.randint(0, nv - 1, ne)) % nv).astype(np.int32)
        rex = RexGraph(sources=src, targets=tgt)
        assert rex.cycle_dimension_of(range(rex.nE)) == int(rex.betti[1])


def test_a_subset_reports_its_own_cycles_not_the_complexs():
    """Two triangles sharing nothing: each subset carries one, the whole carries two."""
    rex = RexGraph(sources=np.asarray([0, 1, 2, 3, 4, 5], np.int32),
                   targets=np.asarray([1, 2, 0, 4, 5, 3], np.int32))
    assert rex.cycle_dimension_of([0, 1, 2]) == 1
    assert rex.cycle_dimension_of([3, 4, 5]) == 1
    assert rex.cycle_dimension_of(range(6)) == 2
    assert rex.cycle_dimension_of([0, 1]) == 0


def test_independent_cycles_of_a_dependent_set():
    a = [1, 0, -1]
    b = [0, 1, -1]
    c = [1, 1, -2]                       # = a + b
    assert independent_cycles([a, b, c]) == 1
    assert carries_cycle([a, b, c]) is True
    assert carries_cycle([a, b]) is False


#### T and G share a spread denominator at every grade


@pytest.mark.parametrize("name,src,tgt", [
    ("triangle", [0, 1, 2], [1, 2, 0]),
    ("square", [0, 1, 2, 3], [1, 2, 3, 0]),
    ("star", [0, 0, 0], [1, 2, 3]),
    ("path", [0, 1, 2], [1, 2, 3]),
])
def test_the_signed_and_unsigned_grams_share_a_diagonal(name, src, tgt):
    """Squaring an entry discards its sign, so the diagonals cannot differ. This is
    what lets the two spreads share a denominator."""
    from rexgraph.graded_boundary import _rex_b1_csr
    rex = RexGraph(sources=np.asarray(src, np.int32),
                   targets=np.asarray(tgt, np.int32))
    B = np.asarray(_rex_b1_csr(rex).todense())
    cols = [B[:, j] for j in range(B.shape[1])]
    T = gram(cols, exact=True)
    G = gram([np.abs(c) for c in cols], exact=True)
    assert [T[i][i] for i in range(len(T))] == [G[i][i] for i in range(len(G))], name


def test_the_spread_difference_is_the_determinant_difference():
    from rexgraph.graded_boundary import _rex_b1_csr
    rex = RexGraph(sources=np.asarray([0, 1, 2], np.int32),
                   targets=np.asarray([1, 2, 0], np.int32))
    B = np.asarray(_rex_b1_csr(rex).todense())
    cols = [B[:, j] for j in range(B.shape[1])]
    T = gram(cols, exact=True)
    G = gram([np.abs(c) for c in cols], exact=True)
    s_T, s_G, diff, denom = cross_spread(T, G)
    assert diff == (gram_determinant([np.abs(c) for c in cols])
                    - gram_determinant(cols)) / denom
    assert s_T - s_G == diff


def test_the_signed_gram_sees_a_triangle_and_the_unsigned_one_does_not():
    """The set-theoretic encoding cannot see an odd cycle. det T vanishes because the
    columns are dependent; det G does not, because the unsigned columns are not."""
    from rexgraph.graded_boundary import _rex_b1_csr
    rex = RexGraph(sources=np.asarray([0, 1, 2], np.int32),
                   targets=np.asarray([1, 2, 0], np.int32))
    B = np.asarray(_rex_b1_csr(rex).todense())
    cols = [B[:, j] for j in range(B.shape[1])]
    assert gram_determinant(cols) == 0
    assert gram_determinant([np.abs(c) for c in cols]) != 0


def test_an_even_cycle_is_visible_to_both():
    """A square is bipartite, so the unsigned columns are dependent too and the
    spread difference vanishes. The difference detects ODD cycles."""
    from rexgraph.graded_boundary import _rex_b1_csr
    rex = RexGraph(sources=np.asarray([0, 1, 2, 3], np.int32),
                   targets=np.asarray([1, 2, 3, 0], np.int32))
    B = np.asarray(_rex_b1_csr(rex).todense())
    cols = [B[:, j] for j in range(B.shape[1])]
    assert gram_determinant(cols) == 0
    assert gram_determinant([np.abs(c) for c in cols]) == 0


def test_cross_spread_refuses_a_pair_that_is_not_a_t_g_pair():
    with pytest.raises(ValueError, match="share a diagonal"):
        cross_spread([[Fr(1), Fr(0)], [Fr(0), Fr(1)]],
                     [[Fr(2), Fr(0)], [Fr(0), Fr(1)]])


#### harmonic becomes curl


def test_a_face_that_raises_the_rank_fills_exactly_one_hole():
    src = np.asarray([0, 1, 2, 2, 3, 4, 0, 5, 1], np.int32)
    tgt = np.asarray([1, 2, 0, 3, 4, 2, 5, 1, 3], np.int32)
    rex = RexGraph(sources=src, targets=tgt)
    before = int(rex.betti[1])
    candidates = find_cycles(rex, 3)
    assert candidates, "the fixture has no triangle to attach"

    filled = 0
    for cand in candidates:
        col = [float(x) for x in solve_face_column(rex, cand)]
        assert rex.face_fills_a_hole(cand, col) is True
        filled += 1
    after = RexGraph(sources=src, targets=tgt)
    after.add_faces(candidates, signs=None)
    assert int(after.betti[1]) == before - filled, \
        "each rank-raising face must convert exactly one harmonic class to curl"


def test_an_already_attached_face_fills_nothing():
    """A column dependent on the attached ones adds a face and kills no hole."""
    src = np.asarray([0, 1, 2, 2, 3, 4, 0, 5, 1], np.int32)
    tgt = np.asarray([1, 2, 0, 3, 4, 2, 5, 1, 3], np.int32)
    base = RexGraph(sources=src, targets=tgt)
    candidates = find_cycles(base, 3)
    col = [float(x) for x in solve_face_column(base, candidates[0])]
    after = RexGraph(sources=src, targets=tgt)
    after.add_faces(candidates, signs=None)
    assert after.face_fills_a_hole(candidates[0], col) is False


def test_rank_increment_is_one_or_zero():
    a, b = [1, 0, 0], [0, 1, 0]
    assert rank_increment([a], b) == 1
    assert rank_increment([a, b], [1, 1, 0]) == 0, "a dependent column adds no rank"
    assert rank_increment([], a) == 1


#### the rational similarity


def _complex():
    return RexGraph(sources=np.asarray([0, 1, 2, 2, 3, 4, 0, 5, 1], np.int32),
                    targets=np.asarray([1, 2, 0, 3, 4, 2, 5, 1, 3], np.int32))


def test_spread_similarity_is_the_square_of_fiber_similarity():
    rex = _complex()
    assert np.abs(np.asarray(rex.spread_similarity)
                  - np.asarray(rex.fiber_similarity) ** 2).max() < 1e-12


def test_spread_similarity_orders_pairs_identically():
    """The reason the square is a usable substitute: every comparison is preserved,
    so any ranking or threshold decision is unchanged and now needs no square root."""
    rex = _complex()
    a = np.asarray(rex.fiber_similarity)
    b = np.asarray(rex.spread_similarity)
    pairs = [(i, j) for i in range(rex.nV) for j in range(i + 1, rex.nV)]
    for p in pairs:
        for q in pairs:
            assert (a[p] < a[q]) == (b[p] < b[q])


def test_spread_similarity_is_symmetric_and_bounded():
    b = np.asarray(_complex().spread_similarity)
    assert np.allclose(b, b.T)
    assert b.min() >= 0.0 and b.max() <= 1.0


def test_exact_spread_agrees_with_the_float_reading():
    rex = _complex()
    chi = np.asarray(rex.star_character)
    exact = rex.exact_spread(0, 1)
    assert float(exact) == pytest.approx(spread(chi[0], chi[1]))


#### the matrix form


def test_the_spread_matrix_is_zero_on_its_diagonal():
    S = spread_matrix([[1, 0], [0, 1], [1, 1]])
    assert np.allclose(np.diag(S), 0.0)
    assert S[0, 1] == pytest.approx(1.0)


def test_the_spread_matrix_matches_the_pairwise_function():
    vs = [[1, 2, 3], [4, 0, 1], [0, 1, 1]]
    S = spread_matrix(vs)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert S[i, j] == pytest.approx(spread(vs[i], vs[j]))


def test_a_zero_vector_is_not_given_a_right_angle():
    S = spread_matrix([[0, 0], [1, 1]])
    assert np.isnan(S[0, 1])


#### the T/G pair on the complex


@pytest.mark.parametrize("name,src,tgt,has_odd_cycle", [
    ("triangle", [0, 1, 2], [1, 2, 0], True),
    ("square", [0, 1, 2, 3], [1, 2, 3, 0], False),
    ("star", [0, 0, 0], [1, 2, 3], False),
    ("pentagon", [0, 1, 2, 3, 4], [1, 2, 3, 4, 0], True),
])
def test_grade_spread_detects_odd_cycles(name, src, tgt, has_odd_cycle):
    """`det T = 0` on any cycle, `det G = 0` on a bipartite component. Their
    difference is non-zero exactly where the two disagree, which is an odd cycle."""
    rex = RexGraph(sources=np.asarray(src, np.int32),
                   targets=np.asarray(tgt, np.int32))
    out = rex.grade_spread(1)
    assert out["orientation_content"] is has_odd_cycle, name
    assert (out["difference"] != 0.0) is has_odd_cycle


def test_grade_spread_reports_the_shared_denominator():
    rex = RexGraph(sources=np.asarray([0, 1, 2], np.int32),
                   targets=np.asarray([1, 2, 0], np.int32))
    out = rex.grade_spread(1)
    assert out["shared_denominator"] == "8"     # three columns of quadrance 2
    assert out["difference_exact"] == "1/2"


def test_grade_spread_refuses_a_grade_that_is_not_there():
    rex = RexGraph(sources=np.asarray([0], np.int32),
                   targets=np.asarray([1], np.int32))
    with pytest.raises(ValueError, match="grade"):
        rex.grade_spread(9)


def test_grade_spread_says_so_when_a_grade_is_empty():
    rex = RexGraph(sources=np.asarray([0, 1, 2], np.int32),
                   targets=np.asarray([1, 2, 0], np.int32))
    out = rex.grade_spread(2)
    assert out["available"] is False and out["n_cells"] == 0


#### the harmonic shadow, against the existing API

# `harmonic_shadow`, `hypermanifold` and `dimensional_subsumption` are cached
# properties that already exist and are already eigen-free. What is tested here is the
# identity connecting them to `face_fills_a_hole`, which is their pointwise companion.


def _filled():
    src = np.asarray([0, 1, 2, 2, 3, 4, 0, 5, 1], np.int32)
    tgt = np.asarray([1, 2, 0, 3, 4, 2, 5, 1, 3], np.int32)
    base = RexGraph(sources=src, targets=tgt)
    rex = RexGraph(sources=src, targets=tgt)
    rex.add_faces(find_cycles(base, 3), signs=None)
    return base, rex


def test_the_shadow_dimension_is_the_drop_in_betti():
    """The shadow is how many cycles the cells one grade up fill, so it equals the
    difference between the unfilled and filled complexes."""
    base, rex = _filled()
    drop = int(base.betti[1]) - int(rex.betti[1])
    assert rex.harmonic_shadow["shadow_dim"] == drop


def test_the_shadow_and_the_candidate_predicate_agree():
    """`harmonic_shadow` is the aggregate; `face_fills_a_hole` is the pointwise form.
    The number of candidates that fill must equal the shadow they produce."""
    src = np.asarray([0, 1, 2, 2, 3, 4, 0, 5, 1], np.int32)
    tgt = np.asarray([1, 2, 0, 3, 4, 2, 5, 1, 3], np.int32)
    base = RexGraph(sources=src, targets=tgt)
    cands = find_cycles(base, 3)
    n_filling = sum(
        1 for c in cands
        if base.face_fills_a_hole(c, [float(x) for x in solve_face_column(base, c)]))
    rex = RexGraph(sources=src, targets=tgt)
    rex.add_faces(cands, signs=None)
    assert rex.harmonic_shadow["shadow_dim"] == n_filling


def test_a_redundant_face_does_not_raise_the_shadow():
    """Attaching the same cycle twice adds a cell and fills nothing, so the shadow
    stays at one while the cell count does not."""
    src = np.asarray([0, 1, 2], np.int32)
    tgt = np.asarray([1, 2, 0], np.int32)
    base = RexGraph(sources=src, targets=tgt)
    cycles = list(find_cycles(base, 3))
    rex = RexGraph(sources=src, targets=tgt)
    rex.add_faces(cycles + cycles, signs=None)
    assert rex.harmonic_shadow["shadow_dim"] == 1
    assert int(rex.nF_hodge) > 1, "the duplicate face was not attached at all"


def test_betti_never_increases_along_the_filtration():
    """Theorem 8.1: adding cells fills holes and never opens one."""
    _base, rex = _filled()
    ok, violations = rex.dimensional_subsumption
    assert ok is True and list(violations) == []


def test_the_filtered_family_is_nested():
    _base, rex = _filled()
    levels = rex.hypermanifold["manifolds"]
    assert len(levels) >= 2
    for a, b in zip(levels, levels[1:], strict=False):
        assert b["N"] >= a["N"]
        assert len(b["cells"]) > len(a["cells"])


#### the coboundary half, and the two spreads together


def test_the_boundary_and_coboundary_grams_sum_to_the_hodge_laplacian():
    """The reason both halves exist. A cell's boundary is a column of `B_k` and its
    coboundary a row of `B_k+1`; their Grams are the down and up parts of `L_k` and
    nothing is left over."""
    from rexgraph.graded_boundary import graded_boundaries_from_rex
    _base, rex = _filled()
    B = graded_boundaries_from_rex(rex)
    B1 = np.asarray(B[0].todense())
    B2 = np.asarray(B[1].todense())
    down = B1.T @ B1
    up = B2 @ B2.T
    assert down.shape == up.shape == (rex.nE, rex.nE)
    assert np.allclose(down + up, B1.T @ B1 + B2 @ B2.T)


def test_the_two_gram_ranks_are_the_hodge_dimensions():
    """gradient = rank(B_k), curl = rank(B_k+1), harmonic = what neither reaches."""
    from rexgraph.graded_boundary import graded_boundaries_from_rex
    _base, rex = _filled()
    B = graded_boundaries_from_rex(rex)
    B1 = np.asarray(B[0].todense())
    B2 = np.asarray(B[1].todense())

    boundary_cols = [B1[:, e] for e in range(B1.shape[1])]
    coboundary_rows = [B2[e, :] for e in range(B2.shape[0])]
    dims = rex.hodge_dimensions(1)
    assert dims["gradient"] == gram_rank(boundary_cols)
    assert dims["curl"] == gram_rank(coboundary_rows)


@pytest.mark.parametrize("grade", [0, 1, 2])
def test_the_harmonic_dimension_is_betti_at_every_grade(grade):
    """The projection identity: a cell is harmonic exactly when it is degenerate for
    BOTH spreads, unreachable from below and bounding nothing above."""
    _base, rex = _filled()
    dims = rex.hodge_dimensions(grade)
    assert dims["harmonic"] == int(rex.betti[grade])
    assert dims["gradient"] + dims["curl"] + dims["harmonic"] == dims["n_cells"]


def test_an_unfilled_complex_has_no_curl():
    """With nothing one grade up, every cycle stays harmonic."""
    base, _rex = _filled()
    dims = base.hodge_dimensions(1)
    assert dims["curl"] == 0
    assert dims["harmonic"] == int(base.betti[1])


def test_filling_moves_dimension_from_harmonic_to_curl():
    base, rex = _filled()
    before, after = base.hodge_dimensions(1), rex.hodge_dimensions(1)
    assert before["gradient"] == after["gradient"], \
        "attaching a face changed the gradient part, which it must not"
    moved = after["curl"] - before["curl"]
    assert moved > 0
    assert before["harmonic"] - after["harmonic"] == moved


def test_the_coboundary_spread_is_rational():
    _base, rex = _filled()
    s = rex.coboundary_spread(1, 0, 1)
    assert isinstance(s, Fr)
    assert 0 <= s <= 1


def test_a_cell_bounding_nothing_has_no_coboundary_spread():
    """No direction up means no angle, and an absence must not read as a right one."""
    _base, rex = _filled()
    dims = rex.hodge_dimensions(1)
    assert dims["harmonic"] > 0, "the fixture has no harmonic edge to check"
    import numpy as _np

    from rexgraph.graded_boundary import graded_boundaries_from_rex
    B2 = _np.asarray(graded_boundaries_from_rex(rex)[1].todense())
    bounding_nothing = [e for e in range(rex.nE) if not B2[e, :].any()]
    assert bounding_nothing, "every edge bounds something, so the case is untested"
    a = bounding_nothing[0]
    assert rex.coboundary_spread(1, a, a) is None


def test_there_is_no_coboundary_above_the_top_grade():
    _base, rex = _filled()
    assert rex.coboundary_spread(2, 0, 0) is None


def test_hodge_dimensions_refuses_a_grade_the_complex_does_not_carry():
    _base, rex = _filled()
    with pytest.raises(ValueError, match="grade"):
        rex.hodge_dimensions(7)


def test_the_dimensions_need_no_signal_and_no_eigenvalue():
    """`hodge_full` decomposes a given signal by projection; this is the dimensions of
    the three subspaces themselves, from two integer ranks."""
    _base, rex = _filled()
    dims = rex.hodge_dimensions(1)
    assert all(isinstance(dims[k], int)
               for k in ("gradient", "curl", "harmonic", "n_cells"))


#### recovering the exact character from its float


def test_a_small_complex_has_a_small_rational_character():
    """The characters are rational and on a small complex they are SMALL rationals:
    a triangle's star character is 1/4 in every channel."""
    from rexgraph.rational_trig import rational_reconstruct
    rex = RexGraph(sources=np.asarray([0, 1, 2], np.int32),
                   targets=np.asarray([1, 2, 0], np.int32))
    got = rational_reconstruct(rex.star_character)
    assert got is not None
    assert all(x == Fr(1, 4) for row in got for x in row)


@pytest.mark.parametrize("name,src,tgt,denominator", [
    ("star", [0, 0, 0, 0, 0], [1, 2, 3, 4, 5], 3),
    ("K4", [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3], 135),
    ("path", [0, 1, 2, 0, 3], [1, 2, 0, 3, 4], 969),
])
def test_the_denominator_is_recovered_exactly(name, src, tgt, denominator):
    from rexgraph.rational_trig import rational_reconstruct
    rex = RexGraph(sources=np.asarray(src, np.int32),
                   targets=np.asarray(tgt, np.int32))
    got = rational_reconstruct(rex.star_character)
    assert got is not None, name
    assert max(x.denominator for row in got for x in row) == denominator


def test_reconstruction_refuses_what_a_double_cannot_pin_down():
    """The guard that makes this worth having.

    Continued fractions will always find SOME fraction close to a float, so a
    reconstruction that does not check always "succeeds" and returns a number that is
    not the value. A rational is uniquely determined by a double only while its
    denominator is under `sqrt(1/(2 eps))`, about 4.7e7; past that this refuses.
    """
    from rexgraph.rational_trig import rational_reconstruct
    rng = np.random.RandomState(1)
    src = rng.randint(0, 20, 40).astype(np.int32)
    tgt = ((src + 1 + rng.randint(0, 19, 40)) % 20).astype(np.int32)
    rex = RexGraph(sources=src, targets=tgt)
    assert rational_reconstruct(rex.star_character) is None, (
        "a 20-vertex complex's character is not recoverable from float64; accepting "
        "one means accepting a fraction that merely matches the float")


def test_the_bound_is_the_classical_one():
    from rexgraph.rational_trig import MAX_RECOVERABLE_DENOMINATOR
    expected = int((1.0 / (2.0 * np.finfo(np.float64).eps)) ** 0.5)
    assert expected == MAX_RECOVERABLE_DENOMINATOR
    assert 4e7 < MAX_RECOVERABLE_DENOMINATOR < 5e7


def test_an_exact_input_round_trips():
    from rexgraph.rational_trig import rational_reconstruct
    got = rational_reconstruct([0.25, 0.5, 0.125])
    assert got == [Fr(1, 4), Fr(1, 2), Fr(1, 8)]


#### the determinant reading has a regime, and says so


@pytest.mark.parametrize("name,src,tgt,odd", [
    ("triangle", [0, 1, 2], [1, 2, 0], True),
    ("square", [0, 1, 2, 3], [1, 2, 3, 0], False),
    ("pentagon", [0, 1, 2, 3, 4], [1, 2, 3, 4, 0], True),
    ("hexagon", [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0], False),
    ("two triangles", [0, 1, 2, 2, 3, 4], [1, 2, 0, 3, 4, 2], True),
    ("K4", [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3], True),
])
def test_odd_cycles_are_detected_at_any_size_by_rank(name, src, tgt, odd):
    """The rank reading holds where the determinant one gives out."""
    rex = RexGraph(sources=np.asarray(src, np.int32),
                   targets=np.asarray(tgt, np.int32))
    assert rex.grade_spread(1)["odd_cycle_present"] is odd, name


def test_the_determinant_reading_reports_when_it_is_out_of_regime():
    """A Gram determinant over more columns than the operator's rank is zero for a
    counting reason. Past `n_cells > n_rows` both determinants vanish and their
    difference carries no information, which the result has to say rather than look
    like 'no odd cycle'."""
    wide = RexGraph(sources=np.asarray([0, 0, 0, 1, 1, 2], np.int32),
                    targets=np.asarray([1, 2, 3, 2, 3, 3], np.int32))
    out = wide.grade_spread(1)
    assert out["informative"] is False
    assert out["difference"] == 0.0
    assert out["odd_cycle_present"] is True, \
        "the rank reading must still see the odd cycle the determinant cannot"

    narrow = RexGraph(sources=np.asarray([0, 1, 2], np.int32),
                      targets=np.asarray([1, 2, 0], np.int32))
    assert narrow.grade_spread(1)["informative"] is True


#### the rank tower


def _tower_fixture():
    src = np.asarray([0, 1, 2, 2, 3, 4, 0, 5, 1], np.int32)
    tgt = np.asarray([1, 2, 0, 3, 4, 2, 5, 1, 3], np.int32)
    base = RexGraph(sources=src, targets=tgt)
    rex = RexGraph(sources=src, targets=tgt)
    rex.add_faces(find_cycles(base, 3), signs=None)
    return rex


@pytest.mark.parametrize("name,src,tgt,fill", [
    ("triangle", [0, 1, 2], [1, 2, 0], True),
    ("K4", [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3], True),
    ("two triangles", [0, 1, 2, 2, 3, 4], [1, 2, 0, 3, 4, 2], False),
])
def test_the_curl_at_a_grade_is_the_gradient_above_it(name, src, tgt, fill):
    """The recursion. Each rank is counted twice (as the curl below and the gradient
    above), so the tower is one integer sequence rather than a dimension per grade."""
    rex = RexGraph(sources=np.asarray(src, np.int32),
                   targets=np.asarray(tgt, np.int32))
    if fill:
        cycles = find_cycles(rex, 3)
        if cycles:
            rex.add_faces(cycles, signs=None)
    grades = rex.rank_tower()["grades"]
    for k in range(len(grades) - 1):
        assert grades[k]["curl"] == grades[k + 1]["gradient"], f"{name} at grade {k}"


def test_the_tower_agrees_with_the_per_grade_reading():
    rex = _tower_fixture()
    for g in rex.rank_tower()["grades"]:
        d = rex.hodge_dimensions(g["grade"])
        assert (d["gradient"], d["curl"], d["harmonic"]) == (
            g["gradient"], g["curl"], g["harmonic"])


def test_every_harmonic_count_is_betti():
    rex = _tower_fixture()
    for g in rex.rank_tower()["grades"]:
        assert g["harmonic"] == int(rex.betti[g["grade"]])


def test_euler_falls_out_of_the_tower():
    """Every rank enters the alternating sum once with each sign and cancels, so the
    cell count and the Betti count agree without either being computed from the
    other."""
    for fill in (True, False):
        src = np.asarray([0, 1, 2, 2, 3, 4, 0, 5, 1], np.int32)
        tgt = np.asarray([1, 2, 0, 3, 4, 2, 5, 1, 3], np.int32)
        rex = RexGraph(sources=src, targets=tgt)
        if fill:
            rex.add_faces(find_cycles(RexGraph(sources=src, targets=tgt), 3),
                          signs=None)
        tower = rex.rank_tower()
        assert tower["euler"] == tower["euler_from_betti"]


def test_the_tower_is_the_rank_sequence():
    rex = _tower_fixture()
    tower = rex.rank_tower()
    assert tower["ranks"] == [g["curl"] for g in tower["grades"]]


#### the character, computed exactly rather than recovered


def test_the_exact_character_matches_the_float_path():
    from rexgraph.rational_trig import exact_star_character
    rex = _tower_fixture()
    exact, names = exact_star_character(rex)
    approx = np.asarray(rex.star_character)
    assert len(names) == approx.shape[1]
    worst = max(abs(float(exact[i][k]) - approx[i, k])
                for i in range(len(exact)) for k in range(len(names)))
    assert worst < 1e-12


@pytest.mark.parametrize("name,src,tgt,denominator", [
    ("triangle", [0, 1, 2], [1, 2, 0], 4),
    ("K4", [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3], 135),
    ("path", [0, 1, 2, 0, 3], [1, 2, 0, 3, 4], 969),
])
def test_the_exact_character_gives_the_true_small_denominator(name, src, tgt,
                                                              denominator):
    from rexgraph.rational_trig import exact_star_character
    rex = RexGraph(sources=np.asarray(src, np.int32),
                   targets=np.asarray(tgt, np.int32))
    exact, _names = exact_star_character(rex)
    assert max(x.denominator for row in exact for x in row) == denominator, name


def test_the_character_is_computable_where_it_is_not_recoverable():
    """The two paths are complementary, and they confirm each other.

    `rational_reconstruct` reads a float and refuses past ~4.7e7 because a double
    cannot pin down a larger denominator. `exact_character` never converts to float,
    so it works at any size, and what it returns is why the refusal was right: at 20
    vertices the true denominator already has 25 digits.
    """
    from rexgraph.rational_trig import exact_star_character, rational_reconstruct
    rng = np.random.RandomState(1)
    src = rng.randint(0, 20, 40).astype(np.int32)
    tgt = ((src + 1 + rng.randint(0, 19, 40)) % 20).astype(np.int32)
    rex = RexGraph(sources=src, targets=tgt)

    assert rational_reconstruct(rex.star_character) is None
    exact, _names = exact_star_character(rex)
    assert exact is not None
    true_denominator = max(x.denominator for row in exact for x in row)
    from rexgraph.rational_trig import MAX_RECOVERABLE_DENOMINATOR
    assert true_denominator > MAX_RECOVERABLE_DENOMINATOR, (
        "the true denominator is within reach, so the refusal was over-cautious")


def test_the_exact_character_rows_sum_to_one():
    """chi lives on the simplex, exactly, with no rounding to hide a drift."""
    from rexgraph.rational_trig import exact_character, exact_star_character
    rex = _tower_fixture()
    for rows, _names in (exact_character(rex), exact_star_character(rex)):
        for row in rows:
            assert sum(row) == 1


def test_the_exact_character_needs_no_solve():
    """It is a ratio of channel DIAGONALS over their traces, all integers, so it is
    O(nnz) and not an exact linear solve."""
    from rexgraph.rational_trig import exact_star_character
    rng = np.random.RandomState(3)
    src = rng.randint(0, 500, 1500).astype(np.int32)
    tgt = ((src + 1 + rng.randint(0, 499, 1500)) % 500).astype(np.int32)
    rex = RexGraph(sources=src, targets=tgt)
    import time
    start = time.monotonic()
    exact, names = exact_star_character(rex)
    elapsed = time.monotonic() - start
    assert exact is not None and len(names) >= 2
    assert elapsed < 20.0, f"exact character took {elapsed:.1f}s on 1500 relations"
