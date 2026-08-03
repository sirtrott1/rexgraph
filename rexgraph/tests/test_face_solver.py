"""Face coefficients are SOLVED from the chain condition, not declared.

A grade-1 column is declared: Definition 2.1 gives it the shape (-1, 1/(k-1), ...). A
grade-2 column is not. Nothing imposes a shape on it; it is whatever satisfies
B1 c_f = 0 on the edges it spans, and what the solution owes is cancellation. So the
right primitive is a solver over the rationals, and the wrong one is a caller-supplied
sign vector that the library trusts.

`add_faces` trusted the caller. A wrong orientation was silently dropped by the Hodge
filter: nF_hodge stays 0, the cycle stays open, betti_1 stays 1, and nothing reports
why. Of the four sign/orientation combinations on one triangle, only one lands a face.

Three properties hold here.

EXACT. The nullspace is computed over Fraction, so a face column is exact rational and
the chain condition holds at 0, not at a tolerance. No float, no eigensolve, no brute
force over 2^k sign patterns.

ANY k. The gon is the support size and is independent of the grade. Nothing here tests
3, and a k-gon's boundary is the CYCLE, not every pair.

REFUSAL IS A RESULT. A set of edges that bounds nothing returns None rather than a
forced face. A lone relation encloses nothing, and a partial overlap does not cancel.
"""

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph


def _ring(k):
    return RexGraph(sources=np.arange(k, dtype=np.int32),
                    targets=np.roll(np.arange(k, dtype=np.int32), -1))


def _branching(ptr, idx):
    return RexGraph.from_hypergraph(np.asarray(ptr, np.int32), np.asarray(idx, np.int32))


#### the solver
@pytest.mark.parametrize("k", [3, 4, 5, 6, 8, 12])
def test_a_k_cycle_is_solved_at_any_k(k):
    from rexgraph.faces import solve_face_column

    rex = _ring(k)
    c = solve_face_column(rex, np.arange(k, dtype=np.int32))
    assert c is not None, k
    assert len(c) == k
    assert all(isinstance(x, Fraction) for x in c)
    assert any(x != 0 for x in c)


@pytest.mark.parametrize("k", [3, 4, 5, 6, 8, 12])
def test_the_solved_column_satisfies_the_chain_condition_exactly(k):
    """B1 c_f = 0 over the rationals, so the residual is 0 and not an epsilon."""
    from rexgraph.faces import solve_face_column

    rex = _ring(k)
    edges = np.arange(k, dtype=np.int32)
    c = solve_face_column(rex, edges)
    B1 = np.asarray(rex.B1, dtype=object)
    img = [sum(Fraction(int(round(B1[v, e] * 1))) * c[i] for i, e in enumerate(edges))
           for v in range(int(rex.nV))]
    assert all(x == 0 for x in img), (k, img)


@pytest.mark.parametrize("k", [3, 4, 5, 6, 8])
def test_a_cycle_face_has_uniform_moduli_and_the_gon_is_its_support(k):
    """Grade 2 does not inherit the grade-1 share. A cycle's coefficients are uniform, so
    the arity ratio reading returns 2 at every k and the gon is |supp(c_f)| instead."""
    from rexgraph.faces import solve_face_column

    c = solve_face_column(_ring(k), np.arange(k, dtype=np.int32))
    mags = {abs(x) for x in c if x != 0}
    assert len(mags) == 1, (k, mags)
    assert len([x for x in c if x != 0]) == k


def test_orientation_does_not_change_whether_a_face_exists():
    """The point of solving. Reversing the closing edge changes the SIGNS of the answer,
    never whether there is one."""
    from rexgraph.faces import solve_face_column

    fwd = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32))
    rev = RexGraph(sources=np.array([0, 1, 0], np.int32), targets=np.array([1, 2, 2], np.int32))
    a = solve_face_column(fwd, np.array([0, 1, 2], np.int32))
    b = solve_face_column(rev, np.array([0, 1, 2], np.int32))
    assert a is not None and b is not None
    assert [abs(x) for x in a] == [abs(x) for x in b]
    assert a != b                                        # the signs differ, the face does not


def test_edges_that_bound_nothing_are_refused():
    """A path encloses no area. Refusal is the correct answer, not a failure."""
    from rexgraph.faces import solve_face_column

    rex = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32))
    assert solve_face_column(rex, np.array([0, 1], np.int32)) is None


def test_a_lone_relation_encloses_nothing():
    from rexgraph.faces import solve_face_column

    rex = _branching([0, 4], [0, 1, 2, 3])
    assert solve_face_column(rex, np.array([0], np.int32)) is None


def test_a_branching_fan_is_closed_by_its_legs():
    """The hyperface: an arity-k relation against the k-1 legs that span its boundary.
    The solution is what cancels, and for the canonical fan that is the negatives of the
    relation's own column."""
    from rexgraph.faces import solve_face_column

    # h over {0,1,2,3} plus legs 0-1, 0-2, 0-3
    rex = _branching([0, 4, 6, 8, 10], [0, 1, 2, 3, 0, 1, 0, 2, 0, 3])
    c = solve_face_column(rex, np.array([0, 1, 2, 3], np.int32))
    assert c is not None
    assert c[0] != 0                                     # the relation itself participates
    B1 = np.asarray(rex.B1, dtype=float)
    img = B1[:, :4] @ np.array([float(x) for x in c])
    assert np.abs(img).max() < 1e-12


def test_a_partial_overlap_is_refused():
    """Two legs of a k=5 fan cannot cancel it, so no hyperface attaches."""
    from rexgraph.faces import solve_face_column

    rex = _branching([0, 5, 7, 9], [0, 1, 2, 3, 4, 0, 1, 0, 2])
    assert solve_face_column(rex, np.array([0, 1, 2], np.int32)) is None


#### integration with add_faces
@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_add_faces_can_solve_instead_of_trusting(k):
    rex = _ring(k)
    rex.add_faces([np.arange(k, dtype=np.int32)], signs=None)
    assert int(rex.nF_hodge) == 1
    B1 = np.asarray(rex.B1, dtype=float)
    B2 = np.asarray(rex.B2, dtype=float)
    assert np.abs(B1 @ B2).max() < 1e-12
    assert int(rex.betti[1]) == 0                        # the cycle is filled


def test_solving_beats_a_wrong_caller_sign():
    """The regression this exists for: the same face, one orientation, guessed vs solved."""
    made = RexGraph(sources=np.array([0, 1, 0], np.int32), targets=np.array([1, 2, 2], np.int32))
    made.add_faces([np.array([0, 1, 2], np.int32)], [np.array([1.0, 1.0, 1.0])])
    assert int(made.nF_hodge) == 0                       # guessed wrong, silently dropped

    solved = RexGraph(sources=np.array([0, 1, 0], np.int32), targets=np.array([1, 2, 2], np.int32))
    solved.add_faces([np.array([0, 1, 2], np.int32)], signs=None)
    assert int(solved.nF_hodge) == 1


def test_an_unsolvable_face_is_not_attached():
    rex = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32))
    rex.add_faces([np.array([0, 1], np.int32)], signs=None)
    assert int(rex.nF_hodge) == 0


#### detection: autoface and auto_hyperface
@pytest.mark.parametrize("k", [3, 4, 5, 6, 7, 8])
def test_autoface_closes_a_k_gon(k):
    """Geometry FROM topology: fill what the connectivity allows, at any k. The gon is
    the shape of the face and is independent of the grade."""
    from rexgraph.faces import autoface

    rex = _ring(k)
    assert autoface(rex, k) == 1
    assert int(rex.nF_hodge) == 1
    assert int(rex.betti[1]) == 0
    B1 = np.asarray(rex.B1, dtype=float)
    B2 = np.asarray(rex.B2, dtype=float)
    assert np.abs(B1 @ B2).max() < 1e-12


def test_autoface_defaults_to_triangles_and_does_not_claim_a_square():
    """`autoface` alone attaches triangles. A square is not a triangle, and silently
    filling it would be inventing topology."""
    from rexgraph.faces import autoface

    rex = _ring(4)
    assert autoface(rex) == 0
    assert int(rex.nF_hodge) == 0
    assert autoface(rex, 4) == 1


def test_autoface_accepts_a_range_of_k():
    from rexgraph.faces import autoface

    rex = _ring(5)
    assert autoface(rex, range(3, 7)) == 1


def test_auto_hyperface_closes_a_fan_and_invents_nothing():
    """The boundary intersection. No pairwise relations are added, which would be clique
    expansion, and no hub vertex is added, which would be star expansion."""
    from rexgraph.faces import auto_hyperface

    rex = _branching([0, 4, 6, 8, 10], [0, 1, 2, 3, 0, 1, 0, 2, 0, 3])
    nV0, nE0 = int(rex.nV), int(rex.nE)
    assert auto_hyperface(rex) == 1
    assert (int(rex.nV), int(rex.nE)) == (nV0, nE0)
    B1 = np.asarray(rex.B1, dtype=float)
    B2 = np.asarray(rex.B2, dtype=float)
    assert np.abs(B1 @ B2).max() < 1e-12


def test_the_fan_face_is_the_negative_of_the_relations_column_up_to_scale():
    """At the canonical orientation the face coefficients are the negatives of the
    hyperedge's own column, the distinguished sign migrating one grade up. Up to scale,
    since a face column is defined up to scale and this solver clears denominators."""
    from rexgraph.faces import auto_hyperface

    rex = _branching([0, 4, 6, 8, 10], [0, 1, 2, 3, 0, 1, 0, 2, 0, 3])
    auto_hyperface(rex)
    col = np.asarray(rex.B2, dtype=float)[:, 0]
    col = col / col[0]                                  # normalise the relation's own entry
    assert np.allclose(col, [1.0, -1 / 3, -1 / 3, -1 / 3], atol=1e-12)


def test_a_lone_relation_bounds_no_area():
    from rexgraph.faces import auto_hyperface

    rex = _branching([0, 4], [0, 1, 2, 3])
    assert auto_hyperface(rex) == 0
    assert int(rex.nF_hodge) == 0


def test_auto_hyperface_ignores_pairwise_only_complexes():
    """Nothing branching, nothing to close. autoface and auto_hyperface stay in their
    own lanes."""
    from rexgraph.faces import auto_hyperface

    rex = _ring(3)
    assert auto_hyperface(rex) == 0


#### the cycle space, at any arity
CYCLE_FIXTURES = [
    ("triangle",               lambda: _ring(3),                                    1),
    ("square",                 lambda: _ring(4),                                    1),
    ("hexagon",                lambda: _ring(6),                                    1),
    ("two disjoint triangles", lambda: RexGraph(sources=np.array([0, 1, 2, 3, 4, 5], np.int32),
                                                targets=np.array([1, 2, 0, 4, 5, 3], np.int32)), 2),
    ("double-T",               lambda: _branching([0, 3, 6], [0, 1, 2, 0, 1, 3]),   0),
    ("h = mean of p1,p2",      lambda: _branching([0, 3, 5, 7], [0, 1, 2, 0, 1, 0, 2]), 1),
    ("3 hyperedges on a pair", lambda: _branching([0, 3, 6, 9],
                                                  [0, 1, 2, 0, 1, 3, 0, 1, 4]),     0),
    ("hyperedge + pairwise",   lambda: _branching([0, 3, 5], [0, 1, 2, 0, 1]),      0),
]


@pytest.mark.parametrize("name,build,want", CYCLE_FIXTURES)
def test_cycle_basis_dimension_is_nE_minus_rank(name, build, want):
    """The definition, not the algorithm. A traversal reports one cycle on the double-T
    because it meets the shared pair twice; the kernel does not."""
    from rexgraph.faces import cycle_basis

    rex = build()
    assert len(cycle_basis(rex)) == want, name
    assert int(rex.betti[1]) == want


@pytest.mark.parametrize("name,build,want", CYCLE_FIXTURES)
def test_every_basis_vector_is_killed_by_the_boundary(name, build, want):
    """B1 c = 0 exactly, on the traversal path and the kernel path alike."""
    from rexgraph.faces import cycle_basis

    rex = build()
    B1 = np.asarray(rex.B1, dtype=float)
    for c in cycle_basis(rex):
        img = B1 @ np.array([float(x) for x in c])
        assert np.abs(img).max() < 1e-12, name


@pytest.mark.parametrize("traversal", ["bfs", "dfs"])
def test_both_gradient_paths_give_a_basis(traversal):
    """A traversal is a gradient: it assigns a potential outward from a root and the tree
    spans im(B1^T). BFS takes that path at width, DFS at length. Either is a basis of the
    complement; they differ in which fundamental cycles come back."""
    from rexgraph.faces import cycle_basis

    rex = RexGraph(sources=np.array([0, 1, 2, 0, 3, 4], np.int32),
                   targets=np.array([1, 2, 0, 2, 4, 0], np.int32))
    basis = cycle_basis(rex, traversal=traversal)
    B1 = np.asarray(rex.B1, dtype=float)
    assert len(basis) == int(rex.betti[1])
    for c in basis:
        assert np.abs(B1 @ np.array([float(x) for x in c])).max() < 1e-12


def test_find_cycles_is_arity_general():
    """A branching relation participates in a cycle exactly when the kernel says so. The
    old pairwise-walk version could not see this one at all."""
    from rexgraph.faces import find_cycles

    rex = _branching([0, 3, 5, 7], [0, 1, 2, 0, 1, 0, 2])
    found = find_cycles(rex, 3)
    assert len(found) == 1
    assert found[0].tolist() == [0, 1, 2]              # the hyperedge is in it


def test_find_cycles_reads_the_gon_off_the_support():
    from rexgraph.faces import find_cycles

    for k in (3, 4, 5, 6):
        rex = _ring(k)
        assert len(find_cycles(rex, k)) == 1
        assert len(find_cycles(rex, k + 1)) == 0
