"""Arity, parity and degree: the three directions of a graded complex.

Arity looks DOWN a grade, degree looks UP, parity reads the orientation. The claim these
pin is that none of the three is derived from another, and that the three are exactly the
canon's separable axes (share, existence, orientation) read per cell instead of per
channel.

The parity result is the one worth stating, because it was measured rather than assumed.
At grade 1 the sign PRODUCT is constant: a canonical `B_1` column is
`(-1, +share, ..., +share)`, exactly one negative whatever the arity and whatever vertex
is distinguished, so it reads -1 for every relation and reversing an edge does not move
it. Parity only carries information from grade 2 up, where the coefficients are solved:
a triangle reads `[1, 1, 1]` and the same triangle with one relation reversed reads
`[1, -1, -1]`. There the product is the holonomy and `n_negative` counts the relations
running against their stored orientation.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.faces import auto_hyperface, autoface
from rexgraph.graph import RexGraph
from rexgraph.tower import apd, surface_identity


def _triangle():
    return RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                    targets=np.array([1, 2, 0], dtype=np.int32))


def _square():
    return RexGraph(sources=np.array([0, 1, 2, 3], dtype=np.int32),
                    targets=np.array([1, 2, 3, 0], dtype=np.int32))


def _branching():
    """One 4-ary relation with two 2-ary legs, so arity varies within a grade."""
    return RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2], dtype=np.int32))


#### the three axes are independent


def test_arity_reads_down_and_degree_reads_up():
    """Arity is how many cells this one bounds, degree is how many contain it. On a bare
    triangle every relation has arity 2 and degree 0; filling it moves degree only."""
    bare = apd(_triangle(), 1)["cells"]
    assert [c["arity"] for c in bare] == [2, 2, 2]
    assert [c["degree"] for c in bare] == [0, 0, 0]

    filled = _triangle()
    autoface(filled)
    cells = apd(filled, 1)["cells"]
    assert [c["arity"] for c in cells] == [2, 2, 2], "filling changed a DOWN reading"
    assert all(c["degree"] == 1 for c in cells), "the face is not seen from below"


def test_arity_varies_within_a_grade_on_a_branching_complex():
    """A wide relation and a 2-ary leg live at the same grade, which is the whole reason
    arity is a per-cell reading rather than a property of the complex."""
    cells = apd(_branching(), 1)["cells"]
    assert sorted(c["arity"] for c in cells) == [2, 2, 4]


def test_a_cell_can_be_wide_and_lonely_or_narrow_and_busy():
    """Independence, stated as the thing that would break if arity determined degree.

    A filled triangle with a 4-ary relation hanging off it: the narrow relations are the
    busy ones and the wide relation is in nothing, so arity runs OPPOSITE to degree here.
    Any rule deriving one from the other has to pick a direction, and this complex would
    falsify either choice.
    """
    rex = RexGraph.from_hypergraph(
        np.array([0, 2, 4, 6, 10], dtype=np.int32),
        np.array([0, 1, 1, 2, 2, 0, 2, 3, 4, 5], dtype=np.int32))
    autoface(rex)
    pairs = [(c["arity"], c["degree"]) for c in apd(rex, 1)["cells"]]
    assert (2, 1) in pairs, "the narrow relations are not the busy ones"
    assert (4, 0) in pairs, "the wide relation is not the lonely one"


#### parity: measured, not assumed


def test_at_grade_one_the_sign_product_is_constant():
    """The canonical column is (-1, +share, ..., +share): exactly one negative at every
    arity, so the product is -1 for every relation and cannot distinguish anything."""
    for build in (_triangle, _square, _branching):
        cells = apd(build(), 1)["cells"]
        assert {c["parity"] for c in cells} == {-1}
        assert {c["n_negative"] for c in cells} == {1}


def test_reversing_a_relation_does_not_move_grade_one_parity():
    """Reversal moves WHICH vertex is distinguished, not how many are negative."""
    a = _triangle()
    b = RexGraph(sources=np.array([1, 1, 2], dtype=np.int32),
                 targets=np.array([0, 2, 0], dtype=np.int32))
    assert ([c["parity"] for c in apd(a, 1)["cells"]]
            == [c["parity"] for c in apd(b, 1)["cells"]])


def test_grade_one_says_so_rather_than_reporting_a_constant_as_a_reading():
    assert apd(_triangle(), 1)["parity_informative"] is False
    assert apd(_triangle(), 1, view="global")["balanced"] is None


def test_parity_carries_information_from_grade_two():
    """Where the coefficients are solved rather than canonical, the signs vary."""
    rex = _triangle()
    autoface(rex)
    face = apd(rex, 2)["cells"][0]
    assert apd(rex, 2)["parity_informative"] is True
    assert face["arity"] == 3
    assert face["n_negative"] == 0

    rev = RexGraph(sources=np.array([1, 1, 2], dtype=np.int32),
                   targets=np.array([0, 2, 0], dtype=np.int32))
    autoface(rev)
    other = apd(rev, 2)["cells"][0]
    assert other["n_negative"] == 2, "reversal is invisible at grade 2 as well"
    assert other["n_negative"] != face["n_negative"]


def test_the_sign_product_is_the_holonomy_so_both_read_balanced():
    """Two negatives is an even count: the cycle still closes. n_negative distinguishes
    the two complexes, parity says they are the same balanced class, and both are true."""
    rex, rev = _triangle(), RexGraph(sources=np.array([1, 1, 2], dtype=np.int32),
                                     targets=np.array([0, 2, 0], dtype=np.int32))
    autoface(rex)
    autoface(rev)
    assert apd(rex, 2)["cells"][0]["parity"] == 1
    assert apd(rev, 2)["cells"][0]["parity"] == 1
    assert apd(rex, 2, view="global")["balanced"] is True
    assert apd(rev, 2, view="global")["n_frustrated"] == 0


#### the global view is the same operator, and it is the identity's terms


def test_the_global_means_are_the_terms_of_the_surface_identity():
    """`a` is mean arity at grade 1 and `c` is mean degree at grade 1, exactly, so the
    identity a/d + c/k = 1 + chi/E is an APD statement about consecutive grades."""
    rex = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10, 12], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32))
    auto_hyperface(rex)
    identity = surface_identity(rex)
    g1 = apd(rex, 1, view="global")
    assert g1["mean_arity"] == identity["mean_arity"]
    assert g1["mean_degree"] == identity["mean_closure"]


def test_the_means_are_exact_rationals():
    g = apd(_branching(), 1, view="global")
    assert Fraction(g["mean_arity"]) == Fraction(8, 3)
    assert "." not in g["mean_arity"], "a mean was rounded on the way out"


def test_the_global_view_is_the_mean_of_the_local_one():
    rex = _branching()
    auto_hyperface(rex)
    cells = apd(rex, 1)["cells"]
    g = apd(rex, 1, view="global")
    assert Fraction(g["mean_arity"]) == Fraction(sum(c["arity"] for c in cells),
                                                 len(cells))
    assert Fraction(g["mean_degree"]) == Fraction(sum(c["degree"] for c in cells),
                                                  len(cells))


#### edges of the operator


def test_an_absent_grade_reports_absence_rather_than_an_empty_reading():
    out = apd(_triangle(), 2)
    assert out["cells"] == [] and "reason" in out


def test_grade_zero_is_refused_because_vertices_bound_nothing():
    assert apd(_triangle(), 0)["cells"] == []


def test_an_unknown_view_is_refused():
    with pytest.raises(ValueError, match="view must be"):
        apd(_triangle(), 1, view="both")


#### a stored zero is not a side


def test_a_solved_face_column_with_a_stored_zero_reports_its_gon():
    """A nullspace basis vector over a group carries an explicit zero for a relation it
    does not use, and `[3,-3,-2,-1,0]` is stored with five entries. Counting stored
    entries made arity the number of relations OFFERED rather than the gon, so apd said 5
    where `face_support` and `surface_identity` said 4."""
    rex = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10, 12], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32))
    auto_hyperface(rex)
    rex._ensure_clean()
    stored = [int((np.asarray(rex.B2)[:, f] != 0).sum()) for f in range(rex.nF)]
    assert [c["arity"] for c in apd(rex, 2)["cells"]] == stored


def test_the_grade_two_means_still_match_the_identity():
    """The claim the operator is built on, on a complex where the columns carry zeros."""
    rex = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10, 12], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32))
    auto_hyperface(rex)
    rex._ensure_clean()
    assert apd(rex, 2, view="global")["mean_arity"] == surface_identity(rex)["mean_face_size"]


def test_parity_is_not_zeroed_by_a_stored_zero():
    """np.sign of a stored zero is 0, so the product collapsed to 0 and the face read
    neither balanced nor frustrated."""
    rex = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10, 12], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32))
    auto_hyperface(rex)
    rex._ensure_clean()
    assert all(c["parity"] in (1, -1) for c in apd(rex, 2)["cells"])
