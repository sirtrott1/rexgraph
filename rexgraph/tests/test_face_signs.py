"""Solving grade-2 signs from the chain condition, and reading the sign context.

A face column is solved, not declared: `B1 c = 0` over the rationals. Two things follow
that were not being said.

`solve_face_column` answers with a column or with None, and None covers two different
situations. Relations that are independent bound nothing and attaching a face would invent
a cell; relations carrying several cycles are not one face but a space of them. Those need
different responses and used to get the same one.

A wrong orientation is invisible. `_B2_hodge_dual` filters chain-invalid faces silently,
so nF_hodge stays 0, the cycle stays open, and nothing says why. `face_reading` with a
column reports validity and the exact residual instead.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.faces import (
    auto_hyperface,
    face_reading,
    find_hyperface_groups,
    solve_face_basis,
    solve_face_column,
)
from rexgraph.graph import RexGraph


def _triangle():
    return RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                    targets=np.array([1, 2, 0], dtype=np.int32))


def _reversed_triangle():
    return RexGraph(sources=np.array([1, 1, 2], dtype=np.int32),
                    targets=np.array([0, 2, 0], dtype=np.int32))


def _path():
    return RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                    targets=np.array([1, 2, 3], dtype=np.int32))


def _k4():
    return RexGraph(sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
                    targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32))


def _wide():
    """A 4-ary relation over {0,1,2,3} with the 4-cycle of legs that spans it."""
    return RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10, 12], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32))


#### the three states


def test_a_cycle_bounds_and_its_column_is_determined():
    out = face_reading(_triangle(), [0, 1, 2])
    assert out["state"] == "bounds"
    assert out["nullity"] == 1
    assert out["column"] == ["1", "1", "1"]
    assert out["chain_valid"] is True


def test_independent_relations_are_open_not_merely_unsolvable():
    """A path bounds nothing. Attaching a face here would invent a cell."""
    out = face_reading(_path(), [0, 1, 2])
    assert out["state"] == "open"
    assert out["nullity"] == 0
    assert out["column"] is None
    assert "bound nothing" in out["reason"]


def test_several_cycles_are_degenerate_not_one_face():
    """K4 carries three independent cycles. Its six relations are not a face."""
    out = face_reading(_k4(), [0, 1, 2, 3, 4, 5])
    assert out["state"] == "degenerate"
    assert out["nullity"] == 3 == _k4().betti[1]
    assert "3 independent cycles" in out["reason"]
    assert "solve_face_basis" in out["reason"]


#### the sign context


def test_reversal_moves_the_signs_not_the_support():
    """Same support, different orientation: the same cell to homology, a different one
    here. Existence says there is a cycle, the sign says how it closes."""
    a, b = face_reading(_triangle(), [0, 1, 2]), face_reading(_reversed_triangle(),
                                                              [0, 1, 2])
    assert a["support"] == b["support"] == 3
    assert a["column"] != b["column"]
    assert b["column"] == ["1", "-1", "-1"]
    assert b["reversed_relations"] == [1, 2]


def test_both_orientations_close_so_the_holonomy_reads_balanced():
    """Two negatives is an even count. n_reversed distinguishes the complexes, the
    holonomy says they are the same balanced class, and both are true."""
    assert face_reading(_triangle(), [0, 1, 2])["holonomy"] == 1
    assert face_reading(_reversed_triangle(), [0, 1, 2])["holonomy"] == 1


def test_the_gon_is_the_support_not_the_number_of_relations_given():
    out = face_reading(_wide(), [0, 1, 2, 3, 4])
    assert out["gon"] == 5
    assert out["support"] < out["gon"], "a zero coefficient is not a side"


#### explicit signs are checked instead of silently dropped


def test_wrong_signs_are_reported_with_the_residual():
    out = face_reading(_triangle(), [0, 1, 2], column=[1, -1, 1])
    assert out["chain_valid"] is False
    assert out["residual"] == ["2", "-2"]
    assert out["solved_column"] == ["1", "1", "1"]
    assert "dropped by the chain filter" in out["reason"]


def test_right_signs_pass_with_no_residual():
    out = face_reading(_triangle(), [0, 1, 2], column=[1, 1, 1])
    assert out["chain_valid"] is True and out["residual"] == []


def test_the_overall_sign_does_not_matter():
    """A face and its reverse are the same cell."""
    assert face_reading(_triangle(), [0, 1, 2], column=[-1, -1, -1])["chain_valid"]


def test_a_column_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="one coefficient per relation"):
        face_reading(_triangle(), [0, 1, 2], column=[1, 1])


#### a group bounds a space of faces, and all of it gets attached


def test_the_basis_has_one_column_per_independent_cycle():
    rex = _wide()
    group = find_hyperface_groups(rex)[0]
    assert face_reading(rex, group)["nullity"] == 2
    assert len(solve_face_basis(rex, group)) == 2


def test_attaching_the_whole_basis_closes_the_group():
    """One arbitrary vector left b1 = 1 and a face that claimed five relations while using
    four. The basis leaves nothing half-filled."""
    rex = _wide()
    auto_hyperface(rex)
    rex._ensure_clean()
    assert rex.nF == 2
    assert rex.nF_hodge == 2, "a solved face was dropped by the chain filter"
    assert rex.betti[1] == 0
    residual = np.abs(np.asarray(rex.B1) @ np.asarray(rex.B2)).max()
    assert residual == 0.0, "B1 B2 = 0 must hold exactly, not approximately"


def test_a_single_cycle_still_attaches_exactly_one_face():
    """The nullity-1 case is the common one and must not have moved."""
    rex = _triangle()
    from rexgraph.faces import autoface
    autoface(rex)
    rex._ensure_clean()
    assert rex.nF == 1


def test_the_first_basis_column_is_what_the_single_solver_returns():
    rex = _wide()
    group = find_hyperface_groups(rex)[0]
    assert solve_face_basis(rex, group)[0] == solve_face_column(rex, group)


def test_relations_that_bound_nothing_contribute_no_face():
    assert solve_face_basis(_path(), [0, 1, 2]) == []


def test_every_solved_column_satisfies_the_chain_condition_exactly():
    rex = _wide()
    group = find_hyperface_groups(rex)[0]
    for column in solve_face_basis(rex, group):
        out = face_reading(rex, group, column=column)
        assert out["chain_valid"] is True
        assert all(isinstance(x, Fraction) for x in column)
