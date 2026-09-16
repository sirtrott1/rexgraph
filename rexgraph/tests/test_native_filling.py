"""Exact cycle attachment preserves native state and the complete tower."""
from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest
from rexgraph.io.partition_state import partition_tower
from rexgraph.io.rex_state import from_state, to_state
from rexgraph.native_sparse import sparse_arrays


def tower():
    face = [(0, 2), (1, -1), (2, -1)]
    difference = [(0, 1), (1, -1)]
    rex = RexGraph.from_cells([5, [[0, 1, 2], [0, 1], [0, 2]],
        [face, face], [difference, difference], [difference]],
        relation_ids=[17, 4, 93], w_E=np.array([Fraction(1, 3), 2, 7], dtype=object),
        signs=[-1, 1, -1], g_channel="raw")
    rex._c_channel = "count"
    return rex


def test_full_tower_state_and_owned_storage(monkeypatch):
    rex = tower()
    rex._agent_meta = {"vertex_labels": ["a", "b", "c", "d", "e"], "title": "sample"}
    rex._signals = np.array([1., 2., 3.])
    rex._w_boundary = {(0, 0): np.array([3., 7.])}
    before = object_digest(rex)
    monkeypatch.setattr(RexGraph, "B2", property(lambda self: pytest.fail("dense B2 read")))
    result = rex.fill_cycle([2, -1, -1])
    assert object_digest(rex) == before and result is not rex
    assert (result.nV, result.nE, result.nF) == (5, 3, 3)
    assert result.relation_ids.tolist() == [17, 4, 93]
    assert result.edge_metric_exact == rex.edge_metric_exact
    np.testing.assert_array_equal(result._signs, rex._signs)
    assert result._g_channel == "raw" and result._c_channel == "count"
    assert result._agent_meta == rex._agent_meta
    np.testing.assert_array_equal(result._signals, rex._signals)
    for grade, (old, new) in enumerate(zip(rex._graded_duals, result._graded_duals, strict=True)):
        a, b = sparse_arrays(old), sparse_arrays(new)
        assert b[3] == (a[3][0] + (grade == 0), a[3][1])
        for index in (1, 2):
            np.testing.assert_array_equal(a[index], b[index])
        assert b[2].dtype == a[2].dtype
    partition_tower(result)
    restored = from_state(to_state(result))
    assert object_digest(restored) == object_digest(result)
    result._agent_meta["title"] = "changed"
    result._signals[0] = 99
    result._w_boundary[(0, 0)][0] = 99
    assert rex._agent_meta["title"] == "sample" and rex._signals[0] == 1
    assert rex._w_boundary[(0, 0)][0] == 3


@pytest.mark.parametrize("cells,cycle,cycles", [
    ([[0, 1, 2], [0, 1], [0, 2]], [2, -1, -1], 1),
    ([[0], [0]], [1, -1], 1),
    ([[0, 0]], [1], 1),
    ([[0, 1], [1, 2], [0, 2]], [1, 1, -1], 1),
])
def test_shadow_and_filling_agree_for_primary_relations(cells, cycle, cycles):
    rex = RexGraph.from_cells([3, cells])
    assert rex.harmonic_shadow == {"shadow_dim": 0, "beta_1_at_d1": cycles, "beta_1_at_d2": cycles}
    child = rex.fill_cycle(cycle)
    assert child.harmonic_shadow == {"shadow_dim": 1, "beta_1_at_d1": cycles, "beta_1_at_d2": 0}
    duplicate = child.fill_cycle(cycle)
    assert duplicate.nF == 2 and duplicate.harmonic_shadow == child.harmonic_shadow


@pytest.mark.parametrize("cycle", [[], [0, 0, 0], [1, 0, 0], [[2], [-1], [-1]],
    [Fraction(1, 2), Fraction(-1, 4), Fraction(-1, 4)], [True, 0, 0],
    [float("nan"), 0, 0], [float("inf"), 0, 0], [2j, -1, -1],
    [2*(2**53+1), -(2**53+1), -(2**53+1)]])
def test_bad_filling_never_changes_source(cycle):
    rex = tower()
    before = object_digest(rex)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        rex.fill_cycle(cycle)
    assert object_digest(rex) == before


def test_invalid_stored_face_is_not_silently_filtered_for_attachment():
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]]])
    rex.add_faces([[0]], [[1]])
    with pytest.raises(ValueError, match="chain condition"):
        rex.fill_cycle([1, 1, -1])


def test_shadow_handles_empty_and_top_grades():
    assert RexGraph.from_cells([3, []]).harmonic_shadow == {
        "shadow_dim": 0, "beta_1_at_d1": 0, "beta_1_at_d2": 0}
    assert tower().harmonic_shadow == {"shadow_dim": 1, "beta_1_at_d1": 1, "beta_1_at_d2": 0}
