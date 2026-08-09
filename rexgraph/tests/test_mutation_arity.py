"""Mutations must return what was constructed, at every arity.

`insert_edges`, `delete_edges` and `subgraph` all read `(sources, targets)` and rebuilt
from it. That form holds two vertices per relation, so every mutation of a branching
complex flattened its wide relations to their first two vertices and orphaned every
vertex past them: a 4-ary relation came back 2-ary and indistinguishable from a leg, and
the complex silently lost cells it was never asked to lose.

They now go through the boundary CSR, which carries the whole column. These hold that,
the round trip (a mutation that removes nothing must return the same complex), and the
settings a mutation has no business changing.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.faces import autoface
from rexgraph.graph import RexGraph

#: a 4-ary relation over {0,1,2,3} with legs 0-1, 1-2, 2-4
_OFFSETS = np.array([0, 4, 6, 8, 10], dtype=np.int32)
_VERTICES = np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 4], dtype=np.int32)


def _arities(rex):
    return np.diff(np.asarray(rex._boundary_ptr)).tolist()


@pytest.fixture
def rex():
    r = RexGraph.from_hypergraph(_OFFSETS.copy(), _VERTICES.copy())
    r._ensure_clean()
    return r


@pytest.fixture
def filled():
    """A filled triangle with a 4-ary relation hanging off it."""
    r = RexGraph.from_hypergraph(
        np.array([0, 2, 4, 6, 10], dtype=np.int32),
        np.array([0, 1, 1, 2, 2, 0, 2, 3, 4, 5], dtype=np.int32))
    autoface(r)
    r._ensure_clean()
    return r


#### arity survives every mutation


def test_inserting_leaves_existing_arity_alone(rex):
    out = rex.insert_edges(np.array([0], np.int32), np.array([4], np.int32))
    assert _arities(out) == [4, 2, 2, 2, 2]


def test_deleting_leaves_the_survivors_alone(rex):
    mask = np.zeros(rex.nE, np.int32)
    mask[2] = 1
    assert _arities(rex.delete_edges(mask)) == [4, 2, 2]


def test_a_subcomplex_keeps_the_whole_boundary_column(rex):
    sub, _v_map, e_map = rex.subgraph(np.array([1, 0, 1, 1], bool))
    assert _arities(sub) == [4, 2, 2]
    assert e_map.tolist() == [0, 2, 3]


def test_the_orphaned_vertex_keeps_its_share(rex):
    """Vertex 3 is only in the 4-ary relation. After a mutation its whole B1 row was
    zero: not narrowed, removed from the complex."""
    out = rex.insert_edges(np.array([0], np.int32), np.array([4], np.int32))
    column = np.asarray(out.B1)[:, 0]
    assert column[3] == pytest.approx(1 / 3)
    assert column.sum() == pytest.approx(0.0), "the column stopped summing to zero"


def test_relations_of_any_arity_can_be_inserted(rex):
    out = rex.insert_relations([[0, 2, 4], [1, 3]])
    assert _arities(out) == [4, 2, 2, 2, 3, 2]


def test_an_empty_relation_is_refused(rex):
    with pytest.raises(ValueError, match="at least one boundary vertex"):
        rex.insert_relations([[]])


def test_a_mask_of_the_wrong_length_is_refused(rex):
    with pytest.raises(ValueError, match="one entry per relation"):
        rex.delete_edges(np.zeros(rex.nE + 3, np.int32))


#### round trips


def test_removing_nothing_returns_the_same_complex(rex):
    same, v_map, e_map = rex.subgraph(np.ones(rex.nE, bool))
    assert _arities(same) == _arities(rex)
    assert np.array_equal(np.asarray(same.B1), np.asarray(rex.B1))
    assert v_map.tolist() == list(range(rex.nV))
    assert e_map.tolist() == list(range(rex.nE))


def test_deleting_nothing_returns_the_same_complex(rex):
    same = rex.delete_edges(np.zeros(rex.nE, np.int32))
    assert np.array_equal(np.asarray(same.B1), np.asarray(rex.B1))


def test_delete_is_the_complement_of_subgraph(rex):
    """They are one operation. Having them as two implementations is how one of them
    stayed pairwise while the other was fixed."""
    mask = np.array([0, 1, 0, 1], np.int32)
    kept, _v, _e = rex.subgraph(~mask.astype(bool))
    assert np.array_equal(np.asarray(rex.delete_edges(mask).B1), np.asarray(kept.B1))


#### faces and settings come across


def test_a_face_dies_with_any_relation_it_bounds(filled):
    assert filled.nF == 1
    assert filled.subgraph(np.array([1, 1, 0, 1], bool))[0].nF == 0


def test_a_face_survives_when_all_its_relations_do(filled):
    sub, _v, _e = filled.subgraph(np.array([1, 1, 1, 0], bool))
    assert sub.nF == 1
    assert sub.nV == 3, "vertices left in no relation should be dropped"


def test_the_face_operator_is_remapped_not_dropped(filled):
    sub, _v, _e = filled.subgraph(np.ones(filled.nE, bool))
    assert np.array_equal(np.asarray(sub.B2), np.asarray(filled.B2))


def test_weights_come_across_and_are_sliced():
    rex = RexGraph(boundary_ptr=np.array([0, 4, 6], np.int32),
                   boundary_idx=np.array([0, 1, 2, 3, 0, 1], np.int32),
                   w_E=np.array([2.0, 5.0]))
    sub, _v, _e = rex.subgraph(np.array([0, 1], bool))
    assert np.asarray(sub._w_E).tolist() == [5.0]


def test_the_channel_selection_is_not_changed_by_a_mutation(rex):
    """A mutation should change what was asked for and nothing else."""
    assert rex.delete_edges(np.zeros(rex.nE, np.int32)).g_channel == rex.g_channel


def test_an_empty_result_is_a_complex_not_a_crash(rex):
    empty, v_map, e_map = rex.subgraph(np.zeros(rex.nE, bool))
    assert empty.nE == 0
    assert v_map.tolist() == [] and e_map.tolist() == []


#### the character follows the structure


def test_the_character_of_an_unchanged_mutation_is_unchanged(rex):
    same = rex.delete_edges(np.zeros(rex.nE, np.int32))
    assert np.asarray(same.structural_character) == pytest.approx(
        np.asarray(rex.structural_character))
