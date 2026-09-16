"""Explicit selection policies reuse full exact partition closure."""
from copy import deepcopy

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.io.catalog import object_digest
from rexgraph.io.partition_state import build_rex_partition, partition_from_policy, partition_policy, partition_tower


def fixture():
    face = [(0, 1), (1, 1), (2, -1)]
    difference = [(0, 1), (1, -1)]
    return RexGraph.from_cells([4, [[0, 1], [1, 2], [0, 2]],
        [face, face], [difference, difference], [difference]], relation_ids=[91, 17, 53])


def policy(rex, cells=None):
    return {"source_state": object_digest(rex), "cells": [[4, [0]], [0, [3]]] if cells is None else cells}


def test_full_tower_and_explicit_isolated_vertices_reuse_core(monkeypatch):
    import rexgraph.io.partition_state as core
    rex = fixture()
    expected = build_rex_partition(rex, [0, 0, 0], v_mask=[0, 0, 0, 1], grade_masks={4: [1]})
    calls = []
    def build(*args, **kwargs):
        calls.append(kwargs)
        return build_rex_partition(*args, **kwargs)
    monkeypatch.setattr(core, "build_rex_partition", build)
    p = partition_from_policy(rex, policy(rex), authority_digest="authority")
    assert len(calls) == 1
    assert p.state.result_state == expected.state.result_state
    assert p.cell_maps == expected.cell_maps
    assert p.rex.relation_ids.tolist() == [91, 17, 53]
    assert len(partition_tower(p.rex)[0]) == 4
    assert p.state.source_state == object_digest(rex)
    assert p.state.policy_digest != expected.state.policy_digest


def test_policy_order_normalization_and_authority_binding():
    rex = fixture()
    a = policy(rex, [[1, [2, 0]], [0, [3]]])
    b = policy(rex, [[0, [3]], [1, [0, 2]], [2, []]])
    before = deepcopy(a)
    assert partition_policy(rex, a) == partition_policy(rex, b)
    x = partition_from_policy(rex, a, authority_digest="a")
    y = partition_from_policy(rex, b, authority_digest="a")
    z = partition_from_policy(rex, b, authority_digest="b")
    assert x.digest == y.digest and x.digest != z.digest
    assert x.state.result_state == z.state.result_state
    assert a == before


@pytest.mark.parametrize("cells", [True, {}, [[1]], [[1, 0]], [[True, []]], [[1.0, []]],
    [[-1, []]], [[5, []]], [[1, [True]]], [[1, [np.bool_(False)]]], [[1, [0.0]]],
    [[1, [-1]]], [[1, [3]]], [[1, [0, 0]]], [[1, []], [1, []]]])
def test_invalid_selection_refused(cells):
    rex = fixture()
    with pytest.raises((ValueError, TypeError)):
        partition_from_policy(rex, policy(rex, cells))


@pytest.mark.parametrize("change", [{"closure": "projection"}, {"permissions": ["train"]},
    {"source_state": "wrong"}, {"source_state": 1}])
def test_policy_cannot_change_binding_closure_or_authority(change):
    rex = fixture()
    with pytest.raises((ValueError, TypeError)):
        partition_from_policy(rex, policy(rex) | change)


def test_empty_owned_partition_and_changed_state():
    rex = fixture()
    selected = policy(rex, [])
    result = partition_from_policy(rex, selected)
    assert result.rex is not rex and result.rex.nV == result.rex.nE == 0
    rex.add_edges([0], [1], relation_ids=[101])
    with pytest.raises(ValueError, match="source_state"):
        partition_from_policy(rex, selected)


def test_branching_witness_self_loop_signs_and_weights():
    rex = RexGraph.from_cells([5, [[0, 1, 2, 3], [3], [4, 4]]],
        relation_ids=[10, 20, 30], w_E=[2, 3, 4], signs=[-1, 1, -1])
    p = partition_from_policy(rex, policy(rex, [[1, [0, 1, 2]]]))
    assert p.state.result_state == object_digest(rex)
    assert p.cell_maps == ((0, 1, 2, 3, 4), (0, 1, 2))
    assert p.rex is not rex


def test_raw_invalid_chain_is_not_filtered():
    rex = RexGraph(sources=np.array([0, 1], dtype=np.int32),
        targets=np.array([1, 2], dtype=np.int32),
        B2_col_ptr=np.array([0, 1], dtype=np.int32), B2_row_idx=np.array([0], dtype=np.int32),
        B2_vals=np.array([1.0]))
    with pytest.raises(ValueError, match="chain condition"):
        partition_from_policy(rex, policy(rex, []))
