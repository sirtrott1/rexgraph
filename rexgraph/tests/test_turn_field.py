"""Conversation capture preserves primary turns and preview leaves no mutation."""
from copy import deepcopy

import numpy as np
import pytest

from rexgraph.flow.turn_field import TurnField
from rexgraph.flow.gate import MalaughGate, malaugh_entropy


def conversation():
    field = TurnField()
    for text in ["alpha beta gamma", "gamma delta", "alpha beta gamma"]:
        field.observe(text)
    return field


def test_parallel_turns_keep_distinct_ids_and_full_support():
    field = conversation()
    history = field.snapshot()
    assert history.T == 3
    for step in range(3):
        rex = history.reconstruct_at(step)
        assert rex.nE == step + 1
        assert rex.relation_ids.tolist() == list(range(step + 1))
    rex = history.reconstruct_at(2)
    assert np.diff(rex._boundary_ptr).tolist() == [3, 2, 3]
    assert rex._agent_meta["vertex_labels"] == ["alpha", "beta", "gamma", "delta"]


def test_preview_matches_observation_without_advancing_baseline():
    field = conversation()
    before = deepcopy((field._turns, field._vocab, field._gate.__dict__, field._last))
    rex = field.rex
    preview = field.preview("epsilon zeta")
    assert preview == field.preview("epsilon zeta")
    assert (field._turns, field._vocab, field._gate.__dict__, field._last) == before
    assert field.rex is rex
    assert field.observe("epsilon zeta") == preview


def test_failed_observation_does_not_partially_append(monkeypatch):
    import rexgraph.scale_propagator as scale
    field = conversation()
    before = deepcopy((field._turns, field._vocab, field._gate.__dict__, field._last))
    def fail(*args, **kwargs):
        raise RuntimeError("kernel failed")
    monkeypatch.setattr(scale, "malaugh_quantities", fail)
    with pytest.raises(RuntimeError, match="kernel failed"):
        field.observe("new unseen terms")
    assert (field._turns, field._vocab, field._gate.__dict__, field._last) == before


def test_empty_and_witness_turns_have_explicit_status():
    field = TurnField()
    assert field.snapshot().T == 0
    assert field.observe("...")["status"] == "no_terms"
    out = field.observe("alone")
    assert field.n_turns == 1 and field.rex.nE == 1
    assert out["status"] in {"observed", "undefined_entropy"}
    assert field.snapshot().reconstruct_at(0).nV == 1


def test_snapshot_is_owned_and_interval_retains_prefixes():
    field = conversation()
    history = field.snapshot(1, 3)
    assert history.T == 2 and history.time_at(0) == 1
    assert history.reconstruct_at(0).nE == 2
    field.observe("fresh terms")
    assert history.T == 2 and history.reconstruct_at(1).nE == 3
    assert field.snapshot(2, 2).T == 0


def test_gate_scalar_entry_uses_the_same_rule():
    field = conversation()
    a, b = MalaughGate(), MalaughGate()
    for step in range(3):
        rex = field.snapshot().reconstruct_at(step)
        assert a.observe(rex) == b.observe_entropy(malaugh_entropy(rex))


@pytest.mark.parametrize("start,stop", [(-1, None), (0, 4), (2, 1), (True, 2), (0, 1.5)])
def test_invalid_interval(start, stop):
    with pytest.raises((TypeError, ValueError)):
        conversation().snapshot(start, stop)


@pytest.mark.parametrize("kwargs", [{"fence_k": -1}, {"fence_k": float("nan")},
                                    {"warmup": True}, {"warmup": 0}, {"warmup": 1.5}])
def test_invalid_gate_configuration(kwargs):
    with pytest.raises(ValueError):
        TurnField(**kwargs)


def test_agent_import_is_the_core_class():
    module = pytest.importorskip("agent.turn_field")
    assert module.TurnField is TurnField


def test_pairwise_first_turn_can_grow_into_branching_history():
    field = TurnField()
    field.observe("alpha beta")
    field.observe("beta gamma delta")
    assert np.diff(field.snapshot().at(1).boundary_ptr).tolist() == [2, 3]


def test_temporal_labels_are_owned_and_sealed():
    from rexgraph.io.temporal_state import to_temporal_state, from_temporal_state, verify_temporal_state
    history = conversation().snapshot()
    state = to_temporal_state(history)
    restored = from_temporal_state(state)
    for step in range(history.T):
        assert restored.at(step)._agent_meta == history.at(step)._agent_meta
    restored.at(0)._agent_meta["vertex_labels"][0] = "detached"
    assert restored.at(0)._agent_meta["vertex_labels"][0] == "alpha"
    state.header["vertex_labels"][0][0] = "tampered"
    assert not verify_temporal_state(state)
    assert restored.at(0)._agent_meta["vertex_labels"][0] == "alpha"


@pytest.mark.parametrize("labels", [[], {}, [1, None, None], [[3], None, None]])
def test_temporal_label_schema_is_checked(labels):
    from rexgraph.io.temporal_state import to_temporal_state, from_temporal_state
    state = to_temporal_state(conversation().snapshot())
    state.header["vertex_labels"] = labels
    with pytest.raises(ValueError, match="vertex_labels"):
        from_temporal_state(state, verify=False)


def test_labels_are_optional_for_existing_histories():
    from rexgraph.graph import RexGraph, TemporalRex
    from rexgraph.io.temporal_state import to_temporal_state, from_temporal_state
    history = TemporalRex([])
    rex = RexGraph.from_graph([0], [1])
    history.append_snapshot(rex)
    plain = to_temporal_state(history)
    assert "vertex_labels" not in plain.header
    assert from_temporal_state(plain).at(0).nE == 1
    rex._agent_meta = {"vertex_labels": ["a", "b"]}
    history.append_snapshot(rex)
    rex._agent_meta["vertex_labels"][0] = "changed"
    restored = from_temporal_state(to_temporal_state(history))
    assert not hasattr(restored.at(0), "_agent_meta")
    assert restored.at(1)._agent_meta["vertex_labels"] == ["a", "b"]
