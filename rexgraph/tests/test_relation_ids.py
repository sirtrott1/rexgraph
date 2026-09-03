"""Exact C1 instance identity: no support-key collapse in temporal deltas."""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph, TemporalRex, apply_edge_delta
from rexgraph.io.rex_state import from_state, to_state
from rexgraph.io.temporal_state import from_temporal_state, to_temporal_state
from rexgraph.temporal_signal import temporal_signal


def _parallel_snapshot(ids):
    """Four C1 cells, with the first two deliberately sharing one support."""
    return RexGraph.from_hypergraph(
        np.asarray([0, 2, 4, 6, 8], dtype=np.int32),
        np.asarray([0, 1, 0, 1, 2, 3, 3, 4], dtype=np.int32),
        relation_ids=np.asarray(ids, dtype=np.int64),
    )


def test_relation_ids_are_exact_validated_and_survive_compaction_and_state():
    with pytest.raises(ValueError, match="unique"):
        _parallel_snapshot([1, 1, 2, 3])
    with pytest.raises(ValueError, match="integral"):
        RexGraph(sources=[0], targets=[1], relation_ids=np.asarray([1.5]))

    rex = _parallel_snapshot([10, 11, 12, 13])
    rex.remove_edges(np.asarray([1, 0, 0, 0], dtype=np.int32))
    rex.compact()
    rex.add_edges([4], [5], relation_ids=np.asarray([14], dtype=np.int64))
    assert rex.relation_ids.tolist() == [11, 12, 13, 14]

    rebuilt = from_state(to_state(rex))
    assert rebuilt.relation_ids.tolist() == [11, 12, 13, 14]


def test_parallel_equal_support_relations_keep_independent_id_histories():
    previous = _parallel_snapshot([101, 102, 103, 104])
    # ID 101 dies while a distinct ID 105 is born on the identical [0, 1]
    # boundary.  A support key sees no change; the exact identity delta sees both.
    current = _parallel_snapshot([102, 105, 103, 104])
    history = TemporalRex([])
    history.append_snapshot(previous)
    history.append_snapshot(current)

    delta = history._index_deltas[1]
    assert delta is not None
    assert delta.died_ids.tolist() == [101]
    assert delta.born_ids.tolist() == [105]

    rebuilt = history.reconstruct_at(1)
    assert rebuilt.relation_ids.tolist() == [102, 103, 104, 105]
    assert rebuilt.relation_supports().count([0, 1]) == 2

    replay = _parallel_snapshot([101, 102, 103, 104])
    apply_edge_delta(replay, delta)
    assert set(replay.relation_ids.tolist()) == {102, 103, 104, 105}
    assert replay.relation_supports().count([0, 1]) == 2

    events = history.delta_tensor()
    assert dict(zip(events["key"], events["existence"], strict=True)) == {101: -1, 105: 1}
    signal = temporal_signal(history, 1)
    assert signal.event(101).existence == -1
    assert signal.event(105).existence == 1
    assert history.edge_lifecycle[0].tolist() == [101, 102, 103, 104, 105]
    counts, born, died = history.edge_metrics
    assert counts.tolist() == [4, 4]
    assert born.tolist() == [0, 1]
    assert died.tolist() == [0, 1]

    restored = from_temporal_state(to_temporal_state(history))
    restored_events = restored.delta_tensor()
    assert dict(zip(restored_events["key"], restored_events["existence"], strict=True)) == {
        101: -1, 105: 1
    }
    assert restored.reconstruct_at(1).relation_ids.tolist() == [102, 103, 104, 105]


def test_tuple_temporal_constructor_accepts_complete_relation_id_vectors():
    history = TemporalRex(
        [
            (np.asarray([0, 0]), np.asarray([1, 1])),
            (np.asarray([0, 0]), np.asarray([1, 1])),
        ],
        relation_ids=[np.asarray([7, 8], dtype=np.int64), np.asarray([8, 9], dtype=np.int64)],
    )
    assert history.at(0).relation_ids.tolist() == [7, 8]
    assert history.at(1).relation_ids.tolist() == [8, 9]
    counts, born, died = history.edge_metrics
    assert counts.tolist() == [2, 2]
    assert born.tolist() == [0, 1]
    assert died.tolist() == [0, 1]


def test_relation_ids_round_trip_through_bundle_hdf5_zarr_and_safetensors(tmp_path):
    history = TemporalRex(
        [
            (np.asarray([0, 0]), np.asarray([1, 1])),
            (np.asarray([0, 0]), np.asarray([1, 1])),
        ],
        relation_ids=[np.asarray([31, 32], dtype=np.int64), np.asarray([32, 33], dtype=np.int64)],
    )

    from rexgraph.io.bundle import load_rex, save_rex
    bundle = tmp_path / "identity.rex"
    save_rex(str(bundle), history)
    assert load_rex(str(bundle)).at(1).relation_ids.tolist() == [32, 33]

    from rexgraph.io.safetensors_bridge import (
        safetensors_to_temporal_rex,
        temporal_rex_to_safetensors,
    )
    tensor_path = tmp_path / "identity.safetensors"
    temporal_rex_to_safetensors(history, tensor_path)
    assert safetensors_to_temporal_rex(tensor_path).reconstruct_at(1).relation_ids.tolist() == [32, 33]

    pytest.importorskip("h5py")
    from rexgraph.io.hdf5_format import load_hdf5, save_hdf5
    h5_path = tmp_path / "identity.h5"
    save_hdf5(str(h5_path), history)
    assert load_hdf5(str(h5_path)).at(1).relation_ids.tolist() == [32, 33]

    pytest.importorskip("zarr")
    from rexgraph.io.zarr_format import load_zarr, save_zarr
    zarr_path = tmp_path / "identity.zarr"
    save_zarr(str(zarr_path), history)
    assert load_zarr(str(zarr_path)).at(1).relation_ids.tolist() == [32, 33]
