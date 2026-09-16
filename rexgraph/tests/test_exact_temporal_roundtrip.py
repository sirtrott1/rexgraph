"""Exact temporal deltas, canonical containers and RCQL commit acceptance."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.graph import RexGraph, TemporalRex, apply_edge_delta
from rexgraph.io import load, save
from rexgraph.io.catalog import object_digest
from rexgraph.io.mutation import prepare_mutation, verify_mutation, mutation_from_bytes, mutation_to_bytes
from rexgraph.io.temporal_state import from_temporal_state, to_temporal_state, verify_temporal_state


def state(weights, ids=False):
    return RexGraph(boundary_ptr=np.array([0, 4, 6], np.int32),
                    boundary_idx=np.array([2, 0, 1, 3, 0, 2], np.int32),
                    relation_ids=np.array([11, 12]) if ids else None,
                    w_E=np.asarray(weights), signs=[1, -1], c_channel="count")


@pytest.mark.parametrize("ids", [False, True])
@pytest.mark.parametrize("weights,next_weights", [
    ([Q(2, 3), Q(-5, 7)], [Q(2, 3) + Q(1, 2**70), Q(-5, 7)]),
    ([2**60, 2**60+1], [2**60+1, 2**60+2]),
])
def test_exact_delta_roundtrip_and_mutation(ids, weights, next_weights):
    first, second = state(weights, ids), state(next_weights, ids)
    history = TemporalRex([], general=True)
    history._checkpoint_threshold = 100
    history.append_snapshot(first, at=10)
    history.append_snapshot(second, at=20)
    assert history._index_deltas[1] is not None  # exercise replay, not only checkpoints
    delta = history._index_deltas[1]
    assert sorted(delta.mod_wE) == sorted(after for before, after in zip(weights, next_weights, strict=True) if before != after)
    restored = from_temporal_state(to_temporal_state(history))
    assert object_digest(restored) == object_digest(history)
    for step, expected in enumerate((first, second)):
        np.testing.assert_array_equal(restored.reconstruct_at(step).w_E, expected.w_E)
    apply_edge_delta(first, delta)
    np.testing.assert_array_equal(first.w_E, second.w_E)
    first = state(weights, ids)
    package = prepare_mutation(first, second, tx_time=20)
    decoded = mutation_from_bytes(mutation_to_bytes(package))
    assert verify_mutation(decoded, previous=first)


@pytest.mark.parametrize("suffix", ["rcbd", "safetensors", "h5", "zarr"])
@pytest.mark.parametrize("lazy", [False, True])
def test_canonical_temporal_formats_keep_exact_metadata(tmp_path, suffix, lazy):
    pytest.importorskip({"rcbd": "safetensors", "h5": "h5py", "zarr": "zarr", "safetensors": "safetensors"}[suffix])
    history = TemporalRex([], general=True)
    history._checkpoint_threshold = 100
    history.append_snapshot(state([Q(2, 3), Q(5, 7)], True), at=3)
    history.append_snapshot(state([Q(7, 11), Q(-2, 13)], True), at=8)
    if lazy:
        history = from_temporal_state(to_temporal_state(history))
        assert not history._snapshots_materialized
    before = object_digest(history)
    path = str(tmp_path / f"timeline.{suffix}")
    save(path, history)
    restored = load(path)
    assert object_digest(restored) == before
    np.testing.assert_array_equal(restored.times, [3, 8])
    for step in range(2):
        a, b = history.reconstruct_at(step), restored.reconstruct_at(step)
        assert object_digest(a) == object_digest(b)


def test_v2_numeric_identity_is_unchanged_and_v3_codec_is_sealed():
    numeric = TemporalRex([])
    numeric.append_snapshot(RexGraph.from_graph([0], [1]), at=1)
    st = to_temporal_state(numeric)
    assert st.header["temporal_state_version"] == 2
    assert verify_temporal_state(st)
    exact = TemporalRex([], general=True)
    exact.append_snapshot(state([Q(1, 3), Q(2, 7)]))
    st = to_temporal_state(exact)
    assert st.header["temporal_state_version"] == 3
    assert verify_temporal_state(st)
    next(iter(st.header["tensor_codecs"].values()))["shape"] = [1]
    assert not verify_temporal_state(st)


def test_appended_snapshot_does_not_alias_live_weights():
    rex = state([Q(1, 3), Q(2, 7)])
    history = TemporalRex([], general=True)
    history.append_snapshot(rex)
    before = object_digest(history)
    rex.w_E[0] = Q(99)
    assert object_digest(history) == before


@pytest.mark.parametrize("weights", [
    (2**100, 2**100+1),
    (Q(1, 3), Q(1, 3)+Q(1, 2**100)),
    (2**2000, 2**2000+1),
])
def test_signal_event_detection_preserves_exact_weight_changes(weights):
    from rexgraph.temporal_signal import temporal_signal
    history = TemporalRex([], general=True)
    for weight in weights:
        history.append_snapshot(state([weight, Q(1)], True))
    signal = temporal_signal(history, 1)
    assert len(signal.events) == 1
    event = signal.event(11)
    assert event.previous_amplitude == weights[0] and event.amplitude == weights[1]
    assert not event.existence and not event.orientation and not event.signing


@pytest.mark.parametrize("kind", ["hdf5", "zarr", "rcbd", "safetensors"])
def test_legacy_snapshot_readers_remain_compatible(tmp_path, kind):
    """Handwritten old layout; intentionally does not call the new writer."""
    source, target = np.array([0], np.int32), np.array([1], np.int32)
    if kind in {"hdf5", "zarr"}:
        if kind == "hdf5":
            h5py = pytest.importorskip('h5py')
            from rexgraph.io.hdf5_format import RexHDF5Format
            group = h5py.File(tmp_path / 'legacy.h5', 'w')
            reader = RexHDF5Format()
        else:
            zarr = pytest.importorskip('zarr')
            from rexgraph.io.zarr_format import RexZarrFormat
            group = zarr.open_group(str(tmp_path / 'legacy.zarr'), mode='w')
            reader = RexZarrFormat()
        try:
            group.attrs['T'] = 1
            snapshot = group.create_group('snapshots').create_group('0')
            reader._store(snapshot, 'sources', source)
            reader._store(snapshot, 'targets', target)
            restored = reader._read_temporal_rex(group)
        finally:
            if kind == 'hdf5':
                group.close()
    elif kind == "rcbd":
        from rexgraph.io.bundle import _read_temporal_rex
        tensors = {'snapshots/0/sources': source, 'snapshots/0/targets': target}
        restored = _read_temporal_rex(tmp_path, tensor_reader=tensors.__getitem__, manifest={'T': 1})
    else:
        from rexgraph.io.safetensors_bridge import _temporal_from_loaded
        restored = _temporal_from_loaded({'snapshot/0/sources': source, 'snapshot/0/targets': target}, {'T': 1})
    assert restored.T == 1 and restored.at(0).nE == 1
    np.testing.assert_array_equal(restored.at(0)._boundary_idx, [0, 1])


def test_exact_weights_survive_core_appends_and_attribute_updates():
    rex = RexGraph.from_graph([0], [1])
    rex.add_hyperedges([[1, 2, 3]], w_E=[Q(1, 3)])
    rex._ensure_clean()
    assert rex.w_E[1] == Q(1, 3)
    rex.add_edges([3], [4], w_E=np.array([2**60+1], np.int64))
    rex._ensure_clean()
    assert rex.w_E[2] == 2**60+1 and rex.w_E[1] == Q(1, 3)
    rex.set_cell_attrs([0], w_E=[Q(2, 7)])
    assert rex.w_E[0] == Q(2, 7) and rex.w_E[2] == 2**60+1


@pytest.mark.parametrize("backend", ["memory", "file", "rex", "sql", "object"])
def test_rcql_rational_commit_and_replacement(tmp_path, backend):
    rcdb = pytest.importorskip("rcdb")
    rcql = pytest.importorskip("rcql")
    if backend == "sql":
        pytest.importorskip("sqlalchemy")
    if backend == "object":
        pytest.importorskip("fsspec")
    constructors = {"memory": lambda: rcdb.MemoryStore(),
                    "file": lambda: rcdb.FileStore(str(tmp_path / "file")),
                    "rex": lambda: rcdb.RexStore(str(tmp_path / "rex")),
                    "sql": lambda: rcdb.SQLStore(f"sqlite:///{tmp_path / 'sql.db'}"),
                    "object": lambda: rcdb.ObjectStore(f"file://{tmp_path / 'objects'}")}
    store = constructors[backend]().configure_security(require_commits=True)
    try:
        for version in range(2):
            rex = state([Q(2, 3) + version * Q(1, 2**70), Q(-5, 7)])
            executor = rcql.Executor(sources={"db": store}, params={"r": rex})
            result = executor.execute(rcql.parse('FROM $db MUTATE "r" SET state=$r, '
                                                 f'expected_version={version} COMMIT'))
            assert result.values[0].version == version+1
            assert store.verify_commits("r")
            assert store.read_record("r").state_digest == object_digest(rex)
    finally:
        store.close()
