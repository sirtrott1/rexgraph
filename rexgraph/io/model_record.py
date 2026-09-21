"""Native model records and full histories over existing storage formats."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from rexgraph.model_state import ModelState, ModelTimeline

__all__ = ["model_record", "read_model_record", "model_history", "read_model_history",
           "save_model", "load_model"]


def _copy(source):
    from rexgraph.io.rex_state import to_state, from_state
    return from_state(to_state(source))


def model_record(source, state):
    """Keep a model assertion separate from the native source it observes."""
    from rexgraph.graph import RexGraph
    if not isinstance(state, ModelState): raise TypeError("model record requires ModelState")
    state.bind(source)
    record = RexGraph.from_hypergraph([0, 1], [0])
    record.attach_metadata(1, 0, "model", state)
    record.attach_metadata(1, 0, "model_source", _copy(source))
    return record


def read_model_record(record, index=0):
    state = record.get_metadata(1, index, "model")
    source = record.get_metadata(1, index, "model_source")
    if not isinstance(state, ModelState) or source is None:
        raise TypeError("selected record does not carry a native model and its source")
    state.bind(source)
    return state, source


def model_history(states, sources, times, *, time_axis="model_observation", time_unit="step"):
    """Store complete snapshots beside an ordinal TemporalState acceleration index."""
    from rexgraph.graph import RexGraph, TemporalRex
    from rexgraph.io.temporal_state import to_temporal_state
    states, sources, times = tuple(states), tuple(sources), tuple(times)
    if not states or not len(states) == len(sources) == len(times):
        raise ValueError("model history requires aligned states, sources and exact times")
    history = TemporalRex([])
    for i, (state, source) in enumerate(zip(states, sources, strict=True)):
        if not isinstance(state, ModelState): raise TypeError("history requires ModelState entries")
        state.bind(source)
        if i and state.parent != states[i - 1].coefficient_digest:
            raise ValueError("model history must follow its declared parent chain")
        history.append_snapshot(_copy(source), at=i)
    temporal = to_temporal_state(history)
    timeline = ModelTimeline(times, tuple(v.coefficient_digest for v in states),
        tuple(v.source.state_digest for v in states), temporal.header, temporal.tensors, time_axis, time_unit)
    record = RexGraph.from_hypergraph(np.arange(len(states) + 1), np.arange(len(states)))
    for i, (state, source) in enumerate(zip(states, sources, strict=True)):
        record.attach_metadata(1, i, "model", state)
        record.attach_metadata(1, i, "model_source", _copy(source))
    record.attach_metadata(1, 0, "model_timeline", timeline)
    return record


def _index_projection(source):
    import json
    from rexgraph.io.rex_state import to_state, decode_tensors, CODEC_TENSOR
    state = to_state(source)
    tensors = dict(state.tensors)
    if CODEC_TENSOR in tensors:
        decode_tensors(tensors, json.loads(tensors[CODEC_TENSOR].tobytes().decode()))
    names = ("boundary_ptr", "boundary_idx", "w_E", "signs", "B2_col_ptr", "B2_row_idx", "B2_vals", "relation_ids")
    return {k: tensors[k] for k in names if k in tensors}, {
        k: state.header[k] for k in ("nV", "nE", "nF", "directed", "g_channel", "c_channel")}


def _verify_index(timeline, snapshots):
    from rexgraph.io.temporal_state import from_temporal_state
    indexed = from_temporal_state(timeline.temporal_state())
    for i, (_, source) in enumerate(snapshots):
        actual, header = _index_projection(indexed.reconstruct_at(i))
        expected, wanted = _index_projection(source)
        if header != wanted or actual.keys() != expected.keys() or any(
                not np.array_equal(actual[k], expected[k]) for k in expected):
            raise ValueError("temporal index differs from its complete source snapshot")


def read_model_history(record, time=None):
    timeline = record.get_metadata(1, 0, "model_timeline")
    if not isinstance(timeline, ModelTimeline): raise TypeError("record has no model timeline")
    timeline.check_state()
    snapshots = tuple(read_model_record(record, i) for i in range(len(timeline.times)))
    if tuple(s.coefficient_digest for s, _ in snapshots) != timeline.model_digests:
        raise ValueError("model timeline and complete snapshots differ")
    if tuple(s.source.state_digest for s, _ in snapshots) != timeline.source_digests:
        raise ValueError("model timeline source identities differ")
    for i in range(1, len(snapshots)):
        if snapshots[i][0].parent != snapshots[i - 1][0].coefficient_digest:
            raise ValueError("model history parent chain differs")
    _verify_index(timeline, snapshots)
    return (timeline, snapshots) if time is None else snapshots[timeline.at(time)]


def save_model(path, record, **options):
    """Use the canonical bundle or safetensors writer for a native model record."""
    path = Path(path)
    if record.get_metadata(1, 0, "model_timeline") is not None: read_model_history(record)
    else: read_model_record(record)
    if path.suffix == ".rcbd":
        from rexgraph.io.bundle import save_rcbd
        return save_rcbd(path, record, **options)
    if path.suffix == ".safetensors":
        from rexgraph.io.safetensors_bridge import rex_to_safetensors
        return rex_to_safetensors(record, path, **options)
    raise ValueError("model transport requires .rcbd or .safetensors")


def load_model(path, **options):
    path = Path(path)
    if path.suffix == ".rcbd":
        from rexgraph.io.bundle import load_rcbd
        record = load_rcbd(path, **options)
    elif path.suffix == ".safetensors":
        from rexgraph.io.safetensors_bridge import safetensors_to_rex
        record = safetensors_to_rex(path, **options)
    else: raise ValueError("model transport requires .rcbd or .safetensors")
    if record.get_metadata(1, 0, "model_timeline") is not None: read_model_history(record)
    else: read_model_record(record)
    return record
