"""Complete state survives compositions of supported interchange containers."""
from fractions import Fraction

import numpy as np
import pytest

from rexgraph import Absent, Relations, TemporalRex, RexGraph, VertexTable
from rexgraph.object_identity import object_digest

FORMATS = ("rcbd", "safetensors", "h5", "zarr", "arrow", "json")


def fixture():
    relations = Relations.from_supports(
        [[0, 1, 2], [1, 2], [2, 2]],
        vertices=VertexTable(("a", "b", "c", "isolated")),
        weights=[Fraction(2, 3), Absent, 0], signs=[-1, 1, 1],
        relation_ids=["branch", "pair", "loop"],
        attributes={1: {0: {"rational": Fraction(1, 10), "large": 2**512,
                            "unknown": Absent}}},
    )
    return RexGraph.from_relations(relations, c_channel="count")


def test_temporal_restoration_owns_checkpoint_buffers():
    from rexgraph.temporal_state import to_temporal_state, from_temporal_state

    history = TemporalRex([])
    history.append_snapshot(RexGraph.from_graph([0], [1]), at=0)
    state = to_temporal_state(history)
    original = history.at(0).relation_supports()
    restored = from_temporal_state(state)
    state.tensors["checkpoint/0/boundary_idx"][:] = 0
    assert restored.at(0).relation_supports() == original


def test_json_native_save_does_not_compute_a_numeric_graph_view(tmp_path, monkeypatch):
    from rexgraph.io import save, load
    graph = fixture()
    def refuse(*args, **kwargs):
        pytest.fail("native JSON serialization requested layout or diagnostics")
    monkeypatch.setattr(RexGraph, "to_json", refuse)
    path = tmp_path / "native.json"
    save(path, graph)
    assert object_digest(load(path)) == object_digest(graph)


def test_json_native_state_rejects_a_changed_semantic_header(tmp_path):
    import base64
    import json
    from rexgraph.io import save, load
    from rexgraph.value_codec import pack_value, unpack_value

    path = tmp_path / "altered.json"
    save(path, fixture())
    envelope = json.loads(path.read_text())
    state = unpack_value(base64.b64decode(envelope["state"]))
    state["header"]["digest"] = "0"*64
    envelope["state"] = base64.b64encode(pack_value(state)).decode("ascii")
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="digest|seal|modified"):
        load(path)


def convert(graph, kind, path):
    dependency = {"safetensors": "safetensors", "h5": "h5py", "zarr": "zarr", "arrow": "pyarrow"}.get(kind)
    if dependency:
        pytest.importorskip(dependency)
    if kind == "arrow":
        from rexgraph.io.arrow_bridge import arrow_to_rex, rex_to_arrow
        return arrow_to_rex(rex_to_arrow(graph))
    from rexgraph.io import save, load
    save(path, graph)
    return load(path)


@pytest.mark.parametrize("first", FORMATS)
@pytest.mark.parametrize("second", FORMATS)
def test_graph_format_chains_preserve_complete_semantic_state(first, second, tmp_path):
    graph = fixture()
    intermediate = convert(graph, first, tmp_path / ("first." + first))
    restored = convert(intermediate, second, tmp_path / ("second." + second))
    assert object_digest(intermediate) == object_digest(graph)
    assert object_digest(restored) == object_digest(graph)
    assert restored.relation_keys == graph.relation_keys
    assert restored.get_metadata(1, 0, "rational") == Fraction(1, 10)


@pytest.mark.parametrize("first", ("rcbd", "safetensors", "h5", "zarr", "json"))
@pytest.mark.parametrize("second", ("rcbd", "safetensors", "h5", "zarr", "json"))
def test_temporal_format_chains_preserve_history_and_cutoffs(first, second, tmp_path):
    graph = TemporalRex([])
    graph.append_snapshot(RexGraph.from_graph([0], [1]), at=Fraction(1, 2))
    graph.append_snapshot(RexGraph.from_graph([0, 1], [1, 2]), at=Fraction(3, 2))
    intermediate = convert(graph, first, tmp_path / ("first." + first))
    restored = convert(intermediate, second, tmp_path / ("second." + second))
    assert object_digest(intermediate) == object_digest(graph)
    assert object_digest(restored) == object_digest(graph)
    assert list(restored._times) == list(graph._times)
    assert restored.at(0).nE == 1 and restored.at(1).nE == 2
