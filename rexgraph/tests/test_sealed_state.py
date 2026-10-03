"""V10 semantic ownership, tamper refusal and legacy migration conformance."""
from copy import deepcopy
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.components import ComponentCodec, ComponentRegistry
from rexgraph.graph import RexGraph
from rexgraph.native_rank import primary_columns
from rexgraph.sealed_state import SEMANTICS_TENSOR, migrate_state, state_identity, to_sealed_state
from rexgraph.state import RexState, from_state, to_state, verify_state
from rexgraph.value import Absent, Approx
from rexgraph.value_codec import pack_value, unpack_value


def fixture():
    graph = RexGraph.from_cells([5, [[(0, -1), (1, Q(1, 4)), (2, Q(1, 2)), (3, Q(1, 4))]]])
    graph._w_E = np.array([Q(2, 3)], object)
    graph._w_boundary = {(0, 0): Q(1, 10), (0, 1): np.array([Q(1, 3)], object)}
    graph.attach_metadata(1, 0, "exact", {"missing": Absent, "count": 2**100, "value": Q(1, 10), "estimate": Approx(.1, "instrument")})
    graph._agent_meta = {"vertex_labels": ["isolated", "a", "b", "c", "d"], "typed": Q(1, 7)}
    return graph


def clone(state):
    return RexState({name: value.copy() for name, value in state.tensors.items()}, deepcopy(state.header))


def test_v10_roundtrip_seals_exact_values_absence_identity_and_isolates():
    graph = fixture()
    state = to_sealed_state(graph)
    assert state.header["format_version"] == 10
    assert verify_state(state)
    assert b"RGVL\x01" == state.tensors[SEMANTICS_TENSOR].tobytes()[:5]
    assert "codec_spec" not in state.tensors
    restored = from_state(state)
    assert primary_columns(restored) == primary_columns(graph)
    assert restored._exact_column_norms_B1() == [Q(11, 8)]
    assert restored.nV == 5
    assert restored._w_E.tolist() == [Q(2, 3)]
    assert restored._w_boundary[(0, 0)] == Q(1, 10)
    assert restored.get_metadata(1, 0, "exact")["missing"] is Absent
    assert restored._agent_meta["typed"] == Q(1, 7)
    assert state_identity(to_sealed_state(restored)) == state_identity(state)


@pytest.mark.parametrize("field,value", [("nV", 6), ("nE", 2), ("nF", 1), ("directed", True),
                                         ("g_channel", "normalized"), ("c_channel", "count"),
                                         ("format_version", 9), ("cell_meta", [])])
def test_external_header_tampering_and_downgrade_are_refused(field, value):
    state = clone(to_sealed_state(fixture()))
    state.header[field] = value
    assert not verify_state(state)
    with pytest.raises(ValueError):
        from_state(state)


@pytest.mark.parametrize("change", ["payload", "name", "dtype", "shape", "extra", "missing", "codec", "component", "claim", "schema", "version"])
def test_state_tamper_matrix(change):
    state = clone(to_sealed_state(fixture()))
    if change == "payload":
        state.tensors["w_E"].flat[0] ^= 1
    elif change == "name":
        state.tensors["renamed"] = state.tensors.pop("w_E")
    elif change == "dtype":
        state.tensors["w_E"] = state.tensors["w_E"].astype(np.int64)
    elif change == "shape":
        state.tensors["w_E"] = state.tensors["w_E"].reshape(-1, 1)
    elif change == "extra":
        state.tensors["unknown"] = np.zeros(1)
    elif change == "missing":
        state.tensors.pop("w_E")
    else:
        record = unpack_value(state.tensors[SEMANTICS_TENSOR].tobytes())
        if change == "codec":
            record["tensor_codecs"]["unknown"] = {"c": "arange", "start": 0, "n": 1, "dtype": "<i8"}
        elif change == "component":
            record["components"][0]["name"] = "unregistered"
        elif change == "claim":
            record["components"][0]["tensors"].append("boundary_idx")
        elif change == "schema":
            record["header"]["cell_meta"][0]["dim"] = 0
        elif change == "version":
            record["format_version"] = 9
        state.tensors[SEMANTICS_TENSOR] = np.frombuffer(pack_value(record), np.uint8)
    assert not verify_state(state)
    with pytest.raises(ValueError):
        from_state(state)


def test_unknown_state_is_refused_even_when_caller_recomputes_digest():
    from rexgraph.identity import tensor_digest
    state = clone(to_sealed_state(fixture()))
    state.tensors["unknown"] = np.zeros(1)
    state.header.update(digest_names=sorted(state.tensors), digest=tensor_digest(state.tensors))
    assert not verify_state(state)


def test_component_registry_refuses_overlapping_owners():
    registry = ComponentRegistry()
    for name in ("one", "two"):
        registry.register(ComponentCodec(name, 1, lambda header, names: names))
    with pytest.raises(ValueError, match="duplicate component claim"):
        registry.describe({}, {"x": np.zeros(1)}, {})


def test_nested_positions_and_states_are_sealed():
    outer = RexGraph.from_graph([0], [1])
    outer.attach_metadata(0, 1, "child", fixture())
    state = to_sealed_state(outer)
    assert verify_state(state)
    restored = from_state(state)
    assert restored.get_metadata(0, 1, "child").get_metadata(1, 0, "exact")["missing"] is Absent


def test_legacy_state_migrates_without_rewriting_it():
    graph = RexGraph.from_cells([4, [[(0, -1), (1, Q(1, 4)), (2, Q(1, 2)), (3, Q(1, 4))]]])
    legacy = to_state(graph, _native=False)
    old_identity = legacy.header["digest"]
    native = migrate_state(legacy)
    assert native.header["format_version"] == 10
    assert native.header["digest"] != old_identity
    assert legacy.header["format_version"] == 9
    assert legacy.header["digest"] == old_identity
    assert primary_columns(from_state(native)) == primary_columns(graph)


def test_native_attribute_identity_ignores_mapping_insertion_order():
    first, second = fixture(), fixture()
    first.attach_metadata(1, 0, "order", {"a": 1, "b": Q(1, 3)})
    second.attach_metadata(1, 0, "order", {"b": Q(1, 3), "a": 1})
    assert state_identity(to_sealed_state(first)) == state_identity(to_sealed_state(second))


def test_container_metadata_does_not_change_native_object_identity():
    from rexgraph.object_identity import state_object_digest
    state = to_state(fixture())
    original = state_object_digest(state)
    state.header.update(magic="RCBD", cached_arrays=["derived"], wire_digest="framing")
    assert state_object_digest(state) == original


def test_cyclic_graph_metadata_is_refused_and_does_not_poison_the_next_write():
    graph = RexGraph.from_graph([0], [1])
    graph.attach_metadata(1, 0, "cycle", graph)
    with pytest.raises(ValueError, match="cyclic"):
        to_state(graph)
    graph._cell_metadata.clear()
    assert verify_state(to_state(graph))


@pytest.mark.parametrize("kind", ["field", "span", "recipe", "family", "model"])
def test_v10_retained_components_use_binary_specs_and_keep_mathematical_identity(kind):
    from rexgraph.coordinate_map import CoordinateMap
    from rexgraph.nn.lifecycle import create_checkpoint
    from rexgraph.sheaf import ExactSheaf
    from rexgraph.span import SpanAttachment, SpanBlock
    from rexgraph.tensor_field import FieldSource, TensorField
    from rexgraph.type_accession import CoordinateSpace
    source = RexGraph.from_hypergraph([0, 2], [0, 1], relation_ids=np.array([17]))
    if kind == "field":
        value = TensorField(CoordinateSpace("relations", ("17",)), [Q(2**200+1, 7)], source=FieldSource(source), grade=1)
    elif kind == "span":
        text = SpanBlock("text", "doc:fixture", "character", (("a", 0, 3),))
        time = SpanBlock("time", "utc", "second", (("b", Q(1, 3), Q(2, 3)),))
        value = SpanAttachment("annotation", "17", "time", "fixture", text, time,
                               CoordinateMap(text.coordinates, time.coordinates, ((0, 0, Q(1, 7)),)))
    elif kind in {"recipe", "family"}:
        system = ExactSheaf(source, grade=0, stalk_dim=2).section_system()
        value = system.recipe if kind == "recipe" else system.complete(*system.pins({("0", "0"): 2**100+1}))
    else:
        pytest.importorskip("torch")
        value = create_checkpoint(source, configuration={"n_classes": 2}, seed=7)
    holder = RexGraph.from_hypergraph([0, 1], [0])
    holder.attach_metadata(1, 0, "retained", value)
    state = to_sealed_state(holder)
    assert verify_state(state)
    specs = [a for name, a in state.tensors.items() if name.endswith("/spec")]
    assert specs and all(a.tobytes().startswith(b"RGVL\x01") for a in specs)
    restored = from_state(state).get_metadata(1, 0, "retained")
    assert restored.coefficient_digest == value.coefficient_digest
    assert state_identity(to_sealed_state(from_state(state))) == state_identity(state)
