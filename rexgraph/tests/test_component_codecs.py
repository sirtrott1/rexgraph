"""Owned native payloads and explicit carried state partition selection."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import Absent, Relations, RexGraph, VertexTable
from rexgraph.sealed_state import open_state, seal_state, state_identity
from rexgraph.state import CODEC_TENSOR, decode_tensors, encode_tensors, from_state, to_state
from rexgraph.sectioning import add_sectioning, add_coarsening, sectionings_of
from rexgraph.partition_state import build_rex_partition, partition_from_policy, partition_policy
from rexgraph.object_identity import object_digest
from rexgraph.native_rank import primary_columns
from rexgraph.native_sparse import sparse_arrays
from rexgraph.value_codec import unpack_value


def fixture():
    declared = Relations.from_supports([[0, 1, 2, 3], [3, 4]],
        vertices=VertexTable(tuple(f"v{i}" for i in range(6)), tuple(f"label{i}" for i in range(6)),
                             tuple((f"alias{i}",) for i in range(6))),
        heads=[1, 0], shares=[Q(1, 4), 0, Q(1, 2), Q(1, 4), Absent, Absent],
        weights=[Absent, Q(2**100+1, 7)], signs=[-1, 1], relation_ids=[17, 23],
        relation_types=["branch", "pair"], embedding=[[i, Q(1, 7)] for i in range(6)],
        provenance={"origin": "codec fixture", "mutable": [1, 2]})
    rex = RexGraph.from_relations(declared, g_channel="normalized", c_channel="count")
    rex.attach_metadata(1, 1, "exact", [Q(1, 7), Absent, np.array([3, 4])])
    rex._signals = np.array([[7, 8], [9, 10]])
    rex._w_boundary = {(1, 4): np.array([11, 12])}
    add_sectioning(rex, "fine", {"first": [0], "second": [1]})
    add_coarsening(rex, "coarse", "fine", [0, 0], ["all"])
    return rex


def retained_fixture(kind):
    domain = RexGraph.from_graph([0], [1], relation_ids=[17])
    if kind == "field":
        from rexgraph.tensor_field import TensorField, FieldSource
        from rexgraph.type_accession import CoordinateSpace
        value = TensorField(CoordinateSpace("C1", ("17",)), [Q(2**100+1, 7)], source=FieldSource(domain), grade=1)
    elif kind in {"recipe", "family"}:
        from rexgraph.sheaf import ExactSheaf
        system = ExactSheaf(domain, grade=0, stalk_dim=2).section_system()
        value = system.recipe if kind == "recipe" else system.complete(*system.pins({("0", "0"): 2**100+1}))
    elif kind == "span":
        from rexgraph.span import SpanAttachment, SpanBlock
        from rexgraph.coordinate_map import CoordinateMap
        text = SpanBlock("text", "doc", "character", (("a", 0, 3),))
        time = SpanBlock("time", "utc", "second", (("b", Q(1, 3), Q(2, 3)),))
        value = SpanAttachment("annotation", "17", "time", "fixture", text, time,
                              CoordinateMap(text.coordinates, time.coordinates, ((0, 0, Q(1, 7)),)))
    elif kind == "model":
        pytest.importorskip("torch")
        from rexgraph.nn.lifecycle import create_checkpoint
        value = create_checkpoint(domain, configuration={"n_classes": 2}, seed=7)
    else:
        value = domain
    holder = fixture()
    holder.attach_metadata(1, 1, "retained", value)
    return holder, value


def logical(rex):
    state = open_state(to_state(rex))
    tensors = dict(state.tensors)
    codecs = unpack_value(tensors.pop(CODEC_TENSOR).tobytes()) if CODEC_TENSOR in tensors else {}
    decode_tensors(tensors, codecs)
    return tensors, dict(state.header)


def test_native_dispatch_uses_each_owner_and_bypasses_legacy_packers(monkeypatch):
    import rexgraph.components as components
    import rexgraph.state as state
    rex, value = retained_fixture("field")
    registry = components.ComponentRegistry()
    encoded, decoded = [], []
    for name, codec in components.component_registry()._codecs.items():
        def encoder(rex, context, codec=codec):
            encoded.append(codec.name)
            return codec.encode(rex, context)
        def decoder(payload, context, codec=codec):
            decoded.append(codec.name)
            assert set(payload.header) <= set(codec.header_fields)
            return codec.decode(payload, context)
        registry.register(replace(codec, encode=encoder, decode=decoder))
    monkeypatch.setattr(components, "_registry", registry)
    monkeypatch.setattr(state, "_pack_cell_metadata", lambda *a, **kw: pytest.fail("legacy packer reached"))
    monkeypatch.setattr(state, "_unpack_cell_metadata", lambda *a, **kw: pytest.fail("legacy unpacker reached"))
    back = from_state(to_state(rex))
    assert encoded == decoded == registry._ordered()
    assert back.get_metadata(1, 1, "retained").coefficient_digest == value.coefficient_digest


def test_snapshot_and_decoded_graph_own_their_arrays_and_metadata():
    rex = fixture()
    saved = to_state(rex)
    digest = state_identity(saved)
    back = from_state(saved)
    back._signals[0, 0] = 99
    back._w_boundary[(1, 4)][0] = 99
    back._agent_meta["mutable"][0] = 99
    for name in ("fine", "coarse"):
        a, b = sectionings_of(rex)[name], sectionings_of(back)[name]
        for attr in ("indptr", "indices", "parent", "spans"):
            original, copied = getattr(a, attr), getattr(b, attr)
            if isinstance(original, np.ndarray):
                assert not np.shares_memory(original, copied)
    rex._boundary_idx[0] = 5
    rex._signals[0, 0] = 42
    rex._agent_meta["mutable"][0] = 42
    sectionings_of(rex)["fine"].indices[0] = 1
    assert state_identity(saved) == digest
    pristine = from_state(saved)
    assert pristine._signals[0, 0] == 7 and pristine._w_boundary[(1, 4)][0] == 11
    assert pristine._agent_meta["mutable"] == [1, 2]
    assert sectionings_of(pristine)["fine"].cells(0).tolist() == [0]


def test_previous_partial_provenance_labels_remain_readable():
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 3])
    rex._agent_meta = {"vertex_labels": ["legacy first", "legacy second"]}
    saved = to_state(rex)
    back = from_state(saved)
    assert back.vertex_labels == ("legacy first", "legacy second")
    assert state_identity(to_state(back)) == state_identity(saved)
    with pytest.raises(ValueError, match="vertex labels"):
        RexGraph.from_graph([0, 1, 2], [1, 2, 3]).set_provenance(rex._agent_meta)


@pytest.mark.parametrize("mode", ["structural", "all"])
@pytest.mark.parametrize("mask", [[1, 0], [0, 1], [0, 0]])
def test_partition_keeps_declared_math_and_exact_presence(mode, mask):
    rex = fixture()
    before = object_digest(rex)
    p = build_rex_partition(rex, mask, carried_state=mode)
    back = from_state(to_state(p.rex))
    vertices, edges = p.cell_maps
    remap = {old: new for new, old in enumerate(vertices)}
    assert primary_columns(back) == [{remap[v]: c for v, c in primary_columns(rex)[i].items()} for i in edges]
    assert back.relations.weight.values().tolist() == [rex.relations.weight.values()[i] for i in edges]
    assert back.relations.vertices.ids == tuple(rex.relations.vertices.ids[i] for i in vertices)
    assert back.relations.vertices.aliases == tuple(rex.relations.vertices.aliases[i] for i in vertices)
    assert object_digest(back) == p.state.result_state and object_digest(rex) == before
    if mode == "all":
        sub, _, _ = rex.subgraph(mask)
        assert object_digest(back) == object_digest(sub)
    else:
        assert not getattr(back, "_agent_meta", {}) and not getattr(back, "_cell_metadata", {})
        assert not hasattr(back, "_signals") and not sectionings_of(back)


@pytest.mark.parametrize("kind", ["field", "recipe", "family", "span", "model", "nested"])
def test_partition_carries_retained_certificate_independently(kind):
    rex, value = retained_fixture(kind)
    result = build_rex_partition(rex, [0, 1], carried_state="all")
    back = from_state(to_state(result.rex)).get_metadata(1, 0, "retained")
    if kind == "nested":
        assert object_digest(back) == object_digest(value)
        back.set_cell_attrs([0], w_E=[3])
        assert value.relations.weight.values().tolist() == [Absent]
    else:
        assert back.coefficient_digest == value.coefficient_digest


def test_explicit_upper_selection_does_not_reintroduce_surviving_cofaces():
    face = [(0, 1), (1, 1), (2, -1)]
    difference = [(0, 1), (1, -1)]
    rex = RexGraph.from_cells([4, [[0, 1], [1, 2], [0, 2]], [face, face],
                               [difference, difference], [difference]])
    rex.attach_metadata(3, 1, "upper", Q(2**100+1, 7))
    p = build_rex_partition(rex, [0, 0, 0], grade_masks={3: [0, 1]}, carried_state="all")
    assert p.cell_maps == ((0, 1, 2), (0, 1, 2), (0, 1), (1,), ())
    assert [sparse_arrays(b)[3] for b in p.rex._graded_duals] == [(2, 1), (1, 0)]
    assert p.rex.get_metadata(3, 0, "upper") == Q(2**100+1, 7)
    assert p.rex.chain_valid and object_digest(from_state(to_state(p.rex))) == p.state.result_state


def test_policy_binds_carried_state_and_preserves_structural_default_identity():
    rex = fixture()
    policy = {"source_state": object_digest(rex), "cells": [[1, [1]]]}
    a = partition_from_policy(rex, policy)
    b = partition_from_policy(rex, dict(policy, carried_state="structural"))
    c = partition_from_policy(rex, dict(policy, carried_state="all"))
    assert a.state == b.state and a.state.selection_digest == c.state.selection_digest
    assert a.state.policy_digest != c.state.policy_digest and a.state.result_state != c.state.result_state
    assert c.rex.get_metadata(1, 0, "exact")[0] == Q(1, 7)


@pytest.mark.parametrize("mode", ["none", "ALL", None, 1])
def test_bad_partition_mode_is_refused(mode):
    rex = fixture()
    with pytest.raises(ValueError, match="carried_state"):
        build_rex_partition(rex, [1, 1], carried_state=mode)
    with pytest.raises(ValueError, match="carried_state"):
        partition_policy(rex, {"source_state": object_digest(rex), "cells": [], "carried_state": mode})


@pytest.mark.parametrize("kind", ["field", "recipe", "span", "nested"])
def test_resealed_orphan_payloads_are_refused(kind):
    rex, _ = retained_fixture(kind)
    tensors, header = logical(rex)
    prefix = {"field": "field/", "recipe": "section/", "span": "annotation/", "nested": "nested/"}[kind]
    if kind == "nested":
        next(col for col in header["cell_meta"] if col["kind"] == "rex")["idx"] = []
    else:
        tensors[prefix+"cm_1_retained/9/extra"] = np.array([1])
    malformed = seal_state(tensors, header, encode_tensors(tensors, native=True))
    with pytest.raises(ValueError, match="unclaimed"):
        from_state(malformed)


@pytest.mark.parametrize("indices", [[1, 1], [-1], [0.5], [[1]]])
def test_resealed_invalid_attribute_addresses_are_refused(indices):
    tensors, header = logical(fixture())
    tensors["cm_1_exact_idx"] = np.asarray(indices)
    malformed = seal_state(tensors, header, encode_tensors(tensors, native=True))
    with pytest.raises(ValueError, match="attribute cell indices"):
        from_state(malformed)


@pytest.mark.parametrize("fault", ["missing", "claims", "header", "duplicate", "decode_arguments"])
def test_registered_codec_contract_refuses_incomplete_or_conflicting_owners(fault):
    from rexgraph.components import ComponentCodec, ComponentRegistry, ComponentPayload, DecodedComponent
    registry = ComponentRegistry()
    codec = ComponentCodec("extension", 1, lambda h, n: n & {"extra"},
        encode=lambda rex, ctx: ComponentPayload({"extra": np.array([1])}),
        decode=lambda p, ctx: DecodedComponent())
    if fault == "missing":
        codec = replace(codec, decode=None)
    elif fault == "claims":
        codec = replace(codec, claim=lambda h, n: ())
    elif fault == "header":
        codec = replace(codec, encode=lambda r, c: ComponentPayload({}, {"unowned": 1}))
    registry.register(codec)
    if fault == "duplicate":
        registry.register(replace(codec, name="second"))
    if fault == "decode_arguments":
        registry = ComponentRegistry()
        for name in ("first", "second"):
            registry.register(ComponentCodec(name, 1, lambda h, n: (),
                encode=lambda r, c: ComponentPayload(), decode=lambda p, c: DecodedComponent({"directed": False})))
        with pytest.raises(ValueError, match="conflicting constructor"):
            registry.decode({}, {})
    else:
        with pytest.raises(ValueError, match="contract|ownership|header|conflicting"):
            registry.encode(RexGraph.from_graph([0], [1]))


def test_trusted_extension_roundtrips_and_unregistered_reader_refuses_it(monkeypatch):
    import rexgraph.components as components
    base = components.component_registry()
    registry = components.ComponentRegistry()
    for codec in base._codecs.values():
        registry.register(codec)
    registry.register(components.ComponentCodec("extension", 1, lambda h, n: n & {"extra"},
        encode=lambda rex, c: components.ComponentPayload({"extra": rex._extra.copy()}),
        decode=lambda p, c: components.DecodedComponent(restore=lambda rex: setattr(rex, "_extra", p.tensors["extra"].copy()))))
    monkeypatch.setattr(components, "_registry", registry)
    rex = RexGraph.from_graph([0], [1])
    rex._extra = np.array([2**60+1])
    saved = to_state(rex)
    back = from_state(saved)
    assert back._extra.tolist() == [2**60+1]
    assert not np.shares_memory(rex._extra, back._extra)
    assert state_identity(to_state(back)) == state_identity(saved)
    with pytest.raises(ValueError, match="transport contract"):
        rex.copy()
    monkeypatch.setattr(components, "_registry", base)
    with pytest.raises(ValueError, match="unknown component"):
        from_state(saved)


# Captured from the previous verified transport wheel, before codec dispatch was
# introduced. These pin the actual v10 semantic bytes, not a encoded again fixture.
@pytest.mark.parametrize("kind,expected", [
    ("generic", "832cf46001bc214f88553a29e013797ea48b3e51c01559c06872d9631961b16c"),
    ("field", "131dce237331e6347da9afcdf8418d4e38ddc830db025479fef4d00b0d718ece"),
    ("recipe", "ee83492c2b9bff31c8d81431bf1aba448517152f71364cbb0f72a17e84c7e92d"),
    ("family", "ec1da9812b5ba33f927d924f15d906221dac5c46a253e7fa15102e4c8b400e7f"),
    ("span", "36489356e691d2b48541175ef720014ef24e149598bbebcb9c9c1c7079e361af"),
    ("nested", "aa1ce35071dab073892e941cca9c93152703079f55b5abda731060f084f5792c"),
])
def test_previous_v10_semantic_identity_is_preserved(kind, expected):
    rex = fixture() if kind == "generic" else retained_fixture(kind)[0]
    assert state_identity(to_state(rex)) == expected
    assert state_identity(to_state(from_state(to_state(rex)))) == expected
