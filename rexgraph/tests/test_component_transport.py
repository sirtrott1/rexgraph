"""Declared carriers survive real append, restriction, replay and persistence."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest

from rexgraph import Absent, Relations, RexGraph, VertexTable
from rexgraph.components import CellMaps, ComponentCodec, ComponentRegistry
from rexgraph.native_rank import primary_columns
from rexgraph.sealed_state import open_state, seal_state, state_identity
from rexgraph.sectioning import add_coarsening, add_sectioning, sectionings_of
from rexgraph.state import CODEC_TENSOR, decode_tensors, encode_tensors, from_state, to_state
from rexgraph.value_codec import pack_value, unpack_value


def rich():
    source = Relations.from_supports(
        [[0, 1, 2, 3], [3, 4]],
        vertices=VertexTable(tuple(f"v{i}" for i in range(6)), tuple(f"label{i}" for i in range(6)),
                             tuple((f"alias{i}",) for i in range(6))),
        heads=[1, 0], shares=[Q(1, 4), 0, Q(1, 2), Q(1, 4), Absent, Absent],
        weights=[Absent, Q(2**100+1, 7)], signs=[-1, 1],
        relation_ids=[10, 11], relation_types=["branch", "pair"],
        attributes={0: {4: {"note": {"value": Q(1, 7)}}},
                    1: {0: {"assertion": [Q(1, 3), Absent]}, 1: {"array": np.array([5, 6])}}},
        embedding=[[i, Q(1, 7)] for i in range(6)], provenance={"origin": "transport fixture"})
    rex = RexGraph.from_relations(source, g_channel="normalized", c_channel="count")
    rex._w_boundary = {(0, 3): np.array([1, 2]), (1, 4): np.array([3, 4])}
    rex._signals = np.array([[7, 8], [9, 10]])
    add_sectioning(rex, "fine", {"first": [0], "second": [1]})
    add_coarsening(rex, "coarse", "fine", [0, 0], ["all"])
    return rex


def tetrahedron():
    edges = list(combinations(range(4), 2))
    faces = list(combinations(range(4), 3))
    edge_lookup = {c: i for i, c in enumerate(edges)}
    face_lookup = {c: i for i, c in enumerate(faces)}
    face_columns = [[(edge_lookup[c[:j]+c[j+1:]], (-1)**j) for j in range(3)] for c in faces]
    top = [(face_lookup[tuple(i for i in range(4) if i != j)], (-1)**j) for j in range(4)]
    rex = RexGraph.from_cells([4, edges, face_columns, [top]])
    rex.attach_metadata(3, 0, "certificate", Q(2, 7))
    return rex


def test_simplicial_endpoints_are_owned_and_ready_for_native_faces():
    sources = np.array([0, 1, 0], np.int32)
    targets = np.array([1, 2, 2], np.int32)
    rex = RexGraph.from_simplicial(sources, targets, [[0, 1, 2]])
    from rexgraph.core._faces import vertex_face_count
    assert vertex_face_count(rex._B2_dual, rex._sources, rex._targets, rex.nV).tolist() == [1, 1, 1]
    sources[:] = 99
    targets[:] = 99
    assert rex._sources.tolist() == [0, 1, 0]
    assert rex._targets.tolist() == [1, 2, 2]


def test_primary_constructor_owns_pairwise_endpoints():
    sources = np.array([0, 1], np.int32)
    targets = np.array([1, 2], np.int32)
    rex = RexGraph(sources=sources, targets=targets)
    sources[:] = 99
    targets[:] = 99
    assert rex.sources.tolist() == [0, 1]
    assert rex.targets.tolist() == [1, 2]
    assert rex.relation_supports() == [[0, 1], [1, 2]]


def test_copy_preserves_upper_boundary_index_storage_and_identity():
    from rexgraph.native_sparse import csr_carrier, sparse_arrays
    rex = tetrahedron()
    ptr, indices, data, shape = sparse_arrays(rex._graded_duals[0])
    rex._graded_duals = [csr_carrier(ptr.astype(np.int32), indices.astype(np.int32), data, shape)]
    rex = from_state(to_state(rex))
    copy = rex.copy()
    assert state_identity(to_state(copy)) == state_identity(to_state(rex))
    for old, new in zip(sparse_arrays(rex._graded_duals[0])[:3], sparse_arrays(copy._graded_duals[0])[:3], strict=True):
        assert old.dtype == new.dtype and not np.shares_memory(old, new)


@pytest.mark.parametrize("method", ["add_edges", "add_hyperedges"])
def test_append_preserves_missing_zero_and_unit_weights(method):
    rex = RexGraph.from_graph([0], [1])
    if method == "add_edges":
        rex.add_edges([1, 2, 3], [2, 3, 4], w_E=[0, 1, Absent])
    else:
        rex.add_hyperedges([[1, 2], [2, 3, 4], [4]], w_E=[0, 1, Absent])
    assert rex.relations.weight.values().tolist() == [Absent, 0, 1, Absent]
    assert rex.w_E.tolist() == [1, 0, 1, 1]
    back = from_state(to_state(rex))
    assert back.relations.weight.values().tolist() == [Absent, 0, 1, Absent]
    assert state_identity(to_state(back)) == state_identity(to_state(rex))


def test_setting_an_absent_weight_to_unit_declares_its_presence():
    rex = RexGraph.from_graph([0, 1], [1, 2])
    rex.set_cell_attrs([0], w_E=[1])
    assert rex.relations.weight.values().tolist() == [1, Absent]
    rex.set_cell_attrs([0], w_E=[Absent])
    assert rex.relations.weight.values().tolist() == [Absent, Absent]
    assert rex.w_E is None
    rex.set_cell_attrs([1], w_E=[2**100+1], signs=[-1])
    assert rex.w_E.tolist() == [1, 2**100+1]
    assert from_state(to_state(rex)).relations.weight.values().tolist() == [Absent, 2**100+1]


def test_interleaved_appends_keep_call_order_and_boundary_keys():
    rex = RexGraph.from_graph([0], [1], relation_ids=[10], w_E=np.array([2]))
    rex.add_hyperedges([[0, 2, 3]], relation_ids=[11], w_E=[Q(1, 7)], signs=[-1])
    rex.add_edges([3], [4], relation_ids=[12], w_E=[0], w_boundary={(0, 4): np.array([3])})
    rex.add_hyperedges([[4]], relation_ids=[13])
    assert rex.relation_supports() == [[0, 1], [0, 2, 3], [3, 4], [4]]
    assert rex.relation_keys == (10, 11, 12, 13)
    assert rex.relations.weight.values().tolist() == [2, Q(1, 7), 0, Absent]
    assert rex.relations.sign.tolist() == [1, -1, 1, 1]
    assert rex._w_boundary[(2, 4)].tolist() == [3]


def test_append_extends_declared_columns_and_carried_domains():
    rex = rich()
    before = primary_columns(rex)
    rex.add_hyperedges([[4, 6, 7]], relation_ids=[12])
    assert primary_columns(rex)[:2] == before
    assert rex.relations.weight.values().tolist() == [Absent, Q(2**100+1, 7), Absent]
    assert rex.vertex_labels[-2:] == ("", "")
    assert rex.embedding[-2:] == [Absent, Absent]
    assert rex.relations.vertices.ids[-2:] == (Absent, Absent)
    assert rex._signals[2].tolist() == [Absent, Absent]
    assert sectionings_of(rex)["fine"].n_cells == 3
    assert sectionings_of(rex)["coarse"].resolved(sectionings_of(rex)).cells(0).tolist() == [0, 1]
    assert state_identity(to_state(from_state(to_state(rex)))) == state_identity(to_state(rex))


@pytest.mark.parametrize("keep", [[1, 0], [0, 1], [0, 0]])
def test_functional_subgraph_and_compaction_share_one_transport(keep):
    source = rich()
    sub, vertices, edges = source.subgraph(keep)
    inplace = source.copy()
    inplace.remove_edges(np.logical_not(keep).astype(np.int32))
    inplace.compact()
    assert state_identity(to_state(sub)) == state_identity(to_state(inplace))
    assert state_identity(to_state(from_state(to_state(sub)))) == state_identity(to_state(sub))
    assert sub.relation_keys == tuple(source.relation_keys[i] for i in edges)
    assert sub.relations.vertices.ids == tuple(source.relations.vertices.ids[i] for i in vertices)
    assert sub.relations.vertices.aliases == tuple(source.relations.vertices.aliases[i] for i in vertices)
    assert sub.embedding == [source.embedding[i] for i in vertices]
    assert sub._signals.tolist() == source._signals[edges].tolist()
    assert sub._g_channel == source._g_channel and sub._c_channel == source._c_channel
    if keep == [1, 0]:
        assert primary_columns(sub)[0] == primary_columns(source)[0]
        assert sub.relations.share.presence.all()
        assert sub.get_metadata(1, 0, "assertion") == [Q(1, 3), Absent]
    elif keep == [0, 1]:
        assert sub._w_boundary[(0, 1)].tolist() == [3, 4]
        sub._w_boundary[(0, 1)][0] = 99
        sub.get_metadata(1, 0, "array")[0] = 99
        assert source._w_boundary[(1, 4)][0] == 3
        assert source.get_metadata(1, 1, "array")[0] == 5


def test_insert_preserves_faces_upper_grades_and_independent_ownership():
    source = tetrahedron()
    source.set_provenance({"vertex_labels": ["a", "b", "c", "d"]})
    inserted = source.insert_relations([[0, 4, 5]])
    assert inserted.nE == source.nE+1 and inserted.nF == source.nF
    assert inserted.get_metadata(3, 0, "certificate") == Q(2, 7)
    from rexgraph.native_sparse import sparse_arrays
    assert sparse_arrays(inserted._graded_duals[0])[3] == sparse_arrays(source._graded_duals[0])[3]
    assert inserted.chain_valid and inserted.vertex_labels[-2:] == ("", "")
    inserted.attach_metadata(3, 0, "certificate", 99)
    assert source.get_metadata(3, 0, "certificate") == Q(2, 7)


def test_restriction_drops_upper_cofaces_without_truncating_their_boundaries():
    source = tetrahedron()
    sub, _, _ = source.subgraph([0, 1, 1, 1, 1, 1])
    assert sub.nF == 2 and sub._graded_duals[0].shape == (2, 0)
    assert sub.get_metadata(3, 0, "certificate") is None
    assert sub.chain_valid


def test_face_append_extends_upper_boundary_row_domain():
    rex = tetrahedron()
    rex.add_faces([[0, 1, 3]], [[1, -1, 1]])
    rex.compact()
    assert rex._graded_duals[0].shape == (5, 1)
    assert rex.get_metadata(3, 0, "certificate") == Q(2, 7)
    assert rex.chain_valid
    assert state_identity(to_state(from_state(to_state(rex)))) == state_identity(to_state(rex))


@pytest.mark.parametrize("kind", ["field", "span", "recipe", "family", "model", "nested"])
def test_registered_attachment_owners_keep_certificates(kind):
    from rexgraph.coordinate_map import CoordinateMap
    from rexgraph.sheaf import ExactSheaf
    from rexgraph.span import SpanAttachment, SpanBlock
    from rexgraph.tensor_field import FieldSource, TensorField
    from rexgraph.type_accession import CoordinateSpace
    domain = RexGraph.from_graph([0], [1], relation_ids=[17])
    if kind == "field":
        value = TensorField(CoordinateSpace("C1", ("17",)), [Q(2**100+1, 7)], source=FieldSource(domain), grade=1)
    elif kind == "span":
        text = SpanBlock("text", "doc", "character", (("a", 0, 3),))
        time = SpanBlock("time", "utc", "second", (("b", Q(1, 3), Q(2, 3)),))
        value = SpanAttachment("annotation", "17", "time", "fixture", text, time,
                               CoordinateMap(text.coordinates, time.coordinates, ((0, 0, Q(1, 7)),)))
    elif kind in {"recipe", "family"}:
        system = ExactSheaf(domain, grade=0, stalk_dim=2).section_system()
        value = system.recipe if kind == "recipe" else system.complete(*system.pins({("0", "0"): 2**100+1}))
    elif kind == "model":
        pytest.importorskip("torch")
        from rexgraph.nn.lifecycle import create_checkpoint
        value = create_checkpoint(domain, configuration={"n_classes": 2}, seed=7)
    else:
        value = domain
        _ = value.B1  # ensure copying a nested graph does not copy native cache handles
    holder = RexGraph.from_graph([0, 1], [1, 2])
    holder.attach_metadata(1, 1, "retained", value)
    sub, _, _ = holder.subgraph([0, 1])
    back = from_state(to_state(sub)).get_metadata(1, 0, "retained")
    if kind == "nested":
        assert state_identity(to_state(back)) == state_identity(to_state(value))
        sub.get_metadata(1, 0, "retained").set_cell_attrs([0], w_E=[3])
        assert domain.relations.weight.values().tolist() == [Absent]
    else:
        assert back.coefficient_digest == value.coefficient_digest


def logical_state(rex):
    state = open_state(to_state(rex))
    tensors = dict(state.tensors)
    codecs = unpack_value(tensors.pop(CODEC_TENSOR).tobytes()) if CODEC_TENSOR in tensors else {}
    decode_tensors(tensors, codecs)
    return tensors, state.header


def test_previous_v10_relation_record_reads_and_migrates_to_owned_presence():
    rex = rich()
    tensors, header = logical_state(rex)
    tensors.pop("weight_presence")
    tensors["relation_record"] = np.frombuffer(pack_value(rex.relations.as_state_record(native_ids=rex.relation_ids, version=2)), np.uint8)
    legacy_v10 = seal_state(tensors, header, encode_tensors(tensors, native=True))
    back = from_state(legacy_v10)
    assert back.relations.weight.values().tolist() == rex.relations.weight.values().tolist()
    native_tensors, _ = logical_state(back)
    assert unpack_value(native_tensors["relation_record"].tobytes())["version"] == 3
    assert native_tensors["weight_presence"].tolist() == [0, 1]


def test_legacy_full_relation_record_normalizes_all_absent_metrics():
    rex = RexGraph.from_graph([0, 1], [1, 2])
    tensors, header = logical_state(rex)
    tensors["w_E"] = np.ones(2)
    tensors["relation_record"] = np.frombuffer(pack_value(rex.relations.as_record()), np.uint8)
    legacy = seal_state(tensors, header, encode_tensors(tensors, native=True))
    back = from_state(legacy)
    assert back.w_E is None
    assert back.relations.weight.values().tolist() == [Absent, Absent]
    assert state_identity(to_state(from_state(to_state(back)))) == state_identity(to_state(back))


@pytest.mark.parametrize("mask", [[0, 0], [1, 1], [0, 2], [0], [1, 0]])
def test_resealed_invalid_presence_is_refused(mask):
    tensors, header = logical_state(rich())
    tensors["weight_presence"] = np.asarray(mask, np.uint8)
    malformed = seal_state(tensors, header, encode_tensors(tensors, native=True))
    with pytest.raises(ValueError, match="presence|unit view"):
        from_state(malformed)


@pytest.mark.parametrize("kwargs", [{"w_E": [0, 1]}, {"w_E": [float("nan")]}, {"signs": [0]}, {"signs": [1.5]}])
def test_invalid_append_does_not_stage_partial_state(kwargs):
    rex = RexGraph.from_graph([], [])
    before = state_identity(to_state(rex))
    with pytest.raises((ValueError, TypeError)):
        rex.add_edges([0], [1], relation_ids=[7], **kwargs)
    assert state_identity(to_state(rex)) == before
    assert rex.relation_ids is None


@pytest.mark.parametrize("mapping", [[0, 0], [-2, 0], [0.5, 1], [False, True]])
def test_invalid_cell_maps_are_refused(mapping):
    with pytest.raises(ValueError):
        CellMaps({0: mapping, 1: [], 2: []})


def test_registered_extension_must_supply_a_transport_contract():
    registry = ComponentRegistry()
    registry.register(ComponentCodec("extension", 1, lambda header, names: ()))
    with pytest.raises(ValueError, match="transport contract"):
        registry.capture(RexGraph.from_graph([0], [1]))


def test_adjacency_keeps_near_unit_and_exact_weights_and_isolated_vertices():
    value = np.nextafter(1., 2.)
    matrix = np.zeros((4, 4))
    matrix[0, 1] = value
    matrix[1, 2] = 1
    rex = RexGraph.from_adjacency(matrix, c_channel="count")
    assert rex.nV == 4 and rex.w_E.tolist() == [value, 1.]
    assert rex.relations.weight.presence.tolist() == [True, True]
    assert rex._c_channel == "count"
    exact = np.zeros((3, 3), dtype=object)
    exact[0, 1] = Q(2**100+1, 7)
    assert RexGraph.from_adjacency(exact).relations.weight.values().tolist() == [Q(2**100+1, 7)]


def test_empty_simplicial_import_and_common_carrier():
    assert RexGraph.from_simplicial([], [], []).nE == 0
    rex = RexGraph.from_simplicial([0, 0, 1], [1, 2, 2], [[0, 1, 2]],
                                  w_E=[Absent, 1, Q(2, 3)], c_channel="count")
    assert rex.nF == 1 and rex.chain_valid
    assert rex.relations.weight.values().tolist() == [Absent, 1, Q(2, 3)]
    assert rex._c_channel == "count"


@pytest.mark.parametrize("vertices", [[0., 1., 2.], [False, True, False], [0, 0, 2]])
def test_simplicial_import_refuses_changed_structural_values(vertices):
    with pytest.raises(ValueError):
        RexGraph.from_simplicial([0, 0, 1], [1, 2, 2], [vertices])


def test_staged_batches_own_caller_arrays():
    source, target = np.array([1], np.int32), np.array([2], np.int32)
    ids, weights, signs = np.array([8]), np.array([3]), np.array([-1], np.int32)
    feature = np.array([7])
    rex = RexGraph.from_graph([0], [1], relation_ids=[7])
    rex.add_edges(source, target, relation_ids=ids, w_E=weights, signs=signs, w_boundary={(0, 2): feature})
    source[0], target[0], ids[0], weights[0], signs[0], feature[0] = 9, 9, 9, 9, 1, 9
    assert rex.relation_supports() == [[0, 1], [1, 2]]
    assert rex.relation_keys == (7, 8)
    assert rex.relations.weight.values().tolist() == [Absent, 3]
    assert rex.relations.sign.tolist() == [1, -1]
    assert rex._w_boundary[(1, 2)].tolist() == [7]


def test_invalid_identity_and_face_batch_leave_no_partial_append():
    rex = RexGraph.from_graph([], [])
    with pytest.raises(ValueError):
        rex.add_edges([0], [1], relation_ids=[1.5])
    assert rex.relation_ids is None and rex.nE == 0
    triangle = RexGraph.from_graph([0, 1, 0], [1, 2, 2])
    with pytest.raises(ValueError):
        triangle.add_faces([[0, 1, 2], [0, 1]], [[1, 1, -1], [1]])
    assert triangle.nF == 0 and triangle._pending_faces is None


def test_dictionary_roundtrip_preserves_full_state_and_rejects_conflicting_projection():
    source = rich()
    payload = source.to_dict()
    copied = RexGraph.from_dict(payload)
    assert state_identity(to_state(copied)) == state_identity(to_state(source))
    copied.set_cell_attrs([0], w_E=[1])
    assert source.relations.weight.values()[0] is Absent
    payload["w_E"] = np.array([3, Q(2**100+1, 7)], dtype=object)
    with pytest.raises(ValueError, match="projection"):
        RexGraph.from_dict(payload)


def test_legacy_primary_dictionary_remains_readable():
    original = RexGraph.from_graph([0], [1], w_E=np.array([Q(1, 3)], object))
    payload = original.to_dict()
    payload.pop("rex_state")
    assert RexGraph.from_dict(payload).w_E.tolist() == [Q(1, 3)]


def test_simplicial_triangle_refuses_ambiguous_parallel_relations():
    with pytest.raises(ValueError, match="multiple primary relations"):
        RexGraph.from_simplicial([0, 0, 0, 1], [1, 1, 2, 2], [[0, 1, 2]])


def test_orientation_update_keeps_explicit_pair_share_presence_and_numeric_kind():
    from rexgraph.graph import _set_cell_heads
    rex = RexGraph.from_relations(Relations.from_supports([[0, 1]], shares=[Q(0), Q(1)]))
    _set_cell_heads(rex, [0], [1])
    assert primary_columns(rex) == [{0: Q(1), 1: Q(-1)}]
    assert rex.relations.share.values().tolist() == [Q(0), Q(1)]
    assert rex.relations.share.kind.tolist() == [1, 1]
    assert rex.relations.share.presence.all()
    assert state_identity(to_state(from_state(to_state(rex)))) == state_identity(to_state(rex))
