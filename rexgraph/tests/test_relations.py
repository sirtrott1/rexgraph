"""Public relation construction preserves declarations through real operations."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import Absent, Relations, RexGraph, VertexTable
from rexgraph.native_rank import primary_columns
from rexgraph.sealed_state import state_identity, to_sealed_state
from rexgraph.state import from_state


def declared():
    return Relations.from_supports(
        [[0, 1], [0, 1, 2, 3]],
        vertices=VertexTable(("one", "two", "three", "four", "isolated"),
                             ("same", "same", "c", "d", "alone"),
                             (("A",), ("B",), (), (), ())),
        weights=[Absent, Q(2, 3)], signs=[1, -1],
        shares=[Absent, Absent, 0, Q(1, 4), Q(1, 2), Q(1, 4)],
        relation_ids=["pair", "branch"], relation_types=[Absent, "group"],
        embedding=[[Q(i, 10), 0] for i in range(5)],
        attributes={1: {1: {"assertion": Q(1, 7)}}}, provenance={"origin": "fixture"},
    )


def test_mixed_relation_construction_and_native_roundtrip():
    r = RexGraph.from_relations(declared(), g_channel="normalized", c_channel="count")
    assert r.nV == 5 and r.nE == 2
    assert primary_columns(r)[1] == {0: Q(-1), 1: Q(1, 4), 2: Q(1, 2), 3: Q(1, 4)}
    assert r._exact_column_norms_B1() == [2, Q(11, 8)]
    assert r.relations.weight.values().tolist() == [Absent, Q(2, 3)]
    assert r.relation_keys == ("pair", "branch")
    assert r.relations.vertices.ids == declared().vertices.ids
    assert r.relations.vertices.aliases[0] == ("A",)
    assert r.get_metadata(1, 1, "assertion") == Q(1, 7)
    state = to_sealed_state(r)
    back = from_state(state)
    assert state_identity(to_sealed_state(back)) == state_identity(state)
    assert back.embedding == r.embedding
    assert back.relations.weight.presence.tolist() == [False, True]
    assert back.vertex_labels == r.vertex_labels


@pytest.mark.parametrize("support", [[0., 1.], [False, True], [-1, 1], [0, 2**31], [], [0, 1, 1]])
def test_invalid_support_refused_before_native_execution(support):
    with pytest.raises((ValueError, TypeError)):
        RexGraph.from_relations(Relations.from_supports([support], n_vertices=2))


@pytest.mark.parametrize("weights", [[.1], [True], [float("nan")], [float("inf")]])
def test_exact_construction_requires_declared_number_conversion(weights):
    with pytest.raises((ValueError, TypeError)):
        Relations.from_supports([[0, 1]], weights=weights)


def test_identity_has_no_label_or_generated_value_fallback():
    r = RexGraph.from_relations(Relations.from_supports([[0, 1]], n_vertices=3))
    assert r.relations.vertices.ids == (Absent,)*3
    assert r.relation_keys == (Absent,)
    with pytest.raises(ValueError, match="duplicate"):
        VertexTable(("A", "A"))


def test_pair_head_and_explicit_share_survive_native_canonicalization():
    source = Relations.from_supports([[0, 1]], heads=[1], shares=[1, 0], relation_ids=["oriented"])
    r = RexGraph.from_relations(source)
    assert primary_columns(r) == [{0: Q(1), 1: Q(-1)}]
    assert r.relations.share.values().tolist() == [0, 1]
    assert r.relations.share.presence.all()
    assert state_identity(to_sealed_state(from_state(to_sealed_state(r)))) == state_identity(to_sealed_state(r))


def test_cell_compaction_transports_source_tables_and_embedding():
    r = RexGraph.from_relations(declared())
    r.remove_edges(np.array([1, 0], np.int32))
    r.compact()
    assert r.relation_keys == ("branch",)
    assert r.relations.relation_type == ("group",)
    assert r.relations.weight.values().tolist() == [Q(2, 3)]
    assert r.relations.vertices.ids == ("one", "two", "three", "four")
    assert len(r.embedding) == r.nV
    assert r.get_metadata(1, 0, "assertion") == Q(1, 7)
    assert from_state(to_sealed_state(r)).relation_keys == ("branch",)


@pytest.mark.parametrize("types,faces", [([0, 0, 0], 1), ([0, 1, 0], 0)])
def test_typed_faces_preserve_all_primary_carriers(types, faces):
    source = Relations.from_supports([[0, 1], [1, 2], [0, 2]], n_vertices=4,
                                    weights=[Q(2, 3), 1, 1], relation_ids=[11, 12, 13],
                                    relation_types=["a", "b", "a"],
                                    embedding=[[i, 0] for i in range(4)],
                                    attributes={1: {0: {"note": "retained"}}})
    r = RexGraph.from_relations(source, g_channel="normalized", c_channel="count")
    filled = r.typed_face_selection(np.array(types, np.int32))
    assert filled.nF == faces and filled.nV == 4 and filled.nE == 3
    assert filled.relation_keys == (11, 12, 13)
    assert filled.embedding == r.embedding
    assert filled.get_metadata(1, 0, "note") == "retained"
    assert filled._g_channel == r._g_channel and filled._c_channel == r._c_channel
    assert filled.chain_valid
    assert r.nF == 0
