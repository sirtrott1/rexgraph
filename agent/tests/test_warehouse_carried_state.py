"""Warehouse publication restricts the actual declared primary complex."""
from contextlib import closing
from fractions import Fraction as Q

import numpy as np
import pytest

from agent.warehouse.source import EdgeData, edge_complex
from agent.warehouse.assemble import _subcomplex
from rexgraph import Absent, RexGraph
from rexgraph.object_identity import object_digest


def data():
    return EdgeData(np.array([0, 1, 0]), np.array([2, 3, 3]),
                    np.array([Q(1,10), Absent, 2**53+1], dtype=object), 2, 3)


def test_primary_weights_presence_and_declared_isolates_survive():
    ed = data(); graph = edge_complex(ed)
    assert graph.nV == 5
    assert graph.relations.weight.values().tolist() == [Q(1,10), Absent, 2**53+1]
    ed.weight[0] = Q(3,7)
    assert graph.relations.weight.values()[0] == Q(1,10)


def test_tier_preserves_actual_primary_attributes_channels_and_storage():
    from rcdb import MemoryStore
    ed = data(); graph = RexGraph.from_relations(edge_complex(ed).relations, c_channel="count")
    graph.set_provenance({"warehouse": {"run": 2**53+1}, "vertex_labels": ["a", "b", "c", "d", "isolated"]})
    graph.attach_metadata(1, 2, "prediction", Q(1,3))
    tier = _subcomplex(ed, np.array([0, 2]), rex=graph)
    assert tier.nV == 3 and tier.nE == 2
    assert tier.relations.weight.values().tolist() == [Q(1,10), 2**53+1]
    assert tier.get_metadata(1, 1, "prediction") == Q(1,3)
    assert tier.provenance["warehouse"] == graph.provenance["warehouse"]
    assert tier.vertex_labels == ("a", "c", "d") and tier.c_channel == "count"
    with closing(MemoryStore()) as store:
        store.put("tier", tier, analytics=False)
        assert object_digest(store.read_record("tier").value) == object_digest(tier)


def test_tier_fallback_and_empty_tier_preserve_metric_contract():
    ed = data()
    assert _subcomplex(ed, np.array([1])).relations.weight.values()[0] is Absent
    assert _subcomplex(ed, np.array([], dtype=int)).nE == 0


def test_tier_keeps_surviving_faces_and_their_owned_attributes():
    ed = EdgeData(np.array([0,1,0]), np.array([1,2,2]), np.array([Q(1,10)]*3, dtype=object), 3, 0)
    graph = RexGraph.from_simplicial(ed.src_idx, ed.dst_idx, np.array([[0,1,2]]))
    graph.attach_metadata(2, 0, "face", {"certificate": Q(1,3)})
    full = _subcomplex(ed, np.arange(3), rex=graph)
    assert full.nF == 1 and full.get_metadata(2, 0, "face") == {"certificate": Q(1,3)}
    assert _subcomplex(ed, np.array([0,1]), rex=graph).nF == 0


@pytest.mark.parametrize("mask", [np.array([2, 0]), np.array([1, 1]), np.array([-1]),
    np.array([3]), np.array([.5]), np.array([True, False, True])])
def test_invalid_tier_addresses_are_refused(mask):
    with pytest.raises(ValueError, match="tier indices"):
        _subcomplex(data(), mask)
