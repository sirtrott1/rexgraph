"""Agent normalizes into core Relations without changing source declarations."""
from fractions import Fraction as Q

import numpy as np
import pytest

from agent.adapters import EdgeConstruction
from agent.auto import build_rex_from_edges
from rexgraph import Absent
from rexgraph.sealed_state import to_sealed_state
from rexgraph.state import from_state


def construction(**kwargs):
    values = dict(sources=np.array([0], np.int32), targets=np.array([1], np.int32),
                  weights=np.array([1.0000000001]), signs=np.array([-1.]),
                  type_labels=np.array([0], np.int32), vertex_labels=["a", "b", "c", "d", "isolated"],
                  n_types=1, type_names=["pair"], branching=[[0, 1, 2, 3]],
                  relation_ids=["pair/id", "branch/id"], origin="declared source")
    values.update(kwargs)
    return EdgeConstruction(**values)


def test_missing_branch_metric_is_absent_and_near_unit_pair_is_not_rounded():
    r = build_rex_from_edges(construction())
    assert r.nV == 5 and r.nE == 2
    assert r.w_E[0] == 1.0000000001
    assert r.relations.weight.presence.tolist() == [True, False]
    assert r.relations.weight.values()[1] is Absent
    assert r.relation_keys == ("pair/id", "branch/id")
    assert r.relations.relation_type == ("pair", Absent)
    back = from_state(to_sealed_state(r))
    assert back.relations.weight.presence.tolist() == [True, False]
    assert back.provenance["origin"] == "declared source"


def test_branch_weight_sign_shares_and_identity_are_independent():
    r = build_rex_from_edges(construction(branching_weights=[Q(2, 3)], branching_signs=[-1],
                                        shares=[Absent, Absent, 0, Q(1, 4), Q(1, 2), Q(1, 4)],
                                        embedding=[[Q(i, 10), 0] for i in range(5)],
                                        vertex_ids=["a/id", "b/id", "c/id", "d/id", "isolated/id"]))
    assert r.relations.weight.values()[1] == Q(2, 3)
    assert r.relations.sign.tolist() == [-1, -1]
    assert r._exact_column_norms_B1() == [2, Q(11, 8)]
    back = from_state(to_sealed_state(r))
    assert back.embedding == r.embedding
    assert back.relations.vertices.ids == r.relations.vertices.ids


@pytest.mark.parametrize("change", [{"sources": np.array([.5])}, {"signs": np.array([0.])},
                                   {"weights": []}, {"branching_weights": [1, 2]},
                                   {"type_labels": np.array([2])}, {"vertex_labels": ["a"]}])
def test_bad_declarations_refuse_before_graph_construction(change):
    with pytest.raises((ValueError, TypeError)):
        build_rex_from_edges(construction(**change))
