"""Retrieval by closure, and reading a reasoning path.

Ranking answers "which items are most like the query". Closure answers a different
question: "what is the whole of what this complex says about these entities, and how do I
know I have all of it". There is no top_k, because top_k is a number someone picked; the
boundary is where the reading stops moving.

The same machinery reads a reasoning path, because a conversation is one: turns are cells
and transitions are relations. A path that only advances is pure gradient. A path that
returns to an earlier turn has a cycle, and the cycle is not explainable by any potential,
so it lands in the harmonic part. betti_1 counts the returns exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from agent.graph_view import flow_positions
from agent.query_engine import retrieve_closure
from rexgraph.graph import RexGraph


def _chain(pairs):
    rex = RexGraph(sources=np.array([a for a, _ in pairs], dtype=np.int32),
                   targets=np.array([b for _, b in pairs], dtype=np.int32))
    rex._ensure_clean()
    return rex


#### closure retrieval


@pytest.fixture
def two_hubs():
    """Two hubs that share a leaf, so one of them reaches further than the other."""
    rex = RexGraph(sources=np.array([0, 0, 0, 3, 3], dtype=np.int32),
                   targets=np.array([1, 2, 9, 2, 4], dtype=np.int32))
    rex._ensure_clean()
    return rex


def test_it_returns_the_relations_rather_than_a_ranking(two_hubs):
    out = retrieve_closure(two_hubs, [0])
    assert out["n_relations"] > 0
    assert sorted(out["relations"]) == out["relations"]


def test_each_seed_keeps_its_own_depth(two_hubs):
    """A seed that closes at 1 and one that needs 3 are different facts about those
    entities, and averaging them would lose the more interesting one."""
    out = retrieve_closure(two_hubs, [0, 4])
    assert {c["seed"] for c in out["closures"]} == {0, 4}
    assert all("depth" in c for c in out["closures"])


def test_the_audit_trail_says_what_arrived_when(two_hubs):
    steps = retrieve_closure(two_hubs, [0])["closures"][0]["steps"]
    assert [s["depth"] for s in steps] == list(range(1, len(steps) + 1))
    assert all("betti" in s for s in steps)


def test_labels_come_back_when_they_are_known(two_hubs):
    labels = [f"e{i}" for i in range(two_hubs.nV)]
    assert retrieve_closure(two_hubs, [0], labels=labels)["closures"][0]["label"] == "e0"


def test_no_seeds_is_answered_not_raised(two_hubs):
    out = retrieve_closure(two_hubs, [])
    assert out["n_relations"] == 0 and "reason" in out


def test_an_unclosed_seed_is_named(two_hubs):
    """So a caller knows the context is incomplete rather than assuming it is not."""
    long_chain = _chain([(i, i + 1) for i in range(10)])
    out = retrieve_closure(long_chain, [0], max_depth=2)
    assert out["all_converged"] is False
    assert out["unclosed"] == [0]


#### the same reading on a reasoning path


def test_a_path_that_only_advances_is_pure_gradient():
    rex = _chain([(i, i + 1) for i in range(6)])
    flow = flow_positions(rex, [1.0] * 6)
    assert rex.betti[1] == 0
    assert flow["decomposition"]["gradient"] == pytest.approx(1.0)


def test_a_return_puts_content_where_no_potential_can_explain_it():
    """The model went back round. A constant flow around a loop is exactly what a
    potential cannot account for, so it lands in the harmonic part."""
    rex = _chain([(i, i + 1) for i in range(6)] + [(6, 2)])
    flow = flow_positions(rex, [1.0] * 7)
    assert rex.betti[1] == 1
    assert flow["decomposition"]["harmonic"] > 0.5


def test_betti_counts_the_returns_exactly():
    once = _chain([(i, i + 1) for i in range(6)] + [(6, 2)])
    twice = _chain([(i, i + 1) for i in range(6)] + [(6, 2), (5, 1)])
    assert once.betti[1] == 1
    assert twice.betti[1] == 2


def test_more_returns_leave_less_explained_by_progress():
    def gradient(pairs):
        return flow_positions(_chain(pairs), [1.0] * len(pairs))["decomposition"]["gradient"]

    straight = [(i, i + 1) for i in range(6)]
    assert gradient(straight) > gradient(straight + [(6, 2)])
    assert gradient(straight + [(6, 2)]) > gradient(straight + [(6, 2), (5, 1)])


def test_a_varying_signal_splits_progress_from_circulation():
    """The useful reading: how much of this reasoning was going forward."""
    rex = _chain([(i, i + 1) for i in range(6)] + [(6, 2)])
    decomposition = flow_positions(rex, [0.9, 0.8, 0.7, 0.3, 0.4, 0.6, 0.2])["decomposition"]
    assert 0.0 < decomposition["gradient"] < 1.0
    assert decomposition["gradient"] + decomposition["harmonic"] == pytest.approx(1.0, abs=1e-6)


def test_an_open_loop_is_harmonic_and_not_curl():
    """Worth keeping straight: curl needs a FACE. A cycle nobody has closed is a hole,
    which is a different statement from circulation inside something accounted for."""
    rex = _chain([(i, i + 1) for i in range(6)] + [(6, 2)])
    decomposition = flow_positions(rex, [1.0] * 7)["decomposition"]
    assert decomposition["curl"] == pytest.approx(0.0)
    assert decomposition["harmonic"] > 0.0
