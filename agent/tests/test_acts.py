"""Acts as relations: the orientation an event carries, and the share a k-ary one carries.

An entity and a verb say that something happened and cannot say which way it went, so
nothing composes and no cycle reads as consistent or not. These pin the two halves that
fixes: the sign, which is what the frustration channel reads, and the share, which is what
lets a relation over k participants stay one relation.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
from agent.agent_complex import act_complex

from agent import activity
from rexgraph.graph import RexGraph


@pytest.fixture(autouse=True)
def _journal(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_ACTIVITY_JOURNAL", str(tmp_path / "activity.jsonl"))
    activity.reset()
    yield tmp_path
    activity.get_log().close()


def test_an_unoriented_event_is_recorded_exactly_as_before(_journal):
    """Every caller that predates orientation writes the line it always wrote, so an old
    reader is unaffected and the field is absent rather than empty."""
    ev = activity.record("worker:coder", "dispatch", detail={"q": "x"})
    assert ev.oriented is False
    assert set(ev.public()) == {"ts", "entity", "scope", "action", "detail"}


def test_an_oriented_event_carries_its_pair(_journal):
    ev = activity.record("worker:mule", "deliver", on="hive:beta", flow="write")
    assert ev.oriented and ev.public()["on"] == "hive:beta"
    line = json.loads((_journal / "activity.jsonl").read_text().strip().splitlines()[-1])
    assert line["on"] == "hive:beta" and line["flow"] == "write"


def test_an_unknown_flow_is_dropped_rather_than_believed(_journal):
    """The log must never be why a caller fails, and a direction that means nothing is
    worse than none: it would be read as a boundary."""
    ev = activity.record("worker:mule", "deliver", on="hive:beta", flow="sideways")
    assert ev.flow == "" and ev.oriented is False


def test_direction_decides_the_frustration_and_existence_cannot_see_it():
    """The control that makes the whole exercise worth it. Same acts, same betti, and the
    frustration channel reads 0 without direction and lights up with it."""
    acts = [{"entity": "mule", "on": "beta", "flow": "write"},
            {"entity": "ox", "on": "beta", "flow": "read"}]
    rex, labels = act_complex(acts)
    assert rex.nV == 3 and rex.nE == 2
    F = np.asarray(rex.frustration_exact.diagonal(), float)
    assert F.sum() > 0, "a write and a read meeting at one object is the contested case"

    flat = RexGraph.from_graph(np.array([0, 1], np.int32), np.array([2, 2], np.int32))
    assert flat.betti == rex.betti, "existence cannot tell the two apart"
    assert np.asarray(flat.frustration_exact.diagonal(), float).sum() == 0


def test_two_writers_agreeing_are_not_frustrated():
    """The half worth checking as hard: a channel that fires on everything reads nothing."""
    rex, _ = act_complex([{"entity": "mule", "on": "beta", "flow": "write"},
                          {"entity": "ox", "on": "beta", "flow": "write"}])
    assert np.asarray(rex.frustration_exact.diagonal(), float).sum() == 0


def test_events_without_an_orientation_are_skipped_not_guessed(_journal):
    activity.record("worker:mule", "deliver")
    activity.record("worker:mule", "deliver", on="hive:beta", flow="write")
    rex, labels = act_complex(activity.get_log().events())
    assert rex.nE == 1 and set(labels) == {"worker:mule", "hive:beta"}
    assert act_complex([{"entity": "a", "on": "b"}]) == (None, [])


def test_a_courier_trip_records_the_two_ends_it_actually_is(_journal, tmp_path):
    from agent.courier import Courier

    from agent import rcdb
    a, b = rcdb.open_store("memory://"), rcdb.open_store("memory://")
    v = np.arange(3, dtype=np.int32)
    a.put("r", RexGraph(sources=v, targets=np.roll(v, -1).astype(np.int32)), tags=["t"])
    c = Courier("mule")
    c.attach_store("alpha", a); c.attach_store("beta", b)
    c.deliver("alpha", "beta")

    acts = [e for e in activity.get_log().events(entity="worker:mule") if e.get("on")]
    assert {(e["on"], e["flow"]) for e in acts} == {("hive:alpha", "read"),
                                                    ("hive:beta", "write")}


#### the share: what keeps a relation over k participants one relation

def _k_ary(k):
    return RexGraph.from_hypergraph(np.array([0, k], np.int32),
                                    np.arange(k, dtype=np.int32))


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_a_k_ary_relation_is_signed_and_sums_to_zero(k):
    """The condition that makes a column a boundary. Nothing about arity enters it."""
    g = _k_ary(k)
    col = np.asarray(g.B1.todense() if hasattr(g.B1, "todense") else g.B1)[:, 0]
    assert abs(col.sum()) < 1e-12
    assert col.min() == pytest.approx(-1.0)
    assert float(col @ col) == pytest.approx(1 + 1 / (k - 1)), "T is the share's concentration"


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_zero_column_sum_propagates_to_zero_row_sum(k):
    """The identity that links the levels: sum(c)=0 makes row i of L0 vanish."""
    L0 = np.asarray(_k_ary(k).L0_sparse.todense())
    assert np.abs(L0.sum(1)).max() < 1e-12
    assert np.linalg.matrix_rank(L0) == 1


def test_expanding_a_relation_reports_holes_that_are_not_there():
    """Why the branching edge is not a convenience. Five 3-ary trips carry no cycle;
    clique expansion invents ten edges and nine of them, star expansion invents five
    vertices and four."""
    trips = [("mule", "alpha", "beta"), ("ox", "beta", "gamma"), ("ant", "gamma", "alpha"),
             ("ox", "beta", "delta"), ("mule", "delta", "alpha")]
    names = sorted({x for t in trips for x in t})
    ix = {v: i for i, v in enumerate(names)}

    rel = RexGraph.from_hypergraph(
        np.arange(0, 3 * len(trips) + 1, 3, dtype=np.int32),
        np.array([ix[x] for t in trips for x in t], np.int32))
    assert rel.nE == 5 and rel.betti[1] == 0

    cl = [(a, b) for t in trips for a, b in itertools.combinations(sorted(t), 2)]
    clq = RexGraph.from_graph(np.array([ix[a] for a, _ in cl], np.int32),
                              np.array([ix[b] for _, b in cl], np.int32))
    assert clq.nE == 15 and clq.betti[1] == 9

    hub = {t: len(names) + i for i, t in enumerate(trips)}
    se = [(hub[t], ix[x]) for t in trips for x in t]
    star = RexGraph.from_graph(np.array([a for a, _ in se], np.int32),
                               np.array([b for _, b in se], np.int32))
    assert star.nV == rel.nV + 5 and star.betti[1] == 4


#### grade 2: a carrier goes somewhere and comes back, and the return is what closes

def _acts(*rows):
    out = []
    for actor, on, flow, tid in rows:
        out.append({"entity": actor, "on": on, "flow": flow, "detail": {"trip": tid}})
    return out


def test_direction_is_positional_because_a_column_and_its_negation_are_one_cell():
    """The library canonicalises an explicit sign, which is right: re-signing is a gauge.
    So orientation is WHICH participant carries the single -1, and encoding a read as the
    negated write silently produces the write."""
    same = RexGraph(boundary_ptr=np.array([0, 2, 4], np.int32),
                    boundary_idx=np.array([0, 1, 0, 1], np.int32),
                    signs=np.array([-1.0, 1.0, 1.0, -1.0]))
    B = np.asarray(same.B1.todense() if hasattr(same.B1, "todense") else same.B1)
    assert np.allclose(B[:, 0], B[:, 1]), "opposite signs, one cell"

    rex, _ = act_complex(_acts(("m", "a", "write", "t1"), ("m", "a", "read", "t1")))
    Bo = np.asarray(rex.B1.todense() if hasattr(rex.B1, "todense") else rex.B1)
    assert not np.allclose(Bo[:, 0], Bo[:, 1]), "position must keep them apart"


def test_a_round_trip_closes_and_its_cycles_bound():
    rex, _ = act_complex(_acts(("m", "a", "read", "t1"), ("m", "b", "write", "t1"),
                               ("m", "b", "read", "t2"), ("m", "a", "write", "t2")))
    assert rex.nF > 0, "the return was never given a face"
    B1 = np.asarray(rex.B1.todense() if hasattr(rex.B1, "todense") else rex.B1)
    assert np.abs(B1 @ np.asarray(rex.B2)).max() < 1e-9, "B1 B2 = 0 must hold exactly"
    assert rex.betti[1] == 0
    assert len(rex.cycle_basis) - rex.betti[1] == len(rex.cycle_basis)


def test_circulation_nobody_reciprocated_stays_a_hole():
    """The reading the whole grade exists for. Three carriers hand work around a ring and
    none of them comes back: the cycle is real and it does not bound."""
    rex, _ = act_complex(_acts(("m", "a", "read", "t1"), ("m", "b", "write", "t1"),
                               ("o", "b", "read", "t2"), ("o", "c", "write", "t2"),
                               ("n", "c", "read", "t3"), ("n", "a", "write", "t3")))
    assert rex.nF == 0 and rex.betti[1] == 1


def test_reciprocating_every_leg_does_not_fill_the_ring():
    """Local closure is not global closure: each pair comes home, and the circulation
    around the three still has nothing spanning it."""
    rows = _acts(("m", "a", "read", "t1"), ("m", "b", "write", "t1"),
                 ("o", "b", "read", "t2"), ("o", "c", "write", "t2"),
                 ("n", "c", "read", "t3"), ("n", "a", "write", "t3"),
                 ("m", "b", "read", "t4"), ("m", "a", "write", "t4"),
                 ("o", "c", "read", "t5"), ("o", "b", "write", "t5"),
                 ("n", "a", "read", "t6"), ("n", "c", "write", "t6"))
    rex, _ = act_complex(rows)
    assert rex.nF > 0 and rex.betti[1] == 1, "the ring is a hole the pairs cannot fill"


def test_a_broadcast_is_one_relation_of_its_own_arity():
    """One carrier writing to three destinations performed one act, not three."""
    rex, labels = act_complex([{"entity": "m", "on": ["a", "b", "c"], "flow": "write"}])
    assert rex.nE == 1 and rex.nV == 4
    col = np.asarray(rex.B1.todense() if hasattr(rex.B1, "todense") else rex.B1)[:, 0]
    assert abs(col.sum()) < 1e-12
    assert float(col @ col) == pytest.approx(1 + 1 / 3), "T is the share it carries"


def test_a_gather_is_several_acts_because_a_relation_has_one_distinguished_end():
    rex, _ = act_complex([{"entity": "m", "on": ["a", "b"], "flow": "read"}])
    assert rex.nE == 2


def test_a_hive_can_be_cut_out_of_the_ambient_complex():
    """Workers and hives are cells of one complex, so a slice is a subcomplex rather than
    a second structure built for the purpose."""
    rex, labels = act_complex(_acts(("worker:m", "hive:a", "read", "t1"),
                                    ("worker:m", "hive:b", "write", "t1"),
                                    ("worker:o", "hive:c", "read", "t2"),
                                    ("worker:o", "hive:d", "write", "t2")))
    from agent.agent_complex import slice_participants
    mine = {n for n in labels if n.startswith("hive:")} | {"worker:m"}
    _v, e_mask, _f = slice_participants(rex, labels, mine)
    assert int(e_mask.sum()) == 2, "only the acts wholly inside the slice survive"

    everything = slice_participants(rex, labels, lambda n: True)
    assert int(everything[1].sum()) == rex.nE, "a slice of all of it is all of it"
