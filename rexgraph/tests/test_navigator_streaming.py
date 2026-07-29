import numpy as np
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.flow.navigator import (
    FieldNavigator, changed_edges, removed_region_for, EdgeChange,
)


def _rex(src, tgt):
    return RexGraph(sources=np.asarray(src, np.int32), targets=np.asarray(tgt, np.int32))


class _AlwaysGate:
    def observe(self, rex):
        return {"H_T": 0.0, "delta": 1.0, "event": True}


class _IdleGate:
    def observe(self, rex):
        return {"event": False}


def test_step_active_unions_added_and_removed_region():
    r = _rex([0, 1, 2, 3], [1, 2, 3, 4])            # 4 edges
    nav = FieldNavigator(gate=_AlwaysGate())
    change = EdgeChange(added=np.array([0], np.int64), removed=np.array([], np.int64))
    out = nav.step(r, change, removed_region=np.array([2], np.int64))
    assert out["event"] is True
    assert out["region"].tolist() == [0, 2]         # added union removed-region, uniqued+sorted
    assert nav.flow_calls == 1
    assert "draining" in out["flow"] and "circulating" in out["flow"]


def test_step_idle_does_no_flow():
    r = _rex([0, 1], [1, 2])
    nav = FieldNavigator(gate=_IdleGate())
    out = nav.step(r, EdgeChange(added=np.array([0], np.int64), removed=np.array([], np.int64)))
    assert out == {"event": False}
    assert nav.flow_calls == 0


def test_step_change_none_is_all_added():
    r = _rex([0, 1, 2], [1, 2, 3])
    nav = FieldNavigator(gate=_AlwaysGate())
    out = nav.step(r, change=None)
    assert out["region"].tolist() == [0, 1, 2]


def _snaps():
    S = [
        ([0, 0, 1], [1, 2, 3]),
        ([0, 0, 1, 2], [1, 2, 3, 4]),
        ([0, 0, 1, 2, 3], [1, 2, 3, 4, 5]),
        ([0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]),
        ([0, 0, 1, 2, 3, 4, 4], [1, 2, 3, 4, 5, 6, 0]),        # cycle-close (surprise)
        ([0, 0, 1, 2, 3, 4, 4, 5], [1, 2, 3, 4, 5, 6, 0, 7]),
    ]
    return TemporalRex([(np.asarray(s, np.int32), np.asarray(t, np.int32)) for s, t in S])


def test_removed_region_maps_keys_to_incident_current_edges():
    prev = _rex([0, 1, 2], [1, 2, 3])               # edges (0,1)(1,2)(2,3)
    curr = _rex([0, 2], [1, 3])                      # (1,2) removed; (0,1) idx0, (2,3) idx1
    ch = changed_edges(prev, curr)
    assert ch.removed.size == 1                      # exactly (1,2) died
    region = removed_region_for(prev, curr, ch.removed)
    # removed edge (1,2) touched vertices 1 and 2; current edges incident: idx0 (0,1) via v1, idx1 (2,3) via v2
    assert region.tolist() == [0, 1]


def test_removed_region_empty_when_no_removals():
    prev = _rex([0, 1], [1, 2])
    curr = _rex([0, 1, 2], [1, 2, 3])                # pure growth, nothing removed
    ch = changed_edges(prev, curr)
    assert removed_region_for(prev, curr, ch.removed).tolist() == []


def test_run_threads_real_nonempty_removed_region():
    # snapshot 0 has edge (1,2); snapshot 1 DROPS it (genuine removal), keeps/adds others
    S = [([0, 1, 2], [1, 2, 3]),        # edges (0,1)(1,2)(2,3)
         ([0, 2, 3], [1, 3, 4])]        # (1,2) removed; (3,4) added; (0,1)(2,3) persist
    trex = TemporalRex([(np.asarray(s, np.int32), np.asarray(t, np.int32)) for s, t in S])
    prev, curr = trex.at(0), trex.at(1)
    change = changed_edges(prev, curr)
    assert change.removed.size >= 1                      # a real removal happened
    expected_rr = removed_region_for(prev, curr, change.removed)
    assert expected_rr.size >= 1                         # non-empty: this is the path under test
    expected_region = np.unique(np.concatenate(
        [np.asarray(change.added, np.int64), expected_rr]))
    log = FieldNavigator(gate=_AlwaysGate()).run(trex)
    step1 = log[1]
    assert step1["event"] is True
    assert step1["region"].tolist() == expected_region.tolist()   # run threaded the real removed_region


def test_run_equals_manual_step_sequence():
    trex = _snaps()
    log = FieldNavigator(gate=_AlwaysGate()).run(trex)
    nav_step = FieldNavigator(gate=_AlwaysGate())    # _AlwaysGate is stateless -> identical
    manual = []
    for i in range(trex.T):
        rex_i = trex.at(i)
        if i > 0:
            prev = trex.at(i - 1)
            change = changed_edges(prev, rex_i)
            rr = removed_region_for(prev, rex_i, change.removed)
        else:
            change, rr = None, None
        manual.append({"t": i, **nav_step.step(rex_i, change, rr)})
    assert [e["event"] for e in log] == [e["event"] for e in manual]
    for a, b in zip(log, manual):
        assert a["t"] == b["t"]
        if a["event"]:
            assert a["region"].tolist() == b["region"].tolist()
