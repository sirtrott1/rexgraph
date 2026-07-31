"""Step index and wall clock are one coordinate system, not two.

TemporalRex addressed history by step index; the RCDB addressed it by transaction
and validity time. Nothing mapped between them, so "reconstruct the complex as it
stood when that measurement was taken" had no answer -- you could get the step or
the timestamp, never both. For an experiment whose timepoints are hours or passages
rather than integers, that gap is the whole difficulty.

A store with no times supplied behaves exactly as before: the step index IS the
time, which is the identity bridge, so nothing that worked stops working.
"""

import numpy as np
import pytest

from rexgraph.graph import RexGraph, TemporalRex


def _rex(n):
    return RexGraph(sources=np.arange(n, dtype=np.int32),
                    targets=np.arange(1, n + 1, dtype=np.int32))


def _store(times=None, n=4):
    tr = TemporalRex([])
    for k in range(n):
        tr.append_snapshot(_rex(k + 2), at=None if times is None else times[k])
    return tr


def test_without_timestamps_the_step_index_is_the_time():
    """The identity bridge. Nothing that worked before changes."""
    tr = _store()
    assert list(tr.times) == [0, 1, 2, 3]
    assert tr.time_at(2) == 2
    assert tr.step_at(2) == 2


def test_a_snapshot_can_carry_the_moment_it_was_taken():
    tr = _store(times=[10.0, 20.0, 35.0, 60.0])
    assert list(tr.times) == [10.0, 20.0, 35.0, 60.0]
    assert tr.time_at(2) == 35.0


def test_step_at_finds_the_snapshot_current_at_an_instant():
    """Between measurements, the complex is whatever the last measurement said."""
    tr = _store(times=[10.0, 20.0, 35.0, 60.0])
    assert tr.step_at(10.0) == 0
    assert tr.step_at(19.9) == 0
    assert tr.step_at(20.0) == 1
    assert tr.step_at(59.0) == 2
    assert tr.step_at(1000.0) == 3


def test_before_the_first_measurement_there_is_no_step():
    tr = _store(times=[10.0, 20.0, 35.0, 60.0])
    assert tr.step_at(9.9) is None


def test_reconstruct_at_time_is_the_bridge():
    tr = _store(times=[10.0, 20.0, 35.0, 60.0])
    assert tr.reconstruct_at_time(35.0).nE == tr.reconstruct_at(2).nE
    assert tr.reconstruct_at_time(59.9).nE == tr.reconstruct_at(2).nE
    assert tr.reconstruct_at_time(5.0) is None


def test_times_must_not_go_backwards():
    """An out-of-order timestamp makes step_at ambiguous, so it is refused at the
    point the mistake is made rather than silently mis-answering later."""
    tr = TemporalRex([])
    tr.append_snapshot(_rex(2), at=100.0)
    with pytest.raises(ValueError):
        tr.append_snapshot(_rex(3), at=50.0)


def test_the_grid_can_be_read_on_the_real_time_axis():
    tr = _store(times=[10.0, 20.0, 35.0, 60.0])
    g = tr.bioes_grid()
    assert list(g["times"]) == [10.0, 20.0, 35.0, 60.0]
    assert g["tags"].shape[0] == len(g["times"])


def test_the_delta_tensor_reports_when_in_real_time():
    tr = _store(times=[10.0, 20.0, 35.0, 60.0])
    d = tr.delta_tensor()
    assert "when" in d
    assert len(d["when"]) == len(d["t"])
    for t, when in zip(d["t"], d["when"]):
        assert when == tr.time_at(int(t))


def test_times_survive_the_rcdb_round_trip():
    """The point of the bridge: a stored TemporalRex keeps the coordinate system
    that lets its steps be lined up against anything else recorded in wall clock."""
    from agent import rcdb

    tr = _store(times=[10.0, 20.0, 35.0, 60.0])
    store = rcdb.MemoryStore()
    store.put("hist", tr)
    back = store.get("hist")
    assert list(back.times) == [10.0, 20.0, 35.0, 60.0]
    assert back.step_at(59.0) == 2


def test_the_signature_exposes_the_time_span_for_querying():
    from agent import rcdb

    tr = _store(times=[10.0, 20.0, 35.0, 60.0])
    store = rcdb.MemoryStore()
    store.put("hist", tr)
    sig = store.get_record("hist").signature
    assert sig["object_type"] == "TemporalRex"
    assert sig["t_first"] == 10.0 and sig["t_last"] == 60.0
