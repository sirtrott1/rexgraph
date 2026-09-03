"""An edit history is a temporal complex, not a log beside one.

`record` builds a complex from labels through a lineage adapter, which is right when the
caller has a list of things and no complex. Most callers have the complex: an edit to a
stored graph, an agent's state, a hive's placement. Flattening one to labels so an adapter
can rebuild a different one loses what it knew, so `record_complex` takes it directly.

The lineage stays one object with two coordinates, as `record` leaves it:

    version   the RCDB chain, bitemporal, read with get_version or as_of
    step      the position inside the TemporalRex, read with at or step_at

so any past state reconstructs as a RexGraph and can be analysed, queried or drawn like
any other, and `append_snapshot` keeps the checkpoint/delta index incrementally, a diff
per edit rather than a rebuild.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from agent import work_recorder as wr
from rexgraph.graph import RexGraph, TemporalRex


@pytest.fixture
def lineage():
    return f"test-edits-{int(time.time() * 1e6)}"


def _edits():
    """Three states of one complex: a path, the cycle it closes, a 3-ary relation."""
    first = RexGraph(sources=np.array([0, 1], dtype=np.int32),
                     targets=np.array([1, 2], dtype=np.int32))
    second = first.insert_edges(np.array([2], dtype=np.int32),
                                np.array([0], dtype=np.int32))
    return [first, second, second.insert_relations([[0, 1, 2]])]


def _record(lineage, states, start=1000.0):
    return [wr.record_complex("edit", rex, lineage_id=lineage, force=True,
                              when=start + i)
            for i, rex in enumerate(states)]


#### the history is the store


def test_each_edit_is_a_version_and_a_step(lineage):
    infos = _record(lineage, _edits())
    assert [i["version"] for i in infos] == [1, 2, 3]
    assert [i["step"] for i in infos] == [0, 1, 2]


def test_the_lineage_is_one_temporal_complex(lineage):
    _record(lineage, _edits())
    assert isinstance(wr._store().get(lineage), TemporalRex)


def test_the_history_reads_back_in_order(lineage):
    _record(lineage, _edits())
    steps = wr.history(lineage)
    assert [h["step"] for h in steps] == [0, 1, 2]
    assert [h["nE"] for h in steps] == [2, 3, 4]
    assert [h["at"] for h in steps] == [1000.0, 1001.0, 1002.0]


#### any past state is a complex again


def test_a_past_state_reconstructs(lineage):
    _record(lineage, _edits())
    step, rex = wr.state_at(lineage, 1001.0)
    assert step == 1
    assert rex.nE == 3


def test_the_topology_of_each_state_is_its_own(lineage):
    """b1 goes 0 -> 1 when the cycle closes, which is the point of keeping the state
    rather than a description of it."""
    _record(lineage, _edits())
    assert wr.state_at(lineage, 1000.0)[1].betti[1] == 0
    assert wr.state_at(lineage, 1001.0)[1].betti[1] == 1


def test_a_branching_relation_survives_the_round_trip(lineage):
    """Through append_snapshot, the checkpoint/delta index and the store. A 3-ary
    relation that came back 2-ary would make the history a record of a different graph."""
    _record(lineage, _edits())
    _step, rex = wr.state_at(lineage, 1002.0)
    rex._ensure_clean()
    assert sorted(np.diff(np.asarray(rex.boundary_ptr)).tolist()) == [2, 2, 2, 3]


def test_a_past_state_draws_like_any_other(lineage):
    """Which is what makes an edit history viewable rather than only queryable."""
    from agent.graph_view import render_payload
    from agent.render_svg import render_svg

    _record(lineage, _edits())
    _step, rex = wr.state_at(lineage, 1002.0)
    svg = render_svg(render_payload(rex))
    assert svg.count("<title>relation") == rex.nE


#### the contract


def test_an_unknown_kind_is_refused(lineage):
    with pytest.raises(ValueError, match="unknown kind"):
        wr.record_complex("whatever", _edits()[0], lineage_id=lineage, force=True)


def test_the_complex_kinds_cover_the_callers():
    assert {"edit", "agent-state", "hive-state"} <= set(wr.COMPLEX_KINDS)


def test_nothing_is_recorded_without_a_complex(lineage):
    assert wr.record_complex("edit", None, lineage_id=lineage, force=True) is None


def test_repeated_states_are_still_separate_edits(lineage):
    """Unlike `record`, which de-duplicates. Two edits producing the same complex are
    still two edits, and a history that drops one is not a history."""
    first = _edits()[0]
    infos = _record(lineage, [first, first])
    assert [i["step"] for i in infos] == [0, 1]


def test_the_clock_may_not_run_backwards(lineage):
    """`step_at` would be ambiguous, so the store refuses rather than guessing."""
    _record(lineage, _edits())
    with pytest.raises(ValueError, match="precedes step"):
        wr.record_complex("edit", _edits()[0], lineage_id=lineage, force=True, when=999.0)


def test_the_meta_carries_the_shape_of_each_state(lineage):
    _record(lineage, _edits())
    record = wr._store().get_record(lineage)
    assert record.meta["kind"] == "edit"
    assert record.meta["step"] == 2
