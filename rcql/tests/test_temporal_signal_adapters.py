"""RCQL queries temporal tensor fields directly, without lowering to table walks."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.metric_field import MetricCurvature
from rexgraph.temporal_signal import TemporalSignal, TemporalSignalFlow

from rcql import Executor, call, query, source
from rcql.operators import (
    relation_signal,
    signal_at,
    signal_field_flow,
    signal_hodge,
    signal_source,
    temporal_delta,
)
from rcql.types import Exactness


def _timeline():
    timeline = TemporalRex([])
    timeline.append_snapshot(RexGraph(
        sources=np.asarray([0, 1], dtype=np.int32),
        targets=np.asarray([1, 2], dtype=np.int32),
        w_E=np.asarray([2.0, 1.0]),
        signs=np.asarray([1, 1], dtype=np.int32),
    ))
    timeline.append_snapshot(RexGraph(
        sources=np.asarray([1, 1, 2], dtype=np.int32),
        targets=np.asarray([0, 2, 3], dtype=np.int32),
        w_E=np.asarray([3.0, 1.0, 4.0]),
        signs=np.asarray([-1, 1, 1], dtype=np.int32),
    ))
    return timeline


def test_temporal_adapters_expose_event_lookup_source_and_local_field_action():
    timeline = _timeline()
    delta = temporal_delta(timeline, 1)
    assert isinstance(delta, TemporalSignal)
    assert signal_at(timeline, delta, (0, 1)).head == 1

    source_field = signal_source(timeline, delta)
    assert source_field.values.tolist() == [Fraction(2), Fraction(-2), Fraction(-1), Fraction(1)]

    flow = signal_field_flow(timeline, delta)
    assert isinstance(flow, TemporalSignalFlow)
    assert flow.relation_response.values.tolist() == [Fraction(4), Fraction(1), Fraction(2)]

    direct = relation_signal(timeline, delta)
    assert direct.values.tolist() == [1.0, 0.0, 4.0]
    split = signal_hodge(timeline, delta)
    assert set(split) == {"gradient", "curl", "harmonic"}
    assert all(value.source is delta.current for value in split.values())


def test_executor_composes_temporal_delta_and_signal_flow_as_a_native_field_query():
    timeline = _timeline()
    result = Executor(sources={"timeline": timeline}).execute(query(
        source("timeline"),
        call("SIGNAL_FLOW", call("TEMPORAL_DELTA", 1)),
    ))
    flow = result.values[0]
    assert isinstance(flow, TemporalSignalFlow)
    assert flow.returned_boundary.values.tolist() == [Fraction(4), Fraction(-5), Fraction(-1), Fraction(2)]
    assert result.exactness == (Exactness.RATIONAL,)


def test_executor_reads_the_direct_c1_delta_and_marks_its_hodge_action_numerical():
    timeline = _timeline()
    executor = Executor(sources={"timeline": timeline})

    direct = executor.execute(query(
        source("timeline"),
        call("RELATION_SIGNAL", call("TEMPORAL_DELTA", 1)),
    ))
    assert direct.values[0].grade == 1
    assert direct.values[0].values.tolist() == [1.0, 0.0, 4.0]
    assert direct.exactness == (Exactness.APPROXIMATE,)

    split = executor.execute(query(
        source("timeline"),
        call("SIGNAL_HODGE", call("TEMPORAL_DELTA", 1)),
    ))
    assert set(split.values[0]) == {"gradient", "curl", "harmonic"}
    assert split.exactness == (Exactness.APPROXIMATE,)


def test_executor_routes_one_direct_temporal_c1_field_to_its_current_rex_basis():
    """Generic field actions retain the delta's current C1 basis, not the history object."""
    timeline = _timeline()
    executor = Executor(sources={"timeline": timeline})
    direct = call("RELATION_SIGNAL", call("TEMPORAL_DELTA", 1))

    hodge = executor.execute(query(source("timeline"), call("HODGE", direct)))
    curvature = executor.execute(query(source("timeline"), call("METRIC_CURVATURE", direct)))

    assert set(hodge.values[0]) == {"gradient", "curl", "harmonic"}
    assert hodge.exactness == (Exactness.APPROXIMATE,)
    assert isinstance(curvature.values[0], MetricCurvature)
    assert curvature.values[0].metric.source is not timeline
    assert curvature.values[0].metric.source.nE == 3
    assert curvature.exactness == (Exactness.APPROXIMATE,)


def test_one_phrase_reuses_its_exact_temporal_delta_before_multiple_field_actions():
    """A repeated phrase fragment reads one transition, then feeds both C1 actions."""
    timeline = _timeline()
    observed_steps = []
    reconstruct = timeline.reconstruct_at

    def counted_reconstruct(step):
        observed_steps.append(step)
        return reconstruct(step)

    timeline.reconstruct_at = counted_reconstruct
    direct = call("RELATION_SIGNAL", call("TEMPORAL_DELTA", 1))
    result = Executor(sources={"timeline": timeline}).execute(query(
        source("timeline"),
        call("HODGE", direct),
        call("METRIC_CURVATURE", direct),
    ))

    assert set(result.values[0]) == {"gradient", "curl", "harmonic"}
    assert isinstance(result.values[1], MetricCurvature)
    # TEMPORAL_DELTA reconstructs the preceding and current states exactly once;
    # the second whole-phrase use consumes that same carrier rather than replaying it.
    assert observed_steps == [0, 1]


def test_executor_accumulates_aligned_temporal_c1_fields_without_a_path_merge():
    timeline = _timeline()
    direct = call("RELATION_SIGNAL", call("TEMPORAL_DELTA", 1))
    result = Executor(sources={"timeline": timeline}).execute(query(
        source("timeline"), call("ACCUMULATE", direct, direct),
    ))

    assert result.values[0].grade == 1
    assert result.values[0].values.tolist() == [2.0, 0.0, 8.0]
    assert result.exactness == (Exactness.APPROXIMATE,)
