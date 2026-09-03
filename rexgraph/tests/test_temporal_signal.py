"""C1 temporal source fields are exact, typed, and do not become graph walks."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.temporal_signal import TemporalSignal, relation_key, signal_flow, temporal_signal


def _timeline() -> TemporalRex:
    """One reversal/signing change plus one new relation with a measured amplitude."""
    out = TemporalRex([])
    out.append_snapshot(RexGraph(
        sources=np.asarray([0, 1], dtype=np.int32),
        targets=np.asarray([1, 2], dtype=np.int32),
        w_E=np.asarray([2.0, 1.0]),
        signs=np.asarray([1, 1], dtype=np.int32),
    ), at=10.0)
    out.append_snapshot(RexGraph(
        sources=np.asarray([1, 1, 2], dtype=np.int32),
        targets=np.asarray([0, 2, 3], dtype=np.int32),
        w_E=np.asarray([3.0, 1.0, 4.0]),
        signs=np.asarray([-1, 1, 1], dtype=np.int32),
    ), at=15.0)
    return out


def test_temporal_signal_preserves_each_channel_and_has_exact_support_lookup():
    signal = temporal_signal(_timeline(), 1)

    assert isinstance(signal, TemporalSignal)
    assert signal.when == 15.0
    reversed_relation = signal.event((0, 1))
    born_relation = signal.event([2, 3])
    assert reversed_relation.key == relation_key(signal.current, 0) == (0, 1)
    assert (reversed_relation.existence, reversed_relation.orientation, reversed_relation.signing) == (0, -1, -1)
    assert (reversed_relation.previous_head, reversed_relation.head) == (0, 1)
    assert born_relation.existence == 1
    assert born_relation.head == 2 and born_relation.previous_head is None
    assert born_relation.amplitude_delta == 4.0


def test_structural_and_amplitude_sources_are_not_conflated():
    signal = temporal_signal(_timeline(), 1)

    structural = signal.source_field("structural")
    # Reversal: [1,-1] - [-1,1] = [2,-2].  Birth: [0,0,-1,1].
    assert structural.values.tolist() == [Fraction(2), Fraction(-2), Fraction(-1), Fraction(1)]
    assert signal.is_exact("structural") is True

    amplitude = signal.source_field("amplitude")
    # The same boundary change carries 2 -> 3 amplitude, and the born C1 cell
    # carries its measured 4.  This is numerical attribution, not a claim that
    # the measured field became exact topology.
    np.testing.assert_allclose(amplitude.values, [5.0, -5.0, -4.0, 4.0])
    assert signal.is_exact("amplitude") is False

    # Signing remains an independently queryable event but cannot be injected
    # into B1 as though a gauge change altered relation boundary geometry.
    assert signal.event((0, 1)).signing == -1
    assert signal.source_field("signing").values.tolist() == [Fraction(0)] * 4


def test_local_dirac_square_response_uses_current_relations_not_paths():
    flow = signal_flow(temporal_signal(_timeline(), 1))

    assert flow.exact is True
    assert flow.source_field.values.tolist() == [Fraction(2), Fraction(-2), Fraction(-1), Fraction(1)]
    # B1* source on the current C1 basis: (1->0, 1->2, 2->3).
    assert flow.relation_response.values.tolist() == [Fraction(4), Fraction(1), Fraction(2)]
    assert flow.returned_boundary.values.tolist() == [Fraction(4), Fraction(-5), Fraction(-1), Fraction(2)]


def test_direct_c1_amplitude_delta_is_not_preprojected_to_a_gradient_source():
    signal = temporal_signal(_timeline(), 1)
    field = signal.relation_field("amplitude")

    assert field.source is signal.current
    assert field.cell_keys == ((0, 1), (1, 2), (2, 3))
    assert field.values.tolist() == [1.0, 0.0, 4.0]


def test_head_identity_catches_a_branching_move_the_coarse_orientation_channel_cannot():
    ptr = np.asarray([0, 4], dtype=np.int64)
    timeline = TemporalRex([])
    timeline.append_snapshot(RexGraph.from_hypergraph(ptr, np.asarray([5, 1, 2, 3], dtype=np.int64)))
    timeline.append_snapshot(RexGraph.from_hypergraph(ptr, np.asarray([3, 1, 2, 5], dtype=np.int64)))

    signal = temporal_signal(timeline, 1)
    event = signal.event((1, 2, 3, 5))
    assert event.orientation == 0
    assert (event.previous_head, event.head, event.head_changed) == (5, 3, True)
    assert signal.source_field("geometry").values.tolist() == [
        Fraction(0), Fraction(0), Fraction(0), Fraction(-4, 3), Fraction(0), Fraction(4, 3)
    ]


def test_parallel_supports_refuse_until_a_stable_relation_identity_exists():
    ptr = np.asarray([0, 2, 4], dtype=np.int64)
    idx = np.asarray([0, 1, 0, 1], dtype=np.int64)
    timeline = TemporalRex([])
    timeline.append_snapshot(RexGraph.from_hypergraph(ptr, idx))
    timeline.append_snapshot(RexGraph.from_hypergraph(ptr, idx))

    with pytest.raises(ValueError, match="stable relation identity"):
        temporal_signal(timeline, 1)


def test_temporal_signal_refuses_an_ambiguous_repeated_primary_incidence():
    ptr = np.asarray([0, 3], dtype=np.int64)
    timeline = TemporalRex([])
    with pytest.raises(ValueError, match="only an exact \\[v, v\\]"):
        timeline.append_snapshot(
            RexGraph.from_hypergraph(ptr, np.asarray([0, 0, 1], dtype=np.int64))
        )


def test_weight_only_change_is_a_signal_without_an_invented_structural_event():
    timeline = TemporalRex([])
    for weight in (2.0, 5.0):
        timeline.append_snapshot(RexGraph(
            sources=np.asarray([0], dtype=np.int32),
            targets=np.asarray([1], dtype=np.int32),
            w_E=np.asarray([weight]),
        ))
    signal = temporal_signal(timeline, 1)
    event = signal.event((0, 1))
    assert (event.existence, event.orientation, event.signing, event.amplitude_delta) == (0, 0, 0, 3.0)
    assert signal.source_field("structural").values.tolist() == [Fraction(0), Fraction(0)]
    np.testing.assert_allclose(signal.source_field("amplitude").values, [-3.0, 3.0])
