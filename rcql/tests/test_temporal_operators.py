"""Combined temporal identities, sparse lifetime labels and exact stored weights."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.temporal_signal import temporal_signal

from rcql import Executor, call, query, source
from rcql.temporal_operators import ADAPTERS


def run(timeline, name, *args, explain=False):
    return Executor(sources={"t": timeline}).execute(replace(query(source("t"), call(name, *args)), explain=explain)).values[0]


def graph(supports, ids, weights=None, signs=None):
    ptr = [0]
    for support in supports:
        ptr.append(ptr[-1] + len(support))
    return RexGraph.from_hypergraph(np.array(ptr), np.array([v for s in supports for v in s]),
                                   relation_ids=np.array(ids), w_E=None if weights is None else np.array(weights, dtype=object),
                                   signs=None if signs is None else np.array(signs))


@pytest.fixture
def timeline():
    value = TemporalRex([])
    value.append_snapshot(graph([(0, 1, 2), (3, 3), (4,), (5, 6)], [11, 12, 13, 14],
                                [Q(1, 3), 1, 1, 2], [1, 1, 1, 1]), at=10)
    value.append_snapshot(graph([(1, 0, 2), (3, 3), (7, 8)], [11, 12, 15],
                                [Q(2, 3), 1, 3], [1, -1, 1]), at=20)
    return value


def entries_by_key(record):
    return {record["keys"][i]: value for i, value in record["entries"]}


def test_scalar_delta_channels_keep_deaths_and_independent_changes(timeline):
    value = call("DELTA", 1)
    existence = run(timeline, "EXISTENCE_DELTA", value)
    assert existence["keys"] == (11, 12, 13, 14, 15)
    assert entries_by_key(existence) == {13: -1, 14: -1, 15: 1}
    assert entries_by_key(run(timeline, "SIGNING_DELTA", value)) == {12: 1}
    assert entries_by_key(run(timeline, "METRIC_DELTA", value)) == {11: Q(1, 3), 13: Q(-1), 14: Q(-2), 15: Q(3)}
    assert existence["shape"] == (5,) and existence["implicit_zero"] == 0


def test_full_boundary_delta_does_not_sum_away_relation_identity(timeline):
    delta = run(timeline, "DELTA", 1)
    result = run(timeline, "STRUCTURAL_DELTA", delta)
    columns = {key: {} for key in result["keys"]}
    for i, j, coefficient in result["entries"]:
        columns[result["keys"][j]][i] = coefficient
    assert columns[11] == {0: Q(3, 2), 1: Q(-3, 2)}
    assert columns[12] == {}  # A gauge sign does not change B1.
    assert columns[13] == {4: Q(-1)}  # A deleted witness is retained.
    assert result["shape"] == (9, 5)
    source = delta.source_field("structural").values
    assert all(sum(column.get(i, 0) for column in columns.values()) == source[i] for i in range(9))
    heads = run(timeline, "HEAD_DELTA", delta)
    assert {(i, heads["keys"][j]): v for i, j, v in heads["entries"]} == {
        (0, 11): -1, (1, 11): 1, (5, 14): -1, (7, 15): 1}
    orientation = run(timeline, "ORIENTATION_DELTA", delta)
    assert orientation["entries"] == heads["entries"] and orientation["shape"] == heads["shape"]


def test_metric_delta_does_not_depend_on_numerical_event_detection():
    timeline = TemporalRex([])
    for weight in (2**100, 2**100 + 1):
        timeline.append_snapshot(graph([(0, 1)], [7], [weight]))
    value = run(timeline, "METRIC_DELTA", call("DELTA", 1))
    assert entries_by_key(value) == {7: Q(1)}


def test_literal_and_nested_deltas_have_the_same_contract(timeline):
    literal = temporal_signal(timeline, 1)
    for name in ("EXISTENCE_DELTA", "ORIENTATION_DELTA", "SIGNING_DELTA", "HEAD_DELTA", "STRUCTURAL_DELTA", "METRIC_DELTA"):
        assert run(timeline, name, literal) == run(timeline, name, call("DELTA", 1))
        assert run(timeline, name, literal, explain=True)["returns"][0]["result"]["exactness"] == "structural"


@pytest.mark.parametrize("explain", [False, True])
def test_temporal_readings_reject_invalid_steps_axes_and_foreign_sources(timeline, explain):
    for step in (0, 2, -1, True):
        with pytest.raises((ValueError, TypeError)):
            run(timeline, "DELTA", step, explain=explain)
    other = TemporalRex([])
    other.append_snapshot(graph([(0, 1)], [7]))
    other.append_snapshot(graph([(1, 0)], [7]))
    with pytest.raises((ValueError, TypeError)):
        run(timeline, "EXISTENCE_DELTA", temporal_signal(other, 1), explain=explain)
    with pytest.raises((ValueError, TypeError)):
        run(timeline, "BETWEEN", 10, 20, "transaction", explain=explain)
    for key, start, stop in ((True, 0, 2), ((), 0, 2), (None, -1, 2), (None, 1, 0), (None, 0, 3)):
        with pytest.raises((ValueError, TypeError)):
            run(timeline, "BIOES", key, start, stop, explain=explain)


def test_sparse_history_and_labels_preserve_gaps_and_window_boundaries():
    timeline = TemporalRex([])
    for ids in ([7], [7], [7], [8], [7], [7]):
        timeline.append_snapshot(graph([(0, 1)]*len(ids), ids))
    history = run(timeline, "EXISTENCE_HISTORY", 7)
    assert history["shape"] == (6, 1)
    assert history["entries"] == ((0, 0, 1), (1, 0, 1), (2, 0, 1), (4, 0, 1), (5, 0, 1))
    labels = run(timeline, "BIOES", 7)
    assert labels["entries"] == ((0, 0, "B"), (1, 0, "I"), (2, 0, "E"), (4, 0, "B"), (5, 0, "E"))
    clipped = run(timeline, "BIOES", 7, 1, 5)
    assert clipped["entries"] == ((0, 0, "I"), (1, 0, "E"), (3, 0, "B"))
    assert clipped["implicit_zero"] == "O"
    assert run(timeline, "BIOES", 8)["entries"] == ((3, 0, "S"),)
    assert run(timeline, "EXISTENCE_HISTORY", None, 2, 2)["shape"] == (0, 0)
    assert run(timeline, "EXISTENCE_HISTORY", 999)["entries"] == ()


def test_temporal_capture_and_slice_preserve_attributes_and_clocks(timeline):
    sliced = run(timeline, "BETWEEN", 20, 20)
    assert sliced.T == 1 and sliced.time_at(0) == 20
    assert sliced.reconstruct_at(0).edge_metric_exact == [Q(2, 3), Q(1), Q(3)]
    captured = run(timeline.reconstruct_at(0), "TEMPORAL")
    assert captured.T == 1
    assert captured.reconstruct_at(0).relation_ids.tolist() == [11, 12, 13, 14]
    assert run(timeline, "BETWEEN", 30, 40).T == 0


def test_all_temporal_names_have_matching_argument_contracts():
    from rcql import catalogued
    from rcql.arguments import EXPRESSION_ARGUMENTS
    from rcql.temporal_contracts import ARGUMENTS
    assert set(ADAPTERS) == set(ARGUMENTS)
    assert set(ADAPTERS) <= catalogued()
    assert all(EXPRESSION_ARGUMENTS[name] == value for name, value in ARGUMENTS.items())


def test_branching_orientation_delta_separates_two_nonminimal_heads():
    timeline = TemporalRex([])
    timeline.append_snapshot(graph([(1, 0, 2)], [7]))
    timeline.append_snapshot(graph([(2, 0, 1)], [7]))
    value = run(timeline, "ORIENTATION_DELTA", call("DELTA", 1))
    assert value["entries"] == ((1, 0, -1), (2, 0, 1))
    assert value["shape"] == (3, 1)


@pytest.mark.parametrize("backend", ["memory", "rex"])
def test_temporal_readings_roundtrip_through_rcdb(timeline, tmp_path, backend):
    rcdb = pytest.importorskip("rcdb")
    from rcql import parse
    store = rcdb.MemoryStore() if backend == "memory" else rcdb.open_store(f"rex://{tmp_path / 'store'}")
    try:
        store.put("history", timeline, analytics=False)
        if backend == "rex":
            store.close()
            store = rcdb.open_store(f"rex://{tmp_path / 'store'}")
        result = Executor(sources={"db": store}).execute(parse(
            'FROM RCDB_GET($db,"history") LET d=DELTA(1) '
            'RETURN METRIC_DELTA(d), STRUCTURAL_DELTA(d), SIGNING_DELTA(d), '
            'ORIENTATION_DELTA(d), EXISTENCE_HISTORY(), BIOES(), BETWEEN(10,20)'))
        assert entries_by_key(result.values[0]) == {11: Q(1, 3), 13: Q(-1), 14: Q(-2), 15: Q(3)}
        assert result.values[1]["shape"] == (9, 5)
        assert entries_by_key(result.values[2]) == {12: 1}
        assert result.values[3]["shape"] == (9, 5)
        assert result.values[4]["shape"] == result.values[5]["shape"] == (2, 5)
        assert result.values[6].times.tolist() == [10, 20]
    finally:
        store.close()
