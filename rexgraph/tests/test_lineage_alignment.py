"""Native relation identity alignment without dense missing value fills."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.lineage_alignment import align_by_lineage, alignment_values


def timeline():
    value = TemporalRex([])
    for supports, ids in [([[0, 1, 2], [0, 1, 2], [2]], [9, 3, 7]),
                           ([[2, 0, 1], [1, 1], [0, 2, 3, 4]], [3, 8, 9]),
                           ([[1, 1], [2]], [8, 7])]:
        value.append_snapshot(RexGraph.from_cells([5, supports], relation_ids=ids), at=value.T+10)
    return value


def values_for(source, records):
    return [[record[int(k)] for k in source.reconstruct_at(t).relation_ids] for t, record in enumerate(records)]


def test_identity_reorder_birth_death_parallel_head_and_support_change():
    source = timeline()
    records = [{9: Q(1, 3), 3: 0, 7: 2**100+1}, {3: 4, 8: 0, 9: -5}, {8: 2, 7: 0}]
    values = values_for(source, records)
    out = align_by_lineage(source, values)
    assert out["identity"] == "relation_ids" and out["coefficient_domain"] == "Q"
    assert out["shape"] == (3, 4) and out["steps"] == (0, 1, 2) and out["times"] == (10, 11, 12)
    assert set(out["keys"]) == {3, 7, 8, 9}
    lookup = {(t, out["keys"][j]): v for t, j, v in out["entries"]}
    present = {(t, out["keys"][j]) for t, j in out["presence"]}
    assert present == {(t, k) for t, record in enumerate(records) for k in record}
    assert lookup == {(t, k): v for t, record in enumerate(records) for k, v in record.items() if v}
    # The zero at relation 3 is observed; the absence of relation 8 is not.
    assert (0, 3) in present and (0, 3) not in lookup
    assert (0, 8) not in present
    for t, cell_map in enumerate(out["cell_maps"]):
        assert tuple(out["keys"][i] for i in cell_map) == tuple(source.reconstruct_at(t).relation_ids)
    values[0][0] = 999
    assert {(t, out["keys"][j]): v for t, j, v in out["entries"]} == lookup


def test_anonymous_support_is_not_a_boundary_column_or_orientation_transport():
    source = TemporalRex([])
    source.append_snapshot(RexGraph.from_cells([4, [[0, 1, 2], [3], [3, 3]]]))
    source.append_snapshot(RexGraph.from_cells([4, [[2, 0, 1], [3, 3]]]))
    out = align_by_lineage(source, [[1, 2, 3], [4, 5]])
    assert out["identity"] == "support"
    assert set(out["keys"]) == {(0, 1, 2), (3,), (3, 3)}
    assert out["entries"] == ((0, 0, 1), (0, 1, 2), (0, 2, 3), (1, 0, 4), (1, 2, 5))


def test_anonymous_parallel_refusal_uses_core_identity_rule():
    source = TemporalRex([])
    source.append_snapshot(RexGraph.from_cells([2, [[0, 1], [0, 1]]]))
    with pytest.raises(ValueError, match="parallel"):
        align_by_lineage(source, [[1, 2]])


def test_interval_empty_and_absent_readings():
    source = timeline()
    out = align_by_lineage(source, [[0]*source.reconstruct_at(1).nE], start=1, stop=2)
    assert out["steps"] == (1,) and out["times"] == (11,)
    assert out["entries"] == () and len(out["presence"]) == 3
    assert align_by_lineage(source, [], start=2, stop=2)["shape"] == (0, 0)
    empty = TemporalRex([])
    assert align_by_lineage(empty, [])["identity"] == "empty"
    empty.append_snapshot(RexGraph.from_cells([0, []]))
    assert align_by_lineage(empty, [[]])["shape"] == (1, 0)


@pytest.mark.parametrize("number,domain", [(0, "Z"), (2**100+1, "Z"), (Q(1, 7), "Q"), (0.0, "real"), (1.25, "real")])
def test_scalar_domain_preserved(number, domain):
    source = TemporalRex([])
    source.append_snapshot(RexGraph.from_cells([1, [[0]]]))
    result = align_by_lineage(source, [[number]])
    assert result["coefficient_domain"] == domain
    assert result["entries"] == (((0, 0, number),) if number else ())
    assert result["presence"] == ((0, 0),)


@pytest.mark.parametrize("number", [True, np.bool_(False), "3", None, 1j, float("nan"), float("inf")])
def test_bad_scalar(number):
    source = TemporalRex([])
    source.append_snapshot(RexGraph.from_cells([1, [[0]]]))
    with pytest.raises((TypeError, ValueError)):
        align_by_lineage(source, [[number]])


@pytest.mark.parametrize("values,start,stop", [([], 0, None), ([[], [], []], True, None),
    ([[], [], []], 0.0, None), ([], -1, 0), ([], 2, 1), ([], 3, 4),
    (None, 0, None), ([[], [], []], 0, None), ([[[1]]], 0, 1)])
def test_bad_axes_or_interval(values, start, stop):
    with pytest.raises((TypeError, ValueError)):
        align_by_lineage(timeline(), values, start=start, stop=stop)


def test_preflight_does_not_reconstruct(monkeypatch):
    source = timeline()
    monkeypatch.setattr(TemporalRex, "reconstruct_at", lambda *a: pytest.fail("reconstructed"))
    assert alignment_values(source, [[1, 2, 3], [3, 4, 5], [6, 7]])[1:] == (0, 3)


def test_disjoint_sparse_union_has_no_rectangular_allocation():
    source = TemporalRex([])
    for i in range(30):
        source.append_snapshot(RexGraph.from_cells([i+1, [[i]]], relation_ids=[i]))
    # Only one live C1 value in each frame, despite the 30 by 30 conceptual shape.
    out = align_by_lineage(source, [[1] for _ in range(source.T)])
    assert out["shape"] == (30, 30)
    assert len(out["presence"]) == len(out["entries"]) == 30
    assert out["cell_maps"] == tuple((i,) for i in range(30))


@pytest.mark.parametrize("start,stop", [(0, 6), (1, 5), (2, 2), (0, 1), (3, 4), (5, 6)])
@pytest.mark.parametrize("key", [None, 7, 8, 999])
def test_history_and_labels_match_independent_lifetimes(start, stop, key):
    from rexgraph.lineage_alignment import existence_history, lifetime_labels
    source = TemporalRex([])
    frames = [{7}, {7}, {7}, {8}, {7}, {7}]
    for ids in frames:
        source.append_snapshot(RexGraph.from_cells([2, [[0, 1]]], relation_ids=list(ids)))
    out = existence_history(source, key, start, stop)
    labels = lifetime_labels(source, key, start, stop)
    expected = {(t-start, k) for t in range(start, stop) for k in frames[t] if key is None or k == key}
    assert {(row, out["keys"][column]) for row, column, value in out["entries"] if value == 1} == expected
    for row, column, label in labels["entries"]:
        t, k = row+start, labels["keys"][column]
        lo, hi = t, t
        while lo > 0 and k in frames[lo-1]:
            lo -= 1
        while hi+1 < len(frames) and k in frames[hi+1]:
            hi += 1
        assert label == ("S" if lo == hi else "B" if t == lo else "E" if t == hi else "I")
    aligned = align_by_lineage(source, [[1]*len(frames[t]) for t in range(start, stop)], start=start, stop=stop)
    if key is None:
        assert {(r, aligned["keys"][c]) for r, c in aligned["presence"]} == expected


def test_history_reconstructs_only_required_frames(monkeypatch):
    from rexgraph.lineage_alignment import existence_history, lifetime_labels
    source = timeline()
    original, calls = TemporalRex.reconstruct_at, []
    def observed(self, step):
        calls.append(step)
        return original(self, step)
    monkeypatch.setattr(TemporalRex, "reconstruct_at", observed)
    existence_history(source, start=1, stop=2)
    assert calls == [1]
    calls.clear()
    lifetime_labels(source, start=1, stop=2)
    assert calls == [0, 1, 2]
    calls.clear()
    assert lifetime_labels(source, start=1, stop=1)["entries"] == ()
    assert calls == []


def test_identity_reads_do_not_construct_signal_boundary_columns(monkeypatch):
    from importlib import import_module
    from rexgraph.lineage_alignment import existence_history, lifetime_labels
    signals = import_module("rexgraph.temporal_signal")
    source = timeline()
    monkeypatch.setattr(signals, "_column", lambda *a: pytest.fail("signal boundary column built"))
    assert len(align_by_lineage(source, [[0, 0, 0], [0, 0, 0], [0, 0]])["presence"]) == 8
    assert len(existence_history(source)["entries"]) == 8
    assert len(lifetime_labels(source)["entries"]) == 8
