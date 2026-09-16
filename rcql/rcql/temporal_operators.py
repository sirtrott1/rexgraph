"""Sparse C1 temporal readings with explicit union identities and clocks."""
from __future__ import annotations

from fractions import Fraction

from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.temporal_signal import _snapshot_cells, relation_identity, temporal_signal

from .execution_trace import record_method


def _timeline(value):
    if not isinstance(value, TemporalRex):
        raise TypeError("temporal reading requires a native TemporalRex")
    return value


def delta(source, step):
    from .temporal_contracts import transition_index
    _timeline(source)
    transition_index(step, source.T)
    return temporal_signal(source, step)


def _signal(source, value):
    from .operators import _typed_temporal_signal
    _timeline(source)
    return _typed_temporal_signal(source, value, operator="DELTA")


def _column_delta(source, value, channel):
    signal = _signal(source, value)
    before, after = _snapshot_cells(signal.previous), _snapshot_cells(signal.current)
    keys = tuple(sorted(before.keys() | after.keys()))
    entries = []
    scalar = channel in {"existence", "signing", "metric"}
    if channel == "metric":
        weights = []
        for rex in (signal.previous, signal.current):
            exact = rex.edge_metric_exact
            weights.append({relation_identity(rex, i): Fraction(1) if exact is None else exact[i]
                            for i in range(rex.nE)})
    for j, key in enumerate(keys):
        a, b = before.get(key), after.get(key)
        if channel == "existence":
            value = int(b is not None) - int(a is not None)
        elif channel == "signing":
            # Birth/death belong to existence, not to a fictitious prior bit.
            # SIGNING reads 0 for +1 and 1 for -1, independently of the head.
            if any(cell is not None and cell.sign not in {-1, 1} for cell in (a, b)):
                raise ValueError("SIGNING_DELTA requires declared gauge signs in {-1,+1}")
            value = 0 if a is None or b is None else (a.sign - b.sign) // 2
        elif channel == "metric":
            value = weights[1].get(key, Fraction(0)) - weights[0].get(key, Fraction(0))
        else:
            if channel == "structural":
                left = {} if a is None else dict(a.column.entries)
                right = {} if b is None else dict(b.column.entries)
            else:
                # Singleton witnesses and cancelling loops have no distinguished
                # negative participant in the composite boundary convention.
                left = {} if a is None else {i: 1 for i, c in a.column.entries if c < 0}
                right = {} if b is None else {i: 1 for i, c in b.column.entries if c < 0}
            for i in sorted(left.keys() | right.keys()):
                value = right.get(i, 0) - left.get(i, 0)
                if value:
                    entries.append((i, j, value))
            continue
        if value:
            entries.append((j, value))
    record_method("exact-c1-union-delta", channel=channel, relations=len(keys), entries=len(entries))
    return {"channel": channel, "keys": keys, "entries": tuple(entries),
            "shape": (len(keys),) if scalar else (signal.vertex_count, len(keys)),
            "from_step": signal.step - 1, "to_step": signal.step,
            "basis": "union-C1-identities", "implicit_zero": 0,
            "coefficient_domain": "Q" if channel in {"metric", "structural"} else "Z"}


def existence_delta(source, value):
    return _column_delta(source, value, "existence")


def orientation_delta(source, value):
    return _column_delta(source, value, "orientation")


def signing_delta(source, value):
    return _column_delta(source, value, "signing")


def head_delta(source, value):
    return _column_delta(source, value, "head")


def structural_delta(source, value):
    """Full B1 change on a union basis, not its sum into a C0 source field."""
    return _column_delta(source, value, "structural")


def metric_delta(source, value):
    """Exact stored C1 metric change, including changes hidden by float rounding."""
    return _column_delta(source, value, "metric")


def existence_history(source, key=None, start=0, stop=None):
    """Sparse existence over [start,stop), with no dense time by cell array."""
    from rexgraph.lineage_alignment import existence_history as core_history
    history = core_history(source, key, start, stop)
    record_method("sparse-c1-existence-history", entries=len(history["entries"]))
    return history


def bioes(source, key=None, start=0, stop=None):
    """Lifetime labels on present coordinates; O is implicit outside existence."""
    from rexgraph.lineage_alignment import lifetime_labels
    result = lifetime_labels(source, key, start, stop)
    record_method("sparse-c1-lifetime-labels", entries=len(result["entries"]))
    return result


def between(source, start, end, axis="rex_time"):
    """Copy a closed interval on one declared timeline axis into a TemporalRex."""
    from .temporal_contracts import time_interval
    source = _timeline(source)
    time_interval(start, end, axis, source.T)
    result = TemporalRex([], general=source._general, directed=source._directed)
    for step in range(source.T):
        position = step if axis == "step" else source.time_at(step)
        if start <= position <= end:
            result.append_snapshot(source.reconstruct_at(step), at=source.time_at(step))
    record_method("native-temporal-interval", axis=axis, snapshots=result.T)
    return result


def temporal(source):
    """Capture one full Rex state as a one step TemporalRex, preserving attributes."""
    if not isinstance(source, RexGraph):
        raise TypeError("TEMPORAL requires a native RexGraph source")
    result = TemporalRex([])
    result.append_snapshot(source)
    return result


def align_by_lineage(source, values, start=0, stop=None):
    from rexgraph.lineage_alignment import align_by_lineage as core_align
    result = core_align(source, values, start=start, stop=stop)
    record_method("core-sparse-c1-lineage-alignment", identity=result["identity"],
                  present=len(result["presence"]), entries=len(result["entries"]))
    return result


ADAPTERS = {"DELTA": delta, "EXISTENCE_DELTA": existence_delta, "ORIENTATION_DELTA": orientation_delta,
            "SIGNING_DELTA": signing_delta, "HEAD_DELTA": head_delta, "STRUCTURAL_DELTA": structural_delta,
            "METRIC_DELTA": metric_delta, "EXISTENCE_HISTORY": existence_history, "BIOES": bioes,
            "BETWEEN": between, "TEMPORAL": temporal, "ALIGN_BY_LINEAGE": align_by_lineage}


def install(register):
    for name, function in ADAPTERS.items():
        register(name)(function)
