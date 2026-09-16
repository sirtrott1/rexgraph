"""Non executing bounds and axis contracts for C1 temporal readings."""
from __future__ import annotations

from math import isfinite

from .types import Domain, Exactness, RCType, ValueKind


ARGUMENTS = {
    "ALIGN_BY_LINEAGE": (("values", "start", "stop"), (0, None)),
    "DELTA": (("step",), ()),
    "EXISTENCE_DELTA": (("value",), ()), "ORIENTATION_DELTA": (("value",), ()),
    "SIGNING_DELTA": (("value",), ()), "HEAD_DELTA": (("value",), ()),
    "STRUCTURAL_DELTA": (("value",), ()), "METRIC_DELTA": (("value",), ()),
    "EXISTENCE_HISTORY": (("key", "start", "stop"), (None, 0, None)),
    "BIOES": (("key", "start", "stop"), (None, 0, None)),
    "BETWEEN": (("start", "end", "axis"), ("rex_time",)),
    "TEMPORAL": ((), ()),
}


def transition_index(step, count):
    if isinstance(step, bool) or not isinstance(step, int) or not 0 < step < count:
        raise ValueError("DELTA requires an interior transition index")


def history_key(key):
    from rexgraph.lineage_alignment import history_key as core_key
    return core_key(key)


def step_interval(start, stop, count):
    from rexgraph.lineage_alignment import step_interval as core_interval
    return core_interval(start, stop, count)


def time_interval(start, end, axis, count):
    if axis not in {"rex_time", "step"}:
        raise ValueError("BETWEEN axis must be rex_time or step; valid and transaction clocks are separate")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in (start, end)):
        raise TypeError("BETWEEN endpoints must be finite real clock values")
    if any(not isfinite(x) for x in (start, end)) or start > end:
        raise ValueError("BETWEEN requires finite ordered endpoints")
    if axis == "step" and (not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end < count):
        raise ValueError("BETWEEN step endpoints must name present snapshots")


def _record(name):
    return RCType(name, kind=ValueKind.RECORD, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def refine(typed, context):
    from .binding import classify
    from rexgraph.graph import RexGraph, TemporalRex
    name, args, value = typed.operator, typed.args, context.binding.value
    if name == "TEMPORAL":
        if not isinstance(value, RexGraph):
            raise TypeError("TEMPORAL requires a native RexGraph")
        return typed.result
    if not isinstance(value, TemporalRex) or classify(value) is not ValueKind.TEMPORAL_REX:
        raise TypeError("temporal reading requires a native TemporalRex")
    if name == "ALIGN_BY_LINEAGE":
        from rexgraph.lineage_alignment import alignment_values
        alignment_values(value, args[0], args[1] if len(args) > 1 else 0, args[2] if len(args) > 2 else None)
        return typed.result
    if name == "DELTA":
        transition_index(args[0], value.T)
    elif name in {"EXISTENCE_HISTORY", "BIOES"}:
        history_key(args[0] if args else None)
        step_interval(args[1] if len(args) > 1 else 0, args[2] if len(args) > 2 else None, value.T)
    elif name == "BETWEEN":
        time_interval(*args[:2], args[2] if len(args) > 2 else "rex_time", value.T)
    else:
        signal = args[0]
        if signal.source != typed.binding.ref:
            raise ValueError("temporal delta must belong to its bound source")
        if signal.temporal is None or signal.temporal.version is None:
            raise ValueError("temporal delta must name its transition index")
        transition_index(signal.temporal.version, value.T)
    return typed.result


def install(register):
    from .signatures import OperatorSignature, TypePattern, _TEMPORAL_DELTA, _temporal_delta_result
    integer = TypePattern("step", literal=int)
    history = (TypePattern("key", literal=(type(None), int, tuple, list), optional=True),
               TypePattern("start", literal=int, optional=True),
               TypePattern("stop", literal=(int, type(None)), optional=True))
    snapshot = RCType("TemporalRex", kind=ValueKind.TEMPORAL_REX, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)
    specs = {"DELTA": ((integer,), _temporal_delta_result),
             "ALIGN_BY_LINEAGE": ((TypePattern("values", literal=(list, tuple)),
                                   TypePattern("start", literal=int, optional=True),
                                   TypePattern("stop", literal=(int, type(None)), optional=True)), _record("LineageAlignment")),
             "EXISTENCE_HISTORY": (history, _record("ExistenceHistory")),
             "BIOES": (history, _record("LifetimeLabels")),
             "BETWEEN": ((TypePattern("start", literal=(int, float)), TypePattern("end", literal=(int, float)),
                          TypePattern("axis", literal=str, optional=True)), snapshot)}
    for name in ("EXISTENCE_DELTA", "ORIENTATION_DELTA", "SIGNING_DELTA", "HEAD_DELTA", "STRUCTURAL_DELTA", "METRIC_DELTA"):
        specs[name] = ((_TEMPORAL_DELTA,), _record("C1DeltaReading"))
    for name, (inputs, result) in specs.items():
        register(OperatorSignature(name=name, source_kind=ValueKind.TEMPORAL_REX, inputs=inputs, result=result,
                                   implementation_key={"ALIGN_BY_LINEAGE": "rexgraph.lineage_alignment.align_by_lineage",
                                                       "EXISTENCE_HISTORY": "rexgraph.lineage_alignment.existence_history",
                                                       "BIOES": "rexgraph.lineage_alignment.lifetime_labels"}.get(name, f"rcql.temporal.{name.lower()}"), memoizable=True,
                                   preconditions=(("canonical scalar C1 vectors for the selected snapshot interval",
                                                   "snapshot reconstruction, axes and identity ambiguity checked at execution",
                                                   "presence separates missing cells from measured zeros; no coefficient transport")
                                                  if name == "ALIGN_BY_LINEAGE" else
                                                  ("explicit C1 identities; no higher grade temporal correspondence is inferred",))))
    register(OperatorSignature(name="TEMPORAL", source_kind=ValueKind.REX, inputs=(), result=snapshot,
                               implementation_key="rcql.temporal.capture", memoizable=True))
