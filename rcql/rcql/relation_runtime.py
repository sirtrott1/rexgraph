"""Invocation budgets and source checks shared by recursive and named operations."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_RUNTIME = ContextVar("rcql_relation_runtime", default=None)


class RecursionLimitError(RuntimeError):
    """The requested calculation did not complete within its declared budget."""


class RecursiveCycleError(ValueError):
    """An active invocation recurred without a declared change of state."""


@dataclass(frozen=True)
class RecursionLimits:
    calls: int = 10000
    depth: int = 2048
    evaluations: int = 100000

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.calls, self.depth, self.evaluations)):
            raise ValueError("recursion limits must be positive integers")

    @classmethod
    def from_value(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict) and set(value) <= {"calls", "depth", "evaluations"}:
            return cls(**value)
        raise TypeError("limits require explicit call, depth and evaluation bounds")


def fingerprint(value):
    from .query_cache import _fingerprint
    from .program_codec import digest
    from .name_relation import NameRelation
    if isinstance(value, NameRelation):
        return ("name", value.coefficient_digest)
    from rexgraph.cochain import Chain, Cochain
    from rexgraph.linear_operator import RexOperator
    from rexgraph.type_accession import TypeAccession, TypeRealization
    from rexgraph.tensor_field import FieldSource
    if isinstance(value, (Chain, Cochain)):
        import numpy as np
        array = np.asarray(value.numpy())
        if array.dtype.hasobject:
            from rexgraph.io.rex_state import _encode_exact
            payload = _encode_exact(array).tobytes()
        elif array.dtype.kind in "biufc":
            payload = np.ascontiguousarray(array).tobytes()
        else:
            raise TypeError("chain coefficients need an explicit numeric encoding")
        return (type(value).__name__, value.grade, array.dtype.str, array.shape, payload,
                None if value.cell_keys is None else tuple(value.cell_keys),
                None if value.source is None else FieldSource(value.source).coefficient_digest)
    if isinstance(value, RexOperator):
        from rexgraph.chain_map import CoordinateComplex
        from rexgraph.native_field import NativeAction
        if value.source is None:
            raise TypeError("named operator parameters require their native owner")
        complex_ = CoordinateComplex.from_rex(value.source)
        bridge = NativeAction(value, complex_, complex_.spaces[value.domain_grade], complex_.spaces[value.codomain_grade])
        return ("native_operator", bridge.coefficient_digest)
    if isinstance(value, (TypeAccession, TypeRealization)):
        return (type(value).__name__, value.coefficient_digest, FieldSource(value.source).coefficient_digest)

    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("recursive record keys must be strings")
        return ("mapping", tuple((k, fingerprint(value[k])) for k in sorted(value)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(fingerprint(v) for v in value))
    try:
        return _fingerprint(value)
    except TypeError:
        from .types import RCType
        from .planning import _plain_type
        if isinstance(value, RCType):
            return ("type", digest(_plain_type(value)))
        raise


class RelationRuntime:
    def __init__(self, binding, limits=None, cancellation=None):
        from .execution_trace import current_evidence
        from rexgraph.tensor_field import FieldSource
        from rexgraph.chain_map import CoordinateComplex, _chain_residual
        self.binding = binding
        self.limits = RecursionLimits.from_value(limits)
        self.cancellation = cancellation
        self.evidence = current_evidence()
        native = binding.value if hasattr(binding.value, "_boundary_ptr") else getattr(binding.value, "rex", None)
        if native is None:
            raise TypeError("relational execution requires a native source complex")
        tower = CoordinateComplex.from_rex(native)
        if _chain_residual(tower):
            raise ValueError("the source violates the chain condition")
        self.source = FieldSource(native)
        self.calls = 0
        self.evaluations = 0
        self.names = set()
        self.check()

    def check(self, *values):
        from .source_context import field_references
        if self.cancellation is not None and self.cancellation.is_set():
            from .scheduling import QueryCancelledError
            raise QueryCancelledError("relational execution was cancelled")
        self.source.check()
        for ref in field_references(values):
            ref.check()
        if self.evidence is not None:
            self.evidence.validate_binding(self.binding)
            self.evidence.validate_values(values)

    def tick(self, *values):
        self.check(*values)
        self.evaluations += 1
        if self.evaluations > self.limits.evaluations:
            raise RecursionLimitError("the operation evaluation budget was exhausted")

    def call(self, depth):
        self.check()
        self.calls += 1
        if self.calls > self.limits.calls or depth > self.limits.depth:
            raise RecursionLimitError("the recursive call or depth budget was exhausted")

    def enter_name(self, relation, arguments):
        from .program_codec import digest
        self.tick(arguments)
        key = digest((relation.coefficient_digest, fingerprint(arguments)))
        if key in self.names:
            raise RecursiveCycleError("an active named operation repeats the same arguments")
        if len(self.names) >= min(self.limits.depth, 64):
            raise RecursionLimitError("nested name application exceeded its declared depth")
        self.names.add(key)
        return key

    def leave_name(self, key):
        self.names.remove(key)


@contextmanager
def runtime_scope(binding, limits=None, cancellation=None):
    previous = _RUNTIME.get()
    if previous is not None:
        if previous.binding.value is not binding.value or previous.binding.ref != binding.ref:
            raise ValueError("nested operation changed its pinned source")
        yield previous
        return
    runtime = RelationRuntime(binding, limits, cancellation)
    token = _RUNTIME.set(runtime)
    try:
        yield runtime
    finally:
        _RUNTIME.reset(token)
