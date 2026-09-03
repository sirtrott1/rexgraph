"""Whole-expression static plans for RCQL.

The parser currently supplies a compact call tree, but a relational-tensor phrase has
meaning only after all of its modifiers and nested readings are considered together.
This module composes the existing operator signatures across that tree before an adapter
executes.  It deliberately does not choose a new query spelling: it is the semantic seam
on which a later phrase grammar, modifier syntax, and optimizer can depend.

Every planned call retains the source, grade, basis, coefficient domain, exactness, and
temporal state established by inference.  A literal remains a literal; a supplied
``RCType`` parameter remains an explicitly declared carrier rather than being guessed
from array length or a Python class.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from .ast import Call, Expr, Literal, Parameter, Query
from .binding import Binding
from .inference import TypedCall, infer
from .types import BasisRef, Domain, Exactness, RCType, SourceRef, ValueKind, Variance

__all__ = ["PlannedExpression", "QueryPlan", "plan_query"]


def _plain_type(value: RCType) -> dict[str, object]:
    """Render one declared carrier without exposing a live source value."""
    return {
        "kind": value.kind.value,
        "grade": value.grade,
        "variance": None if value.variance is None else value.variance.value,
        "domain": None if value.domain is None else value.domain.value,
        "exactness": None if value.exactness is None else value.exactness.value,
        "source": None if value.source is None else value.source.name,
        "temporal": None if value.temporal is None else {
            "version": value.temporal.version,
            "as_of": value.temporal.as_of,
            "valid_at": value.temporal.valid_at,
        },
        "basis": None if value.basis is None else {
            "source_id": value.basis.source_id,
            "grade": value.basis.grade,
            "ordering": value.basis.ordering,
        },
    }


def _plain_literal(value: object) -> object:
    """Keep a plan renderable without leaking a live Python object."""
    if isinstance(value, RCType):
        return {"type": _plain_type(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)) and all(
        item is None or isinstance(item, (bool, int, float, str)) for item in value
    ):
        return list(value)
    return {"python_type": type(value).__name__}


def _coefficient_contract(value: object) -> tuple[Domain, Exactness]:
    """Classify an already-declared carrier without running a tensor action.

    A typed Chain/Cochain is a legitimate RCQL literal.  Its coefficient array
    still matters to the phrase contract: an integer metric can retain rational
    curvature, while a measured float must not acquire an exact label merely
    because it happened to be integral in one observation.  This only reads
    the array protocol; it never calls an adapter or reconstructs a source.
    """
    dtype = getattr(value, "dtype", None)
    kind = getattr(dtype, "kind", None)
    if kind in {"i", "u"}:
        return Domain.INTEGER, Exactness.INTEGER
    if kind == "O":
        # A native object carrier has no coefficient-domain tag. Certifying one
        # explicit literal is necessary to distinguish a real Fraction tensor
        # from arbitrary Python objects before an exact functional is admitted.
        # This is O(nnz) only for externally supplied object literals; nested
        # RCQL structural operations already declare their exact contract and
        # never take this path.
        flat = getattr(value, "flat", None)
        seen, fractions, integers = False, True, True
        if flat is not None:
            for item in flat:
                seen = True
                fractions &= isinstance(item, Fraction)
                integers &= isinstance(item, int) and not isinstance(item, bool)
        if seen and fractions:
            return Domain.RATIONAL, Exactness.RATIONAL
        if seen and integers:
            return Domain.INTEGER, Exactness.INTEGER
        return Domain.SYMBOLIC, Exactness.STRUCTURAL
    if kind == "c":
        return Domain.COMPLEX, Exactness.APPROXIMATE
    return Domain.REAL, Exactness.APPROXIMATE


def _carrier_literal(binding: Binding, value: object) -> RCType | None:
    """Turn a core typed carrier literal into its non-executing RCQL contract.

    RCQL accepts native typed values as expression literals for programmatic
    callers.  Planning must preserve their grade and variance rather than
    treating them as opaque Python objects, or ordinary execution would bypass
    the same source/basis checks that protect nested phrase results.
    """
    from rexgraph.cochain import Chain, Cochain, Field

    field = isinstance(value, Field)
    carrier = value.cochain if field else value
    if not isinstance(carrier, (Chain, Cochain)):
        return None

    # A foreign or unbound carrier must be rejected by a source-bound
    # signature, not silently adopted because its shape happens to match.
    source = binding.ref if carrier.source is binding.value else SourceRef("foreign")
    domain, exactness = _coefficient_contract(carrier.values)
    if field:
        kind, variance, name = ValueKind.FIELD, Variance.COCHAIN, "Field"
    elif isinstance(carrier, Chain):
        kind, variance, name = ValueKind.CHAIN, Variance.CHAIN, "Chain"
    else:
        kind, variance, name = ValueKind.COCHAIN, Variance.COCHAIN, "Cochain"
    return RCType(
        name, grade=carrier.grade, kind=kind, variance=variance,
        domain=domain, exactness=exactness,
        source=source, basis=BasisRef(source.name, carrier.grade),
    )


@dataclass(frozen=True)
class PlannedExpression:
    """One static phrase fragment and the carrier it establishes."""

    expr: Expr
    result: object
    call: TypedCall | None = None
    children: tuple[PlannedExpression, ...] = ()

    def explain(self) -> object:
        """Return a plain, recursive structural account of this phrase fragment."""
        if self.call is None:
            return {"literal": _plain_literal(self.result)}
        detail = self.call.explain()
        detail["arguments"] = [child.explain() for child in self.children]
        return detail


@dataclass(frozen=True)
class QueryPlan:
    """A fully typed, non-executing plan for every returned phrase."""

    binding: Binding
    returns: tuple[PlannedExpression, ...]
    query: Query

    @property
    def effects(self) -> frozenset:
        """The union of effects declared by all planned expression calls."""
        effects = frozenset()
        pending = list(self.returns)
        while pending:
            expression = pending.pop()
            if expression.call is not None:
                effects |= expression.call.effects
            pending.extend(expression.children)
        return effects

    def explain(self) -> dict[str, object]:
        """Describe the full phrase without evaluating an operator or serializing a source."""
        return {
            "source": self.binding.ref.name,
            "source_kind": self.binding.schema.kind.value,
            "policy_digest": self.binding.ref.policy_digest,
            "effects": sorted(effect.value for effect in self.effects),
            "returns": [item.explain() for item in self.returns],
        }


def _plan_expression(binding: Binding, expr: Expr, parameters: Mapping[str, object]) -> PlannedExpression:
    if isinstance(expr, Literal):
        return PlannedExpression(expr, _carrier_literal(binding, expr.value) or expr.value)
    if isinstance(expr, Parameter):
        try:
            value = parameters[expr.name]
        except KeyError as exc:
            raise KeyError(f"no static value declared for parameter ${expr.name}") from exc
        return PlannedExpression(expr, value)
    if isinstance(expr, Call):
        children = tuple(_plan_expression(binding, item, parameters) for item in expr.args)
        typed = infer(binding, expr.name, tuple(item.result for item in children))
        return PlannedExpression(expr, typed.result, typed, children)
    raise TypeError(f"cannot statically plan {type(expr).__name__}")


def plan_query(binding: Binding, query: Query, *, parameters: Mapping[str, object] | None = None) -> QueryPlan:
    """Type every returned expression of a query before any runtime operator resolves.

    ``binding`` is supplied explicitly so the caller cannot accidentally plan one source
    and run another.  The source expression in ``query`` remains parser-level syntax;
    this function checks the returned phrase against the already-bound source.
    """
    if not isinstance(query, Query):
        raise TypeError("plan_query expects a Query")
    supplied = {} if parameters is None else parameters
    return QueryPlan(
        binding=binding,
        returns=tuple(_plan_expression(binding, expression, supplied) for expression in query.returns),
        query=query,
    )
