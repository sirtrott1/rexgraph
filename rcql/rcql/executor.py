"""RCQL execution."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Any

from .ast import Call, Expr, Literal, MutationQuery, Parameter, Query
from .operators import get_operator
from .optimizer import Rewrite, optimize
from .types import Exactness


@dataclass(frozen=True)
class Result:
    values: tuple[Any, ...]
    rewrites: tuple[Rewrite, ...] = ()
    plan: tuple[str, ...] = ()
    exactness: tuple[Exactness, ...] = ()


#: What each operator must be permitted to do. The executor asks the source for exactly
#: this before evaluating, so this table IS the access-control surface: anything absent
#: falls to "read", which is why every operator that reaches identity, history, files or
#: the store's own configuration has to be named here.
_PERMISSION = {
    "RCDB_SECURITY": "admin",
    "RCDB_HISTORY": ("history", "identity"),
    "RCDB_COMMITS": ("history", "identity"),
    "RCDB_VERIFY": ("history", "identity"),
    "RCDB_GET": "identity",
    "RCDB_HASH": "identity",
    "RCDB_LIST": "records",
    "RCDB_SEARCH": "records",
    "SEARCH": "search",
    "SEARCH_TENSORS": "search",
    "FILES": "file_read",
    "FILE_INFO": "file_read",
    "FILE_HASH": "file_read",
    "HASH_FILES": "file_read",
    "TENSORS": "file_read",
}

# A direct C1 temporal field is carried on the reconstructed current Rex, not on the
# TemporalRex history object that supplied it.  These operators consume that carrier's
# actual source basis.  This is source routing, not a projection: the C1 field itself is
# passed unchanged and the temporal source remains in its provenance contract.
_CARRIER_SOURCE_OPERATORS = frozenset({
    "ACCUMULATE", "HODGE", "HARMONIC", "HODGE_COORDS", "METRIC_CURVATURE", "QUADRANCE",
    "WINDING",
})

# Caching is deliberately an allow-list.  Catalog hashing and storage operations can
# change a cache, expose a new record state, or consume an external capability, so they
# must run each time they appear.  These native readings have no such side effect and
# may safely share a source-bound intermediate across one whole query phrase.
_MEMOIZABLE_OPERATORS = frozenset({
    "ACCUMULATE", "APPLY", "ARITY", "BETTI", "BOUNDARY", "CELL", "CELLS", "CHARACTER", "CLOSURE",
    "COBOUNDARY", "COMPOSITE", "CORELATIONS", "DESCRIBE", "ENCLOSURE", "EXISTENCE",
    "GRADE", "GREEN", "HARMONIC", "HEAD", "HODGE", "HODGE_COORDS", "HODGE_OPERATOR",
    "INDICATOR", "METRIC_CURVATURE", "NULLITY", "QUADRANCE", "RANK", "RELATION_SIGNAL", "SHARE",
    "SHARE_SUPPORT", "SIGNAL_AT", "SIGNAL_FLOW", "SIGNAL_HODGE", "SIGNAL_SOURCE",
    "SIGNIFICANCE", "SPREAD", "STAR", "TEMPORAL_DELTA", "WINDING", "ZERO",
})


class Executor:
    """Evaluate RCQL against explicit source and parameter bindings."""

    def __init__(self, *, sources=None, params=None):
        self.sources = dict(sources or {})
        self.params = dict(params or {})

    @staticmethod
    def _unwrap(source, permission):
        from .capabilities import BoundSource
        if isinstance(source, BoundSource):
            permissions = (permission,) if isinstance(permission, str) else tuple(permission)
            for item in permissions:
                source.require(item)
            return source.value, source.policy
        return source, None

    @staticmethod
    def _carrier_source(source, args: tuple[object, ...], operator: str):
        """Use one carried field's current Rex for a temporal field operation.

        A whole temporal history is not a Rex snapshot, so handing it to a C1 action
        would lose the field's basis.  Routing is allowed only for the named single-field
        operations and only when every carried source candidate agrees by identity;
        multi-field alignment remains a static-plan responsibility rather than a guess.
        """
        if operator not in _CARRIER_SOURCE_OPERATORS:
            return source
        if not hasattr(source, "reconstruct_at") or not hasattr(source, "T"):
            return source
        candidates = []
        for value in args:
            value = getattr(value, "cochain", value)
            candidate = getattr(value, "source", None)
            if candidate is not None:
                candidates.append(candidate)
        if candidates and all(candidate is candidates[0] for candidate in candidates):
            return candidates[0]
        return source

    def _eval_source(self, expr: Expr):
        if isinstance(expr, Parameter):
            if expr.name not in self.sources:
                raise KeyError(f"unknown source ${expr.name}")
            return self.sources[expr.name]
        if isinstance(expr, Call) and expr.name in {"REX", "CATALOG"} and len(expr.args) == 1:
            name = self._eval(expr.args[0], None)
            if name not in self.sources:
                raise KeyError(f"unknown source {name!r}")
            return self.sources[name]
        if isinstance(expr, Call) and expr.name == "FILE" and len(expr.args) == 2:
            catalog_name = self._eval(expr.args[0], None)
            entry_name = self._eval(expr.args[1], None)
            if catalog_name not in self.sources:
                raise KeyError(f"unknown catalog {catalog_name!r}")
            catalog = self.sources[catalog_name]
            raw, policy = self._unwrap(catalog, "file_read")
            from rexgraph.io.catalog import FileCatalog
            if not isinstance(raw, FileCatalog):
                raise TypeError(f"source {catalog_name!r} is not a file catalog")
            value = raw.load(str(entry_name))
            if policy is not None:
                from .capabilities import BoundSource
                return BoundSource(value, policy)
            return value
        if isinstance(expr, Call) and expr.name == "AT" and len(expr.args) == 2:
            parent = self._eval_source(expr.args[0])
            from .capabilities import BoundSource

            temporal = parent.value if isinstance(parent, BoundSource) else parent
            version = self._eval(expr.args[1], None)
            if isinstance(version, bool) or not isinstance(version, int):
                raise TypeError("AT expects an integer TemporalRex snapshot version")
            if not hasattr(temporal, "reconstruct_at") or not hasattr(temporal, "T"):
                raise TypeError("AT expects a TemporalRex source")
            if version < 0 or version >= int(temporal.T):
                raise ValueError(
                    f"AT version must lie in [0, {int(temporal.T) - 1}], got {version}"
                )
            snapshot = temporal.reconstruct_at(version)
            return BoundSource(snapshot, parent.policy) if isinstance(parent, BoundSource) else snapshot
        if isinstance(expr, Call) and expr.name == "AT_TIME" and len(expr.args) == 2:
            parent = self._eval_source(expr.args[0])
            from .capabilities import BoundSource

            temporal = parent.value if isinstance(parent, BoundSource) else parent
            when = self._eval(expr.args[1], None)
            if isinstance(when, bool) or not isinstance(when, (int, float)):
                raise TypeError("AT_TIME expects a numeric TemporalRex clock time")
            if not isfinite(float(when)):
                raise ValueError("AT_TIME expects a finite TemporalRex clock time")
            if not hasattr(temporal, "reconstruct_at_time") or not hasattr(temporal, "T"):
                raise TypeError("AT_TIME expects a TemporalRex source")
            snapshot = temporal.reconstruct_at_time(float(when))
            if snapshot is None:
                raise ValueError(f"AT_TIME has no declared TemporalRex state at {when}")
            return BoundSource(snapshot, parent.policy) if isinstance(parent, BoundSource) else snapshot
        raise TypeError("FROM expects a source parameter, REX(name), CATALOG(name), "
                        "FILE(catalog, name), AT(temporal_source, version), or "
                        "AT_TIME(temporal_source, time)")

    @staticmethod
    def _source_label(expr: Expr) -> str:
        """Name a bound query source without evaluating an expression under it."""
        if isinstance(expr, Parameter):
            return expr.name
        if isinstance(expr, Call) and expr.name in {"REX", "CATALOG"}:
            if len(expr.args) == 1 and isinstance(expr.args[0], Literal):
                return str(expr.args[0].value)
            return expr.name.lower()
        if isinstance(expr, Call) and expr.name in {"AT", "AT_TIME"} and expr.args:
            return Executor._source_label(expr.args[0])
        if isinstance(expr, Call) and expr.name == "FILE":
            return "file"
        return "source"

    def _source_temporal(self, expr: Expr):
        """Attach a snapshot version to the static phrase binding when declared."""
        if not (isinstance(expr, Call) and expr.name in {"AT", "AT_TIME"} and len(expr.args) == 2):
            return None
        from .types import TemporalRef

        value = self._eval(expr.args[1], None)
        if expr.name == "AT":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("AT expects an integer TemporalRex snapshot version")
            return TemporalRef(version=value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("AT_TIME expects a numeric TemporalRex clock time")
        if not isfinite(float(value)):
            raise ValueError("AT_TIME expects a finite TemporalRex clock time")
        return TemporalRef(as_of=float(value))

    def _planning_binding(self, source_expr: Expr, source):
        """Make the same policy-aware binding used by static phrase planning."""
        from .binding import bind
        from .capabilities import BoundSource, SourcePolicy

        if isinstance(source, BoundSource):
            return bind(self._source_label(source_expr), source.value, source.policy,
                        temporal=self._source_temporal(source_expr))
        return bind(self._source_label(source_expr), source, SourcePolicy.allow("*"),
                    temporal=self._source_temporal(source_expr))

    def _eval(self, expr: Expr, source, *, memo: dict[Call, object] | None = None):
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, Parameter):
            if expr.name not in self.params:
                raise KeyError(f"unknown parameter ${expr.name}")
            return self.params[expr.name]
        if isinstance(expr, Call):
            name = expr.name.upper()
            cache_this = memo is not None and name in _MEMOIZABLE_OPERATORS
            if cache_this:
                try:
                    return memo[expr]
                except KeyError:
                    pass
                except TypeError:
                    # A literal can carry an unhashable Python value.  It remains a
                    # legitimate query input; only this outer expression cannot be
                    # a memoization key.  Its nested pure fragments may still share.
                    cache_this = False
            args = tuple(self._eval(arg, source, memo=memo) for arg in expr.args)
            permission = _PERMISSION.get(name, "read")
            raw, policy = self._unwrap(source, permission)
            value = get_operator(expr.name).fn(self._carrier_source(raw, args, name), *args)
            if policy is not None and name in {"RCDB_LIST", "RCDB_SEARCH", "RCDB_HISTORY"}:
                value = policy.project_record(value)
            if cache_this:
                with suppress(TypeError):
                    memo[expr] = value
            return value
        raise TypeError(f"unsupported expression {type(expr).__name__}")

    def execute(self, query: Query | MutationQuery) -> Result:
        source = self._eval_source(query.source)
        if isinstance(query, MutationQuery):
            raw, _policy = self._unwrap(source, ("mutate", "identity"))
            if not hasattr(raw, "commit_mutation"):
                raise TypeError("RCQL mutation expects an RCDB store")
            record_id = str(self._eval(query.record_id, source))
            resulting = self._eval(query.resulting, source)
            actor = str(self._eval(query.actor, source))
            valid_from = self._eval(query.valid_from, source)
            valid_to = self._eval(query.valid_to, source)
            rec = raw.commit_mutation(record_id, resulting, actor=actor,
                                      valid_from=valid_from, valid_to=valid_to)
            return Result((rec,), (), (f"COMMIT({record_id!r})",), (Exactness.STRUCTURAL,))
        planned, rewrites = optimize(query)
        if planned.explain:
            # No expression operator is evaluated in this branch.  The source is bound
            # once so type/capability/provenance checks have a real contract, then the
            # whole AST is typed recursively and returned as a plain structural value.
            from .planning import plan_query

            phrase = plan_query(
                self._planning_binding(planned.source, source), planned, parameters=self.params,
            )
            return Result(
                (phrase.explain(),), tuple(rewrites),
                tuple(format_expr(expr) for expr in planned.returns),
                (Exactness.STRUCTURAL,),
            )
        # A normal phrase receives the same contract check as EXPLAIN before even its
        # first adapter runs.  The returned plan is intentionally not discarded work:
        # it is the source/grade/basis/time proof for this execution, while runtime
        # remains responsible for data-dependent bounds and numerical residuals.
        from .planning import plan_query

        plan_query(
            self._planning_binding(planned.source, source), planned, parameters=self.params,
        )
        memo: dict[Call, object] = {}
        values = tuple(self._eval(expr, source, memo=memo) for expr in planned.returns)
        plan = tuple(format_expr(expr) for expr in planned.returns)
        exactness = tuple(value_exactness(value) for value in values)
        return Result(values, tuple(rewrites), plan, exactness)



def format_expr(expr: Expr) -> str:
    """Return a compact RCQL expression string."""
    if isinstance(expr, Literal):
        return repr(expr.value)
    if isinstance(expr, Parameter):
        return "$" + expr.name
    if isinstance(expr, Call):
        return f"{expr.name}({', '.join(format_expr(a) for a in expr.args)})"
    return repr(expr)


def value_exactness(value: Any) -> Exactness:
    """Classify the numeric representation returned by one expression.

    This reads a finished value, which is not type inference: it cannot answer before the
    work happens and it cannot distinguish a rational that was rendered to a float from a
    float that was never exact. Signature-driven inference replaces it wherever a
    signature exists; this remains the fallback for expressions that have none yet.

    Deliberately written without importing numpy. An array is recognised by carrying a
    ``dtype`` whose ``kind`` is a single character from the array protocol, so this
    classifies numpy values correctly without RCQL depending on numpy to do it. The
    exact tensor carriers come from the core library, and the only numpy in this stack
    belongs to the binary bundles beneath it.
    """
    from rexgraph.cells import CellBoundary, CellCoboundary, CompositeBinary
    from rexgraph.cochain import Chain, Cochain, Field
    from rexgraph.linear_operator import RexOperator
    from rexgraph.metric_field import MetricCurvature
    from rexgraph.temporal_signal import TemporalSignal, TemporalSignalFlow

    if isinstance(value, (bool, RexOperator)):
        return Exactness.STRUCTURAL
    if isinstance(value, int):
        return Exactness.INTEGER
    if isinstance(value, Fraction):
        return Exactness.RATIONAL
    if isinstance(value, (Chain, Cochain, Field)):
        return value_exactness(value.values)
    if isinstance(value, CellBoundary):
        return Exactness.STRUCTURAL if value.chain is None else value_exactness(value.chain)
    if isinstance(value, CellCoboundary):
        return value_exactness(value.cochain)
    if isinstance(value, CompositeBinary):
        return value_exactness(value.boundary)
    if isinstance(value, TemporalSignal):
        # The carrier is structural: individual channels retain their own
        # exactness, notably a numerical amplitude field beside exact topology.
        return Exactness.STRUCTURAL
    if isinstance(value, TemporalSignalFlow):
        return value_exactness(value.returned_boundary)
    if isinstance(value, MetricCurvature):
        # Integer relation metrics can produce rational local means and strain
        # through declared C1 share coefficients.  Classify the actual returned
        # field, not merely its input dtype.
        return value_exactness(value.local_mean)
    if isinstance(value, Mapping):
        # Compound field results must not hide the contract of their members.
        # In particular, SIGNAL_HODGE returns a named C1 split whose numerical
        # solver components are approximate even though their carrier/basis is
        # structurally well defined.  Preserve a uniform contract; a mixed
        # compound has no single arithmetic contract to promise.
        contracts = tuple(value_exactness(item) for item in value.values())
        if contracts and all(contract is contracts[0] for contract in contracts):
            return contracts[0]
        return Exactness.STRUCTURAL

    kind = getattr(getattr(value, "dtype", None), "kind", None)
    if kind is not None:
        if kind == "b":
            return Exactness.STRUCTURAL
        if kind in "iu":
            return Exactness.INTEGER
        if kind == "O":
            flat = getattr(value, "flat", None)
            if (getattr(value, "size", 1) and flat is not None
                    and all(isinstance(entry, Fraction) for entry in flat)):
                return Exactness.RATIONAL
            return Exactness.STRUCTURAL
        if kind in "fc":
            return Exactness.APPROXIMATE
        return Exactness.STRUCTURAL

    if isinstance(value, (float, complex)):
        return Exactness.APPROXIMATE
    return Exactness.STRUCTURAL
