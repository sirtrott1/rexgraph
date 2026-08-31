"""RCQL execution."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

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
        raise TypeError("FROM expects a source parameter, REX(name), CATALOG(name), "
                        "or FILE(catalog, name)")

    def _eval(self, expr: Expr, source):
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, Parameter):
            if expr.name not in self.params:
                raise KeyError(f"unknown parameter ${expr.name}")
            return self.params[expr.name]
        if isinstance(expr, Call):
            args = tuple(self._eval(arg, source) for arg in expr.args)
            name = expr.name.upper()
            permission = _PERMISSION.get(name, "read")
            raw, policy = self._unwrap(source, permission)
            value = get_operator(expr.name).fn(raw, *args)
            if policy is not None and name in {"RCDB_LIST", "RCDB_SEARCH", "RCDB_HISTORY"}:
                value = policy.project_record(value)
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
        values = tuple(self._eval(expr, source) for expr in planned.returns)
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
    """Classify the numeric representation returned by one expression."""
    from rexgraph.cochain import Cochain, Field
    from rexgraph.linear_operator import RexOperator

    if isinstance(value, (bool, np.bool_, RexOperator)):
        return Exactness.STRUCTURAL
    if isinstance(value, (int, np.integer)):
        return Exactness.INTEGER
    if isinstance(value, Fraction):
        return Exactness.RATIONAL
    if isinstance(value, (Cochain, Field)):
        return value_exactness(value.values)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.integer):
            return Exactness.INTEGER
        if (value.dtype == object and value.size
                and all(isinstance(x, Fraction) for x in value.flat)):
            return Exactness.RATIONAL
        if (np.issubdtype(value.dtype, np.floating)
                or np.issubdtype(value.dtype, np.complexfloating)):
            return Exactness.APPROXIMATE
        return Exactness.STRUCTURAL
    if isinstance(value, (float, complex, np.floating, np.complexfloating)):
        return Exactness.APPROXIMATE
    return Exactness.STRUCTURAL
