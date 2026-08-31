"""Algebraic rewrites for RCQL expressions."""
from __future__ import annotations

from dataclasses import dataclass

from .ast import Call, Expr, Literal, Query


@dataclass(frozen=True)
class Rewrite:
    before: Expr
    after: Expr
    reason: str


def optimize_expr(expr: Expr) -> tuple[Expr, list[Rewrite]]:
    if not isinstance(expr, Call):
        return expr, []
    args = []
    rewrites = []
    for arg in expr.args:
        got, used = optimize_expr(arg)
        args.append(got)
        rewrites.extend(used)
    current = Call(expr.name, tuple(args))
    if current.name == "BOUNDARY" and len(current.args) == 2:
        inner = current.args[1]
        if isinstance(inner, Call) and inner.name == "BOUNDARY" and len(inner.args) == 2:
            outer_grade = current.args[0]
            inner_grade = inner.args[0]
            if (isinstance(outer_grade, Literal) and isinstance(inner_grade, Literal)
                    and int(inner_grade.value) == int(outer_grade.value) + 1):
                after = Call("ZERO", (Literal(int(outer_grade.value) - 1),))
                rewrites.append(Rewrite(current, after, "consecutive boundaries compose to zero"))
                return after, rewrites
    return current, rewrites


def optimize(query: Query) -> tuple[Query, list[Rewrite]]:
    returns = []
    rewrites = []
    for expr in query.returns:
        got, used = optimize_expr(expr)
        returns.append(got)
        rewrites.extend(used)
    return Query(query.source, tuple(returns), query.explain), rewrites
