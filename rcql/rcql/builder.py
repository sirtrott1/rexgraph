"""Small builders for typed RCQL queries."""
from __future__ import annotations

from .ast import Call, Expr, Literal, MutationQuery, Parameter, Query


def expr(value) -> Expr:
    """Convert a Python value to an RCQL expression."""
    return value if isinstance(value, Expr) else Literal(value)


def param(name: str) -> Parameter:
    """Return a named query parameter."""
    return Parameter(str(name))


def source(name: str) -> Parameter:
    """Return a named query source."""
    return Parameter(str(name))


def at(source_expr, version: int) -> Call:
    """Bind a TemporalRex source to one exact snapshot version."""
    return call("AT", source_expr, version)


def at_time(source_expr, when: float) -> Call:
    """Bind a TemporalRex source to the state declared at one clock time."""
    return call("AT_TIME", source_expr, when)


def call(name: str, *args) -> Call:
    """Build one operator call."""
    return Call(str(name).upper(), tuple(expr(arg) for arg in args))


def query(source_expr, *returns, explain: bool = False) -> Query:
    """Build a query from typed expressions."""
    return Query(expr(source_expr), tuple(expr(item) for item in returns), bool(explain))


def mutation(source_expr, record_id, resulting, *, actor="", valid_from=None, valid_to=None):
    """Build one typed RCQL mutation without executable query text."""
    return MutationQuery(expr(source_expr), expr(record_id), expr(resulting), expr(actor),
                         expr(valid_from), expr(valid_to))
