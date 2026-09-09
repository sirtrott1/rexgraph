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


def rcdb_get(source_expr, record_id) -> Call:
    """Bind one current RCDB record as the source for a structural phrase."""
    return call("RCDB_GET", source_expr, record_id)


def rcdb_version(source_expr, record_id, version: int) -> Call:
    """Bind one exact persisted RCDB version as the source for a structural phrase."""
    return call("RCDB_VERSION", source_expr, record_id, version)


def rcdb_as_of(source_expr, record_id, when: float) -> Call:
    """Bind the record state current at one exact RCDB transaction time."""
    return call("RCDB_AS_OF", source_expr, record_id, when)


def rcdb_valid_at(source_expr, record_id, when: float) -> Call:
    """Bind the record state valid at one exact RCDB valid time."""
    return call("RCDB_VALID_AT", source_expr, record_id, when)


def phrase(section) -> Call:
    """Bind one policy-aware explicit phrase section as the source of a RCQL query."""
    return call("PHRASE", section)


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
