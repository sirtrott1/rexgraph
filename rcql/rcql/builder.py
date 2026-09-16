"""Small builders for typed RCQL queries."""
from __future__ import annotations

from .ast import (
    Alias,
    Call,
    Expr,
    LetBinding,
    ListExpr,
    Literal,
    Member,
    MutationQuery,
    Parameter,
    Query,
    Reference,
    StructuralEdit,
)


def expr(value) -> Expr:
    """Convert a Python value to an RCQL expression."""
    if isinstance(value, list):
        return ListExpr(tuple(expr(item) for item in value))
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


def rcdb(name) -> Call:
    """Resolve an explicitly registered store by name, not a backend URI."""
    return call("RCDB", name)


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
    """Bind one policy aware explicit phrase section as the source of a RCQL query."""
    return call("PHRASE", section)


def call(name: str, /, *args, **keywords) -> Call:
    """Build one operator call."""
    from .arguments import bind_arguments
    from .names import canonical_name
    name = canonical_name(name)
    return Call(name, bind_arguments(name, tuple(expr(arg) for arg in args),
                                     tuple((key, expr(value)) for key, value in keywords.items())))


def source_call(name: str, /, *args, **keywords) -> Call:
    """Build a FROM transform with its source specific argument contract."""
    from .arguments import bind_arguments
    from .names import canonical_name
    name = canonical_name(name)
    return Call(name, bind_arguments(name, tuple(expr(arg) for arg in args),
               tuple((key, expr(value)) for key, value in keywords.items()), source=True))


def alias(value, name: str) -> Alias:
    """Name one returned expression; not a query local variable assignment."""
    return Alias(expr(value), name)


def member(value, name: str) -> Member:
    """Project a declared record member without Python attribute access."""
    return Member(expr(value), name)


def ref(name: str) -> Reference:
    """Refer to a prior LET binding, not an external parameter or operator."""
    return Reference(name)


def let(name: str, value) -> LetBinding:
    """Bind one expression for query local reuse."""
    return LetBinding(name, expr(value))


def query(source_expr, *returns, explain: bool = False, bindings=(), source_alias=None,
          matches=(), where=None, order=(), limit=None, offset=0) -> Query:
    """Build a query from typed expressions."""
    return Query(expr(source_expr), tuple(expr(item) for item in returns), bool(explain), tuple(bindings), source_alias,
                 tuple(matches), None if where is None else expr(where),
                 tuple((expr(value), reverse) for value, reverse in order), limit, offset)


def mutation(source_expr, record_id, resulting, *, actor="", valid_from=None, valid_to=None,
             expected_version=None, expected_hash=None, explain=False, bindings=(), source_alias=None,
             edits=()):
    """Build one typed RCQL mutation without executable query text."""
    state = expr(resulting)
    for operation, value in edits:
        state = StructuralEdit(state, operation, expr(value))
    return MutationQuery(expr(source_expr), expr(record_id), state, expr(actor),
                         expr(valid_from), expr(valid_to), expr(expected_version),
                         bool(explain), tuple(bindings), source_alias, expr(expected_hash))
