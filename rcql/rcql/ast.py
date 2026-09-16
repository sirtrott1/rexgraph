"""Typed RCQL syntax tree."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class Expr:
    pass


@dataclass(frozen=True)
class Literal(Expr):
    value: Any


@dataclass(frozen=True)
class Parameter(Expr):
    name: str


def _local_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError("local binding name must be an ASCII identifier")
    if name.upper() in {"LET", "RETURN", "FROM", "EXPLAIN", "TRUE", "FALSE", "NONE", "MUTATE", "SET", "COMMIT",
                        "AS", "MATCH", "WHERE", "ORDER", "BY", "LIMIT", "OFFSET", "ASC", "DESC",
                        "AND", "OR", "NOT", "IN", "USING", "VERIFY", "ADD", "REMOVE"}:
        raise ValueError(f"{name!r} is reserved and cannot name a local binding")


@dataclass(frozen=True)
class Reference(Expr):
    """A query local name, distinct from an external $parameter."""

    name: str

    def __post_init__(self):
        _local_name(self.name)


@dataclass(frozen=True)
class LetBinding:
    """One eager, single assignment expression in declaration order."""

    name: str
    value: Expr

    def __post_init__(self):
        _local_name(self.name)
        if not isinstance(self.value, Expr):
            raise TypeError("LetBinding value must be an expression")


@dataclass(frozen=True)
class Call(Expr):
    name: str
    args: tuple[Expr, ...] = ()

    def __post_init__(self):
        from .names import canonical_name
        object.__setattr__(self, "name", canonical_name(self.name))


@dataclass(frozen=True)
class ListExpr(Expr):
    items: tuple[Expr, ...] = ()

    def __post_init__(self):
        if not isinstance(self.items, tuple) or not all(isinstance(item, Expr) for item in self.items):
            raise TypeError("ListExpr items must be a tuple of expressions")


@dataclass(frozen=True)
class StructuralEdit(Expr):
    state: Expr
    operation: str
    value: Expr

    def __post_init__(self):
        if self.operation not in {"ADD", "REMOVE"}:
            raise ValueError("structural edit must be ADD or REMOVE")
        if not isinstance(self.state, Expr) or not isinstance(self.value, Expr):
            raise TypeError("structural edit requires expressions")


@dataclass(frozen=True)
class Comparison(Expr):
    operation: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class MatchBinding:
    name: str
    value: Expr

    def __post_init__(self):
        _local_name(self.name)
        if not isinstance(self.value, Expr):
            raise TypeError("MATCH value must be an expression")


@dataclass(frozen=True)
class Member(Expr):
    value: Expr
    name: str

    def __post_init__(self):
        _local_name(self.name)
        if self.name.startswith("_"):
            raise ValueError("member names must be public record keys")
        if not isinstance(self.value, Expr):
            raise TypeError("Member value must be an expression")


@dataclass(frozen=True)
class Alias(Expr):
    value: Expr
    name: str

    def __post_init__(self):
        _local_name(self.name)
        if not isinstance(self.value, Expr):
            raise TypeError("Alias value must be an expression")


@dataclass(frozen=True)
class Query:
    source: Expr
    returns: tuple[Expr, ...]
    explain: bool = False
    bindings: tuple[LetBinding, ...] = ()
    source_alias: str | None = None
    matches: tuple[MatchBinding, ...] = ()
    where: Expr | None = None
    order: tuple[tuple[Expr, bool], ...] = ()
    limit: int | None = None
    offset: int = 0

    def __post_init__(self):
        if self.source_alias is not None:
            _local_name(self.source_alias)
        if not isinstance(self.source, Expr) or not self.returns or any(
                not isinstance(value, Expr) for value in self.returns):
            raise TypeError("query requires a source expression and returned expressions")
        if any(not isinstance(value, MatchBinding) for value in self.matches):
            raise TypeError("MATCH requires MatchBinding values")
        if self.where is not None and not isinstance(self.where, Expr):
            raise TypeError("WHERE requires an expression")
        for item in self.order:
            if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], Expr) or type(item[1]) is not bool:
                raise TypeError("ORDER requires expression and boolean direction pairs")
        for value in ((self.offset,) if self.limit is None else (self.limit, self.offset)):
            if type(value) is not int or value < 0:
                raise ValueError("LIMIT and OFFSET require nonnegative integers")
        if not self.matches and (self.where is not None or self.order or self.limit is not None or self.offset):
            raise ValueError("WHERE/ORDER/LIMIT/OFFSET require MATCH bindings")


@dataclass(frozen=True)
class MutationQuery:
    """Typed RCQL request to append one canonical Rex state transition."""

    source: Expr
    record_id: Expr
    resulting: Expr
    actor: Expr = Literal("")
    valid_from: Expr = Literal(None)
    valid_to: Expr = Literal(None)
    expected_version: Expr = Literal(None)
    explain: bool = False
    bindings: tuple[LetBinding, ...] = ()
    source_alias: str | None = None
    expected_hash: Expr = Literal(None)

    def __post_init__(self):
        if self.source_alias is not None:
            _local_name(self.source_alias)
