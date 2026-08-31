"""Typed RCQL syntax tree."""
from __future__ import annotations

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


@dataclass(frozen=True)
class Call(Expr):
    name: str
    args: tuple[Expr, ...] = ()


@dataclass(frozen=True)
class Query:
    source: Expr
    returns: tuple[Expr, ...]
    explain: bool = False


@dataclass(frozen=True)
class MutationQuery:
    """Typed RCQL request to append one canonical Rex state transition."""

    source: Expr
    record_id: Expr
    resulting: Expr
    actor: Expr = Literal("")
    valid_from: Expr = Literal(None)
    valid_to: Expr = Literal(None)
