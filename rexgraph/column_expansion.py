"""Exact elementary coordinates with the primary boundary lift retained.

For each column b and reference h, b = sum_i b_i (e_i-e_h) + sum(b) e_h.
Legs are internal coordinates, never independent cells of a replacement Rex.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property

import numpy as np

from rexgraph.chain_map import _columns, _exact_entries
from rexgraph.linear_operator import RexOperator
from rexgraph.native_rank import exact_tower
from rexgraph.native_sparse import native_coo
from rexgraph.type_accession import _action, _digest

__all__ = ["ColumnExpansion", "ColumnLegs", "PrimaryColumnLift", "PrimaryBoundary", "primary_lift"]


def _capture(source):
    shapes, columns = exact_tower(source)
    maps = tuple(tuple(sorted((i, j, Fraction(c)) for j, col in enumerate(cols) for i, c in col.items()))
                 for cols in columns)
    primary = tuple(tuple(map(int, s)) for s in source.relation_supports())
    ids = None if source.relation_ids is None else tuple(map(int, source.relation_ids))
    return tuple(shapes), maps, primary, ids


def _check_boundary(boundary):
    b = boundary
    if (not isinstance(b, RexOperator) or b.construction != "boundary" or b.source is None
            or b.codomain_grade != b.domain_grade - 1 or b.variance != "chain"
            or b.boundary_entries is None or b.exact_matvec is None):
        raise TypeError("column expansion requires a certified canonical boundary operator")
    state = _capture(b.source)
    shapes, maps, _, _ = state
    if (not 1 <= b.domain_grade <= len(shapes) or b.shape != shapes[b.domain_grade-1]
            or _exact_entries(b.boundary_entries, b.shape) != maps[b.domain_grade-1]):
        raise ValueError("boundary operator differs from the current exact source; bind it again")
    return state


@dataclass(frozen=True, eq=False)
class _Factor:
    expansion: ColumnExpansion
    shape: tuple[int, int]
    entries: tuple

    @property
    def source(self):
        return self.expansion.source

    @property
    def grade(self):
        return self.expansion.grade

    @cached_property
    def _native(self):
        try:
            weights = np.asarray([float(c) for _, _, c in self.entries])
        except OverflowError as exc:
            raise FloatingPointError("column factors cannot be represented numerically") from exc
        if not np.all(np.isfinite(weights)) or np.any(weights == 0):
            raise FloatingPointError("column factors cannot be represented numerically")
        return native_coo([i for i, _, _ in self.entries], [j for _, j, _ in self.entries], weights, self.shape)

    def _apply(self, values, *, exact=False, transpose=False):
        if exact:
            entries = tuple((j, i, c) for i, j, c in self.entries) if transpose else self.entries
            return _action(entries, self.shape[::-1] if transpose else self.shape, values, exact=True)
        return self._native.apply(values, transpose=transpose)

    def apply(self, values, *, exact=False, transpose=False):
        if not isinstance(exact, (bool, np.bool_)) or not isinstance(transpose, (bool, np.bool_)):
            raise TypeError("exact and transpose must be booleans")
        self.expansion.check_state()
        return self._apply(values, exact=exact, transpose=transpose)


class ColumnLegs(_Factor):
    """P maps the ordered internal leg coordinates into the lower cell grade."""


class PrimaryColumnLift(_Factor):
    """Lambda maps primary coefficients into their own ordered leg groups."""


@dataclass(frozen=True, eq=False)
class ColumnExpansion:
    """Factor one canonical boundary operator over Q in linear incidence storage.

The reference is the declared C1 head, even if repeated slots cancel it. Above
C1 it is the first nonzero row. A zero column has no legs but retains its
primary position. Nonzero column sums have an explicit witness leg.
"""

    boundary: RexOperator
    _state: tuple = field(init=False, repr=False)
    legs: ColumnLegs = field(init=False)
    lift: PrimaryColumnLift = field(init=False)
    references: tuple = field(init=False)
    groups: tuple = field(init=False)
    coefficient_digest: str = field(init=False)

    def __post_init__(self):
        b = self.boundary
        state = _check_boundary(b)
        _, maps, primary, _ = state
        entries = maps[self.grade-1]
        p, lift, references, groups = [], [], [], []
        count = 0
        for j, column in enumerate(_columns(entries, b.shape[1])):
            support = primary[j] if self.grade == 1 else ()
            # the reference is the head: the participant carrying the negative
            # coefficient, which is slot zero only when nothing else is declared
            head = next((i for i, c in sorted(column.items()) if c < 0), None)
            h = head if (self.grade == 1 and head is not None) else (
                support[0] if support else min(column, default=None))
            references.append(h)
            start = count
            for i, value in sorted(column.items()):
                if i == h:
                    continue
                p.extend(((i, count, Fraction(1)), (h, count, Fraction(-1))))
                lift.append((count, j, value))
                count += 1
            witness = sum(column.values(), Fraction(0))
            if witness:
                p.append((h, count, Fraction(1)))
                lift.append((count, j, witness))
                count += 1
            groups.append((start, count))
        p, lift = tuple(p), tuple(lift)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "legs", ColumnLegs(self, (b.shape[0], count), p))
        object.__setattr__(self, "lift", PrimaryColumnLift(self, (count, b.shape[1]), lift))
        object.__setattr__(self, "references", tuple(references))
        object.__setattr__(self, "groups", tuple(groups))
        object.__setattr__(self, "coefficient_digest", _digest(p + lift,
            f"column-expansion-v1|{self.grade}|{state}|{references}|{groups}|"))

    @property
    def source(self):
        return self.boundary.source

    @property
    def grade(self):
        return self.boundary.domain_grade

    def check_state(self):
        if _capture(self.source) != self._state:
            raise ValueError("column expansion source boundary state changed; bind it again")


class PrimaryBoundary(RexOperator):
    """B = P Lambda, applied through its exact or compiled sparse factors."""

    def __init__(self, expansion):
        p, lift = expansion.legs, expansion.lift

        def action(values, *, exact=False, transpose=False):
            expansion.check_state()
            first, second = (p, lift) if transpose else (lift, p)
            return second._apply(first._apply(values, exact=exact, transpose=transpose),
                                 exact=exact, transpose=transpose)

        def matrix():
            expansion.check_state()
            return expansion.boundary.as_native()

        def rank():
            expansion.check_state()
            return expansion.boundary.exact_rank_factory()

        super().__init__(f"PrimaryB{expansion.grade}", expansion.boundary.shape,
            expansion.grade, expansion.grade-1, action, source=expansion.source, matrix_factory=matrix,
            construction="primary-lift", variance="chain",
            transpose_matvec=lambda v: action(v, transpose=True),
            exact_matvec=lambda v: action(v, exact=True),
            exact_transpose_matvec=lambda v: action(v, exact=True, transpose=True),
            parameters=(("expansion_digest", expansion.coefficient_digest),),
            exact_rank_factory=rank if expansion.boundary.exact_rank_factory is not None else None,
            boundary_entries=expansion.boundary.boundary_entries)
        object.__setattr__(self, "expansion", expansion)


def primary_lift(legs, lift):
    """Reconstruct B from factors of the identical certified expansion."""
    if not isinstance(legs, ColumnLegs) or not isinstance(lift, PrimaryColumnLift):
        raise TypeError("primary lift requires ColumnLegs and PrimaryColumnLift")
    if legs.expansion is not lift.expansion or legs is not legs.expansion.legs or lift is not lift.expansion.lift:
        raise ValueError("primary lift requires the original factors of one expansion")
    legs.expansion.check_state()
    return PrimaryBoundary(legs.expansion)
