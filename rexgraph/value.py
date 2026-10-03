"""Declared scalar values at reader, state, query and persistence boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral

import numpy as np

from .exact_value import binary_fraction, decimal_fraction, exact_fraction

__all__ = ["Absent", "Approx", "NumberRule", "ValueRules", "convert_number", "ExactTime", "TimeRange"]


def _absent():
    return Absent


class _AbsentType:
    __slots__ = ()

    def __repr__(self):
        return "Absent"

    def __bool__(self):
        raise TypeError("absence has no truth value; test 'value is Absent'")

    def __reduce__(self):
        return _absent, ()


Absent = _AbsentType()


@dataclass(frozen=True)
class Approx:
    value: float
    source: str | None = None

    def __post_init__(self):
        if isinstance(self.value, (bool, np.bool_)) or not isinstance(self.value, (float, np.floating, Integral)):
            raise TypeError("Approx requires a real numeric value")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("Approx requires a finite value")
        if self.source is not None and not isinstance(self.source, str):
            raise TypeError("Approx source must be text or None")
        object.__setattr__(self, "value", value)


class NumberRule(str, Enum):
    EXACT = "exact"
    DECIMAL_EXACT = "decimal_exact"
    XLSX_SHORTEST_DECIMAL = "xlsx_shortest_decimal"
    BINARY_EXACT = "binary_exact"
    JSON_RATIONAL = "json_rational"
    TYPED_COLUMNAR = "typed_columnar"


@dataclass(frozen=True)
class ValueRules:
    number: NumberRule = NumberRule.EXACT
    absent_tokens: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "number", NumberRule(self.number))
        tokens = tuple(self.absent_tokens)
        if any(type(token) is not str for token in tokens) or len(set(tokens)) != len(tokens):
            raise ValueError("absence tokens must be distinct strings")
        object.__setattr__(self, "absent_tokens", tokens)

    def convert(self, value, *, kind: str = "number", context: str = "source value"):
        if value is Absent or value is None or (isinstance(value, str) and value in self.absent_tokens):
            return Absent
        if kind == "number":
            if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
                return int(value)
            return convert_number(value, self.number, context=context)
        if kind == "string" and isinstance(value, str):
            return value
        if kind == "bytes" and isinstance(value, bytes):
            return value
        if kind == "bool" and isinstance(value, (bool, np.bool_)):
            return bool(value)
        raise TypeError(f"{context} does not match declared kind {kind!r}")


def convert_number(value, rule: NumberRule = NumberRule.EXACT, *, context="source number"):
    """Convert using the source declaration, never a value's visual appearance."""
    rule = NumberRule(rule)
    if value is Absent:
        return Absent
    if isinstance(value, Approx):
        return value
    if rule == NumberRule.DECIMAL_EXACT:
        return decimal_fraction(value, context=context)
    if rule == NumberRule.XLSX_SHORTEST_DECIMAL:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                raise ValueError(f"{context} must be finite")
            value = str(value)
        return decimal_fraction(value, context=context)
    if rule in (NumberRule.BINARY_EXACT, NumberRule.TYPED_COLUMNAR):
        return binary_fraction(value, context=context)
    if rule == NumberRule.JSON_RATIONAL:
        if isinstance(value, dict):
            if set(value) != {"numerator", "denominator"}:
                raise ValueError(f"{context} requires numerator and denominator")
            value = (value["numerator"], value["denominator"])
        if isinstance(value, list):
            value = tuple(value)
        return exact_fraction(value, allow_pair=True, context=context)
    return exact_fraction(value, context=context)


@dataclass(frozen=True, order=True)
class ExactTime:
    """Exact UTC seconds since the Unix epoch; no binary timestamp rounding."""
    seconds: Fraction

    def __post_init__(self):
        object.__setattr__(self, "seconds", exact_fraction(self.seconds, context="timestamp"))

    @classmethod
    def from_datetime(cls, value: datetime):
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ValueError("timestamp requires an aware datetime")
        delta = value.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
        return cls(Fraction(delta.days * 86400 + delta.seconds) + Fraction(delta.microseconds, 1_000_000))


@dataclass(frozen=True)
class TimeRange:
    """Half open UTC interval; an absent end means it has no upper bound."""
    start: ExactTime
    end: ExactTime | _AbsentType = Absent

    def __post_init__(self):
        if not isinstance(self.start, ExactTime) or (self.end is not Absent and not isinstance(self.end, ExactTime)):
            raise TypeError("time range requires ExactTime endpoints")
        if self.end is not Absent and self.end < self.start:
            raise ValueError("time range end precedes start")

    def contains(self, value: ExactTime):
        if not isinstance(value, ExactTime):
            raise TypeError("time range membership requires ExactTime")
        return self.start <= value and (self.end is Absent or value < self.end)
