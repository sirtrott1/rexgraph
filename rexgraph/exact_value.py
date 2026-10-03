"""One source aware conversion from scalar carriers to exact rationals.

Exactness is a property of the source, not of the act of wrapping a value in
``Fraction``.  This module keeps the three materially different cases explicit:

``EXACT``
    The source is already an exact carrier.  Integers, NumPy integers, Fractions,
    Decimals and (when requested) ``(numerator, denominator)`` pairs are accepted;
    binary floating values are refused.

``BINARY_FLOAT``
    A stored IEEE floating value is itself the source value.  It is converted to the
    exact rational represented by that binary value.  This is the contract used for
    persisted numerical measurements and legacy ``w_E`` arrays.

``DECIMAL_TEXT``
    Decimal text is the source value.  Strings and Decimals are converted exactly as
    decimal numbers.  A binary float is never silently reinterpreted as decimal text.

Callers therefore choose the source contract once, at the boundary where the value's
provenance is known, instead of each mathematical subsystem inventing a different
``Fraction`` policy.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
from numbers import Integral
from typing import Any, Literal

import numpy as np

__all__ = [
    "EXACT",
    "BINARY_FLOAT",
    "DECIMAL_TEXT",
    "RationalSource",
    "as_fraction",
    "exact_fraction",
    "binary_fraction",
    "decimal_fraction",
    "validate_exact_array",
]

RationalSource = Literal["exact", "binary-float", "decimal-text"]
EXACT: RationalSource = "exact"
BINARY_FLOAT: RationalSource = "binary-float"
DECIMAL_TEXT: RationalSource = "decimal-text"


def _pair(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, tuple) or len(value) != 2:
        return None
    numerator, denominator = value
    if (isinstance(numerator, (bool, np.bool_))
            or isinstance(denominator, (bool, np.bool_))
            or not isinstance(numerator, Integral)
            or not isinstance(denominator, Integral)):
        return None
    return int(numerator), int(denominator)


def as_fraction(
    value: Any,
    *,
    source: RationalSource = EXACT,
    allow_pair: bool = False,
    context: str = "exact coefficient",
) -> Fraction:
    """Convert one scalar according to its declared source kind.

    Booleans are always refused.  NumPy integer and floating scalar types follow the
    same contract as their Python counterparts.  Nonfinite floating values are
    refused before conversion so no subsystem gets its own NaN/Inf policy by accident.
    """
    if source not in {EXACT, BINARY_FLOAT, DECIMAL_TEXT}:
        raise ValueError(f"unknown rational source kind {source!r}")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{context} cannot be boolean")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if allow_pair:
        pair = _pair(value)
        if pair is not None:
            return Fraction(*pair)

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{context} must be finite")
        return Fraction(value)

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"{context} must be finite")
        if source == BINARY_FLOAT:
            return Fraction(*value.as_integer_ratio())
        raise TypeError(
            f"{context} received {type(value).__name__}, an approximate binary value; "
            "convert at the source or declare the binary-float source contract"
        )

    if isinstance(value, str):
        if source != DECIMAL_TEXT:
            raise TypeError(f"{context} accepts decimal text only under the decimal-text contract")
        try:
            decimal = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"{context} is not valid decimal text") from exc
        if not decimal.is_finite():
            raise ValueError(f"{context} must be finite")
        return Fraction(decimal)

    if source == DECIMAL_TEXT:
        raise TypeError(f"{context} must be decimal text, Decimal, integer, or Fraction")
    if source == BINARY_FLOAT:
        raise TypeError(f"{context} must be an integer, Fraction, Decimal, or real floating scalar")
    suffix = ", or an integer pair" if allow_pair else ""
    raise TypeError(f"{context} must be an integer, Fraction, Decimal{suffix}")


def exact_fraction(value: Any, *, allow_pair: bool = False, context: str = "exact coefficient") -> Fraction:
    return as_fraction(value, source=EXACT, allow_pair=allow_pair, context=context)


def binary_fraction(value: Any, *, context: str = "stored numeric coefficient") -> Fraction:
    return as_fraction(value, source=BINARY_FLOAT, context=context)


def decimal_fraction(value: Any, *, context: str = "decimal coefficient") -> Fraction:
    return as_fraction(value, source=DECIMAL_TEXT, context=context)


def validate_exact_array(values: Any, *, context: str = "exact object tensor") -> np.ndarray:
    """Return an object array view after validating its exact coefficient carriers.

    This is intentionally a semantic validator, not a persistence codec.  Storage
    layers may encode the validated integers/Fractions however they choose, while
    model/state code can enforce the exact carrier contract without importing IO.
    """
    array = np.asarray(values, dtype=object)
    for value in array.flat:
        if isinstance(value, Fraction):
            continue
        if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
            continue
        raise TypeError(f"{context} require integer or Fraction coefficients")
    return array
