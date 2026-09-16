"""Scalar predicates retain integer/rational values; no float threshold cast."""
from math import isfinite
from numbers import Rational, Real
import operator

from .types import RCType, ValueKind, Domain, Exactness

OPS = {"=": operator.eq, "==": operator.eq, "!=": operator.ne, "<": operator.lt,
       "<=": operator.le, ">": operator.gt, ">=": operator.ge,
       "AND": operator.and_, "OR": operator.or_, "NOT": lambda a, b: not a}


def scalar_kind(value):
    """Classify a scalar without casting integers or rationals to floats."""
    if isinstance(value, RCType):
        if value.kind is ValueKind.BOOLEAN:
            return "boolean"
        if value.kind in {ValueKind.EXACT_INTEGER, ValueKind.EXACT_RATIONAL, ValueKind.REAL}:
            return "number"
        if value.kind is ValueKind.TEXT:
            return "text"
        if value.kind is ValueKind.UNKNOWN and value.domain is Domain.METADATA:
            return "deferred"
        raise TypeError("comparison and ORDER BY require scalar readings")
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "text"
    if isinstance(value, Real):
        if not isinstance(value, Rational) and not isfinite(value):
            raise ValueError("comparison and ORDER BY require finite numbers")
        return "number"
    raise TypeError("comparison and ORDER BY require scalar readings")


def order_key(value):
    scalar_kind(value)
    return (value is not None, value)


def validate(operation, values):
    if operation not in OPS:
        raise TypeError("unknown comparison")
    kinds = {scalar_kind(value) for value in values}
    if operation in {"AND", "OR", "NOT"}:
        if kinds != {"boolean"}:
            raise TypeError("AND/OR/NOT require boolean predicates")
    elif operation not in {"=", "==", "!="}:
        concrete = kinds - {"deferred"}
        if "none" in concrete or len(concrete) > 1:
            raise TypeError("ordered comparison requires compatible scalar readings")
    return RCType("Boolean", kind=ValueKind.BOOLEAN, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)


def evaluate(operation, left, right):
    validate(operation, (left, right))
    return bool(OPS[operation](left, right))
