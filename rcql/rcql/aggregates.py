"""Scalar sequence reductions with declared arithmetic and empty values."""
from fractions import Fraction
from math import fsum, isfinite
from numbers import Integral, Real

from .types import Domain, Exactness, RCType, ValueKind


def numeric_kind(value):
    if isinstance(value, RCType):
        kinds = {ValueKind.EXACT_INTEGER: "integer", ValueKind.EXACT_RATIONAL: "rational",
                 ValueKind.REAL: "real"}
        if value.kind in kinds:
            return kinds[value.kind]
    elif isinstance(value, bool):
        pass
    elif isinstance(value, Integral):
        return "integer"
    elif isinstance(value, Fraction):
        return "rational"
    elif isinstance(value, Real):
        if not isfinite(value):
            raise ValueError("SUM and MEAN require finite scalar coefficients")
        return "real"
    raise TypeError("SUM and MEAN require a sequence of real scalar coefficients")


def result_type(values, *, mean=False):
    if not isinstance(values, (list, tuple)):
        raise TypeError("SUM and MEAN require an explicit scalar sequence")
    kinds = {numeric_kind(value) for value in values}
    if mean and not values:
        return RCType("EmptyMean", domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)
    if "real" in kinds:
        return RCType("Real", kind=ValueKind.REAL, domain=Domain.REAL, exactness=Exactness.APPROXIMATE)
    if mean or "rational" in kinds:
        return RCType("Rational", kind=ValueKind.EXACT_RATIONAL, domain=Domain.RATIONAL,
                      exactness=Exactness.RATIONAL)
    return RCType("Integer", kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER, exactness=Exactness.INTEGER)


def reduce_scalars(values, *, mean=False):
    contract = result_type(values, mean=mean)
    if mean and not values:
        return None
    if contract.domain is Domain.REAL:
        total = fsum(values)
    else:
        start = Fraction(0) if contract.domain is Domain.RATIONAL else 0
        total = sum((int(value) if isinstance(value, Integral) else value for value in values), start)
    return total / len(values) if mean else total


def count_type(values):
    if not isinstance(values, (list, tuple)) and not (
            isinstance(values, RCType) and values.kind in {
                ValueKind.CELL_SET, ValueKind.RECORD_SET, ValueKind.CATALOG_ENTRY_SET,
                ValueKind.SEQUENCE, ValueKind.OPERATOR_SIGNATURE_SET}):
        raise TypeError("COUNT requires a finite sequence or native cell selection")
    return RCType("Integer", kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER, exactness=Exactness.INTEGER)
