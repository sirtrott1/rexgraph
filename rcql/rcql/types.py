"""RCQL value types."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Exactness(str, Enum):
    INTEGER = "integer"
    RATIONAL = "rational"
    ROUNDED = "rounded"
    APPROXIMATE = "approximate"
    ANALYTIC = "analytic"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class RCType:
    """The RCQL type of one value."""

    name: str
    grade: int | None = None
    exactness: Exactness | None = None


INTEGER = RCType("Integer", exactness=Exactness.INTEGER)
RATIONAL = RCType("Rational", exactness=Exactness.RATIONAL)
BOOLEAN = RCType("Boolean", exactness=Exactness.STRUCTURAL)
REX = RCType("Rex")
TENSOR = RCType("Tensor")
