"""Typed cochains and collections of graded values.

A value array does not identify the cell space on which it lives.  These small
wrappers keep its grade and, when available, its ordered basis and source rex
beside it so callers can preserve that identity across operator applications.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Cochain", "Field", "GradedState"]


@dataclass(frozen=True)
class Cochain:
    """Values on one grade of a relational complex.

    ``values.shape[0]`` is the cell axis.  ``cell_keys`` is optional but, when
    present, names that basis in the same order.
    """

    grade: int
    values: Any
    cell_keys: Any = None
    source: Any = None

    def __post_init__(self) -> None:
        grade = int(self.grade)
        if grade < 0:
            raise ValueError("grade must be >= 0")
        shape = getattr(self.values, "shape", None)
        if shape is None or len(shape) == 0:
            raise ValueError("values must have a cell axis")
        if self.cell_keys is not None and len(self.cell_keys) != int(shape[0]):
            raise ValueError("cell_keys must match the first value axis")
        object.__setattr__(self, "grade", grade)

    @property
    def n_cells(self) -> int:
        """Number of cells in the cochain's basis."""
        return int(self.values.shape[0])

    def numpy(self, *, dtype=None, copy: bool = False) -> np.ndarray:
        """Return the values as a NumPy array, detaching a torch tensor if needed."""
        if hasattr(self.values, "detach"):
            out = self.values.detach().cpu().numpy()
        else:
            out = np.asarray(self.values)
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out.copy() if copy else out

    def with_values(self, values) -> Cochain:
        """Return new values on the same grade and ordered basis."""
        return Cochain(self.grade, values, self.cell_keys, self.source)


@dataclass(frozen=True)
class Field:
    """A cochain together with the operator that gives it field meaning."""

    cochain: Cochain
    operator: Any
    kind: str = "field"

    def __post_init__(self) -> None:
        if not isinstance(self.cochain, Cochain):
            raise TypeError("field cochain must be a Cochain")

    @property
    def grade(self) -> int:
        return self.cochain.grade

    @property
    def values(self):
        return self.cochain.values


class GradedState:
    """A collection containing at most one cochain at each grade."""

    def __init__(self, cochains=()):
        self._grades: dict[int, Cochain] = {}
        if isinstance(cochains, dict):
            for grade, cochain in cochains.items():
                if int(grade) != getattr(cochain, "grade", None):
                    raise ValueError("graded state key must match the cochain grade")
                self.add(cochain)
        else:
            for cochain in cochains:
                self.add(cochain)

    def add(self, cochain: Cochain) -> None:
        """Add or replace the cochain at its grade."""
        if not isinstance(cochain, Cochain):
            raise TypeError("graded state entries must be Cochain objects")
        self._grades[cochain.grade] = cochain

    def __getitem__(self, grade: int) -> Cochain:
        return self._grades[int(grade)]

    def __contains__(self, grade: int) -> bool:
        return int(grade) in self._grades

    def __iter__(self) -> Iterator[Cochain]:
        for grade in sorted(self._grades):
            yield self._grades[grade]

    @property
    def grades(self) -> tuple[int, ...]:
        return tuple(sorted(self._grades))
