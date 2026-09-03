"""Primary graded-cell values and exact local readings of a relational complex.

Relations are cells, not rows in a derived table.  These carriers retain the
source Rex, grade, and basis position needed to ask direct questions about one
cell, its boundary, its co-relations, or its upward enclosure.  Chains and
cochains remain coefficient spaces *over* these cells; they do not replace the
primary cell structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_boundary import graded_boundaries_from_rex
from rexgraph.linear_operator import boundary_operator, coboundary_operator

__all__ = [
    "Cell",
    "CellSet",
    "GradedCellPattern",
    "CompositeBinary",
    "CellBoundary",
    "CellCoboundary",
    "cell",
    "cells",
    "cell_count",
    "composite_binary",
    "boundary_of",
    "coboundary_of",
    "corelations",
    "star",
    "enclosure",
]


def _grade_sizes(rex) -> tuple[int, ...]:
    rex._ensure_clean()
    boundaries = graded_boundaries_from_rex(rex)
    if not boundaries:
        return (0,)
    return tuple([int(boundaries[0].shape[0])] + [int(B.shape[1]) for B in boundaries])


def cell_count(rex, grade: int, *, allow_empty_upper: bool = False) -> int:
    """Return the carried cell count at one grade.

    The one permitted non-carried space is the empty grade immediately above the
    top.  It names the codomain of the top co-boundary, never an invented cell
    population.
    """
    grade = int(grade)
    sizes = _grade_sizes(rex)
    if 0 <= grade < len(sizes):
        return sizes[grade]
    if allow_empty_upper and grade == len(sizes):
        return 0
    raise ValueError(f"grade {grade} is not present")


@dataclass(frozen=True, eq=False)
class Cell:
    """One carried cell in the ordered basis of a source relational complex."""

    source: Any
    grade: int
    index: int

    def __post_init__(self) -> None:
        if isinstance(self.grade, (bool, np.bool_)) or isinstance(self.index, (bool, np.bool_)):
            raise TypeError("cell grade and index must be integers, not booleans")
        grade, index = int(self.grade), int(self.index)
        count = cell_count(self.source, grade)
        if index < 0 or index >= count:
            raise ValueError(f"cell index {index} is not present at grade {grade}")
        object.__setattr__(self, "grade", grade)
        object.__setattr__(self, "index", index)


@dataclass(frozen=True, eq=False)
class CellSet:
    """An ordered-basis selection at one grade of one source Rex."""

    source: Any
    grade: int
    indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        grade = int(self.grade)
        raw = tuple(self.indices)
        if any(isinstance(index, (bool, np.bool_)) for index in raw):
            raise TypeError("cell indices must be integers, not booleans")
        indices = tuple(sorted({int(index) for index in raw}))
        count = cell_count(self.source, grade, allow_empty_upper=not indices)
        if any(index < 0 or index >= count for index in indices):
            raise ValueError(f"cell set contains an index absent at grade {grade}")
        object.__setattr__(self, "grade", grade)
        object.__setattr__(self, "indices", indices)

    @property
    def cells(self) -> tuple[Cell, ...]:
        """Materialize the selected primary cells in canonical basis order."""
        return tuple(Cell(self.source, self.grade, index) for index in self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        return iter(self.cells)


@dataclass(frozen=True, eq=False)
class GradedCellPattern:
    """A source-bound pattern of selected cells across one or more grades."""

    source: Any
    cell_sets: tuple[CellSet, ...]

    def __post_init__(self) -> None:
        ordered: dict[int, CellSet] = {}
        for cell_set in self.cell_sets:
            if not isinstance(cell_set, CellSet):
                raise TypeError("graded patterns contain CellSet values")
            if cell_set.source is not self.source:
                raise ValueError("all pattern cells must be bound to one source Rex")
            ordered[cell_set.grade] = cell_set
        object.__setattr__(self, "cell_sets", tuple(ordered[g] for g in sorted(ordered)))

    @property
    def grades(self) -> tuple[int, ...]:
        return tuple(cell_set.grade for cell_set in self.cell_sets)

    def at(self, grade: int) -> CellSet:
        grade = int(grade)
        for cell_set in self.cell_sets:
            if cell_set.grade == grade:
                return cell_set
        return CellSet(self.source, grade, ())


@dataclass(frozen=True, eq=False)
class CompositeBinary:
    """The existence/head/share binary decomposition of one primary C1 relation.

    ``existence``, ``head`` and ``share_support`` are 0/1 chains over the C0
    basis.  ``share`` is the exact rational normalization of share support.
    ``boundary`` is ``share - head`` and ``integer_boundary`` is its projective
    integer representative for arity at least two.  A witness has its own
    positive C0 boundary and no head/share split.  A deliberate ``[v, v]``
    self-loop has existence at ``v`` and a zero C0 boundary; it therefore has
    no head/share split either, while remaining a carried primary relation.
    """

    cell: Cell
    existence: Chain
    head: Chain
    share_support: Chain
    share: Chain
    boundary: Chain
    integer_boundary: Chain
    arity: int
    witness: bool
    self_loop: bool

    @property
    def source(self):
        """The source relational complex that declares this C1 relation."""
        return self.cell.source

    @property
    def grade(self) -> int:
        """The primary relation grade, distinct from its C0 tensor basis."""
        return self.cell.grade


@dataclass(frozen=True, eq=False)
class CellBoundary:
    """A cell's direct lower-grade participants and its coefficient chain."""

    cell: Cell
    cells: CellSet | None
    chain: Chain | None
    composite: CompositeBinary | None = None

    @property
    def source(self):
        """The source relational complex of the boundary reading."""
        return self.cell.source

    @property
    def grade(self) -> int | None:
        """The lower coefficient grade, or None when a C0 cell has no boundary."""
        return None if self.chain is None else self.chain.grade


@dataclass(frozen=True, eq=False)
class CellCoboundary:
    """A cell's direct co-relations and its cochain coefficient reading."""

    cell: Cell
    cells: CellSet
    cochain: Cochain

    @property
    def source(self):
        """The source relational complex of the co-boundary reading."""
        return self.cell.source

    @property
    def grade(self) -> int:
        """The cochain grade, including the empty virtual top sector."""
        return self.cochain.grade


def cell(source, grade: int, index: int) -> Cell:
    """Address one primary cell by grade and ordered-basis index."""
    return Cell(source, grade, index)


def cells(source, grade: int, indices=None) -> CellSet:
    """Address all or a selected set of cells at one carried grade."""
    if indices is None:
        indices = range(cell_count(source, grade))
    return CellSet(source, grade, tuple(indices))


def _exact_c1_boundary(source, index: int) -> tuple[list[int], np.ndarray]:
    """Reconstruct one C1 boundary from declared relation incidence, not floats."""
    support = source.relation_supports()[int(index)]
    values = [Fraction(0) for _ in range(cell_count(source, 0))]
    arity = len(support)
    if arity == 1:
        values[support[0]] += Fraction(1)
    elif arity >= 2:
        values[support[0]] -= Fraction(1)
        share = Fraction(1, arity - 1)
        for vertex in support[1:]:
            values[vertex] += share
    return support, np.asarray(values, dtype=object)


def composite_binary(value: Cell) -> CompositeBinary:
    """Read the exact 0/1 existence, head, and share-support tensors of one C1 cell.

    The only repeated incidence admitted by the primary carrier is an exact
    ``[v, v]`` self-loop.  It has a first-class zero-boundary composite instead
    of being mistaken for an arbitrary repeated branching relation.
    """
    if not isinstance(value, Cell):
        raise TypeError("COMPOSITE expects a Cell")
    if value.grade != 1:
        raise ValueError("COMPOSITE currently applies to grade-1 relation cells")
    support, boundary = _exact_c1_boundary(value.source, value.index)
    self_loop = len(support) == 2 and support[0] == support[1]
    if not self_loop and len(set(support)) != len(support):
        raise ValueError(
            "COMPOSITE refuses repeated C1 incidence: vertex-basis binary masks "
            "would collapse occurrences"
        )
    n_vertices = cell_count(value.source, 0)
    existence = np.zeros(n_vertices, dtype=np.uint8)
    head = np.zeros(n_vertices, dtype=np.uint8)
    share_support = np.zeros(n_vertices, dtype=np.uint8)
    share = [Fraction(0) for _ in range(n_vertices)]
    existence[support] = 1
    witness = len(support) == 1
    if not witness and not self_loop and support:
        head[support[0]] = 1
        share_support[support[1:]] = 1
        share_value = Fraction(1, len(support) - 1)
        for vertex in support[1:]:
            share[vertex] = share_value
    integer_values = boundary
    if not witness:
        integer_values = [Fraction(len(support) - 1) * coefficient for coefficient in boundary]
    # This is an exact integral representative, not a float-rounded rendering.
    integer_boundary = np.asarray(
        [int(coefficient) for coefficient in integer_values], dtype=np.int64
    )
    return CompositeBinary(
        cell=value,
        existence=Chain(0, existence, source=value.source),
        head=Chain(0, head, source=value.source),
        share_support=Chain(0, share_support, source=value.source),
        share=Chain(0, np.asarray(share, dtype=object), source=value.source),
        boundary=Chain(0, boundary, source=value.source),
        integer_boundary=Chain(0, integer_boundary, source=value.source),
        arity=len(support),
        witness=witness,
        self_loop=self_loop,
    )


def _indicator_chain(value: CellSet) -> Chain:
    vector = np.zeros(cell_count(value.source, value.grade), dtype=np.int64)
    vector[list(value.indices)] = 1
    return Chain(value.grade, vector, source=value.source)


def _indicator_cochain(value: CellSet) -> Cochain:
    vector = np.zeros(cell_count(value.source, value.grade), dtype=np.int64)
    vector[list(value.indices)] = 1
    return Cochain(value.grade, vector, source=value.source)


def boundary_of(value: Cell | CellSet) -> CellBoundary | Chain:
    """Read the direct boundary of a primary cell or aggregate cell pattern."""
    if isinstance(value, CellSet):
        if value.grade == 0:
            raise ValueError("C0 cells have no lower chain grade")
        if value.grade == 1:
            # B1 stores shares as floating point for sparse numerical work.  A
            # direct relational complex reading instead retains the declared
            # rational relation boundaries, including cancellation from repeated
            # incidence, before any numerical operator is applied.
            coefficients = [Fraction(0) for _ in range(cell_count(value.source, 0))]
            for index in value.indices:
                _support, part = _exact_c1_boundary(value.source, index)
                for vertex, coefficient in enumerate(part):
                    coefficients[vertex] += coefficient
            return Chain(0, np.asarray(coefficients, dtype=object), source=value.source)
        operator = boundary_operator(value.source, value.grade)
        return Chain(value.grade - 1, operator.apply(_indicator_chain(value).values),
                     source=value.source)
    if not isinstance(value, Cell):
        raise TypeError("BOUNDARY expects a Cell, CellSet, or typed Chain")
    if value.grade == 0:
        return CellBoundary(value, None, None)
    if value.grade == 1:
        support, coefficients = _exact_c1_boundary(value.source, value.index)
        participants = CellSet(value.source, 0, tuple(set(support)))
        composite = None
        if (len(set(support)) == len(support)
                or (len(support) == 2 and support[0] == support[1])):
            composite = composite_binary(value)
        return CellBoundary(
            value,
            participants,
            Chain(0, coefficients, source=value.source),
            composite,
        )
    operator = boundary_operator(value.source, value.grade)
    unit = np.zeros(cell_count(value.source, value.grade), dtype=np.float64)
    unit[value.index] = 1.0
    coefficients = operator.apply(unit)
    participants = CellSet(value.source, value.grade - 1, tuple(np.flatnonzero(coefficients)))
    return CellBoundary(
        value,
        participants,
        Chain(value.grade - 1, coefficients, source=value.source),
    )


def coboundary_of(value: Cell | CellSet) -> CellCoboundary | Cochain:
    """Read direct co-relations/cofaces without projecting the complex to a graph."""
    if isinstance(value, CellSet):
        if value.grade == 0:
            # The dual numerical action B1^T has the same coefficients, but its
            # float representation would discard the exact share denominator.
            coefficients = [Fraction(0) for _ in range(cell_count(value.source, 1))]
            selected = set(value.indices)
            for index in range(cell_count(value.source, 1)):
                _support, part = _exact_c1_boundary(value.source, index)
                coefficients[index] = sum(
                    (part[vertex] for vertex in selected), Fraction(0)
                )
            return Cochain(1, np.asarray(coefficients, dtype=object), source=value.source)
        operator = coboundary_operator(value.source, value.grade)
        return Cochain(value.grade + 1, operator.apply(_indicator_cochain(value).values),
                       source=value.source)
    if not isinstance(value, Cell):
        raise TypeError("COBOUNDARY expects a Cell, CellSet, or typed Cochain")
    if value.grade == 0:
        supports = value.source.relation_supports()
        related = tuple(index for index, support in enumerate(supports) if value.index in support)
        coefficients = np.asarray(
            [_exact_c1_boundary(value.source, index)[1][value.index]
             for index in range(cell_count(value.source, 1))],
            dtype=object,
        )
        return CellCoboundary(
            value,
            CellSet(value.source, 1, related),
            Cochain(1, coefficients, source=value.source),
        )
    operator = coboundary_operator(value.source, value.grade)
    unit = np.zeros(cell_count(value.source, value.grade), dtype=np.float64)
    unit[value.index] = 1.0
    coefficients = operator.apply(unit)
    cofaces = CellSet(value.source, value.grade + 1, tuple(np.flatnonzero(coefficients)))
    return CellCoboundary(
        value,
        cofaces,
        Cochain(value.grade + 1, coefficients, source=value.source),
    )


def corelations(value: Cell | CellSet) -> CellSet:
    """Return direct higher-grade co-relations of one cell or cell pattern."""
    if isinstance(value, Cell):
        return coboundary_of(value).cells
    if not isinstance(value, CellSet):
        raise TypeError("CORELATIONS expects a Cell or CellSet")
    top = len(_grade_sizes(value.source)) - 1
    if value.grade >= top:
        return CellSet(value.source, value.grade + 1, ())
    related: set[int] = set()
    for item in value:
        related.update(corelations(item).indices)
    return CellSet(value.source, value.grade + 1, tuple(related))


def star(value: Cell | CellSet) -> GradedCellPattern:
    """Return the upward graded enclosure generated by a cell or cell pattern."""
    if isinstance(value, Cell):
        current = CellSet(value.source, value.grade, (value.index,))
    elif isinstance(value, CellSet):
        current = value
    else:
        raise TypeError("STAR expects a Cell or CellSet")
    sets = [current]
    while current.indices and current.grade < len(_grade_sizes(current.source)) - 1:
        current = corelations(current)
        if current.indices:
            sets.append(current)
    return GradedCellPattern(current.source if sets else value.source, tuple(sets))


def enclosure(value: Cell | CellSet) -> GradedCellPattern:
    """Alias for the full upward cell enclosure (the graded star)."""
    return star(value)
