"""Exact missing triangular faces on an explicitly selected pairwise C1 region."""
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property

import numpy as np

from .cells import CellSet
from .io.partition_state import partition_tower


def validate_region(region):
    if not isinstance(region, CellSet) or region.grade != 1:
        raise TypeError("VOID requires a C1 CellSet region")
    source = region.source
    carriers, columns = partition_tower(source)
    CellSet(source, 1, region.indices)
    if len(region.indices) > np.iinfo(np.int32).max // 2:
        raise ValueError("triangle enumeration exceeds the compiled int32 incidence axis")
    for i in region.indices:
        column = columns[0][i]
        if len(column) != 2 or sorted(column.values()) != [Fraction(-1), Fraction(1)]:
            raise ValueError("VOID currently requires distinct pairwise participants in its region; select pairwise C1 cells explicitly")
    return carriers, columns


@dataclass(frozen=True, eq=False)
class VoidState:
    """Sparse missing face boundaries, with lazy exact joint filling rank."""

    source: object = field(repr=False)
    source_state: str
    region: tuple
    potential: tuple
    void_indices: tuple
    columns: tuple
    _n_cells: int = field(repr=False)

    def check_state(self):
        from .io.catalog import object_digest
        if object_digest(self.source) != self.source_state:
            raise ValueError("void source changed; bind a fresh region")

    @property
    def shape(self):
        return (self._n_cells, len(self.columns))

    @property
    def n_voids(self):
        return len(self.columns)

    @property
    def n_potential(self):
        return len(self.potential)

    @property
    def strain(self):
        return sum(value*value for column in self.columns for _, value in column)

    @property
    def homology(self):
        self.check_state()
        return dict(self._homology)

    @cached_property
    def _homology(self):
        from .graded_boundary import _rank_integer_columns
        _, tower = partition_tower(self.source)
        faces = tower[1] if len(tower) > 1 else []
        base, first = _rank_integer_columns((self.shape[0], len(faces)), faces)
        combined = [*faces, *(dict(column) for column in self.columns)]
        total, second = _rank_integer_columns((self.shape[0], len(combined)), combined)
        return {"rank_before": base, "rank_after": total, "independent_fillings": total-base,
                "methods": (first, second)}


def void_state(region):
    """Read missing triangular C2 supports; do not attach cells or choose a metric.

    The selected region is pairwise, but the rest of the source may carry
    arbitrary arity and grades. Parallel cells remain distinct candidates.
    Homology uses every original C2 column, not just faces in the region.
    """
    from .core import _cycles, _void
    from .faces import solve_face_column
    from .io.catalog import object_digest
    carriers, tower = validate_region(region)
    vertices, sources, targets = {}, [], []
    for i in region.indices:
        negative, positive = sorted(tower[0][i], key=tower[0][i].get)
        sources.append(vertices.setdefault(negative, len(vertices)))
        targets.append(vertices.setdefault(positive, len(vertices)))
    adjacency = _cycles.build_symmetric_adjacency(len(vertices), len(region.indices),
        np.array(sources, np.int32), np.array(targets, np.int32))
    triangles, count = _void.find_potential_triangles(*adjacency, len(vertices), len(region.indices))
    potential = tuple(tuple(region.indices[int(i)] for i in triple) for triple in triangles)
    raw_faces = carriers[1] if len(carriers) > 1 else None
    _, missing, _ = _void.classify_triangles(raw_faces, np.array(potential, np.int32).reshape(-1, 3), count, region.source.nE)
    columns = []
    for j in missing:
        support = potential[int(j)]
        coefficients = solve_face_column(region.source, support)
        if coefficients is None:
            raise ValueError("compiled triangle candidate does not bound an exact cycle")
        columns.append(tuple((i, int(v)) for i, v in zip(support, coefficients, strict=True)))
    return VoidState(region.source, object_digest(region.source), tuple(region.indices), potential,
                     tuple(map(int, missing)), tuple(columns), int(region.source.nE))
