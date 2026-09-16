"""Exact relative chains C(X)/C(A) for a selected native subcomplex.

The surviving boundary coefficients keep their original shares. A relative
column need not satisfy the primary relation normalization, so this result is
a coordinate complex and a certified projection, not a reconstructed Rex.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .cells import Cell, CellSet, GradedCellPattern
from .chain_map import CoordinateComplex, GradedMap
from .io.partition_state import _downward_closure, partition_tower
from .type_accession import CoordinateSpace


@dataclass(frozen=True, eq=False)
class RelativeQuotient:
    """A full relative tower, original coordinate maps and its projection proof."""

    projection: object
    cell_maps: tuple
    removed_cells: tuple

    def check_state(self):
        self.projection.check_state()

    @property
    def complex(self):
        return self.projection.declaration.codomain

    @property
    def sizes(self):
        return self.complex.sizes

    @property
    def boundaries(self):
        return self.complex.boundaries

    @property
    def source_boundary_digest(self):
        return self.projection.declaration.domain.coefficient_digest

    @property
    def coefficient_digest(self):
        return self.complex.coefficient_digest

    @property
    def residuals(self):
        return self.projection.commutation_residuals

    @property
    def readings(self):
        self.check_state()
        betti, ranks, methods = self._homology
        return {"betti": betti, "ranks": ranks, "methods": methods}

    @cached_property
    def _homology(self):
        from .chain_map import _columns
        from .graded_boundary import _rank_integer_columns
        from .native_rank import clear_column_denominators
        ranks, methods = [], []
        for grade, entries in enumerate(self.boundaries, 1):
            columns = clear_column_denominators(_columns(entries, self.sizes[grade]))
            rank, method = _rank_integer_columns((self.sizes[grade-1], self.sizes[grade]), columns)
            ranks.append(rank)
            methods.append(method)
        padded = (0, *ranks, 0)
        betti = tuple(n - padded[k] - padded[k+1] for k, n in enumerate(self.sizes))
        return betti, tuple(ranks), tuple(methods)


def relative_quotient(selection):
    """Quotient by the downward closure of a cell selection across any grades.

    P_k deletes selected coordinates and Bbar_k retains the corresponding
    rows and columns of B_k. Core certifies both chain laws and P B = Bbar P.
    Primary slot closure retains loop vertices even when their B1 cancels.
    Construction is sparse; homology ranks are evaluated only on demand.
    """
    if not isinstance(selection, (Cell, CellSet, GradedCellPattern)):
        raise TypeError("relative quotient requires a Cell, CellSet or GradedCellPattern")
    source = selection.source
    sets = (selection.cell_sets if isinstance(selection, GradedCellPattern) else
            (CellSet(source, selection.grade, (selection.index,)),) if isinstance(selection, Cell) else (selection,))
    carriers, columns = partition_tower(source)
    sizes = (int(carriers[0].shape[0]), *(int(b.shape[1]) for b in carriers))
    requested = [np.zeros(n, dtype=bool) for n in sizes]
    for selected in sets:
        CellSet(source, selected.grade, selected.indices)
        if not 0 <= selected.grade < len(sizes):
            raise ValueError("relative quotient selection requires a carried grade")
        requested[selected.grade][list(selected.indices)] = True
    closed = _downward_closure(source, columns, requested)
    removed = tuple(tuple(map(int, np.flatnonzero(mask))) for mask in closed)
    kept = tuple(tuple(map(int, np.flatnonzero(~mask))) for mask in closed)
    inverse = tuple({old: new for new, old in enumerate(keys)} for keys in kept)
    domain = CoordinateComplex.from_rex(source)
    boundaries = tuple(tuple((inverse[k-1][i], inverse[k][j], value)
        for i, j, value in entries if i in inverse[k-1] and j in inverse[k])
        for k, entries in enumerate(domain.boundaries, 1))
    spaces = tuple(CoordinateSpace(f"C{k}/A", tuple(domain.spaces[k].keys[i] for i in keys))
                   for k, keys in enumerate(kept))
    target = CoordinateComplex(spaces, boundaries)
    maps = tuple(tuple((new, old, 1) for new, old in enumerate(keys)) for keys in kept)
    proof = GradedMap(domain, target, maps).verify()
    return RelativeQuotient(proof, kept, removed)
