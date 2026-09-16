"""Local graded incidence views without an adjacency matrix or coefficient field."""
from dataclasses import dataclass

from rexgraph.cells import Cell, CellSet
from rexgraph.native_sparse import NativeSparse, boundary_carriers, sparse_arrays

__all__ = ["Hyperslice", "hyperslice"]


def _row(matrix, index):
    ptr, columns, values, _ = sparse_arrays(matrix)
    result = {}
    for pos in range(int(ptr[index]), int(ptr[index+1])):
        j = int(columns[pos])
        result[j] = result.get(j, 0) + values[pos]
    return {j for j, value in result.items() if value}


def _column(matrix, index):
    if isinstance(matrix, NativeSparse):
        dual = matrix.dual
        result = {}
        for pos in range(int(dual.col_ptr[index]), int(dual.col_ptr[index+1])):
            i = int(dual.row_idx[pos])
            result[i] = result.get(i, 0) + dual.vals_csc[pos]
        return {i for i, value in result.items() if value}
    # Original higher grade CSR may contain integers beyond floating range.
    # Read its arrays, without numerical conversion or an optional sparse import.
    return {i for i in range(matrix.shape[0]) if index in _row(matrix, i)}


@dataclass(frozen=True, eq=False)
class Hyperslice:
    """Immediate lower, upper and lateral CellSets around one primary cell.

Lateral means sharing a lower participant, except at C0 where it means sharing
an upper relation. The center is excluded. There is no lower grade below C0;
above the top is an explicitly empty CellSet. Types and time remain on the
source binding, not an inferred projection of this neighborhood.
"""

    cell: Cell
    below: CellSet | None
    above: CellSet
    lateral: CellSet

    @property
    def source(self):
        return self.cell.source

    @property
    def grade(self):
        return self.cell.grade


def hyperslice(value):
    if not isinstance(value, Cell):
        raise TypeError("hyperslice requires one primary Cell")
    rex, grade, index = value.source, value.grade, value.index
    # Revalidate a saved cell's population before indexing native buffers.
    Cell(rex, grade, index)
    maps = boundary_carriers(rex)

    def below(g, i):
        if g == 1:
            ptr, indices = rex._boundary_ptr, rex._boundary_idx
            return set(map(int, indices[int(ptr[i]):int(ptr[i+1])]))
        return _column(maps[g-1], i)

    def above(g, i):
        if g == len(maps):
            return set()
        if g == 0:
            ptr, indices = rex._v2e
            return set(map(int, indices[int(ptr[i]):int(ptr[i+1])]))
        return _row(maps[g], i)

    lower = below(grade, index) if grade else None
    upper = above(grade, index)
    lateral = set()
    if grade:
        for i in lower:
            lateral.update(above(grade-1, i))
    else:
        for i in upper:
            lateral.update(below(1, i))
    lateral.discard(index)
    return Hyperslice(value, None if lower is None else CellSet(rex, grade-1, tuple(lower)),
                      CellSet(rex, grade+1, tuple(upper)), CellSet(rex, grade, tuple(lateral)))
