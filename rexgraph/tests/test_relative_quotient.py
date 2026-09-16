"""Exact relative towers against independent small rational matrix oracles."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest

from rexgraph.cells import Cell, CellSet, GradedCellPattern
from rexgraph.chain_map import CoordinateComplex, GradedMap
from rexgraph.type_accession import CoordinateSpace
from rexgraph.graph import RexGraph
from rexgraph.relative_quotient import relative_quotient


def simplex():
    simplices = [list(combinations(range(5), k+1)) for k in range(5)]
    cells = [5, simplices[1]]
    for k in range(2, 5):
        index = {s: i for i, s in enumerate(simplices[k-1])}
        cells.append([[(index[s[:i]+s[i+1:]], (-1)**i) for i in range(len(s))] for s in simplices[k]])
    return RexGraph.from_cells(cells)


def dense(entries, rows, cols):
    result = np.full((rows, cols), Q(0), object)
    for i, j, v in entries:
        result[i, j] += v
    return result


def rank(a):
    a = a.copy()
    pivot = 0
    for j in range(a.shape[1]):
        row = next((i for i in range(pivot, a.shape[0]) if a[i, j]), None)
        if row is None:
            continue
        a[[row, pivot]] = a[[pivot, row]]
        a[pivot] /= a[pivot, j]
        for i in range(pivot+1, a.shape[0]):
            a[i] -= a[i, j] * a[pivot]
        pivot += 1
    return pivot


@pytest.mark.parametrize("grade", range(5))
@pytest.mark.parametrize("selection", ["empty", "one", "all"])
def test_full_tower_matches_independent_relative_matrices(grade, selection):
    source = simplex()
    original = CoordinateComplex.from_rex(source)
    indices = () if selection == "empty" else (0,) if selection == "one" else range(original.sizes[grade])
    value = relative_quotient(CellSet(source, grade, indices))
    assert len(value.sizes) == 5 and len(value.boundaries) == 4
    matrices, ranks = [], [0]
    for k, entries in enumerate(original.boundaries, 1):
        b = dense(entries, original.sizes[k-1], original.sizes[k])
        expected = b[np.ix_(value.cell_maps[k-1], value.cell_maps[k])]
        actual = dense(value.boundaries[k-1], value.sizes[k-1], value.sizes[k])
        assert np.array_equal(actual, expected)
        p = dense(value.projection.declaration.components[k], value.sizes[k], original.sizes[k])
        below = dense(value.projection.declaration.components[k-1], value.sizes[k-1], original.sizes[k-1])
        assert np.array_equal(actual @ p, below @ b)
        if matrices:
            assert not np.any(matrices[-1] @ actual)
        matrices.append(actual)
        ranks.append(rank(actual))
    ranks.append(0)
    expected_betti = tuple(n-ranks[k]-ranks[k+1] for k, n in enumerate(value.sizes))
    assert value.readings["betti"] == expected_betti
    assert value.residuals == (Q(0),)*4
    if grade == 3 and selection == "all":
        assert expected_betti == (0, 0, 0, 0, 1)
    if grade == 4 and selection == "all":
        assert value.sizes == (0,)*5


@pytest.mark.parametrize("removed", [0, 1, 2, 3])
def test_partial_branching_coefficients_are_not_renormalized(removed):
    source = RexGraph.from_cells([4, [[0, 1, 2, 3]]], w_E=np.array([Q(2**100+1, 7)], object))
    value = relative_quotient(Cell(source, 0, removed))
    original = {0: Q(-1), 1: Q(1, 3), 2: Q(1, 3), 3: Q(1, 3)}
    assert value.boundaries[0] == tuple((i, 0, original[j]) for i, j in enumerate(value.cell_maps[0]))
    assert value.sizes == (3, 1) and value.readings["betti"] == (2, 0)
    assert source.nV == 4 and source.nE == 1


def test_loop_witness_parallel_and_isolated_cells_remain_distinct():
    source = RexGraph.from_cells([5, [[0, 0], [1], [1, 2, 3], [1, 2, 3]]], relation_ids=[11, 12, 13, 14])
    loop = relative_quotient(Cell(source, 1, 0))
    assert loop.removed_cells == ((0,), (0,))
    assert loop.cell_maps == ((1, 2, 3, 4), (1, 2, 3))
    witness = relative_quotient(Cell(source, 1, 1))
    assert witness.removed_cells == ((1,), (1,))
    assert witness.cell_maps[1] == (0, 2, 3)
    assert witness.readings["betti"] == (3, 2)
    assert source.relation_ids.tolist() == [11, 12, 13, 14]
    # Retaining a basepoint is a different construction, not the relative tower.
    single = RexGraph.from_cells([1, [[0]]])
    domain = CoordinateComplex.from_rex(single)
    collapsed = CoordinateComplex((CoordinateSpace("point", ("p",)), CoordinateSpace("C1", ())), ((),))
    with pytest.raises(ValueError, match="square failed at grade 1"):
        GradedMap(domain, collapsed, (((0, 0, 1),), ())).verify()
    relative = relative_quotient(Cell(single, 1, 0))
    assert relative.sizes == (0, 0) and relative.residuals == (Q(0),)


def test_rank_is_lazy_and_cached_without_exposing_mutable_cache(monkeypatch):
    import rexgraph.graded_boundary as core
    calls = []
    original = core._rank_integer_columns
    def measured(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(core, "_rank_integer_columns", measured)
    value = relative_quotient(Cell(simplex(), 0, 0))
    assert calls == []
    first = value.readings
    assert len(calls) == 4
    first["betti"] = "changed"
    assert value.readings["betti"] == (0, 0, 0, 0, 0)
    assert len(calls) == 4


def test_changed_boundary_refuses_cached_homology_and_projection():
    source = RexGraph.from_cells([3, [[0, 1], [1, 2]]])
    value = relative_quotient(Cell(source, 0, 0))
    _ = value.readings
    source.add_edges([0], [2])
    with pytest.raises(ValueError, match="changed"):
        _ = value.readings
    with pytest.raises(ValueError, match="changed"):
        value.projection.check_state()


def test_noncomplex_is_refused_not_filtered():
    source = RexGraph(sources=np.array([0]), targets=np.array([1]), B2_col_ptr=np.array([0, 1], np.int32),
                      B2_row_idx=np.array([0], np.int32), B2_vals=np.array([1.0]))
    with pytest.raises(ValueError, match="chain condition"):
        relative_quotient(Cell(source, 0, 0))


def test_mixed_grade_selection_and_absent_grade_contract():
    source = RexGraph.from_cells([4, [[0, 1], [1, 2]]])
    selection = GradedCellPattern(source, (CellSet(source, 0, (3,)), CellSet(source, 1, (0,))))
    value = relative_quotient(selection)
    assert value.removed_cells == ((0, 1, 3), (0,))
    assert value.sizes == (1, 1) and value.boundaries == (((0, 0, Q(1)),),)
    with pytest.raises(ValueError, match="carried grade"):
        relative_quotient(CellSet(source, 2, ()))


def test_sparse_construction_does_not_allocate_square_arrays(monkeypatch):
    source = RexGraph.from_cells([1001, [[i, i+1] for i in range(1000)]])
    original = np.zeros
    def checked(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) > 1 and min(shape) > 1:
            pytest.fail("dense allocation")
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(np, "zeros", checked)
    value = relative_quotient(CellSet(source, 1, range(500)))
    assert value.sizes == (500, 500)
    assert sum(len(entries) for entries in value.boundaries) == 999
