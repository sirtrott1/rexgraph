"""Full incidence neighborhoods, including primary zero columns and higher grades."""
import pytest

from rexgraph.cell_neighborhood import hyperslice
from rexgraph.cells import Cell
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graph import RexGraph


def test_primary_loops_witnesses_and_branches_are_not_collapsed():
    supports = [[0, 1, 2], [2, 2], [2], [0, 3], [0, 1]]
    rex = RexGraph.from_cells([4, supports])
    relation = hyperslice(Cell(rex, 1, 0))
    assert relation.below.indices == (0, 1, 2)
    assert relation.above.indices == () and relation.above.grade == 2
    assert relation.lateral.indices == (1, 2, 3, 4)
    vertex = hyperslice(Cell(rex, 0, 2))
    assert vertex.below is None
    assert vertex.above.indices == (0, 1, 2)
    assert vertex.lateral.indices == (0, 1)
    assert hyperslice(Cell(rex, 1, 1)).below.indices == (2,)
    assert hyperslice(Cell(rex, 1, 3)).below.indices == (0, 3)


def test_all_grades_and_explicit_empty_upper_are_preserved():
    rex = RexGraph.from_cells(solid_octahedron_3rex())
    for grade, count in ((0, rex.nV), (1, rex.nE), (2, rex.nF), (3, 1)):
        for index in range(count):
            value = hyperslice(Cell(rex, grade, index))
            assert value.cell.index == index and index not in value.lateral.indices
            if grade:
                for lower in value.below.indices:
                    assert index in hyperslice(Cell(rex, grade-1, lower)).above.indices
            if grade == 3:
                assert value.above.grade == 4 and value.above.indices == ()
            if grade == 2:
                assert value.above.indices == (0,)


@pytest.mark.parametrize("grade,index", [(0, 0), (1, 0), (2, 0)])
def test_matches_existing_compiled_convenience_view_where_defined(grade, index):
    rex = RexGraph.from_simplicial([0, 1, 2], [1, 2, 0], [[0, 1, 2]])
    result = hyperslice(Cell(rex, grade, index))
    old = rex.hyperslice(grade, index)
    expected = ((result.above.indices, result.lateral.indices) if grade == 0 else
                (result.below.indices, result.lateral.indices) if grade == 2 else
                (result.below.indices, result.above.indices, result.lateral.indices))
    assert expected == tuple(tuple(sorted(set(map(int, a)))) for a in old)


def test_higher_native_coefficients_and_absent_population():
    from rexgraph.native_sparse import native_coo
    rex = RexGraph.from_cells([1, [[0, 0]]])
    rex._graded_duals = [native_coo([0], [0], [2**100], (1, 1))]
    assert hyperslice(Cell(rex, 2, 0)).above.indices == (0,)
    assert hyperslice(Cell(rex, 3, 0)).below.indices == (0,)
    with pytest.raises(ValueError):
        hyperslice(Cell(rex, 4, 0))
