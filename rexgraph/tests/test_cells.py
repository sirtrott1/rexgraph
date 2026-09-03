"""Direct graded-cell readings preserve the relational complex before algebra."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from rexgraph.cells import (
    Cell,
    CellBoundary,
    CellCoboundary,
    CellSet,
    CompositeBinary,
    GradedCellPattern,
    boundary_of,
    coboundary_of,
    composite_binary,
    corelations,
    star,
)
from rexgraph.cochain import Chain
from rexgraph.graph import RexGraph


def test_c1_composite_binary_retains_exact_head_share_boundary():
    rex = RexGraph.from_hypergraph(
        np.array([0, 3], dtype=np.int32), np.array([0, 1, 2], dtype=np.int32)
    )
    relation = Cell(rex, 1, 0)

    composite = composite_binary(relation)
    assert isinstance(composite, CompositeBinary)
    assert composite.arity == 3 and not composite.witness
    np.testing.assert_array_equal(composite.existence.values, [1, 1, 1])
    np.testing.assert_array_equal(composite.head.values, [1, 0, 0])
    np.testing.assert_array_equal(composite.share_support.values, [0, 1, 1])
    assert composite.boundary.values.tolist() == [
        Fraction(-1), Fraction(1, 2), Fraction(1, 2)
    ]

    direct = boundary_of(relation)
    assert isinstance(direct, CellBoundary)
    assert direct.cells.indices == (0, 1, 2)
    assert direct.chain.values.tolist() == composite.boundary.values.tolist()


def test_cell_corelations_keep_a_self_loop_that_cancels_in_b1():
    rex = RexGraph.from_hypergraph(
        np.array([0, 2], dtype=np.int32), np.array([0, 0], dtype=np.int32)
    )
    vertex, relation = Cell(rex, 0, 0), Cell(rex, 1, 0)

    direct = boundary_of(relation)
    assert direct.cells.indices == (0,)
    assert direct.chain.values.tolist() == [Fraction(0)]

    upward = coboundary_of(vertex)
    assert isinstance(upward, CellCoboundary)
    assert upward.cells.indices == (0,)
    assert upward.cochain.values.tolist() == [Fraction(0)]
    assert corelations(vertex).indices == (0,)
    composite = composite_binary(relation)
    assert composite.self_loop and not composite.witness
    np.testing.assert_array_equal(composite.existence.values, [1])
    np.testing.assert_array_equal(composite.head.values, [0])
    np.testing.assert_array_equal(composite.share_support.values, [0])
    assert composite.boundary.values.tolist() == [Fraction(0)]


def test_higher_grade_boundary_and_enclosure_follow_the_grading():
    rex = RexGraph.from_simplicial(
        np.array([0, 1, 0], dtype=np.int32),
        np.array([1, 2, 2], dtype=np.int32),
        np.array([[0, 1, 2]], dtype=np.int32),
    )
    relation, face = Cell(rex, 1, 0), Cell(rex, 2, 0)

    lower = boundary_of(face)
    assert lower.cells.grade == 1
    assert lower.cells.indices == (0, 1, 2)
    assert isinstance(lower.chain, Chain)

    upward = coboundary_of(relation)
    assert upward.cells.grade == 2
    assert upward.cells.indices == (0,)
    enclosure = star(relation)
    assert isinstance(enclosure, GradedCellPattern)
    assert enclosure.grades == (1, 2)
    assert enclosure.at(1).indices == (0,)
    assert enclosure.at(2).indices == (0,)

    # The top co-boundary has an empty codomain, not an invented fourth grade.
    assert corelations(face).grade == 3
    assert corelations(face).indices == ()


def test_cell_set_boundary_and_coboundary_preserve_exact_share_coefficients():
    rex = RexGraph.from_hypergraph(
        np.array([0, 3, 5], dtype=np.int32),
        np.array([0, 1, 2, 2, 3], dtype=np.int32),
    )
    boundary = boundary_of(CellSet(rex, 1, (0, 1)))
    assert boundary.values.tolist() == [
        Fraction(-1), Fraction(1, 2), Fraction(-1, 2), Fraction(1)
    ]

    coboundary = coboundary_of(CellSet(rex, 0, (0, 1)))
    assert coboundary.values.tolist() == [Fraction(-1, 2), Fraction(0)]
