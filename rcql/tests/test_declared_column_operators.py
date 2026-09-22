"""RCQL reads a DECLARED grade 1 column as declared.

COMPOSITE and the three masks it carries are the query layer's reading of the composite
binary: existence, the distinguished head, and the share. Each used to be derived from
the slot order and the arity, so a relation declaring its own head or share was reported
as the canonical one. They are read off the column now, and these are the contracts.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from rexgraph.graph import RexGraph

from rcql import Executor, call, query, source

DECLARED = [(0, -1), (1, Fraction(1, 4)), (2, Fraction(1, 2)), (3, Fraction(1, 4))]


@pytest.fixture
def complexes():
    """One canonical relation and one declaring shares, over the same four objects."""
    return RexGraph.from_cells([4, [[0, 1, 2, 3], DECLARED]])


def read(rex, operator, relation):
    return Executor(sources={"complex": rex}).execute(
        query(source("complex"), call(operator, call("COMPOSITE", call("CELL", 1, relation))))
    ).values[0]


def test_share_returns_the_declared_vector_and_the_equal_share_elsewhere(complexes):
    assert [str(v) for v in read(complexes, "SHARE", 0).values] == ["0", "1/3", "1/3", "1/3"]
    assert [str(v) for v in read(complexes, "SHARE", 1).values] == ["0", "1/4", "1/2", "1/4"]


def test_head_is_the_participant_carrying_the_minus_one(complexes):
    for relation in (0, 1):
        np.testing.assert_array_equal(read(complexes, "HEAD", relation).values, [1, 0, 0, 0])
    # a declared head off slot zero is reported where it was declared
    moved = RexGraph.from_cells([4, [[(2, -1), (0, Fraction(1, 2)), (1, Fraction(1, 4)),
                                      (3, Fraction(1, 4))]]])
    np.testing.assert_array_equal(read(moved, "HEAD", 0).values, [0, 0, 1, 0])
    np.testing.assert_array_equal(read(moved, "SHARE_SUPPORT", 0).values, [1, 1, 0, 1])


def test_existence_and_arity_are_unchanged_by_a_declaration(complexes):
    for relation in (0, 1):
        np.testing.assert_array_equal(read(complexes, "EXISTENCE", relation).values, [1, 1, 1, 1])
        assert read(complexes, "ARITY", relation) == 4


def test_the_composite_carries_the_declared_column_and_its_integer_representative(complexes):
    from rcql.operators import composite
    from rexgraph.cells import Cell

    canonical = composite(complexes, Cell(complexes, 1, 0))
    declared = composite(complexes, Cell(complexes, 1, 1))
    assert canonical.boundary.values.tolist() == [Fraction(-1), Fraction(1, 3),
                                                  Fraction(1, 3), Fraction(1, 3)]
    assert declared.boundary.values.tolist() == [Fraction(-1), Fraction(1, 4),
                                                 Fraction(1, 2), Fraction(1, 4)]
    # the integer representative clears the column's OWN denominator, 3 against 4 here
    assert canonical.integer_boundary.values.tolist() == [-3, 1, 1, 1]
    assert declared.integer_boundary.values.tolist() == [-4, 1, 2, 1]


def test_the_character_reads_the_declared_column(complexes):
    """CHARACTER goes through the compiled channel tower, which takes the coefficients:
    T is the quadrance 1 + sum s_i^2, so the two relations read differently."""
    chi = Executor(sources={"complex": complexes}).execute(
        query(source("complex"), call("CHARACTER"))
    ).values[0]
    values = np.asarray(chi.values if hasattr(chi, "values") else chi, dtype=float)
    assert values.shape[0] == 2
    assert not np.allclose(values[0], values[1])
