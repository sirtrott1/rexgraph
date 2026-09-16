"""Exact factorization and metric pullback against small independent oracles."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.column_expansion import ColumnExpansion, primary_lift
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import boundary_operator, hodge_operator
from rexgraph.native_rank import boundary_columns


def dense(entries, shape):
    values = np.full(shape, Q(0), object)
    for i, j, value in entries:
        values[i, j] += value
    return values


def source(supports):
    return RexGraph.from_cells([4, supports])


@pytest.mark.parametrize("supports", [[], [[0]], [[0, 0]], [[0, 1]], [[0, 1, 2, 3]],
    [[0, 0], [1, 1]], [[0, 1], [0, 1], [1, 0]], [[0], [1], [0, 1, 2], [2, 2]]])
def test_exact_and_numeric_actions_preserve_primary_columns_and_transposes(supports):
    rex = source(supports)
    boundary = boundary_operator(rex, 1)
    expansion = ColumnExpansion(boundary)
    result = primary_lift(expansion.legs, expansion.lift)
    shape, columns = boundary_columns(rex, 1)
    expected = dense(((i, j, c) for j, col in enumerate(columns) for i, c in col.items()), shape)
    p = dense(expansion.legs.entries, expansion.legs.shape)
    lift = dense(expansion.lift.entries, expansion.lift.shape)
    np.testing.assert_array_equal(p @ lift, expected)
    x = np.array([[Q(j+1, 7), 2**100+j] for j in range(shape[1])], object).reshape(shape[1], 2)
    y = np.array([Q(i+1, 5) for i in range(shape[0])], object)
    np.testing.assert_array_equal(result.apply(x, exact=True), expected @ x)
    np.testing.assert_array_equal(result.transpose_apply(y, exact=True), expected.T @ y)
    small = np.asarray([[Q(j+1, 7), j+1] for j in range(shape[1])], object).reshape(shape[1], 2)
    np.testing.assert_allclose(result.apply(small), np.asarray(expected @ small, float), atol=1e-14)
    np.testing.assert_allclose(result.transpose_apply(y), np.asarray(expected.T @ y, float), atol=1e-14)
    # Independent arbitrary boundary form, not only a diagonal metric.
    m = np.eye(shape[0], dtype=object) + np.ones((shape[0], shape[0]), object)
    np.testing.assert_array_equal(expected.T @ m @ expected, lift.T @ p.T @ m @ p @ lift)
    assert result.shape == boundary.shape and expansion.lift.shape[1] == len(supports)
    assert len(expansion.groups) == len(supports)


def test_witness_loop_and_branch_have_distinct_primary_readings():
    rex = source([[0], [0, 0], [0, 1, 2, 3]])
    expansion = ColumnExpansion(boundary_operator(rex, 1))
    assert expansion.references == (0, 0, 0)
    assert expansion.groups == ((0, 1), (1, 1), (1, 4))
    assert expansion.lift.entries == ((0, 0, Q(1)), (1, 2, Q(1, 3)), (2, 2, Q(1, 3)), (3, 2, Q(1, 3)))
    assert rex.nE == 3 and expansion.legs.shape[1] == 4


@pytest.mark.parametrize("grade", [1, 2, 3])
def test_every_carried_grade_reconstructs_and_keeps_chain_composition(grade):
    rex = RexGraph.from_cells(solid_octahedron_3rex())
    b = boundary_operator(rex, grade)
    expansion = ColumnExpansion(b)
    result = primary_lift(expansion.legs, expansion.lift)
    x = np.asarray([Q(i+1, 11) for i in range(b.shape[1])], object)
    np.testing.assert_array_equal(result.apply(x, exact=True), b.apply(x, exact=True))
    if grade > 1:
        residual = boundary_operator(rex, grade-1).apply(result.apply(x, exact=True), exact=True)
        assert all(v == 0 for v in residual)


def test_unbalanced_higher_column_has_a_witness_and_huge_integer_stays_exact():
    from rexgraph.native_sparse import native_coo
    rex = RexGraph.from_cells([1, [[0, 0]]])
    # Explicit zero B2 (1,1), then B3=[2**100]. Chain law is still exact.
    rex._graded_duals = [native_coo([0], [0], [2**100], (1, 1))]
    expansion = ColumnExpansion(boundary_operator(rex, 3))
    assert expansion.legs.entries == ((0, 0, Q(1)),)
    assert expansion.lift.entries == ((0, 0, Q(2**100)),)
    assert primary_lift(expansion.legs, expansion.lift).apply(np.array([Q(1, 3)], object), exact=True)[0] == Q(2**100, 3)


def test_mixed_expansions_and_modified_factors_are_refused():
    rex = RexGraph.from_cells([3, [[0, 1, 2]]])
    a, b = (ColumnExpansion(boundary_operator(rex, 1)) for _ in range(2))
    for legs, lift in ((a.legs, b.lift), (replace(a.legs, entries=()), a.lift)):
        with pytest.raises(ValueError, match="original factors"):
            primary_lift(legs, lift)
    with pytest.raises(TypeError, match="canonical boundary"):
        ColumnExpansion(hodge_operator(rex, 1))


def test_source_mutation_refuses_saved_boundary_and_expansion():
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2]]])
    boundary = boundary_operator(rex, 1)
    expansion = ColumnExpansion(boundary)
    action = primary_lift(expansion.legs, expansion.lift)
    rex._boundary_idx[0], rex._boundary_idx[1] = rex._boundary_idx[1], rex._boundary_idx[0]
    for operation in (expansion.check_state, lambda: ColumnExpansion(boundary),
                      lambda: action.apply(np.ones(2, dtype=int), exact=True)):
        with pytest.raises(ValueError):
            operation()


def test_numeric_factors_are_cached_native_and_exact_action_never_materializes(monkeypatch):
    from rexgraph.column_expansion import _Factor
    rex = RexGraph.from_cells([4, [[0, 1, 2, 3]]])
    expansion = ColumnExpansion(boundary_operator(rex, 1))
    result = primary_lift(expansion.legs, expansion.lift)
    first = expansion.legs._native
    result.apply(np.ones(1))
    assert expansion.legs._native is first
    monkeypatch.setattr(_Factor, "_native", property(lambda s: pytest.fail("exact action built numeric factors")))
    assert result.apply(np.array([Q(3)], object), exact=True).tolist() == [-3, 1, 1, 1]
