"""Exact adjugate identities, including rank deficient and empty operators."""
from fractions import Fraction as Q
from itertools import permutations

import numpy as np
import pytest

from rexgraph.adjugate_operator import AdjugateOperator
from rexgraph.channel_operator import channel_operator
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import RexOperator


def determinant(a):
    n = len(a)
    total = Q(0)
    for p in permutations(range(n)):
        sign = (-1)**sum(p[i] > p[j] for i in range(n) for j in range(i+1, n))
        term = Q(sign)
        for i in range(n):
            term *= a[i, p[i]]
        total += term
    return total


def primal(a):
    n = len(a)
    rex = RexGraph.from_cells([max(1, n+1), [[0, i+1] for i in range(n)]])
    return RexOperator("explicit-test-action", (n, n), 1, 1, lambda x: np.asarray(a, float) @ x,
                       source=rex, transpose_matvec=lambda x: np.asarray(a, float).T @ x,
                       exact_matvec=lambda x: a @ x, exact_transpose_matvec=lambda x: a.T @ x)


@pytest.mark.parametrize("a", [
    np.empty((0, 0), dtype=object), np.array([[Q(0)]], object), np.array([[2**200]], object),
    np.array([[Q(1, 3), Q(2, 7)], [-3, 4]], object),
    np.array([[1, 2, 3], [1, 2, 3], [0, 1, 1]], object),
    np.array([[1, 2, 3], [2, 4, 6], [-1, -2, -3]], object),
    np.array([[3, 1, 0, 2], [4, 2, 0, 1], [3, 2, 1, 1], [0, -3, 2, 1]], object),
])
def test_exact_cofactor_and_determinant_identities_without_inverting(a):
    a = np.array([Q(v) for v in a.flat], object).reshape(a.shape)
    n = len(a)
    expected = np.empty((n, n), object)
    for i in range(n):
        for j in range(n):
            expected[i, j] = (-1)**(i+j) * determinant(np.delete(np.delete(a, j, 0), i, 1))
    op = AdjugateOperator(primal(a))
    result = op.apply(np.eye(n, dtype=int), exact=True)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(a @ result, determinant(a) * np.eye(n, dtype=int))
    np.testing.assert_array_equal(result @ a, determinant(a) * np.eye(n, dtype=int))
    np.testing.assert_array_equal(op.transpose_apply(np.eye(n, dtype=int), exact=True), result.T)
    np.testing.assert_allclose(op.apply(np.ones(n)), np.asarray(result @ np.ones(n, dtype=int), float))
    assert op.apply(np.empty((n, 0), dtype=int), exact=True).shape == (n, 0)


def test_native_channel_adjugate_never_materializes_or_uses_numeric_inverse(monkeypatch):
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2], [2, 0]]])
    a = channel_operator(rex, "T")
    def fail(*args, **kwargs):
        pytest.fail("adjugate used a matrix, inverse or eigensolver")
    monkeypatch.setattr(RexOperator, "as_native", fail)
    monkeypatch.setattr(RexGraph, "B1_dense", property(fail))
    monkeypatch.setattr(np.linalg, "inv", fail)
    monkeypatch.setattr(np.linalg, "eigh", fail)
    x = np.array([Q(1, 3), Q(2, 5), 2**100], object)
    result = AdjugateOperator(a).apply(x, exact=True)
    assert np.any(result != 0)
    assert not np.any(a.apply(result, exact=True))


def test_coefficients_are_not_cached_after_a_primal_change():
    a = np.array([[Q(1), Q(2)], [Q(3), Q(4)]], object)
    op = AdjugateOperator(primal(a))
    x = np.array([1, 0])
    assert op.apply(x, exact=True).tolist() == [4, -3]
    a[1, 1] = Q(9)
    assert op.apply(x, exact=True).tolist() == [9, -3]


def test_numerical_only_primal_is_not_given_a_rational_certificate():
    a = primal(np.eye(2, dtype=object))
    from dataclasses import replace
    with pytest.raises(TypeError, match="certified Q"):
        AdjugateOperator(replace(a, exact_matvec=None))
