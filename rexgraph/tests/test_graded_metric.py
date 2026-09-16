"""Explicit diagonal forms: exact oracles and carrier/refusal contracts."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import DiagonalMetric, diagonal_metric
from rexgraph.graph import RexGraph


def fixture():
    return RexGraph.from_graph([0, 1], [1, 2])


def co(rex, values, **kw):
    return Cochain(1, np.asarray(values), source=rex, **kw)


def test_exact_signed_and_block_contractions_copy_coefficients():
    rex = fixture()
    weights = co(rex, [Q(1, 3), Q(5, 2)])
    metric = diagonal_metric(rex, 1, weights)
    digest = metric.coefficient_digest
    weights.values[0] = 100
    a, b = co(rex, [[1, 2], [3, 4]]), co(rex, [[-3, 1], [-2, 0]])
    assert metric.moment(a, b, exact=True) == Q(-46, 3)
    assert metric.moment(a, a, exact=True) == Q(385, 6)
    assert metric.weights == (Q(1, 3), Q(5, 2))
    assert metric.coefficient_digest == digest
    assert diagonal_metric(rex, 1, co(rex, [Q(1, 3), Q(5, 2)])).coefficient_digest == digest
    assert diagonal_metric(rex, 1).coefficient_digest != digest


def test_complex_hermitian_not_bilinear_or_squared_moment():
    rex = fixture()
    metric = diagonal_metric(rex, 1, co(rex, [2, 3]))
    a, b = co(rex, [1j, 1]), co(rex, [1, 1j])
    assert metric.moment(a, b) == 1j
    assert metric.moment(b, a) == -1j
    assert metric.moment(a, a) == 5
    with pytest.raises(TypeError, match="exact coefficients"):
        metric.moment(a, b, exact=True)


@pytest.mark.parametrize("weights", [[0, 1], [-1, 1], [True, 1], [1j, 1],
                                    [float("nan"), 1], [float("inf"), 1], [[1, 2]], [1]])
def test_invalid_diagonals_refuse(weights):
    with pytest.raises((ValueError, TypeError)):
        DiagonalMetric(fixture(), 1, weights)


@pytest.mark.parametrize("weight", [Q(10**400), Q(1, 10**400)])
def test_exact_extremes_are_not_forced_through_double(weight):
    rex = fixture()
    metric = diagonal_metric(rex, 1, co(rex, [weight, weight]))
    a = co(rex, [1, 2])
    assert metric.moment(a, a, exact=True) == 5 * weight
    with pytest.raises(FloatingPointError, match="representable"):
        metric.moment(a, a)


def test_ordered_basis_source_grade_variance_and_shape_are_not_length_equivalence():
    rex = fixture()
    metric = diagonal_metric(rex, 1, co(rex, [1, 2], cell_keys=("b", "a")))
    a = co(rex, [1, 2], cell_keys=("b", "a"))
    assert metric.moment(a, a, exact=True) == 9
    bad = [co(rex, [1, 2], cell_keys=("a", "b")), co(fixture(), [1, 2]),
           Cochain(0, np.array([1, 2]), source=rex),
           Chain(1, np.array([1, 2]), cell_keys=("b", "a"), source=rex),
           co(rex, [[1], [2]], cell_keys=("b", "a"))]
    for value in bad:
        with pytest.raises(ValueError):
            metric.moment(a, value)


def test_empty_metric_has_exact_zero_contraction():
    rex = RexGraph.from_graph([], [])
    metric = diagonal_metric(rex, 1)
    empty = co(rex, np.empty((0, 3), dtype=object))
    assert metric.moment(empty, empty, exact=True) == Q(0)
    assert metric.moment(empty, empty) == 0.0


def test_large_diagonal_contraction_without_matrix_allocation(monkeypatch):
    n = 4096
    rex = RexGraph.from_graph(np.zeros(n, dtype=int), np.arange(1, n+1))
    def forbidden(*args, **kw):
        pytest.fail("diagonal contraction assembled a matrix or diagonalized")
    monkeypatch.setattr(np, "diag", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    metric = diagonal_metric(rex, 1, co(rex, np.arange(1, n+1)))
    ones = co(rex, np.ones(n, dtype=int))
    assert metric.moment(ones, ones, exact=True) == Q(n*(n+1), 2)
    assert metric.moment(ones, ones) == n*(n+1)/2
