"""Canonical dual evaluation, separate from a metric/Riesz contraction."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import diagonal_metric, integrate
from rexgraph.graph import RexGraph


def source():
    return RexGraph.from_graph([0, 1], [1, 2])


def pair(rex, omega, chain, **kwargs):
    return Cochain(1, np.asarray(omega), source=rex, **kwargs), Chain(1, np.asarray(chain), source=rex, **kwargs)


def test_signed_exact_pairing_and_block_sum():
    rex = source()
    a, c = pair(rex, [Q(1, 3), Q(5, 2)], [-3, 2])
    assert integrate(a, c, exact=True) == Q(4)
    assert integrate(a, c) == 4.
    a, c = pair(rex, [[1, 2], [3, 4]], [[-3, 1], [-2, 0]])
    assert integrate(a, c, exact=True) == Q(-7)


@pytest.mark.parametrize('object_array', [False, True])
def test_complex_dual_pairing_is_bilinear_not_hermitian(object_array):
    rex = source()
    a, c = pair(rex, [1j, 1], [1, 2j])
    if object_array:
        a, c = a.with_values(a.values.astype(object)), c.with_values(c.values.astype(object))
    assert integrate(a, c) == 3j
    # Metric moments retain their separate antilinear first convention.
    if not object_array:
        assert diagonal_metric(rex, 1).moment(a, Cochain(1, c.values, source=rex)) == 1j
    with pytest.raises(TypeError, match='exact coefficients'):
        integrate(a, c, exact=True)


@pytest.mark.parametrize('mismatch', ['source', 'unbound', 'grade', 'basis', 'shape', 'count', 'variance', 'raw'])
def test_dual_spaces_are_not_inferred_from_equal_lengths(mismatch):
    rex = source()
    a, c = pair(rex, [1, 2], [3, 4], cell_keys=('a', 'b'))
    if mismatch == 'source':
        c = Chain(1, c.values, c.cell_keys, source())
    elif mismatch == 'unbound':
        a, c = Cochain(1, a.values), Chain(1, c.values)
    elif mismatch == 'grade':
        c = Chain(0, c.values, c.cell_keys, rex)
    elif mismatch == 'basis':
        c = Chain(1, c.values, ('b', 'a'), rex)
    elif mismatch == 'shape':
        c = c.with_values(c.values[:, None])
    elif mismatch == 'count':
        a, c = pair(rex, [1], [2])
    elif mismatch == 'variance':
        c = Cochain(1, c.values, c.cell_keys, rex)
    elif mismatch == 'raw':
        a = a.values
    with pytest.raises((TypeError, ValueError)):
        integrate(a, c, exact=True)


@pytest.mark.parametrize('values', [[True, False], ['1', '2'], [float('nan'), 1], [float('inf'), 1]])
def test_non_numeric_or_nonfinite_inputs_are_refused(values):
    a, c = pair(source(), values, [1, 0])
    with pytest.raises((TypeError, FloatingPointError)):
        integrate(a, c)


@pytest.mark.parametrize('value', [Q(10**400), Q(1, 10**400)])
def test_exact_coefficients_never_round_trip_through_float(value):
    a, c = pair(source(), [value, -value], [2, 1])
    assert integrate(a, c, exact=True) == value


def test_exact_empty_pairing_has_no_unknown_coefficient():
    rex = RexGraph.from_graph([], [])
    a, c = pair(rex, np.empty((0, 3), dtype=object), np.empty((0, 3), dtype=object))
    assert integrate(a, c, exact=True) == Q(0)


def test_no_matrix_or_eigenbasis_is_built(monkeypatch):
    n = 2048
    rex = RexGraph.from_graph(np.zeros(n, dtype=int), np.arange(1, n+1))
    a, c = pair(rex, np.ones(n, dtype=int), np.arange(n))
    def forbidden(*args, **kwargs):
        pytest.fail('dual pairing built a matrix or eigenbasis')
    monkeypatch.setattr(np, 'diag', forbidden)
    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    assert integrate(a, c, exact=True) == Q(n*(n-1), 2)


@pytest.mark.parametrize('bad', [None, 1, 'true'])
def test_exact_flag_is_not_truthiness(bad):
    a, c = pair(source(), [1, 2], [3, 4])
    with pytest.raises(TypeError, match='boolean'):
        integrate(a, c, exact=bad)
