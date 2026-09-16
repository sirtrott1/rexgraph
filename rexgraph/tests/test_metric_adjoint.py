"""Metric adjoints: independent finite oracles and inherently sparse actions."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.cochain import Chain
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graded_metric import DiagonalMetric, diagonal_metric
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import (
    RexOperator,
    boundary_operator,
    coboundary_operator,
    hodge_operator,
    metric_adjoint,
)


def make(case='branching'):
    if case == 'branching':
        return RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2]), 1
    if case == 'witness':
        return RexGraph.from_hypergraph([0, 1, 3], [0, 0, 1]), 1
    if case == 'parallel':
        return RexGraph.from_graph([0, 0, 1], [1, 1, 2]), 1
    if case == 'face':
        return RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]]), 2
    return RexGraph.from_cells(solid_octahedron_3rex()), 3


def metric(rex, grade, n):
    return DiagonalMetric(rex, grade, tuple(Q(i+1, i+2) for i in range(n)))


@pytest.mark.parametrize('case', ['branching', 'witness', 'parallel', 'face', 'grade3'])
@pytest.mark.parametrize('columns', [None, 0, 3])
def test_exact_adjoint_identity_and_numeric_reference(case, columns):
    rex, k = make(case)
    b = boundary_operator(rex, k)
    md, mc = metric(rex, k, b.shape[1]), metric(rex, k-1, b.shape[0])
    a = metric_adjoint(b, md, mc)
    def vector(n):
        shape = (n,) if columns is None else (n, columns)
        return np.asarray([Q(i-3, i+1) for i in range(int(np.prod(shape)))], object).reshape(shape)
    x, y = vector(b.shape[1]), vector(b.shape[0])
    left = mc.moment(Chain(k-1, b.apply(x, exact=True), source=rex), Chain(k-1, y, source=rex), exact=True)
    right = md.moment(Chain(k, x, source=rex), Chain(k, a.apply(y, exact=True), source=rex), exact=True)
    assert left == right
    assert isinstance(left, Q)
    assert a.variance == 'chain'
    matrix = b.as_scipy().toarray()
    reference = (matrix.T * np.asarray(mc.weights, float)) / np.asarray(md.weights, float)[:, None]
    np.testing.assert_allclose(a.apply(np.asarray(y, float)), reference @ np.asarray(y, float), atol=1e-14)
    np.testing.assert_allclose(a.transpose_apply(np.asarray(x, float)), reference.T @ np.asarray(x, float), atol=1e-14)
    twice = metric_adjoint(a, mc, md)
    np.testing.assert_array_equal(twice.apply(x, exact=True), b.apply(x, exact=True))


def test_branching_independent_exact_values_and_dual_coboundary_difference():
    rex, _ = make()
    md = DiagonalMetric(rex, 1, (2, 3))
    mc = DiagonalMetric(rex, 0, (1, 2, 3, 4))
    y = np.array([1, 2, 3, 4])
    # B columns (1/3,1/3,-1,1/3) and (-1,0,1,0).
    # B.T Mc y = (-2,8); divide by (2,3).
    a = metric_adjoint(boundary_operator(rex, 1), md, mc)
    assert list(a.apply(y, exact=True)) == [Q(-1), Q(8, 3)]
    assert list(coboundary_operator(rex, 0).apply(y, exact=True)) == [Q(-2, 3), Q(2)]


@pytest.mark.parametrize('case', ['branching', 'face', 'grade3'])
def test_weighted_sectors_annihilate_without_euclidean_symmetry_claim(case):
    rex, _ = make(case)
    b = boundary_operator(rex, 1)
    # If absent, B2 is the transpose of the explicit zero d1, not a new grade.
    c = (boundary_operator(rex, 2) if len(rex.graded_boundaries()) > 1
         else metric_adjoint(coboundary_operator(rex, 1)))
    m0, m1, m2 = (metric(rex, k, n) for k, n in enumerate((b.shape[0], b.shape[1], c.shape[1])))
    bd, cd = metric_adjoint(b, m1, m0), metric_adjoint(c, m2, m1)
    x = np.arange(b.shape[1]) + 1
    def down(v):
        return bd.apply(b.apply(v, exact=True), exact=True)
    def up(v):
        return c.apply(cd.apply(v, exact=True), exact=True)
    assert all(v == 0 for v in down(up(x)))
    assert all(v == 0 for v in up(down(x)))
    for adj in (bd, cd):
        assert not adj.symmetric and not adj.psd


def test_weighted_square_adjoint_is_not_euclidean_symmetric_or_psd():
    rex, _ = make('face')
    op = hodge_operator(rex, 0)
    m = DiagonalMetric(rex, 0, (1, 2, 5))
    a = metric_adjoint(op, m, m)
    dense = a.apply(np.eye(3))
    assert not np.allclose(dense, dense.T)
    assert not a.symmetric and not a.psd
    assert a.exact_matvec is None  # Floating Hodge action is not upgraded to Q.
    with pytest.raises(TypeError, match='certified exact'):
        a.apply(np.ones(3, dtype=int), exact=True)


@pytest.mark.parametrize('objects', [False, True])
def test_complex_numeric_carriers_keep_hermitian_adjoint(objects):
    rex, _ = make()
    b = boundary_operator(rex, 1)
    m1, m0 = metric(rex, 1, 2), metric(rex, 0, 4)
    x, y = np.array([1j, 2]), np.array([1, 2j, 3-1j, 4])
    a = metric_adjoint(b, m1, m0)
    values = y.astype(object) if objects else y
    left = np.vdot(b.apply(x), np.asarray(m0.weights, float)*y)
    right = np.vdot(x, np.asarray(m1.weights, float)*a.apply(values))
    assert left == pytest.approx(right)


def test_missing_upper_grade_adjoint_is_typed_exact_zero():
    rex, _ = make('face')
    d = coboundary_operator(rex, 2)
    a = metric_adjoint(d)
    assert a.shape == (1, 0) and a.domain_grade == 3 and a.codomain_grade == 2
    assert a.variance == 'cochain'
    assert a.apply(np.empty((0, 4), object), exact=True).tolist() == [[Q(0)]*4]
    assert a.transpose_apply(np.ones((1, 4), int), exact=True).shape == (0, 4)


@pytest.mark.parametrize('bad', ['source', 'grade', 'basis', 'transpose'])
def test_operator_endpoint_mismatches_are_refused(bad):
    rex, _ = make()
    b = boundary_operator(rex, 1)
    md, mc = diagonal_metric(rex, 1), diagonal_metric(rex, 0)
    if bad == 'source':
        md = diagonal_metric(make()[0], 1)
    elif bad == 'grade':
        md = mc
    elif bad == 'basis':
        md = DiagonalMetric(rex, 1, (1, 1), ('b', 'a'))
    else:
        b = RexOperator('opaque', b.shape, 1, 0, lambda v: v, source=rex,
                        matrix_factory=lambda: pytest.fail('implicit materialization'))
    with pytest.raises((TypeError, ValueError)):
        metric_adjoint(b, md, mc)


@pytest.mark.parametrize('values', [[True]*4, ['1']*4, [float('nan')]*4, [float('inf')]*4])
@pytest.mark.parametrize('exact', [False, True])
def test_bad_inputs_are_validated_even_for_zero_actions(values, exact):
    rex, _ = make()
    a = metric_adjoint(boundary_operator(rex, 1))
    with pytest.raises((TypeError, FloatingPointError)):
        a.apply(np.asarray(values), exact=exact)


@pytest.mark.parametrize('weight', [Q(10**400), Q(1, 10**400)])
def test_exact_extreme_metrics_do_not_cast_through_float(weight):
    rex, _ = make()
    a = metric_adjoint(boundary_operator(rex, 1), DiagonalMetric(rex, 1, (weight, weight)))
    assert a.apply(np.array([1, 2, 3, 4]), exact=True).tolist() == [Q(-2, 3)/weight, Q(2)/weight]
    with pytest.raises(FloatingPointError):
        a.apply(np.array([1, 2, 3, 4]))


def test_no_gram_materialization_inverse_or_eigenbasis(monkeypatch):
    rex = RexGraph.from_graph(np.zeros(4096, int), np.arange(1, 4097))
    b = boundary_operator(rex, 1)
    h = hodge_operator(rex, 1)
    def forbidden(*a, **kw):
        pytest.fail('adjoint materialized or diagonalized')
    monkeypatch.setattr(RexOperator, 'as_scipy', forbidden)
    monkeypatch.setattr(np, 'diag', forbidden)
    monkeypatch.setattr(np.linalg, 'inv', forbidden)
    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    a = metric_adjoint(b)
    assert all(v == 0 for v in a.apply(np.ones(4097, int), exact=True))
    assert np.array_equal(a.apply(np.ones(4097)), np.zeros(4096))
    assert metric_adjoint(h).apply(np.ones(4096)).shape == (4096,)


@pytest.mark.parametrize('bad', [None, 1, 'true'])
def test_exact_flag_is_boolean(bad):
    rex, _ = make()
    a = metric_adjoint(boundary_operator(rex, 1))
    with pytest.raises(TypeError, match='boolean'):
        a.apply(np.ones(4, int), exact=bad)
