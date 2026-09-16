"""Independent weighted sector oracles, exact identities and no Gram paths."""
from fractions import Fraction as Q

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.cells import cell_count
from rexgraph.cochain import Chain
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import RexOperator, boundary_operator, hodge_operator, metric_adjoint
from rexgraph.weighted_hodge import weighted_hodge


def make(case):
    if case == 'branching':
        return RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2])
    if case == 'witness':
        return RexGraph.from_hypergraph([0, 1, 3], [0, 0, 1])
    if case == 'parallel':
        return RexGraph.from_graph([0, 0, 1], [1, 1, 2])
    if case == 'face':
        return RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])
    return RexGraph.from_cells(solid_octahedron_3rex())


def metrics(rex, k):
    def m(g):
        return DiagonalMetric(rex, g, tuple(Q(i+2, i+1) for i in range(cell_count(rex, g, allow_empty_upper=True))))
    return dict(metric=m(k), lower_metric=m(k-1) if k else None, upper_metric=m(k+1))


def operators(rex, k, ms):
    return {
        'down': weighted_hodge(rex, k, sector='down', metric=ms['metric'], lower_metric=ms['lower_metric']),
        'up': weighted_hodge(rex, k, sector='up', metric=ms['metric'], upper_metric=ms['upper_metric']),
        'sum': weighted_hodge(rex, k, **ms),
        'difference': weighted_hodge(rex, k, sector='difference', **ms),
    }


@pytest.mark.parametrize('case', ['branching', 'witness', 'parallel', 'face', 'grade3'])
@pytest.mark.parametrize('columns', [None, 0, 2])
def test_all_carried_grades_match_independent_coordinate_formulas(case, columns):
    rex = make(case)
    boundaries = rex.graded_boundaries()
    for k in range(len(boundaries)+1):
        n = cell_count(rex, k)
        ms = metrics(rex, k)
        m = np.asarray(ms['metric'].weights, float)
        down, up = np.zeros((n, n)), np.zeros((n, n))
        if k:
            b = boundaries[k-1].toarray()
            down = (b.T * np.asarray(ms['lower_metric'].weights, float)) @ b / m[:, None]
        if k < len(boundaries):
            b = boundaries[k].toarray()
            up = (b / np.asarray(ms['upper_metric'].weights, float)) @ b.T * m
        shape = (n,) if columns is None else (n, columns)
        x = np.arange(int(np.prod(shape)), dtype=float).reshape(shape) - 2
        for name, ref in {'down': down, 'up': up, 'sum': down+up, 'difference': down-up}.items():
            op = operators(rex, k, ms)[name]
            np.testing.assert_allclose(op.apply(x), ref @ x, atol=2e-13)
            np.testing.assert_allclose(op.transpose_apply(x), ref.T @ x, atol=2e-13)
            assert op.variance == 'chain' and op.metric_self_adjoint
            assert not op.symmetric and not op.psd
            assert op.metric_psd is (None if name == 'difference' else True)


@pytest.mark.parametrize('case', ['branching', 'witness', 'parallel', 'face', 'grade3'])
@pytest.mark.parametrize('columns', [None, 0, 2])
def test_exact_metric_energy_sector_laws_and_adjoint(case, columns):
    rex = make(case)
    for k in range(len(rex.graded_boundaries())+1):
        ms = metrics(rex, k)
        ops = operators(rex, k, ms)
        n = cell_count(rex, k)
        shape = (n,) if columns is None else (n, columns)
        x = np.array([Q(i+1, i+2) for i in range(int(np.prod(shape)))], object).reshape(shape)
        y = x*3-1
        applied = {s: op.apply(x, exact=True) for s, op in ops.items()}
        np.testing.assert_array_equal(applied['sum'], applied['down'] + applied['up'])
        np.testing.assert_array_equal(applied['difference'], applied['down'] - applied['up'])
        assert not any(ops['down'].apply(applied['up'], exact=True).flat)
        assert not any(ops['up'].apply(applied['down'], exact=True).flat)
        np.testing.assert_array_equal(ops['sum'].apply(applied['sum'], exact=True),
                                      ops['difference'].apply(applied['difference'], exact=True))
        def moment(a, b, m=ms['metric'], grade=k):
            return m.moment(Chain(grade, a, source=rex), Chain(grade, b, source=rex), exact=True)
        for s, op in ops.items():
            assert moment(x, op.apply(y, exact=True)) == moment(applied[s], y)
            if s != 'difference':
                assert moment(x, applied[s]) >= 0
            adj = metric_adjoint(op, ms['metric'], ms['metric'])
            np.testing.assert_array_equal(adj.apply(x, exact=True), applied[s])
        # The sum's energy equals the energies of the two differentials.
        energy = Q(0)
        if k:
            bx = boundary_operator(rex, k).apply(x, exact=True)
            c = Chain(k-1, bx, source=rex)
            energy += ms['lower_metric'].moment(c, c, exact=True)
        if k < len(rex.graded_boundaries()):
            adj = metric_adjoint(boundary_operator(rex, k+1), ms['upper_metric'], ms['metric'])
            ax = Chain(k+1, adj.apply(x, exact=True), source=rex)
            energy += ms['upper_metric'].moment(ax, ax, exact=True)
        assert moment(x, applied['sum']) == energy


def test_branching_has_exact_hand_values_and_is_not_euclidean_symmetric():
    rex = make('branching')
    md, ml = DiagonalMetric(rex, 1, (2, 3)), DiagonalMetric(rex, 0, (1, 2, 3, 4))
    op = weighted_hodge(rex, 1, metric=md, lower_metric=ml)
    x = np.array([3, 2])
    assert list(op.apply(x, exact=True)) == [Q(7, 3), Q(-2, 3)]
    assert md.moment(Chain(1, x, source=rex), Chain(1, op.apply(x, exact=True), source=rex), exact=True) == 10
    dense = op.apply(np.eye(2))
    np.testing.assert_allclose(dense, [[17/9, -5/3], [-10/9, 4/3]])
    assert not np.allclose(dense, dense.T)


def test_difference_has_both_signs_on_a_filled_triangle():
    rex = make('face')
    ms = metrics(rex, 1)
    op = weighted_hodge(rex, 1, sector='difference', **ms)
    curl = boundary_operator(rex, 2).apply(np.ones(1, int), exact=True)
    grad = metric_adjoint(boundary_operator(rex, 1), ms['metric'], ms['lower_metric']).apply(np.array([1, 0, 0]), exact=True)
    def energy(x):
        return ms['metric'].moment(Chain(1, x, source=rex), Chain(1, op.apply(x, exact=True), source=rex), exact=True)
    assert energy(curl) < 0 and energy(grad) > 0


@pytest.mark.parametrize('case', ['branching', 'face', 'grade3'])
def test_identity_metrics_agree_with_existing_euclidean_reference(case):
    rex = make(case)
    for k in range(len(rex.graded_boundaries())+1):
        x = np.arange(cell_count(rex, k), dtype=float)+1
        np.testing.assert_allclose(weighted_hodge(rex, k).apply(x), hodge_operator(rex, k).apply(x))


def test_complex_action_is_hermitian_in_its_metric():
    rex = make('branching')
    ms = metrics(rex, 1)
    op = weighted_hodge(rex, 1, **ms)
    x, y = np.array([1j, 2]), np.array([3, 4j])
    m = np.asarray(ms['metric'].weights, float)
    assert np.vdot(x, m*op.apply(y)) == pytest.approx(np.vdot(op.apply(x), m*y))


@pytest.mark.parametrize('sector,k', [('down', 0), ('up', 1)])
def test_missing_sector_is_exact_zero_even_with_a_numeric_metric(sector, k):
    rex = make('branching')
    n = cell_count(rex, k)
    op = weighted_hodge(rex, k, sector=sector, metric=DiagonalMetric(rex, k, (2.,)*n))
    assert op.active_sectors == ()
    assert op.apply(np.ones((n, 2), int), exact=True).tolist() == [[Q(0), Q(0)]]*n
    with pytest.raises(TypeError, match='exact coefficients'):
        op.apply(np.ones(n), exact=True)
    with pytest.raises(FloatingPointError):
        op.apply(np.full(n, float('nan')))


@pytest.mark.parametrize('bad', ['source', 'grade', 'basis', 'lower', 'unused', 'grade-type', 'unknown-sector'])
def test_bad_operator_spaces_and_options_are_refused(bad):
    rex = make('branching')
    kw, k = {}, 1
    if bad == 'source':
        kw['metric'] = DiagonalMetric(make('branching'), 1, (1, 2))
    elif bad == 'grade':
        kw['metric'] = DiagonalMetric(rex, 0, (1,)*4)
    elif bad == 'basis':
        kw['metric'] = DiagonalMetric(rex, 1, (1, 2), ('b', 'a'))
    elif bad == 'lower':
        k, kw['lower_metric'] = 0, DiagonalMetric(rex, 0, (1,)*4)
    elif bad == 'unused':
        kw = dict(sector='down', upper_metric=DiagonalMetric(rex, 2, ()))
    elif bad == 'grade-type':
        k = True
    else:
        kw['sector'] = 'G'
    with pytest.raises((TypeError, ValueError)):
        weighted_hodge(rex, k, **kw)


@pytest.mark.parametrize('weight', [Q(10**400), Q(1, 10**400)])
def test_exact_metric_range_is_not_limited_by_float(weight):
    rex = make('branching')
    op = weighted_hodge(rex, 1, metric=DiagonalMetric(rex, 1, (weight, weight)))
    x = np.array([1, 0])
    assert op.apply(x, exact=True).tolist() == [Q(4, 3)/weight, Q(-4, 3)/weight]
    with pytest.raises(FloatingPointError):
        op.apply(x)


def test_hub_actions_never_form_grams_or_inverses(monkeypatch):
    rex = RexGraph.from_graph(np.zeros(4096, int), np.arange(1, 4097))
    def forbidden(*a, **kw):
        pytest.fail('weighted Hodge assembled or diagonalized a matrix')
    monkeypatch.setattr(RexOperator, 'as_scipy', forbidden)
    monkeypatch.setattr(sp.csr_matrix, 'toarray', forbidden)
    monkeypatch.setattr(np, 'diag', forbidden)
    monkeypatch.setattr(np.linalg, 'inv', forbidden)
    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    op = weighted_hodge(rex, 1)
    assert np.array_equal(op.apply(np.ones(4096)), np.full(4096, 4097.))
    assert all(v == Q(4097) for v in op.apply(np.ones(4096, int), exact=True))


def test_nonintegral_higher_boundary_stays_numeric_not_reconstructed_rational():
    from rexgraph.native_sparse import as_native
    rex = make('grade3')
    upper = as_native(rex._graded_duals[0])
    rex._graded_duals = [upper.with_data(upper.data * .5).dual]
    op = weighted_hodge(rex, 3)
    assert op.exact_matvec is None and op.exact_transpose_matvec is None
    np.testing.assert_allclose(op.apply(np.ones(1)), [2.])
    with pytest.raises(TypeError, match='certified exact'):
        op.apply(np.ones(1, int), exact=True)


def test_nonfinite_boundary_cannot_acquire_a_metric_psd_certificate():
    from rexgraph.native_sparse import as_native
    rex = make('grade3')
    upper = as_native(rex._graded_duals[0])
    rex._graded_duals = [upper.with_data(upper.data * float('nan')).dual]
    with pytest.raises(ValueError, match='finite boundary'):
        weighted_hodge(rex, 3)


@pytest.mark.parametrize('sector', ['down', 'up', 'sum', 'difference'])
def test_empty_complex_preserves_zero_width_and_zero_population(sector):
    rex = RexGraph.from_graph([], [])
    op = weighted_hodge(rex, 0, sector=sector)
    for width in (0, 3):
        result = op.apply(np.empty((0, width), object), exact=True)
        assert result.shape == (0, width) and result.dtype == object
