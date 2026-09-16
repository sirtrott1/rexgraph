"""Channel actions and exact geometry against small independent oracles."""
from fractions import Fraction as Q

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.graph import RexGraph
from rexgraph.rational_trig import (
    cross_spread,
    exact_channel_diagonals,
    exact_character,
    quadrance,
    spread,
)
from rexgraph.sparse_character import (
    _trace_normalized,
    build_sparse_channels,
    build_sparse_character_cheap,
    channel_diagonals,
    closed_form_applies,
)


def make(cells, weights=None, **kw):
    return RexGraph(boundary_ptr=np.array([0, *np.cumsum([len(c) for c in cells])], np.int32),
                    boundary_idx=np.array([v for c in cells for v in c], np.int32),
                    w_E=None if weights is None else np.array(weights, dtype=object), **kw)


TRI = [(0, 1), (1, 2), (2, 0)]


@pytest.mark.parametrize("weight", [Q(1000001, 1000000), Q(2**60 + 1, 2**60)])
def test_unweighted_predicate_is_exact(weight):
    r = make(TRI, [weight, 1, 1])
    assert not closed_form_applies(r)
    actual = channel_diagonals(r)
    exact, _ = exact_channel_diagonals(r)
    for name in exact:
        np.testing.assert_allclose(actual[name], np.array(exact[name], float), rtol=1e-15)


@pytest.mark.parametrize("weight", [Q(1, 10**9), Q(1, 10**150)])
def test_nonzero_channels_are_not_dropped_by_magnitude(weight):
    r = make(TRI, [weight] * 3)
    bundle = build_sparse_character_cheap(r)
    np.testing.assert_allclose(bundle["chi"], np.full((3, 4), .25))
    for hat in bundle["hats"]:
        assert hat.diagonal().sum() == pytest.approx(1)


def test_subnormal_trace_normalizes_without_reciprocal_overflow():
    tiny = np.nextafter(0., 1.)
    hat = _trace_normalized(sp.csr_matrix([[tiny]]), tiny, 1)
    assert hat[0, 0] == 1


@pytest.mark.parametrize("trace", [float('nan'), float('inf'), -1.])
def test_invalid_traces_refuse(trace):
    with pytest.raises((ValueError, FloatingPointError)):
        _trace_normalized(sp.eye(1, format='csr'), trace, 1)


def test_normalized_G_diagonal_is_exact_without_assembly(monkeypatch):
    r = make([(0, 1, 2), (1, 0, 2)], [2, 3], g_channel="normalized")
    def forbidden(*a, **k):
        raise AssertionError("exact diagonal assembled an operator")
    monkeypatch.setattr(RexGraph, "overlap_gramian_sparse", property(forbidden))
    d, names = exact_channel_diagonals(r)
    assert d['L_O'] == [Q(5, 9), Q(5, 14)]
    chi, actual_names = exact_character(r)
    assert actual_names == names
    assert chi == [[Q(92, 573), Q(182, 573), Q(299, 1146), Q(299, 1146)],
                   [Q(207, 623), Q(117, 623), Q(299, 1246), Q(299, 1246)]]


def test_cross_spread_uses_same_convention_as_vector_spread():
    a, b = [Q(-1), Q(1, 2), Q(1, 2)], [Q(1, 2), Q(-1), Q(1, 2)]
    t = [[Q(3, 2), Q(-3, 4)], [Q(-3, 4), Q(3, 2)]]
    g = [[Q(3, 2), Q(5, 4)], [Q(5, 4), Q(3, 2)]]
    st, sg, diff, den = cross_spread(t, g)
    assert st == spread(a, b, exact=True) == Q(3, 4)
    assert sg == spread(np.abs(a), np.abs(b), exact=True) == Q(11, 36)
    assert diff == st - sg == Q(4, 9)
    assert den == Q(9, 4)


def test_complex_quadrance_and_spread_are_Hermitian():
    assert quadrance([1j, 0]) == 1
    assert spread([1j, 0], [1, 0]) == 0
    assert spread([1j, 0], [0, 1]) == 1
    with pytest.raises(TypeError):
        quadrance([1j, 0], exact=True)


@pytest.mark.parametrize("cells,weights", [(TRI, [1, 2, 3]),
    ([(0,), (0, 1), (1, 0, 2), (2, 0, 1, 3)], [-2, 3, -.5, 0])])
@pytest.mark.parametrize("c_channel", ["share", "count"])
def test_factored_channels_read_actual_weights_and_C(cells, weights, c_channel):
    from rexgraph._experimental import build_factored_operator
    r = make(cells, weights, c_channel=c_channel)
    matrices = dict(build_sparse_channels(r))
    names = list(matrices)
    traces = [float(m.diagonal().sum()) for m in matrices.values()]
    # No pre assembled channels are needed by the factored implementation.
    action, hat, _ = build_factored_operator(r, None, names, traces)
    x = np.arange(r.nE * 2, dtype=float).reshape(r.nE, 2) - 1
    total = np.zeros_like(x)
    for name, tr in zip(names, traces, strict=True):
        expected = matrices[name] @ x / tr if tr else np.zeros_like(x)
        np.testing.assert_allclose(hat(name, x), expected, atol=1e-13)
        np.testing.assert_allclose(hat(name, x[:, 0]), expected[:, 0], atol=1e-13)
        total += expected
    np.testing.assert_allclose(action(x), total, atol=1e-13)


def test_exact_rank_refuses_near_integer_numeric_matrix():
    from rexgraph.graded_boundary import _sparse_rank
    a = sp.csr_matrix([[1., 1.], [1., 1.0000001]])
    with pytest.raises(ValueError, match="exact"):
        _sparse_rank(a, exact=True)


def test_branching_rank_shortcut_requires_zero_sum():
    from rexgraph.graded_boundary import _sparse_rank
    a = sp.csr_matrix([[-1, 0, 1], [1, -1, 1], [0, 1, 1]])
    assert _sparse_rank(a, exact=True) == 3


@pytest.mark.parametrize('weights', [[0, 0, 0], [1, 2, 3], [0, 1, 2]])
def test_normalized_factored_and_diagonal_readers(weights, monkeypatch):
    from rexgraph._experimental import build_factored_operator
    r = make(TRI, weights, g_channel='normalized')
    oracle = dict(build_sparse_channels(r))
    names = list(oracle)
    traces = [float(m.diagonal().sum()) for m in oracle.values()]
    def forbidden(*a, **k):
        raise AssertionError('diagonal/action assembled a Gram')
    monkeypatch.setattr(RexGraph, 'overlap_gramian_sparse', property(forbidden))
    d = channel_diagonals(r)
    exact, _ = exact_channel_diagonals(r)
    action, hat, _ = build_factored_operator(r, None, names, traces)
    x = np.array([1., -2., 3.])
    for name, tr in zip(names, traces, strict=True):
        np.testing.assert_allclose(d[name], oracle[name].diagonal(), atol=1e-14)
        np.testing.assert_allclose(np.array(exact[name], float), d[name], atol=1e-14)
        np.testing.assert_allclose(hat(name, x), oracle[name] @ x / tr if tr else np.zeros(3), atol=1e-14)


def test_float_mass_underflow_is_not_a_zero_channel():
    r = make(TRI, [Q(1, 10**200)] * 3)
    with pytest.raises(FloatingPointError, match='exact_character'):
        build_sparse_character_cheap(r)
    exact, _ = exact_character(r)
    assert exact == [[Q(1, 4)] * 4] * 3


def test_signed_normalized_weights_are_refused():
    r = make(TRI, [1, -1, 1], g_channel='normalized')
    for reader in (exact_channel_diagonals, channel_diagonals, build_sparse_channels):
        with pytest.raises(ValueError, match='nonnegative relation weights'):
            reader(r)


def test_complex_gram_and_zero_spread_matrix():
    from rexgraph.rational_trig import gram, spread_matrix
    rows = [[1j, 1], [1, 1j], [0, 0]]
    g = gram(rows)
    np.testing.assert_array_equal(g, np.asarray(rows).conj() @ np.asarray(rows).T)
    spreads = spread_matrix(rows)
    assert spreads[0, 1] == 1
    assert np.isnan(spreads[2]).all()
    assert spread_matrix([[0, 0]], exact=True) == [[None]]


@pytest.mark.parametrize('arity', [2, 3, 4, 5, 8])
@pytest.mark.parametrize('weight', [1, 3, Q(1, 10**9)])
def test_isolated_normalized_G_is_exactly_zero(arity, weight):
    r = make([tuple(range(arity))], [weight], g_channel='normalized')
    d = channel_diagonals(r)
    assert d['L_O'].tolist() == [0.]
    assert dict(build_sparse_channels(r))['L_O'].nnz == 0
    np.testing.assert_array_equal(build_sparse_character_cheap(r)['chi'], [[1, 0, 0, 0]])


def test_repeated_slot_factored_contract_refuses_instead_of_changing_meaning():
    from rexgraph._experimental import build_factored_operator
    r = make([(0, 0), (0, 1)])
    with pytest.raises(ValueError, match='distinct participants'):
        build_factored_operator(r, None, [], [])
