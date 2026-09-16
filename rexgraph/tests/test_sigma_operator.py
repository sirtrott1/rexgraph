"""Small matrix oracles for the explicit sparse sigma family and its derivative."""
from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.linear_operator import RexOperator
from rexgraph.native_sparse import NativeSparse
from rexgraph.sigma_operator import critical_commutator, critical_rate, sigma_operator


def make(cells, weights=None):
    return RexGraph(boundary_ptr=np.array([0, *np.cumsum([len(c) for c in cells])], dtype=int),
                    boundary_idx=np.array([v for c in cells for v in c], dtype=int),
                    w_E=None if weights is None else np.array(weights, dtype=object))


def reference(rex, sigma, amplitudes, rates, channels, c_channel):
    b = NativeSparse(rex._B1_dual).apply(np.eye(rex.nE))
    w = np.asarray(amplitudes) * np.exp(sigma * np.asarray(rates))
    q = (b*b).T @ w
    d = np.zeros_like(b)
    mask = q > 0
    d[:, mask] = np.sqrt(w[:, None]) * b[:, mask] / np.sqrt(q[None, mask])
    m = np.ones(rex.nE) if rex.edge_metric_exact is None else np.array(rex.edge_metric_exact, float)
    t = (d*m).T @ (d*m)
    g = (abs(d)*m).T @ (abs(d)*m)
    f = t-g
    np.fill_diagonal(f, 0)
    f += np.diag(abs(f).sum(axis=1))
    support = (b != 0).astype(float) if c_channel == "count" else abs(b)
    c = support.T @ support
    c = np.diag(c.sum(axis=1)) - c
    matrices = dict(zip("TGFC", (t, g, f, c), strict=True))
    out = np.zeros((rex.nE, rex.nE))
    for key in channels:
        trace = np.trace(matrices[key])
        if trace > 0:
            out += matrices[key] / trace
    return out


@pytest.mark.parametrize("cells,weights", [
    ([(0, 1), (1, 2), (2, 0)], None),
    ([(0,), (0, 1), (1, 0, 2), (2, 0, 1, 3)], [-2, 3, 0.2, 0]),
    ([(0, 1, 2, 3, 4), (1, 0, 2, 3, 4), (6,)], [2, -3, 0]),
])
@pytest.mark.parametrize("channels", [("T",), ("G",), ("F",), ("C",), tuple("TGFC")])
@pytest.mark.parametrize("c_channel", ["count", "share"])
def test_actions_and_analytic_derivatives_match_small_oracles(cells, weights, channels, c_channel):
    rex = make(cells, weights)
    amplitudes = list(range(1, rex.nV + 1))
    rates = list(np.linspace(-2, 1, rex.nV))
    args = (amplitudes, rates, channels, c_channel)
    op = sigma_operator(rex, 0.63, *args)
    x = np.arange(rex.nE*2).reshape(rex.nE, 2) - 1
    matrix = reference(rex, 0.63, *args)
    np.testing.assert_allclose(op.apply(x), matrix @ x, atol=3e-14)
    np.testing.assert_allclose(op.transpose_apply(x), matrix.T @ x, atol=3e-14)
    h = 2e-5
    derivative = (reference(rex, 0.63+h, *args) - reference(rex, 0.63-h, *args))/(2*h)
    np.testing.assert_allclose(op.derivative.apply(x), derivative @ x, atol=2e-9)
    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-12
    assert op.exact_matvec is None and op.derivative.exact_matvec is None
    assert op.apply(np.empty((rex.nE, 0))).shape == (rex.nE, 0)


def test_prime_weighted_theorem_family_is_explicit_not_a_default():
    rex = make([(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)])
    bases = np.array([2, 3, 5, 7])
    args = (np.log(bases).tolist(), (-np.log(bases)).tolist(), list("TGFC"), "count")
    op = sigma_operator(rex, 0.5, *args)
    np.testing.assert_allclose(op.apply(np.eye(rex.nE)), reference(rex, 0.5, *args), atol=1e-14)
    alternative = sigma_operator(rex, 0.5, [1]*4, [0, -1, -2, -3], list("TGFC"), "count")
    assert not np.allclose(op.apply(np.ones(rex.nE)), alternative.apply(np.ones(rex.nE)))


def test_critical_generator_and_involution_slope_have_the_correct_sign_and_factor():
    rex = make([(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)])
    args = ([1, 2, 4, 3], [-2, -1, 0, 1], list("TGFC"), "share")
    x = np.eye(rex.nE)
    center = critical_commutator(rex, 0.5, *args).apply(x)
    assert not np.any(center)
    generator = critical_rate(rex, *args).apply(x)
    slope = critical_rate(rex, *args, "slope").apply(x)
    assert np.max(abs(generator)) > 1e-4
    np.testing.assert_allclose(slope, -2 * generator)
    np.testing.assert_allclose(generator.T, -generator, atol=1e-14)
    h = 1e-4
    difference = critical_commutator(rex, 0.5+h, *args).apply(x) / h
    np.testing.assert_allclose(difference, slope, atol=2e-9)
    np.testing.assert_allclose(critical_commutator(rex, 0.5-h, *args).apply(x), -difference*h, atol=1e-14)


def test_common_weight_and_rate_scaling_cancel_and_zero_channels_stay_zero():
    rex = make([(0, 1), (0, 2), (3,)], [1, 1, 0])
    x = np.arange(3.0)
    args = ([1, 2, 3, 4], [-1, -2, -3, -4], list("TGFC"), "share")
    op = sigma_operator(rex, 0.5, *args)
    changed = sigma_operator(rex, 0.5, [10, 20, 30, 40], [1, 0, -1, -2], list("TGFC"), "share")
    np.testing.assert_allclose(op.apply(x), changed.apply(x), atol=1e-14)
    np.testing.assert_allclose(op.derivative.apply(x), changed.derivative.apply(x), atol=1e-14)
    frustration = sigma_operator(rex, 0.5, args[0], args[1], ["F"], "share")
    assert not np.any(frustration.apply(x))
    assert not np.any(frustration.derivative.apply(x))


def test_hub_actions_do_not_materialize_or_use_eigensolvers(monkeypatch):
    n = 4096
    rex = RexGraph.from_graph(np.zeros(n, int), np.arange(1, n+1))
    def fail(*args, **kwargs):
        pytest.fail("sigma action requested a matrix product or spectral oracle")
    monkeypatch.setattr(RexGraph, "B1_dense", property(fail))
    monkeypatch.setattr(NativeSparse, "product", fail)
    monkeypatch.setattr(RexOperator, "as_native", fail)
    monkeypatch.setattr(np.linalg, "eigh", fail)
    monkeypatch.setattr(np.linalg, "eigvalsh", fail)
    op = sigma_operator(rex, 0.5, [1]*(n+1), [0]*(n+1), list("TGFC"), "share")
    assert np.isfinite(op.apply(np.ones(n))).all()
    assert not np.any(op.derivative.apply(np.ones(n)))


def test_invalid_weights_underflow_and_changed_source_are_not_silently_accepted():
    rex = make([(0, 1), (1, 2)])
    with pytest.raises(FloatingPointError, match="dynamic range"):
        sigma_operator(rex, 1, [1]*3, [0, -10000, 0], list("TGFC"), "share")
    op = sigma_operator(rex, 0.5, [1]*3, [0, -1, -2], list("TGFC"), "share")
    rex.set_cell_attrs(np.array([0, 1]), w_E=np.array([1, 2]))
    with pytest.raises(ValueError, match="changed"):
        op.apply(np.ones(2))


def test_repeated_slots_are_refused_as_in_the_current_channel_contract():
    rex = make([(0, 0)])
    with pytest.raises(ValueError, match="distinct participants"):
        sigma_operator(rex, 0.5, [1], [0], ["T"], "share")


def test_tiny_nonzero_frustration_is_normalized_without_gram_cancellation():
    rex = make([(0, 1), (1, 2)], [1e-150, 1e-150])
    op = sigma_operator(rex, 1, [1]*3, [0, -700, -700], ["F"], "share")
    np.testing.assert_allclose(op.apply(np.array([1., 0.])), [0.5, -0.5])
    np.testing.assert_allclose(op.apply(np.array([1., 1.])), [0, 0], atol=1e-14)
    np.testing.assert_allclose(op.derivative.apply(np.array([1., 0.])), [0, 0], atol=1e-12)


@pytest.mark.parametrize("weight", [10**400, Fraction(1, 10**400)])
def test_common_metric_scale_cancels_before_float_conversion(weight):
    args = (0.63, [1, 2, 3], [-1, -2, -3], list("TGFC"), "share")
    rex = make([(0, 1), (1, 2)], [weight, -2*weight])
    unit = make([(0, 1), (1, 2)], [1, -2])
    x = np.array([2, 3])
    np.testing.assert_allclose(sigma_operator(rex, *args).apply(x), sigma_operator(unit, *args).apply(x))


def test_C_only_does_not_evaluate_unused_vertex_or_relation_weights():
    rex = make([(0, 1), (1, 2)], [10**400, 1])
    op = sigma_operator(rex, 1, [1]*3, [0, -10000, -10000], ["C"], "share")
    np.testing.assert_allclose(op.apply(np.array([1, 0])), [0.5, -0.5])
    assert not np.any(op.derivative.apply(np.array([1, 0])))
