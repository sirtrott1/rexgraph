"""Indefinite field/metric contracts; dense operations below are independent oracles."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from rexgraph import field_propagator as fp
from rexgraph.graph import RexGraph


def triangle_and_edges():
    return RexGraph.from_simplicial(
        np.array([0, 0, 1] + list(range(3, 43, 2))),
        np.array([1, 2, 2] + list(range(4, 44, 2))), np.array([[0, 1, 2]]))


@pytest.fixture
def indefinite():
    rex = triangle_and_edges()
    M = fp.assemble_field_operator(rex)
    assert rex.c2_E == Fraction(9, 98)
    assert Fraction(3) * rex.c2_E - Fraction(1, 3) == Fraction(-17, 294)
    assert fp.field_coupling(rex) ** 2 == pytest.approx(1 / 3)
    assert np.linalg.eigvalsh(M.toarray())[0] < -0.04
    return rex, M


def metric(n, kind):
    if kind == "identity":
        return None, np.eye(n)
    if kind == "diagonal":
        d = np.linspace(.7, 1.4, n)
        return d, np.diag(d)
    W = sp.diags([np.full(n - 1, -.12), np.linspace(.8, 1.8, n),
                  np.full(n - 1, -.12)], [-1, 0, 1], format="csr")
    return W, W.toarray()


@pytest.mark.parametrize("kind", ["identity", "diagonal", "full"])
@pytest.mark.parametrize("t", [-3., 0., .5, 4.])
def test_heat_and_wave_solve_the_defined_indefinite_equations(indefinite, kind, t):
    rex, M = indefinite
    n = M.shape[0]
    W, dense_W = metric(n, kind)
    K = la.solve(dense_W, M.toarray(), assume_a="pos")
    rng = np.random.default_rng(17)
    F = rng.normal(size=n)
    V = rng.normal(size=n)
    generator = np.block([[np.zeros((n, n)), np.eye(n)], [-K, np.zeros((n, n))]])
    reference = la.expm(t * generator) @ np.concatenate([F, V])
    pos, vel = fp.field_wave_full(rex, F, [t], W=W, velocity=V)
    np.testing.assert_allclose(pos[0], reference[:n], atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(vel[0], reference[n:], atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(fp.field_heat(rex, F, t, W=W), la.expm(-t * K) @ F,
                               atol=2e-9, rtol=2e-10)


def test_negative_mode_grows_instead_of_freezing(indefinite):
    rex, M = indefinite
    values, basis = np.linalg.eigh(M.toarray())
    value, F = values[0], basis[:, 0]
    times = np.array([-2., 0., 1., 4.])
    rate = np.sqrt(-value)
    pos, vel = fp.field_wave_full(rex, F, times)
    np.testing.assert_allclose(pos, np.cosh(times * rate)[:, None] * F, atol=2e-12)
    np.testing.assert_allclose(vel, (rate * np.sinh(times * rate))[:, None] * F, atol=2e-12)
    np.testing.assert_allclose(fp.field_wave_trajectory(rex, F, times), pos, atol=2e-12)
    energy = .5 * (np.sum(vel * vel, axis=1) + np.einsum("ti,it->t", pos, M @ pos.T))
    np.testing.assert_allclose(energy, .5 * value, atol=2e-12)
    assert np.linalg.norm(pos[-1]) > 1.1


@pytest.mark.parametrize("kind", ["identity", "diagonal", "full"])
def test_native_path_never_densifies_or_eigensolves(indefinite, monkeypatch, kind):
    rex, M = indefinite
    W, dense_W = metric(M.shape[0], kind)
    F = np.arange(M.shape[0], dtype=float)
    expected = la.expm(-.2 * la.solve(dense_W, M.toarray())) @ F
    def forbidden(*args, **kw):
        pytest.fail("native field evolution reached a dense/eigen operation")
    for module in (np.linalg, la):
        for name in ("eigh", "eigvalsh", "eig", "cholesky", "solve"):
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(sp.linalg, "eigsh", forbidden)
    for cls in (sp.csr_matrix, sp.csc_matrix, sp.csr_array, sp.csc_array):
        monkeypatch.setattr(cls, "toarray", forbidden)
        monkeypatch.setattr(cls, "todense", forbidden)
    np.testing.assert_allclose(fp.field_heat(rex, F, .2, W=W), expected, atol=1e-10)
    assert np.isfinite(fp.field_wave_full(rex, F, [-.2, 0., .2], W=W)[0]).all()


@pytest.mark.parametrize("kind", ["identity", "diagonal", "full"])
def test_complex_tensor_trajectories_keep_all_component_axes(indefinite, kind):
    rex, M = indefinite
    n = M.shape[0]
    W, dense_W = metric(n, kind)
    F = np.arange(n * 6).reshape(n, 2, 3) * (1. + .5j)
    V = F * .02j
    times = [0., .2, -.3]
    heat = fp.field_heat_trajectory(rex, F, times, W=W)
    pos, vel = fp.field_wave_full(rex, F, times, W=W, velocity=V)
    assert heat.shape == pos.shape == vel.shape == (3, n, 2, 3)
    K = la.solve(dense_W, M.toarray())
    np.testing.assert_allclose(heat[1].reshape(n, 6), la.expm(-.2 * K) @ F.reshape(n, 6), atol=2e-10)
    np.testing.assert_array_equal(pos[0], F)
    np.testing.assert_array_equal(vel[0], V)
    for j in range(2):
        for k in range(3):
            p, v = fp.field_wave_full(rex, F[:, j, k], times, W=W, velocity=V[:, j, k])
            np.testing.assert_allclose(pos[:, :, j, k], p, atol=2e-11)
            np.testing.assert_allclose(vel[:, :, j, k], v, atol=2e-11)


def test_initial_velocity_and_kernel_drift_reach_the_graph_wrapper():
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    F = np.zeros(3)
    V = np.ones(3)
    assert np.array_equal(fp.assemble_field_operator(rex) @ V, np.zeros(3))
    times = np.array([-2., 0., 3.])
    pos, vel = rex.field_wave_evolve(F, V, times)
    np.testing.assert_allclose(pos, times[:, None] * V, atol=2e-12)
    np.testing.assert_allclose(vel, np.broadcast_to(V, pos.shape), atol=2e-12)


def test_empty_times_zero_operator_and_empty_cell_space():
    rex = SimpleNamespace(nE=3, nF_hodge=0, w_E=None)
    F = np.arange(6).reshape(3, 2)
    M = sp.csr_matrix((3, 3))
    assert fp.field_heat_trajectory(rex, F, [], M=M).shape == (0, 3, 2)
    assert fp.field_wave_full(rex, F, [], M=M)[0].shape == (0, 3, 2)
    pos, vel = fp.field_wave_full(rex, F, [-1., 0., 2.], M=M, velocity=F)
    np.testing.assert_array_equal(pos, np.array([0., 1., 3.])[:, None, None] * F)
    np.testing.assert_array_equal(vel, np.broadcast_to(F, pos.shape))
    empty = SimpleNamespace(nE=0, nF_hodge=0, w_E=None)
    assert fp.field_heat(empty, np.zeros((0, 2)), 1., M=sp.csr_matrix((0, 0))).shape == (0, 2)


@pytest.mark.parametrize("bad", [True, 1, 2.5, -3, 0])
def test_invalid_orders_are_not_coerced(bad):
    rex = SimpleNamespace(nE=2, nF_hodge=0, w_E=None)
    with pytest.raises(ValueError, match="order"):
        fp.field_wave(rex, np.ones(2), .1, M=sp.eye(2), order=bad)


@pytest.mark.parametrize("bad", [True, np.nan, np.inf, 1j, [1.], "1"])
def test_invalid_times_are_not_coerced(bad):
    rex = SimpleNamespace(nE=2, nF_hodge=0, w_E=None)
    with pytest.raises(ValueError, match="time"):
        fp.field_heat(rex, np.ones(2), bad, M=sp.eye(2))


@pytest.mark.parametrize("W", [[1., 0.], [1., -1.], [1., np.nan], [1.],
                               [[1., 2.], [2., 1.]], [[1., .1], [0., 1.]],
                               [[1., 1.], [1., 1.]], np.eye(3), [1j, 1j]])
def test_invalid_metrics_are_refused(W):
    rex = SimpleNamespace(nE=2, nF_hodge=0, w_E=None)
    with pytest.raises(ValueError, match="metric"):
        fp.field_heat(rex, np.ones(2), .1, M=sp.eye(2), W=W)


def test_near_unit_weights_are_not_replaced_by_identity():
    rex = SimpleNamespace(nE=2, nF_hodge=0, w_E=np.array([1., 1.000001]))
    kind, d = fp.field_metric(rex)
    assert kind == "diag" and d[1] == 1.000001
    expected = np.exp(-1. / d)
    np.testing.assert_allclose(fp.field_heat(rex, np.ones(2), 1., M=sp.eye(2)), expected, atol=1e-14)
    rex.w_E = np.array([1., 0.])
    with pytest.raises(ValueError, match="positive"):
        fp.field_metric(rex)


def test_large_negative_mode_overflow_is_reported():
    rex = SimpleNamespace(nE=1, nF_hodge=0, w_E=None)
    with pytest.raises(FloatingPointError):
        fp.field_wave(rex, np.ones(1), 1000., M=-sp.eye(1))


def test_full_metric_factor_bound_and_conjugation():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(9, 9))
    W = A @ A.T + np.eye(9)
    factor = fp._FactoredMetric(sp.csr_matrix(W))
    X = rng.normal(size=(9, 3))
    np.testing.assert_allclose(factor.decode(factor.encode(X)), X, atol=1e-12)
    assert factor.inverse_norm_squared_bound >= 1 / la.eigvalsh(W)[0]
    M = sp.diags(np.arange(9) - 4., format="csr")
    S = factor.action(M, np.eye(9))  # explicit materialization is test only
    np.testing.assert_allclose(S, S.T, atol=1e-12)
    assert max(abs(la.eigvalsh(S))) <= fp._spg._gershgorin_bound(M) * factor.inverse_norm_squared_bound


@pytest.mark.parametrize("kind", ["identity", "diagonal", "full"])
def test_long_trajectories_unsorted_and_repeated_times(indefinite, kind):
    rex, M = indefinite
    n = M.shape[0]
    W, dense_W = metric(n, kind)
    K = la.solve(dense_W, M.toarray())
    generator = np.block([[np.zeros((n, n)), np.eye(n)], [-K, np.zeros((n, n))]])
    rng = np.random.default_rng(201)
    F, V = rng.normal(size=(2, n))
    times = np.array([12., -8., 0., 3., 12., -2.])
    pos, vel = fp.field_wave_full(rex, F, times, W=W, velocity=V)
    for i, t in enumerate(times):
        expected = la.expm(t * generator) @ np.concatenate([F, V])
        np.testing.assert_allclose(pos[i], expected[:n], atol=5e-10, rtol=5e-10)
        np.testing.assert_allclose(vel[i], expected[n:], atol=5e-10, rtol=5e-10)
    heat_times = np.array([6., -1., 0., 2., 6., -.2])
    heat = fp.field_heat_trajectory(rex, F, heat_times, W=W)
    for i, t in enumerate(heat_times):
        np.testing.assert_allclose(heat[i], la.expm(-t * K) @ F, atol=5e-10, rtol=5e-10)
    np.testing.assert_array_equal(pos[0], pos[4])
    np.testing.assert_array_equal(heat[0], heat[4])


@pytest.mark.parametrize("multi", [False, True])
def test_existing_gpu_dispatch_receives_shifted_signed_operator(monkeypatch, multi):
    rex = SimpleNamespace(nE=2, nF_hodge=0, w_E=None)
    M = sp.diags([-1., 2.], format="csr")
    F = np.arange(6, dtype=float).reshape(2, 3) + 1
    calls = []
    def device_stub(L, X, coefficients, upper, order, *devices):
        calls.append(devices)
        assert fp._interval(L)[0] >= -1e-12
        scaled = (2. / upper) * L.toarray() - np.eye(2)
        # Independent Chebyshev scalar evaluation on this diagonal test operator.
        return np.polynomial.chebyshev.chebval(scaled.diagonal(), coefficients)[:, None] * X
    monkeypatch.setattr(fp._spg, "_GPU_MIN_WORK", 0)
    monkeypatch.setattr(fp._spg, "_resolve_backend", lambda _: "gpu")
    monkeypatch.setattr(fp._spg, "_multi_gpu_plan", lambda *args: [0, 1] if multi else None)
    monkeypatch.setattr(fp._spg, "_matfunc_gpu", device_stub)
    monkeypatch.setattr(fp._spg, "_matfunc_gpu_multi", device_stub)
    got = fp.field_heat(rex, F, .5, M=M)
    np.testing.assert_allclose(got, np.exp(-.5 * M.diagonal())[:, None] * F, atol=1e-12)
    assert calls == [([0, 1],)] if multi else calls == [()]
    calls.clear()
    fp.field_heat(rex, F * (1 + 1j), .5, M=M)
    assert calls == []  # The existing device primitive cannot carry complex data.


def test_gpu_failure_falls_back_to_signed_cpu_recurrence(monkeypatch):
    rex = SimpleNamespace(nE=2, nF_hodge=0, w_E=None)
    monkeypatch.setattr(fp._spg, "_GPU_MIN_WORK", 0)
    monkeypatch.setattr(fp._spg, "_resolve_backend", lambda _: "gpu")
    monkeypatch.setattr(fp._spg, "_multi_gpu_plan", lambda *args: None)
    def broken(*args):
        raise RuntimeError("device unavailable")
    monkeypatch.setattr(fp._spg, "_matfunc_gpu", broken)
    np.testing.assert_allclose(fp.field_wave(rex, np.ones(2), .5, M=sp.diags([-1., 2.])),
                               [np.cosh(.5), np.cos(.5 * np.sqrt(2))], atol=1e-12)


def test_dense_reference_preserves_negative_modes_and_tiny_amplitudes():
    from rexgraph.core import _field
    values = np.array([-1., 0., 2.])
    basis = np.eye(3)
    F = np.full(3, 1e-20)
    freqs = np.sqrt(np.maximum(values, 0.))
    times = np.array([0., .5, 2.])
    pos, vel = _field.wave_evolve_trajectory(F, values, basis, freqs, times)
    heat = _field.field_diffusion_trajectory(F, values, basis, times)
    for i, t in enumerate(times):
        np.testing.assert_allclose(pos[i], fp._wave_values(values, t, "cos") * F, atol=0, rtol=1e-14)
        np.testing.assert_allclose(vel[i], fp._wave_values(values, t, "derivative") * F, atol=0, rtol=1e-14)
        np.testing.assert_allclose(heat[i], np.exp(-t * values) * F, atol=0, rtol=1e-14)
        np.testing.assert_array_equal(_field.wave_evolve(F, values, basis, freqs, t)[0], pos[i])
        np.testing.assert_array_equal(_field.field_diffusion_spectral(F, values, basis, t), heat[i])
    tiny_values, _, _ = _field.field_eigendecomposition(np.diag([-1e-18, 1e-18]))
    assert tiny_values[0] < 0 < tiny_values[1]


def test_dense_reference_empty_dimensions_and_bad_shapes():
    from rexgraph.core import _field
    empty, basis = np.zeros(0), np.zeros((0, 0))
    assert _field.wave_evolve(empty, empty, basis, empty, 1.)[0].shape == (0,)
    assert _field.wave_evolve_trajectory(empty, empty, basis, empty, np.zeros(2))[0].shape == (2, 0)
    assert _field.field_diffusion_spectral(empty, empty, basis, 1.).shape == (0,)
    assert _field.field_diffusion_trajectory(empty, empty, basis, np.zeros(2)).shape == (2, 0)
    with pytest.raises(ValueError, match="dimensions"):
        _field.wave_energy(np.ones(2), np.ones(1), np.eye(2))
    with pytest.raises(ValueError, match="dimensions"):
        _field.wave_evolve(np.ones(2), empty, basis, empty, 1.)


def test_weighted_perturbation_energy_uses_metric_and_sparse_readers(monkeypatch):
    rex = RexGraph.from_simplicial([0, 0, 1], [1, 2, 2], [[0, 1, 2]],
                                  w_E=np.array([2., 3., 4.]))
    M = fp.assemble_field_operator(rex)
    _, W = fp.field_metric(rex)
    F = np.array([1., -.2, .4, .3])
    times = np.linspace(0., 3., 13)
    pos, vel = fp.field_wave_full(rex, F, times)
    def forbidden(*args):
        pytest.fail("perturbation analysis read an implicit dense oracle")
    for name in ("L1_down", "L1_up", "B1", "field_eigen"):
        monkeypatch.setattr(RexGraph, name, property(forbidden))
    result = rex.analyze_perturbation_field(F[:3], F[3:], times=times, mode="wave")
    np.testing.assert_allclose(result["field_trajectory"], pos, atol=1e-12)
    expected_ke = .5 * np.sum(vel * vel * W, axis=1)
    expected_pe = .5 * np.einsum("ti,it->t", pos, M @ pos.T)
    np.testing.assert_allclose(result["wave_KE"], expected_ke, atol=1e-12)
    np.testing.assert_allclose(result["wave_PE"], expected_pe, atol=1e-12)
    np.testing.assert_allclose(result["wave_total"], .5 * F @ (M @ F), atol=2e-11)
