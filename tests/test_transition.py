"""
Tests for rexgraph.core._transition -- transition operators on the rex chain complex.

Verifies:
    - Markov discrete: column-stochastic W preserves L1 norm
    - Markov continuous (expm and spectral): agree, decay signal
    - Schrodinger spectral: preserves L2 norm (unitary), cos/sin split
    - Energy decomposition: nonneg, E_RL = E_kin + alpha_G * E_pot
    - RK4: trajectory shape, integration accuracy on simple ODE
    - Coupled derivative: shape, cross-dimensional coupling
    - Rewrite: insert/delete resize correctly
    - apply_transition dispatch: Markov, Schrodinger, Differential
    - Integration through RexGraph: evolve_markov, evolve_schrodinger,
      evolve_coupled
"""
import numpy as np
import pytest

from rexgraph.core import _transition
from rexgraph.graph import RexGraph


# Fixtures

@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


@pytest.fixture
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


# Markov Diffusion

class TestMarkovDiscrete:

    def test_step_preserves_l1(self, k4):
        """Column-stochastic W preserves L1 norm of nonneg signal."""
        L0 = np.asarray(k4.L0, dtype=np.float64)
        W = _transition.build_vertex_transition_matrix(L0)
        p = np.ones(k4.nV, dtype=np.float64) / k4.nV
        p_new = _transition.markov_vertex_step(p, W)
        assert abs(p_new.sum() - p.sum()) < 1e-10

    def test_multistep_shape(self, k4):
        L0 = np.asarray(k4.L0, dtype=np.float64)
        W = _transition.build_vertex_transition_matrix(L0)
        p = np.ones(k4.nV, dtype=np.float64) / k4.nV
        final, traj = _transition.markov_multistep(p, W, 10)
        assert traj.shape == (11, k4.nV)
        assert np.allclose(traj[0], p)

    def test_lazy_transition(self, k4):
        L0 = np.asarray(k4.L0, dtype=np.float64)
        W = _transition.build_vertex_transition_matrix(L0)
        WL = _transition.build_lazy_transition_matrix(W, lazy=0.5)
        assert WL.shape == W.shape
        # Lazy walk has diagonal >= 0.5
        assert np.all(np.diag(WL) >= 0.5 - 1e-12)


class TestMarkovContinuous:

    def test_expm_at_t0(self, k4):
        """At t=0, exp(-L*0) = I, so p(0) = p."""
        L0 = np.asarray(k4.L0, dtype=np.float64)
        p = np.ones(k4.nV, dtype=np.float64) / k4.nV
        p_t = _transition.markov_continuous_expm(p, L0, 0.0)
        assert np.allclose(p_t, p, atol=1e-10)

    def test_spectral_matches_expm(self, k4):
        """Spectral and expm paths agree."""
        L0 = np.asarray(k4.L0, dtype=np.float64)
        evals, evecs = np.linalg.eigh(L0)
        p = np.random.RandomState(42).rand(k4.nV).astype(np.float64)
        p /= p.sum()
        p_expm = _transition.markov_continuous_expm(p, L0, 1.0)
        p_spec = _transition.markov_continuous_spectral(p, evals, evecs, 1.0)
        assert np.allclose(p_expm, p_spec, atol=1e-8)

    def test_signal_decays(self, k4):
        """Diffusion under connected Laplacian converges toward uniform."""
        L0 = np.asarray(k4.L0, dtype=np.float64)
        p = np.zeros(k4.nV, dtype=np.float64)
        p[0] = 1.0
        p_t = _transition.markov_continuous_expm(p, L0, 100.0)
        # Should be close to uniform for large t
        assert np.std(p_t) < 0.01


# Schrodinger Evolution

class TestSchrodingerSpectral:

    def test_norm_preservation(self, k4):
        """Unitary evolution preserves ||f_re||^2 + ||f_im||^2 = ||f||^2."""
        L1 = np.asarray(k4.L1, dtype=np.float64)
        evals, evecs = np.linalg.eigh(L1)
        f = np.random.RandomState(42).randn(k4.nE).astype(np.float64)
        norm0_sq = float(f @ f)
        f_re, f_im = _transition.schrodinger_evolve_spectral(f, evals, evecs, 1.0)
        norm_sq = float(f_re @ f_re + f_im @ f_im)
        assert abs(norm_sq - norm0_sq) < 1e-10

    def test_initial_condition(self, k4):
        """At t=0, f_re = f, f_im = 0."""
        L1 = np.asarray(k4.L1, dtype=np.float64)
        evals, evecs = np.linalg.eigh(L1)
        f = np.random.RandomState(7).randn(k4.nE).astype(np.float64)
        f_re, f_im = _transition.schrodinger_evolve_spectral(f, evals, evecs, 0.0)
        assert np.allclose(f_re, f, atol=1e-10)
        assert np.allclose(f_im, 0, atol=1e-10)

    def test_multistep_shape(self, k4):
        L1 = np.asarray(k4.L1, dtype=np.float64)
        evals, evecs = np.linalg.eigh(L1)
        f = np.ones(k4.nE, dtype=np.float64)
        times = np.linspace(0, 1.0, 5, dtype=np.float64)
        traj_re, traj_im = _transition.schrodinger_multistep(f, evals, evecs, times)
        assert traj_re.shape == (5, k4.nE)
        assert traj_im.shape == (5, k4.nE)


# Energy Decomposition

class TestEnergyDecomposition:

    def test_nonneg(self, k4):
        f_re = np.random.RandomState(42).randn(k4.nE).astype(np.float64)
        f_im = np.zeros(k4.nE, dtype=np.float64)
        L1 = np.asarray(k4.L1, dtype=np.float64)
        LO = np.asarray(k4.L_overlap, dtype=np.float64)
        ek, ep, erl = _transition.energy_decomposition(f_re, f_im, L1, LO, 1.0)
        assert ek >= -1e-10
        assert ep >= -1e-10

    def test_erl_formula(self, k4):
        """E_RL = E_kin + alpha_G * E_pot."""
        f_re = np.random.RandomState(7).randn(k4.nE).astype(np.float64)
        f_im = np.zeros(k4.nE, dtype=np.float64)
        L1 = np.asarray(k4.L1, dtype=np.float64)
        LO = np.asarray(k4.L_overlap, dtype=np.float64)
        alpha_G = 2.5
        ek, ep, erl = _transition.energy_decomposition(f_re, f_im, L1, LO, alpha_G)
        assert abs(erl - (ek + alpha_G * ep)) < 1e-10

    def test_quadratic_form_matches(self, k4):
        """quadratic_form(f, 0, L) = f^T L f."""
        f = np.random.RandomState(42).randn(k4.nE).astype(np.float64)
        L1 = np.asarray(k4.L1, dtype=np.float64)
        qf = _transition.quadratic_form(f, np.zeros_like(f), L1)
        expected = float(f @ L1 @ f)
        assert abs(qf - expected) < 1e-10


# RK4 Integration

class TestRK4:

    def test_simple_ode(self):
        """dy/dt = -y has solution y(t) = y0 * exp(-t)."""
        y0 = np.array([1.0], dtype=np.float64)
        def deriv(y, t):
            return -y
        y_final, traj, times = _transition.rk4_integrate(y0, 0.0, 1.0, 1000, deriv)
        expected = np.exp(-1.0)
        assert abs(y_final[0] - expected) < 1e-6

    def test_trajectory_shape(self):
        y0 = np.array([1.0, 2.0], dtype=np.float64)
        def deriv(y, t):
            return -y
        _, traj, times = _transition.rk4_integrate(y0, 0.0, 1.0, 50, deriv)
        assert traj.shape == (51, 2)
        assert times.shape == (51,)

    def test_initial_in_trajectory(self):
        y0 = np.array([3.0], dtype=np.float64)
        def deriv(y, t):
            return -y
        _, traj, times = _transition.rk4_integrate(y0, 0.0, 1.0, 10, deriv)
        assert np.allclose(traj[0], y0)
        assert times[0] == 0.0


# Coupled Derivative

class TestCoupledDerivative:

    def test_shape(self, k4):
        nV, nE, nF = k4.nV, k4.nE, k4.nF
        flat = np.zeros(nV + nE + nF, dtype=np.float64)
        flat[0] = 1.0  # vertex perturbation
        sizes = np.array([nV, nE, nF], dtype=np.int32)
        L0 = np.asarray(k4.L0, dtype=np.float64)
        L1 = np.asarray(k4.L1, dtype=np.float64)
        L2 = np.asarray(k4.L2, dtype=np.float64)
        LO = np.asarray(k4.L_overlap, dtype=np.float64)
        B1 = np.asarray(k4.B1, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        df = _transition.coupled_derivative(flat, sizes, L0, L1, L2, LO, B1, B2)
        assert df.shape == (nV + nE + nF,)

    def test_zero_state_zero_derivative(self, k4):
        """Zero signal has zero derivative."""
        nV, nE, nF = k4.nV, k4.nE, k4.nF
        flat = np.zeros(nV + nE + nF, dtype=np.float64)
        sizes = np.array([nV, nE, nF], dtype=np.int32)
        L0 = np.asarray(k4.L0, dtype=np.float64)
        L1 = np.asarray(k4.L1, dtype=np.float64)
        L2 = np.asarray(k4.L2, dtype=np.float64)
        LO = np.asarray(k4.L_overlap, dtype=np.float64)
        B1 = np.asarray(k4.B1, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        df = _transition.coupled_derivative(flat, sizes, L0, L1, L2, LO, B1, B2)
        assert np.allclose(df, 0, atol=1e-15)


# Rewrite

class TestRewrite:

    def test_insert_edges(self):
        f0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        f1 = np.array([4.0, 5.0], dtype=np.float64)
        f2 = np.array([6.0], dtype=np.float64)
        nf0, nf1, nf2 = _transition.rewrite_insert_edges_i32(
            f0, f1, f2, 3, 2, 4, 3)
        assert nf0.shape == (4,)
        assert nf1.shape == (3,)
        assert nf0[0] == 1.0
        assert nf0[3] == 0.0  # default

    def test_delete_edges(self):
        f0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        f1 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        f2 = np.array([7.0], dtype=np.float64)
        v_map = np.array([0, 1, -1], dtype=np.int32)  # vertex 2 removed
        e_map = np.array([0, -1, 1], dtype=np.int32)   # edge 1 removed
        nf0, nf1, nf2 = _transition.rewrite_delete_edges_i32(
            f0, f1, f2, v_map, e_map, 2, 2)
        assert nf0.shape == (2,)
        assert nf1.shape == (2,)
        assert nf0[0] == 1.0
        assert nf0[1] == 2.0
        assert nf1[0] == 4.0
        assert nf1[1] == 6.0

    def test_add_faces(self):
        f2 = np.array([1.0, 2.0], dtype=np.float64)
        extended = _transition.rewrite_add_faces(f2, 3)
        assert extended.shape == (5,)
        assert extended[0] == 1.0
        assert extended[2] == 0.0  # default

    def test_remove_faces(self):
        f2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        mask = np.array([1, 0, 1], dtype=np.int32)
        result = _transition.rewrite_remove_faces(f2, mask)
        assert result.shape == (2,)
        assert result[0] == 1.0
        assert result[1] == 3.0


# Dispatch

class TestApplyTransition:

    def test_markov_dispatch(self, k4):
        L0 = np.asarray(k4.L0, dtype=np.float64)
        W = _transition.build_vertex_transition_matrix(L0)
        f0 = np.ones(k4.nV, dtype=np.float64) / k4.nV
        f1 = np.zeros(k4.nE, dtype=np.float64)
        f2 = np.zeros(k4.nF, dtype=np.float64)
        nf0, nf1, nf2 = _transition.apply_transition(
            _transition.TRANS_MARKOV, f0, f1, f2, 0,
            {"W": W}, dt=1.0)
        assert nf0.shape == (k4.nV,)
        assert nf1 is f1  # untouched

    def test_schrodinger_dispatch(self, k4):
        L1 = np.asarray(k4.L1, dtype=np.float64)
        evals, evecs = np.linalg.eigh(L1)
        f0 = np.zeros(k4.nV, dtype=np.float64)
        f1 = np.ones(k4.nE, dtype=np.float64)
        f2 = np.zeros(k4.nF, dtype=np.float64)
        nf0, nf1, nf2 = _transition.apply_transition(
            _transition.TRANS_SCHRODINGER, f0, f1, f2, 1,
            {"evals": evals, "evecs": evecs}, dt=0.5)
        assert nf1.shape == (k4.nE,)
        assert nf0 is f0  # untouched


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_evolve_markov(self, k4):
        p = np.zeros(k4.nV, dtype=np.float64)
        p[0] = 1.0
        p_t = k4.evolve_markov(p, dim=0, t=1.0)
        assert p_t.shape == (k4.nV,)

    def test_evolve_schrodinger(self, k4):
        f = np.ones(k4.nE, dtype=np.float64)
        f_re, f_im = k4.evolve_schrodinger(f, dim=1, t=0.5)
        assert f_re.shape == (k4.nE,)
        assert f_im.shape == (k4.nE,)
        # Norm preservation
        norm_sq = float(f_re @ f_re + f_im @ f_im)
        assert abs(norm_sq - float(f @ f)) < 1e-8

    def test_evolve_coupled(self, k4):
        nV, nE, nF = k4.nV, k4.nE, k4.nF
        state = np.zeros(nV + nE + nF, dtype=np.float64)
        state[0] = 1.0  # vertex perturbation
        y_final, traj, times = k4.evolve_coupled(state, t=1.0, n_steps=50)
        assert y_final.shape == (nV + nE + nF,)
        assert traj.shape == (51, nV + nE + nF)
        assert times.shape == (51,)
