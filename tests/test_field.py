"""
Tests for rexgraph.core._field -- cross-dimensional field dynamics on (E, F).

Verifies:
    - Field operator: symmetric, correct shape, PSD for auto coupling
    - Eigendecomposition: eigenvalues nonneg (PSD), eigenvectors orthonormal
    - Wave evolution: energy conservation, initial condition, trajectory shapes
    - Diffusion: signal decays, initial condition at t=0
    - Mode classification: labels valid, weights sum to 1
    - Vertex derivation: f_V = B1 f_E
    - Integration through RexGraph: field_operator, field_eigen, field_diffuse,
      field_wave_evolve, classify_modes, derive_vertex_state
"""
import numpy as np
import pytest

from rexgraph.core import _field
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
def filled_triangle():
    return RexGraph.from_simplicial(
        sources=np.array([0, 1, 0], dtype=np.int32),
        targets=np.array([1, 2, 2], dtype=np.int32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )


@pytest.fixture
def tree():
    return RexGraph.from_graph([0, 1, 2], [1, 2, 3])


# Field Operator Construction

class TestBuildFieldOperator:

    def test_shape(self, k4):
        RL = np.asarray(k4.RL, dtype=np.float64)
        L2 = np.asarray(k4.L2, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        M, g, is_psd = _field.build_field_operator(RL, L2, B2)
        n = k4.nE + k4.nF_hodge
        assert M.shape == (n, n)

    def test_symmetric(self, k4):
        RL = np.asarray(k4.RL, dtype=np.float64)
        L2 = np.asarray(k4.L2, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        M, _, _ = _field.build_field_operator(RL, L2, B2)
        assert np.allclose(M, M.T, atol=1e-12)

    def test_auto_coupling_psd(self, k4):
        """Auto coupling g should keep M PSD."""
        RL = np.asarray(k4.RL, dtype=np.float64)
        L2 = np.asarray(k4.L2, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        M, g, is_psd = _field.build_field_operator(RL, L2, B2)
        assert g > 0
        assert is_psd

    def test_block_structure(self, filled_triangle):
        """Top-left is RL, bottom-right is L2."""
        RL = np.asarray(filled_triangle.RL, dtype=np.float64)
        L2 = np.asarray(filled_triangle.L2, dtype=np.float64)
        B2 = np.asarray(filled_triangle.B2_hodge, dtype=np.float64)
        M, g, _ = _field.build_field_operator(RL, L2, B2)
        nE = filled_triangle.nE
        nF = filled_triangle.nF_hodge
        assert np.allclose(M[:nE, :nE], RL)
        assert np.allclose(M[nE:, nE:], L2)

    def test_no_faces(self, tree):
        """Tree with nF=0: M is just RL (nE x nE)."""
        RL = np.asarray(tree.RL, dtype=np.float64)
        L2 = np.zeros((0, 0), dtype=np.float64)
        B2 = np.asarray(tree.B2_hodge, dtype=np.float64)
        M, _, _ = _field.build_field_operator(RL, L2, B2)
        assert M.shape == (tree.nE, tree.nE)


# Eigendecomposition

class TestFieldEigen:

    def test_eigenvalues_nonneg(self, k4):
        M, _, _ = k4.field_operator
        evals, evecs, freqs = _field.field_eigendecomposition(M)
        assert np.all(evals >= -1e-10)

    def test_frequencies_nonneg(self, k4):
        M, _, _ = k4.field_operator
        _, _, freqs = _field.field_eigendecomposition(M)
        assert np.all(freqs >= 0)

    def test_eigenvectors_orthonormal(self, k4):
        M, _, _ = k4.field_operator
        _, evecs, _ = _field.field_eigendecomposition(M)
        prod = evecs.T @ evecs
        assert np.allclose(prod, np.eye(prod.shape[0]), atol=1e-10)

    def test_reconstruction(self, k4):
        """M = V diag(evals) V^T."""
        M, _, _ = k4.field_operator
        evals, evecs, _ = _field.field_eigendecomposition(M)
        reconstructed = evecs @ np.diag(evals) @ evecs.T
        assert np.allclose(M, reconstructed, atol=1e-10)


# Wave Evolution

class TestWaveEvolution:

    def test_initial_condition(self, k4):
        """At t=0, F(0) = F0."""
        M, _, _ = k4.field_operator
        evals, evecs, freqs = _field.field_eigendecomposition(M)
        n = k4.nE + k4.nF_hodge
        F0 = np.zeros(n, dtype=np.float64)
        F0[0] = 1.0
        Ft, dFdt = _field.wave_evolve(F0, evals, evecs, freqs, 0.0)
        assert np.allclose(Ft, F0, atol=1e-10)
        assert np.allclose(dFdt, 0, atol=1e-10)

    def test_energy_conservation(self, k4):
        """Total energy KE + PE is conserved."""
        M, _, _ = k4.field_operator
        evals, evecs, freqs = _field.field_eigendecomposition(M)
        n = k4.nE + k4.nF_hodge
        F0 = np.random.RandomState(42).randn(n).astype(np.float64)
        _, _, E0 = _field.wave_energy(F0, np.zeros(n, dtype=np.float64), M)
        for t in [0.5, 1.0, 5.0]:
            Ft, dFdt = _field.wave_evolve(F0, evals, evecs, freqs, t)
            _, _, Et = _field.wave_energy(Ft, dFdt, M)
            assert abs(Et - E0) < 1e-8

    def test_trajectory_shape(self, k4):
        M, _, _ = k4.field_operator
        evals, evecs, freqs = _field.field_eigendecomposition(M)
        n = k4.nE + k4.nF_hodge
        F0 = np.zeros(n, dtype=np.float64)
        F0[0] = 1.0
        times = np.linspace(0, 1.0, 10, dtype=np.float64)
        traj, vel = _field.wave_evolve_trajectory(F0, evals, evecs, freqs, times)
        assert traj.shape == (10, n)
        assert vel.shape == (10, n)


# Diffusion

class TestDiffusion:

    def test_initial_condition(self, k4):
        M, _, _ = k4.field_operator
        evals, evecs, _ = _field.field_eigendecomposition(M)
        n = k4.nE + k4.nF_hodge
        F0 = np.random.RandomState(7).randn(n).astype(np.float64)
        Ft = _field.field_diffusion_spectral(F0, evals, evecs, 0.0)
        assert np.allclose(Ft, F0, atol=1e-10)

    def test_signal_decays(self, k4):
        """Diffusion decreases signal norm."""
        M, _, _ = k4.field_operator
        evals, evecs, _ = _field.field_eigendecomposition(M)
        n = k4.nE + k4.nF_hodge
        F0 = np.zeros(n, dtype=np.float64)
        F0[0] = 1.0
        Ft = _field.field_diffusion_spectral(F0, evals, evecs, 10.0)
        assert np.linalg.norm(Ft) <= np.linalg.norm(F0) + 1e-10

    def test_trajectory_shape(self, k4):
        M, _, _ = k4.field_operator
        evals, evecs, _ = _field.field_eigendecomposition(M)
        n = k4.nE + k4.nF_hodge
        F0 = np.zeros(n, dtype=np.float64)
        F0[0] = 1.0
        times = np.linspace(0, 1.0, 5, dtype=np.float64)
        traj = _field.field_diffusion_trajectory(F0, evals, evecs, times)
        assert traj.shape == (5, n)


# Mode Classification

class TestClassifyModes:

    def test_labels_valid(self, k4):
        evals, evecs, _ = k4.field_eigen
        labels, wE, wF, n_res = _field.classify_modes(
            evals, evecs, k4.nE, k4.nF_hodge)
        assert np.all((labels >= 0) & (labels <= 2))

    def test_weights_sum_to_one(self, k4):
        evals, evecs, _ = k4.field_eigen
        _, wE, wF, _ = _field.classify_modes(
            evals, evecs, k4.nE, k4.nF_hodge)
        assert np.allclose(wE + wF, 1.0, atol=1e-10)

    def test_shapes(self, k4):
        evals, evecs, _ = k4.field_eigen
        labels, wE, wF, n_res = _field.classify_modes(
            evals, evecs, k4.nE, k4.nF_hodge)
        n = k4.nE + k4.nF_hodge
        assert labels.shape == (n,)
        assert wE.shape == (n,)


# Vertex Derivation

class TestVertexDerivation:

    def test_single_state(self, k4):
        """f_V = B1 @ F[:nE]."""
        n = k4.nE + k4.nF_hodge
        F = np.random.RandomState(42).randn(n).astype(np.float64)
        f_V = _field.derive_vertex_state(F, k4.B1, k4.nE)
        expected = np.asarray(k4.B1 @ F[:k4.nE], dtype=np.float64)
        assert np.allclose(f_V, expected, atol=1e-12)

    def test_trajectory(self, k4):
        n = k4.nE + k4.nF_hodge
        traj = np.random.RandomState(7).randn(5, n).astype(np.float64)
        traj_V = _field.derive_vertex_trajectory(traj, k4.B1, k4.nE)
        assert traj_V.shape == (5, k4.nV)
        # Check first row
        expected = np.asarray(k4.B1 @ traj[0, :k4.nE], dtype=np.float64)
        assert np.allclose(traj_V[0], expected, atol=1e-12)


# Wave Energy

class TestWaveEnergy:

    def test_ke_pe_nonneg(self, k4):
        M, _, _ = k4.field_operator
        n = k4.nE + k4.nF_hodge
        F = np.random.RandomState(42).randn(n).astype(np.float64)
        dFdt = np.random.RandomState(7).randn(n).astype(np.float64)
        ke, pe, total = _field.wave_energy(F, dFdt, M)
        assert ke >= -1e-10
        assert pe >= -1e-10
        assert abs(total - ke - pe) < 1e-12

    def test_zero_velocity_zero_ke(self, k4):
        M, _, _ = k4.field_operator
        n = k4.nE + k4.nF_hodge
        F = np.ones(n, dtype=np.float64)
        dFdt = np.zeros(n, dtype=np.float64)
        ke, pe, _ = _field.wave_energy(F, dFdt, M)
        assert abs(ke) < 1e-15


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_field_operator(self, k4):
        M, g, is_psd = k4.field_operator
        n = k4.nE + k4.nF_hodge
        assert M.shape == (n, n)
        assert is_psd

    def test_field_eigen(self, k4):
        evals, evecs, freqs = k4.field_eigen
        n = k4.nE + k4.nF_hodge
        assert evals.shape == (n,)
        assert freqs.shape == (n,)

    def test_field_diffuse(self, k4):
        n = k4.nE + k4.nF_hodge
        F0 = np.zeros(n, dtype=np.float64)
        F0[0] = 1.0
        times = np.linspace(0, 1.0, 5, dtype=np.float64)
        traj = k4.field_diffuse(F0, times)
        assert traj.shape == (5, n)

    def test_classify_modes(self, k4):
        result = k4.classify_modes()
        # Returns (labels, wE, wF, n_resonant) tuple
        assert len(result) == 4
        labels, wE, wF, n_res = result
        assert labels.shape[0] == k4.nE + k4.nF_hodge

    def test_derive_vertex_state(self, k4):
        n = k4.nE + k4.nF_hodge
        F = np.zeros(n, dtype=np.float64)
        F[0] = 1.0
        f_V = k4.derive_vertex_state(F)
        assert f_V.shape == (k4.nV,)
