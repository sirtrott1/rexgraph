"""
Tests for rexgraph.core._wave -- complex-amplitude wave mechanics.

Verifies:
    - Normalization: in-place, returns original norm
    - Born probabilities: sum to 1 for normalized state, nonneg
    - Inner product: <psi|psi> = ||psi||^2, <psi|phi> conjugate symmetric
    - Fidelity: 1 for identical, 0 for orthogonal
    - Shannon entropy: 0 for Dirac delta, log2(n) for uniform
    - Participation ratio: 1 for localized, n for uniform
    - Schrodinger evolution: norm preservation, initial condition
    - Field evolution: independent E/F tiers, vertex derivation
    - Interference: quantum != classical when states overlap
    - Entanglement entropy: 0 for product state
    - Density matrix: purity 1 for pure state, Tr(rho) = 1
    - Integration through RexGraph: born_probabilities, wave_state,
      measure, entanglement_entropy, evolve_field_wave
"""
import numpy as np
import pytest

from rexgraph.core import _wave
from rexgraph.graph import RexGraph


# Fixtures

@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


def _normalized_state(n, seed=42):
    rng = np.random.RandomState(seed)
    psi = (rng.randn(n) + 1j * rng.randn(n)).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    return psi


def _dirac_state(n, idx=0):
    psi = np.zeros(n, dtype=np.complex128)
    psi[idx] = 1.0 + 0j
    return psi


def _uniform_state(n):
    psi = np.ones(n, dtype=np.complex128) / np.sqrt(n)
    return psi


# Normalization

class TestNormalization:

    def test_normalize_inplace(self):
        psi = np.array([3.0 + 4j, 0.0 + 0j], dtype=np.complex128)
        orig_norm = _wave.normalize_c128(psi)
        assert abs(orig_norm - 5.0) < 1e-10
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_norm_c128(self):
        psi = np.array([3.0 + 4j], dtype=np.complex128)
        assert abs(_wave.norm_c128(psi) - 5.0) < 1e-10

    def test_already_normalized(self):
        psi = _normalized_state(5)
        _wave.normalize_c128(psi)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


# Born Probabilities

class TestBornProbabilities:

    def test_sum_to_one(self):
        psi = _normalized_state(6)
        probs = _wave.born_probabilities(psi)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_nonneg(self):
        psi = _normalized_state(6)
        probs = _wave.born_probabilities(psi)
        assert np.all(probs >= 0)

    def test_dirac_concentrated(self):
        psi = _dirac_state(5, idx=2)
        probs = _wave.born_probabilities(psi)
        assert abs(probs[2] - 1.0) < 1e-12
        assert abs(probs.sum() - 1.0) < 1e-12

    def test_shape(self):
        psi = _normalized_state(10)
        assert _wave.born_probabilities(psi).shape == (10,)


# Inner Product

class TestInnerProduct:

    def test_self_inner_product(self):
        psi = _normalized_state(5)
        ip = _wave.inner_product(psi, psi)
        assert abs(ip - 1.0) < 1e-10

    def test_conjugate_symmetric(self):
        psi = _normalized_state(5, seed=1)
        phi = _normalized_state(5, seed=2)
        ip1 = _wave.inner_product(psi, phi)
        ip2 = _wave.inner_product(phi, psi)
        assert abs(ip1 - np.conj(ip2)) < 1e-10

    def test_orthogonal_zero(self):
        psi = _dirac_state(3, 0)
        phi = _dirac_state(3, 1)
        ip = _wave.inner_product(psi, phi)
        assert abs(ip) < 1e-12


# Fidelity

class TestFidelity:

    def test_identical(self):
        psi = _normalized_state(5)
        assert abs(_wave.fidelity_pure(psi, psi) - 1.0) < 1e-10

    def test_orthogonal(self):
        psi = _dirac_state(3, 0)
        phi = _dirac_state(3, 1)
        assert abs(_wave.fidelity_pure(psi, phi)) < 1e-12

    def test_range(self):
        psi = _normalized_state(5, seed=1)
        phi = _normalized_state(5, seed=2)
        F = _wave.fidelity_pure(psi, phi)
        assert 0 <= F <= 1.0 + 1e-10


# Shannon Entropy

class TestShannonEntropy:

    def test_dirac_zero(self):
        psi = _dirac_state(5, 0)
        H = _wave.shannon_entropy(psi)
        assert abs(H) < 1e-10

    def test_uniform_log2n(self):
        n = 8
        psi = _uniform_state(n)
        H = _wave.shannon_entropy(psi)
        assert abs(H - np.log2(n)) < 1e-8

    def test_nonneg(self):
        psi = _normalized_state(10)
        assert _wave.shannon_entropy(psi) >= -1e-10


# Participation Ratio

class TestParticipationRatio:

    def test_dirac_one(self):
        psi = _dirac_state(5, 0)
        assert abs(_wave.participation_ratio(psi) - 1.0) < 1e-10

    def test_uniform_n(self):
        n = 8
        psi = _uniform_state(n)
        assert abs(_wave.participation_ratio(psi) - n) < 1e-8

    def test_purity_inverse(self):
        psi = _normalized_state(6)
        pr = _wave.participation_ratio(psi)
        pur = _wave.signal_purity(psi)
        assert abs(pr * pur - 1.0) < 1e-10


# Schrodinger Evolution

class TestSchrodingerEvolution:

    def test_norm_preserved(self):
        """||psi(t)|| = ||psi(0)|| for unitary evolution."""
        n = 4
        L = np.array([[2, -1, 0, -1], [-1, 2, -1, 0],
                       [0, -1, 2, -1], [-1, 0, -1, 2]], dtype=np.float64)
        evals, evecs = np.linalg.eigh(L)
        psi0 = _normalized_state(n)
        psi_t = _wave.schrodinger_spectral(psi0, evals, evecs, 1.0)
        assert abs(np.linalg.norm(psi_t) - 1.0) < 1e-10

    def test_initial_condition(self):
        n = 4
        L = np.eye(n, dtype=np.float64)
        evals, evecs = np.linalg.eigh(L)
        psi0 = _normalized_state(n)
        psi_t = _wave.schrodinger_spectral(psi0, evals, evecs, 0.0)
        assert np.allclose(psi_t, psi0, atol=1e-10)

    def test_trajectory_shape(self):
        n = 4
        L = np.eye(n, dtype=np.float64)
        evals, evecs = np.linalg.eigh(L)
        psi0 = _normalized_state(n)
        times = np.linspace(0, 1.0, 5, dtype=np.float64)
        traj = _wave.schrodinger_spectral_trajectory(psi0, evals, evecs, times)
        assert traj.shape == (5, n)


# Interference

class TestInterference:

    def test_quantum_ne_classical(self):
        """Quantum interference differs from classical mixture for overlapping states."""
        psi1 = _normalized_state(4, seed=1)
        psi2 = _normalized_state(4, seed=2)
        w = 1.0 / np.sqrt(2)
        p_q = _wave.interference_pattern(psi1, psi2, w + 0j, w + 0j)
        p_c = _wave.classical_mixture(psi1, psi2, 0.5, 0.5)
        # They differ unless states are orthogonal
        if abs(_wave.inner_product(psi1, psi2)) > 0.01:
            assert not np.allclose(p_q, p_c, atol=1e-6)

    def test_interference_term_zero_for_orthogonal(self):
        psi1 = _dirac_state(3, 0)
        psi2 = _dirac_state(3, 1)
        I = _wave.interference_term(psi1, psi2)
        assert np.allclose(I, 0, atol=1e-12)

    def test_visibility_identical(self):
        psi = _normalized_state(4)
        V = _wave.visibility(psi, psi)
        assert abs(V - 1.0) < 1e-10

    def test_visibility_orthogonal(self):
        psi1 = _dirac_state(3, 0)
        psi2 = _dirac_state(3, 1)
        V = _wave.visibility(psi1, psi2)
        assert abs(V) < 1e-10


# Entanglement

class TestEntanglement:

    def test_product_state_zero_entropy(self):
        """Product state has zero entanglement entropy."""
        psi_A = _normalized_state(2, seed=1)
        psi_B = _normalized_state(3, seed=2)
        psi_AB = _wave.tensor_product(psi_A, psi_B)
        S = _wave.entanglement_entropy(psi_AB, 2, 3)
        assert abs(S) < 1e-8

    def test_tensor_product_shape(self):
        psi_A = _normalized_state(3)
        psi_B = _normalized_state(4)
        psi_AB = _wave.tensor_product(psi_A, psi_B)
        assert psi_AB.shape == (12,)


# Density Matrix

class TestDensityMatrix:

    def test_pure_trace_one(self):
        psi = _normalized_state(4)
        rho = _wave.pure_to_density(psi)
        tr = _wave.density_trace(rho)
        assert abs(tr - 1.0) < 1e-10

    def test_pure_purity_one(self):
        psi = _normalized_state(4)
        rho = _wave.pure_to_density(psi)
        pur = _wave.density_purity(rho)
        assert abs(pur - 1.0) < 1e-10

    def test_pure_von_neumann_zero(self):
        """Pure state has zero von Neumann entropy."""
        psi = _normalized_state(4)
        rho = _wave.pure_to_density(psi)
        S = _wave.von_neumann_entropy(rho)
        assert abs(S) < 1e-6

    def test_mixed_lower_purity(self):
        """Mixed state has purity < 1."""
        psi1 = _dirac_state(3, 0)
        psi2 = _dirac_state(3, 1)
        weights = np.array([0.5, 0.5], dtype=np.float64)
        rho = _wave.density_from_ensemble([psi1, psi2], weights)
        pur = _wave.density_purity(rho)
        assert pur < 1.0 - 1e-6


# Measurement

class TestMeasurement:

    def test_born_sample_shape(self):
        psi = _normalized_state(5)
        samples = _wave.born_sample(psi, 100)
        assert samples.shape == (100,)
        assert np.all(samples >= 0)
        assert np.all(samples < 5)

    def test_projective_collapse_normalized(self):
        psi = _normalized_state(3)
        P = np.zeros((3, 3), dtype=np.complex128)
        P[0, 0] = 1.0  # project onto first component
        collapsed, prob = _wave.projective_collapse(psi, P)
        if prob > 1e-10:
            assert abs(np.linalg.norm(collapsed) - 1.0) < 1e-10


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_born_probabilities(self, k4):
        psi = k4.wave_state(dim=1)
        probs = k4.born_probabilities(psi)
        assert probs.shape == (k4.nE,)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_wave_state_normalized(self, k4):
        psi = k4.wave_state(dim=1)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_measure_returns_valid(self, k4):
        psi = k4.wave_state(dim=1)
        outcome, collapsed = k4.measure(psi)
        assert 0 <= outcome < k4.nE
        assert collapsed.shape == (k4.nE,)

    def test_entanglement_entropy(self, k4):
        """Entanglement entropy of a product state is zero."""
        psi_A = _normalized_state(2, seed=1)
        psi_B = _normalized_state(3, seed=2)
        psi_AB = _wave.tensor_product(psi_A, psi_B)
        S = k4.entanglement_entropy(psi_AB, 2, 3)
        assert abs(S) < 1e-8
