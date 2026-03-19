"""
Tests for rexgraph.core._character - structural character decomposition.

RexGraph properties:
    structural_character -> chi (nE, nhats)
    vertex_character -> phi (nV, nhats)
    star_character -> chi_star (nV, nhats)
    coherence -> kappa (nV,)
    nhats -> int
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph


# Fixtures

@pytest.fixture
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32),
    )


@pytest.fixture
def tree():
    return RexGraph.from_graph([0, 1, 2], [1, 2, 3])


# Chi (structural_character)

class TestChi:

    def test_shape(self, k4):
        chi = k4.structural_character
        assert chi.shape[0] == k4.nE
        assert chi.shape[1] == k4.nhats

    def test_rows_sum_to_one(self, k4):
        chi = k4.structural_character
        row_sums = chi.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_nonnegative(self, k4):
        chi = k4.structural_character
        assert np.all(chi >= -1e-12)

    def test_k4_uniform(self, k4):
        """K4 is edge-transitive: all edges have the same chi."""
        chi = k4.structural_character
        nhats = k4.nhats
        expected = 1.0 / nhats
        assert np.allclose(chi, expected, atol=1e-8)

    def test_triangle_rows_sum_to_one(self, triangle):
        chi = triangle.structural_character
        assert np.allclose(chi.sum(axis=1), 1.0, atol=1e-10)


# Phi (vertex_character)

class TestPhi:

    def test_shape(self, k4):
        phi = k4.vertex_character
        assert phi.shape == (k4.nV, k4.nhats)

    def test_rows_sum_to_one(self, k4):
        phi = k4.vertex_character
        row_sums = phi.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-8)

    def test_nonnegative(self, k4):
        phi = k4.vertex_character
        assert np.all(phi >= -1e-6)

    def test_k4_uniform(self, k4):
        """K4 phi rows should all sum to 1 and have similar magnitudes."""
        phi = k4.vertex_character
        # All rows sum to 1
        assert np.allclose(phi.sum(axis=1), 1.0, atol=1e-8)
        # Standard deviation across vertices is small (near-uniform)
        assert np.std(phi, axis=0).max() < 0.25


# Kappa (coherence)

class TestKappa:

    def test_range(self, k4):
        kappa = k4.coherence
        assert np.all(kappa >= -1e-10)
        assert np.all(kappa <= 1.0 + 1e-10)

    def test_k4_high_coherence(self, k4):
        """K4 coherence should be positive (phi and chi_star not fully misaligned)."""
        kappa = k4.coherence
        assert np.all(kappa > 0.5)

    def test_shape(self, triangle):
        kappa = triangle.coherence
        assert kappa.shape == (triangle.nV,)


# Chi_star (star_character)

class TestChiStar:

    def test_shape(self, k4):
        cs = k4.star_character
        assert cs.shape == (k4.nV, k4.nhats)

    def test_rows_sum_to_one(self, k4):
        cs = k4.star_character
        assert np.allclose(cs.sum(axis=1), 1.0, atol=1e-10)


# Structural Summary

class TestStructuralSummary:

    def test_returns_dict(self, k4):
        from rexgraph.core._character import structural_summary
        chi = k4.structural_character
        phi = k4.vertex_character
        kappa = k4.coherence
        s = structural_summary(chi, phi, kappa, k4.nE, k4.nV, k4.nhats)
        assert isinstance(s, dict)
        assert 'mean_kappa' in s

    def test_kappa_stats(self, k4):
        from rexgraph.core._character import structural_summary
        s = structural_summary(k4.structural_character, k4.vertex_character,
                             k4.coherence, k4.nE, k4.nV, k4.nhats)
        assert 0 <= s['mean_kappa'] <= 1


# Structural Entropy

class TestStructuralEntropy:

    def test_bounded_by_ln_nhats(self, k4):
        from rexgraph.core._character import structural_entropy
        H = structural_entropy(k4.structural_character, k4.nE, k4.nhats)
        assert H >= -1e-10
        assert H <= np.log(k4.nhats) + 1e-10

    def test_k4_maximum_entropy(self, k4):
        """K4 uniform chi should give maximum entropy = ln(nhats)."""
        from rexgraph.core._character import structural_entropy
        H = structural_entropy(k4.structural_character, k4.nE, k4.nhats)
        assert abs(H - np.log(k4.nhats)) < 1e-6


# Self Response

class TestSelfResponse:

    def test_shape(self, k4):
        from rexgraph.core._character import self_response
        RLp = np.linalg.pinv(np.asarray(k4.RL, dtype=np.float64))
        rs = self_response(RLp, k4.nE)
        assert rs.shape == (k4.nE,)

    def test_nonnegative(self, k4):
        from rexgraph.core._character import self_response
        RLp = np.linalg.pinv(np.asarray(k4.RL, dtype=np.float64))
        rs = self_response(RLp, k4.nE)
        assert np.all(rs >= -1e-10)


# Mixing Time

class TestMixingTime:

    def test_positive_connected(self, k4):
        from rexgraph.core._character import mixing_time
        evals = np.linalg.eigvalsh(np.asarray(k4.RL, dtype=np.float64))
        evals = np.sort(evals)
        tau = mixing_time(evals, k4.nE)
        assert tau > 0
        assert np.isfinite(tau)


# Derived Constants

class TestDerivedConstants:

    def test_values(self):
        from rexgraph.core._character import derived_constants
        d = derived_constants(10)
        assert d['H0'] == 1.0
        assert 0 < d['amp_coeff'] < 1
        assert d['scaffold_floor'] > 0
        assert d['probe_floor'] > 0

    def test_zero_vertices(self):
        from rexgraph.core._character import derived_constants
        d = derived_constants(0)
        assert d['amp_coeff'] == 0.0
