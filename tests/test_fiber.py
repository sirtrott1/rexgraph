"""
Tests for rexgraph.core._fiber -- fiber character and similarity complex.

Verifies:
    - chi_cosine: symmetric, diagonal 1, values in [0, 1] for nonneg chi
    - phi_cosine: symmetric, diagonal 1
    - phi_similarity_score: 1 for identical vectors, 0 for maximally different
    - phi_similarity_matrix: symmetric, diagonal 1, values in [0, 1]
    - sfb_similarity_matrix: symmetric, values in [0, 1], zero diagonal
    - threshold_graph: correct edge count, weights above threshold
    - similarity_complex: valid chain complex with Betti numbers
    - signal_sphere_proj: correct shape, nhats=3 maps to 2D plane
    - Integration through RexGraph: phi_similarity, fiber_similarity
"""
import numpy as np
import pytest

from rexgraph.core import _fiber
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


# Chi Cosine

class TestChiCosine:

    def test_symmetric(self, k4):
        chi = k4.structural_character
        sim = _fiber.chi_cosine(chi, k4.nE, k4.nhats)
        assert np.allclose(sim, sim.T)

    def test_diagonal_one(self, k4):
        chi = k4.structural_character
        sim = _fiber.chi_cosine(chi, k4.nE, k4.nhats)
        assert np.allclose(np.diag(sim), 1.0, atol=1e-10)

    def test_shape(self, k4):
        chi = k4.structural_character
        sim = _fiber.chi_cosine(chi, k4.nE, k4.nhats)
        assert sim.shape == (k4.nE, k4.nE)

    def test_nonneg_for_nonneg_chi(self, k4):
        """chi vectors are nonneg, so cosine is in [0, 1]."""
        chi = k4.structural_character
        sim = _fiber.chi_cosine(chi, k4.nE, k4.nhats)
        assert np.all(sim >= -1e-10)

    def test_k4_uniform_all_ones(self, k4):
        """K4 has uniform chi: all cosine similarities are 1."""
        chi = k4.structural_character
        sim = _fiber.chi_cosine(chi, k4.nE, k4.nhats)
        assert np.allclose(sim, 1.0, atol=1e-8)


# Phi Cosine

class TestPhiCosine:

    def test_symmetric(self, k4):
        phi = k4.vertex_character
        sim = _fiber.phi_cosine(phi, k4.nV, k4.nhats)
        assert np.allclose(sim, sim.T)

    def test_diagonal_one(self, k4):
        phi = k4.vertex_character
        sim = _fiber.phi_cosine(phi, k4.nV, k4.nhats)
        assert np.allclose(np.diag(sim), 1.0, atol=1e-10)

    def test_shape(self, k4):
        phi = k4.vertex_character
        sim = _fiber.phi_cosine(phi, k4.nV, k4.nhats)
        assert sim.shape == (k4.nV, k4.nV)


# Phi Similarity Score

class TestPhiSimilarityScore:

    def test_identical_returns_one(self):
        phi = np.array([0.3, 0.3, 0.4], dtype=np.float64)
        assert abs(_fiber.phi_similarity_score(phi, phi, 3) - 1.0) < 1e-12

    def test_maximally_different(self):
        """Two Dirac deltas on different channels have similarity 0."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        score = _fiber.phi_similarity_score(a, b, 3)
        assert abs(score) < 1e-12

    def test_range(self):
        rng = np.random.RandomState(42)
        for _ in range(10):
            a = rng.dirichlet([1, 1, 1]).astype(np.float64)
            b = rng.dirichlet([1, 1, 1]).astype(np.float64)
            score = _fiber.phi_similarity_score(a, b, 3)
            assert -1e-10 <= score <= 1.0 + 1e-10


# Phi Similarity Matrix

class TestPhiSimilarityMatrix:

    def test_symmetric(self, k4):
        sim = _fiber.phi_similarity_matrix(k4.vertex_character, k4.nV, k4.nhats)
        assert np.allclose(sim, sim.T)

    def test_diagonal_one(self, k4):
        sim = _fiber.phi_similarity_matrix(k4.vertex_character, k4.nV, k4.nhats)
        assert np.allclose(np.diag(sim), 1.0, atol=1e-10)

    def test_shape(self, k4):
        sim = _fiber.phi_similarity_matrix(k4.vertex_character, k4.nV, k4.nhats)
        assert sim.shape == (k4.nV, k4.nV)

    def test_values_in_range(self, k4):
        sim = _fiber.phi_similarity_matrix(k4.vertex_character, k4.nV, k4.nhats)
        assert np.all(sim >= -1e-10)
        assert np.all(sim <= 1.0 + 1e-10)


# SFB Similarity Matrix

class TestSFBSimilarity:

    def test_symmetric(self, k4):
        sfb = _fiber.sfb_similarity_matrix(
            k4.star_character, k4.vertex_character, k4.nV, k4.nhats)
        assert np.allclose(sfb, sfb.T)

    def test_diagonal_zero(self, k4):
        """sfb only fills off-diagonal (i < j loop), diagonal stays 0."""
        sfb = _fiber.sfb_similarity_matrix(
            k4.star_character, k4.vertex_character, k4.nV, k4.nhats)
        assert np.allclose(np.diag(sfb), 0)

    def test_nonneg(self, k4):
        """sfb values are nonneg (cosine clamped to 0, phi_sim in [0,1])."""
        sfb = _fiber.sfb_similarity_matrix(
            k4.star_character, k4.vertex_character, k4.nV, k4.nhats)
        assert np.all(sfb >= -1e-10)

    def test_shape(self, k4):
        sfb = _fiber.sfb_similarity_matrix(
            k4.star_character, k4.vertex_character, k4.nV, k4.nhats)
        assert sfb.shape == (k4.nV, k4.nV)


# Threshold Graph

class TestThresholdGraph:

    def test_all_above(self):
        sim = np.ones((3, 3), dtype=np.float64)
        src, tgt, wt, ne = _fiber.threshold_graph(sim, 3, 0.5)
        assert ne == 3  # 3 upper-triangular pairs

    def test_none_above(self):
        sim = np.zeros((3, 3), dtype=np.float64)
        src, tgt, wt, ne = _fiber.threshold_graph(sim, 3, 0.5)
        assert ne == 0

    def test_weights_above_threshold(self):
        sim = np.array([[1, 0.8, 0.3], [0.8, 1, 0.6], [0.3, 0.6, 1]],
                        dtype=np.float64)
        src, tgt, wt, ne = _fiber.threshold_graph(sim, 3, 0.5)
        assert np.all(wt > 0.5)
        assert ne == 2  # (0,1)=0.8, (1,2)=0.6

    def test_edge_indices_valid(self):
        sim = np.ones((4, 4), dtype=np.float64)
        src, tgt, wt, ne = _fiber.threshold_graph(sim, 4, 0.5)
        assert np.all(src < tgt)
        assert np.all(src >= 0)
        assert np.all(tgt < 4)


# Similarity Complex

class TestSimilarityComplex:

    def test_returns_dict(self, k4):
        chi = k4.structural_character
        sim = _fiber.chi_cosine(chi, k4.nE, k4.nhats)
        result = _fiber.similarity_complex(sim, k4.nE, 0.5)
        assert isinstance(result, dict)
        assert 'nV' in result
        assert 'beta' in result

    def test_empty_at_high_threshold(self, triangle):
        """Very high threshold produces no edges."""
        chi = triangle.structural_character
        sim = _fiber.chi_cosine(chi, triangle.nE, triangle.nhats)
        result = _fiber.similarity_complex(sim, triangle.nE, 2.0)
        assert result['n_edges'] == 0


# Sphere Projection

class TestSphereProj:

    def test_shape_nhats3(self):
        chi = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]], dtype=np.float64)
        pts = _fiber.signal_sphere_proj(chi, 2, 3)
        assert pts.shape == (2, 3)

    def test_z_zero_nhats3(self):
        """nhats=3 maps to the xy-plane (z=0)."""
        chi = np.array([[0.5, 0.3, 0.2]], dtype=np.float64)
        pts = _fiber.signal_sphere_proj(chi, 1, 3)
        assert pts[0, 2] == 0.0

    def test_shape_nhats4(self):
        chi = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float64)
        pts = _fiber.signal_sphere_proj(chi, 1, 4)
        assert pts.shape == (1, 3)

    def test_nhats4_z_nonzero(self):
        """nhats=4 with nonzero fourth component lifts into z."""
        chi = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        pts = _fiber.signal_sphere_proj(chi, 1, 4)
        assert pts[0, 2] > 0

    def test_uniform_nhats3_centroid(self):
        """Uniform chi = (1/3, 1/3, 1/3) maps to centroid of triangle."""
        chi = np.array([[1.0 / 3, 1.0 / 3, 1.0 / 3]], dtype=np.float64)
        pts = _fiber.signal_sphere_proj(chi, 1, 3)
        # Centroid of (0,0), (1,0), (0.5, sqrt(3)/2) = (0.5, sqrt(3)/6)
        assert abs(pts[0, 0] - 0.5) < 1e-10
        assert abs(pts[0, 1] - np.sqrt(3) / 6.0) < 1e-10


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_phi_similarity_shape(self, k4):
        sim = k4.phi_similarity
        assert sim.shape == (k4.nV, k4.nV)

    def test_phi_similarity_symmetric(self, k4):
        sim = k4.phi_similarity
        assert np.allclose(sim, sim.T)

    def test_phi_similarity_diagonal_one(self, k4):
        sim = k4.phi_similarity
        assert np.allclose(np.diag(sim), 1.0, atol=1e-10)

    def test_fiber_similarity_shape(self, k4):
        sim = k4.fiber_similarity
        assert sim.shape == (k4.nV, k4.nV)

    def test_fiber_similarity_symmetric(self, k4):
        sim = k4.fiber_similarity
        assert np.allclose(sim, sim.T)

    def test_fiber_similarity_nonneg(self, k4):
        sim = k4.fiber_similarity
        assert np.all(sim >= -1e-10)
