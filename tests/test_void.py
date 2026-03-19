"""
Tests for rexgraph.core._void -- void spectral theory.

Verifies:
    - Triangle enumeration: correct count for K4, triangle, tree
    - Classification: realized vs void triangles match nF
    - Void boundary: B1 @ Bvoid = 0, correct shape
    - Harmonic content: eta in [0, 1], nonzero when beta_1 > 0
    - Void identity: L_up + Lvoid = Bfull @ Bfull^T
    - Void strain: S^void = 3 * n_voids (for triangles)
    - fills_beta: 1 iff eta > 0
    - build_void_complex: returns all keys
    - Integration through RexGraph: void_complex property
"""
import numpy as np
import pytest

from rexgraph.core import _void
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
def k4_partial():
    """K4 with 3 of 4 faces filled (1 void)."""
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3]], dtype=np.int32),
    )


@pytest.fixture
def triangle():
    """Unfilled triangle: 1 potential triangle, 0 realized."""
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.fixture
def tree():
    return RexGraph.from_graph([0, 1, 2], [1, 2, 3])


def _get_adj(rex):
    """Get adjacency bundle for a RexGraph."""
    return rex._adjacency_bundle


# Triangle Enumeration

class TestTriangleEnumeration:

    def test_k4_four_triangles(self, k4):
        adj_ptr, adj_idx, adj_edge = _get_adj(k4)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE)
        assert nT == 4

    def test_triangle_one_triangle(self, triangle):
        adj_ptr, adj_idx, adj_edge = _get_adj(triangle)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, triangle.nV, triangle.nE)
        assert nT == 1

    def test_tree_no_triangles(self, tree):
        adj_ptr, adj_idx, adj_edge = _get_adj(tree)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, tree.nV, tree.nE)
        assert nT == 0

    def test_tri_edges_shape(self, k4):
        adj_ptr, adj_idx, adj_edge = _get_adj(k4)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE)
        assert tri_edges.shape == (nT, 3)

    def test_edge_indices_valid(self, k4):
        adj_ptr, adj_idx, adj_edge = _get_adj(k4)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE)
        assert np.all(tri_edges >= 0)
        assert np.all(tri_edges < k4.nE)


# Classification

class TestClassification:

    def test_k4_all_realized(self, k4):
        adj_ptr, adj_idx, adj_edge = _get_adj(k4)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE)
        realized, void_idx, n_voids = _void.classify_triangles(
            k4.B2_hodge, tri_edges, nT, k4.nE)
        assert n_voids == 0
        assert realized.sum() == 4

    def test_k4_partial_one_void(self, k4_partial):
        adj_ptr, adj_idx, adj_edge = _get_adj(k4_partial)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, k4_partial.nV, k4_partial.nE)
        realized, void_idx, n_voids = _void.classify_triangles(
            k4_partial.B2_hodge, tri_edges, nT, k4_partial.nE)
        assert n_voids == 1
        assert realized.sum() == 3

    def test_unfilled_triangle_one_void(self, triangle):
        adj_ptr, adj_idx, adj_edge = _get_adj(triangle)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, triangle.nV, triangle.nE)
        realized, void_idx, n_voids = _void.classify_triangles(
            triangle.B2_hodge, tri_edges, nT, triangle.nE)
        assert n_voids == 1
        assert realized.sum() == 0


# Void Boundary

class TestVoidBoundary:

    def test_b1_bvoid_zero(self, k4_partial):
        """B1 @ Bvoid = 0 (void boundaries in ker(B1))."""
        adj_ptr, adj_idx, adj_edge = _get_adj(k4_partial)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, k4_partial.nV, k4_partial.nE)
        Bvoid, _, n_voids = _void.build_void_boundary(
            k4_partial.B1, k4_partial.B2_hodge, tri_edges, nT,
            k4_partial.nV, k4_partial.nE)
        assert n_voids == 1
        product = np.asarray(k4_partial.B1, dtype=np.float64) @ Bvoid
        assert np.max(np.abs(product)) < 1e-10

    def test_bvoid_shape(self, k4_partial):
        adj_ptr, adj_idx, adj_edge = _get_adj(k4_partial)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, k4_partial.nV, k4_partial.nE)
        Bvoid, _, n_voids = _void.build_void_boundary(
            k4_partial.B1, k4_partial.B2_hodge, tri_edges, nT,
            k4_partial.nV, k4_partial.nE)
        assert Bvoid.shape == (k4_partial.nE, n_voids)

    def test_bvoid_entries_pm1(self, k4_partial):
        """Each column has exactly 3 nonzeros, all +/-1."""
        adj_ptr, adj_idx, adj_edge = _get_adj(k4_partial)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, k4_partial.nV, k4_partial.nE)
        Bvoid, _, _ = _void.build_void_boundary(
            k4_partial.B1, k4_partial.B2_hodge, tri_edges, nT,
            k4_partial.nV, k4_partial.nE)
        for col in range(Bvoid.shape[1]):
            nonzero = np.abs(Bvoid[:, col]) > 0.5
            assert nonzero.sum() == 3
            assert np.allclose(np.abs(Bvoid[nonzero, col]), 1.0)

    def test_no_voids_returns_none(self, k4):
        adj_ptr, adj_idx, adj_edge = _get_adj(k4)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, k4.nV, k4.nE)
        Bvoid, _, n_voids = _void.build_void_boundary(
            k4.B1, k4.B2_hodge, tri_edges, nT, k4.nV, k4.nE)
        assert n_voids == 0
        assert Bvoid is None


# Harmonic Content

class TestHarmonicContent:

    def test_eta_range(self, k4_partial):
        vc = k4_partial.void_complex
        if vc['n_voids'] > 0:
            assert np.all(vc['eta'] >= -1e-10)
            assert np.all(vc['eta'] <= 1.0 + 1e-10)

    def test_unfilled_triangle_eta_one(self, triangle):
        """Unfilled triangle has beta_1 = 1 and one void with eta = 1."""
        vc = triangle.void_complex
        assert vc['n_voids'] == 1
        # The void is the only cycle and it's fully harmonic
        assert vc['eta'][0] > 0.9


# Void Identity

class TestVoidIdentity:

    def test_l_up_plus_lvoid(self, k4_partial):
        """L_up + Lvoid = Bfull @ Bfull^T."""
        adj_ptr, adj_idx, adj_edge = _get_adj(k4_partial)
        tri_edges, nT = _void.find_potential_triangles(
            adj_ptr, adj_idx, adj_edge, k4_partial.nV, k4_partial.nE)
        Bvoid, _, n_voids = _void.build_void_boundary(
            k4_partial.B1, k4_partial.B2_hodge, tri_edges, nT,
            k4_partial.nV, k4_partial.nE)
        ok, res = _void.verify_void_identity(
            k4_partial.B2_hodge, Bvoid, k4_partial.nE)
        assert ok
        assert res < 1e-10

    def test_no_voids_still_valid(self, k4):
        ok, res = _void.verify_void_identity(k4.B2_hodge, None, k4.nE)
        assert ok


# Void Strain

class TestVoidStrain:

    def test_strain_equals_3n(self, k4_partial):
        """For triangular voids, S^void = 3 * n_voids."""
        vc = k4_partial.void_complex
        if vc['n_voids'] > 0:
            assert abs(vc['void_strain'] - 3.0 * vc['n_voids']) < 1e-10

    def test_zero_when_no_voids(self, k4):
        vc = k4.void_complex
        assert vc['void_strain'] == 0.0


# Fills Beta

class TestFillsBeta:

    def test_fills_when_eta_positive(self):
        eta = np.array([0.5, 0.0, 1.0], dtype=np.float64)
        fb = _void.fills_beta(eta, 3)
        assert fb[0] == 1
        assert fb[1] == 0
        assert fb[2] == 1

    def test_all_zero_eta(self):
        eta = np.zeros(4, dtype=np.float64)
        fb = _void.fills_beta(eta, 4)
        assert np.all(fb == 0)


# build_void_complex

class TestBuildVoidComplex:

    def test_returns_all_keys(self, k4_partial):
        vc = k4_partial.void_complex
        for key in ['Bvoid', 'n_voids', 'n_potential', 'eta',
                     'chi_void', 'fills_beta', 'void_strain']:
            assert key in vc

    def test_n_potential(self, k4_partial):
        vc = k4_partial.void_complex
        assert vc['n_potential'] == 4  # K4 has 4 potential triangles

    def test_no_voids_empty(self, k4):
        vc = k4.void_complex
        assert vc['n_voids'] == 0
        assert vc['void_strain'] == 0.0


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_void_complex_property(self, k4):
        vc = k4.void_complex
        assert isinstance(vc, dict)
        assert 'n_voids' in vc

    def test_partial_k4_has_void(self, k4_partial):
        vc = k4_partial.void_complex
        assert vc['n_voids'] == 1

    def test_tree_no_voids(self, tree):
        vc = tree.void_complex
        assert vc['n_voids'] == 0
        assert vc['n_potential'] == 0
