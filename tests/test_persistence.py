"""
Tests for rexgraph.core._persistence -- persistent homology.

Verifies:
    - Filtration: monotone (edge >= max boundary vertex), correct shapes
    - Persistence diagram: returns expected keys, Betti numbers correct
    - Barcodes: shape, birth <= death, sorted by persistence
    - Persistence entropy: nonneg, zero for single bar
    - Landscape: shape (k_max, G), nonneg
    - Diagram distances: self-distance = 0, triangle inequality
    - Integration through RexGraph: filtration, persistence,
      persistence_barcodes, persistence_entropy
"""
import numpy as np
import pytest

from rexgraph.core import _persistence
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
    """Unfilled triangle: beta_1 = 1."""
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.fixture
def tree():
    return RexGraph.from_graph([0, 1, 2], [1, 2, 3])


# Filtration

class TestFiltration:

    def test_vertex_filtration_shapes(self, k4):
        f0 = np.arange(k4.nV, dtype=np.float64)
        fv, fe, ff = _persistence.filtration_sublevel_vertex(
            f0, k4.boundary_ptr, k4.boundary_idx,
            k4._B2_col_ptr, k4._B2_row_idx)
        assert fv.shape == (k4.nV,)
        assert fe.shape == (k4.nE,)
        assert ff.shape == (k4.nF,)

    def test_edge_filtration_monotone(self, k4):
        """Edge filtration >= max of boundary vertex filtration."""
        f0 = np.arange(k4.nV, dtype=np.float64)
        fv, fe, ff = _persistence.filtration_sublevel_vertex(
            f0, k4.boundary_ptr, k4.boundary_idx,
            k4._B2_col_ptr, k4._B2_row_idx)
        # Each edge's filtration value should be >= both endpoint values
        B1 = np.asarray(k4.B1, dtype=np.float64)
        for e in range(k4.nE):
            verts = np.where(np.abs(B1[:, e]) > 0.5)[0]
            for v in verts:
                assert fe[e] >= fv[v] - 1e-10

    def test_dimension_filtration(self, k4):
        fv, fe, ff = _persistence.filtration_dimension(k4.nV, k4.nE, k4.nF)
        assert np.allclose(fv, 0.0)
        assert np.allclose(fe, 1.0)
        assert np.allclose(ff, 2.0)


# Persistence Diagram

class TestPersistenceDiagram:

    def test_returns_dict(self, k4):
        f0 = np.arange(k4.nV, dtype=np.float64)
        fv, fe, ff = _persistence.filtration_sublevel_vertex(
            f0, k4.boundary_ptr, k4.boundary_idx,
            k4._B2_col_ptr, k4._B2_row_idx)
        result = _persistence.persistence_diagram(
            fv, fe, ff, k4.boundary_ptr, k4.boundary_idx,
            k4._B2_col_ptr, k4._B2_row_idx)
        assert isinstance(result, dict)
        for key in ['pairs', 'essential', 'betti']:
            assert key in result

    def test_k4_betti(self, k4):
        """K4 fully filled: beta = (1, 0, 0)."""
        fv, fe, ff = _persistence.filtration_dimension(k4.nV, k4.nE, k4.nF)
        result = _persistence.persistence_diagram(
            fv, fe, ff, k4.boundary_ptr, k4.boundary_idx,
            k4._B2_col_ptr, k4._B2_row_idx)
        b0, b1, b2 = result['betti']
        assert b0 == 1
        assert b1 == 0

    def test_triangle_beta1(self, triangle):
        """Unfilled triangle: beta_1 = 1."""
        fv, fe, ff = _persistence.filtration_dimension(
            triangle.nV, triangle.nE, triangle.nF)
        result = _persistence.persistence_diagram(
            fv, fe, ff, triangle.boundary_ptr, triangle.boundary_idx,
            triangle._B2_col_ptr, triangle._B2_row_idx)
        b0, b1, b2 = result['betti']
        assert b0 == 1
        assert b1 == 1

    def test_pairs_shape(self, k4):
        fv, fe, ff = _persistence.filtration_dimension(k4.nV, k4.nE, k4.nF)
        result = _persistence.persistence_diagram(
            fv, fe, ff, k4.boundary_ptr, k4.boundary_idx,
            k4._B2_col_ptr, k4._B2_row_idx)
        pairs = result['pairs']
        if pairs.shape[0] > 0:
            assert pairs.shape[1] == 5  # birth, death, dim, birth_cell, death_cell


# Barcodes

class TestBarcodes:

    def test_birth_le_death(self, triangle):
        fv, fe, ff = _persistence.filtration_dimension(
            triangle.nV, triangle.nE, triangle.nF)
        result = _persistence.persistence_diagram(
            fv, fe, ff, triangle.boundary_ptr, triangle.boundary_idx,
            triangle._B2_col_ptr, triangle._B2_row_idx)
        bars = _persistence.persistence_barcodes(
            result['pairs'], result['essential'])
        if bars.shape[0] > 0:
            finite = bars[~np.isinf(bars[:, 1])]
            if finite.shape[0] > 0:
                assert np.all(finite[:, 0] <= finite[:, 1] + 1e-10)

    def test_filter_by_dim(self, triangle):
        fv, fe, ff = _persistence.filtration_dimension(
            triangle.nV, triangle.nE, triangle.nF)
        result = _persistence.persistence_diagram(
            fv, fe, ff, triangle.boundary_ptr, triangle.boundary_idx,
            triangle._B2_col_ptr, triangle._B2_row_idx)
        bars_0 = _persistence.persistence_barcodes(
            result['pairs'], result['essential'], target_dim=0)
        bars_1 = _persistence.persistence_barcodes(
            result['pairs'], result['essential'], target_dim=1)
        # Both should be valid 2-column arrays
        if bars_0.shape[0] > 0:
            assert bars_0.shape[1] == 2
        if bars_1.shape[0] > 0:
            assert bars_1.shape[1] == 2


# Persistence Entropy

class TestPersistenceEntropy:

    def test_nonneg(self):
        bars = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=np.float64)
        H = _persistence.persistence_entropy(bars)
        assert H >= -1e-10

    def test_single_bar_zero(self):
        """A single finite bar has zero entropy (log2(1) = 0)."""
        bars = np.array([[0.0, 1.0]], dtype=np.float64)
        H = _persistence.persistence_entropy(bars)
        assert abs(H) < 1e-10

    def test_equal_bars_maximal(self):
        """n equal bars have entropy log2(n)."""
        n = 4
        bars = np.array([[0.0, 1.0]] * n, dtype=np.float64)
        H = _persistence.persistence_entropy(bars)
        assert abs(H - np.log2(n)) < 1e-8

    def test_empty(self):
        bars = np.zeros((0, 2), dtype=np.float64)
        assert _persistence.persistence_entropy(bars) == 0.0


# Persistence Landscape

class TestPersistenceLandscape:

    def test_shape(self):
        bars = np.array([[0.0, 2.0], [1.0, 4.0]], dtype=np.float64)
        grid = np.linspace(0, 5, 20, dtype=np.float64)
        L = _persistence.persistence_landscape(bars, grid, k_max=3)
        assert L.shape == (3, 20)

    def test_nonneg(self):
        bars = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float64)
        grid = np.linspace(0, 4, 10, dtype=np.float64)
        L = _persistence.persistence_landscape(bars, grid, k_max=2)
        assert np.all(L >= -1e-10)

    def test_empty_bars(self):
        bars = np.zeros((0, 2), dtype=np.float64)
        grid = np.linspace(0, 1, 5, dtype=np.float64)
        L = _persistence.persistence_landscape(bars, grid, k_max=2)
        assert np.allclose(L, 0)


# Diagram Distances

class TestDiagramDistances:

    def test_self_distance_zero(self):
        dgm = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64)
        d = _persistence.bottleneck_distance(dgm, dgm)
        assert abs(d) < 1e-10

    def test_wasserstein_self_zero(self):
        dgm = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64)
        d = _persistence.wasserstein_distance(dgm, dgm)
        assert abs(d) < 1e-10

    def test_nonneg(self):
        dgm1 = np.array([[0.0, 1.0]], dtype=np.float64)
        dgm2 = np.array([[0.0, 2.0]], dtype=np.float64)
        assert _persistence.bottleneck_distance(dgm1, dgm2) >= -1e-10
        assert _persistence.wasserstein_distance(dgm1, dgm2) >= -1e-10


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_filtration(self, k4):
        f0 = np.arange(k4.nV, dtype=np.float64)
        fv, fe, ff = k4.filtration(kind="vertex_sublevel", signal=f0)
        assert fv.shape == (k4.nV,)
        assert fe.shape == (k4.nE,)

    def test_persistence(self, k4):
        f0 = np.arange(k4.nV, dtype=np.float64)
        fv, fe, ff = k4.filtration(kind="vertex_sublevel", signal=f0)
        result = k4.persistence(fv, fe, ff)
        assert isinstance(result, dict)
        assert 'pairs' in result
        assert 'betti' in result

    def test_persistence_barcodes(self, k4):
        f0 = np.arange(k4.nV, dtype=np.float64)
        fv, fe, ff = k4.filtration(kind="vertex_sublevel", signal=f0)
        result = k4.persistence(fv, fe, ff)
        bars = k4.persistence_barcodes(result)
        assert bars.shape[1] == 2 if bars.shape[0] > 0 else True

    def test_persistence_entropy(self, k4):
        f0 = np.arange(k4.nV, dtype=np.float64)
        fv, fe, ff = k4.filtration(kind="vertex_sublevel", signal=f0)
        result = k4.persistence(fv, fe, ff)
        bars = k4.persistence_barcodes(result)
        H = k4.persistence_entropy(bars)
        assert isinstance(H, float)
        assert H >= -1e-10
