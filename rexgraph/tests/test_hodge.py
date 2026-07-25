"""
Tests for rexgraph.core._hodge - Hodge decomposition of edge signals.

Verifies:
    - g = grad + curl + harm (reconstruction)
    - Mutual orthogonality of components
    - Energy: ||g||^2 = ||grad||^2 + ||curl||^2 + ||harm||^2
    - Tree graph: curl = 0, harm = 0 (all signal is gradient)
    - Filled triangle: harm = 0 (no independent cycles)
    - Unfilled triangle: harm != 0 (one independent cycle)
    - Per-edge rho in [0, 1]
    - Signal construction from types and negative mask
    - Normalization
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph


# Fixtures

@pytest.fixture
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.fixture
def filled_triangle():
    return RexGraph.from_simplicial(
        sources=np.array([0, 1, 0], dtype=np.int32),
        targets=np.array([1, 2, 2], dtype=np.int32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )


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


# Reconstruction

class TestReconstruction:

    def test_sum_equals_original(self, filled_triangle):
        flow = np.array([1.0, -0.5, 0.3], dtype=np.float64)
        grad, curl, harm = filled_triangle.hodge(flow)
        assert np.allclose(flow, grad + curl + harm, atol=1e-10)

    def test_k4_reconstruction(self, k4):
        rng = np.random.RandomState(42)
        flow = rng.randn(k4.nE)
        grad, curl, harm = k4.hodge(flow)
        assert np.allclose(flow, grad + curl + harm, atol=1e-10)

    def test_random_signal(self, filled_triangle):
        rng = np.random.RandomState(7)
        flow = rng.randn(filled_triangle.nE)
        grad, curl, harm = filled_triangle.hodge(flow)
        assert np.allclose(flow, grad + curl + harm, atol=1e-10)


# Orthogonality

class TestOrthogonality:

    def test_components_orthogonal(self, filled_triangle):
        flow = np.array([1.0, -0.5, 0.3], dtype=np.float64)
        grad, curl, harm = filled_triangle.hodge(flow)
        assert abs(grad @ curl) < 1e-10
        assert abs(grad @ harm) < 1e-10
        assert abs(curl @ harm) < 1e-10

    def test_k4_orthogonality(self, k4):
        rng = np.random.RandomState(42)
        flow = rng.randn(k4.nE)
        h = k4.hodge_full(flow)
        orth = h['orthogonality']
        assert orth['orthogonal']
        assert orth['max_inner'] < 1e-6


# Energy

class TestEnergy:

    def test_energy_sums_to_one(self, filled_triangle):
        flow = np.array([1.0, -0.5, 0.3], dtype=np.float64)
        h = filled_triangle.hodge_full(flow)
        total = h['pct_grad'] + h['pct_curl'] + h['pct_harm']
        assert abs(total - 1.0) < 1e-10

    def test_energy_equals_norm_squared(self, k4):
        rng = np.random.RandomState(42)
        flow = rng.randn(k4.nE)
        grad, curl, harm = k4.hodge(flow)
        flow_energy = float(flow @ flow)
        component_energy = (float(grad @ grad) +
                          float(curl @ curl) +
                          float(harm @ harm))
        assert abs(flow_energy - component_energy) < 1e-10

    def test_zero_signal(self, filled_triangle):
        flow = np.zeros(filled_triangle.nE, dtype=np.float64)
        h = filled_triangle.hodge_full(flow)
        assert h['pct_grad'] == 0.0
        assert h['pct_curl'] == 0.0
        assert h['pct_harm'] == 0.0


# Tree (no cycles, no faces)

class TestTree:

    def test_all_gradient(self, tree):
        """On a tree, all signal is gradient (no curl, no harmonic)."""
        flow = np.array([1.0, -0.5, 0.3], dtype=np.float64)
        grad, curl, harm = tree.hodge(flow)
        assert np.allclose(curl, 0, atol=1e-10)
        assert np.allclose(harm, 0, atol=1e-10)
        assert np.allclose(grad, flow, atol=1e-10)

    def test_energy_all_gradient(self, tree):
        flow = np.array([1.0, -0.5, 0.3], dtype=np.float64)
        h = tree.hodge_full(flow)
        assert abs(h['pct_grad'] - 1.0) < 1e-10


# Filled vs Unfilled Triangle

class TestFilling:

    def test_filled_no_harmonic(self, filled_triangle):
        """Filled triangle: beta_1 = 0, so harmonic component is zero."""
        flow = np.array([1.0, -0.5, 0.3], dtype=np.float64)
        grad, curl, harm = filled_triangle.hodge(flow)
        assert np.allclose(harm, 0, atol=1e-10)

    def test_unfilled_has_harmonic(self, triangle):
        """Unfilled triangle: beta_1 = 1, so harmonic component is nonzero."""
        flow = np.array([1.0, 1.0, -1.0], dtype=np.float64)
        grad, curl, harm = triangle.hodge(flow)
        assert np.linalg.norm(harm) > 0.1


# Resistance Ratio

class TestRho:

    def test_rho_range(self, k4):
        rng = np.random.RandomState(42)
        flow = rng.randn(k4.nE)
        h = k4.hodge_full(flow)
        rho = h['rho']
        assert np.all(rho >= -1e-12)
        assert np.all(rho <= 1.0 + 1e-12)

    def test_rho_zero_for_gradient(self, tree):
        """On a tree, rho = 0 everywhere (no harmonic content)."""
        flow = np.array([1.0, -0.5, 0.3], dtype=np.float64)
        h = tree.hodge_full(flow)
        assert np.allclose(h['rho'], 0, atol=1e-10)


# Signal Construction

class TestSignalConstruction:

    def test_build_flow_no_types(self):
        from rexgraph.core._hodge import build_flow_signal
        w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        flow = build_flow_signal(w)
        assert np.array_equal(flow, w)

    def test_build_flow_with_negatives(self):
        from rexgraph.core._hodge import build_flow_signal
        w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        types = np.array([0, 1, 0], dtype=np.int32)
        mask = np.array([0, 1], dtype=np.uint8)  # type 1 is negative
        flow = build_flow_signal(w, types, mask)
        assert flow[0] == 1.0
        assert flow[1] == -2.0
        assert flow[2] == 3.0

    def test_normalize(self):
        from rexgraph.core._hodge import normalize_signal
        x = np.array([2.0, -4.0, 1.0], dtype=np.float64)
        normed = normalize_signal(x)
        assert abs(np.max(np.abs(normed)) - 1.0) < 1e-12

    def test_normalize_zero(self):
        from rexgraph.core._hodge import normalize_signal
        x = np.zeros(5, dtype=np.float64)
        normed = normalize_signal(x)
        assert np.allclose(normed, 0)
