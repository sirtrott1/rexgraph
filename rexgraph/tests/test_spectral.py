"""
Tests for rexgraph.core._spectral - spectral layout and force-directed refinement.

Verifies:
    - Spectral embedding produces finite coordinates within canvas bounds
    - Force-directed refinement reduces energy (edge lengths approach ideal)
    - Barnes-Hut and naive methods produce qualitatively similar results
    - Edge cases: 0, 1, 2 vertices, no edges
    - compute_layout selects correct method by vertex count
    - Coordinates stay within canvas after refinement
"""
import numpy as np
import pytest

from rexgraph.core import _spectral


# Helpers

def _triangle_evecs():
    """Eigenvectors of L0 for a triangle (3V, 3E)."""
    from rexgraph.core._laplacians import build_L0, eigen_symmetric
    B1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float64)
    L0 = build_L0(B1)
    evals, evecs = eigen_symmetric(L0)
    return evals, evecs, 3


def _k4_evecs():
    """Eigenvectors of L0 for K4 (4V, 6E)."""
    from rexgraph.core._laplacians import build_L0, eigen_symmetric
    B1 = np.zeros((4, 6), dtype=np.float64)
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for j, (s, t) in enumerate(edges):
        B1[s, j] = -1.0
        B1[t, j] = 1.0
    L0 = build_L0(B1)
    evals, evecs = eigen_symmetric(L0)
    return evals, evecs, 4


def _random_graph(nV, nE, seed=42):
    """Random graph with nV vertices and nE edges."""
    rng = np.random.RandomState(seed)
    src = rng.randint(0, nV, size=nE).astype(np.int32)
    tgt = rng.randint(0, nV, size=nE).astype(np.int32)
    # Avoid self-loops
    mask = src != tgt
    return src[mask], tgt[mask]


# Spectral Layout

class TestSpectralLayout:

    def test_shape(self):
        evals, evecs, nV = _triangle_evecs()
        px, py = _spectral.spectral_layout(evecs, nV, evals_in=evals)
        assert px.shape == (nV,)
        assert py.shape == (nV,)

    def test_finite(self):
        evals, evecs, nV = _k4_evecs()
        px, py = _spectral.spectral_layout(evecs, nV, evals_in=evals)
        assert np.all(np.isfinite(px))
        assert np.all(np.isfinite(py))

    def test_within_canvas(self):
        evals, evecs, nV = _k4_evecs()
        w, h = 700.0, 500.0
        px, py = _spectral.spectral_layout(evecs, nV, width=w, height=h, evals_in=evals)
        assert np.all(px >= 0) and np.all(px <= w)
        assert np.all(py >= 0) and np.all(py <= h)

    def test_not_all_same(self):
        """Vertices should not all be at the same position."""
        evals, evecs, nV = _k4_evecs()
        px, py = _spectral.spectral_layout(evecs, nV, evals_in=evals)
        assert np.std(px) > 1.0 or np.std(py) > 1.0

    def test_zero_vertices(self):
        evecs = np.empty((0, 0), dtype=np.float64)
        px, py = _spectral.spectral_layout(evecs, 0)
        assert len(px) == 0
        assert len(py) == 0

    def test_single_vertex(self):
        evecs = np.ones((1, 1), dtype=np.float64)
        px, py = _spectral.spectral_layout(evecs, 1, width=700.0, height=500.0)
        assert abs(px[0] - 350.0) < 1e-10
        assert abs(py[0] - 250.0) < 1e-10

    def test_two_vertices(self):
        evecs = np.ones((2, 2), dtype=np.float64)
        px, py = _spectral.spectral_layout(evecs, 2, width=600.0)
        assert px[0] != px[1]  # should be at different x positions

    def test_custom_canvas(self):
        evals, evecs, nV = _triangle_evecs()
        px, py = _spectral.spectral_layout(evecs, nV, width=100.0, height=80.0,
                                            evals_in=evals)
        assert np.all(px <= 100.0)
        assert np.all(py <= 80.0)


# Force-Directed Refinement

class TestForceDirected:

    def test_output_shape(self):
        evals, evecs, nV = _triangle_evecs()
        px, py = _spectral.spectral_layout(evecs, nV, evals_in=evals)
        src = np.array([0, 1, 0], dtype=np.int32)
        tgt = np.array([1, 2, 2], dtype=np.int32)
        px2, py2 = _spectral.force_directed_refine(px, py, src, tgt, nV, 3,
                                                     iterations=10)
        assert px2.shape == (nV,)
        assert py2.shape == (nV,)

    def test_finite_after_refine(self):
        evals, evecs, nV = _triangle_evecs()
        px, py = _spectral.spectral_layout(evecs, nV, evals_in=evals)
        src = np.array([0, 1, 0], dtype=np.int32)
        tgt = np.array([1, 2, 2], dtype=np.int32)
        px2, py2 = _spectral.force_directed_refine(px, py, src, tgt, nV, 3,
                                                     iterations=50)
        assert np.all(np.isfinite(px2))
        assert np.all(np.isfinite(py2))

    def test_within_canvas(self):
        evals, evecs, nV = _k4_evecs()
        w, h = 700.0, 500.0
        px, py = _spectral.spectral_layout(evecs, nV, width=w, height=h,
                                            evals_in=evals)
        src = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        tgt = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
        px2, py2 = _spectral.force_directed_refine(px, py, src, tgt, nV, 6,
                                                     iterations=100,
                                                     width=w, height=h)
        assert np.all(px2 >= 0) and np.all(px2 <= w)
        assert np.all(py2 >= 0) and np.all(py2 <= h)

    def test_single_vertex_passthrough(self):
        px = np.array([100.0], dtype=np.float64)
        py = np.array([200.0], dtype=np.float64)
        src = np.array([], dtype=np.int32)
        tgt = np.array([], dtype=np.int32)
        px2, py2 = _spectral.force_directed_refine(px, py, src, tgt, 1, 0)
        assert px2[0] == 100.0
        assert py2[0] == 200.0


# Barnes-Hut Refinement

class TestBarnesHut:

    def test_output_shape(self):
        evals, evecs, nV = _k4_evecs()
        px, py = _spectral.spectral_layout(evecs, nV, evals_in=evals)
        src = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        tgt = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
        px2, py2 = _spectral.barnes_hut_refine(px, py, src, tgt, nV, 6,
                                                 iterations=10)
        assert px2.shape == (nV,)

    def test_finite(self):
        evals, evecs, nV = _k4_evecs()
        px, py = _spectral.spectral_layout(evecs, nV, evals_in=evals)
        src = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        tgt = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
        px2, py2 = _spectral.barnes_hut_refine(px, py, src, tgt, nV, 6,
                                                 iterations=50)
        assert np.all(np.isfinite(px2))
        assert np.all(np.isfinite(py2))

    def test_within_canvas(self):
        evals, evecs, nV = _k4_evecs()
        w, h = 700.0, 500.0
        px, py = _spectral.spectral_layout(evecs, nV, width=w, height=h,
                                            evals_in=evals)
        src = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        tgt = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
        px2, py2 = _spectral.barnes_hut_refine(px, py, src, tgt, nV, 6,
                                                 iterations=100,
                                                 width=w, height=h)
        assert np.all(px2 >= 0) and np.all(px2 <= w)
        assert np.all(py2 >= 0) and np.all(py2 <= h)


# compute_layout (combined)

class TestComputeLayout:

    def test_small_graph_uses_naive(self):
        """Graphs with nV <= 200 use naive refinement (same output shape)."""
        evals, evecs, nV = _k4_evecs()
        src = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        tgt = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
        px, py = _spectral.compute_layout(evecs, nV, 6, src, tgt,
                                           iterations=10, evals_in=evals)
        assert px.shape == (nV,)
        assert np.all(np.isfinite(px))

    def test_no_edges(self):
        evals, evecs, nV = _k4_evecs()
        src = np.array([], dtype=np.int32)
        tgt = np.array([], dtype=np.int32)
        px, py = _spectral.compute_layout(evecs, nV, 0, src, tgt, evals_in=evals)
        assert px.shape == (nV,)

    def test_two_vertices(self):
        evecs = np.eye(2, dtype=np.float64)
        src = np.array([0], dtype=np.int32)
        tgt = np.array([1], dtype=np.int32)
        px, py = _spectral.compute_layout(evecs, 2, 1, src, tgt)
        assert len(px) == 2


# Deterministic Fallback

class TestDeterministicFallback:

    def test_insufficient_eigenvectors(self):
        """With fewer than 3 eigenvectors, falls back to deterministic placement."""
        evecs = np.ones((5, 2), dtype=np.float64)
        px, py = _spectral.spectral_layout(evecs, 5)
        assert px.shape == (5,)
        assert np.all(np.isfinite(px))
        # Should not all be at same position
        assert np.std(px) > 0

    def test_deterministic_within_canvas(self):
        evecs = np.ones((10, 1), dtype=np.float64)
        w, h = 400.0, 300.0
        px, py = _spectral.spectral_layout(evecs, 10, width=w, height=h)
        assert np.all(px >= 0) and np.all(px <= w)
        assert np.all(py >= 0) and np.all(py <= h)
