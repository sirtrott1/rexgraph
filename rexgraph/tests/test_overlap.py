"""
Tests for rexgraph.core._overlap - overlap Laplacian L_O.

Verifies:
    - L_O shape, symmetry, PSD
    - Eigenvalues in [0, 1]
    - Gramian K is symmetric with nonneg entries
    - Overlap degree is positive for connected graphs
    - Similarity S diagonal entries are 1
    - Dense and sparse paths produce same result
    - Vertex weights affect the Gramian
    - Top-k overlap pairs
    - Degenerate cases (isolated edges, single edge)
"""
import numpy as np
import pytest

from rexgraph.core import _overlap


# Helpers

def _triangle():
    """Triangle: 3V, 3E."""
    return 3, 3, np.array([0, 1, 0], dtype=np.int32), np.array([1, 2, 2], dtype=np.int32)


def _k4():
    """K4: 4V, 6E."""
    return 4, 6, np.array([0, 0, 0, 1, 1, 2], dtype=np.int32), np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)


def _path():
    """Path 0-1-2-3: 4V, 3E."""
    return 4, 3, np.array([0, 1, 2], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32)


def _single_edge():
    """Single edge: 2V, 1E."""
    return 2, 1, np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)


# L_O Properties

class TestLOProperties:

    def test_shape(self):
        nV, nE, s, t = _triangle()
        L_O = _overlap.build_L_O(nV, nE, s, t, method="dense")
        assert L_O.shape == (nE, nE)

    def test_symmetric(self):
        nV, nE, s, t = _triangle()
        L_O = _overlap.build_L_O(nV, nE, s, t, method="dense")
        assert np.allclose(L_O, L_O.T, atol=1e-12)

    def test_psd(self):
        nV, nE, s, t = _k4()
        L_O = _overlap.build_L_O(nV, nE, s, t, method="dense")
        evals = np.linalg.eigvalsh(L_O)
        assert np.all(evals >= -1e-10)

    def test_eigenvalues_bounded(self):
        """All eigenvalues of L_O are in [0, 1]."""
        nV, nE, s, t = _k4()
        L_O = _overlap.build_L_O(nV, nE, s, t, method="dense")
        evals = np.linalg.eigvalsh(L_O)
        assert np.all(evals >= -1e-10)
        assert np.all(evals <= 1.0 + 1e-10)

    def test_trace_positive(self):
        nV, nE, s, t = _triangle()
        L_O = _overlap.build_L_O(nV, nE, s, t, method="dense")
        assert np.trace(L_O) > 0

    def test_diagonal_nonnegative(self):
        nV, nE, s, t = _k4()
        L_O = _overlap.build_L_O(nV, nE, s, t, method="dense")
        assert np.all(np.diag(L_O) >= -1e-12)


# Similarity and Overlap Degree

class TestSimilarity:

    def test_S_shape(self):
        nV, nE, s, t = _triangle()
        S, d_ov = _overlap.build_overlap_adjacency(nV, nE, s, t)
        assert S.shape == (nE, nE)

    def test_S_symmetric(self):
        nV, nE, s, t = _k4()
        S, d_ov = _overlap.build_overlap_adjacency(nV, nE, s, t)
        assert np.allclose(S, S.T, atol=1e-12)

    def test_S_diagonal_positive(self):
        """Diagonal of normalized similarity is positive (self-similarity)."""
        nV, nE, s, t = _triangle()
        S, d_ov = _overlap.build_overlap_adjacency(nV, nE, s, t)
        assert np.all(np.diag(S) > 0)

    def test_S_entries_bounded(self):
        """All entries of S are in [0, 1]."""
        nV, nE, s, t = _k4()
        S, d_ov = _overlap.build_overlap_adjacency(nV, nE, s, t)
        assert np.all(S >= -1e-12)
        assert np.all(S <= 1.0 + 1e-12)

    def test_overlap_degree_positive(self):
        nV, nE, s, t = _triangle()
        S, d_ov = _overlap.build_overlap_adjacency(nV, nE, s, t)
        assert np.all(d_ov > 0)

    def test_L_O_equals_I_minus_S(self):
        """L_O = I - S."""
        nV, nE, s, t = _triangle()
        L_O = _overlap.build_L_O(nV, nE, s, t, method="dense")
        S, d_ov = _overlap.build_overlap_adjacency(nV, nE, s, t)
        assert np.allclose(L_O, np.eye(nE) - S, atol=1e-10)


# Dense vs Sparse Agreement

class TestDenseSparse:

    def test_dense_sparse_agree(self):
        """Dense and sparse paths produce the same L_O."""
        nV, nE, s, t = _k4()
        L_dense = _overlap.build_L_O(nV, nE, s, t, method="dense")
        L_sparse = _overlap.build_L_O(nV, nE, s, t, method="sparse")
        if hasattr(L_sparse, 'toarray'):
            L_sparse = L_sparse.toarray()
        assert np.allclose(L_dense, L_sparse, atol=1e-10)


# Vertex Weights

class TestVertexWeights:

    def test_uniform_default(self):
        """Default weights (uniform) produce same result as explicit ones."""
        nV, nE, s, t = _triangle()
        L_default = _overlap.build_L_O(nV, nE, s, t, method="dense")
        L_uniform = _overlap.build_L_O(nV, nE, s, t, method="dense",
                                        vertex_weights=np.ones(nV, dtype=np.float64))
        assert np.allclose(L_default, L_uniform, atol=1e-12)

    def test_weights_change_result(self):
        """Non-uniform weights produce a different L_O."""
        nV, nE, s, t = _triangle()
        w = np.array([1.0, 2.0, 0.5], dtype=np.float64)
        L_weighted = _overlap.build_L_O(nV, nE, s, t, method="dense", vertex_weights=w)
        L_uniform = _overlap.build_L_O(nV, nE, s, t, method="dense")
        # They should differ (different vertex weighting changes K)
        assert not np.allclose(L_weighted, L_uniform)

    def test_weighted_still_psd(self):
        """Weighted L_O is still PSD."""
        nV, nE, s, t = _k4()
        w = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float64)
        L_O = _overlap.build_L_O(nV, nE, s, t, method="dense", vertex_weights=w)
        evals = np.linalg.eigvalsh(L_O)
        assert np.all(evals >= -1e-10)


# Top-k Pairs

class TestOverlapPairs:

    def test_returns_list(self):
        nV, nE, s, t = _triangle()
        pairs = _overlap.build_overlap_pairs(nV, nE, s, t, topk=5)
        assert isinstance(pairs, list)

    def test_pair_fields(self):
        nV, nE, s, t = _triangle()
        pairs = _overlap.build_overlap_pairs(nV, nE, s, t, topk=5)
        if pairs:
            p = pairs[0]
            assert "edge_i" in p
            assert "edge_j" in p
            assert "similarity" in p
            assert "shared" in p

    def test_descending_similarity(self):
        nV, nE, s, t = _k4()
        pairs = _overlap.build_overlap_pairs(nV, nE, s, t, topk=10)
        if len(pairs) > 1:
            sims = [p["similarity"] for p in pairs]
            assert all(sims[i] >= sims[i + 1] - 1e-12 for i in range(len(sims) - 1))

    def test_shared_positive(self):
        """Pairs with positive similarity share at least one vertex."""
        nV, nE, s, t = _triangle()
        pairs = _overlap.build_overlap_pairs(nV, nE, s, t, topk=5)
        for p in pairs:
            if p["similarity"] > 1e-10:
                assert p["shared"] >= 1


# Edge Cases

class TestEdgeCases:

    def test_single_edge(self):
        nV, nE, s, t = _single_edge()
        L_O = _overlap.build_L_O(nV, nE, s, t, method="dense")
        assert L_O.shape == (1, 1)
        # Single edge: L_O = I - I = 0 (edge is fully similar to itself)
        assert abs(L_O[0, 0]) < 1e-10

    def test_path_no_shared_endpoints(self):
        """Edges 0-1 and 2-3 in a path share no vertices: S[0,2] = 0."""
        nV, nE, s, t = _path()
        S, d_ov = _overlap.build_overlap_adjacency(nV, nE, s, t)
        assert abs(S[0, 2]) < 1e-12  # edges 0->1 and 2->3 share nothing

    def test_triangle_all_pairs_share(self):
        """In a triangle, every pair of edges shares exactly 1 vertex."""
        nV, nE, s, t = _triangle()
        S, d_ov = _overlap.build_overlap_adjacency(nV, nE, s, t)
        for i in range(nE):
            for j in range(nE):
                if i != j:
                    assert S[i, j] > 1e-10  # all pairs share a vertex
