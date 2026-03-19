"""
Tests for rexgraph.core._frustration -- frustration Laplacian L_SG.

Verifies:
    - Vertex weights: positive, decrease with degree, w = 1/log(deg+e)
    - Signed Gramian K_s: symmetric, diagonal = sum of vertex weights
    - L_SG: symmetric, PSD, row sums zero, shape matches nE
    - All-positive signs: L_SG matches manual construction
    - Negative signs: off-diagonal sign flips propagate correctly
    - frustration_rate: correct fraction per type
    - Integration through RexGraph: L_frustration property
"""
import numpy as np
import pytest

from rexgraph.core import _frustration
from rexgraph.graph import RexGraph


# Helpers

def _triangle():
    nV, nE = 3, 3
    src = np.array([0, 1, 0], dtype=np.int32)
    tgt = np.array([1, 2, 2], dtype=np.int32)
    return nV, nE, src, tgt


def _k4():
    nV, nE = 4, 6
    src = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
    tgt = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
    return nV, nE, src, tgt


def _tree():
    nV, nE = 4, 3
    src = np.array([0, 1, 2], dtype=np.int32)
    tgt = np.array([1, 2, 3], dtype=np.int32)
    return nV, nE, src, tgt


def _manual_L_SG(nV, nE, src, tgt, signs, vertex_weights):
    """Reference implementation matching Cython vertex-driven enumeration.

    Diagonal: K_s[i,i] = sum w(v) for v in boundary(i)  (unsigned)
    Off-diag: K_s[i,j] = sum w(v) * sign(i) * sign(j)   for shared v
    Then L_SG = D_{|K_off|} - K_off.
    """
    # Build v2e map
    v2e = [[] for _ in range(nV)]
    for e in range(nE):
        v2e[src[e]].append(e)
        v2e[tgt[e]].append(e)

    Ks = np.zeros((nE, nE), dtype=np.float64)
    for v in range(nV):
        wv = vertex_weights[v]
        edges = v2e[v]
        for ei in edges:
            Ks[ei, ei] += wv
        for a in range(len(edges)):
            for b in range(a + 1, len(edges)):
                ei, ej = edges[a], edges[b]
                Ks[ei, ej] += wv * signs[ei] * signs[ej]
                Ks[ej, ei] += wv * signs[ei] * signs[ej]

    Koff = Ks.copy()
    np.fill_diagonal(Koff, 0)
    L = np.diag(np.abs(Koff).sum(axis=1)) - Koff
    return L


# Fixtures

@pytest.fixture
def tri():
    return _triangle()


@pytest.fixture
def k4_data():
    return _k4()


@pytest.fixture
def k4_rex():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


# Vertex Weights

class TestVertexWeights:

    def test_positive(self, tri):
        nV, nE, src, tgt = tri
        w = _frustration.build_vertex_weights(nV, nE, src, tgt)
        assert np.all(w > 0)

    def test_shape(self, tri):
        nV, nE, src, tgt = tri
        w = _frustration.build_vertex_weights(nV, nE, src, tgt)
        assert w.shape == (nV,)

    def test_inverse_log_formula(self, tri):
        """w(v) = 1 / log(deg(v) + e) for a triangle where all degrees are 2."""
        nV, nE, src, tgt = tri
        w = _frustration.build_vertex_weights(nV, nE, src, tgt)
        expected = 1.0 / np.log(2.0 + np.e)
        assert np.allclose(w, expected, atol=1e-12)

    def test_higher_degree_lower_weight(self):
        """Hub vertex (deg 3) has lower weight than leaf (deg 1)."""
        # Star graph: 0 connected to 1,2,3
        nV, nE = 4, 3
        src = np.array([0, 0, 0], dtype=np.int32)
        tgt = np.array([1, 2, 3], dtype=np.int32)
        w = _frustration.build_vertex_weights(nV, nE, src, tgt)
        # vertex 0 has degree 3, vertices 1,2,3 have degree 1
        assert w[0] < w[1]

    def test_k4_uniform(self, k4_data):
        """K4 is vertex-transitive: all weights equal."""
        nV, nE, src, tgt = k4_data
        w = _frustration.build_vertex_weights(nV, nE, src, tgt)
        assert np.allclose(w, w[0], atol=1e-12)

    def test_i64_variant(self, tri):
        nV, nE, src, tgt = tri
        src64 = src.astype(np.int64)
        tgt64 = tgt.astype(np.int64)
        w64 = _frustration.build_vertex_weights_i64(nV, nE, src64, tgt64)
        w32 = _frustration.build_vertex_weights(nV, nE, src, tgt)
        assert np.allclose(w64, w32, atol=1e-14)


# Signed Gramian

class TestSignedGramian:

    def test_symmetric(self, tri):
        nV, nE, src, tgt = tri
        signs = np.ones(nE, dtype=np.float64)
        w = _frustration.build_vertex_weights(nV, nE, src, tgt)
        Ks = _frustration.build_signed_gramian_dense(nV, nE, src, tgt, signs, w)
        assert np.allclose(Ks, Ks.T)

    def test_shape(self, tri):
        nV, nE, src, tgt = tri
        signs = np.ones(nE, dtype=np.float64)
        w = _frustration.build_vertex_weights(nV, nE, src, tgt)
        Ks = _frustration.build_signed_gramian_dense(nV, nE, src, tgt, signs, w)
        assert Ks.shape == (nE, nE)

    def test_diagonal_is_sum_of_weights(self, tri):
        """K_s[i,i] = sum of w(v) for v in boundary(i)."""
        nV, nE, src, tgt = tri
        signs = np.ones(nE, dtype=np.float64)
        w = _frustration.build_vertex_weights(nV, nE, src, tgt)
        Ks = _frustration.build_signed_gramian_dense(nV, nE, src, tgt, signs, w)
        # Each edge connects 2 vertices, so K_s[e,e] = w[src[e]] + w[tgt[e]]
        for e in range(nE):
            expected = w[src[e]] + w[tgt[e]]
            assert abs(Ks[e, e] - expected) < 1e-12

    def test_signs_flip_offdiag(self, tri):
        """Flipping one edge sign negates its off-diagonal entries."""
        nV, nE, src, tgt = tri
        w = _frustration.build_vertex_weights(nV, nE, src, tgt)
        signs_pos = np.ones(nE, dtype=np.float64)
        signs_neg = np.ones(nE, dtype=np.float64)
        signs_neg[0] = -1.0
        Ks_pos = _frustration.build_signed_gramian_dense(nV, nE, src, tgt, signs_pos, w)
        Ks_neg = _frustration.build_signed_gramian_dense(nV, nE, src, tgt, signs_neg, w)
        # Off-diagonal entries involving edge 0 should flip sign
        for j in range(1, nE):
            if abs(Ks_pos[0, j]) > 1e-15:
                assert abs(Ks_neg[0, j] + Ks_pos[0, j]) < 1e-12
        # Diagonal should be unchanged
        assert np.allclose(np.diag(Ks_neg), np.diag(Ks_pos), atol=1e-12)


# L_SG Construction

class TestBuildLSG:

    def test_symmetric(self, tri):
        nV, nE, src, tgt = tri
        L = _frustration.build_L_SG(nV, nE, src, tgt)
        assert np.allclose(L, L.T)

    def test_psd(self, tri):
        nV, nE, src, tgt = tri
        L = _frustration.build_L_SG(nV, nE, src, tgt)
        evals = np.linalg.eigvalsh(L)
        assert np.all(evals >= -1e-10)

    def test_shape(self, k4_data):
        nV, nE, src, tgt = k4_data
        L = _frustration.build_L_SG(nV, nE, src, tgt)
        assert L.shape == (nE, nE)

    def test_row_sums_zero(self, tri):
        """L_SG = D - K_off implies row sums are zero when K_off has no sign."""
        nV, nE, src, tgt = tri
        L = _frustration.build_L_SG(nV, nE, src, tgt)
        # With all +1 signs, K_off is nonneg so L is a standard graph Laplacian
        # and row sums = 0
        row_sums = L.sum(axis=1)
        assert np.allclose(row_sums, 0, atol=1e-10)

    def test_default_signs_all_positive(self, tri):
        """No signs argument defaults to all +1."""
        nV, nE, src, tgt = tri
        L1 = _frustration.build_L_SG(nV, nE, src, tgt)
        L2 = _frustration.build_L_SG(nV, nE, src, tgt,
                                      signs=np.ones(nE, dtype=np.float64))
        assert np.allclose(L1, L2, atol=1e-14)

    def test_matches_manual(self, tri):
        """Dense L_SG matches manual vertex-driven construction."""
        nV, nE, src, tgt = tri
        signs = np.ones(nE, dtype=np.float64)
        w = _frustration.build_vertex_weights(nV, nE, src, tgt)
        L_manual = _manual_L_SG(nV, nE, src, tgt, signs, w)
        L_cython = _frustration.build_L_SG(nV, nE, src, tgt, signs=signs)
        assert np.allclose(L_cython, L_manual, atol=1e-10)

    def test_k4_psd(self, k4_data):
        nV, nE, src, tgt = k4_data
        L = _frustration.build_L_SG(nV, nE, src, tgt)
        evals = np.linalg.eigvalsh(L)
        assert np.all(evals >= -1e-10)

    def test_tree(self):
        """Tree L_SG is nonzero (edges still share vertices)."""
        nV, nE, src, tgt = _tree()
        L = _frustration.build_L_SG(nV, nE, src, tgt)
        assert np.trace(L) > 0

    def test_dense_method(self, tri):
        nV, nE, src, tgt = tri
        L = _frustration.build_L_SG(nV, nE, src, tgt, method="dense")
        assert L.shape == (nE, nE)

    def test_sparse_method(self, tri):
        nV, nE, src, tgt = tri
        L = _frustration.build_L_SG(nV, nE, src, tgt, method="sparse")
        assert L.shape == (nE, nE)


# Negative Signs

class TestNegativeSigns:

    def test_lsg_still_psd_with_mixed_signs(self, tri):
        """L_SG is PSD even with mixed +/-1 edge signs."""
        nV, nE, src, tgt = tri
        signs = np.array([1.0, -1.0, 1.0], dtype=np.float64)
        L = _frustration.build_L_SG(nV, nE, src, tgt, signs=signs)
        evals = np.linalg.eigvalsh(L)
        assert np.all(evals >= -1e-10)

    def test_all_negative_symmetric(self, tri):
        nV, nE, src, tgt = tri
        signs = -np.ones(nE, dtype=np.float64)
        L = _frustration.build_L_SG(nV, nE, src, tgt, signs=signs)
        assert np.allclose(L, L.T)

    def test_all_negative_equals_all_positive(self, tri):
        """Flipping all signs should produce the same L_SG since
        sign(i)*sign(j) = (-1)*(-1) = 1 for all pairs."""
        nV, nE, src, tgt = tri
        L_pos = _frustration.build_L_SG(nV, nE, src, tgt,
                                         signs=np.ones(nE, dtype=np.float64))
        L_neg = _frustration.build_L_SG(nV, nE, src, tgt,
                                         signs=-np.ones(nE, dtype=np.float64))
        assert np.allclose(L_pos, L_neg, atol=1e-12)


# Frustration Rate

class TestFrustrationRate:

    def test_all_positive(self):
        signs = np.ones(5, dtype=np.float64)
        types = np.zeros(5, dtype=np.int32)
        rates = _frustration.frustration_rate(signs, types, 5, 1)
        assert rates[0] == 0.0

    def test_all_negative(self):
        signs = -np.ones(5, dtype=np.float64)
        types = np.zeros(5, dtype=np.int32)
        rates = _frustration.frustration_rate(signs, types, 5, 1)
        assert rates[0] == 1.0

    def test_mixed(self):
        signs = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        types = np.zeros(4, dtype=np.int32)
        rates = _frustration.frustration_rate(signs, types, 4, 1)
        assert abs(rates[0] - 0.5) < 1e-12

    def test_per_type(self):
        """Two types: type 0 has 1/3 negative, type 1 has 1/2 negative."""
        signs = np.array([1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        types = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        rates = _frustration.frustration_rate(signs, types, 5, 2)
        assert abs(rates[0] - 1.0 / 3.0) < 1e-12
        assert abs(rates[1] - 0.5) < 1e-12

    def test_empty_type_zero(self):
        """Type with no edges gets rate 0."""
        signs = np.array([1.0, -1.0], dtype=np.float64)
        types = np.array([0, 0], dtype=np.int32)
        rates = _frustration.frustration_rate(signs, types, 2, 3)
        assert rates[1] == 0.0
        assert rates[2] == 0.0


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_l_frustration_shape(self, k4_rex):
        L = k4_rex.L_frustration
        assert L.shape == (k4_rex.nE, k4_rex.nE)

    def test_l_frustration_symmetric(self, k4_rex):
        L = k4_rex.L_frustration
        assert np.allclose(L, L.T)

    def test_l_frustration_psd(self, k4_rex):
        L = k4_rex.L_frustration
        evals = np.linalg.eigvalsh(L)
        assert np.all(evals >= -1e-10)

    def test_nonzero_trace(self, k4_rex):
        """K4 has shared vertices between edges, so L_SG is nonzero."""
        L = k4_rex.L_frustration
        assert np.trace(L) > 0

    def test_triangle_frustration(self):
        rex = RexGraph.from_graph([0, 1, 0], [1, 2, 2])
        L = rex.L_frustration
        assert L.shape == (3, 3)
        assert np.allclose(L, L.T)
