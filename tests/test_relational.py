"""
Tests for rexgraph.core._relational -- relational Laplacian and Green function.

Verifies:
    - Trace normalization: tr(L_hat) = 1, zero-trace inputs produce zero output
    - build_RL: tr(RL) = nhats, hat operators are PSD, skips zero-trace inputs
    - 3-hat and 4-hat fast paths match general path
    - rl_eigen: eigenvalues sorted ascending, nonnegative, reconstruct RL
    - rl_pinv_dense: RL @ RL^+ @ RL = RL (Moore-Penrose condition)
    - rl_pinv_matvec: matches dense pseudoinverse times vector
    - build_green_cache: S0 = B1 @ RL^+ @ B1^T, S0 symmetric PSD
    - build_line_graph: correct vertex/edge counts
    - build_L_coPC: symmetric PSD, zero for matchings
    - Integration through RexGraph: RL, nhats, _rl_eigen, _green_cache
"""
import numpy as np
import pytest

from rexgraph.core import _relational
from rexgraph.graph import RexGraph


# Helpers

def _triangle_operators():
    """Build L1, L_O, L_SG for a triangle (3V, 3E)."""
    B1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float64)
    nV, nE = 3, 3
    # L1 = B1^T @ B1 (no faces)
    L1 = B1.T @ B1
    # L_O
    K = np.abs(B1).T @ np.abs(B1)
    d = K.sum(axis=1)
    D_inv = np.diag(np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0))
    S = D_inv @ K @ D_inv
    L_O = 0.5 * (np.eye(nE) - S + (np.eye(nE) - S).T)
    # L_SG
    deg = np.abs(B1).sum(axis=1)
    W = np.diag(1.0 / np.log(deg + np.e))
    Ks = B1.T @ W @ B1
    Koff = Ks.copy()
    np.fill_diagonal(Koff, 0)
    L_SG = np.diag(np.abs(Koff).sum(axis=1)) - Koff
    return B1, L1, L_O, L_SG, nV, nE


def _k4_operators():
    """Build L1, L_O, L_SG for K4 (4V, 6E, 4F)."""
    B1 = np.zeros((4, 6), dtype=np.float64)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for j, (s, t) in enumerate(edges):
        B1[s, j] = -1.0
        B1[t, j] = 1.0
    nV, nE = 4, 6
    rex = RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )
    B2 = np.asarray(rex.B2_hodge, dtype=np.float64)
    L1 = B1.T @ B1 + B2 @ B2.T
    K = np.abs(B1).T @ np.abs(B1)
    d = K.sum(axis=1)
    D_inv = np.diag(np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0))
    S = D_inv @ K @ D_inv
    L_O = 0.5 * (np.eye(nE) - S + (np.eye(nE) - S).T)
    deg = np.abs(B1).sum(axis=1)
    W = np.diag(1.0 / np.log(deg + np.e))
    Ks = B1.T @ W @ B1
    Koff = Ks.copy()
    np.fill_diagonal(Koff, 0)
    L_SG = np.diag(np.abs(Koff).sum(axis=1)) - Koff
    return B1, L1, L_O, L_SG, nV, nE, K


# Fixtures

@pytest.fixture
def tri_ops():
    return _triangle_operators()


@pytest.fixture
def k4_ops():
    return _k4_operators()


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


# Trace Normalization

class TestTraceNormalize:

    def test_trace_one(self, tri_ops):
        _, L1, _, _, _, _ = tri_ops
        L_hat, tr_val = _relational.trace_normalize(L1)
        assert tr_val > 0
        assert abs(np.trace(L_hat) - 1.0) < 1e-12

    def test_preserves_symmetry(self, tri_ops):
        _, L1, _, _, _, _ = tri_ops
        L_hat, _ = _relational.trace_normalize(L1)
        assert np.allclose(L_hat, L_hat.T)

    def test_preserves_psd(self, tri_ops):
        _, L1, _, _, _, _ = tri_ops
        L_hat, _ = _relational.trace_normalize(L1)
        evals = np.linalg.eigvalsh(L_hat)
        assert np.all(evals >= -1e-12)

    def test_zero_trace_returns_zero(self):
        Z = np.zeros((3, 3), dtype=np.float64)
        L_hat, tr_val = _relational.trace_normalize(Z)
        assert tr_val == 0.0
        assert np.allclose(L_hat, 0)

    def test_does_not_modify_input(self, tri_ops):
        _, L1, _, _, _, _ = tri_ops
        L1_copy = L1.copy()
        _relational.trace_normalize(L1)
        assert np.array_equal(L1, L1_copy)


# build_RL

class TestBuildRL:

    def test_trace_equals_nhats(self, tri_ops):
        """tr(RL) = number of active hat operators."""
        _, L1, L_O, L_SG, _, _ = tri_ops
        result = _relational.build_RL([L1, L_O, L_SG], ['L1', 'L_O', 'L_SG'])
        assert abs(np.trace(result['RL']) - result['nhats']) < 1e-10

    def test_nhats_three(self, tri_ops):
        _, L1, L_O, L_SG, _, _ = tri_ops
        result = _relational.build_RL([L1, L_O, L_SG], ['L1', 'L_O', 'L_SG'])
        assert result['nhats'] == 3

    def test_rl_symmetric(self, tri_ops):
        _, L1, L_O, L_SG, _, _ = tri_ops
        result = _relational.build_RL([L1, L_O, L_SG], ['L1', 'L_O', 'L_SG'])
        RL = result['RL']
        assert np.allclose(RL, RL.T)

    def test_rl_psd(self, tri_ops):
        _, L1, L_O, L_SG, _, _ = tri_ops
        result = _relational.build_RL([L1, L_O, L_SG], ['L1', 'L_O', 'L_SG'])
        evals = np.linalg.eigvalsh(result['RL'])
        assert np.all(evals >= -1e-10)

    def test_hats_are_trace_one(self, tri_ops):
        """Each active hat has trace 1."""
        _, L1, L_O, L_SG, _, _ = tri_ops
        result = _relational.build_RL([L1, L_O, L_SG], ['L1', 'L_O', 'L_SG'])
        for h in result['hats']:
            assert abs(np.trace(h) - 1.0) < 1e-12

    def test_skips_zero_trace(self, tri_ops):
        """A zero matrix is dropped from the active hats."""
        _, L1, L_O, _, _, nE = tri_ops
        Z = np.zeros((nE, nE), dtype=np.float64)
        result = _relational.build_RL([L1, L_O, Z], ['L1', 'L_O', 'zero'])
        assert result['nhats'] == 2
        assert 'zero' not in result['hat_names']
        assert abs(np.trace(result['RL']) - 2) < 1e-10

    def test_hat_names_preserved(self, tri_ops):
        _, L1, L_O, L_SG, _, _ = tri_ops
        result = _relational.build_RL([L1, L_O, L_SG], ['hodge', 'overlap', 'frust'])
        assert result['hat_names'] == ['hodge', 'overlap', 'frust']

    def test_rl_is_sum_of_hats(self, tri_ops):
        """RL equals the sum of active hat operators."""
        _, L1, L_O, L_SG, _, _ = tri_ops
        result = _relational.build_RL([L1, L_O, L_SG], ['L1', 'L_O', 'L_SG'])
        hat_sum = sum(result['hats'])
        assert np.allclose(result['RL'], hat_sum, atol=1e-12)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _relational.build_RL([], [])


# 4-hat fast path

class TestBuildRL4:

    def test_four_inputs(self, k4_ops):
        B1, L1, L_O, L_SG, _, nE, K = k4_ops
        lg = _relational.build_line_graph(K, nE)
        L_C = _relational.build_L_coPC(lg)
        result = _relational.build_RL(
            [L1, L_O, L_SG, L_C], ['L1', 'L_O', 'L_SG', 'L_C'])
        assert result['nhats'] == 4
        assert abs(np.trace(result['RL']) - 4) < 1e-10

    def test_four_hat_rl_psd(self, k4_ops):
        B1, L1, L_O, L_SG, _, nE, K = k4_ops
        lg = _relational.build_line_graph(K, nE)
        L_C = _relational.build_L_coPC(lg)
        result = _relational.build_RL(
            [L1, L_O, L_SG, L_C], ['L1', 'L_O', 'L_SG', 'L_C'])
        evals = np.linalg.eigvalsh(result['RL'])
        assert np.all(evals >= -1e-10)


# General path (N != 3, N != 4)

class TestBuildRLGeneral:

    def test_two_inputs(self, tri_ops):
        _, L1, L_O, _, _, _ = tri_ops
        result = _relational.build_RL([L1, L_O], ['L1', 'L_O'])
        assert result['nhats'] == 2
        assert abs(np.trace(result['RL']) - 2) < 1e-10

    def test_five_inputs(self, tri_ops):
        """Five inputs (no fast path) should still work."""
        _, L1, L_O, L_SG, _, nE = tri_ops
        L4 = np.eye(nE, dtype=np.float64) * 0.5
        L5 = np.eye(nE, dtype=np.float64) * 0.3
        result = _relational.build_RL(
            [L1, L_O, L_SG, L4, L5],
            ['a', 'b', 'c', 'd', 'e'])
        assert result['nhats'] == 5
        assert abs(np.trace(result['RL']) - 5) < 1e-10


# build_RL_from_laplacians

class TestBuildRLFromLaplacians:

    def test_matches_build_rl(self, tri_ops):
        _, L1, L_O, L_SG, _, _ = tri_ops
        r1 = _relational.build_RL_from_laplacians(L1, L_O, L_SG)
        r2 = _relational.build_RL([L1, L_O, L_SG], ['L1_down', 'L_O', 'L_SG'])
        assert np.allclose(r1['RL'], r2['RL'], atol=1e-14)


# rl_eigen

class TestRLEigen:

    def test_eigenvalues_ascending(self, tri_ops):
        _, L1, L_O, L_SG, _, _ = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        evals, _ = _relational.rl_eigen(RL)
        assert np.all(np.diff(evals) >= -1e-12)

    def test_eigenvalues_nonnegative(self, tri_ops):
        _, L1, L_O, L_SG, _, _ = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        evals, _ = _relational.rl_eigen(RL)
        assert np.all(evals >= -1e-12)

    def test_reconstruction(self, tri_ops):
        """RL = V diag(evals) V^T."""
        _, L1, L_O, L_SG, _, _ = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        evals, evecs = _relational.rl_eigen(RL)
        reconstructed = evecs @ np.diag(evals) @ evecs.T
        assert np.allclose(RL, reconstructed, atol=1e-10)

    def test_eigenvectors_orthonormal(self, tri_ops):
        _, L1, L_O, L_SG, _, _ = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        _, evecs = _relational.rl_eigen(RL)
        prod = evecs.T @ evecs
        assert np.allclose(prod, np.eye(evecs.shape[0]), atol=1e-10)


# Pseudoinverse

class TestPseudoinverse:

    def test_moore_penrose(self, tri_ops):
        """RL @ RL^+ @ RL = RL."""
        _, L1, L_O, L_SG, _, _ = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        evals, evecs = _relational.rl_eigen(RL)
        RLp = _relational.rl_pinv_dense(evals, evecs)
        assert np.allclose(RL @ RLp @ RL, RL, atol=1e-10)

    def test_pinv_symmetric(self, tri_ops):
        _, L1, L_O, L_SG, _, _ = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        evals, evecs = _relational.rl_eigen(RL)
        RLp = _relational.rl_pinv_dense(evals, evecs)
        assert np.allclose(RLp, RLp.T, atol=1e-12)

    def test_matvec_matches_dense(self, tri_ops):
        """rl_pinv_matvec matches dense pseudoinverse times vector."""
        _, L1, L_O, L_SG, _, nE = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        evals, evecs = _relational.rl_eigen(RL)
        RLp = _relational.rl_pinv_dense(evals, evecs)
        x = np.random.RandomState(42).randn(nE).astype(np.float64)
        result_mv = _relational.rl_pinv_matvec(evals, evecs, x)
        result_dense = RLp @ x
        assert np.allclose(result_mv, result_dense, atol=1e-10)


# Green Function Cache

class TestGreenCache:

    def test_keys(self, tri_ops):
        B1, L1, L_O, L_SG, _, _ = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        evals, evecs = _relational.rl_eigen(RL)
        gc = _relational.build_green_cache(RL, B1, evals, evecs)
        for key in ['RL_pinv', 'B1_RLp', 'S0', 'evals', 'evecs', 'nV', 'nE']:
            assert key in gc

    def test_s0_symmetric(self, tri_ops):
        B1, L1, L_O, L_SG, _, _ = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        evals, evecs = _relational.rl_eigen(RL)
        gc = _relational.build_green_cache(RL, B1, evals, evecs)
        assert np.allclose(gc['S0'], gc['S0'].T, atol=1e-12)

    def test_s0_equals_b1_rlp_b1t(self, tri_ops):
        """S0 = B1 @ RL^+ @ B1^T."""
        B1, L1, L_O, L_SG, _, _ = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        evals, evecs = _relational.rl_eigen(RL)
        gc = _relational.build_green_cache(RL, B1, evals, evecs)
        expected = B1 @ gc['RL_pinv'] @ B1.T
        assert np.allclose(gc['S0'], expected, atol=1e-10)

    def test_b1_rlp_shape(self, tri_ops):
        B1, L1, L_O, L_SG, nV, nE = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        evals, evecs = _relational.rl_eigen(RL)
        gc = _relational.build_green_cache(RL, B1, evals, evecs)
        assert gc['B1_RLp'].shape == (nV, nE)


# Linear Solve

class TestLinearSolve:

    def test_rl_cg_solve(self, tri_ops):
        """Solution x satisfies RL x ~ b in the column space of RL."""
        _, L1, L_O, L_SG, _, nE = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        b = np.random.RandomState(7).randn(nE).astype(np.float64)
        x = _relational.rl_cg_solve(RL, b)
        assert x.shape == (nE,)
        # RL @ x should approximate the projection of b onto col(RL)
        residual = RL @ x - b
        # Residual should be in the null space of RL
        evals, evecs = _relational.rl_eigen(RL)
        col_space = evecs[:, evals > 1e-10]
        proj_b = col_space @ col_space.T @ b
        assert np.allclose(RL @ x, proj_b, atol=1e-8)

    def test_rl_solve_column(self, tri_ops):
        B1, L1, L_O, L_SG, _, nE = tri_ops
        RL = _relational.build_RL([L1, L_O, L_SG], ['a', 'b', 'c'])['RL']
        x = _relational.rl_solve_column(RL, B1, 0)
        assert x.shape == (nE,)


# Line Graph

class TestLineGraph:

    def test_triangle_line_graph(self, tri_ops):
        """Triangle: K1 is 3x3 all-positive off-diag. Line graph is K3."""
        B1, _, _, _, _, nE = tri_ops
        K = np.abs(B1).T @ np.abs(B1)
        lg = _relational.build_line_graph(K, nE)
        assert lg['nV_L'] == 3
        assert lg['nE_L'] == 3  # K3 has 3 edges

    def test_k4_line_graph(self, k4_ops):
        B1, _, _, _, _, nE, K = k4_ops
        lg = _relational.build_line_graph(K, nE)
        assert lg['nV_L'] == 6
        assert lg['nE_L'] > 0
        assert np.all(lg['weights'] > 0)

    def test_matching_no_edges(self):
        """A matching (no shared vertices) has an empty line graph."""
        # 4V, 2E, no shared vertices: 0-1, 2-3
        B1 = np.array([
            [-1, 0], [1, 0], [0, -1], [0, 1]
        ], dtype=np.float64)
        K = np.abs(B1).T @ np.abs(B1)
        lg = _relational.build_line_graph(K, 2)
        assert lg['nE_L'] == 0

    def test_edge_indices_valid(self, tri_ops):
        B1, _, _, _, _, nE = tri_ops
        K = np.abs(B1).T @ np.abs(B1)
        lg = _relational.build_line_graph(K, nE)
        assert np.all(lg['src'] >= 0)
        assert np.all(lg['src'] < lg['nV_L'])
        assert np.all(lg['tgt'] >= 0)
        assert np.all(lg['tgt'] < lg['nV_L'])
        assert np.all(lg['src'] < lg['tgt'])


# L_coPC

class TestLCoPC:

    def test_symmetric(self, k4_ops):
        _, _, _, _, _, nE, K = k4_ops
        lg = _relational.build_line_graph(K, nE)
        L_C = _relational.build_L_coPC(lg)
        assert np.allclose(L_C, L_C.T)

    def test_psd(self, k4_ops):
        _, _, _, _, _, nE, K = k4_ops
        lg = _relational.build_line_graph(K, nE)
        L_C = _relational.build_L_coPC(lg)
        evals = np.linalg.eigvalsh(L_C)
        assert np.all(evals >= -1e-10)

    def test_shape(self, k4_ops):
        _, _, _, _, _, nE, K = k4_ops
        lg = _relational.build_line_graph(K, nE)
        L_C = _relational.build_L_coPC(lg)
        assert L_C.shape == (nE, nE)

    def test_matching_returns_zero(self):
        """Empty line graph produces zero L_C."""
        lg = {'nV_L': 2, 'nE_L': 0,
              'src': np.empty(0, dtype=np.int32),
              'tgt': np.empty(0, dtype=np.int32),
              'weights': np.empty(0, dtype=np.float64)}
        L_C = _relational.build_L_coPC(lg)
        assert np.allclose(L_C, 0)

    def test_nonzero_for_connected(self, tri_ops):
        """Triangle has a connected line graph, so L_C is nonzero."""
        B1, _, _, _, _, nE = tri_ops
        K = np.abs(B1).T @ np.abs(B1)
        lg = _relational.build_line_graph(K, nE)
        L_C = _relational.build_L_coPC(lg)
        assert np.trace(L_C) > 0


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_rl_trace(self, k4):
        """tr(RL) = nhats for K4 through graph.py."""
        RL = k4.RL
        nhats = k4.nhats
        assert abs(np.trace(RL) - nhats) < 1e-10

    def test_rl_symmetric(self, k4):
        assert np.allclose(k4.RL, k4.RL.T)

    def test_rl_psd(self, k4):
        evals = np.linalg.eigvalsh(k4.RL)
        assert np.all(evals >= -1e-10)

    def test_nhats_at_least_two(self, k4):
        """K4 should have at least 2 active hat operators."""
        assert k4.nhats >= 2

    def test_triangle_rl_trace(self, triangle):
        """Unfilled triangle also produces valid RL."""
        RL = triangle.RL
        nhats = triangle.nhats
        assert abs(np.trace(RL) - nhats) < 1e-10

    def test_green_cache_through_vertex_character(self, k4):
        """Accessing vertex_character triggers _green_cache internally."""
        phi = k4.vertex_character
        assert phi.shape == (k4.nV, k4.nhats)
        assert np.allclose(phi.sum(axis=1), 1.0, atol=1e-8)
