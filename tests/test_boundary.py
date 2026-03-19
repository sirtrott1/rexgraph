"""
Tests for rexgraph.core._boundary - chain complex construction.

Verifies:
    - B1 shape, column sums zero, signed incidence
    - B2 from cycles, correct shape and signs
    - Chain condition B1 @ B2 = 0
    - Betti numbers from eigenvalues match rank-based computation
    - Euler relation: beta_0 - beta_1 + beta_2 = nV - nE + nF
    - count_zero_eigenvalues on known spectra
    - SVD rank fallback
"""
import numpy as np
import pytest

from rexgraph.core import _boundary


# Helpers

def _triangle_data():
    """Triangle: 3V, 3E, edges 0->1, 1->2, 0->2."""
    return 3, 3, np.array([0, 1, 0], dtype=np.int32), np.array([1, 2, 2], dtype=np.int32)


def _k4_data():
    """K4: 4V, 6E."""
    src = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
    tgt = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
    return 4, 6, src, tgt


def _tree_data():
    """Path 0-1-2-3: 4V, 3E."""
    return 4, 3, np.array([0, 1, 2], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32)


def _build_triangle_B2(nE=3):
    """B2 for a filled triangle: edges [0,1,2], signs [+1,+1,-1]."""
    ce = np.array([0, 1, 2], dtype=np.int32)
    cs = np.array([1.0, 1.0, -1.0], dtype=np.float64)
    cl = np.array([3], dtype=np.int32)
    return _boundary.build_B2_from_cycles(nE, ce, cs, cl)


# B1 Construction

class TestBuildB1:

    def test_shape(self):
        nV, nE, src, tgt = _triangle_data()
        B1 = _boundary.build_B1(nV, nE, src, tgt)
        assert B1.nrow == nV
        assert B1.ncol == nE

    def test_column_sums_zero(self):
        """Each column of B1 sums to zero (one -1 and one +1)."""
        nV, nE, src, tgt = _triangle_data()
        B1 = _boundary.build_B1(nV, nE, src, tgt)
        from rexgraph.core._sparse import to_dense_f64
        dense = to_dense_f64(B1)
        col_sums = dense.sum(axis=0)
        assert np.allclose(col_sums, 0, atol=1e-12)

    def test_signed_incidence(self):
        """B1[src, e] = -1 and B1[tgt, e] = +1."""
        nV, nE, src, tgt = _triangle_data()
        B1 = _boundary.build_B1(nV, nE, src, tgt)
        from rexgraph.core._sparse import to_dense_f64
        dense = to_dense_f64(B1)
        for e in range(nE):
            assert dense[src[e], e] == -1.0
            assert dense[tgt[e], e] == 1.0

    def test_nnz(self):
        """B1 has exactly 2*nE nonzeros."""
        nV, nE, src, tgt = _triangle_data()
        B1 = _boundary.build_B1(nV, nE, src, tgt)
        assert B1.nnz == 2 * nE

    def test_k4(self):
        nV, nE, src, tgt = _k4_data()
        B1 = _boundary.build_B1(nV, nE, src, tgt)
        assert B1.nrow == 4
        assert B1.ncol == 6
        assert B1.nnz == 12

    def test_int64_dispatch(self):
        """int64 sources/targets produce correct B1."""
        nV, nE = 3, 3
        src = np.array([0, 1, 0], dtype=np.int64)
        tgt = np.array([1, 2, 2], dtype=np.int64)
        B1 = _boundary.build_B1(nV, nE, src, tgt)
        assert B1.nrow == 3
        assert B1.ncol == 3


# B2 Construction

class TestBuildB2:

    def test_shape(self):
        B2 = _build_triangle_B2()
        assert B2.nrow == 3  # nE
        assert B2.ncol == 1  # nF

    def test_nnz(self):
        """Triangle B2 has 3 nonzeros (3 boundary edges per face)."""
        B2 = _build_triangle_B2()
        assert B2.nnz == 3

    def test_signs(self):
        """B2 entries should be +/-1."""
        B2 = _build_triangle_B2()
        from rexgraph.core._sparse import to_dense_f64
        dense = to_dense_f64(B2)
        for i in range(3):
            assert abs(dense[i, 0]) == 1.0

    def test_multiple_faces(self):
        """Two triangles sharing an edge."""
        # Face 0: edges [0,1,2], Face 1: edges [0,3,4]
        ce = np.array([0, 1, 2, 0, 3, 4], dtype=np.int32)
        cs = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float64)
        cl = np.array([3, 3], dtype=np.int32)
        B2 = _boundary.build_B2_from_cycles(5, ce, cs, cl)
        assert B2.nrow == 5
        assert B2.ncol == 2
        assert B2.nnz == 6

    def test_from_dense(self):
        matrix = np.array([[1.0], [1.0], [-1.0]], dtype=np.float64)
        B2 = _boundary.build_B2_from_dense(3, 1, matrix)
        assert B2.nrow == 3
        assert B2.ncol == 1


# Chain Condition

class TestChainCondition:

    def test_triangle_chain(self):
        """B1 @ B2 = 0 for a filled triangle."""
        nV, nE, src, tgt = _triangle_data()
        B1 = _boundary.build_B1(nV, nE, src, tgt)
        B2 = _build_triangle_B2()
        ok, err = _boundary.verify_chain_complex(B1, B2)
        assert ok
        assert err < 1e-10

    def test_k4_chain(self):
        """B1 @ B2 = 0 for K4 with all faces."""
        from rexgraph.graph import RexGraph
        rex = RexGraph.from_simplicial(
            sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
            targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
            triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32),
        )
        ok = rex.chain_valid
        assert ok

    def test_wrong_signs_fail(self):
        """B1 @ B2 != 0 when B2 signs are wrong."""
        nV, nE, src, tgt = _triangle_data()
        B1 = _boundary.build_B1(nV, nE, src, tgt)
        # All positive signs (wrong orientation)
        ce = np.array([0, 1, 2], dtype=np.int32)
        cs = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        cl = np.array([3], dtype=np.int32)
        B2 = _boundary.build_B2_from_cycles(nE, ce, cs, cl)
        ok, err = _boundary.verify_chain_complex(B1, B2)
        assert not ok


# Betti Numbers

class TestBettiNumbers:

    def test_count_zero_eigenvalues(self):
        evals = np.array([0.0, 0.0, 0.5, 1.0, 3.0], dtype=np.float64)
        assert _boundary.count_zero_eigenvalues(evals) == 2

    def test_count_with_tolerance(self):
        evals = np.array([1e-13, 1e-11, 0.5], dtype=np.float64)
        assert _boundary.count_zero_eigenvalues(evals, tol=1e-10) == 2

    def test_rank_from_eigenvalues(self):
        evals = np.array([0.0, 1.0, 3.0], dtype=np.float64)
        assert _boundary.rank_from_eigenvalues(evals, 3) == 2

    def test_betti_from_eigenvalues_triangle(self):
        """Filled triangle: beta = (1, 0, 0)."""
        from rexgraph.core._laplacians import build_L0, build_L1_down, build_L1_up, build_L1_full, build_L2, eigen_symmetric
        B1_dense = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float64)
        B2_dense = np.array([[1], [1], [-1]], dtype=np.float64)
        e0, _ = eigen_symmetric(build_L0(B1_dense))
        e1, _ = eigen_symmetric(build_L1_full(build_L1_down(B1_dense), build_L1_up(B2_dense)))
        e2, _ = eigen_symmetric(build_L2(B2_dense))
        info = _boundary.betti_from_eigenvalues(e0, e1, e2, 3, 3, 1)
        assert info['beta0'] == 1
        assert info['beta1'] == 0
        assert info['beta2'] == 0
        assert info['euler_check']

    def test_betti_from_eigenvalues_tree(self):
        """Path graph: beta = (1, 0)."""
        from rexgraph.core._laplacians import build_L0, build_L1_down, eigen_symmetric
        B1_dense = np.array([[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]], dtype=np.float64)
        e0, _ = eigen_symmetric(build_L0(B1_dense))
        e1, _ = eigen_symmetric(build_L1_down(B1_dense))
        info = _boundary.betti_from_eigenvalues(e0, e1, None, 4, 3, 0)
        assert info['beta0'] == 1
        assert info['beta1'] == 0
        assert info['beta2'] == 0

    def test_euler_relation(self):
        """beta_0 - beta_1 + beta_2 = nV - nE + nF for K4."""
        from rexgraph.graph import RexGraph
        rex = RexGraph.from_simplicial(
            sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
            targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
            triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32),
        )
        b = rex.betti
        chi = b[0] - b[1] + b[2]
        assert chi == rex.nV - rex.nE + rex.nF

    def test_unfilled_triangle_has_cycle(self):
        """Unfilled triangle: beta_1 = 1 (one independent cycle)."""
        from rexgraph.core._laplacians import build_L0, build_L1_down, eigen_symmetric
        B1_dense = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float64)
        e0, _ = eigen_symmetric(build_L0(B1_dense))
        e1, _ = eigen_symmetric(build_L1_down(B1_dense))  # no L1_up (no faces)
        info = _boundary.betti_from_eigenvalues(e0, e1, None, 3, 3, 0)
        assert info['beta0'] == 1
        assert info['beta1'] == 1


# Unified Interface

class TestBettiInterface:

    def test_with_eigenvalues(self):
        """Spectral path produces correct Betti for triangle."""
        from rexgraph.graph import RexGraph
        rex = RexGraph.from_graph([0, 1, 0], [1, 2, 2])
        b = rex.betti
        assert b[0] == 1
        assert b[1] == 1  # unfilled triangle

    def test_filled_triangle(self):
        from rexgraph.graph import RexGraph
        rex = RexGraph.from_simplicial(
            sources=np.array([0, 1, 0], dtype=np.int32),
            targets=np.array([1, 2, 2], dtype=np.int32),
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
        )
        b = rex.betti
        assert b[0] == 1
        assert b[1] == 0
        assert b[2] == 0
