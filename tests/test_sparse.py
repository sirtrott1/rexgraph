"""
Tests for rexgraph.core._sparse -- DualCSR sparse storage and operations.

Verifies:
    - Construction: COO -> CSR -> DualCSR, correct dimensions
    - Sorted indices: column indices sorted within each row
    - Matvec: A @ x matches dense result
    - Rmatvec: A^T @ x matches dense result
    - Gram: A^T A and A A^T match dense
    - Diagonal extraction matches dense
    - Dense roundtrip: to_dense_f64 -> from_dense_f64 preserves values
    - Scipy roundtrip: to_scipy_csr -> from_scipy_csr preserves structure
    - Memory: memory_bytes > 0, validate_csr passes
    - Integration through RexGraph: B1 stored as DualCSR
"""
import numpy as np
import pytest

from rexgraph.core import _sparse
from rexgraph.graph import RexGraph


# Helpers

def _make_dual_3x4():
    """Build a simple 3x4 DualCSR from COO."""
    # [[1, 0, 2, 0],
    #  [0, 3, 0, 4],
    #  [5, 0, 0, 6]]
    rows = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    cols = np.array([0, 2, 1, 3, 0, 3], dtype=np.int32)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    return _sparse.dual_from_coo(rows, cols, vals, 3, 4)


def _dense_3x4():
    return np.array([[1, 0, 2, 0],
                     [0, 3, 0, 4],
                     [5, 0, 0, 6]], dtype=np.float64)


# Fixtures

@pytest.fixture
def dual():
    return _make_dual_3x4()


@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


# Construction

class TestConstruction:

    def test_dimensions(self, dual):
        assert dual.nrow == 3
        assert dual.ncol == 4
        assert dual.nnz == 6

    def test_idx_bits(self, dual):
        assert dual.idx_bits in (32, 64)

    def test_val_bits(self, dual):
        assert dual.val_bits in (32, 64)

    def test_csr_from_coo(self):
        rows = np.array([0, 1, 1], dtype=np.int32)
        cols = np.array([0, 0, 1], dtype=np.int32)
        vals = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        csr = _sparse.csr_from_coo(rows, cols, vals, 2, 2)
        assert csr.nrow == 2
        assert csr.ncol == 2
        assert csr.nnz == 3

    def test_sorted_col_indices(self, dual):
        """Column indices sorted within each row."""
        rp = np.asarray(dual.row_ptr)
        ci = np.asarray(dual.col_idx)
        for i in range(dual.nrow):
            row_cols = ci[rp[i]:rp[i+1]]
            assert np.all(row_cols[:-1] <= row_cols[1:])

    def test_duplicate_entries(self):
        """Duplicate COO entries create separate CSR slots."""
        rows = np.array([0, 0], dtype=np.int32)
        cols = np.array([0, 0], dtype=np.int32)
        vals = np.array([1.0, 2.0], dtype=np.float64)
        d = _sparse.dual_from_coo(rows, cols, vals, 1, 1)
        assert d.nnz == 2  # both entries stored


# Matvec

class TestMatvec:

    def test_matches_dense(self, dual):
        D = _dense_3x4()
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        y_sparse = _sparse.matvec(dual, x)
        y_dense = D @ x
        assert np.allclose(y_sparse, y_dense, atol=1e-10)

    def test_shape(self, dual):
        x = np.ones(4, dtype=np.float64)
        y = _sparse.matvec(dual, x)
        assert y.shape == (3,)


# Rmatvec

class TestRmatvec:

    def test_matches_dense(self, dual):
        D = _dense_3x4()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        y_sparse = _sparse.rmatvec(dual, x)
        y_dense = D.T @ x
        assert np.allclose(y_sparse, y_dense, atol=1e-10)

    def test_shape(self, dual):
        x = np.ones(3, dtype=np.float64)
        y = _sparse.rmatvec(dual, x)
        assert y.shape == (4,)


# Gram Products

class TestGram:

    def test_ata(self, dual):
        D = _dense_3x4()
        AtA_sparse = _sparse.spmm_AtA_dense_f64(dual)
        AtA_dense = D.T @ D
        assert np.allclose(AtA_sparse, AtA_dense, atol=1e-10)

    def test_aat(self, dual):
        D = _dense_3x4()
        AAt_sparse = _sparse.spmm_AAt_dense_f64(dual)
        AAt_dense = D @ D.T
        assert np.allclose(AAt_sparse, AAt_dense, atol=1e-10)


# Diagonal

class TestDiag:

    def test_matches_dense(self, dual):
        D = _dense_3x4()
        d = _sparse.diag(dual)
        assert d.shape == (3,)  # min(3, 4)
        assert abs(d[0] - D[0, 0]) < 1e-12
        assert abs(d[1] - D[1, 1]) < 1e-12
        assert abs(d[2] - D[2, 2]) < 1e-12


# Dense Roundtrip

class TestDenseRoundtrip:

    def test_to_dense(self, dual):
        D = _sparse.to_dense_f64(dual)
        expected = _dense_3x4()
        assert np.allclose(D, expected, atol=1e-12)

    def test_from_dense_roundtrip(self):
        D = _dense_3x4()
        dual = _sparse.from_dense_f64(D)
        D2 = _sparse.to_dense_f64(dual)
        assert np.allclose(D, D2, atol=1e-12)


# Scipy Roundtrip

class TestScipyRoundtrip:

    def test_roundtrip(self, dual):
        sp = _sparse.to_scipy_csr(dual)
        dual2 = _sparse.from_scipy_csr(sp)
        D1 = _sparse.to_dense_f64(dual)
        D2 = _sparse.to_dense_f64(dual2)
        assert np.allclose(D1, D2, atol=1e-12)


# Memory and Validation

class TestMemoryValidation:

    def test_memory_bytes_positive(self, dual):
        assert _sparse.memory_bytes(dual) > 0

    def test_memory_report_string(self, dual):
        report = _sparse.memory_report(dual)
        assert isinstance(report, str)

    def test_validate_passes(self, dual):
        ok, msg = _sparse.validate_csr(dual)
        assert ok, msg


# Row/Col Access

class TestAccess:

    def test_row_entries(self, dual):
        cols, vals = _sparse.row_entries(dual, 0)
        assert len(cols) == 2  # row 0 has entries at cols 0 and 2

    def test_col_entries(self, dual):
        rows, vals = _sparse.col_entries(dual, 0)
        assert len(rows) == 2  # col 0 has entries at rows 0 and 2

    def test_row_nnz(self, dual):
        assert _sparse.row_nnz(dual.csr, 0) == 2

    def test_col_nnz(self, dual):
        assert _sparse.col_nnz(dual, 0) == 2


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_b1_is_dual_csr(self, k4):
        """B1 is stored internally as DualCSR."""
        assert k4._B1_dual is not None
        assert isinstance(k4._B1_dual, _sparse.DualCSR)

    def test_b1_dense_matches(self, k4):
        """Dense B1 from DualCSR matches graph.py B1 property."""
        B1_dense = _sparse.to_dense_f64(k4._B1_dual)
        B1_prop = np.asarray(k4.B1, dtype=np.float64)
        assert np.allclose(B1_dense, B1_prop, atol=1e-12)
