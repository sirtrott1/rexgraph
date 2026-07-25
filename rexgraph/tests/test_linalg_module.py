"""
Tests for rexgraph.core._linalg - LAPACK/BLAS wrappers and RL pipeline.

Verifies:
    - eigh: eigenvalues ascending, nonneg for PSD, reconstruction A = V D V^T
    - svd: A = U S Vt reconstruction, singular values nonneg
    - lstsq: solves known system
    - matrix_rank: correct for known matrices
    - gemm_nn/nt/tn: match numpy matmul
    - pinv_spectral: A @ A^+ @ A = A
    - pinv_matvec: matches full pinv @ x
    - rl_pipeline: tr(RL) = 3, chi sums to 1, kappa in [0,1]
"""
import numpy as np
import pytest

from rexgraph.core import _linalg


# Helpers

def _random_psd(n, seed=42):
    """Random positive semi-definite matrix."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    return (A @ A.T) / n


def _triangle_B1():
    return np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float64)


# eigh

class TestEigh:

    def test_ascending(self):
        L = _random_psd(5)
        evals, evecs = _linalg.eigh(L)
        assert np.all(np.diff(evals) >= -1e-12)

    def test_nonneg_psd(self):
        L = _random_psd(5)
        evals, _ = _linalg.eigh(L)
        assert np.all(evals >= -1e-10)

    def test_reconstruction(self):
        """A = V diag(evals) V^T."""
        L = _random_psd(4)
        evals, evecs = _linalg.eigh(L)
        recon = evecs @ np.diag(evals) @ evecs.T
        assert np.allclose(L, recon, atol=1e-10)

    def test_orthonormal_evecs(self):
        L = _random_psd(5)
        _, evecs = _linalg.eigh(L)
        assert np.allclose(evecs.T @ evecs, np.eye(5), atol=1e-10)

    def test_identity(self):
        evals, evecs = _linalg.eigh(np.eye(3, dtype=np.float64))
        assert np.allclose(evals, 1.0)


# svd

class TestSVD:

    def test_reconstruction(self):
        rng = np.random.RandomState(42)
        A = rng.randn(4, 3).astype(np.float64)
        U, S, Vt = _linalg.svd(A)
        recon = U[:, :3] @ np.diag(S) @ Vt[:3, :]
        assert np.allclose(A, recon, atol=1e-10)

    def test_singular_values_nonneg(self):
        rng = np.random.RandomState(42)
        A = rng.randn(5, 3).astype(np.float64)
        _, S, _ = _linalg.svd(A)
        assert np.all(S >= -1e-12)

    def test_square(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        U, S, Vt = _linalg.svd(A)
        recon = U @ np.diag(S) @ Vt
        assert np.allclose(A, recon, atol=1e-10)


# lstsq

class TestLstsq:

    def test_exact_system(self):
        """Solve A x = b where A is invertible."""
        A = np.array([[2, 1], [1, 3]], dtype=np.float64)
        b = np.array([5, 7], dtype=np.float64)
        x, rank = _linalg.lstsq(A, b)
        assert rank == 2
        assert np.allclose(A @ x, b, atol=1e-10)

    def test_overdetermined(self):
        """Least squares for overdetermined system."""
        A = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
        b = np.array([1, 2, 4], dtype=np.float64)
        x, rank = _linalg.lstsq(A, b)
        assert rank == 2
        # Residual should be smaller than random guess
        residual = np.linalg.norm(A @ x - b)
        random_residual = np.linalg.norm(A @ np.ones(2) - b)
        assert residual <= random_residual + 1e-10


# matrix_rank

class TestMatrixRank:

    def test_full_rank(self):
        A = np.eye(3, dtype=np.float64)
        assert _linalg.matrix_rank(A) == 3

    def test_rank_deficient(self):
        A = np.array([[1, 2], [2, 4]], dtype=np.float64)
        assert _linalg.matrix_rank(A) == 1

    def test_zero_matrix(self):
        A = np.zeros((3, 3), dtype=np.float64)
        assert _linalg.matrix_rank(A) == 0


# gemm

class TestGemm:

    def test_nn(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.array([[5, 6], [7, 8]], dtype=np.float64)
        C = _linalg.gemm_nn(A, B)
        assert np.allclose(C, A @ B)

    def test_nt(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.array([[5, 6], [7, 8]], dtype=np.float64)
        C = _linalg.gemm_nt(A, B)
        assert np.allclose(C, A @ B.T)

    def test_tn(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.array([[5, 6], [7, 8]], dtype=np.float64)
        C = _linalg.gemm_tn(A, B)
        assert np.allclose(C, A.T @ B)

    def test_rectangular(self):
        A = np.ones((3, 4), dtype=np.float64)
        B = np.ones((4, 2), dtype=np.float64)
        C = _linalg.gemm_nn(A, B)
        assert C.shape == (3, 2)
        assert np.allclose(C, 4.0)


# Spectral Pseudoinverse

class TestPinv:

    def test_identity_pinv(self):
        evals = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        evecs = np.eye(3, dtype=np.float64)
        P = _linalg.pinv_spectral(evals, evecs)
        assert np.allclose(P, np.eye(3), atol=1e-10)

    def test_psd_pinv(self):
        """A @ A^+ @ A = A for PSD matrix."""
        L = _random_psd(4)
        evals, evecs = _linalg.eigh(L)
        P = _linalg.pinv_spectral(evals, evecs)
        assert np.allclose(L @ P @ L, L, atol=1e-8)

    def test_pinv_matvec_matches_full(self):
        """pinv_matvec produces same result as full pinv @ x."""
        L = _random_psd(4)
        evals, evecs = _linalg.eigh(L)
        P = _linalg.pinv_spectral(evals, evecs)
        x = np.array([1, 2, 3, 4], dtype=np.float64)
        y_full = P @ x
        y_fast = _linalg.pinv_matvec(evals, evecs, x)
        assert np.allclose(y_full, y_fast, atol=1e-10)

    def test_zero_eigenvalues_excluded(self):
        """Eigenvalues below tol are excluded from pinv."""
        evals = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float64)
        evecs = np.eye(4, dtype=np.float64)
        P = _linalg.pinv_spectral(evals, evecs)
        # First two eigenvalues are zero -> pinv diagonal is [0, 0, 1, 0.5]
        assert abs(P[0, 0]) < 1e-12
        assert abs(P[1, 1]) < 1e-12
        assert abs(P[2, 2] - 1.0) < 1e-12
        assert abs(P[3, 3] - 0.5) < 1e-12


# rl_pipeline

class TestRLPipeline:

    def _build_inputs(self):
        B1 = _triangle_B1()
        nV, nE = B1.shape
        L1 = B1.T @ B1
        K = np.abs(B1).T @ np.abs(B1)
        d = K.sum(axis=1)
        D_inv = np.diag(np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0))
        L_O = np.eye(nE) - D_inv @ K @ D_inv
        L_O = 0.5 * (L_O + L_O.T)
        deg = np.abs(B1).sum(axis=1)
        W = np.diag(1.0 / np.log(deg + np.e))
        Ks = B1.T @ W @ B1
        Koff = Ks.copy()
        np.fill_diagonal(Koff, 0)
        L_SG = np.diag(np.abs(Koff).sum(axis=1)) - Koff
        return B1, L1, L_O, L_SG

    def test_trace_equals_nhats(self):
        B1, L1, L_O, L_SG = self._build_inputs()
        r = _linalg.rl_pipeline(B1, L1, L_O, L_SG)
        assert abs(np.trace(r['RL']) - r['nhats']) < 1e-10

    def test_chi_simplex(self):
        """Every row of chi sums to 1."""
        B1, L1, L_O, L_SG = self._build_inputs()
        r = _linalg.rl_pipeline(B1, L1, L_O, L_SG)
        assert np.allclose(r['chi'].sum(axis=1), 1.0, atol=1e-10)

    def test_phi_simplex(self):
        """Every row of phi sums to 1."""
        B1, L1, L_O, L_SG = self._build_inputs()
        r = _linalg.rl_pipeline(B1, L1, L_O, L_SG)
        assert np.allclose(r['phi'].sum(axis=1), 1.0, atol=1e-8)

    def test_kappa_range(self):
        B1, L1, L_O, L_SG = self._build_inputs()
        r = _linalg.rl_pipeline(B1, L1, L_O, L_SG)
        assert np.all(r['kappa'] >= -1e-10)
        assert np.all(r['kappa'] <= 1.0 + 1e-10)

    def test_rl_symmetric(self):
        B1, L1, L_O, L_SG = self._build_inputs()
        r = _linalg.rl_pipeline(B1, L1, L_O, L_SG)
        assert np.allclose(r['RL'], r['RL'].T, atol=1e-12)

    def test_rl_psd(self):
        B1, L1, L_O, L_SG = self._build_inputs()
        r = _linalg.rl_pipeline(B1, L1, L_O, L_SG)
        assert np.all(r['evals'] >= -1e-10)

    def test_returns_all_keys(self):
        B1, L1, L_O, L_SG = self._build_inputs()
        r = _linalg.rl_pipeline(B1, L1, L_O, L_SG)
        for key in ['RL', 'evals', 'evecs', 'RLp', 'chi', 'phi',
                     'chi_star', 'kappa', 'B1_RLp', 'S0_diag', 'hats', 'nhats']:
            assert key in r
