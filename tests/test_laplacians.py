"""
Tests for rexgraph.core._laplacians - Hodge Laplacians and spectral decomposition.

Verifies:
    - L0, L1_down, L1_up, L1_full, L2 shapes and symmetry
    - Positive semi-definiteness of all Laplacians
    - Betti numbers from eigenvalue nullities
    - Hodge decomposition: ker(L1) = ker(L1_down) intersect ker(L1_up)
    - Fiedler value and vector
    - Diagonal extraction matches full construction
    - Trace normalization
    - Coupling constants
    - Composite operators (L1_alpha, Lambda)
    - build_all_laplacians returns complete bundle
"""
import numpy as np
import pytest

from rexgraph.core import _laplacians


# Helpers

def _triangle_B1():
    """B1 for triangle: 3V, 3E. Edges: 0->1, 1->2, 0->2."""
    return np.array([
        [-1,  0, -1],
        [ 1, -1,  0],
        [ 0,  1,  1],
    ], dtype=np.float64)


def _triangle_B2():
    """B2 for filled triangle: 3E, 1F."""
    return np.array([
        [ 1],
        [ 1],
        [-1],
    ], dtype=np.float64)


def _k4_B1():
    """B1 for K4: 4V, 6E."""
    B1 = np.zeros((4, 6), dtype=np.float64)
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for j, (s, t) in enumerate(edges):
        B1[s, j] = -1.0
        B1[t, j] = 1.0
    return B1


def _k4_B2():
    """B2 for K4: 6E, 4F. All 4 triangles filled."""
    from rexgraph.graph import RexGraph
    rex = RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32),
    )
    return np.asarray(rex.B2_hodge, dtype=np.float64)


def _tree_B1():
    """B1 for path graph: 4V, 3E. Edges: 0->1, 1->2, 2->3."""
    return np.array([
        [-1,  0,  0],
        [ 1, -1,  0],
        [ 0,  1, -1],
        [ 0,  0,  1],
    ], dtype=np.float64)


# Laplacian Construction

class TestLaplacianConstruction:

    def test_L0_shape(self):
        B1 = _triangle_B1()
        L0 = _laplacians.build_L0(B1)
        assert L0.shape == (3, 3)

    def test_L0_symmetric(self):
        L0 = _laplacians.build_L0(_triangle_B1())
        assert np.allclose(L0, L0.T)

    def test_L0_psd(self):
        L0 = _laplacians.build_L0(_triangle_B1())
        evals = np.linalg.eigvalsh(L0)
        assert np.all(evals >= -1e-10)

    def test_L0_row_sums_zero(self):
        """Row sums of L0 are zero (constant vector in kernel)."""
        L0 = _laplacians.build_L0(_triangle_B1())
        assert np.allclose(L0.sum(axis=1), 0, atol=1e-12)

    def test_L1_down_shape(self):
        L1d = _laplacians.build_L1_down(_triangle_B1())
        assert L1d.shape == (3, 3)

    def test_L1_down_symmetric(self):
        L1d = _laplacians.build_L1_down(_triangle_B1())
        assert np.allclose(L1d, L1d.T)

    def test_L1_up_zero_without_faces(self):
        B2 = np.zeros((3, 0), dtype=np.float64)
        L1u = _laplacians.build_L1_up(B2)
        assert np.allclose(L1u, 0)

    def test_L1_up_nonzero_with_faces(self):
        L1u = _laplacians.build_L1_up(_triangle_B2())
        assert L1u.shape == (3, 3)
        assert np.trace(L1u) > 0

    def test_L1_full_is_sum(self):
        B1 = _triangle_B1()
        B2 = _triangle_B2()
        L1d = _laplacians.build_L1_down(B1)
        L1u = _laplacians.build_L1_up(B2)
        L1f = _laplacians.build_L1_full(L1d, L1u)
        assert np.allclose(L1f, L1d + L1u)

    def test_L1_full_psd(self):
        B1 = _triangle_B1()
        B2 = _triangle_B2()
        L1d = _laplacians.build_L1_down(B1)
        L1u = _laplacians.build_L1_up(B2)
        L1f = _laplacians.build_L1_full(L1d, L1u)
        evals = np.linalg.eigvalsh(L1f)
        assert np.all(evals >= -1e-10)

    def test_L2_shape(self):
        B2 = _triangle_B2()
        L2 = _laplacians.build_L2(B2)
        assert L2.shape == (1, 1)

    def test_L2_empty(self):
        B2 = np.zeros((3, 0), dtype=np.float64)
        L2 = _laplacians.build_L2(B2)
        assert L2.shape == (0, 0)


# Eigendecomposition

class TestEigendecomposition:

    def test_eigenvalues_ascending(self):
        L0 = _laplacians.build_L0(_triangle_B1())
        evals, evecs = _laplacians.eigen_symmetric(L0)
        assert np.all(np.diff(evals) >= -1e-12)

    def test_eigenvalues_nonnegative(self):
        L0 = _laplacians.build_L0(_triangle_B1())
        evals, _ = _laplacians.eigen_symmetric(L0)
        assert np.all(evals >= -1e-12)

    def test_eigenvectors_orthonormal(self):
        L0 = _laplacians.build_L0(_k4_B1())
        _, evecs = _laplacians.eigen_symmetric(L0)
        prod = evecs.T @ evecs
        assert np.allclose(prod, np.eye(4), atol=1e-10)

    def test_reconstruction(self):
        """L = V diag(evals) V^T."""
        L0 = _laplacians.build_L0(_triangle_B1())
        evals, evecs = _laplacians.eigen_symmetric(L0)
        reconstructed = evecs @ np.diag(evals) @ evecs.T
        assert np.allclose(L0, reconstructed, atol=1e-10)

    def test_empty_matrix(self):
        L = np.zeros((0, 0), dtype=np.float64)
        evals, evecs = _laplacians.eigen_symmetric(L)
        assert len(evals) == 0

    def test_clean_eigenvalues(self):
        evals = np.array([-1e-13, 1e-14, 0.5, 1.0], dtype=np.float64)
        cleaned = _laplacians.clean_eigenvalues(evals)
        assert cleaned[0] == 0.0
        assert cleaned[1] == 0.0
        assert cleaned[2] == 0.5


# Betti Numbers

class TestBettiNumbers:

    def test_triangle_betti(self):
        """Filled triangle: beta_0=1, beta_1=0, beta_2=0."""
        B1 = _triangle_B1()
        B2 = _triangle_B2()
        L0 = _laplacians.build_L0(B1)
        L1d = _laplacians.build_L1_down(B1)
        L1u = _laplacians.build_L1_up(B2)
        L1 = _laplacians.build_L1_full(L1d, L1u)
        L2 = _laplacians.build_L2(B2)

        e0, _ = _laplacians.eigen_symmetric(L0)
        e1, _ = _laplacians.eigen_symmetric(L1)
        e2, _ = _laplacians.eigen_symmetric(L2)

        b0 = np.sum(np.abs(e0) < 1e-8)
        b1 = np.sum(np.abs(e1) < 1e-8)
        b2 = np.sum(np.abs(e2) < 1e-8)
        assert b0 == 1
        assert b1 == 0
        assert b2 == 0

    def test_tree_betti(self):
        """Path graph: beta_0=1, beta_1=0."""
        B1 = _tree_B1()
        L0 = _laplacians.build_L0(B1)
        L1d = _laplacians.build_L1_down(B1)
        e0, _ = _laplacians.eigen_symmetric(L0)
        e1, _ = _laplacians.eigen_symmetric(L1d)
        assert np.sum(np.abs(e0) < 1e-8) == 1
        assert np.sum(np.abs(e1) < 1e-8) == 0

    def test_unfilled_triangle_betti(self):
        """Unfilled triangle: beta_0=1, beta_1=1 (one cycle)."""
        B1 = _triangle_B1()
        L1d = _laplacians.build_L1_down(B1)
        e1, _ = _laplacians.eigen_symmetric(L1d)
        # Without faces, L1 = L1_down only. beta_1 = nullity(L1_down) if no faces.
        # But beta_1 = nE - rank(B1) - rank(B2) = 3 - 2 - 0 = 1
        assert np.sum(np.abs(e1) < 1e-8) == 1

    def test_euler_relation(self):
        """chi = beta_0 - beta_1 + beta_2 = nV - nE + nF."""
        B1 = _triangle_B1()
        B2 = _triangle_B2()
        L0 = _laplacians.build_L0(B1)
        L1 = _laplacians.build_L1_full(
            _laplacians.build_L1_down(B1),
            _laplacians.build_L1_up(B2))
        L2 = _laplacians.build_L2(B2)
        e0, _ = _laplacians.eigen_symmetric(L0)
        e1, _ = _laplacians.eigen_symmetric(L1)
        e2, _ = _laplacians.eigen_symmetric(L2)
        b0 = np.sum(np.abs(e0) < 1e-8)
        b1 = np.sum(np.abs(e1) < 1e-8)
        b2 = np.sum(np.abs(e2) < 1e-8)
        nV, nE, nF = 3, 3, 1
        assert b0 - b1 + b2 == nV - nE + nF


# Fiedler

class TestFiedler:

    def test_fiedler_positive_connected(self):
        """Connected graph has positive Fiedler value."""
        L0 = _laplacians.build_L0(_triangle_B1())
        evals, _ = _laplacians.eigen_symmetric(L0)
        fv = _laplacians.fiedler_value(evals)
        assert fv > 0

    def test_fiedler_vector_orthogonal_to_constant(self):
        L0 = _laplacians.build_L0(_k4_B1())
        evals, evecs = _laplacians.eigen_symmetric(L0)
        fvec = _laplacians.fiedler_vector(evecs, evals)
        # Fiedler vector is orthogonal to the all-ones vector
        assert abs(np.sum(fvec)) < 1e-10


# Diagonal Extraction

class TestDiagonalExtraction:

    def test_matches_full(self):
        """Diagonal extraction matches diag of full L1_down and L1_up."""
        B1 = _triangle_B1()
        B2 = _triangle_B2()
        dd, du = _laplacians.extract_diag_L1(B1, B2)
        L1d = _laplacians.build_L1_down(B1)
        L1u = _laplacians.build_L1_up(B2)
        assert np.allclose(dd, np.diag(L1d))
        assert np.allclose(du, np.diag(L1u))

    def test_diag_nonnegative(self):
        """Diagonals of PSD matrices are nonnegative."""
        B1 = _k4_B1()
        B2 = _k4_B2()
        dd, du = _laplacians.extract_diag_L1(B1, B2)
        assert np.all(dd >= -1e-12)
        assert np.all(du >= -1e-12)


# Trace Normalization

class TestTraceNormalization:

    def test_trace_one(self):
        L1d = _laplacians.build_L1_down(_triangle_B1())
        L_hat, tr = _laplacians.trace_normalize(L1d)
        assert tr > 0
        assert abs(np.trace(L_hat) - 1.0) < 1e-12

    def test_zero_trace(self):
        """Zero matrix returns zero hat and trace 0."""
        Z = np.zeros((3, 3), dtype=np.float64)
        L_hat, tr = _laplacians.trace_normalize(Z)
        assert tr == 0.0
        assert np.allclose(L_hat, 0)

    def test_preserves_symmetry(self):
        L1d = _laplacians.build_L1_down(_triangle_B1())
        L_hat, _ = _laplacians.trace_normalize(L1d)
        assert np.allclose(L_hat, L_hat.T)

    def test_preserves_psd(self):
        L1d = _laplacians.build_L1_down(_triangle_B1())
        L_hat, _ = _laplacians.trace_normalize(L1d)
        evals = np.linalg.eigvalsh(L_hat)
        assert np.all(evals >= -1e-12)


# Composite Operators

class TestCompositeOperators:

    def test_L1_alpha_at_zero(self):
        """alpha=0 returns L1 unchanged."""
        B1 = _triangle_B1()
        L1 = _laplacians.build_L1_down(B1)
        L_O = np.eye(3, dtype=np.float64)
        result = _laplacians.build_L1_alpha(L1, L_O, 0.0)
        assert np.allclose(result, L1)

    def test_L1_alpha_linearity(self):
        B1 = _triangle_B1()
        L1 = _laplacians.build_L1_down(B1)
        L_O = np.eye(3, dtype=np.float64)
        r1 = _laplacians.build_L1_alpha(L1, L_O, 1.0)
        r2 = _laplacians.build_L1_alpha(L1, L_O, 2.0)
        assert np.allclose(r2 - r1, L_O)

    def test_Lambda_shape(self):
        B1 = _triangle_B1()
        L_O = np.eye(3, dtype=np.float64)
        lam = _laplacians.build_Lambda(B1, L_O)
        assert lam.shape == (3, 3)

    def test_Lambda_symmetric(self):
        B1 = _triangle_B1()
        L_O = np.eye(3, dtype=np.float64)
        lam = _laplacians.build_Lambda(B1, L_O)
        assert np.allclose(lam, lam.T)


# Coupling Constants

class TestCouplingConstants:

    def test_alpha_G_positive(self):
        B1 = _triangle_B1()
        L1 = _laplacians.build_L1_down(B1)
        L_O = np.eye(3, dtype=np.float64) * 0.5
        e1, _ = _laplacians.eigen_symmetric(L1)
        eO, _ = _laplacians.eigen_symmetric(L_O)
        aG, aT = _laplacians.compute_coupling_constants(e1, eO, 1, 3)
        assert aG > 0

    def test_alpha_T_range(self):
        B1 = _triangle_B1()
        L1 = _laplacians.build_L1_down(B1)
        evals = np.array([0.0, 1.0, 3.0], dtype=np.float64)
        _, aT = _laplacians.compute_coupling_constants(evals, evals, 1, 3)
        assert 0 <= aT <= 1


# build_all_laplacians

class TestBuildAllLaplacians:

    def _get_L_O(self, B1):
        """Build L_O through RexGraph to use correct API."""
        nV, nE = B1.shape
        # Build L_O manually: K = |B1|^T |B1|, then normalize
        K = np.abs(B1).T @ np.abs(B1)
        d = K.sum(axis=1)
        d_inv_sqrt = np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0)
        D = np.diag(d_inv_sqrt)
        S = D @ K @ D
        L_O = np.eye(nE, dtype=np.float64) - S
        # Symmetrize
        L_O = 0.5 * (L_O + L_O.T)
        return L_O

    def test_returns_dict(self):
        B1 = _triangle_B1()
        B2 = _triangle_B2()
        L_O = self._get_L_O(B1)
        result = _laplacians.build_all_laplacians(B1, B2, L_O)
        assert isinstance(result, dict)

    def test_contains_all_keys(self):
        B1 = _triangle_B1()
        B2 = _triangle_B2()
        L_O = self._get_L_O(B1)
        result = _laplacians.build_all_laplacians(B1, B2, L_O)
        for key in ['L0', 'L1_down', 'L1_up', 'L1_full', 'L2',
                     'evals_L0', 'evals_L1', 'evals_L2',
                     'beta0', 'beta1', 'beta2',
                     'RL', 'hats', 'nhats', 'hat_names', 'chi']:
            assert key in result, f"Missing key: {key}"

    def test_betti_from_bundle(self):
        B1 = _triangle_B1()
        B2 = _triangle_B2()
        L_O = self._get_L_O(B1)
        result = _laplacians.build_all_laplacians(B1, B2, L_O)
        assert result['beta0'] == 1
        assert result['beta1'] == 0

    def test_rl_trace_equals_nhats(self):
        """tr(RL) = number of hat operators with nonzero trace."""
        B1 = _triangle_B1()
        B2 = _triangle_B2()
        L_O = self._get_L_O(B1)
        result = _laplacians.build_all_laplacians(B1, B2, L_O)
        if result['nhats'] > 0:
            assert abs(np.trace(result['RL']) - result['nhats']) < 1e-10

    def test_k4_with_frustration(self):
        """K4 with frustration Laplacian produces RL with more hat operators."""
        B1 = _k4_B1()
        B2 = _k4_B2()
        L_O = self._get_L_O(B1)
        # Build L_SG manually: signed Gramian -> frustration Laplacian
        nV = B1.shape[0]
        deg = np.abs(B1).sum(axis=1)
        W = np.diag(1.0 / np.log(deg + np.e))
        Ks = B1.T @ W @ B1
        Koff = Ks.copy()
        np.fill_diagonal(Koff, 0)
        L_SG = np.diag(np.abs(Koff).sum(axis=1)) - Koff
        result = _laplacians.build_all_laplacians(B1, B2, L_O, L_SG_in=L_SG)
        assert result['nhats'] >= 2

    def test_no_faces(self):
        """Tree graph with no faces still produces a valid bundle."""
        B1 = _tree_B1()
        B2 = np.zeros((3, 0), dtype=np.float64)
        result = _laplacians.build_all_laplacians(B1, B2, None)
        assert result['beta0'] == 1
        assert result['L1_up'].shape == (3, 3)
        assert np.allclose(result['L1_up'], 0)
