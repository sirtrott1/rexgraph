"""
Tests for rexgraph.core._dirac -- Dirac operator and graded state evolution.

Verifies:
    - D is real symmetric, correct shape (nV+nE+nF)^2
    - D^2 = blkdiag(L0, L1, L2) when B1 @ B2 = 0
    - Eigenvalues can be positive and negative (D is not PSD)
    - Schrodinger evolution preserves ||Psi||^2
    - psi_re(0) = psi0, psi_im(0) = 0
    - Canonical collapse: face component is zero, state is normalized
    - Born probabilities sum to 1 per timestep
    - Energy partition sums to 1
    - Integration through RexGraph: dirac_operator, dirac_eigenvalues,
      graded_state, canonical_collapse, born_graded, energy_partition
"""
import numpy as np
import pytest

from rexgraph.core import _dirac
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
def filled_triangle():
    return RexGraph.from_simplicial(
        sources=np.array([0, 1, 0], dtype=np.int32),
        targets=np.array([1, 2, 2], dtype=np.int32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )


@pytest.fixture
def tree():
    return RexGraph.from_graph([0, 1, 2], [1, 2, 3])


# Dirac Construction

class TestBuildDirac:

    def test_shape(self, k4):
        B1 = np.asarray(k4.B1, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        D, sizes = _dirac.build_dirac_operator(B1, B2)
        nV, nE, nF = sizes
        N = nV + nE + nF
        assert D.shape == (N, N)

    def test_symmetric(self, k4):
        B1 = np.asarray(k4.B1, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        D, _ = _dirac.build_dirac_operator(B1, B2)
        assert np.allclose(D, D.T, atol=1e-14)

    def test_block_structure(self, filled_triangle):
        """D has B1 in (V,E) block and B2 in (E,F) block, zeros elsewhere."""
        B1 = np.asarray(filled_triangle.B1, dtype=np.float64)
        B2 = np.asarray(filled_triangle.B2_hodge, dtype=np.float64)
        D, (nV, nE, nF) = _dirac.build_dirac_operator(B1, B2)
        # (V,V) block is zero
        assert np.allclose(D[:nV, :nV], 0)
        # (E,E) block is zero
        assert np.allclose(D[nV:nV+nE, nV:nV+nE], 0)
        # (F,F) block is zero
        assert np.allclose(D[nV+nE:, nV+nE:], 0)
        # (V,E) block is B1
        assert np.allclose(D[:nV, nV:nV+nE], B1)
        # (E,F) block is B2
        assert np.allclose(D[nV:nV+nE, nV+nE:], B2)
        # (V,F) block is zero
        assert np.allclose(D[:nV, nV+nE:], 0)

    def test_no_faces(self, tree):
        """Tree with nF=0: D is (nV+nE) x (nV+nE)."""
        B1 = np.asarray(tree.B1, dtype=np.float64)
        B2 = np.asarray(tree.B2_hodge, dtype=np.float64)
        D, (nV, nE, nF) = _dirac.build_dirac_operator(B1, B2)
        assert nF == 0
        assert D.shape == (nV + nE, nV + nE)

    def test_sizes_tuple(self, k4):
        B1 = np.asarray(k4.B1, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        _, (nV, nE, nF) = _dirac.build_dirac_operator(B1, B2)
        assert nV == k4.nV
        assert nE == k4.nE
        assert nF == k4.nF_hodge


# D^2 Verification

class TestDSquared:

    def test_d_squared_equals_laplacians(self, k4):
        """D^2 = blkdiag(L0, L1, L2) for K4."""
        B1 = np.asarray(k4.B1, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        D, (nV, nE, nF) = _dirac.build_dirac_operator(B1, B2)
        L0 = np.asarray(k4.L0, dtype=np.float64)
        L1 = np.asarray(k4.L1, dtype=np.float64)
        L2 = np.asarray(k4.L2, dtype=np.float64)
        ok, err = _dirac.verify_d_squared(D, L0, L1, L2, nV, nE, nF)
        assert ok
        assert err < 1e-10

    def test_d_squared_tree(self, tree):
        """D^2 = blkdiag(L0, L1) for tree (nF=0)."""
        B1 = np.asarray(tree.B1, dtype=np.float64)
        B2 = np.asarray(tree.B2_hodge, dtype=np.float64)
        D, (nV, nE, nF) = _dirac.build_dirac_operator(B1, B2)
        L0 = np.asarray(tree.L0, dtype=np.float64)
        L1 = np.asarray(tree.L1, dtype=np.float64)
        L2 = np.zeros((0, 0), dtype=np.float64)
        ok, err = _dirac.verify_d_squared(D, L0, L1, L2, nV, nE, nF)
        assert ok


# Eigendecomposition

class TestDiracEigen:

    def test_has_negative_eigenvalues(self, k4):
        """D is not PSD: some eigenvalues are negative."""
        D = k4.dirac_operator
        evals, _ = _dirac.dirac_eigen(D)
        assert np.any(evals < -1e-10)

    def test_has_positive_eigenvalues(self, k4):
        D = k4.dirac_operator
        evals, _ = _dirac.dirac_eigen(D)
        assert np.any(evals > 1e-10)

    def test_eigenvectors_orthonormal(self, filled_triangle):
        B1 = np.asarray(filled_triangle.B1, dtype=np.float64)
        B2 = np.asarray(filled_triangle.B2_hodge, dtype=np.float64)
        D, _ = _dirac.build_dirac_operator(B1, B2)
        _, evecs = _dirac.dirac_eigen(D)
        prod = evecs.T @ evecs
        assert np.allclose(prod, np.eye(prod.shape[0]), atol=1e-10)

    def test_reconstruction(self, filled_triangle):
        """D = V diag(evals) V^T."""
        B1 = np.asarray(filled_triangle.B1, dtype=np.float64)
        B2 = np.asarray(filled_triangle.B2_hodge, dtype=np.float64)
        D, _ = _dirac.build_dirac_operator(B1, B2)
        evals, evecs = _dirac.dirac_eigen(D)
        reconstructed = evecs @ np.diag(evals) @ evecs.T
        assert np.allclose(D, reconstructed, atol=1e-10)


# Schrodinger Evolution

class TestSchrodingerEvolve:

    def test_norm_preservation(self, k4):
        """||psi_re||^2 + ||psi_im||^2 = ||psi0||^2 for all t."""
        psi0 = k4.canonical_collapse(vertex_idx=0)
        evals, evecs = _dirac.dirac_eigen(k4.dirac_operator)
        norm0_sq = float(psi0 @ psi0)
        for t in [0.0, 0.5, 1.0, 5.0]:
            psi_re, psi_im = _dirac.schrodinger_evolve(evals, evecs, psi0, t)
            norm_sq = float(psi_re @ psi_re + psi_im @ psi_im)
            assert abs(norm_sq - norm0_sq) < 1e-10

    def test_initial_condition(self, k4):
        """At t=0, psi_re = psi0, psi_im = 0."""
        psi0 = k4.canonical_collapse(vertex_idx=0)
        evals, evecs = _dirac.dirac_eigen(k4.dirac_operator)
        psi_re, psi_im = _dirac.schrodinger_evolve(evals, evecs, psi0, 0.0)
        assert np.allclose(psi_re, psi0, atol=1e-10)
        assert np.allclose(psi_im, 0, atol=1e-10)

    def test_nonzero_imaginary_at_t_positive(self, k4):
        """At t > 0, psi_im is generally nonzero."""
        psi0 = k4.canonical_collapse(vertex_idx=0)
        evals, evecs = _dirac.dirac_eigen(k4.dirac_operator)
        _, psi_im = _dirac.schrodinger_evolve(evals, evecs, psi0, 1.0)
        assert np.linalg.norm(psi_im) > 1e-6


class TestSchrodingerTrajectory:

    def test_shapes(self, k4):
        psi0 = k4.canonical_collapse(vertex_idx=0)
        evals, evecs = _dirac.dirac_eigen(k4.dirac_operator)
        times = np.linspace(0, 2.0, 10, dtype=np.float64)
        traj_re, traj_im, born = _dirac.schrodinger_trajectory(
            evals, evecs, psi0, times)
        N = k4.nV + k4.nE + k4.nF_hodge
        assert traj_re.shape == (10, N)
        assert traj_im.shape == (10, N)
        assert born.shape == (10, N)

    def test_born_sum_to_one(self, k4):
        """Born probabilities sum to 1 at each timestep."""
        psi0 = k4.canonical_collapse(vertex_idx=0)
        evals, evecs = _dirac.dirac_eigen(k4.dirac_operator)
        times = np.linspace(0, 2.0, 5, dtype=np.float64)
        _, _, born = _dirac.schrodinger_trajectory(evals, evecs, psi0, times)
        row_sums = born.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_born_nonnegative(self, k4):
        psi0 = k4.canonical_collapse(vertex_idx=0)
        evals, evecs = _dirac.dirac_eigen(k4.dirac_operator)
        times = np.linspace(0, 2.0, 5, dtype=np.float64)
        _, _, born = _dirac.schrodinger_trajectory(evals, evecs, psi0, times)
        assert np.all(born >= -1e-15)


# Canonical Collapse

class TestCanonicalCollapse:

    def test_normalized(self, k4):
        psi = k4.canonical_collapse(vertex_idx=0)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_face_component_zero(self, k4):
        """Face sector of canonical collapse is exactly zero."""
        psi = k4.canonical_collapse(vertex_idx=0)
        nV, nE, nF = k4.nV, k4.nE, k4.nF_hodge
        face_part = psi[nV + nE:]
        assert np.allclose(face_part, 0, atol=1e-14)

    def test_vertex_component_nonzero(self, k4):
        psi = k4.canonical_collapse(vertex_idx=0)
        vertex_part = psi[:k4.nV]
        assert np.linalg.norm(vertex_part) > 0

    def test_edge_component_is_b1t_delta(self, k4):
        """Edge sector = B1^T delta_v (before normalization)."""
        B1 = np.asarray(k4.B1, dtype=np.float64)
        nV, nE, nF = k4.nV, k4.nE, k4.nF_hodge
        psi = _dirac.canonical_collapse(B1, nV, nE, nF, 0)
        # Reconstruct unnormalized
        unnorm = np.zeros(nV + nE + nF, dtype=np.float64)
        unnorm[0] = 1.0
        unnorm[nV:nV+nE] = B1[0, :]
        unnorm /= np.linalg.norm(unnorm)
        assert np.allclose(psi, unnorm, atol=1e-10)

    def test_different_vertices(self, k4):
        psi0 = k4.canonical_collapse(vertex_idx=0)
        psi1 = k4.canonical_collapse(vertex_idx=1)
        assert not np.allclose(psi0, psi1)


# Born Graded

class TestBornGraded:

    def test_per_cell_sums_to_one(self, k4):
        psi_re, psi_im = k4.graded_state(t=1.0)
        per_cell, per_dim = k4.born_graded(psi_re, psi_im)
        assert abs(per_cell.sum() - 1.0) < 1e-10

    def test_per_dim_sums_to_one(self, k4):
        psi_re, psi_im = k4.graded_state(t=1.0)
        _, per_dim = k4.born_graded(psi_re, psi_im)
        assert abs(per_dim.sum() - 1.0) < 1e-10

    def test_per_dim_shape(self, k4):
        psi_re, psi_im = k4.graded_state(t=0.0)
        _, per_dim = k4.born_graded(psi_re, psi_im)
        assert per_dim.shape == (3,)

    def test_nonnegative(self, k4):
        psi_re, psi_im = k4.graded_state(t=0.5)
        per_cell, per_dim = k4.born_graded(psi_re, psi_im)
        assert np.all(per_cell >= -1e-15)
        assert np.all(per_dim >= -1e-15)


# Energy Partition

class TestEnergyPartition:

    def test_sums_to_one(self, k4):
        psi_re, psi_im = k4.graded_state(t=1.0)
        ep = k4.energy_partition(psi_re, psi_im)
        assert abs(ep.sum() - 1.0) < 1e-10

    def test_shape(self, k4):
        psi_re, psi_im = k4.graded_state(t=0.0)
        ep = k4.energy_partition(psi_re, psi_im)
        assert ep.shape == (3,)

    def test_matches_born_dim(self, k4):
        """Energy partition equals per_dim from born_graded."""
        psi_re, psi_im = k4.graded_state(t=1.0)
        _, per_dim = k4.born_graded(psi_re, psi_im)
        ep = k4.energy_partition(psi_re, psi_im)
        # They should agree (energy_partition normalizes per_dim)
        expected = per_dim / per_dim.sum()
        assert np.allclose(ep, expected, atol=1e-10)


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_dirac_operator_shape(self, k4):
        D = k4.dirac_operator
        N = k4.nV + k4.nE + k4.nF_hodge
        assert D.shape == (N, N)

    def test_dirac_operator_symmetric(self, k4):
        D = k4.dirac_operator
        assert np.allclose(D, D.T)

    def test_dirac_eigenvalues_shape(self, k4):
        ev = k4.dirac_eigenvalues
        N = k4.nV + k4.nE + k4.nF_hodge
        assert ev.shape == (N,)

    def test_graded_state_returns_pair(self, k4):
        psi_re, psi_im = k4.graded_state(t=0.5)
        N = k4.nV + k4.nE + k4.nF_hodge
        assert psi_re.shape == (N,)
        assert psi_im.shape == (N,)

    def test_graded_trajectory_keys(self, k4):
        times = np.linspace(0, 1.0, 5, dtype=np.float64)
        result = k4.graded_trajectory(times)
        for key in ['traj_re', 'traj_im', 'born', 'times']:
            assert key in result

    def test_tree_dirac(self, tree):
        """Tree (no faces) still produces valid Dirac operator."""
        D = tree.dirac_operator
        N = tree.nV + tree.nE
        assert D.shape == (N, N)
        assert np.allclose(D, D.T)
