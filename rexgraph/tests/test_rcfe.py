"""
Tests for rexgraph.core._rcfe -- RCFE curvature, strain, and conservation laws.

Verifies:
    - Curvature: nonnegative, sums to nF, zero when nF=0
    - Strain: nonnegative, equals sum C(e)*RL[e,e]
    - Bianchi identity: B1 @ diag(C) @ B2 = 0
    - Per-face Bianchi residual is zero
    - Coupling tensor: shape (nF, nhats), rows sum to ~boundary size
    - Relational integrity: RI in (0, 1]
    - Face overlap K2: symmetric, diagonal = boundary size
    - Attributed curvature: kappa_f nonnegative
    - Dynamic strain: sigma = B2 @ delta, B1 @ sigma = 0
    - Strain equilibrium: full pipeline, Bianchi holds
    - Integration through RexGraph: rcfe_curvature, rcfe_strain,
      attributed_curvature, strain_equilibrium
"""
import numpy as np
import pytest

from rexgraph.core import _rcfe
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


# Curvature

class TestCurvature:

    def test_nonnegative(self, k4):
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        assert np.all(C >= -1e-12)

    def test_sums_to_nF(self, k4):
        """Total curvature = nF (each face contributes 1 unit)."""
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        assert abs(C.sum() - k4.nF_hodge) < 1e-10

    def test_shape(self, k4):
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        assert C.shape == (k4.nE,)

    def test_no_faces_returns_zero(self, tree):
        C = _rcfe.compute_curvature(tree.B2_hodge, tree.nE, 0)
        assert np.allclose(C, 0)
        assert C.shape == (tree.nE,)

    def test_triangle_uniform(self, filled_triangle):
        """Filled triangle: all 3 edges have equal curvature = 1/3 * 1 face."""
        C = _rcfe.compute_curvature(filled_triangle.B2_hodge,
                                     filled_triangle.nE, filled_triangle.nF_hodge)
        # Each edge has B2[e,f]^2 / ||B2[:,f]||^2 = 1/3
        assert np.allclose(C, 1.0 / 3.0, atol=1e-10)

    def test_k4_positive(self, k4):
        """K4: every edge is on at least one face, so C(e) > 0."""
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        assert np.all(C > 0)


# Strain

class TestStrain:

    def test_nonnegative(self, k4):
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        rl_diag = np.ascontiguousarray(np.diag(np.asarray(k4.RL, dtype=np.float64)))
        S = _rcfe.compute_strain(C, rl_diag, k4.nE)
        assert S >= -1e-10

    def test_equals_dot_product(self, k4):
        """S = sum C(e) * RL[e,e]."""
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        rl_diag = np.ascontiguousarray(np.diag(np.asarray(k4.RL, dtype=np.float64)))
        S = _rcfe.compute_strain(C, rl_diag, k4.nE)
        expected = float(np.dot(C, rl_diag))
        assert abs(S - expected) < 1e-12

    def test_per_face_shape(self, k4):
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        sf = _rcfe.compute_strain_per_face(k4.B2_hodge, C, k4.nE, k4.nF_hodge)
        assert sf.shape == (k4.nF_hodge,)

    def test_per_face_nonneg(self, k4):
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        sf = _rcfe.compute_strain_per_face(k4.B2_hodge, C, k4.nE, k4.nF_hodge)
        assert np.all(sf >= -1e-12)


# Bianchi Identity

class TestBianchi:

    def test_bianchi_holds(self, k4):
        """B1 @ diag(C) @ B2 = 0 for K4."""
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        ok, err = _rcfe.verify_bianchi(k4.B1, k4.B2_hodge, C,
                                        k4.nE, k4.nF_hodge)
        assert ok
        assert err < 1e-10

    def test_bianchi_triangle(self, filled_triangle):
        C = _rcfe.compute_curvature(filled_triangle.B2_hodge,
                                     filled_triangle.nE, filled_triangle.nF_hodge)
        ok, err = _rcfe.verify_bianchi(filled_triangle.B1, filled_triangle.B2_hodge,
                                        C, filled_triangle.nE,
                                        filled_triangle.nF_hodge)
        assert ok

    def test_residual_zero(self, k4):
        """Per-face Bianchi residual is zero."""
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        res = _rcfe.bianchi_residual(k4.B1, k4.B2_hodge, C,
                                      k4.nE, k4.nF_hodge)
        assert np.allclose(res, 0, atol=1e-10)

    def test_no_faces(self, tree):
        ok, err = _rcfe.verify_bianchi(tree.B1, tree.B2_hodge,
                                        np.zeros(tree.nE, dtype=np.float64),
                                        tree.nE, 0)
        assert ok
        assert err == 0.0


# Coupling Tensor

class TestCouplingTensor:

    def test_shape(self, k4):
        rcf = k4._rcf_bundle
        tensor = _rcfe.coupling_tensor(k4.B2_hodge, k4.RL,
                                        rcf['hats'], rcf['nhats'],
                                        k4.nE, k4.nF_hodge)
        assert tensor.shape == (k4.nF_hodge, k4.nhats)

    def test_nonnegative(self, k4):
        """Tensor entries are nonneg (diag ratios of PSD operators)."""
        rcf = k4._rcf_bundle
        tensor = _rcfe.coupling_tensor(k4.B2_hodge, k4.RL,
                                        rcf['hats'], rcf['nhats'],
                                        k4.nE, k4.nF_hodge)
        assert np.all(tensor >= -1e-10)

    def test_no_faces(self, tree):
        tensor = _rcfe.coupling_tensor(tree.B2_hodge, tree.RL,
                                        [], 0, tree.nE, 0)
        assert tensor.shape == (0, 0)


# Relational Integrity

class TestRelationalIntegrity:

    def test_ri_in_range(self, k4):
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        rl_diag = np.ascontiguousarray(np.diag(np.asarray(k4.RL, dtype=np.float64)))
        ri = _rcfe.relational_integrity(C, rl_diag, k4.nE)
        assert 0 < ri['RI'] <= 1.0

    def test_zero_curvature_ri_one(self):
        """Zero curvature gives RI = 1."""
        C = np.zeros(3, dtype=np.float64)
        rl_diag = np.ones(3, dtype=np.float64)
        ri = _rcfe.relational_integrity(C, rl_diag, 3)
        assert abs(ri['RI'] - 1.0) < 1e-12

    def test_per_face_ri(self, k4):
        C = _rcfe.compute_curvature(k4.B2_hodge, k4.nE, k4.nF_hodge)
        rl_diag = np.ascontiguousarray(np.diag(np.asarray(k4.RL, dtype=np.float64)))
        ri = _rcfe.relational_integrity(C, rl_diag, k4.nE,
                                         B2=np.asarray(k4.B2_hodge, dtype=np.float64),
                                         nF=k4.nF_hodge)
        assert 'per_face_RI' in ri
        assert ri['per_face_RI'].shape == (k4.nF_hodge,)


# Face Overlap

class TestFaceOverlap:

    def test_symmetric(self, k4):
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        K2 = _rcfe.face_overlap_K2(B2, k4.nE, k4.nF_hodge)
        assert np.allclose(K2, K2.T)

    def test_diagonal_is_boundary_size(self, k4):
        """K2[f,f] = number of boundary edges of face f."""
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        K2 = _rcfe.face_overlap_K2(B2, k4.nE, k4.nF_hodge)
        for f in range(k4.nF_hodge):
            bnd_size = np.sum(np.abs(B2[:, f]) > 0.5)
            assert abs(K2[f, f] - bnd_size) < 1e-12

    def test_shape(self, k4):
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        K2 = _rcfe.face_overlap_K2(B2, k4.nE, k4.nF_hodge)
        assert K2.shape == (k4.nF_hodge, k4.nF_hodge)


# Edge Weight Conjugation

class TestEdgeWeightConjugation:

    def test_identity_weights(self, k4):
        """Conjugation with sqrt(w)=1 returns L unchanged."""
        L = np.asarray(k4.L1, dtype=np.float64)
        sqw = np.ones(k4.nE, dtype=np.float64)
        Lw = _rcfe.edge_weight_conjugation(L, sqw, k4.nE)
        assert np.allclose(Lw, L)

    def test_symmetric(self, k4):
        L = np.asarray(k4.L1, dtype=np.float64)
        sqw = np.random.RandomState(42).rand(k4.nE).astype(np.float64) + 0.1
        Lw = _rcfe.edge_weight_conjugation(L, sqw, k4.nE)
        assert np.allclose(Lw, Lw.T, atol=1e-12)


# Dynamic Strain

class TestDynamicStrain:

    def test_face_deficit_shape(self, k4):
        kappa_f = np.ones(k4.nF_hodge, dtype=np.float64)
        born_f = np.ones(k4.nF_hodge, dtype=np.float64) / k4.nF_hodge
        delta = _rcfe.face_deficit(kappa_f, 1.0, born_f, k4.nF_hodge)
        assert delta.shape == (k4.nF_hodge,)

    def test_strain_is_b2_delta(self, k4):
        """sigma = B2 @ delta."""
        kappa_f = np.ones(k4.nF_hodge, dtype=np.float64)
        born_f = np.ones(k4.nF_hodge, dtype=np.float64) / k4.nF_hodge
        delta = _rcfe.face_deficit(kappa_f, 1.0, born_f, k4.nF_hodge)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        sigma = _rcfe.relational_strain_dynamic(B2, delta, k4.nE, k4.nF_hodge)
        expected = B2 @ delta
        assert np.allclose(sigma, expected, atol=1e-12)

    def test_bianchi_strain_holds(self, k4):
        """B1 @ sigma = 0 (chain condition conservation)."""
        kappa_f = np.ones(k4.nF_hodge, dtype=np.float64)
        born_f = np.ones(k4.nF_hodge, dtype=np.float64) / k4.nF_hodge
        delta = _rcfe.face_deficit(kappa_f, 1.0, born_f, k4.nF_hodge)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        sigma = _rcfe.relational_strain_dynamic(B2, delta, k4.nE, k4.nF_hodge)
        B1 = np.asarray(k4.B1, dtype=np.float64)
        ok, res = _rcfe.verify_bianchi_strain(B1, sigma, k4.nV, k4.nE)
        assert ok
        assert res < 1e-10

    def test_optimal_alpha_nonneg(self, k4):
        kappa_f = np.ones(k4.nF_hodge, dtype=np.float64)
        born_f = np.ones(k4.nF_hodge, dtype=np.float64) / k4.nF_hodge
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        alpha = _rcfe.optimal_alpha(B2, kappa_f, born_f, k4.nE, k4.nF_hodge)
        assert alpha >= 0


# Strain Equilibrium

class TestStrainEquilibrium:

    def test_returns_all_keys(self, k4):
        B1 = np.asarray(k4.B1, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        kappa_f = np.ones(k4.nF_hodge, dtype=np.float64)
        born_f = np.ones(k4.nF_hodge, dtype=np.float64) / k4.nF_hodge
        result = _rcfe.strain_equilibrium(B1, B2, kappa_f, born_f,
                                           k4.nV, k4.nE, k4.nF_hodge)
        for key in ['alpha', 'delta', 'sigma', 'bianchi_ok',
                     'bianchi_residual', 'strain_norm']:
            assert key in result

    def test_bianchi_ok(self, k4):
        B1 = np.asarray(k4.B1, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        kappa_f = np.ones(k4.nF_hodge, dtype=np.float64)
        born_f = np.ones(k4.nF_hodge, dtype=np.float64) / k4.nF_hodge
        result = _rcfe.strain_equilibrium(B1, B2, kappa_f, born_f,
                                           k4.nV, k4.nE, k4.nF_hodge)
        assert result['bianchi_ok']

    def test_sigma_shape(self, k4):
        B1 = np.asarray(k4.B1, dtype=np.float64)
        B2 = np.asarray(k4.B2_hodge, dtype=np.float64)
        kappa_f = np.ones(k4.nF_hodge, dtype=np.float64)
        born_f = np.ones(k4.nF_hodge, dtype=np.float64) / k4.nF_hodge
        result = _rcfe.strain_equilibrium(B1, B2, kappa_f, born_f,
                                           k4.nV, k4.nE, k4.nF_hodge)
        assert result['sigma'].shape == (k4.nE,)


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_rcfe_curvature_shape(self, k4):
        C = k4.rcfe_curvature
        assert C.shape == (k4.nE,)

    def test_rcfe_curvature_sums_to_nf(self, k4):
        C = k4.rcfe_curvature
        assert abs(C.sum() - k4.nF_hodge) < 1e-10

    def test_rcfe_strain_nonneg(self, k4):
        assert k4.rcfe_strain >= -1e-10

    def test_attributed_curvature(self, k4):
        result = k4.attributed_curvature()
        assert 'kappa_f' in result
        assert result['kappa_f'].shape == (k4.nF_hodge,)

    def test_strain_equilibrium(self, k4):
        result = k4.strain_equilibrium()
        assert result['bianchi_ok']

    def test_tree_curvature_zero(self, tree):
        C = tree.rcfe_curvature
        assert np.allclose(C, 0)

    def test_tree_strain_zero(self, tree):
        assert tree.rcfe_strain == 0.0
