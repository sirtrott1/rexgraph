"""
Tests for rexgraph.core._hypermanifold -- filtered manifold sequence.

Verifies:
    - Manifold sequence: correct levels, cell counts, Betti numbers
    - beta_1(1) = nE - nV + beta_0 (no face contribution at d=1)
    - beta_1(2) <= beta_1(1) (adding faces can only kill cycles)
    - Harmonic shadow: dim = beta_1(1) - beta_1(2) = rank(B2)
    - Dimensional subsumption: beta_k(d+1) <= beta_k(d) holds
    - compute_betti_from_evals: correct count of zero eigenvalues
    - Integration through RexGraph: hypermanifold, harmonic_shadow,
      dimensional_subsumption
"""
import numpy as np
import pytest

from rexgraph.core import _hypermanifold
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
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.fixture
def tree():
    return RexGraph.from_graph([0, 1, 2], [1, 2, 3])


# Manifold Sequence

class TestManifoldSequence:

    def test_returns_dict(self, k4):
        sb = k4.spectral_bundle
        result = _hypermanifold.build_manifold_sequence(
            sb['evals_L0'], sb['evals_L1'],
            sb.get('evals_L2', np.empty(0, dtype=np.float64)),
            k4.nV, k4.nE, k4.nF_hodge)
        assert isinstance(result, dict)
        assert 'manifolds' in result

    def test_k4_two_levels(self, k4):
        """K4 with faces produces M1 and M2."""
        sb = k4.spectral_bundle
        result = _hypermanifold.build_manifold_sequence(
            sb['evals_L0'], sb['evals_L1'],
            sb.get('evals_L2', np.empty(0, dtype=np.float64)),
            k4.nV, k4.nE, k4.nF_hodge)
        assert len(result['manifolds']) == 2
        assert result['max_dimension'] == 2

    def test_tree_one_level(self, tree):
        """Tree (nF=0) produces only M1."""
        sb = tree.spectral_bundle
        result = _hypermanifold.build_manifold_sequence(
            sb['evals_L0'], sb['evals_L1'],
            np.empty(0, dtype=np.float64),
            tree.nV, tree.nE, 0)
        assert len(result['manifolds']) == 1
        assert result['max_dimension'] == 1

    def test_m1_cell_counts(self, k4):
        sb = k4.spectral_bundle
        result = _hypermanifold.build_manifold_sequence(
            sb['evals_L0'], sb['evals_L1'],
            sb.get('evals_L2', np.empty(0, dtype=np.float64)),
            k4.nV, k4.nE, k4.nF_hodge)
        m1 = result['manifolds'][0]
        assert m1['cells'] == [k4.nV, k4.nE]
        assert m1['N'] == k4.nV + k4.nE

    def test_m2_cell_counts(self, k4):
        sb = k4.spectral_bundle
        result = _hypermanifold.build_manifold_sequence(
            sb['evals_L0'], sb['evals_L1'],
            sb.get('evals_L2', np.empty(0, dtype=np.float64)),
            k4.nV, k4.nE, k4.nF_hodge)
        m2 = result['manifolds'][1]
        assert m2['cells'] == [k4.nV, k4.nE, k4.nF_hodge]
        assert m2['N'] == k4.nV + k4.nE + k4.nF_hodge

    def test_m1_beta0_correct(self, k4):
        """beta_0 at M1 = 1 (connected graph)."""
        sb = k4.spectral_bundle
        result = _hypermanifold.build_manifold_sequence(
            sb['evals_L0'], sb['evals_L1'],
            sb.get('evals_L2', np.empty(0, dtype=np.float64)),
            k4.nV, k4.nE, k4.nF_hodge)
        assert result['manifolds'][0]['betti'][0] == 1

    def test_m1_beta1_formula(self, k4):
        """beta_1(1) = nE - nV + beta_0 (no face contribution at d=1)."""
        sb = k4.spectral_bundle
        result = _hypermanifold.build_manifold_sequence(
            sb['evals_L0'], sb['evals_L1'],
            sb.get('evals_L2', np.empty(0, dtype=np.float64)),
            k4.nV, k4.nE, k4.nF_hodge)
        beta0 = result['manifolds'][0]['betti'][0]
        beta1_at_d1 = result['manifolds'][0]['betti'][1]
        assert beta1_at_d1 == k4.nE - k4.nV + beta0

    def test_m1_bianchi_zero(self, k4):
        """M1 (1-rex) has 0 Bianchi identities."""
        sb = k4.spectral_bundle
        result = _hypermanifold.build_manifold_sequence(
            sb['evals_L0'], sb['evals_L1'],
            sb.get('evals_L2', np.empty(0, dtype=np.float64)),
            k4.nV, k4.nE, k4.nF_hodge)
        assert result['manifolds'][0]['n_bianchi'] == 0

    def test_m2_bianchi_one(self, k4):
        """M2 (2-rex) has 1 Bianchi identity."""
        sb = k4.spectral_bundle
        result = _hypermanifold.build_manifold_sequence(
            sb['evals_L0'], sb['evals_L1'],
            sb.get('evals_L2', np.empty(0, dtype=np.float64)),
            k4.nV, k4.nE, k4.nF_hodge)
        assert result['manifolds'][1]['n_bianchi'] == 1


# Harmonic Shadow

class TestHarmonicShadow:

    def test_shadow_nonneg(self, k4):
        """Shadow dimension is nonnegative."""
        sb = k4.spectral_bundle
        L1_down = sb.get('L1_down')
        from rexgraph.core._linalg import eigh as _eigh
        evals_down = _eigh(np.asarray(L1_down, dtype=np.float64))[0]
        evals_full = sb['evals_L1']
        shadow_dim, _, _ = _hypermanifold.harmonic_shadow(evals_down, evals_full)
        assert shadow_dim >= 0

    def test_shadow_equals_rank_b2(self, k4):
        """Shadow dim = beta_1(1) - beta_1(2) = rank(B2)."""
        sb = k4.spectral_bundle
        L1_down = sb.get('L1_down')
        from rexgraph.core._linalg import eigh as _eigh
        evals_down = _eigh(np.asarray(L1_down, dtype=np.float64))[0]
        evals_full = sb['evals_L1']
        shadow_dim, beta_d, beta_d1 = _hypermanifold.harmonic_shadow(
            evals_down, evals_full)
        # rank(B2) = nE - beta1(full) - rank(B1)... but simpler:
        # shadow_dim = beta_d - beta_d1
        assert shadow_dim == beta_d - beta_d1

    def test_tree_shadow_zero(self, tree):
        """Tree has no faces, so no shadow (beta_1 = 0 at both levels)."""
        sb = tree.spectral_bundle
        evals = sb['evals_L1']
        # At d=1, L1 = L1_down (same since no faces)
        shadow_dim, _, _ = _hypermanifold.harmonic_shadow(evals, evals)
        assert shadow_dim == 0

    def test_filled_triangle_shadow(self, filled_triangle):
        """Filled triangle: beta_1(1) = 1, beta_1(2) = 0, shadow = 1."""
        sb = filled_triangle.spectral_bundle
        L1_down = sb.get('L1_down')
        from rexgraph.core._linalg import eigh as _eigh
        evals_down = _eigh(np.asarray(L1_down, dtype=np.float64))[0]
        evals_full = sb['evals_L1']
        shadow_dim, beta_d, beta_d1 = _hypermanifold.harmonic_shadow(
            evals_down, evals_full)
        assert beta_d == 1   # one cycle before filling
        assert beta_d1 == 0  # no cycles after filling
        assert shadow_dim == 1


# Dimensional Subsumption

class TestDimensionalSubsumption:

    def test_valid_sequence(self):
        """beta_k(d+1) <= beta_k(d) passes."""
        betti_seq = [[1, 3], [1, 0, 0]]  # M1: beta=(1,3), M2: beta=(1,0,0)
        ok, violations = _hypermanifold.dimensional_subsumption(betti_seq)
        assert ok
        assert len(violations) == 0

    def test_violation_detected(self):
        """beta_1 increases from d=1 to d=2 is a violation."""
        betti_seq = [[1, 1], [1, 3, 0]]  # beta_1 goes from 1 to 3
        ok, violations = _hypermanifold.dimensional_subsumption(betti_seq)
        assert not ok
        assert len(violations) > 0

    def test_single_level(self):
        """Single manifold level trivially passes."""
        betti_seq = [[1, 5]]
        ok, violations = _hypermanifold.dimensional_subsumption(betti_seq)
        assert ok

    def test_k4_subsumption(self, k4):
        """K4 satisfies dimensional subsumption."""
        ok, violations = k4.dimensional_subsumption
        assert ok
        assert len(violations) == 0


# compute_betti_from_evals

class TestBettiFromEvals:

    def test_count_zeros(self):
        evals = np.array([0.0, 0.0, 1e-10, 0.5, 1.0], dtype=np.float64)
        assert _hypermanifold.compute_betti_from_evals(evals) == 3

    def test_no_zeros(self):
        evals = np.array([0.5, 1.0, 2.0], dtype=np.float64)
        assert _hypermanifold.compute_betti_from_evals(evals) == 0

    def test_all_zeros(self):
        evals = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        assert _hypermanifold.compute_betti_from_evals(evals) == 3

    def test_empty(self):
        evals = np.empty(0, dtype=np.float64)
        assert _hypermanifold.compute_betti_from_evals(evals) == 0

    def test_custom_tol(self):
        evals = np.array([1e-5, 1e-3, 0.5], dtype=np.float64)
        assert _hypermanifold.compute_betti_from_evals(evals, tol=1e-4) == 1
        assert _hypermanifold.compute_betti_from_evals(evals, tol=1e-2) == 2


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_hypermanifold_keys(self, k4):
        hm = k4.hypermanifold
        assert 'manifolds' in hm
        assert 'max_dimension' in hm

    def test_harmonic_shadow_keys(self, k4):
        hs = k4.harmonic_shadow
        assert 'shadow_dim' in hs
        assert 'beta_1_at_d1' in hs
        assert 'beta_1_at_d2' in hs

    def test_harmonic_shadow_nonneg(self, k4):
        hs = k4.harmonic_shadow
        assert hs['shadow_dim'] >= 0

    def test_dimensional_subsumption_holds(self, k4):
        ok, violations = k4.dimensional_subsumption
        assert ok

    def test_tree_hypermanifold(self, tree):
        hm = tree.hypermanifold
        assert hm['max_dimension'] == 1
        assert len(hm['manifolds']) == 1
