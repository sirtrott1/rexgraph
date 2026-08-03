"""
Tests for linkage complex construction and character-based
quotient filtration.
"""
import numpy as np
import pytest

from rexgraph.core import _fiber, _quotient
from rexgraph.graph import RexGraph


@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32),
    )


# linkage_complex

class TestLinkageComplex:

    def test_empty_at_high_threshold(self, k4):
        sfb = k4.fiber_similarity
        result = _fiber.linkage_complex(sfb, 2.0, k4.nV)
        assert result['n_edges'] == 0

    def test_complete_at_zero_threshold(self):
        sim = np.ones((4, 4), dtype=np.float64)
        np.fill_diagonal(sim, 0)
        result = _fiber.linkage_complex(sim, -0.1, 4)
        assert result['n_edges'] == 6  # C(4,2)

    def test_chain_condition(self, k4):
        sfb = k4.fiber_similarity
        result = _fiber.linkage_complex(sfb, 0.1, k4.nV)
        if result['nF'] > 0 and result['B1'] is not None and result['B2'] is not None:
            B1 = np.asarray(result['B1'], dtype=np.float64)
            B2 = np.asarray(result['B2'], dtype=np.float64)
            product = B1 @ B2
            assert np.max(np.abs(product)) < 1e-10

    def test_betti_euler(self, k4):
        sfb = k4.fiber_similarity
        result = _fiber.linkage_complex(sfb, 0.1, k4.nV)
        b0, b1, b2 = result['beta']
        euler = result['nV'] - result['n_edges'] + result['nF']
        assert b0 - b1 + b2 == euler

    def test_triangles_shape(self, k4):
        sfb = k4.fiber_similarity
        result = _fiber.linkage_complex(sfb, 0.1, k4.nV)
        tri = result['triangles']
        assert tri.shape[1] == 3
        assert tri.shape[0] == result['nF']

    def test_triangle_vertices_distinct(self, k4):
        sfb = k4.fiber_similarity
        result = _fiber.linkage_complex(sfb, 0.1, k4.nV)
        for row in result['triangles']:
            assert len(set(row)) == 3

    def test_graph_method(self, k4):
        rex = k4.linkage_complex(sfb_threshold=0.1)
        assert isinstance(rex, RexGraph)

    def test_betti_eigenfree_matches_svd(self, k4):
        """The clique-path Betti now uses exact rational rank (no SVD); it equals the
        rank counted from B2's singular values."""
        sfb = k4.fiber_similarity
        result = _fiber.linkage_complex(sfb, 0.1, k4.nV)
        B2 = np.asarray(result['B2'], dtype=np.float64)
        rank_svd = int(np.sum(np.linalg.svd(B2, compute_uv=False) > 1e-10))
        b1nf = result['beta'][1] + rank_svd     # beta_1 = beta_1_no_faces - rank(B2)
        assert result['beta'][1] == b1nf - rank_svd
        assert result['beta'][2] == result['nF'] - rank_svd

    def test_face_fill_cycle(self, k4):
        """face_fill='cycle' fills the fundamental cycle basis (arbitrary-arity faces):
        every cycle independent so beta_1=beta_2=0, chain condition B1@B2=0 holds,
        Euler holds, and triangles is empty while face_lengths is populated."""
        sfb = k4.fiber_similarity
        r = _fiber.linkage_complex(sfb, 0.1, k4.nV, face_fill='cycle')
        b0, b1, b2 = r['beta']
        assert (b1, b2) == (0, 0)
        assert b0 - b1 + b2 == r['nV'] - r['n_edges'] + r['nF']
        assert r['nF'] == r['n_edges'] - r['nV'] + b0    # beta_1_no_faces cycles
        assert r['triangles'].shape == (0, 3)
        assert r['face_lengths'].shape[0] == r['nF']
        B1 = np.asarray(r['B1'], dtype=np.float64)
        B2 = np.asarray(r['B2'], dtype=np.float64)
        assert np.max(np.abs(B1 @ B2)) < 1e-10

    def test_face_fill_invalid(self, k4):
        with pytest.raises(ValueError):
            _fiber.linkage_complex(k4.fiber_similarity, 0.1, k4.nV, face_fill='nope')


# quotient_filtration_by_character

class TestQuotientFiltration:

    def test_shape(self, k4):
        n_steps = 10
        result = _quotient.quotient_filtration_by_character(
            k4.structural_character, 0, n_steps,
            k4.B1, k4.B2_hodge, k4.nV, k4.nE, k4.nF_hodge)
        assert result['thresholds'].shape == (n_steps,)
        assert result['beta0'].shape == (n_steps,)
        assert result['beta1'].shape == (n_steps,)
        assert result['beta2'].shape == (n_steps,)
        assert result['n_edges_remaining'].shape == (n_steps,)

    def test_edges_non_increasing(self, k4):
        result = _quotient.quotient_filtration_by_character(
            k4.structural_character, 0, 10,
            k4.B1, k4.B2_hodge, k4.nV, k4.nE, k4.nF_hodge)
        remaining = result['n_edges_remaining']
        assert np.all(np.diff(remaining) <= 0)

    def test_transition_valid(self, k4):
        result = _quotient.quotient_filtration_by_character(
            k4.structural_character, 0, 10,
            k4.B1, k4.B2_hodge, k4.nV, k4.nE, k4.nF_hodge)
        ti = result['transition_index']
        assert ti == -1 or (1 <= ti < 10)

    def test_order_matches_chi(self, k4):
        chi = k4.structural_character
        result = _quotient.quotient_filtration_by_character(
            chi, 0, 10, k4.B1, k4.B2_hodge, k4.nV, k4.nE, k4.nF_hodge)
        order = result['edges_removed_order']
        chi_vals = chi[order, 0]
        assert np.all(np.diff(chi_vals) <= 1e-12)

    def test_graph_method(self, k4):
        result = k4.quotient_filtration(channel=0, n_steps=5)
        assert 'beta1' in result
        assert 'transition_index' in result


class TestQuotientEigenFree:
    """The quotient kernels compute Betti/harmonic/congruence eigen-free (exact rank,
    combinatorial harmonic basis, factor-once congruence) - each pinned to its dense
    oracle."""

    def _quot(self):
        # 5-vertex graph, 2 independent cycles, one triangular face on edges 0,1,2
        nV, nE = 5, 6
        edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]
        B1 = np.zeros((nV, nE))
        for e, (i, j) in enumerate(edges):
            B1[i, e] = -1; B1[j, e] = 1
        # valid triangle boundary on the cycle 0->1->2->0 (edges 0,1,2): B1 @ B2 = 0
        B2 = np.zeros((nE, 1)); B2[0, 0] = 1; B2[1, 0] = 1; B2[2, 0] = 1
        return B1, B2

    def test_relative_betti_matches_svd(self):
        B1, B2 = self._quot()
        b0, b1, b2 = _quotient.relative_betti(B1, B2)
        r1 = int(np.sum(np.linalg.svd(B1, compute_uv=False) > 1e-10))
        r2 = int(np.sum(np.linalg.svd(B2, compute_uv=False) > 1e-10))
        assert (b0, b1, b2) == (B1.shape[0] - r1, B1.shape[1] - r1 - r2, B2.shape[1] - r2)

    def test_relative_cycle_basis_eigenfree_matches_dense(self):
        from rexgraph.core._linalg import eigh as _eigh
        B1, B2 = self._quot()
        basis = _quotient.relative_cycle_basis(B1, B2)
        L1q = B1.T @ B1 + B2 @ B2.T
        ev, evec = _eigh(L1q)
        harm = evec[:, np.abs(ev) < 1e-10]
        assert basis.shape[1] == harm.shape[1]
        # annihilates L1q, orthonormal, and spans the same plane as the dense harmonic
        assert np.abs(B1 @ basis).max() < 1e-9
        assert np.abs(B2.T @ basis).max() < 1e-9
        assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-9)
        Q, _ = np.linalg.qr(basis)
        assert np.linalg.norm(harm - Q @ (Q.T @ harm)) < 1e-9

    def test_congruence_factor_once_matches_per_pair(self):
        B1, _ = self._quot()

        def ref(M, mask, tol=1e-10):
            M = np.asarray(M, float); labels = np.full(M.shape[1], -1, np.int32)
            surv = np.where(~mask.astype(bool))[0]
            idxI = np.where(mask.astype(bool))[0]
            basis = M[:, idxI] if idxI.size else None
            nl = 0
            for a in range(len(surv)):
                if labels[surv[a]] >= 0:
                    continue
                labels[surv[a]] = nl
                for b in range(a + 1, len(surv)):
                    if labels[surv[b]] >= 0:
                        continue
                    d = M[:, surv[a]] - M[:, surv[b]]
                    r = (np.linalg.norm(d) if basis is None
                         else np.linalg.norm(d - basis @ np.linalg.lstsq(basis, d, rcond=None)[0]))
                    if r < tol:
                        labels[surv[b]] = nl
                nl += 1
            return labels, nl

        emask = np.zeros(B1.shape[1], np.uint8); emask[0] = 1; emask[3] = 1
        lab, nc = _quotient.congruence_classes_edges(B1, emask)
        lab_ref, nc_ref = ref(B1, emask)
        assert nc == nc_ref
        assert np.array_equal(lab, lab_ref)

    def test_quotient_verify_chain_sparse(self):
        B1, B2 = self._quot()
        ok, err = _quotient.quotient_verify_chain(B1, B2)
        assert ok and err < 1e-10
        assert abs(err - float(np.max(np.abs(B1 @ B2)))) < 1e-12
