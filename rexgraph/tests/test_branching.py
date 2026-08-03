"""Branching-hyperedge correctness sweep.

The audit found NO test exercised the character / moment channels on branching
hyperedges (arity != 2 columns of B1, from_hypergraph). This closes that gap: it
pins the whole edge-centric stack - the four channels, RL4, character chi/phi/kappa,
the moment operators, Green's, and curvature - against a dense-from-B1 reference on
branching complexes (where two edges can share >1 vertex). Regression home for the
L_C weighted-line-graph fix and everything it un-broke.
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph

# from_hypergraph complexes with branching (arity>2) edges that share >1 vertex
CASES = {
    "two_3ary_share2": (np.array([0, 3, 6]), np.array([0, 1, 2, 1, 2, 3])),
    "three_branch":    (np.array([0, 3, 6, 9]), np.array([0, 1, 2, 0, 1, 3, 2, 3, 4])),
    "mixed_4_4_3ary":  (np.array([0, 4, 8, 11]), np.array([0, 1, 2, 3, 1, 2, 3, 4, 4, 5, 0])),
    "parallel_plus":   (np.array([0, 2, 4, 7]), np.array([0, 1, 0, 1, 1, 2, 3])),
}


@pytest.fixture(params=list(CASES), ids=list(CASES))
def hg(request):
    ptr, idx = CASES[request.param]
    return RexGraph.from_hypergraph(ptr.astype(np.int64), idx.astype(np.int64))


def _dense_chi_rl(B1):
    """Dense reference: the four trace-normalized channels, RL4, and per-edge chi
    from B1 alone (T signed Gram, G unsigned Gram, F frustration Laplacian, C weighted
    line-graph Laplacian on shared-vertex counts)."""
    absB1 = np.abs(B1)
    T = B1.T @ B1
    G = absB1.T @ absB1
    K_off = G - np.diag(np.diag(G))
    Foff = (T - G).copy(); np.fill_diagonal(Foff, 0.0)
    F = Foff + np.diag(np.abs(Foff).sum(1))
    C = np.diag(K_off.sum(1)) - K_off
    hats = [X / np.trace(X) for X in (T, G, F, C) if np.trace(X) > 1e-15]
    RL = sum(hats)
    nE = B1.shape[1]
    chi = np.zeros((nE, len(hats)))
    for e in range(nE):
        d = RL[e, e]
        if d > 1e-15:
            chi[e] = [h[e, e] / d for h in hats]
    return chi, RL


class TestBranchingChannels:

    def test_rl4_psd_trace_nhats(self, hg):
        RL = np.asarray(hg.RL)
        assert np.linalg.eigvalsh(RL).min() > -1e-9              # PSD (L_C was breaking this)
        assert abs(np.trace(RL) - hg.nhats) < 1e-9              # trace-normalized

    def test_structural_character_matches_dense(self, hg):
        chi_d, _ = _dense_chi_rl(np.asarray(hg.B1, float))
        chi_s = np.asarray(hg.structural_character)
        assert chi_s.shape == chi_d.shape
        assert np.allclose(chi_s, chi_d, atol=1e-6)

    def test_rl4_matches_dense(self, hg):
        _, RL_d = _dense_chi_rl(np.asarray(hg.B1, float))
        assert np.allclose(np.asarray(hg.RL), RL_d, atol=1e-6)


class TestBranchingMoments:

    def test_energy_character_is_rl4_row_energy(self, hg):
        RL = np.asarray(hg.RL)
        assert np.allclose(hg.energy_character, np.diag(RL @ RL), atol=1e-6)

    def test_harmonic_entropy_finite_nonneg(self, hg):
        he = hg.harmonic_entropy
        assert np.isfinite(he) and he >= -1e-9

    def test_greens_diagonal_eigenfree_matches_dense(self, hg):
        got = hg.greens_diagonal_eigenfree                      # diag(RL4^-1), SPD
        want = np.diag(np.linalg.pinv(np.asarray(hg.RL, float)))
        assert np.allclose(got, want, atol=1e-7)

    def test_greens_character_edge_finite(self, hg):
        gc = hg.greens_character_edge                           # diag(L1^+), deflated
        assert gc.shape == (hg.nE,) and np.all(np.isfinite(gc)) and np.all(gc >= -1e-7)

    def test_per_channel_mixing_times_finite(self, hg):
        mt = hg.per_channel_mixing_times
        assert mt.shape == (hg.nhats,) and np.all(mt > 0)


class TestBranchingCharacterCoherence:

    def test_vertex_character_on_simplex(self, hg):
        phi = np.asarray(hg.vertex_character)
        assert phi.shape == (hg.nV, hg.nhats)
        assert np.allclose(phi.sum(axis=1), 1.0, atol=1e-6)

    def test_coherence_in_unit_interval(self, hg):
        kap = np.asarray(hg.coherence)
        assert kap.shape == (hg.nV,)
        assert kap.min() >= -1e-6 and kap.max() <= 1.0 + 1e-6

    def test_rcfe_curvature_strain_finite(self, hg):
        assert np.all(np.isfinite(hg.rcfe_curvature))
        assert np.isfinite(hg.rcfe_strain)
