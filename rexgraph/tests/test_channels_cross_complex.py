"""
Tests for per-channel signal decomposition, spectral channel scores,
group scoring, cross-complex alignment and bridge analysis, new
types, and graph/analysis integration.
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph, cross_complex_bridge
from rexgraph.core import _channels, _cross_complex, _character


@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32),
    )


@pytest.fixture
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


# primal_signal_character

class TestPrimalSignalCharacter:

    def test_sums_to_one(self, k4):
        psi = np.random.RandomState(42).randn(k4.nE).astype(np.float64)
        he = k4._hat_eigen_bundle
        if len(he) == 0:
            pytest.skip("no hat eigendata")
        evals_list = [h[0] for h in he]
        evecs_list = [h[1] for h in he]
        pc = _channels.primal_signal_character(
            psi, evals_list, evecs_list, k4.nhats, k4.nE)
        assert abs(pc.sum() - 1.0) < 1e-10

    def test_shape(self, k4):
        psi = np.ones(k4.nE, dtype=np.float64)
        pc = k4.primal_signal_character(psi)
        assert pc.shape == (k4.nhats,)

    def test_nonneg(self, k4):
        psi = np.ones(k4.nE, dtype=np.float64)
        pc = k4.primal_signal_character(psi)
        assert np.all(pc >= -1e-10)

    def test_zero_signal_uniform(self, k4):
        psi = np.zeros(k4.nE, dtype=np.float64)
        pc = k4.primal_signal_character(psi)
        expected = 1.0 / k4.nhats
        assert np.allclose(pc, expected, atol=1e-10)


# spectral_channel_score

class TestSpectralChannelScore:

    def test_symmetric(self, k4):
        evals_rl, evecs_rl = k4._rl_eigen
        a = np.random.RandomState(42).randn(k4.nE).astype(np.float64)
        b = np.random.RandomState(43).randn(k4.nE).astype(np.float64)
        s_ab = _channels.spectral_channel_score(a, b, evals_rl, evecs_rl, k4.nE)
        s_ba = _channels.spectral_channel_score(b, a, evals_rl, evecs_rl, k4.nE)
        assert abs(s_ab - s_ba) < 1e-10

    def test_self_positive(self, k4):
        evals_rl, evecs_rl = k4._rl_eigen
        a = np.ones(k4.nE, dtype=np.float64)
        s = _channels.spectral_channel_score(a, a, evals_rl, evecs_rl, k4.nE)
        assert s > 0

    def test_graph_method(self, k4):
        a = np.ones(k4.nE, dtype=np.float64)
        b = np.ones(k4.nE, dtype=np.float64)
        s = k4.spectral_channel_score(a, b)
        assert isinstance(s, float)
        assert s > 0


# group_channel_scores

class TestGroupChannelScores:

    def test_shape(self, k4):
        evals_rl, evecs_rl = k4._rl_eigen
        n_groups = 2
        masks = np.zeros((n_groups, k4.nV), dtype=np.uint8)
        masks[0, 0] = 1
        masks[0, 1] = 1
        masks[1, 2] = 1
        masks[1, 3] = 1
        target = np.ones(k4.nE, dtype=np.float64)
        scores = _channels.group_channel_scores(
            masks, target, evals_rl, evecs_rl, k4.B1, k4.nV, k4.nE, n_groups)
        assert scores.shape == (n_groups,)

    def test_single_vertex_group(self, k4):
        evals_rl, evecs_rl = k4._rl_eigen
        masks = np.zeros((1, k4.nV), dtype=np.uint8)
        masks[0, 0] = 1
        target = np.ones(k4.nE, dtype=np.float64)
        scores = _channels.group_channel_scores(
            masks, target, evals_rl, evecs_rl, k4.B1, k4.nV, k4.nE, 1)
        assert np.isfinite(scores[0])


# multi_channel_profile

class TestMultiChannelProfile:

    def test_keys(self):
        iv = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        pc = np.array([0.3, 0.3, 0.4], dtype=np.float64)
        result = _channels.multi_channel_profile(iv, pc, 0.8, 0.9, 0.7)
        assert 'coverage' in result
        assert 'kappa_mean' in result
        assert 'efficiency' in result
        assert 'iv_T' in result
        assert 'pc_T' in result


# align_by_labels

class TestAlignByLabels:

    def test_shared(self):
        la = ["A", "B", "C", "D"]
        lb = ["C", "D", "E", "F"]
        shared, ia, ib = _cross_complex.align_by_labels(la, lb)
        assert set(shared) == {"C", "D"}
        assert len(ia) == 2
        assert len(ib) == 2

    def test_no_overlap(self):
        shared, ia, ib = _cross_complex.align_by_labels(["A", "B"], ["C", "D"])
        assert len(shared) == 0
        assert ia.shape == (0,)

    def test_full_overlap(self):
        la = ["X", "Y", "Z"]
        shared, ia, ib = _cross_complex.align_by_labels(la, la)
        assert len(shared) == 3

    def test_index_correctness(self):
        la = ["A", "B", "C"]
        lb = ["B", "C", "D"]
        shared, ia, ib = _cross_complex.align_by_labels(la, lb)
        for k in range(len(shared)):
            assert la[ia[k]] == shared[k]
            assert lb[ib[k]] == shared[k]


# cross_complex_kappa

class TestCrossComplexKappa:

    def test_self_correlation_one(self, k4):
        kappa = k4.coherence
        idx = np.arange(k4.nV, dtype=np.int32)
        result = _cross_complex.cross_complex_kappa(kappa, kappa, idx, idx)
        assert abs(result['correlation'] - 1.0) < 1e-10

    def test_range(self, k4):
        kappa = k4.coherence
        kappa2 = np.random.RandomState(42).rand(k4.nV).astype(np.float64)
        idx = np.arange(k4.nV, dtype=np.int32)
        result = _cross_complex.cross_complex_kappa(kappa, kappa2, idx, idx)
        assert -1.0 - 1e-10 <= result['correlation'] <= 1.0 + 1e-10

    def test_insufficient_shared(self):
        ka = np.array([0.5], dtype=np.float64)
        kb = np.array([0.8], dtype=np.float64)
        ia = np.array([0], dtype=np.int32)
        ib = np.array([0], dtype=np.int32)
        result = _cross_complex.cross_complex_kappa(ka, kb, ia, ib)
        assert result['correlation'] == 0.0


# cross_complex_void_fraction

class TestCrossComplexVoidFraction:

    def test_range(self):
        result = _cross_complex.cross_complex_void_fraction(3, 10, 7, 10)
        assert 0.0 <= result['void_fraction_A'] <= 1.0
        assert 0.0 <= result['void_fraction_B'] <= 1.0

    def test_no_voids(self):
        result = _cross_complex.cross_complex_void_fraction(0, 10, 0, 10)
        assert result['void_fraction_A'] == 0.0
        assert result['void_fraction_B'] == 0.0

    def test_difference(self):
        result = _cross_complex.cross_complex_void_fraction(3, 10, 7, 10)
        assert abs(result['difference'] - (0.3 - 0.7)) < 1e-12


# cross_complex_bridge

class TestCrossComplexBridge:

    def test_returns_keys(self, k4):
        kappa = k4.coherence
        idx = np.arange(k4.nV, dtype=np.int32)
        result = _cross_complex.cross_complex_bridge(
            kappa, kappa, idx, idx, 0, 4, 0, 4)
        assert 'kappa' in result
        assert 'void' in result
        assert 'n_shared' in result

    def test_with_channel_scores(self, k4):
        kappa = k4.coherence
        idx = np.arange(k4.nV, dtype=np.int32)
        sa = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sb = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        result = _cross_complex.cross_complex_bridge(
            kappa, kappa, idx, idx, 0, 4, 0, 4,
            channel_scores_A=sa, channel_scores_B=sb)
        assert 'channel' in result

    def test_without_channel_scores(self, k4):
        kappa = k4.coherence
        idx = np.arange(k4.nV, dtype=np.int32)
        result = _cross_complex.cross_complex_bridge(
            kappa, kappa, idx, idx, 0, 4, 0, 4)
        assert 'channel' not in result

    def test_graph_level_function(self, k4):
        labels = [f"v{i}" for i in range(k4.nV)]
        result = cross_complex_bridge(k4, k4, labels, labels)
        assert 'kappa' in result
        assert result['n_shared'] == k4.nV


# types

class TestNewTypes:

    def test_interfacing_result_fields(self):
        from rexgraph.types import InterfacingResult
        assert 'rho' in InterfacingResult._fields
        assert 'iv' in InterfacingResult._fields
        assert 'coverage' in InterfacingResult._fields
        assert 'confidence' in InterfacingResult._fields

    def test_channel_profile_fields(self):
        from rexgraph.types import ChannelProfile
        assert 'iv_T' in ChannelProfile._fields
        assert 'coverage' in ChannelProfile._fields

    def test_filtration_result_fields(self):
        from rexgraph.types import FiltrationResult
        assert 'beta1' in FiltrationResult._fields
        assert 'transition_index' in FiltrationResult._fields

    def test_cross_complex_bridge_fields(self):
        from rexgraph.types import CrossComplexBridge
        assert 'kappa' in CrossComplexBridge._fields
        assert 'void' in CrossComplexBridge._fields
        assert 'channel' in CrossComplexBridge._fields

    def test_cross_complex_bridge_optional_default(self):
        from rexgraph.types import CrossComplexBridge
        ccb = CrossComplexBridge(
            kappa={}, void={}, n_shared=0)
        assert ccb.channel is None


# graph.py integration

class TestGraphIntegration:

    def test_hat_eigen_bundle_cached(self, k4):
        a = k4._hat_eigen_bundle
        b = k4._hat_eigen_bundle
        assert a is b

    def test_inverse_centrality_ratio_shape(self, k4):
        mu = k4.inverse_centrality_ratio
        assert mu.shape == (k4.nV,)

    def test_inverse_centrality_ratio_positive(self, k4):
        mu = k4.inverse_centrality_ratio
        assert np.all(mu >= 0)

    def test_per_channel_mixing_times_shape(self, k4):
        t = k4.per_channel_mixing_times
        assert t.shape == (k4.nhats,)


# analysis.py integration

class TestAnalysisIntegration:

    def test_mixing_times_in_analysis(self, k4):
        from rexgraph.analysis import analyze
        data = analyze(k4, run_perturbation=False)
        sc = data.get('structural_character', {})
        if 'per_channel_mixing_times' in sc:
            assert isinstance(sc['per_channel_mixing_times'], list)

    def test_channels_in_analysis(self, k4):
        from rexgraph.analysis import analyze
        data = analyze(k4, run_perturbation=False)
        if 'channels' in data:
            ch = data['channels']
            assert 'primal_signal_character' in ch
