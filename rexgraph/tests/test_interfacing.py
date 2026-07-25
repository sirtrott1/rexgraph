"""
Tests for the interfacing vector pipeline: vertex source, edge signal,
response operators, channel scores, Schrodinger score, quality gate,
sphere position, coverage, efficiency, confidence, and the full bundle.
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.core import _interfacing


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


def _get_L0_eigen(rex):
    sb = rex.spectral_bundle
    return sb['evals_L0'], np.ascontiguousarray(sb['evecs_L0'], dtype=np.float64)


def _get_RL_eigen(rex):
    return rex._rl_eigen


# build_vertex_source

class TestBuildVertexSource:

    def test_zero_except_targets(self, k4):
        ti = np.array([0, 2], dtype=np.int32)
        tw = np.array([1.0, 1.0], dtype=np.float64)
        vw = np.ones(k4.nV, dtype=np.float64)
        rho = _interfacing.build_vertex_source(ti, tw, vw, k4.nV)
        assert rho[1] == 0.0
        assert rho[3] == 0.0
        assert rho[0] > 0.0
        assert rho[2] > 0.0

    def test_weights_multiply(self, k4):
        ti = np.array([1], dtype=np.int32)
        tw = np.array([3.0], dtype=np.float64)
        vw = np.array([1.0, 2.0, 1.0, 1.0], dtype=np.float64)
        rho = _interfacing.build_vertex_source(ti, tw, vw, k4.nV)
        assert abs(rho[1] - 6.0) < 1e-12  # 3.0 * 2.0

    def test_shape(self, k4):
        ti = np.array([0], dtype=np.int32)
        tw = np.array([1.0], dtype=np.float64)
        vw = np.ones(k4.nV, dtype=np.float64)
        rho = _interfacing.build_vertex_source(ti, tw, vw, k4.nV)
        assert rho.shape == (k4.nV,)


# build_edge_signal

class TestBuildEdgeSignal:

    def test_shape(self, k4):
        evals_L0, evecs_L0 = _get_L0_eigen(k4)
        rho = np.zeros(k4.nV, dtype=np.float64)
        rho[0] = 1.0
        psi = _interfacing.build_edge_signal(
            rho, k4.B1, evals_L0, evecs_L0, k4.nV, k4.nE)
        assert psi.shape == (k4.nE,)

    def test_zero_source_zero_signal(self, k4):
        evals_L0, evecs_L0 = _get_L0_eigen(k4)
        rho = np.zeros(k4.nV, dtype=np.float64)
        psi = _interfacing.build_edge_signal(
            rho, k4.B1, evals_L0, evecs_L0, k4.nV, k4.nE)
        assert np.allclose(psi, 0, atol=1e-12)


# build_response_operators

class TestBuildResponseOperators:

    def test_S_T_shape(self, k4):
        evals_L0, evecs_L0 = _get_L0_eigen(k4)
        ops = _interfacing.build_response_operators(
            k4.B1, evals_L0, evecs_L0, k4.L_overlap, k4.L_frustration,
            k4.nV, k4.nE)
        assert ops['S_T'].shape == (k4.nE, k4.nE)

    def test_S_T_symmetric(self, k4):
        evals_L0, evecs_L0 = _get_L0_eigen(k4)
        ops = _interfacing.build_response_operators(
            k4.B1, evals_L0, evecs_L0, k4.L_overlap, k4.L_frustration,
            k4.nV, k4.nE)
        assert np.allclose(ops['S_T'], ops['S_T'].T, atol=1e-10)

    def test_S_G_is_L_O(self, k4):
        evals_L0, evecs_L0 = _get_L0_eigen(k4)
        ops = _interfacing.build_response_operators(
            k4.B1, evals_L0, evecs_L0, k4.L_overlap, k4.L_frustration,
            k4.nV, k4.nE)
        assert np.allclose(ops['S_G'], k4.L_overlap)


# channel_scores

class TestChannelScores:

    def test_shape(self, k4):
        evals_L0, evecs_L0 = _get_L0_eigen(k4)
        ops = _interfacing.build_response_operators(
            k4.B1, evals_L0, evecs_L0, k4.L_overlap, k4.L_frustration,
            k4.nV, k4.nE)
        psi = np.ones(k4.nE, dtype=np.float64)
        target = np.ones(k4.nE, dtype=np.float64)
        scores = _interfacing.channel_scores(
            psi, ops['S_T'], ops['S_G'], ops['S_F'], target, k4.nE)
        assert scores.shape == (3,)


# schrodinger_score

class TestSchrodingerScore:

    def test_nonnegative(self, k4):
        evals_rl, evecs_rl = _get_RL_eigen(k4)
        psi = np.ones(k4.nE, dtype=np.float64)
        target = np.ones(k4.nE, dtype=np.float64)
        s = _interfacing.schrodinger_score(psi, evals_rl, evecs_rl, target, k4.nE)
        assert s >= -1e-10

    def test_zero_signal(self, k4):
        evals_rl, evecs_rl = _get_RL_eigen(k4)
        psi = np.zeros(k4.nE, dtype=np.float64)
        target = np.ones(k4.nE, dtype=np.float64)
        s = _interfacing.schrodinger_score(psi, evals_rl, evecs_rl, target, k4.nE)
        assert abs(s) < 1e-12


# quality_gate

class TestQualityGate:

    def test_range(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        q = _interfacing.quality_gate(scores)
        assert np.all(q >= -1e-10)
        assert np.all(q <= 1.0 + 1e-10)

    def test_shape(self):
        scores = np.ones((5, 3), dtype=np.float64)
        q = _interfacing.quality_gate(scores)
        assert q.shape == (5, 3)


# sphere_position

class TestSpherePosition:

    def test_unit_norm(self):
        iv = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        sp = _interfacing.sphere_position(iv)
        assert abs(np.linalg.norm(sp) - 1.0) < 1e-12

    def test_zero_vector(self):
        iv = np.zeros(4, dtype=np.float64)
        sp = _interfacing.sphere_position(iv)
        assert np.allclose(sp, 0)


# coverage

class TestCoverage:

    def test_range(self, k4):
        evals_rl, evecs_rl = _get_RL_eigen(k4)
        psi = np.ones(k4.nE, dtype=np.float64)
        c = _interfacing.coverage(psi, evals_rl, evecs_rl, k4.nE, 1e-10)
        assert 0.0 <= c <= 1.0

    def test_dense_signal_high_coverage(self, k4):
        evals_rl, evecs_rl = _get_RL_eigen(k4)
        psi = np.random.RandomState(42).randn(k4.nE).astype(np.float64)
        c = _interfacing.coverage(psi, evals_rl, evecs_rl, k4.nE, 1e-15)
        assert c > 0.5

    def test_poisson_floor(self):
        pf = _interfacing.poisson_floor()
        assert abs(pf - (1.0 - np.exp(-1.0))) < 1e-12


# source_efficiency

class TestSourceEfficiency:

    def test_range(self, k4):
        ti = np.array([0, 1], dtype=np.int32)
        e = _interfacing.source_efficiency(ti, k4.B1, k4.nV, k4.nE)
        assert 0.0 <= e <= 1.0


# confidence_flags

class TestConfidenceFlags:

    def test_confident(self):
        r = _interfacing.confidence_flags(0.8, 0.7, 0.5)
        assert r['flag'] == 'CONFIDENT'

    def test_low_signal(self):
        r = _interfacing.confidence_flags(0.3, 0.7, 0.5)
        assert 'LOW_SIGNAL' in r['reasons']

    def test_channel_conflict(self):
        r = _interfacing.confidence_flags(0.8, 0.3, 0.3)
        assert 'CHANNEL_CONFLICT' in r['reasons']


# build_interfacing_bundle

class TestBuildInterfacingBundle:

    def test_returns_all_keys(self, k4):
        evals_L0, evecs_L0 = _get_L0_eigen(k4)
        evals_rl, evecs_rl = _get_RL_eigen(k4)
        ti = np.array([0, 1], dtype=np.int32)
        tw = np.array([1.0, 1.0], dtype=np.float64)
        vw = np.ones(k4.nV, dtype=np.float64)
        target = np.ones(k4.nE, dtype=np.float64)
        result = _interfacing.build_interfacing_bundle(
            ti, tw, vw, k4.B1, evals_L0, evecs_L0,
            k4.L_overlap, k4.L_frustration,
            evals_rl, evecs_rl, target, k4.nV, k4.nE)
        for key in ['rho', 'psi', 'scores', 'schrodinger', 'iv',
                     'sphere_pos', 'signal_magnitude', 'coverage',
                     'efficiency', 'confidence']:
            assert key in result

    def test_sphere_pos_unit_norm(self, k4):
        evals_L0, evecs_L0 = _get_L0_eigen(k4)
        evals_rl, evecs_rl = _get_RL_eigen(k4)
        ti = np.array([0], dtype=np.int32)
        tw = np.array([1.0], dtype=np.float64)
        vw = np.ones(k4.nV, dtype=np.float64)
        target = np.ones(k4.nE, dtype=np.float64)
        result = _interfacing.build_interfacing_bundle(
            ti, tw, vw, k4.B1, evals_L0, evecs_L0,
            k4.L_overlap, k4.L_frustration,
            evals_rl, evecs_rl, target, k4.nV, k4.nE)
        sp = result['sphere_pos']
        norm = np.linalg.norm(sp)
        if result['signal_magnitude'] > 1e-10:
            assert abs(norm - 1.0) < 1e-10

    def test_graph_method(self, k4):
        ti = np.array([0, 1], dtype=np.int32)
        tw = np.array([1.0, 1.0], dtype=np.float64)
        target = np.ones(k4.nE, dtype=np.float64)
        result = k4.interfacing_vector(ti, tw, target)
        assert 'iv' in result
        assert 'confidence' in result
