"""
Tests for hat eigendecomposition, per-channel mixing times,
mixing time anisotropy, and face-void dipole.
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.core import _character


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


# hat_eigen

class TestHatEigen:

    def test_evals_ascending(self, k4):
        hats = k4._rcf_bundle.get('hats', [])
        if len(hats) == 0:
            pytest.skip("no hats available")
        evals, evecs = _character.hat_eigen(hats[0], k4.nE)
        assert np.all(np.diff(evals) >= -1e-12)

    def test_evals_nonneg(self, k4):
        hats = k4._rcf_bundle.get('hats', [])
        if len(hats) == 0:
            pytest.skip("no hats available")
        evals, _ = _character.hat_eigen(hats[0], k4.nE)
        assert np.all(evals >= -1e-10)

    def test_evecs_shape(self, k4):
        hats = k4._rcf_bundle.get('hats', [])
        if len(hats) == 0:
            pytest.skip("no hats available")
        evals, evecs = _character.hat_eigen(hats[0], k4.nE)
        assert evals.shape == (k4.nE,)
        assert evecs.shape == (k4.nE, k4.nE)

    def test_evecs_orthonormal(self, k4):
        hats = k4._rcf_bundle.get('hats', [])
        if len(hats) == 0:
            pytest.skip("no hats available")
        _, evecs = _character.hat_eigen(hats[0], k4.nE)
        assert np.allclose(evecs.T @ evecs, np.eye(k4.nE), atol=1e-10)

    def test_empty(self):
        evals, evecs = _character.hat_eigen(
            np.empty((0, 0), dtype=np.float64), 0)
        assert evals.shape == (0,)
        assert evecs.shape == (0, 0)


class TestHatEigenAll:

    def test_length(self, k4):
        hats = k4._rcf_bundle.get('hats', [])
        nhats = k4._rcf_bundle.get('nhats', 0)
        if nhats == 0:
            pytest.skip("no hats available")
        result = _character.hat_eigen_all(hats, nhats, k4.nE)
        assert len(result) == nhats

    def test_each_pair(self, k4):
        hats = k4._rcf_bundle.get('hats', [])
        nhats = k4._rcf_bundle.get('nhats', 0)
        if nhats == 0:
            pytest.skip("no hats available")
        result = _character.hat_eigen_all(hats, nhats, k4.nE)
        for evals, evecs in result:
            assert evals.shape == (k4.nE,)
            assert evecs.shape == (k4.nE, k4.nE)


# per_channel_mixing_time

class TestPerChannelMixingTime:

    def test_from_evals_finite(self, k4):
        hats = k4._rcf_bundle.get('hats', [])
        if len(hats) == 0:
            pytest.skip("no hats available")
        evals, _ = _character.hat_eigen(hats[0], k4.nE)
        tau = _character.per_channel_mixing_time(evals, k4.nE)
        assert tau > 0

    def test_zero_gap_returns_inf(self):
        evals = np.zeros(5, dtype=np.float64)
        tau = _character.per_channel_mixing_time(evals, 5)
        assert tau == float('inf')

    def test_single_edge_returns_inf(self):
        evals = np.array([1.0], dtype=np.float64)
        tau = _character.per_channel_mixing_time(evals, 1)
        assert tau == float('inf')


class TestPerChannelMixingTimes:

    def test_shape(self, k4):
        times = k4.per_channel_mixing_times
        assert times.shape == (k4.nhats,)

    def test_positive(self, k4):
        times = k4.per_channel_mixing_times
        for t in times:
            assert t > 0

    def test_from_evals_matches(self, k4):
        hats = k4._rcf_bundle.get('hats', [])
        nhats = k4._rcf_bundle.get('nhats', 0)
        if nhats == 0:
            pytest.skip("no hats available")
        he = _character.hat_eigen_all(hats, nhats, k4.nE)
        evals_list = [h[0] for h in he]
        times_from_evals = _character.per_channel_mixing_times_from_evals(
            evals_list, nhats, k4.nE)
        times_direct = _character.per_channel_mixing_times(hats, nhats, k4.nE)
        assert np.allclose(times_from_evals, times_direct, atol=1e-10)


# mixing_time_anisotropy

class TestMixingTimeAnisotropy:

    def test_returns_dict(self, k4):
        times = k4.per_channel_mixing_times
        result = _character.mixing_time_anisotropy(times, k4.nhats)
        assert 'ratios' in result
        assert 'dominant_channel' in result
        assert 'slowest_channel' in result
        assert 'anisotropy' in result

    def test_ratios_shape(self, k4):
        times = k4.per_channel_mixing_times
        result = _character.mixing_time_anisotropy(times, k4.nhats)
        assert result['ratios'].shape == (k4.nhats, k4.nhats)

    def test_diagonal_ones(self, k4):
        times = k4.per_channel_mixing_times
        result = _character.mixing_time_anisotropy(times, k4.nhats)
        assert np.allclose(np.diag(result['ratios']), 1.0, atol=1e-10)

    def test_dominant_has_smallest_time(self, k4):
        times = k4.per_channel_mixing_times
        result = _character.mixing_time_anisotropy(times, k4.nhats)
        dom = result['dominant_channel']
        finite = times[np.isfinite(times)]
        if len(finite) > 0:
            assert times[dom] <= np.min(finite) + 1e-10


# face_void_dipole

class TestFaceVoidDipole:

    def test_zero_signal(self, k4):
        psi = np.zeros(k4.nE, dtype=np.float64)
        d = _character.face_void_dipole(psi, k4.B2_hodge, None, k4.nE, k4.nF_hodge)
        assert d['face_affinity'] == 0.0
        assert d['void_affinity'] == 0.0

    def test_range(self, k4):
        psi = np.random.RandomState(42).randn(k4.nE).astype(np.float64)
        d = _character.face_void_dipole(psi, k4.B2_hodge, None, k4.nE, k4.nF_hodge)
        assert d['face_affinity'] >= 0.0
        assert d['void_affinity'] >= 0.0

    def test_dipole_ratio_range(self, k4):
        psi = np.ones(k4.nE, dtype=np.float64)
        d = _character.face_void_dipole(psi, k4.B2_hodge, None, k4.nE, k4.nF_hodge)
        assert -1.0 - 1e-10 <= d['dipole_ratio'] <= 1.0 + 1e-10

    def test_no_voids_zero_void_affinity(self, k4):
        psi = np.ones(k4.nE, dtype=np.float64)
        d = _character.face_void_dipole(psi, k4.B2_hodge, None, k4.nE, k4.nF_hodge)
        assert d['void_affinity'] == 0.0

    def test_no_faces(self, triangle):
        psi = np.ones(triangle.nE, dtype=np.float64)
        B2 = np.zeros((triangle.nE, 0), dtype=np.float64)
        d = _character.face_void_dipole(psi, B2, None, triangle.nE, 0)
        assert d['face_affinity'] == 0.0

    def test_graph_method(self, k4):
        psi = np.ones(k4.nE, dtype=np.float64)
        d = k4.face_void_dipole(psi)
        assert 'face_affinity' in d
        assert 'dipole_ratio' in d
