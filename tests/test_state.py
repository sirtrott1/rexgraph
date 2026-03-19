"""
Tests for rexgraph.core._state -- rex state representation and signal operations.

Verifies:
    - Signal norms: L1, L2, Linf correct values
    - Normalization: L1 sums to 1, L2 norm is 1, zero signal unchanged
    - Pack/unpack roundtrip: V+E+F and E+F field state
    - State differencing: diff + apply recovers original
    - Energy: E_kin and E_pot nonneg (PSD Laplacians), ratio correct
    - State construction: uniform, dirac, dirac_edge, random
    - RexState class: shapes, set/get, energy caching, derive_vertex_signal
    - Integration through RexGraph: energy_kin_pot, dirac_state, dirac_edge,
      uniform_state, normalize, create_state
"""
import numpy as np
import pytest

from rexgraph.core import _state
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
def tree():
    return RexGraph.from_graph([0, 1, 2], [1, 2, 3])


# Signal Norms

class TestSignalNorms:

    def test_l1(self):
        x = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        assert abs(_state.signal_norm_l1(x) - 6.0) < 1e-12

    def test_l2(self):
        x = np.array([3.0, 4.0], dtype=np.float64)
        assert abs(_state.signal_norm_l2(x) - 5.0) < 1e-12

    def test_linf(self):
        x = np.array([1.0, -5.0, 3.0], dtype=np.float64)
        assert abs(_state.signal_norm_linf(x) - 5.0) < 1e-12

    def test_dispatch(self):
        x = np.array([3.0, 4.0], dtype=np.float64)
        assert abs(_state.signal_norm(x, 0) - 7.0) < 1e-12  # L1
        assert abs(_state.signal_norm(x, 1) - 5.0) < 1e-12  # L2
        assert abs(_state.signal_norm(x, 2) - 4.0) < 1e-12  # Linf

    def test_zero_signal(self):
        x = np.zeros(5, dtype=np.float64)
        assert _state.signal_norm_l1(x) == 0.0
        assert _state.signal_norm_l2(x) == 0.0
        assert _state.signal_norm_linf(x) == 0.0


# Normalization

class TestNormalization:

    def test_l1_sums_to_one(self):
        x = np.array([2.0, 3.0, 5.0], dtype=np.float64)
        normed = _state.normalize_l1(x)
        assert abs(normed.sum() - 1.0) < 1e-12

    def test_l2_norm_one(self):
        x = np.array([3.0, 4.0], dtype=np.float64)
        normed = _state.normalize_l2(x)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-12

    def test_zero_signal_unchanged(self):
        x = np.zeros(3, dtype=np.float64)
        normed = _state.normalize_l1(x)
        assert np.allclose(normed, 0)

    def test_returns_copy(self):
        x = np.array([1.0, 2.0], dtype=np.float64)
        normed = _state.normalize_l2(x)
        normed[0] = 999.0
        assert x[0] == 1.0  # original unchanged


# Pack / Unpack (V+E+F)

class TestPackUnpack:

    def test_roundtrip(self):
        f0 = np.array([1.0, 2.0], dtype=np.float64)
        f1 = np.array([3.0, 4.0, 5.0], dtype=np.float64)
        f2 = np.array([6.0], dtype=np.float64)
        flat, sizes = _state.pack_state(f0, f1, f2)
        assert flat.shape == (6,)
        assert list(sizes) == [2, 3, 1]
        r0, r1, r2 = _state.unpack_state(flat, sizes)
        assert np.array_equal(r0, f0)
        assert np.array_equal(r1, f1)
        assert np.array_equal(r2, f2)

    def test_empty_dimensions(self):
        f0 = np.array([1.0], dtype=np.float64)
        f1 = np.array([2.0, 3.0], dtype=np.float64)
        f2 = np.zeros(0, dtype=np.float64)
        flat, sizes = _state.pack_state(f0, f1, f2)
        assert flat.shape == (3,)
        r0, r1, r2 = _state.unpack_state(flat, sizes)
        assert r2.shape == (0,)


# Field State Pack / Unpack (E+F)

class TestFieldStatePack:

    def test_roundtrip(self):
        f_E = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        f_F = np.array([4.0, 5.0], dtype=np.float64)
        flat, sizes = _state.field_state_pack(f_E, f_F)
        assert flat.shape == (5,)
        assert list(sizes) == [3, 2]
        rE, rF = _state.field_state_unpack(flat, sizes)
        assert np.array_equal(rE, f_E)
        assert np.array_equal(rF, f_F)

    def test_vertex_observable(self, k4):
        f_E = np.random.RandomState(42).randn(k4.nE).astype(np.float64)
        f_V = _state.field_state_vertex_observable(f_E, k4.B1)
        expected = np.asarray(k4.B1 @ f_E, dtype=np.float64)
        assert np.allclose(f_V, expected, atol=1e-12)


# State Differencing

class TestStateDiff:

    def test_diff_and_apply(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        diff = _state.state_diff(a, b)
        assert np.allclose(diff, [3.0, 3.0, 3.0])
        recovered = _state.state_apply_diff(a, diff)
        assert np.allclose(recovered, b)

    def test_zero_diff(self):
        a = np.array([1.0, 2.0], dtype=np.float64)
        diff = _state.state_diff(a, a)
        assert np.allclose(diff, 0)


# Energy Computation

class TestEnergyKinPot:

    def test_nonneg(self, k4):
        f_E = np.random.RandomState(42).randn(k4.nE).astype(np.float64)
        ek, ep, ratio = _state.energy_kin_pot(f_E, k4.L1, k4.L_overlap)
        assert ek >= -1e-10
        assert ep >= -1e-10

    def test_zero_signal_zero_energy(self, k4):
        f_E = np.zeros(k4.nE, dtype=np.float64)
        ek, ep, ratio = _state.energy_kin_pot(f_E, k4.L1, k4.L_overlap)
        assert abs(ek) < 1e-15
        assert abs(ep) < 1e-15
        assert ratio == 1.0  # both zero -> ratio 1

    def test_ratio_correct(self, k4):
        f_E = np.random.RandomState(7).randn(k4.nE).astype(np.float64)
        ek, ep, ratio = _state.energy_kin_pot(f_E, k4.L1, k4.L_overlap)
        if ep > 1e-10:
            assert abs(ratio - ek / ep) < 1e-10


# State Construction

class TestStateConstruction:

    def test_uniform_l1(self):
        f0, f1, f2 = _state.uniform_state(4, 6, 4, norm_type=0)
        assert abs(f0.sum() - 1.0) < 1e-12
        assert abs(f1.sum() - 1.0) < 1e-12
        assert abs(f2.sum() - 1.0) < 1e-12

    def test_uniform_l2(self):
        f0, f1, f2 = _state.uniform_state(4, 6, 4, norm_type=1)
        assert abs(np.linalg.norm(f0) - 1.0) < 1e-12
        assert abs(np.linalg.norm(f1) - 1.0) < 1e-12

    def test_dirac_state(self):
        f0, f1, f2 = _state.dirac_state(3, 5, 2, dim=1, idx=3)
        assert np.allclose(f0, 0)
        assert f1[3] == 1.0
        assert np.count_nonzero(f1) == 1
        assert np.allclose(f2, 0)

    def test_dirac_edge(self):
        f_E, f_F = _state.dirac_edge(5, 2, 1)
        assert f_E[1] == 1.0
        assert np.count_nonzero(f_E) == 1
        assert np.allclose(f_F, 0)

    def test_random_state_shapes(self):
        f0, f1, f2 = _state.random_state(3, 5, 2)
        assert f0.shape == (3,)
        assert f1.shape == (5,)
        assert f2.shape == (2,)

    def test_random_state_l1_normalized(self):
        f0, f1, f2 = _state.random_state(4, 6, 3, norm_type=0)
        assert abs(f0.sum() - 1.0) < 1e-10
        assert abs(f1.sum() - 1.0) < 1e-10


# RexState Class

class TestRexState:

    def test_shapes(self):
        rs = _state.RexState(4, 6, 4, t=0.0)
        assert rs.shapes == (4, 6, 4)

    def test_initial_zero(self):
        rs = _state.RexState(3, 5, 2)
        assert np.allclose(rs.f0, 0)
        assert np.allclose(rs.f1, 0)
        assert np.allclose(rs.f2, 0)
        assert rs.t == 0.0

    def test_set_signals(self):
        rs = _state.RexState(2, 3, 1)
        rs.set_f1(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        assert rs.f1[0] == 1.0
        assert rs.f1[2] == 3.0

    def test_energy_dirty_after_set(self):
        rs = _state.RexState(2, 3, 1)
        rs.set_f1(np.ones(3, dtype=np.float64))
        assert np.isnan(rs.energy)
        assert np.isnan(rs.E_kin)
        assert np.isnan(rs.E_pot)

    def test_derive_vertex_signal(self, k4):
        rs = _state.RexState(k4.nV, k4.nE, k4.nF)
        rs.set_f1(np.ones(k4.nE, dtype=np.float64))
        rs.derive_vertex_signal(k4.B1)
        expected = np.asarray(k4.B1 @ np.ones(k4.nE), dtype=np.float64)
        assert np.allclose(rs.f0, expected)


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_energy_kin_pot(self, k4):
        f_E = np.ones(k4.nE, dtype=np.float64)
        ek, ep, ratio = k4.energy_kin_pot(f_E)
        assert ek >= -1e-10
        assert ep >= -1e-10

    def test_dirac_state(self, k4):
        f0, f1, f2 = k4.dirac_state(dim=1, idx=0)
        assert f0.shape == (k4.nV,)
        assert f1.shape == (k4.nE,)
        assert f1[0] == 1.0

    def test_dirac_edge(self, k4):
        f_E, f_F = k4.dirac_edge(0)
        assert f_E.shape == (k4.nE,)
        assert f_E[0] == 1.0
        assert np.allclose(f_F, 0)

    def test_uniform_state(self, k4):
        f0, f1, f2 = k4.uniform_state(norm="l1")
        assert abs(f0.sum() - 1.0) < 1e-12
        assert abs(f1.sum() - 1.0) < 1e-12

    def test_normalize(self, k4):
        g = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        normed = k4.normalize(g, norm="l2")
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-12

    def test_create_state(self, k4):
        rs = k4.create_state(t=0.5)
        assert rs.shapes == (k4.nV, k4.nE, k4.nF)
        assert rs.t == 0.5
