"""
Tests for rexgraph.core._temporal -- temporal bundle, BIOES, lifecycle.

Verifies:
    - Delta encoding: born/died edges correct between snapshots
    - Temporal index: checkpoints stored, deltas computed
    - Edge lifecycle: birth/death times, unique edge count
    - Edge metrics: counts, births, deaths per timestep
    - Phase detection: constant Betti -> single phase, changes -> multiple
    - BIOES tags: valid tag values, B/I/E/S assignment
    - Energy-ratio BIOES: phases from E_kin/E_pot regime
    - Cascade: activation order consistent, source activates first
    - Face tracking: persist, born, died events
    - Integration through TemporalRex
"""
import numpy as np
import pytest

from rexgraph.core import _temporal
from rexgraph.graph import RexGraph, TemporalRex


# Helpers

def _make_snapshots():
    """Three snapshots: triangle -> triangle+extra edge -> triangle."""
    s0 = (np.array([0, 1, 0], dtype=np.int32),
          np.array([1, 2, 2], dtype=np.int32))
    s1 = (np.array([0, 1, 0, 1], dtype=np.int32),
          np.array([1, 2, 2, 3], dtype=np.int32))
    s2 = (np.array([0, 1, 0], dtype=np.int32),
          np.array([1, 2, 2], dtype=np.int32))
    return [s0, s1, s2]


# Delta Encoding

class TestDeltaEncoding:

    def test_identical_snapshots(self):
        src = np.array([0, 1], dtype=np.int32)
        tgt = np.array([1, 2], dtype=np.int32)
        bs, bt, ds, dt = _temporal.encode_snapshot_delta(src, tgt, src, tgt)
        assert bs.shape[0] == 0
        assert ds.shape[0] == 0

    def test_edge_born(self):
        prev_s = np.array([0, 1], dtype=np.int32)
        prev_t = np.array([1, 2], dtype=np.int32)
        curr_s = np.array([0, 1, 0], dtype=np.int32)
        curr_t = np.array([1, 2, 2], dtype=np.int32)
        bs, bt, ds, dt = _temporal.encode_snapshot_delta(prev_s, prev_t, curr_s, curr_t)
        assert bs.shape[0] == 1  # edge (0,2) born
        assert ds.shape[0] == 0

    def test_edge_died(self):
        prev_s = np.array([0, 1, 0], dtype=np.int32)
        prev_t = np.array([1, 2, 2], dtype=np.int32)
        curr_s = np.array([0, 1], dtype=np.int32)
        curr_t = np.array([1, 2], dtype=np.int32)
        bs, bt, ds, dt = _temporal.encode_snapshot_delta(prev_s, prev_t, curr_s, curr_t)
        assert bs.shape[0] == 0
        assert ds.shape[0] == 1  # edge (0,2) died

    def test_symmetric_undirected(self):
        """(0,1) and (1,0) are the same edge when undirected."""
        s1 = np.array([0], dtype=np.int32)
        t1 = np.array([1], dtype=np.int32)
        s2 = np.array([1], dtype=np.int32)
        t2 = np.array([0], dtype=np.int32)
        bs, bt, ds, dt = _temporal.encode_snapshot_delta(s1, t1, s2, t2, directed=False)
        assert bs.shape[0] == 0
        assert ds.shape[0] == 0


# Temporal Index

class TestTemporalIndex:

    def test_builds_checkpoints(self):
        snaps = _make_snapshots()
        checkpoints, deltas, cp_times = _temporal.build_temporal_index(snaps)
        assert len(checkpoints) >= 1
        assert cp_times[0] == 0

    def test_first_checkpoint_is_snapshot_0(self):
        snaps = _make_snapshots()
        checkpoints, _, _ = _temporal.build_temporal_index(snaps)
        _, s0, t0 = checkpoints[0]
        assert np.array_equal(s0, snaps[0][0])


# Edge Lifecycle

class TestEdgeLifecycle:

    def test_returns_tuple(self):
        snaps = _make_snapshots()
        result = _temporal.edge_lifecycle(snaps)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (edge_ids, birth, death)
        edge_ids, birth, death = result
        assert edge_ids.dtype == np.int64
        assert birth.dtype == np.int32
        assert death.dtype == np.int32

    def test_n_unique_edges(self):
        """3 triangle edges + 1 extra edge = 4 unique edges."""
        snaps = _make_snapshots()
        edge_ids, _, _ = _temporal.edge_lifecycle(snaps)
        assert edge_ids.shape[0] == 4

    def test_birth_times_valid(self):
        snaps = _make_snapshots()
        _, birth, _ = _temporal.edge_lifecycle(snaps)
        T = len(snaps)
        assert np.all(birth >= 0)
        assert np.all(birth < T)

    def test_death_minus_one_means_alive(self):
        """Edges alive at the final snapshot have death = -1."""
        snaps = _make_snapshots()
        _, birth, death = _temporal.edge_lifecycle(snaps)
        # Triangle edges (born t=0, last seen t=2 = T-1) should have death=-1
        alive_mask = death == -1
        assert alive_mask.sum() >= 3  # the 3 triangle edges survive


# Edge Metrics

class TestEdgeMetrics:

    def test_shapes(self):
        snaps = _make_snapshots()
        ec, bc, dc = _temporal.compute_edge_metrics(snaps)
        T = len(snaps)
        assert ec.shape == (T,)
        assert bc.shape == (T,)
        assert dc.shape == (T,)

    def test_counts_match_snapshots(self):
        snaps = _make_snapshots()
        ec, _, _ = _temporal.compute_edge_metrics(snaps)
        assert ec[0] == 3  # triangle
        assert ec[1] == 4  # triangle + extra edge
        assert ec[2] == 3  # back to triangle

    def test_born_at_t1(self):
        snaps = _make_snapshots()
        _, bc, _ = _temporal.compute_edge_metrics(snaps)
        assert bc[1] == 1  # edge (1,3) born at t=1

    def test_died_at_t2(self):
        snaps = _make_snapshots()
        _, _, dc = _temporal.compute_edge_metrics(snaps)
        assert dc[2] == 1  # edge (1,3) died at t=2


# Phase Detection

class TestPhaseDetection:

    def test_constant_betti_single_phase(self):
        beta0 = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        beta1 = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        ps, pe, pb0, pb1 = _temporal.detect_phases(beta0, beta1)
        assert len(ps) == 1
        assert ps[0] == 0
        assert pe[0] == 4

    def test_betti_change_splits_phase(self):
        beta0 = np.array([1, 1, 1, 2, 2], dtype=np.int64)
        beta1 = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        ps, pe, _, _ = _temporal.detect_phases(beta0, beta1)
        assert len(ps) == 2


# BIOES Tags

class TestBIOESTags:

    def test_valid_tags(self):
        ps = np.array([0, 3], dtype=np.int32)
        pe = np.array([2, 4], dtype=np.int32)
        tags = _temporal.assign_bioes_tags(5, ps, pe, min_phase_len=2)
        assert tags.shape == (5,)
        assert np.all((tags >= 0) & (tags <= 4))

    def test_single_step_phase(self):
        """Phase of length 1 gets tag S=4."""
        ps = np.array([0, 1, 2], dtype=np.int32)
        pe = np.array([0, 1, 2], dtype=np.int32)
        tags = _temporal.assign_bioes_tags(3, ps, pe, min_phase_len=2)
        assert np.all(tags == 4)  # all single-step phases

    def test_long_phase_bie(self):
        """Phase of length >= 3 has B at start, I in middle, E at end."""
        ps = np.array([0], dtype=np.int32)
        pe = np.array([4], dtype=np.int32)
        tags = _temporal.assign_bioes_tags(5, ps, pe, min_phase_len=2)
        assert tags[0] == 0   # B
        assert tags[4] == 3   # E
        for t in range(1, 4):
            assert tags[t] == 1  # I


# Energy-Ratio BIOES

class TestEnergyBIOES:

    def test_shapes(self):
        E_kin = np.array([1.0, 0.5, 0.1, 0.01, 0.001], dtype=np.float64)
        E_pot = np.array([0.1, 0.5, 1.0, 2.0, 3.0], dtype=np.float64)
        tags, ps, pe, regime, lr, ct = _temporal.compute_bioes_energy(
            E_kin, E_pot)
        assert tags.shape == (5,)
        assert lr.shape == (5,)

    def test_kinetic_regime(self):
        """All kinetic-dominated should be a single regime-0 phase."""
        E_kin = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        E_pot = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        _, ps, pe, regime, _, _ = _temporal.compute_bioes_energy(E_kin, E_pot)
        assert regime[0] == 0  # kinetic


# Cascade

class TestCascade:

    def test_activation_shapes(self):
        signals = np.array([[1.0, 0.0, 0.0],
                            [0.8, 0.5, 0.0],
                            [0.5, 0.8, 0.3]], dtype=np.float64)
        at, ao, ar = _temporal.cascade_edge_activation(signals, 0.1)
        assert at.shape == (3,)
        assert ar.shape == (3,)

    def test_source_activates_first(self):
        signals = np.array([[1.0, 0.0, 0.0],
                            [0.8, 0.5, 0.0],
                            [0.5, 0.8, 0.3]], dtype=np.float64)
        at, ao, ar = _temporal.cascade_edge_activation(signals, 0.1)
        # Edge 0 has signal 1.0 at t=0, should activate first
        assert at[0] == 0
        assert ao[0] == 0

    def test_never_activated(self):
        signals = np.array([[0.0, 0.0],
                            [0.0, 0.0]], dtype=np.float64)
        at, ao, ar = _temporal.cascade_edge_activation(signals, 0.1)
        assert np.all(at == -1)
        assert ao.shape[0] == 0


# Integration through TemporalRex

class TestTemporalRexIntegration:

    def test_temporal_index(self):
        snaps = _make_snapshots()
        trex = TemporalRex(snaps)
        ti = trex.temporal_index
        assert ti is not None

    def test_edge_lifecycle(self):
        snaps = _make_snapshots()
        trex = TemporalRex(snaps)
        lc = trex.edge_lifecycle
        assert lc is not None

    def test_edge_metrics(self):
        snaps = _make_snapshots()
        trex = TemporalRex(snaps)
        ec, bc, dc = trex.edge_metrics
        assert ec.shape == (3,)

    def test_at_returns_rexgraph(self):
        snaps = _make_snapshots()
        trex = TemporalRex(snaps)
        r0 = trex.at(0)
        assert isinstance(r0, RexGraph)
        assert r0.nE == 3

    def test_T_property(self):
        snaps = _make_snapshots()
        trex = TemporalRex(snaps)
        assert trex.T == 3

    def test_bioes_energy(self):
        snaps = _make_snapshots()
        trex = TemporalRex(snaps)
        E_kin = np.array([1.0, 0.5, 0.1], dtype=np.float64)
        E_pot = np.array([0.1, 0.5, 1.0], dtype=np.float64)
        result = trex.bioes_energy(E_kin, E_pot)
        tags = result[0]
        assert tags.shape == (3,)
