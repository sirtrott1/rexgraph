import inspect

import numpy as np

from rexgraph.core._temporal import cell_keys_of
from rexgraph.flow.navigator import changed_edges
from rexgraph.flow.online import GreensCochainField
from rexgraph.graph import RexGraph, TemporalRex


def _rex(src, tgt):
    return RexGraph(sources=np.asarray(src, np.int32), targets=np.asarray(tgt, np.int32))


def test_online_module_imports_no_torch():
    import rexgraph.flow.online as online
    src = inspect.getsource(online)
    assert "torch" not in src                       # native spine: torch never appears in the field


def test_correction_reduces_region_residual():
    r = _rex([0, 1, 2], [1, 2, 0])                   # triangle -> L_C is non-None
    f = GreensCochainField()
    region = np.array([0, 1, 2], np.int64)
    pred = f.predict(r, region)                      # field starts at 0
    target = np.array([1.0, 1.0, 1.0])
    before = float(np.abs(target - pred).mean())
    out = f.correct(r, region, target)
    keys = cell_keys_of(r._boundary_ptr, r._boundary_idx, r._directed)
    after_field = np.array([f.phi[int(keys[i])] for i in region])
    after = float(np.abs(target - after_field).mean())
    assert out["updated"] is True
    assert after < before                            # one relational step moved toward the target


def test_predict_recorded_before_observe():
    S = [([0, 0, 1], [1, 2, 3]), ([0, 0, 1, 2], [1, 2, 3, 4]),
         ([0, 0, 1, 2, 3], [1, 2, 3, 4, 5])]
    trex = TemporalRex([(np.asarray(s, np.int32), np.asarray(t, np.int32)) for s, t in S])
    f = GreensCochainField()
    r0 = f.predict_then_observe(0, None, trex.at(0))
    assert r0["target"] is None and r0["updated"] is False   # nothing to observe on the first step
    r1 = f.predict_then_observe(1, changed_edges(trex.at(0), trex.at(1)), trex.at(1))
    assert r1["pred"] is not None                    # a prediction was recorded for step 1
    assert r1["target"] is not None                  # step 0's pending region was observed at step 1
    assert r1["error"] >= 0.0


def test_field_keyed_by_canonical_key_survives_index_shift():
    f = GreensCochainField()
    r0 = _rex([0, 1, 2], [1, 2, 3])                  # edges (0,1)(1,2)(2,3)
    keys0 = cell_keys_of(r0._boundary_ptr, r0._boundary_idx, r0._directed)
    f.phi[int(keys0[2])] = 5.0                       # field on edge (2,3)
    r1 = _rex([1, 2], [2, 3])                         # (0,1) gone -> (2,3) shifts index
    keys1 = cell_keys_of(r1._boundary_ptr, r1._boundary_idx, r1._directed)
    idx = [j for j in range(r1.nE) if int(keys1[j]) == int(keys0[2])][0]
    vec = f._phi_vec(keys1)
    assert vec[idx] == 5.0                            # same key -> same value despite the index shift


def test_predict_propagates_beyond_seeded_edge():
    # Locks in that the Green's solve is REAL, not the identity fallback: seeding
    # one edge must spread the field to neighbors over L_C. An identity fallback
    # (L_C None) would leave exactly one nonzero entry.
    r = _rex([0, 1, 2], [1, 2, 0])                   # triangle: L_C connects all edges
    f = GreensCochainField()
    keys = cell_keys_of(r._boundary_ptr, r._boundary_idx, r._directed)
    f.phi[int(keys[0])] = 1.0                        # seed only edge 0
    pred = f.predict(r, np.array([0, 1, 2], np.int64))
    assert int(np.sum(np.abs(pred) > 1e-9)) > 1      # propagation spreads; identity gives exactly 1


def test_l_c_cache_reuses_snapshot_build(monkeypatch):
    import rexgraph.flow.online as online
    calls = {"n": 0}
    real = online.build_sparse_channels
    def counting(rex):
        calls["n"] += 1
        return real(rex)
    monkeypatch.setattr(online, "build_sparse_channels", counting)
    S = [([0, 0, 1], [1, 2, 3]), ([0, 0, 1, 2], [1, 2, 3, 4]),
         ([0, 0, 1, 2, 3], [1, 2, 3, 4, 5])]
    trex = TemporalRex([(np.asarray(s, np.int32), np.asarray(t, np.int32)) for s, t in S])
    f = GreensCochainField()
    snaps = [trex.at(i) for i in range(3)]
    f.predict_then_observe(0, None, snaps[0])
    f.predict_then_observe(1, changed_edges(snaps[0], snaps[1]), snaps[1])
    f.predict_then_observe(2, changed_edges(snaps[1], snaps[2]), snaps[2])
    # Each distinct snapshot object's L_C is built at most once (cache reuse across the
    # predict-at-t -> correct-at-(t+1) path). Three distinct snapshots -> at most 3 builds,
    # versus the no-cache path which rebuilds on every predict AND correct.
    assert calls["n"] <= 3
