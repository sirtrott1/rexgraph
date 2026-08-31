import numpy as np
import pytest

from rexgraph.graph import RexGraph, _cell_state, make_edge_delta


def _delta_between(prev, curr, directed=False):
    pp, pi, pw, ps = _cell_state(prev)
    cp, ci, cw, cs = _cell_state(curr)
    pw = np.zeros(prev._nE, np.float64) if pw is None else np.asarray(pw, np.float64)
    ps = np.ones(prev._nE, np.int32) if ps is None else np.asarray(ps, np.int32)
    cw = np.zeros(curr._nE, np.float64) if cw is None else np.asarray(cw, np.float64)
    cs = np.ones(curr._nE, np.int32) if cs is None else np.asarray(cs, np.int32)
    return make_edge_delta(pp, pi, pw, ps, cp, ci, cw, cs, directed=directed)


def _cell_state_full(rex):
    """_cell_state, but with w_E/signs materialized to their zeros/ones defaults
    instead of None, matching the 4-tuple shape TemporalRex._last_state stores
    internally (_append_index_entry always fills these before diffing)."""
    ptr, idx, w_E, signs = _cell_state(rex)
    nE = int(ptr.shape[0] - 1)
    w_E = np.zeros(nE, np.float64) if w_E is None else np.asarray(w_E, np.float64)
    signs = np.ones(nE, np.int32) if signs is None else np.asarray(signs, np.int32)
    return ptr, idx, w_E, signs


def test_encode_delta_full_born_died_modified():
    prev = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32),
                    w_E=np.array([10.0, 20.0], np.float64))
    curr = RexGraph(sources=np.array([0, 2], np.int32), targets=np.array([1, 3], np.int32),
                    w_E=np.array([99.0, 30.0], np.float64))
    # edge (0,1) persists with w_E 10 -> 99 (modified); (1,2) died; (2,3) born
    d = _delta_between(prev, curr)
    # one born cell (2,3), arity 2
    assert list(d.born_offsets) == [0, 2]
    assert list(d.born_cols) == [2, 3]
    assert list(d.born_wE) == [30.0]
    # one died key (the (1,2) edge), one modified key (the (0,1) edge, new w_E 99)
    assert d.died_keys.shape[0] == 1
    assert d.mod_keys.shape[0] == 1
    assert list(d.mod_wE) == [99.0]


def test_make_edge_delta_stamps_directed():
    from rexgraph.graph import make_edge_delta
    prev_ptr = np.array([0, 2], np.int32); prev_idx = np.array([0, 1], np.int32)
    curr_ptr = np.array([0, 2], np.int32); curr_idx = np.array([0, 1], np.int32)
    pw = np.zeros(1, np.float64); ps = np.ones(1, np.int32)
    d = make_edge_delta(prev_ptr, prev_idx, pw, ps, curr_ptr, curr_idx, pw, ps, directed=True)
    assert d.directed is True         # the scheme its keys were computed with is preserved
    d2 = make_edge_delta(prev_ptr, prev_idx, pw, ps, curr_ptr, curr_idx, pw, ps, directed=False)
    assert d2.directed is False


def test_encode_face_delta_born_died_by_key():
    from rexgraph.graph import _face_state, make_face_delta
    # triangle with one face over edges [0,1,2]
    prev = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32),
                    B2_col_ptr=np.array([0, 3], np.int32), B2_row_idx=np.array([0, 1, 2], np.int32),
                    B2_vals=np.array([1.0, 1.0, 1.0], np.float64))
    # curr: same edges, face removed
    curr = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32))
    fd = make_face_delta(_face_state(prev), _face_state(curr), directed=False)
    assert fd.died_face_keys.shape[0] == 1        # the face died
    assert fd.born_offsets.shape[0] == 1          # no born faces (offsets = [0])


def test_make_face_delta_stamps_directed():
    from rexgraph.graph import _face_state, make_face_delta
    tri = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32),
                   B2_col_ptr=np.array([0, 3], np.int32), B2_row_idx=np.array([0, 1, 2], np.int32),
                   B2_vals=np.array([1.0, 1.0, 1.0], np.float64))
    empty = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32))
    fd = make_face_delta(_face_state(tri), _face_state(empty), directed=True)
    assert fd.directed is True and fd.died_face_keys.shape[0] == 1
    fd2 = make_face_delta(_face_state(tri), _face_state(empty), directed=False)
    assert fd2.directed is False


def test_apply_edge_delta_replays_born_died_modified():
    from rexgraph.graph import apply_edge_delta
    prev = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32),
                    w_E=np.array([10.0, 20.0], np.float64))
    curr = RexGraph(sources=np.array([0, 2], np.int32), targets=np.array([1, 3], np.int32),
                    w_E=np.array([99.0, 30.0], np.float64))
    d = _delta_between(prev, curr)          # the same helper the delta-store tests use
    live = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32),
                    w_E=np.array([10.0, 20.0], np.float64))
    apply_edge_delta(live, d)
    live.compact()
    # live should now equal curr's connectivity + weights (order may differ; compare as sets)
    got = set(zip(live.sources.tolist(), live.targets.tolist(), strict=False))
    assert got == {(0, 1), (2, 3)}
    # the persisting (0,1) edge carries its modified weight 99
    idx = [i for i, (s, t) in enumerate(zip(live.sources.tolist(), live.targets.tolist(), strict=False)) if (s, t) == (0, 1)][0]
    assert live._w_E[idx] == 99.0


def test_apply_edge_delta_raises_on_unresolvable_modified_key():
    import pytest

    from rexgraph.graph import TemporalDelta, apply_edge_delta
    live = RexGraph(sources=np.array([0], np.int32), targets=np.array([1], np.int32),
                    w_E=np.array([1.0], np.float64))
    # a delta whose mod_keys references a cell that does not exist in `live`
    bad = TemporalDelta(
        born_cols=np.zeros(0, np.int32), born_offsets=np.zeros(1, np.int32),
        born_wE=np.zeros(0, np.float64), born_signs=np.zeros(0, np.int32),
        died_keys=np.zeros(0, np.int64),
        mod_keys=np.array([999999], np.int64),          # not present in `live`
        mod_wE=np.array([7.0], np.float64), mod_signs=np.array([1], np.int32),
        directed=False)
    with pytest.raises(ValueError):
        apply_edge_delta(live, bad)


def test_apply_face_delta_replays_born_and_died():
    from rexgraph.graph import _face_state, apply_face_delta, make_face_delta
    tri = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32),
                   B2_col_ptr=np.array([0, 3], np.int32), B2_row_idx=np.array([0, 1, 2], np.int32),
                   B2_vals=np.array([1.0, 1.0, 1.0], np.float64))
    empty = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32))

    # died: live starts with the face, the delta records that it died
    died_delta = make_face_delta(_face_state(tri), _face_state(empty), directed=False)
    live_died = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32),
                         B2_col_ptr=np.array([0, 3], np.int32), B2_row_idx=np.array([0, 1, 2], np.int32),
                         B2_vals=np.array([1.0, 1.0, 1.0], np.float64))
    apply_face_delta(live_died, died_delta)
    live_died.compact()
    assert live_died._nF == 0

    # born: live starts without the face, the delta records that it was born
    born_delta = make_face_delta(_face_state(empty), _face_state(tri), directed=False)
    live_born = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32))
    apply_face_delta(live_born, born_delta)
    live_born.compact()
    assert live_born._nF == 1
    assert set(live_born._B2_row_idx.tolist()) == {0, 1, 2}


def test_reconstruct_at_matches_direct_build():
    # build a store from full snapshots via the existing constructor, then reconstruct
    s0 = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32))
    s1 = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 3], np.int32))
    from rexgraph.graph import TemporalRex
    trex = TemporalRex([(s0.sources, s0.targets), (s1.sources, s1.targets)])
    r1 = trex.reconstruct_at(1)
    assert set(zip(r1.sources.tolist(), r1.targets.tolist(), strict=False)) == {(0, 1), (1, 2), (2, 3)}
    assert np.array_equal(np.asarray(r1.betti), np.asarray(s1.betti))


def test_parallel_relations_reconstruct_without_changing_topology():
    from rexgraph import multiplicity_dimension
    from rexgraph.graph import TemporalRex

    t0 = RexGraph(sources=np.array([0, 1], np.int32),
                  targets=np.array([1, 2], np.int32))
    t1 = RexGraph(sources=np.array([0, 0, 1], np.int32),
                  targets=np.array([1, 1, 2], np.int32))
    trex = TemporalRex([(t0.sources, t0.targets), (t1.sources, t1.targets)])
    trex._ensure_index()

    rebuilt = trex.reconstruct_at(1)
    assert trex._index_deltas[1] is None
    assert 1 in trex._index_checkpoints
    assert rebuilt.nE == t1.nE == 3
    assert rebuilt.betti == t1.betti == (1, 1, 0)
    assert multiplicity_dimension(rebuilt) == multiplicity_dimension(t1) == 1
    assert np.array_equal(rebuilt._boundary_ptr, t1._boundary_ptr)
    assert np.array_equal(rebuilt._boundary_idx, t1._boundary_idx)


def test_parallel_relation_checkpoint_preserves_attribution_and_faces():
    from rexgraph.graph import TemporalRex

    parallel = RexGraph(
        sources=np.array([0, 0], np.int32),
        targets=np.array([1, 1], np.int32),
        w_E=np.array([3.0, 7.0], np.float64),
        signs=np.array([1, -1], np.int32),
        B2_col_ptr=np.array([0, 2], np.int32),
        B2_row_idx=np.array([0, 1], np.int32),
        B2_vals=np.array([1.0, -1.0], np.float64),
    )
    trex = TemporalRex([])
    trex.append_snapshot(parallel)

    rebuilt = trex.reconstruct_at(0)
    assert rebuilt.nE == 2 and rebuilt.nF == 1
    for name in ("_boundary_ptr", "_boundary_idx", "_w_E", "_signs",
                 "_B2_col_ptr", "_B2_row_idx", "_B2_vals"):
        assert np.array_equal(getattr(rebuilt, name), getattr(parallel, name)), name


def test_repeated_faces_reconstruct_as_a_multiset():
    from rexgraph.graph import TemporalRex

    edges = dict(
        sources=np.array([0, 1, 2], np.int32),
        targets=np.array([1, 2, 0], np.int32),
    )
    one_face = RexGraph(
        **edges,
        B2_col_ptr=np.array([0, 3], np.int32),
        B2_row_idx=np.array([0, 1, 2], np.int32),
        B2_vals=np.ones(3, np.float64),
    )
    two_faces = RexGraph(
        **edges,
        B2_col_ptr=np.array([0, 3, 6], np.int32),
        B2_row_idx=np.array([0, 1, 2, 0, 1, 2], np.int32),
        B2_vals=np.ones(6, np.float64),
    )
    trex = TemporalRex([])
    trex.append_snapshot(one_face)
    trex.append_snapshot(two_faces)

    rebuilt = trex.reconstruct_at(1)
    assert trex._index_deltas[1] is None
    assert rebuilt.nF == two_faces.nF == 2
    assert rebuilt.betti == two_faces.betti == (1, 0, 1)
    assert np.array_equal(rebuilt._B2_col_ptr, two_faces._B2_col_ptr)
    assert np.array_equal(rebuilt._B2_row_idx, two_faces._B2_row_idx)


def test_leaving_multiplicity_checkpoints_before_resuming_deltas():
    from rexgraph.graph import TemporalRex

    def graph(edges):
        return RexGraph(sources=np.array([s for s, _ in edges], np.int32),
                        targets=np.array([t for _, t in edges], np.int32))

    refs = [
        graph([(0, 1), (0, 1), (1, 2), (2, 3), (3, 4)]),
        graph([(0, 1), (1, 2), (2, 3), (3, 4)]),
        graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
    ]
    trex = TemporalRex([(r.sources, r.targets) for r in refs])
    trex._ensure_index()

    assert trex._index_deltas[1] is None  # the transition away from multiplicity
    assert trex._index_deltas[2] is not None  # injective states may use deltas again
    for t, ref in enumerate(refs):
        rebuilt = trex.reconstruct_at(t)
        assert rebuilt.nE == ref.nE
        assert rebuilt.betti == ref.betti


def test_legacy_parallel_relation_delta_refuses_lossy_replay():
    from rexgraph.graph import TemporalRex

    base = RexGraph(sources=np.array([0, 1], np.int32),
                    targets=np.array([1, 2], np.int32))
    parallel = RexGraph(sources=np.array([0, 0, 1], np.int32),
                        targets=np.array([1, 1, 2], np.int32))
    legacy = TemporalRex([])
    legacy.append_snapshot(base)
    legacy._index_deltas.append(_delta_between(base, parallel))
    legacy._index_face_deltas.append(None)
    legacy._T = 2

    with pytest.raises(ValueError, match="cannot represent parallel relations"):
        legacy.reconstruct_at(1)


def test_parallel_relations_survive_delta_store_serialization(tmp_path):
    pytest.importorskip("safetensors")
    from rexgraph import multiplicity_dimension
    from rexgraph.graph import TemporalRex
    from rexgraph.io.safetensors_bridge import (
        safetensors_to_temporal_rex,
        temporal_rex_to_safetensors,
    )

    base = RexGraph(sources=np.array([0, 1], np.int32),
                    targets=np.array([1, 2], np.int32))
    parallel = RexGraph(sources=np.array([0, 0, 1], np.int32),
                        targets=np.array([1, 1, 2], np.int32),
                        w_E=np.array([2.0, 5.0, 9.0], np.float64),
                        signs=np.array([1, -1, 1], np.int32))
    trex = TemporalRex([])
    trex.append_snapshot(base)
    trex.append_snapshot(parallel)
    path = tmp_path / "parallel.safetensors"
    temporal_rex_to_safetensors(trex, path)

    loaded = safetensors_to_temporal_rex(path)
    rebuilt = loaded.reconstruct_at(1)
    assert rebuilt.nE == 3
    assert rebuilt.betti == parallel.betti
    assert multiplicity_dimension(rebuilt) == 1
    assert np.array_equal(rebuilt._boundary_ptr, parallel._boundary_ptr)
    assert np.array_equal(rebuilt._boundary_idx, parallel._boundary_idx)
    assert np.array_equal(rebuilt._w_E, parallel._w_E)
    assert np.array_equal(rebuilt._signs, parallel._signs)


def test_reconstruct_at_with_deaths_across_deltas():
    # moderate churn against a stable 9-vertex path core (0-1-2-...-8) so at
    # least one reconstruct_at call replays 2+ real deltas including an edge
    # death, not just checkpoints. This is the regression case for the
    # stable-vertex-id bug: in-place delta replay (apply_edge_delta ->
    # remove_edges -> compact on the next _ensure_clean) renumbers vertices
    # to a contiguous range once an edge death orphans one (dropping the
    # orphan and shifting every higher-numbered live vertex down). A later
    # delta's died/mod keys were computed by _ensure_index against the
    # ORIGINAL, un-shifted vertex-id scheme, so they silently fail to match
    # the now-renumbered live complex: a death that should land is dropped
    # (its np.isin match comes back empty) and a phantom edge from the
    # renumbering shows up instead. reconstruct_at must do a key-level
    # replay (never mutating a live rex, never renumbering) instead.
    #
    # Step 1 (t=0 -> t=1) drops edge (0,1), orphaning vertex 0 and shifting
    # every vertex 1..8 down by one under in-place replay. Step 2 (t=1 -> t=2)
    # drops edge (7,8): its died-key was computed against the original ids
    # (7, 8), which no longer resolves once vertex 8 has been silently
    # renumbered to 7, so the buggy replay leaves a phantom (0, 1) edge in
    # place and fails to remove the dead (7, 8) cell.
    e0 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    e1 = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    e2 = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    e3 = e2 + [(50, 51)]
    e4 = list(e2)

    def mk(edges):
        s = np.array([a for a, _ in edges], np.int32)
        tt = np.array([b for _, b in edges], np.int32)
        return RexGraph(sources=s, targets=tt)

    refs = [mk(e0), mk(e1), mk(e2), mk(e3), mk(e4)]
    from rexgraph.graph import TemporalRex
    trex = TemporalRex([(r.sources, r.targets) for r in refs])
    trex._ensure_index()
    # moderate churn: assert not every step is a checkpoint (so deltas are actually applied)
    assert any(d is not None for d in trex._index_deltas), "test must exercise delta application"
    for t in range(len(refs)):
        r = trex.reconstruct_at(t)
        got = set(zip(r.sources.tolist(), r.targets.tolist(), strict=False))
        ref = set(zip(refs[t].sources.tolist(), refs[t].targets.tolist(), strict=False))
        assert got == ref, "t=%d: %s != %s" % (t, got, ref)
        assert np.array_equal(np.asarray(r.betti), np.asarray(refs[t].betti))


def _as_delta_backed(built):
    """Force a snapshots-backed TemporalRex into delta-backed mode: build the
    checkpoint/delta index while snapshots are still materialized, then drop
    the materialized snapshots so every analysis member must go through
    `reconstruct_at` via `_all_snapshots`/`_snapshot_at`."""
    built._ensure_index()
    built._snapshots_materialized = False
    built._snapshots = []
    return built


def _growing_snapshots():
    return [RexGraph(sources=np.arange(k, dtype=np.int32), targets=np.arange(1, k + 1, dtype=np.int32))
            for k in (2, 3, 4)]


def test_analysis_works_delta_backed():
    s = _growing_snapshots()
    from rexgraph.graph import TemporalRex
    built = TemporalRex([(x.sources, x.targets) for x in s])
    ref_life = built.edge_lifecycle
    # a delta-backed store: same data, but force _snapshots_materialized = False
    delta_backed = TemporalRex([(x.sources, x.targets) for x in s])
    delta_backed._ensure_index()
    delta_backed._snapshots_materialized = False
    delta_backed._snapshots = []
    got_life = delta_backed.edge_lifecycle
    assert np.array_equal(np.asarray(ref_life[0]), np.asarray(got_life[0]))


def test_edge_metrics_delta_backed_matches_snapshots_backed():
    s = _growing_snapshots()
    from rexgraph.graph import TemporalRex
    ref = TemporalRex([(x.sources, x.targets) for x in s]).edge_metrics
    got = _as_delta_backed(TemporalRex([(x.sources, x.targets) for x in s])).edge_metrics
    for r, g in zip(ref, got, strict=False):
        assert np.array_equal(np.asarray(r), np.asarray(g))


def test_temporal_index_delta_backed_matches_snapshots_backed():
    # temporal_index returns (checkpoints, deltas, checkpoint_times); checkpoints
    # and deltas are ragged lists of (time, arrays...) tuples, so compare the
    # checkpoint_times array and each checkpoint's (time, sources, targets)
    # element-by-element rather than np.asarray()-ing the whole ragged structure
    s = _growing_snapshots()
    from rexgraph.graph import TemporalRex
    ref_cps, ref_deltas, ref_cp_times = TemporalRex([(x.sources, x.targets) for x in s]).temporal_index
    got_cps, got_deltas, got_cp_times = _as_delta_backed(
        TemporalRex([(x.sources, x.targets) for x in s])).temporal_index
    assert np.array_equal(ref_cp_times, got_cp_times)
    assert len(ref_cps) == len(got_cps)
    for (rt, rs, rtg), (gt, gs, gtg) in zip(ref_cps, got_cps, strict=False):
        assert rt == gt
        assert np.array_equal(rs, gs)
        assert np.array_equal(rtg, gtg)


def test_bioes_delta_backed_matches_snapshots_backed():
    s = _growing_snapshots()
    from rexgraph.graph import TemporalRex
    betti = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.int64)
    ref_trex = TemporalRex([(x.sources, x.targets) for x in s])
    ref = ref_trex.bioes(betti)
    got_trex = _as_delta_backed(TemporalRex([(x.sources, x.targets) for x in s]))
    got = got_trex.bioes(betti)
    for r, g in zip(ref, got, strict=False):
        assert np.array_equal(np.asarray(r), np.asarray(g))


def test_temporal_persistence_delta_backed_matches_snapshots_backed():
    s = _growing_snapshots()
    from rexgraph.graph import TemporalRex
    ref = TemporalRex([(x.sources, x.targets) for x in s]).temporal_persistence()
    got = _as_delta_backed(TemporalRex([(x.sources, x.targets) for x in s])).temporal_persistence()
    assert ref.keys() == got.keys()
    for k in ref:
        assert np.array_equal(np.asarray(ref[k]), np.asarray(got[k]))


def test_cascade_wavefront_delta_backed_matches_snapshots_backed():
    s = _growing_snapshots()
    from rexgraph.graph import TemporalRex
    ref_trex = TemporalRex([(x.sources, x.targets) for x in s])
    # cascade_wavefront reads edge endpoints off snapshot 0 only, so the
    # signal matrix is sized to that first snapshot's edge count
    signals = np.ones((ref_trex.T, s[0]._nE), dtype=np.float64)
    ref_wf, ref_cumul, ref_vact = ref_trex.cascade_wavefront(signals)
    got_trex = _as_delta_backed(TemporalRex([(x.sources, x.targets) for x in s]))
    got_wf, got_cumul, got_vact = got_trex.cascade_wavefront(signals)
    assert np.array_equal(ref_cumul, got_cumul)
    assert np.array_equal(ref_vact, got_vact)
    for r, g in zip(ref_wf, got_wf, strict=False):
        assert np.array_equal(r, g)


def test_streaming_append_equals_batch_build():
    graphs = [RexGraph(sources=np.arange(k, dtype=np.int32), targets=np.arange(1, k + 1, dtype=np.int32))
              for k in (2, 3, 4, 5)]
    from rexgraph.graph import TemporalRex
    batch = TemporalRex([(g.sources, g.targets) for g in graphs])
    batch._ensure_index()
    stream = TemporalRex([(graphs[0].sources, graphs[0].targets)])
    for g in graphs[1:]:
        stream.append_snapshot(g)
    assert stream._T == batch._T
    assert list(stream._index_cp_times) == list(batch._index_cp_times)
    assert np.array_equal(np.asarray(stream.edge_metrics[0]), np.asarray(batch.edge_metrics[0]))


def test_ensure_index_is_atomic_on_failure(monkeypatch):
    from rexgraph.graph import TemporalRex
    graphs = [RexGraph(sources=np.arange(k, dtype=np.int32),
                       targets=np.arange(1, k + 1, dtype=np.int32)) for k in (2, 3, 4, 5)]
    trex = TemporalRex([(g.sources, g.targets) for g in graphs])
    # make the per-entry index build blow up on the 2nd entry
    orig = TemporalRex._append_index_entry
    calls = {"n": 0}
    def boom(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected build failure")
        return orig(self, *a, **k)
    monkeypatch.setattr(TemporalRex, "_append_index_entry", boom)
    with pytest.raises(RuntimeError):
        trex._ensure_index()
    assert trex._index_cp_times is None            # rolled back to unbuilt, not half-built
    # after removing the fault, a retry rebuilds cleanly
    monkeypatch.setattr(TemporalRex, "_append_index_entry", orig)
    trex._ensure_index()
    assert trex._index_cp_times is not None
    r = trex.reconstruct_at(3)
    assert set(zip(r.sources.tolist(), r.targets.tolist(), strict=False)) == \
           set(zip(graphs[3].sources.tolist(), graphs[3].targets.tolist(), strict=False))


def test_delta_serialization_roundtrip(tmp_path):
    pytest.importorskip("safetensors")
    from rexgraph.graph import TemporalRex
    from rexgraph.io.safetensors_bridge import (
        safetensors_to_temporal_rex,
        temporal_rex_to_safetensors,
    )
    graphs = [RexGraph(sources=np.arange(k, dtype=np.int32), targets=np.arange(1, k + 1, dtype=np.int32),
                       w_E=np.arange(k, dtype=np.float64) + 1.0) for k in (2, 3, 4)]
    trex = TemporalRex([(g.sources, g.targets) for g in graphs])
    trex._ensure_index()
    p = tmp_path / "trex.safetensors"
    temporal_rex_to_safetensors(trex, str(p))
    back = safetensors_to_temporal_rex(str(p))
    assert back._snapshots_materialized is False
    for t in range(3):
        r = back.reconstruct_at(t) if not back._snapshots_materialized else back.at(t)
        ref = graphs[t]
        assert set(zip(r.sources.tolist(), r.targets.tolist(), strict=False)) == set(zip(ref.sources.tolist(), ref.targets.tolist(), strict=False))


def test_general_mode_roundtrips_as_general(tmp_path):
    pytest.importorskip("safetensors")
    from rexgraph.graph import TemporalRex
    from rexgraph.io.safetensors_bridge import (
        safetensors_to_temporal_rex,
        temporal_rex_to_safetensors,
    )
    snaps = [(np.array([0, 2, 4], np.int32), np.array([0, 1, 2, 0, 1, 3], np.int32))]  # general CSR
    trex = TemporalRex(snaps, general=True)
    p = tmp_path / "g.safetensors"
    temporal_rex_to_safetensors(trex, str(p))
    back = safetensors_to_temporal_rex(str(p))
    assert back._general is True                      # was silently reverting to False


def test_changed_edges_reports_added_and_removed():
    from rexgraph.flow.navigator import changed_edges
    prev = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 3], np.int32))
    curr = RexGraph(sources=np.array([0, 2, 3], np.int32), targets=np.array([1, 3, 4], np.int32))
    # edge (1,2) removed; edges (2,3)->stays, (3,4) added; (0,1) stays
    ch = changed_edges(prev, curr)
    added_pairs = {(int(curr.sources[i]), int(curr.targets[i])) for i in ch.added}
    assert (3, 4) in added_pairs
    assert ch.removed.shape[0] == 1        # the (1,2) edge's key


def test_changed_edges_handles_branching_hyperedges():
    from rexgraph.core._temporal import cell_keys_of
    from rexgraph.flow.navigator import changed_edges
    # prev: edge (0,1), branching hyperedge over {2,3,4}
    prev = RexGraph(boundary_ptr=np.array([0, 2, 5], np.int32),
                    boundary_idx=np.array([0, 1, 2, 3, 4], np.int32))
    # curr: edge (0,1) persists, the branching hyperedge is dropped, a new one over {5,6,7} is born
    curr = RexGraph(boundary_ptr=np.array([0, 2, 5], np.int32),
                    boundary_idx=np.array([0, 1, 5, 6, 7], np.int32))
    ch = changed_edges(prev, curr)
    assert ch.added.shape[0] == 1 and ch.added[0] == 1     # curr's second cell (the new hyperedge)
    assert ch.removed.shape[0] == 1                        # prev's branching hyperedge died
    prev_keys = cell_keys_of(prev._boundary_ptr, prev._boundary_idx, prev._directed)
    assert ch.removed[0] == prev_keys[1]


def test_changed_edges_edge_identity_respects_directedness():
    from rexgraph.flow.navigator import changed_edges
    # undirected (default): reversing an edge's endpoints is NOT a change
    prev_u = RexGraph(sources=np.array([0], np.int32), targets=np.array([1], np.int32))
    curr_u = RexGraph(sources=np.array([1], np.int32), targets=np.array([0], np.int32))
    ch_u = changed_edges(prev_u, curr_u)
    assert ch_u.added.shape[0] == 0 and ch_u.removed.shape[0] == 0
    # directed: the reversed edge IS a different edge (one removed, one added)
    prev_d = RexGraph(sources=np.array([0], np.int32), targets=np.array([1], np.int32), directed=True)
    curr_d = RexGraph(sources=np.array([1], np.int32), targets=np.array([0], np.int32), directed=True)
    ch_d = changed_edges(prev_d, curr_d)
    assert ch_d.added.shape[0] == 1 and ch_d.removed.shape[0] == 1


def test_face_signs_survive_reconstruct():
    from rexgraph.graph import TemporalRex
    tri = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32),
                   B2_col_ptr=np.array([0, 3], np.int32), B2_row_idx=np.array([0, 1, 2], np.int32),
                   B2_vals=np.array([1.0, -1.0, 1.0], np.float64))
    trex = TemporalRex([(tri.sources, tri.targets)],
                       face_snapshots=[(tri._B2_col_ptr, tri._B2_row_idx, tri._B2_vals)])
    r = trex.reconstruct_at(0)
    assert np.allclose(np.sort(r._B2_vals), np.sort([1.0, -1.0, 1.0]))   # not fabricated ones


def test_time_varying_weights_and_sign_flip_reconstruct():
    a = RexGraph(sources=np.array([0], np.int32), targets=np.array([1], np.int32),
                 w_E=np.array([1.0], np.float64), signs=np.array([1], np.int32))
    b = RexGraph(sources=np.array([0], np.int32), targets=np.array([1], np.int32),
                 w_E=np.array([5.0], np.float64), signs=np.array([-1], np.int32))
    from rexgraph.graph import TemporalRex
    trex = TemporalRex([(a.sources, a.targets)])
    trex._last_state = _cell_state_full(a)     # helper: (ptr, idx, w_E, signs) with defaults filled
    trex.append_snapshot(b)
    r1 = trex.reconstruct_at(1)
    assert r1._w_E[0] == 5.0 and r1._signs[0] == -1


def test_branching_arity_reconstruct():
    # t=0: an ordinary edge (0,1) plus a branching hyperedge over {2,3,4} (arity 3)
    s0 = RexGraph(boundary_ptr=np.array([0, 2, 5], np.int32),
                  boundary_idx=np.array([0, 1, 2, 3, 4], np.int32))
    # t=1: edge (0,1) persists unchanged; the arity-3 hyperedge dies and a new
    # arity-4 hyperedge over {2,3,4,5} is born, so reconstruct_at must replay a
    # born/died pair at the KEY level and preserve the new cell's full arity
    s1 = RexGraph(boundary_ptr=np.array([0, 2, 6], np.int32),
                  boundary_idx=np.array([0, 1, 2, 3, 4, 5], np.int32))
    from rexgraph.graph import TemporalRex
    trex = TemporalRex(
        [(s0._boundary_ptr, s0._boundary_idx), (s1._boundary_ptr, s1._boundary_idx)],
        general=True)
    r1 = trex.reconstruct_at(1)
    ptr = np.asarray(r1._boundary_ptr)
    idx = np.asarray(r1._boundary_idx)
    arities = np.diff(ptr)
    branch = [i for i in range(len(arities)) if arities[i] > 2]
    assert len(branch) == 1, "expected exactly one surviving branching cell"
    b = branch[0]
    assert arities[b] == 4
    assert set(idx[ptr[b]:ptr[b + 1]].tolist()) == {2, 3, 4, 5}


def test_faces_resolve_by_key_after_edge_renumbering():
    # a triangle face over edges (1,2),(2,3),(3,1) sits at boundary positions
    # 1,2,3 alongside an unrelated edge (0,1) at position 0. At t=1 the
    # unrelated edge (0,1) dies, so every triangle edge's POSITION shifts down
    # by one in the rebuilt complex even though none of them changed. The face
    # must still resolve to the right (renumbered) edges because
    # reconstruct_at looks them up by canonical KEY, not by stale position.
    s0 = RexGraph(sources=np.array([0, 1, 2, 3], np.int32), targets=np.array([1, 2, 3, 1], np.int32),
                  B2_col_ptr=np.array([0, 3], np.int32), B2_row_idx=np.array([1, 2, 3], np.int32),
                  B2_vals=np.array([1.0, 1.0, 1.0], np.float64))
    s1 = RexGraph(sources=np.array([1, 2, 3], np.int32), targets=np.array([2, 3, 1], np.int32),
                  B2_col_ptr=np.array([0, 3], np.int32), B2_row_idx=np.array([0, 1, 2], np.int32),
                  B2_vals=np.array([1.0, 1.0, 1.0], np.float64))
    from rexgraph.graph import TemporalRex
    trex = TemporalRex(
        [(s0.sources, s0.targets), (s1.sources, s1.targets)],
        face_snapshots=[(s0._B2_col_ptr, s0._B2_row_idx, s0._B2_vals),
                        (s1._B2_col_ptr, s1._B2_row_idx, s1._B2_vals)])
    r1 = trex.reconstruct_at(1)
    assert r1._nF == 1
    face_edges = set()
    for row in r1._B2_row_idx.tolist():
        s, t = int(r1.sources[row]), int(r1.targets[row])
        face_edges.add(frozenset((s, t)))
    assert face_edges == {frozenset((1, 2)), frozenset((2, 3)), frozenset((3, 1))}


def test_checkpoint_boundary_reconstruct_high_churn():
    # a stream that alternates a small delta step with a step that replaces
    # half the edges (churn/nE > 0.5), forcing repeated full checkpoints, not
    # just an ever growing delta chain. reconstruct_at must be correct both
    # AT a checkpoint and across an intervening delta.
    def mk(edges):
        s = np.array([a for a, _ in edges], np.int32)
        t = np.array([b for _, b in edges], np.int32)
        return RexGraph(sources=s, targets=t)

    e0 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    e1 = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (100, 101)]        # small delta
    e2 = [(5, 6), (6, 7), (7, 8), (100, 101), (200, 201), (202, 203), (204, 205), (206, 207)]  # 4/8 churn
    e3 = [(6, 7), (7, 8), (100, 101), (200, 201), (202, 203), (204, 205), (206, 207), (300, 301)]  # small delta
    e4 = [(7, 8), (100, 101), (400, 401), (402, 403), (404, 405), (406, 407), (408, 409), (410, 411)]  # big churn
    seqs = [e0, e1, e2, e3, e4]
    refs = [mk(e) for e in seqs]

    from rexgraph.graph import TemporalRex
    trex = TemporalRex([(r.sources, r.targets) for r in refs])
    trex._ensure_index()
    assert len(trex._index_cp_times) > 1, "high churn stream must produce more than one checkpoint"
    for t in range(len(refs)):
        r = trex.reconstruct_at(t)
        got = set(zip(r.sources.tolist(), r.targets.tolist(), strict=False))
        assert got == set(seqs[t]), "t=%d: %s != %s" % (t, got, set(seqs[t]))


def test_delta_serialized_smaller_than_full_snapshots(tmp_path):
    pytest.importorskip("safetensors")
    from rexgraph.graph import TemporalRex
    from rexgraph.io.safetensors_bridge import rex_to_safetensors, temporal_rex_to_safetensors
    # a large stable base with a SMALL per-step edit (a handful of edges swapped
    # each step, real world "big graph, tiny periodic edits" shape): full
    # per-step serialization pays for the whole boundary CSR every step, the
    # delta index pays for one full checkpoint plus O(churn) per-step deltas.
    # (A tiny-graph/many-steps version of this test is dominated by the
    # safetensors per-tensor header overhead on both sides rather than by the
    # actual O(nE) vs O(churn) payload difference, so it does not exercise the
    # property this test is checking; this shape does.)
    rng = np.random.default_rng(0)
    n_edges = 2000
    src = np.arange(n_edges, dtype=np.int32)
    tgt = np.arange(1, n_edges + 1, dtype=np.int32)
    graphs = [RexGraph(sources=src.copy(), targets=tgt.copy())]
    for step in range(10):
        idxs = rng.choice(n_edges, size=5, replace=False)
        tgt = tgt.copy()
        tgt[idxs] = tgt[idxs] + 100000 + step   # 5 edges die, 5 new edges born
        graphs.append(RexGraph(sources=src.copy(), targets=tgt.copy()))
    trex = TemporalRex([(g.sources, g.targets) for g in graphs])

    delta_path = tmp_path / "delta.safetensors"
    temporal_rex_to_safetensors(trex, str(delta_path))
    delta_size = delta_path.stat().st_size

    full_size = 0
    for i, g in enumerate(graphs):
        p = tmp_path / ("full_%d.safetensors" % i)
        rex_to_safetensors(g, str(p))
        full_size += p.stat().st_size

    assert delta_size < full_size, "delta-serialized (%d bytes) not smaller than full-snapshot (%d bytes)" % (
        delta_size, full_size)


def test_matrix_free_reconstruct(monkeypatch):
    import numpy.linalg as nla
    import scipy.sparse.linalg as ssla
    calls = []
    for mod, name in [(nla, "eig"), (nla, "eigh"), (nla, "svd"), (nla, "pinv"),
                      (ssla, "eigsh"), (ssla, "svds")]:
        if hasattr(mod, name):
            o = getattr(mod, name)
            monkeypatch.setattr(mod, name, lambda *a, _o=o, _n=name, **k: (calls.append(_n), _o(*a, **k))[1])
    from rexgraph.graph import TemporalRex
    gs = [RexGraph(sources=np.arange(k, dtype=np.int32), targets=np.arange(1, k + 1, dtype=np.int32))
          for k in (2, 3, 4, 5, 6)]
    trex = TemporalRex([(gs[0].sources, gs[0].targets)])
    for g in gs[1:]:
        trex.append_snapshot(g)
    _ = trex.reconstruct_at(4)
    assert calls == [], f"delta path hit a dense solver: {calls}"


def test_serialization_preserves_full_attribution(tmp_path):
    pytest.importorskip("safetensors")
    from rexgraph.graph import TemporalRex
    from rexgraph.io.safetensors_bridge import (
        safetensors_to_temporal_rex,
        temporal_rex_to_safetensors,
    )

    def rx(wbase):
        k = 3
        return RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 3], np.int32),
                        w_E=np.arange(k, dtype=np.float64) + wbase,
                        signs=np.array([1, -1, 1], np.int32))

    a, b = rx(10.0), rx(20.0)                 # same edges, weights change over time
    trex = TemporalRex([])                     # APPEND path (full RexGraphs carry attribution)
    trex.append_snapshot(a)
    trex.append_snapshot(b)
    p = tmp_path / "attr.safetensors"
    temporal_rex_to_safetensors(trex, str(p))
    back = safetensors_to_temporal_rex(str(p))

    def attr_map(r):
        m = {}
        for i in range(r.nE):
            s, t = int(r.sources[i]), int(r.targets[i])
            w = float(r._w_E[i]) if r._w_E is not None else None
            sg = int(r._signs[i]) if r._signs is not None else None
            m[(s, t)] = (w, sg)
        return m

    for t, ref in enumerate([a, b]):
        r = back.reconstruct_at(t)
        assert attr_map(r) == attr_map(ref), "t=%d attribution not preserved through serialization" % t


def test_face_signs_survive_serialization():
    """test_face_signs_survive_reconstruct only checks the in-memory replay path;
    this checks the same signed face survives an actual serialize -> load ->
    reconstruct round trip through the APPEND path (Slice D's real usage)."""
    pytest.importorskip("safetensors")
    import os
    import tempfile

    from rexgraph.graph import TemporalRex
    from rexgraph.io.safetensors_bridge import (
        safetensors_to_temporal_rex,
        temporal_rex_to_safetensors,
    )
    tri = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32),
                   B2_col_ptr=np.array([0, 3], np.int32), B2_row_idx=np.array([0, 1, 2], np.int32),
                   B2_vals=np.array([1.0, -1.0, 1.0], np.float64))
    trex = TemporalRex([])
    trex.append_snapshot(tri)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "face.safetensors")
        temporal_rex_to_safetensors(trex, p)
        back = safetensors_to_temporal_rex(p)
        r = back.reconstruct_at(0)
        assert np.allclose(np.sort(r._B2_vals), np.sort([-1.0, 1.0, 1.0])), "face signs not fabricated as ones"
