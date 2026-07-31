import numpy as np
from rexgraph.graph import RexGraph, _TIER_B1_ONLY


def _line_graph(n=6):
    src = np.arange(n - 1, dtype=np.int32)
    tgt = np.arange(1, n, dtype=np.int32)
    return RexGraph(sources=src, targets=tgt)


def test_invalidate_drops_only_named_tiers():
    g = _line_graph()
    _ = g.L0            # a B1_ONLY cached_property: materializes into __dict__
    _ = g.betti         # a GLOBAL cached_property
    assert "L0" in g.__dict__ and "betti" in g.__dict__
    g._invalidate(_TIER_B1_ONLY)
    assert "L0" not in g.__dict__       # B1_ONLY dropped
    assert "betti" in g.__dict__        # GLOBAL survived (not named)


def test_new_slots_initialized_clean():
    g = _line_graph()
    assert g._pending_edges is None and g._pending_faces is None
    assert g._live_edges is None and g._live_faces is None
    assert g._dirty is False and g._last_remap is None


def test_add_edges_is_deferred_then_golden():
    # deferred: arrays untouched until a read
    g = _line_graph(4)                       # edges (0,1),(1,2),(2,3)
    ptr_before, idx_before = g._boundary_ptr, g._boundary_idx
    g.add_edges(np.array([3], np.int32), np.array([0], np.int32))   # close a cycle
    assert g._boundary_ptr is ptr_before     # no copy at call time
    assert g._dirty is True
    assert g._nE == 4                         # logical count updated eagerly
    # golden: after materialization, identical to a full build of the final edge set
    full = RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
                    targets=np.array([1, 2, 3, 0], np.int32))
    g._ensure_clean()
    assert np.array_equal(g._boundary_ptr, full._boundary_ptr)
    assert np.array_equal(g._boundary_idx, full._boundary_idx)
    assert g._nV == full._nV and g._nE == full._nE
    assert np.allclose(g.L0.toarray() if hasattr(g.L0, "toarray") else g.L0,
                       full.L0.toarray() if hasattr(full.L0, "toarray") else full.L0)
    assert np.array_equal(np.asarray(g.betti), np.asarray(full.betti))


def test_operator_read_triggers_flush():
    g = _line_graph(4)
    g.add_edges(np.array([3], np.int32), np.array([0], np.int32))
    _ = g.L0                                  # reading an operator must flush first
    assert g._dirty is False


def test_e2f_refreshes_after_edge_append():
    g = _line_graph(4)
    _ = g._e2f                               # cache it at nE=3
    g.add_edges(np.array([3], np.int32), np.array([0], np.int32))
    ptr, idx = g._e2f                        # read triggers flush + must be re-derived
    assert ptr.shape[0] == g._nE + 1 == 5    # CSR ptr length tracks the new edge count


def test_subcomplex_sees_appended_edges():
    g = _line_graph(4)                       # edges (0,1),(1,2),(2,3)
    g.add_edges(np.array([3], np.int32), np.array([0], np.int32))   # add (3,0)
    e_mask = np.ones(4, dtype=np.uint8)       # all 4 edges, including the appended one
    v_mask, e_mask_out, f_mask = g.subcomplex(e_mask=e_mask)
    assert e_mask_out.shape[0] == g._nE == 4  # the appended edge is visible, not lost
    assert v_mask.shape[0] == g._nV == 4


def _triangle():
    # 3 edges forming a triangle: (0,1),(1,2),(2,0)
    return RexGraph(sources=np.array([0, 1, 2], np.int32),
                    targets=np.array([1, 2, 0], np.int32))


def test_add_faces_golden_against_from_cells():
    g = _triangle()
    # one face over edges [0,1,2] with consistent orientation signs
    g.add_faces([np.array([0, 1, 2], np.int32)], [np.array([1, 1, 1], np.float64)])
    g._ensure_clean()
    full = RexGraph(
        sources=np.array([0, 1, 2], np.int32),
        targets=np.array([1, 2, 0], np.int32),
        B2_col_ptr=np.array([0, 3], np.int32),
        B2_row_idx=np.array([0, 1, 2], np.int32),
        B2_vals=np.array([1.0, 1.0, 1.0], np.float64),
    )
    assert np.array_equal(g._B2_col_ptr, full._B2_col_ptr)
    assert np.array_equal(g._B2_row_idx, full._B2_row_idx)
    assert np.allclose(g._B2_vals, full._B2_vals)
    assert g._nF == 1
    assert np.array_equal(np.asarray(g.betti), np.asarray(full.betti))


def test_add_faces_keeps_b1_caches_warm():
    g = _triangle()
    _ = g.L0                     # B1_ONLY cache
    g.add_faces([np.array([0, 1, 2], np.int32)], [np.array([1, 1, 1], np.float64)])
    _ = g.betti                  # triggers flush + invalidation
    assert "L0" in g.__dict__    # B1_ONLY survived a face-only append


def test_compact_boundary_kernel_drops_and_renumbers():
    from rexgraph.core import _rex
    # 4 edges over 5 vertices: (0,1),(1,2),(2,3),(3,4); drop edge 1 -> vertex 2 stays
    # (still in edge 2), no orphan; edges renumber 0,1(drop),2->1,3->2
    ptr = np.array([0, 2, 4, 6, 8], np.int32)
    idx = np.array([0, 1, 1, 2, 2, 3, 3, 4], np.int32)
    live = np.array([1, 0, 1, 1], np.int32)
    new_ptr, new_idx, nV_new, v_map, e_map = _rex.compact_boundary(ptr, idx, live)
    assert list(e_map) == [0, -1, 1, 2]
    # surviving columns are (0,1),(2,3),(3,4) renumbered through v_map
    rebuilt = [tuple(new_idx[new_ptr[c]:new_ptr[c + 1]]) for c in range(len(new_ptr) - 1)]
    remap = {int(v): int(v_map[v]) for v in range(len(v_map)) if v_map[v] >= 0}
    assert rebuilt == [(remap[0], remap[1]), (remap[2], remap[3]), (remap[3], remap[4])]
    assert nV_new == 5      # no vertex orphaned in this case


def test_remove_edges_is_deferred_and_carries_state():
    g = RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
                 targets=np.array([1, 2, 3, 0], np.int32),
                 w_E=np.array([10.0, 20.0, 30.0, 40.0], np.float64),
                 signs=np.array([1, -1, 1, -1], np.int32))
    ptr_before = g._boundary_ptr
    g.remove_edges(np.array([0, 1, 0, 0], np.int32))    # drop edge 1 (the (1,2) edge)
    assert g._dirty is True and g._boundary_ptr is ptr_before   # deferred
    remap = g.compact()
    # edge 1 dropped; edges 0,2,3 -> new 0,1,2
    assert list(remap.edge_map) == [0, -1, 1, 2]
    assert np.allclose(g._w_E, [10.0, 30.0, 40.0])       # attribution carried (was dropped before)
    assert np.array_equal(g._signs, [1, 1, -1])          # signs carried
    full = RexGraph(sources=np.array([0, 2, 3], np.int32),
                    targets=np.array([1, 3, 0], np.int32))
    assert np.array_equal(np.asarray(g.betti), np.asarray(full.betti))


def test_remove_faces_drops_column():
    g = _triangle()
    g.add_faces([np.array([0, 1, 2], np.int32)], [np.array([1, 1, 1], np.float64)])
    g.add_faces([np.array([0, 1, 2], np.int32)], [np.array([1, 1, 1], np.float64)])
    g._ensure_clean()
    assert g._nF == 2
    g.remove_faces(np.array([1, 0], np.int32))           # drop face 0, keep face 1
    g.compact()
    assert g._nF == 1


def test_remove_then_append_edge_in_one_batch():
    g = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32))
    g.remove_edges(np.array([0, 0], np.int32))        # keep both, mask sized to nE=2
    g.add_edges(np.array([3], np.int32), np.array([4], np.int32))   # nE -> 3
    remap = g.compact()
    assert g._nE == 3                                  # appended edge survives (not dropped)
    assert list(remap.edge_map) == [0, 1, 2]


def test_remove_then_append_face_in_one_batch():
    g = _triangle()
    g.add_faces([np.array([0, 1, 2], np.int32)], [np.array([1, 1, 1], np.float64)])
    g._ensure_clean()                                  # now nF=1
    g.remove_faces(np.array([1], np.int32))            # mask sized to nF=1; nonzero removes face 0
    g.add_faces([np.array([0, 1, 2], np.int32)], [np.array([1, 1, 1], np.float64)])  # nF -> 2
    g.compact()                                        # must not IndexError
    assert g._nF == 1                                  # original face removed, appended one kept


def test_remove_edges_and_faces_same_batch():
    # face 0 uses edges [0,1,2]; face 1 uses edges [0,1,2] too. Remove edge 2 (drops both
    # edge-referencing faces) is too coarse; instead: 4 edges, face 0 on edges [0,1,3] (uses
    # edge 3), face 1 on edges [0,1,2] (no edge 3). Remove edge 3 -> face 0 auto-dropped;
    # ALSO remove_faces face 1 -> both gone.
    g = RexGraph(sources=np.array([0, 1, 2, 3], np.int32), targets=np.array([1, 2, 0, 0], np.int32))
    g.add_faces([np.array([0, 1, 3], np.int32)], [np.array([1, 1, 1], np.float64)])   # face 0 uses edge 3
    g.add_faces([np.array([0, 1, 2], np.int32)], [np.array([1, 1, 1], np.float64)])   # face 1
    g._ensure_clean()
    assert g._nF == 2
    g.remove_edges(np.array([0, 0, 0, 1], np.int32))   # remove edge 3 -> face 0 drops
    g.remove_faces(np.array([0, 1], np.int32))         # remove face 1 (original index 1)
    g.compact()
    assert g._nF == 0                                  # both faces gone; correct face targeted


def test_selective_invalidation_edge_append_keeps_face_caches():
    g = _triangle()
    g.add_faces([np.array([0, 1, 2], np.int32)], [np.array([1, 1, 1], np.float64)])
    _ = g._B2_hodge_dual         # B2_ONLY cache
    _ = g.L0                     # B1_ONLY cache
    g.add_edges(np.array([2], np.int32), np.array([0], np.int32))   # edge append
    _ = g.betti                  # trigger flush + invalidation
    assert "_B2_hodge_dual" in g.__dict__   # B2_ONLY survived an edge-only append
    assert "L0" not in g.__dict__           # B1_ONLY invalidated


def test_odelta_append_defers_array_work():
    g = _line_graph(200)
    ptr, idx = g._boundary_ptr, g._boundary_idx
    g.add_edges(np.array([199], np.int32), np.array([0], np.int32))
    assert g._boundary_ptr is ptr and g._boundary_idx is idx     # untouched
    assert len(g._pending_edges["src"]) == 1


def test_append_onto_general_boundary_complex():
    # a complex built via the general boundary constructor (not sources/targets),
    # then a 2-arity edge appended onto it; the B1 dual must reflect the new column.
    g = RexGraph(boundary_ptr=np.array([0, 2], np.int32),
                 boundary_idx=np.array([0, 1], np.int32))   # one edge (0,1)
    g.add_edges(np.array([1], np.int32), np.array([2], np.int32))   # append edge (1,2)
    assert g._nE == 2 and g._nV == 3
    from rexgraph.core._sparse import to_scipy_csr
    b1 = to_scipy_csr(g._B1_dual)                          # nV x nE signed incidence
    assert b1.shape == (3, 2)                              # rebuilt to include the appended edge
    assert (b1[:, 1] != 0).sum() == 2                      # the new column has its two endpoints


def test_public_counts_flush_after_deferred_removal():
    g = _line_graph(6)                       # 5 edges
    g.remove_edges(np.concatenate([np.array([1], np.int32), np.zeros(4, np.int32)]))  # drop edge 0
    assert g.nE == 4                          # count reflects the removal immediately, not 5
    sig = np.ones(g.nE)                       # the natural downstream pattern must be consistent
    assert sig.shape[0] == g.nE == 4


def test_compact_returns_identity_on_noop_after_consuming():
    g = _line_graph(4)                                   # edges (0,1),(1,2),(2,3)
    g.remove_edges(np.array([1, 0, 0], np.int32))        # drop edge 0
    r1 = g.compact()
    assert list(r1.edge_map) == [-1, 0, 1]               # real renumbering returned once
    r2 = g.compact()                                     # no new mutation
    assert list(r2.edge_map) == [0, 1]                   # identity over the now-2-edge graph, NOT [-1,0,1]


def test_compact_remap_retrievable_after_operator_flush():
    g = _line_graph(4)
    g.remove_edges(np.array([1, 0, 0], np.int32))
    _ = g.L0                                             # an operator read flushes + sets _last_remap
    r = g.compact()                                     # must still return the compaction's remap
    assert list(r.edge_map) == [-1, 0, 1]


def test_add_hyperedges_arity3_roundtrips():
    from rexgraph.core._sparse import to_scipy_csr
    g = _line_graph(4)                                   # 3 edges over vertices 0..3
    g.add_hyperedges([np.array([0, 1, 2], np.int32)])    # one arity-3 branching cell
    assert g._nE == 4
    b1 = to_scipy_csr(g._B1_dual)
    assert b1.shape[1] == 4                               # a 4th column exists
    assert (b1[:, 3] != 0).sum() == 3                     # the new column has its 3 incident vertices


def test_mixed_edges_and_hyperedges_batch_carries_attribution():
    g = RexGraph(sources=np.array([0], np.int32), targets=np.array([1], np.int32),
                 w_E=np.array([5.0], np.float64))
    g.add_edges(np.array([1], np.int32), np.array([2], np.int32), w_E=np.array([7.0], np.float64))
    g.add_hyperedges([np.array([0, 1, 2], np.int32)], w_E=np.array([9.0], np.float64))
    g._ensure_clean()
    assert g._nE == 3
    # 2-arity edges land first (indices 0,1), the hyperedge last (index 2)
    assert np.allclose(g._w_E, [5.0, 7.0, 9.0])
    from rexgraph.core._sparse import to_scipy_csr
    b1 = to_scipy_csr(g._B1_dual)
    assert (b1[:, 2] != 0).sum() == 3                     # index 2 is the arity-3 cell


def test_hyperedge_removed_and_compacted():
    g = _line_graph(4)
    g.add_hyperedges([np.array([0, 1, 2], np.int32)])    # edge 3 = arity-3 cell
    g._ensure_clean()
    assert g._nE == 4
    g.remove_edges(np.array([0, 0, 0, 1], np.int32))     # remove the hyperedge (index 3)
    g.compact()
    assert g._nE == 3                                     # general-arity removal works end to end


def test_mutation_path_is_matrix_free(monkeypatch):
    import numpy.linalg as nla
    import scipy.sparse.linalg as ssla
    calls = []
    for mod, name in [(nla, "eig"), (nla, "eigh"), (nla, "svd"), (nla, "pinv"),
                      (ssla, "eigsh"), (ssla, "svds")]:
        if hasattr(mod, name):
            orig = getattr(mod, name)
            monkeypatch.setattr(mod, name,
                                lambda *a, _o=orig, _n=name, **k: (calls.append(_n), _o(*a, **k))[1])
    g = _line_graph(8)
    g.add_edges(np.array([7], np.int32), np.array([0], np.int32))
    g.remove_edges(np.concatenate([np.array([1], np.int32), np.zeros(g._nE - 1, np.int32)]))
    g.compact()
    _ = g._B1_dual               # operator read after mutation
    assert calls == [], f"mutation path hit a dense solver: {calls}"
