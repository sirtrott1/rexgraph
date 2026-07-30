import numpy as np
import pytest
from rexgraph.graph import RexGraph
from rexgraph.io.rex_state import to_state, from_state, RexState, FORMAT_VERSION


def _simple():
    return RexGraph(sources=np.array([0, 1, 2, 0], np.int32),
                    targets=np.array([1, 2, 3, 3], np.int32),
                    w_E=np.array([1.5, 2.0, 0.5, 3.0]), signs=[1, -1, 1, -1])


def test_roundtrip_core_and_signs_and_gchannel():
    g = RexGraph.from_graph([0, 1, 2], [1, 2, 0], g_channel="normalized")
    r = from_state(to_state(g))
    assert np.array_equal(np.asarray(r.B1), np.asarray(g.B1))
    assert r._g_channel == "normalized"                      # was silently lost
    assert np.allclose(np.asarray(r.coherence), np.asarray(g.coherence))


def test_roundtrip_signs_exact():
    g = _simple()
    r = from_state(to_state(g))
    assert np.array_equal(np.asarray(r._signs), np.asarray(g._signs))


def test_roundtrip_w_boundary_ragged_no_crash():
    # tuple keys, mixed scalar/array values: the exact shape that crashes today
    g = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32),
                 w_boundary={(0, 0): np.array([1.0, 2.0]), (1, 1): 5.0})
    r = from_state(to_state(g))
    assert set(r._w_boundary.keys()) == {(0, 0), (1, 1)}
    assert np.allclose(np.atleast_1d(r._w_boundary[(0, 0)]), [1.0, 2.0])
    assert float(np.atleast_1d(r._w_boundary[(1, 1)])[0]) == 5.0


def test_roundtrip_attribution():
    g = RexGraph(sources=[0, 1, 2], targets=[1, 2, 0])
    g.set_vertex_attribution(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
    r = from_state(to_state(g))
    assert r._w_boundary  # attribution (stored in _w_boundary) survived; no (0,0) crash


def test_roundtrip_cell_metadata():
    g = _simple()
    g.attach_metadata(1, 0, "kind", "activation")
    g.attach_metadata(1, 0, "confidence", 0.9)
    r = from_state(to_state(g))
    assert r.get_metadata(1, 0, "kind") == "activation"
    assert float(r.get_metadata(1, 0, "confidence")) == 0.9


def test_self_describing_labels_and_types():
    g = _simple()
    g._agent_meta = {"vertex_labels": ["v0", "v1", "v2", "v3"], "source": "unit",
                     "columns": [{"name": "w", "dtype": "float64", "role": "numeric"}]}
    st = to_state(g)
    r = from_state(st)
    assert r._agent_meta["vertex_labels"] == ["v0", "v1", "v2", "v3"]
    assert r._agent_meta["columns"][0]["role"] == "numeric"


def test_nested_rex_roundtrips():
    child = RexGraph.from_graph([0, 1], [1, 2])
    g = _simple()
    g.attach_metadata(0, 0, "schema", child)          # a cell value that is itself a complex
    r = from_state(to_state(g))
    sub = r.get_metadata(0, 0, "schema")
    assert isinstance(sub, RexGraph) and sub.nE == child.nE


def test_header_is_kb_scale_not_fat_json():
    import json
    g = _simple()
    st = to_state(g)
    # all numeric payload is in tensors; the header carries no raw arrays
    hjson = json.dumps(st.header)
    assert len(hjson) < 4096
    assert "boundary_idx" in st.tensors and "signs" in st.tensors
    for v in st.header.values():
        assert not (isinstance(v, list) and len(v) > 64 and all(isinstance(x, (int, float)) for x in v))


def test_version_guard():
    g = _simple()
    st = to_state(g)
    st.header["format_version"] = 999
    with pytest.raises(ValueError):
        from_state(st)


def test_rex_bundle_roundtrips_attribution_and_gchannel(tmp_path):
    from rexgraph.io.bundle import save_rex, load_rex
    g = RexGraph.from_graph([0, 1, 2], [1, 2, 0], g_channel="normalized")
    g.set_vertex_attribution(np.array([[0.1], [0.2], [0.3]]))
    p = str(tmp_path / "g.rex")
    save_rex(p, g)                       # used to crash on load with '(0, 0)'
    r = load_rex(p)
    assert r._g_channel == "normalized" and r._w_boundary


def test_rex_bundle_bad_version_raises(tmp_path):
    import json
    from rexgraph.io.bundle import save_rex, load_rex
    p = str(tmp_path / "g.rex")
    save_rex(p, _simple())
    mf = tmp_path / "g.rex" / "MANIFEST.json"
    d = json.loads(mf.read_text()); d["format_version"] = 999; mf.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_rex(p)


def test_safetensors_roundtrips_signs_and_attribution(tmp_path):
    from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex
    g = _simple()                                # has signs [1,-1,1,-1]
    g.set_vertex_attribution(np.array([[0.1], [0.2], [0.3], [0.4]]))
    p = str(tmp_path / "g.safetensors")
    rex_to_safetensors(g, p)                     # used to drop signs + crash on attribution
    r = safetensors_to_rex(p)
    assert np.array_equal(np.asarray(r._signs), np.asarray(g._signs))
    assert r._w_boundary


def test_rcstore_roundtrips_full_state(tmp_path):
    from agent.rcdb import FileStore
    g = _simple()
    g._agent_meta = {"vertex_labels": ["a", "b", "c", "d"], "source": "unit"}
    store = FileStore(str(tmp_path / "rcdb"))
    store.put("g1", g)
    r = store.get("g1")
    assert np.array_equal(np.asarray(r._signs), np.asarray(g._signs))   # signs survive the RCDB
    assert r._agent_meta["vertex_labels"] == ["a", "b", "c", "d"]        # self-describing


def test_trustgraph_and_agentic_roundtrip_still_work():
    # standalone-plus-interop: an agent-built complex, persisted and reloaded, keeps agent meta and
    # its deterministic invariants recompute identically.
    g = _simple()
    g._agent_meta = {"vertex_labels": ["a", "b", "c", "d"], "source": "csv"}
    r = from_state(to_state(g))
    assert r._agent_meta["source"] == "csv"
    assert list(r.betti) == list(g.betti)


# --- final-review fixes (C1 name collision, I2 grade>=3, M3 scalar/array, M4 edge_types) ---
def test_cell_metadata_key_with_double_underscore_roundtrips(tmp_path):
    from rexgraph.io.bundle import save_rex, load_rex
    from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex
    g = _simple()
    g.attach_metadata(1, 0, "node__id", "X7")       # '__' in a user key used to KeyError on load
    p = str(tmp_path / "g.rex"); save_rex(p, g)
    assert load_rex(p).get_metadata(1, 0, "node__id") == "X7"
    sp = str(tmp_path / "g.safetensors"); rex_to_safetensors(g, sp)
    assert safetensors_to_rex(sp).get_metadata(1, 0, "node__id") == "X7"


def test_cell_metadata_slash_vs_underscore_keys_do_not_collide(tmp_path):
    from rexgraph.io.bundle import save_rex, load_rex
    g = _simple()
    g.attach_metadata(1, 0, "a/b", "SLASH")
    g.attach_metadata(1, 0, "a__b", "UNDER")
    p = str(tmp_path / "g.rex"); save_rex(p, g)
    r = load_rex(p)
    assert r.get_metadata(1, 0, "a/b") == "SLASH"      # no collapse/corruption
    assert r.get_metadata(1, 0, "a__b") == "UNDER"


def test_grade3_boundary_roundtrips():
    verts = 4
    edges = [[i, j] for i in range(4) for j in range(i + 1, 4)]
    eidx = {(e[0], e[1]): k for k, e in enumerate(edges)}
    tris_v = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    triangles = [[eidx[(min(a, b), max(a, b))] for a, b in ((t[0], t[1]), (t[0], t[2]), (t[1], t[2]))]
                 for t in tris_v]
    g = RexGraph.from_cells([verts, edges, triangles, [[0, 1, 2, 3]]])
    assert getattr(g, "_graded_duals", None)             # has grade-3
    r = from_state(to_state(g))
    assert r._graded_duals is not None
    assert list(r.betti) == list(g.betti)                # grade-3 homology survives (was corrupted)


def test_wboundary_scalar_vs_len1_array_fidelity():
    g = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32),
                 w_boundary={(0, 0): 5.0, (1, 1): np.array([7.0])})
    r = from_state(to_state(g))
    assert isinstance(r._w_boundary[(0, 0)], float) and r._w_boundary[(0, 0)] == 5.0
    v = r._w_boundary[(1, 1)]
    assert isinstance(v, np.ndarray) and v.shape == (1,) and v[0] == 7.0


def test_edge_types_not_stored_but_recomputes():
    g = _simple()
    st = to_state(g)
    assert "edge_types" not in st.tensors               # deterministic: not stored (dead weight)
    r = from_state(st)
    assert np.array_equal(np.asarray(r.edge_types), np.asarray(g.edge_types))   # recomputed on load
