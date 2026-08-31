import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.rex_state import from_state, to_state


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
    from rexgraph.io.bundle import load_rex, save_rex
    g = RexGraph.from_graph([0, 1, 2], [1, 2, 0], g_channel="normalized")
    g.set_vertex_attribution(np.array([[0.1], [0.2], [0.3]]))
    p = str(tmp_path / "g.rex")
    save_rex(p, g)                       # used to crash on load with '(0, 0)'
    r = load_rex(p)
    assert r._g_channel == "normalized" and r._w_boundary


def test_rex_bundle_bad_version_raises(tmp_path):
    import json

    from rexgraph.io.bundle import load_rex, save_rex
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
    FileStore = pytest.importorskip(
        "agent.rcdb",
        reason="requires the optional rexgraph-agent package",
    ).FileStore
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


#### final-review fixes (C1 name collision, I2 grade>=3, M3 scalar/array, M4 edge_types)
def test_cell_metadata_key_with_double_underscore_roundtrips(tmp_path):
    from rexgraph.io.bundle import load_rex, save_rex
    from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex
    g = _simple()
    g.attach_metadata(1, 0, "node__id", "X7")       # '__' in a user key used to KeyError on load
    p = str(tmp_path / "g.rex"); save_rex(p, g)
    assert load_rex(p).get_metadata(1, 0, "node__id") == "X7"
    sp = str(tmp_path / "g.safetensors"); rex_to_safetensors(g, sp)
    assert safetensors_to_rex(sp).get_metadata(1, 0, "node__id") == "X7"


def test_cell_metadata_slash_vs_underscore_keys_do_not_collide(tmp_path):
    from rexgraph.io.bundle import load_rex, save_rex
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


#### the tensor codec ###########################################################

def test_an_arange_is_stored_as_its_endpoints():
    from rexgraph.io.rex_state import decode_tensors, encode_tensors
    t = {"idx": np.arange(5, 4005, dtype=np.uint16)}
    spec = encode_tensors(t)
    assert "idx" not in t, "the array IS its endpoints"
    assert spec["idx"] == {"c": "arange", "start": 5, "n": 4000, "dtype": "<u2"}
    decode_tensors(t, spec)
    assert np.array_equal(t["idx"], np.arange(5, 4005))
    assert t["idx"].dtype == np.uint16


def test_a_monotone_array_is_stored_as_its_first_difference():
    """A CSR pointer is the integral of the arity vector; the arities are what is small."""
    from rexgraph.io.rex_state import decode_tensors, encode_tensors
    ptr = np.cumsum(np.concatenate([[0], np.full(2000, 7)])).astype(np.int64)
    t = {"boundary_ptr": ptr.copy()}
    spec = encode_tensors(t)
    assert spec["boundary_ptr"]["c"] == "delta"
    assert set(np.asarray(t["boundary_ptr"])[1:].tolist()) == {7}, "the arity, repeated"
    decode_tensors(t, spec)
    assert np.array_equal(t["boundary_ptr"], ptr)
    assert t["boundary_ptr"].dtype == ptr.dtype


def test_only_the_monotone_columns_of_a_two_column_array_are_differenced():
    """Spans are (start, length): the starts ascend, the lengths do not."""
    from rexgraph.io.rex_state import decode_tensors, encode_tensors
    spans = np.array([[0, 10], [10, 4], [14, 25], [39, 3]], dtype=np.uint32)
    t = {"spans": spans.copy()}
    spec = encode_tensors(t)
    assert spec["spans"]["cols"] == [0], "only the start column ascends"
    decode_tensors(t, spec)
    assert np.array_equal(t["spans"], spans) and t["spans"].dtype == np.uint32


def test_floats_and_byte_buffers_pass_through_untouched():
    from rexgraph.io.rex_state import encode_tensors
    t = {"w_E": np.array([0.5, 0.25, 0.125]),
         "labels": np.frombuffer(b"s0s1s2", dtype=np.uint8).copy()}
    before = {k: v.copy() for k, v in t.items()}
    spec = encode_tensors(t)
    assert "w_E" not in spec
    for k, v in before.items():
        assert np.array_equal(t[k], v)


def test_the_codec_never_reorders_a_boundary_column():
    """The arguments of a column share `1/(k-1)` and look interchangeable, but
    `_leaf_digests` hashes them in order, so sorting them would move the Merkle root.
    Sorting is not among the transforms; `boundary_idx` is not monotone and is left
    exactly as it was."""
    from rexgraph.io.rex_state import decode_tensors, encode_tensors
    idx = np.array([3, 1, 2, 9, 4, 0, 7], dtype=np.int32)
    t = {"boundary_idx": idx.copy()}
    spec = encode_tensors(t)
    decode_tensors(t, spec)
    assert np.array_equal(t["boundary_idx"], idx)


def test_a_complex_round_trips_through_the_codec_byte_for_byte():
    from rexgraph.graph import RexGraph
    from rexgraph.io.rex_state import from_state, to_state
    from rexgraph.sectioning import add_coarsening, add_sectioning, sectionings_of

    r = RexGraph(sources=list(range(64)), targets=[(i + 1) % 64 for i in range(64)])
    add_sectioning(r, "sentence", {f"s{i}": [2 * i, 2 * i + 1] for i in range(32)},
                   spans={f"s{i}": (10 * i, 10) for i in range(32)})
    add_coarsening(r, "paragraph", "sentence", [i // 2 for i in range(32)],
                   [f"p{i}" for i in range(16)])
    st = to_state(r)
    back = from_state(st, verify=True)
    assert int(back.nV) == int(r.nV) and int(back.nE) == int(r.nE)
    assert np.array_equal(np.asarray(back._boundary_ptr), np.asarray(r._boundary_ptr))
    assert np.array_equal(np.asarray(back._boundary_idx), np.asarray(r._boundary_idx))
    a, b = sectionings_of(r), sectionings_of(back)
    for k in a:
        assert np.array_equal(np.asarray(a[k].indices), np.asarray(b[k].indices))
        assert np.array_equal(np.asarray(a[k].spans), np.asarray(b[k].spans)) \
            if a[k].spans is not None else b[k].spans is None


def test_a_version_1_bundle_still_loads():
    """v1 carries no codec, so the reader must accept it and not decode anything."""
    from rexgraph.graph import RexGraph
    from rexgraph.io.rex_state import (CODEC_TENSOR, RexState, from_state,
                                       state_digest, to_state)
    r = RexGraph(sources=[0, 1, 2], targets=[1, 2, 0])
    st = to_state(r)
    t = dict(st.tensors)
    spec = None
    if CODEC_TENSOR in t:
        import json
        spec = json.loads(bytes(np.asarray(t.pop(CODEC_TENSOR)).tobytes()).decode())
        from rexgraph.io.rex_state import decode_tensors
        decode_tensors(t, spec)
    h = dict(st.header)
    h["format_version"] = 1
    h["digest_names"] = sorted(t)
    h["digest"] = state_digest(t)
    back = from_state(RexState(t, h), verify=True)
    assert int(back.nE) == int(r.nE)


def test_an_unknown_format_version_is_refused_by_name():
    from rexgraph.graph import RexGraph
    from rexgraph.io.rex_state import RexState, from_state, to_state
    st = to_state(RexGraph(sources=[0, 1], targets=[1, 0]))
    h = dict(st.header); h["format_version"] = 99
    with pytest.raises(ValueError, match="unsupported rex state format_version"):
        from_state(RexState(st.tensors, h), verify=False)
