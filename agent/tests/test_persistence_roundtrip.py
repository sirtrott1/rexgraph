"""The workspace store, end to end.

`server/persistence.py` is where a workspace actually lives: settings, the
document/session index, saved complexes, activity, query history, conversations and
export. 122 of its lines were never executed by a test, and it is the layer that
decides whether work survives a restart.

Every test here writes and reads back. A store that accepts a write and returns
something different on read is the failure mode worth catching, so the assertions
compare against what went in rather than against a shape.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from agent.server import persistence as P

from rexgraph.graph import RexGraph

WS = "roundtrip-ws"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A workspace rooted in tmp_path.

    The base directory is read when asked, so setting REXGRAPH_CONFIG_DIR in a test
    has no effect: the module attribute is what has to move.
    """
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _rex():
    """A triangle plus a pendant: one cycle, one isolated-ish tail."""
    return RexGraph(sources=np.array([0, 1, 2, 0], np.int32),
                    targets=np.array([1, 2, 0, 3], np.int32))


#### settings


def test_settings_default_to_a_dict_before_anything_is_written(store):
    assert isinstance(P.load_settings(WS), dict)


def test_settings_round_trip(store):
    P.save_settings(WS, {"depth": "full", "face_selection": "auto", "top_k": 5})
    back = P.load_settings(WS)
    assert back["depth"] == "full"
    assert back["face_selection"] == "auto"
    assert back["top_k"] == 5


def test_update_settings_merges_rather_than_replaces(store):
    P.save_settings(WS, {"depth": "full", "top_k": 5})
    P.update_settings(WS, {"top_k": 9})
    back = P.load_settings(WS)
    assert back["top_k"] == 9, "the change did not land"
    assert back["depth"] == "full", "an unrelated setting was dropped by an update"


def test_settings_survive_a_reload_from_disk(store):
    P.save_settings(WS, {"marker": "persisted"})
    raw = json.loads((P._settings_path(WS)).read_text())
    assert raw["marker"] == "persisted", "settings were not written to disk"


#### the document <-> session index


def test_a_linked_document_finds_its_session(store):
    P.link_doc_session(WS, "doc-a", "sess-1")
    assert P.get_doc_session(WS, "doc-a") == "sess-1"


def test_an_unlinked_document_returns_nothing_rather_than_raising(store):
    assert P.get_doc_session(WS, "never-linked") in (None, "", {})


def test_the_index_holds_every_link(store):
    P.link_doc_session(WS, "doc-a", "sess-1")
    P.link_doc_session(WS, "doc-b", "sess-2")
    m = P.doc_session_map(WS)
    assert m.get("doc-a") == "sess-1" and m.get("doc-b") == "sess-2"


def test_relinking_a_document_replaces_its_session(store):
    P.link_doc_session(WS, "doc-a", "sess-1")
    P.link_doc_session(WS, "doc-a", "sess-2")
    assert P.get_doc_session(WS, "doc-a") == "sess-2"
    assert len(P.doc_session_map(WS)) == 1, "relinking added a row instead of replacing"


#### saved complexes


def test_a_saved_complex_comes_back_the_same_shape(store):
    rex = _rex()
    P.save_document_rex(WS, "doc-a", rex)
    back = P.load_document_rex(WS, "doc-a")
    assert back is not None, "a complex that was saved did not load"
    assert (back.nV, back.nE) == (rex.nV, rex.nE)


def test_a_saved_complex_keeps_its_boundary(store):
    rex = _rex()
    P.save_document_rex(WS, "doc-a", rex)
    back = P.load_document_rex(WS, "doc-a")
    assert np.array_equal(np.sort(back.sources), np.sort(rex.sources))
    assert np.array_equal(np.sort(back.targets), np.sort(rex.targets))


def test_a_saved_complex_keeps_its_topology(store):
    """betti is the thing a persisted complex exists to answer later."""
    rex = _rex()
    before = tuple(rex.betti)
    P.save_document_rex(WS, "doc-a", rex)
    assert tuple(P.load_document_rex(WS, "doc-a").betti) == before


def test_loading_a_document_that_was_never_saved_returns_none(store):
    assert P.load_document_rex(WS, "no-such-doc") is None


def test_saved_documents_are_listed(store):
    P.save_document_rex(WS, "doc-a", _rex())
    P.save_document_rex(WS, "doc-b", _rex())
    assert set(P.list_document_bundles(WS)) >= {"doc-a", "doc-b"}


#### activity and query history


def test_activity_round_trips(store):
    P.save_activity(WS, [("art", "upload", "doc-a", 1000.0),
                         ("art", "query", "doc-a", 1001.0)])
    back = P.load_activity(WS)
    assert len(back) >= 2
    assert any(row[1] == "upload" for row in back)


def test_a_query_is_recorded_with_its_text(store):
    P.save_query(WS, "art", "what connects alpha?", "spectral", [])
    hist = P.load_query_history(WS, limit=10)
    assert hist, "a saved query did not appear in history"
    assert any("alpha" in json.dumps(row, default=str) for row in hist)


def test_query_history_honours_its_limit(store):
    for i in range(7):
        P.save_query(WS, "art", f"query number {i}", "spectral", [])
    assert len(P.load_query_history(WS, limit=3)) <= 3


def test_query_history_is_empty_before_any_query(store):
    assert P.load_query_history(WS, limit=10) == []


#### conversations


def test_a_conversation_saves_its_structure(store):
    """`save_conversation` stores the structural reading of each exchange, not the
    text: n_shared, the exchange edge count, kappa and the Hodge split. A caller
    expecting the transcript back will not find it here."""
    ex = SimpleNamespace(n_shared=4, n_exchange_edges=9, kappa_mean=0.31,
                         hodge_gradient=0.8, hodge_curl=0.15, hodge_harmonic=0.05)
    P.save_conversation(WS, "sess-1", [ex])
    path = P._convs_dir(WS) / "sess-1.json"
    assert path.exists(), "the conversation was not written"
    rows = json.loads(path.read_text())
    assert rows[0]["n_shared"] == 4
    assert rows[0]["kappa"] == pytest.approx(0.31)
    assert rows[0]["hodge"] == pytest.approx([0.8, 0.15, 0.05])


#### export and stats


def test_export_json_carries_the_documents_and_the_queries(store, tmp_path):
    P.save_document_rex(WS, "doc-a", _rex())
    P.save_query(WS, "art", "a recorded query", "spectral", [])
    out = tmp_path / "ws.json"
    P.export_workspace(WS, str(out), fmt="json")
    data = json.loads(out.read_text())
    assert "doc-a" in data["documents"]
    assert data["queries"], "export dropped the query history"


def test_export_rex_copies_the_whole_tree(store, tmp_path):
    P.save_document_rex(WS, "doc-a", _rex())
    P.save_settings(WS, {"depth": "full"})
    out = tmp_path / "exported"
    P.export_workspace(WS, str(out), fmt="rex")
    assert (out / "documents").exists()
    assert (out / "settings.json").exists()


def test_an_unknown_export_format_is_refused(store, tmp_path):
    with pytest.raises(ValueError):
        P.export_workspace(WS, str(tmp_path / "x"), fmt="parquet-ish")


def test_workspace_files_report_each_saved_document(store):
    P.save_document_rex(WS, "doc-a", _rex())
    files = P.list_workspace_files(WS)
    assert any(f["doc_id"] == "doc-a" for f in files)
    assert all(f.get("size_bytes", 0) > 0 for f in files), "a file reported no size"


def test_workspace_stats_count_what_was_stored(store):
    P.save_document_rex(WS, "doc-a", _rex())
    P.save_document_rex(WS, "doc-b", _rex())
    P.save_query(WS, "art", "one query", "spectral", [])
    P.save_activity(WS, [("art", "upload", "doc-a", 1000.0)])
    st = P.get_workspace_stats(WS)
    assert st["n_documents"] == 2
    assert st["n_queries"] >= 1
    assert "art" in st["users"]
    assert st["total_size_bytes"] > 0


def test_stats_on_an_untouched_workspace_are_zeros_not_an_error(store):
    st = P.get_workspace_stats("never-used")
    assert st["n_documents"] == 0 and st["n_queries"] == 0


#### isolation


def test_two_workspaces_do_not_see_each_other(store):
    P.save_document_rex("ws-one", "doc-a", _rex())
    P.save_settings("ws-one", {"depth": "full"})
    P.save_settings("ws-two", {"depth": "quick"})
    assert P.list_document_bundles("ws-two") == []
    assert P.load_settings("ws-two")["depth"] == "quick"
    assert P.load_settings("ws-one")["depth"] == "full"
