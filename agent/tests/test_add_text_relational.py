"""Text enters the system the way every other document does: as a complex.

`add-text` used to be a JSON convenience that only touched the corpus, so text was
the one input with no .rex behind it. It now builds the complex and writes the
bundle, and the bundle carries the source text, so a text document is one file.
"""
from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

TEXT = ("Alpha connects beta. Beta connects gamma. "
        "Gamma connects alpha and delta. Delta connects alpha.")


@pytest.fixture
def client(tmp_path, monkeypatch):
    import agent.server.persistence as pers
    monkeypatch.setattr(pers, "_BASE_DIR", tmp_path / "ws", raising=False)
    from agent.server.app import app
    return TestClient(app)


def test_add_text_builds_a_complex_and_writes_a_rex(client):
    r = client.post("/api/v1/corpus/add-text", json={"text": TEXT, "doc_id": "note1"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["doc_id"] == "note1"
    assert d["nV"] > 0 and d["nE"] > 0, d
    assert d["vertex_labels"], "the complex has no labelled vertices"
    assert pathlib.Path(d["path"]).exists(), d["path"]
    assert d["path"].endswith(".rex")


def test_the_bundle_carries_the_text_so_there_is_no_sidecar(client):
    client.post("/api/v1/corpus/add-text", json={"text": TEXT, "doc_id": "note2"})
    from agent.server.persistence import _docs_dir, load_document_rex
    rex = load_document_rex("default", "note2")
    assert rex is not None
    meta = getattr(rex, "_agent_meta", None) or {}
    assert meta.get("input_type") == "text", meta.get("input_type")
    assert TEXT[:20] in (meta.get("source_text") or ""), "the text did not survive"
    assert not (_docs_dir("default") / "note2.txt").exists(), "wrote a sidecar"


def test_persist_false_adds_without_writing(client):
    r = client.post("/api/v1/corpus/add-text",
                    json={"text": TEXT, "doc_id": "note3", "persist": False})
    assert r.status_code == 200, r.text
    assert "path" not in r.json()
    from agent.server.persistence import load_document_rex
    assert load_document_rex("default", "note3") is None


def test_empty_text_is_rejected(client):
    assert client.post("/api/v1/corpus/add-text", json={}).status_code == 400
    assert client.post("/api/v1/corpus/add-text",
                       json={"text": "   "}).status_code == 400


def test_prose_is_read_as_prose_not_as_an_edge_list(tmp_path):
    """A .txt fell through the delimited-file classifier to `edge_csv`, so prose was
    recorded as an edge list. Positive evidence is now required to call it a table."""
    from agent.auto import auto_rex, detect_input_type
    cases = {
        "one line": "Alpha connects beta. Beta connects gamma.",
        "many lines": "The quick brown fox jumps.\nIt was the best of times.\nAll alike.",
        "commas in prose": ("First, we consider the problem. Then, we solve it, carefully.\n"
                            "After that, we write it up, and we publish."),
    }
    for name, body in cases.items():
        p = tmp_path / (name.replace(" ", "_") + ".txt")
        p.write_text(body)
        assert detect_input_type(str(p)) == "text", name

    tabular = tmp_path / "edges.txt"
    tabular.write_text("source,target\na,b\nb,c")
    assert detect_input_type(str(tabular)) == "edge_csv"

    p = tmp_path / "prose.txt"
    p.write_text(cases["one line"])
    rex = auto_rex(str(p))
    assert (rex._agent_meta or {}).get("input_type") == "text"


def test_csv_classification_is_unchanged(tmp_path):
    """The .txt rule must not move where the extension already says table."""
    from agent.auto import detect_input_type
    e = tmp_path / "e.csv"
    e.write_text("source,target\na,b\nb,c")
    assert detect_input_type(str(e)) == "edge_csv"
    f = tmp_path / "f.csv"
    f.write_text("a,b,c,d,e,f\n" + "\n".join("1,2,3,4,5,6" for _ in range(4)))
    assert detect_input_type(str(f)) == "feature_csv"
