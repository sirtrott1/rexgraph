"""The walkthrough a user actually performs, end to end.

Upload a document, get a session, read its analysis, ask about it, store it, find
it again. Each step is the real HTTP route the screen calls, so this fails when the
app stops working rather than when an internal helper changes shape.
"""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

DOC = (b"Alpha connects beta. Beta connects gamma. Gamma connects alpha and delta. "
       b"Delta connects epsilon. Epsilon connects alpha.")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "rcdb.sqlite"))
    import agent.server.persistence as pers
    from agent.rcdb import reset_default_store
    monkeypatch.setattr(pers, "_BASE_DIR", tmp_path / "ws", raising=False)
    reset_default_store()
    from agent.server.app import app
    yield TestClient(app)
    reset_default_store()


def _stream(client, name, body, mime="text/plain", **form):
    form.setdefault("depth", "quick")
    return client.post("/api/v1/pipeline/stream",
                       files=[("files", (name, io.BytesIO(body), mime))], data=form)


def _events(resp):
    kinds, payloads = [], []
    for line in resp.text.split("\n"):
        if line.startswith("event:"):
            kinds.append(line[6:].strip())
        elif line.startswith("data:"):
            try:
                payloads.append(json.loads(line[5:].strip()))
            except Exception:
                pass
    return kinds, payloads


@pytest.mark.parametrize("name,body,mime", [
    ("note.txt", DOC, "text/plain"),
    ("edges.csv", b"source,target\na,b\nb,c\nc,a\nc,d\n", "text/csv"),
    ("s.fasta", b">s1 a\nMKTAYIAKQR\n>s2 b\nMKTAYIAKQS\n>s3 c\nMKTAYIAKQT\n", "text/plain"),
])
def test_the_pipeline_runs_every_file_type_it_offers(client, name, body, mime):
    """The default screen's one job. It streams phases and finishes without an
    error event, whatever the file is."""
    r = _stream(client, name, body, mime, query="what connects to what?")
    assert r.status_code == 200, r.text
    kinds, _ = _events(r)
    assert kinds, "the stream produced no events"
    assert "error" not in kinds, r.text[:600]
    assert "done" in kinds, kinds
    assert kinds.count("phase") >= 3, kinds


def test_a_document_becomes_a_session_you_can_read_and_store(client):
    r = _stream(client, "note.txt", DOC, query="what connects to alpha?")
    assert r.status_code == 200, r.text

    sessions = client.get("/api/sessions").json()
    assert sessions, "the run created no session"
    sid = sessions[-1]["session_id"]

    a = client.get(f"/api/analysis/{sid}")
    assert a.status_code == 200, a.text
    assert a.json(), "the analysis is empty"

    chat = client.post(f"/api/chat/{sid}", json={"message": "what connects to alpha?"})
    assert chat.status_code == 200, chat.text
    assert chat.json().get("response") or chat.json().get("text"), chat.text[:200]

    put = client.post("/api/v1/db/put", json={"session_id": sid, "tags": ["journey"]})
    assert put.status_code == 200, put.text

    listed = client.get("/api/v1/db/list").json()["records"]
    assert any("journey" in (r.get("signature", {}).get("tags") or []) for r in listed), listed


def test_an_unreadable_snapshot_is_reported_not_a_crash(client, tmp_path):
    """A session whose bundle was written by an older format used to raise through
    the middleware as a 500 from seven different routes. It is one 422 now, naming
    the session and why."""
    from agent.server.app import get_store
    from agent.session import SnapshotUnreadable

    r = _stream(client, "note.txt", DOC)
    assert r.status_code == 200
    sid = client.get("/api/sessions").json()[-1]["session_id"]

    session = get_store().get(sid)
    snap = session.snapshots[session.current_step]
    # a bundle that exists but cannot be parsed, which is what an old format is
    (tmp_path / "broken.rex").mkdir()
    (tmp_path / "broken.rex" / "MANIFEST.json").write_text(
        '{"magic":"rex-bundle","version":1,"object_type":"RexGraph"}')
    snap.rex_path = str(tmp_path / "broken.rex")
    session._current_rex = None

    with pytest.raises(SnapshotUnreadable):
        session.current()

    resp = client.get(f"/api/analysis/{sid}")
    assert resp.status_code == 422, f"{resp.status_code}: {resp.text[:200]}"
    assert resp.json()["error"] == "snapshot_unreadable"
    assert sid in resp.json()["detail"]


def test_a_legacy_bundle_says_it_is_legacy(tmp_path):
    """The old manifest carried its version under `version`; the reader looks for
    `format_version` and reported None, which reads as corruption rather than age."""
    from rexgraph.io import load_rex
    b = tmp_path / "old.rex"
    b.mkdir()
    (b / "MANIFEST.json").write_text(
        '{"magic":"rex-bundle","version":1,"object_type":"RexGraph","nV":3,"nE":3}')
    with pytest.raises(ValueError, match="older version"):
        load_rex(str(b))
