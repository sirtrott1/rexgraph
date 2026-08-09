"""The Python SDK, against the real server.

`client.py` is the public surface a notebook imports, and it was 0% executed: 117
statements, 19 requests, and nothing checking that any of them names a route the
server actually serves. An SDK method whose path drifted from its route fails only
for the user, at import-time-plus-one-call.

The client talks httpx, so the transport is redirected onto the ASGI TestClient here
rather than binding a port. That keeps the real client code in the path: its URLs, its
headers, its parameter names and its `raise_for_status`.

A 404 or a 405 is the failure this file exists to catch. A 4xx that is the request's
own fault (no such session, no model deployed) is a route that exists and answered.
"""
from __future__ import annotations

import httpx
import pytest
from agent.client import RexClient
from fastapi.testclient import TestClient

DOC = ("Alpha connects beta. Beta connects gamma. Gamma connects alpha. "
       "Delta relates to alpha. Epsilon relates to delta. Alpha relates to epsilon.")


@pytest.fixture
def rc(tmp_path, monkeypatch):
    """A RexClient whose httpx calls land on the in-process app."""
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "r.sqlite"))
    from agent.rcdb import reset_default_store
    reset_default_store()
    from agent.server.app import app

    tc = TestClient(app, raise_server_exceptions=False)
    monkeypatch.setattr(httpx, "get", lambda url, **kw: tc.get(url, **kw))
    monkeypatch.setattr(httpx, "post", lambda url, **kw: tc.post(url, **kw))
    yield RexClient(url="http://testserver")
    reset_default_store()


def _call(fn, *a, **kw):
    """Run an SDK call, returning (ok, status). A routing failure is 404/405."""
    try:
        fn(*a, **kw)
        return True, 200
    except httpx.HTTPStatusError as e:
        return False, e.response.status_code
    except Exception:                            # noqa: BLE001 - not a routing fault
        return True, 0


#### the routing contract


def test_health_answers(rc):
    assert rc.health(), "the server reported no health"


def test_status_answers(rc):
    assert isinstance(rc.status(), dict)


def test_the_workspace_header_is_sent(rc):
    assert rc._headers()["X-Workspace"] == "default"


def test_an_api_key_becomes_a_bearer_token():
    h = RexClient(url="http://x", api_key="secret")._headers()
    assert h["Authorization"] == "Bearer secret"


def test_a_trailing_slash_is_stripped_from_the_url():
    assert RexClient(url="http://x:8000/").url == "http://x:8000"


#: every read-only SDK method, called with arguments that are valid on their face.
#: The assertion is about routing, not about the answer.
READ_CALLS = [
    ("health", lambda c: c.health()),
    ("status", lambda c: c.status()),
    ("corpus_summary", lambda c: c.corpus_summary()),
    ("corpus_temporal", lambda c: c.corpus_temporal()),
    ("models_list", lambda c: c.models_list()),
    ("list_workspaces", lambda c: c.list_workspaces()),
    ("workspace_activity", lambda c: c.workspace_activity()),
    ("export_workspace", lambda c: c.export_workspace()),
    ("export_queries", lambda c: c.export_queries(limit=5)),
]


@pytest.mark.parametrize("name,call", READ_CALLS, ids=[n for n, _ in READ_CALLS])
def test_a_read_method_names_a_route_that_exists(rc, name, call):
    ok, code = _call(call, rc)
    assert code not in (404, 405), \
        f"RexClient.{name}() got HTTP {code}: its path does not match any route"


#: the write methods, same contract
WRITE_CALLS = [
    ("corpus_add_text", lambda c: c.corpus_add_text(DOC, doc_id="d1")),
    ("corpus_build", lambda c: c.corpus_build(depth="quick")),
    ("corpus_query", lambda c: c.corpus_query("alpha", mode="spectral", top_k=3)),
    ("corpus_bridge", lambda c: c.corpus_bridge(0, 1)),
    ("corpus_reset", lambda c: c.corpus_reset()),
    ("models_pull", lambda c: c.models_pull("no-such-model")),
    ("models_stop", lambda c: c.models_stop()),
    ("generate", lambda c: c.generate("hello")),
    ("create_token", lambda c: c.create_token("art")),
]


@pytest.mark.parametrize("name,call", WRITE_CALLS, ids=[n for n, _ in WRITE_CALLS])
def test_a_write_method_names_a_route_that_exists(rc, name, call):
    ok, code = _call(call, rc)
    assert code not in (404, 405), \
        f"RexClient.{name}() got HTTP {code}: its path does not match any route"


def test_upload_and_analysis_name_routes_that_exist(rc, tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text(DOC)
    ok, code = _call(rc.upload, str(p))
    assert code not in (404, 405), f"RexClient.upload() got HTTP {code}"


def test_corpus_add_file_names_a_route_that_exists(rc, tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text(DOC)
    ok, code = _call(rc.corpus_add_file, str(p))
    assert code not in (404, 405), f"RexClient.corpus_add_file() got HTTP {code}"


def test_session_scoped_methods_name_routes_that_exist(rc):
    """With a session id that does not exist, the route still has to be found. A 404
    from the handler and a 404 from the router look the same over HTTP, so this only
    fails on a method that is not routed at all."""
    for name, call in (("analysis", lambda c: c.analysis("no-such-session")),
                       ("chat", lambda c: c.chat("no-such-session", "hi")),
                       ("export_session", lambda c: c.export_session("no-such"))):
        ok, code = _call(call, rc)
        assert code != 405, f"RexClient.{name}() got HTTP 405: wrong method for its route"


#### the corpus round-trip, through the SDK only


def test_text_added_through_the_sdk_can_be_queried_back(rc):
    """add -> build -> query, which is the documented notebook flow."""
    rc.corpus_reset()
    rc.corpus_add_text(DOC, doc_id="supply")
    built = rc.corpus_build(depth="quick")
    assert built, "corpus_build returned nothing"
    hits = rc.corpus_query("alpha", mode="spectral", top_k=3)
    assert hits, "a query over a built corpus returned nothing"


def test_the_summary_reflects_what_was_added(rc):
    rc.corpus_reset()
    rc.corpus_add_text(DOC, doc_id="supply")
    rc.corpus_build(depth="quick")
    s = rc.corpus_summary()
    assert "supply" in repr(s) or (s.get("n_documents") or 0) >= 1, \
        f"the corpus summary does not mention the document that was added: {s}"


def test_reset_empties_the_corpus(rc):
    rc.corpus_add_text(DOC, doc_id="supply")
    rc.corpus_build(depth="quick")
    rc.corpus_reset()
    s = rc.corpus_summary()
    assert (s.get("n_documents") or 0) == 0, f"reset left {s} behind"


def test_repr_names_the_server(rc):
    assert "testserver" in repr(rc)
