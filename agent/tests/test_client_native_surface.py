"""The client's side of the native contract, including the half it could not do.

A server with `REXGRAPH_FRAME_KEY` set refuses an unsigned frame, so before this the
client could not talk to a signing deployment at all: the server enforced a signature
nothing helped a caller produce. Signing belongs with the caller, so it lives here.

Both directions or neither. A client that authenticates what it sends and accepts
anything back is still talking to whoever is in the path, so a reply whose signature
does not match is refused rather than returned.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rexgraph.graph import RexGraph

KEY = "a shared deployment key"


@pytest.fixture
def rex():
    return RexGraph(sources=np.array([0, 1, 2, 2, 3, 4], dtype=np.int32),
                    targets=np.array([1, 2, 0, 3, 4, 2], dtype=np.int32))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"file://{tmp_path}/rcdb")
    from agent.rcdb import reset_default_store
    from agent.server import audit, auth
    auth.reset_auth_manager()
    audit.reset_cache()
    reset_default_store()
    yield tmp_path
    auth.reset_auth_manager()
    audit.reset_cache()
    reset_default_store()


def _client(monkeypatch, *, signed: bool):
    """A RexClient whose transport is the TestClient, so no socket is involved."""
    if signed:
        monkeypatch.setenv("REXGRAPH_FRAME_KEY", KEY)
    else:
        monkeypatch.delenv("REXGRAPH_FRAME_KEY", raising=False)
    from agent.client import RexClient
    from agent.server.app import app
    transport = TestClient(app)
    rc = RexClient("http://testserver", frame_key=KEY if signed else None)

    import httpx
    # See test_courier_remote: TestClient deprecates the timeout RexClient sets.
    def _strip(kw):
        return {k: v for k, v in kw.items() if k != "timeout"}

    monkeypatch.setattr(httpx, "post", lambda url, **kw: transport.post(
        url.replace("http://testserver", ""), **_strip(kw)))
    monkeypatch.setattr(httpx, "get", lambda url, **kw: transport.get(
        url.replace("http://testserver", ""), **_strip(kw)))
    return rc


def test_the_client_picks_the_key_up_from_the_environment(isolated, monkeypatch):
    """The same variable the server reads, so a local operator gets it for free."""
    monkeypatch.setenv("REXGRAPH_FRAME_KEY", KEY)
    from agent.client import RexClient
    assert RexClient("http://x").frame_key == KEY.encode()


def test_an_unsigned_client_cannot_reach_a_signing_server(isolated, monkeypatch, rex):
    """The gap this closes: the server enforced a signature and nothing produced one."""
    monkeypatch.setenv("REXGRAPH_FRAME_KEY", KEY)
    from agent.client import RexClient
    from agent.server.app import app
    transport = TestClient(app)
    rc = RexClient("http://testserver", frame_key=None)
    import httpx
    monkeypatch.setattr(httpx, "post", lambda url, **kw: transport.post(
        url.replace("http://testserver", ""), **kw))
    from rexgraph.protocol import encode
    r = httpx.post("http://testserver/rex/v1/verify", content=encode(rex),
                   headers=rc._headers())
    assert r.status_code == 401


def test_a_signing_client_round_trips_a_complex(isolated, monkeypatch, rex):
    rc = _client(monkeypatch, signed=True)
    out = rc.rex_verify(rex)
    assert out["fingerprint"]["nE"] == rex.nE
    assert out["fingerprint"]["chain_valid"] is True

    stored = rc.rex_store(rex, note="from the client")
    back = rc.rex_fetch(stored["record_id"])
    assert back.nE == rex.nE
    assert tuple(back.betti) == tuple(rex.betti)


def test_an_unsigned_deployment_round_trips_too(isolated, monkeypatch, rex):
    """Local use is the operator on their own machine; signing is not required there."""
    rc = _client(monkeypatch, signed=False)
    stored = rc.rex_store(rex)
    assert rc.rex_fetch(stored["record_id"]).nE == rex.nE


def test_a_reply_the_key_does_not_match_is_refused(isolated, monkeypatch, rex):
    """The half a digest cannot cover: whoever rewrote the reply recomputes the digest
    over what they substituted, and cannot produce the HMAC."""
    rc = _client(monkeypatch, signed=True)
    stored = rc.rex_store(rex)
    rc.frame_key = b"a different key"
    with pytest.raises(ValueError, match="signature"):
        rc.rex_fetch(stored["record_id"])


def test_the_client_reads_the_ceilings_before_sending(isolated, monkeypatch):
    rc = _client(monkeypatch, signed=True)
    hello = rc.rex_hello()
    assert hello["signed_frames"] is True
    assert hello["limits"]["max_cells"] > 0


def test_upload_returns_a_handle_a_tool_then_reads(isolated, monkeypatch, tmp_path):
    """The whole point of handles: the file crosses once, and what a tool receives is
    a name in the caller's namespace rather than a path in the server's."""
    rc = _client(monkeypatch, signed=True)
    p = tmp_path / "terms.obo"
    p.write_text("[Term]\nid: GO:1\nname: a\n\n[Term]\nid: GO:2\nname: b\nis_a: GO:1\n")
    handle = rc.rex_upload(str(p))["handle"]
    assert handle in {f["handle"] for f in rc.rex_files()["files"]}
    out = rc.rex_call("rexgraph_homology", files=[handle])
    assert out["result"]["betti"]
