"""Saved database connections belong to a workspace.

`routes/dbmanager.py` declares no authorization and `_secrets()` is a process-wide
singleton, so every tenant shared one connection store: a name collision overwrote
someone else's credentials, a DELETE removed them, and `POST /test` resolved another
tenant's saved name and dialled their host with their full credentialed URI.

The header is the other half. `identity_and_workspace` took `X-Workspace` verbatim, so
on any route that does not declare `require_workspace` a caller could name a workspace
their token does not grant and be scoped to it. Scoping the store without that fix only
scopes to whatever the caller asserted.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

ALPHA_URI = "postgresql://alice:alicepw@alpha.example.com:5432/app"
BETA_URI = "postgresql://bob:bobpw@beta.example.com:5432/app"


@pytest.fixture
def tenants(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_SECRETS_URI", f"file://{tmp_path}/connections.json")
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"file://{tmp_path}/rcdb")
    from agent.server import audit, auth, scope
    scope.reset_secret_store()
    auth.reset_auth_manager(); audit.reset_cache()
    from agent.server.app import app
    mgr = auth.get_auth_manager(); mgr.enable_auth()
    mgr.bootstrap_admin()
    alice = mgr.create_token("alice", ["alpha"], role="user")
    bob = mgr.create_token("bob", ["beta"], role="user")
    ah = {"Authorization": f"Bearer {alice}", "X-Workspace": "alpha"}
    bh = {"Authorization": f"Bearer {bob}", "X-Workspace": "beta"}
    yield TestClient(app), ah, bh
    scope.reset_secret_store()
    auth.reset_auth_manager(); audit.reset_cache()


def _save(client, headers, name, uri):
    r = client.post("/api/v1/dbmanager/connections", headers=headers,
                    json={"name": name, "uri": uri})
    assert r.status_code == 200, r.text
    return r


def _names(client, headers):
    r = client.get("/api/v1/dbmanager/connections", headers=headers)
    assert r.status_code == 200, r.text
    return [c.get("name") for c in r.json()["connections"]]


def test_a_tenant_does_not_see_another_tenants_connections(tenants):
    client, ah, bh = tenants
    _save(client, ah, "prod", ALPHA_URI)
    assert "prod" not in _names(client, bh), "bob listed alice's connection"


def test_a_tenant_cannot_resolve_another_tenants_connection(tenants):
    client, ah, bh = tenants
    _save(client, ah, "prod", ALPHA_URI)
    r = client.post("/api/v1/dbmanager/test", headers=bh, json={"name": "prod"})
    assert r.status_code == 404, f"bob resolved alice's credentials: {r.text}"


def test_a_forged_workspace_header_does_not_bind_another_workspace(tenants):
    client, ah, bh = tenants
    _save(client, ah, "prod", ALPHA_URI)
    forged = dict(bh); forged["X-Workspace"] = "alpha"
    assert "prod" not in _names(client, forged), "bob named a workspace he has no role in"


def test_a_colliding_name_does_not_overwrite_the_other_tenant(tenants):
    client, ah, bh = tenants
    _save(client, ah, "prod", ALPHA_URI)
    _save(client, bh, "prod", BETA_URI)
    r = client.post("/api/v1/dbmanager/test", headers=ah, json={"name": "prod"})
    assert "alpha.example.com" in r.json().get("uri", ""), "bob overwrote alice's connection"


def test_deleting_another_tenants_connection_is_a_noop(tenants):
    client, ah, bh = tenants
    _save(client, ah, "prod", ALPHA_URI)
    client.delete("/api/v1/dbmanager/connections/prod", headers=bh)
    assert "prod" in _names(client, ah), "bob deleted alice's connection"


def test_outside_a_request_nothing_is_filtered(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_SECRETS_URI", f"file://{tmp_path}/c.json")
    from agent.secrets import open_secret_store
    store = open_secret_store()
    store.put("cli", ALPHA_URI, "sql")
    assert store.get("cli") == ALPHA_URI
    assert [c.get("name") for c in store.list()] == ["cli"]


def test_connectors_resolves_through_the_same_scoped_view(tenants):
    """routes/connectors reached open_secret_store() directly, so read, validate and
    ingest each dialled another tenant's database with their full credentialed URI."""
    client, ah, bh = tenants
    _save(client, ah, "prod", ALPHA_URI)
    r = client.post("/api/v1/connectors/validate", headers=bh, json={"name": "prod"})
    assert r.status_code == 404, f"bob resolved alice's connection: {r.text}"
