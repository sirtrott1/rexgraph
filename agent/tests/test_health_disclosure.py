"""The health probe reports liveness to anyone and the tenant roster to nobody.

`/api/health` declares no authorization and returned `mgr.list_workspaces()`, so an
anonymous caller got every workspace name on the server. That is the value `X-Workspace`
takes, so it is the namespace an attacker aims at, and the names usually carry customer
identity on their own.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"file://{tmp_path}/rcdb")
    from agent.server import audit, auth
    auth.reset_auth_manager(); audit.reset_cache()
    from agent.server.app import app
    yield TestClient(app), auth.get_auth_manager()
    auth.reset_auth_manager(); audit.reset_cache()


def test_an_anonymous_probe_still_reports_liveness(server):
    client, mgr = server
    mgr.enable_auth(); mgr.bootstrap_admin()
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["auth_enabled"] is True


def test_an_anonymous_probe_does_not_get_the_roster(server):
    client, mgr = server
    mgr.enable_auth(); mgr.bootstrap_admin()
    mgr.create_token("bob", ["beta"], role="user")
    assert "workspaces" not in client.get("/api/health").json()


def test_an_unverifiable_token_does_not_get_the_roster(server):
    client, mgr = server
    mgr.enable_auth(); mgr.bootstrap_admin()
    mgr.create_token("bob", ["beta"], role="user")
    body = client.get("/api/health", headers={"Authorization": "Bearer nope"}).json()
    assert "workspaces" not in body


def test_a_tenant_sees_only_its_own_workspaces(server):
    client, mgr = server
    mgr.enable_auth(); mgr.bootstrap_admin()
    bob = mgr.create_token("bob", ["beta"], role="user")
    mgr.create_token("alice", ["alpha"], role="user")
    body = client.get("/api/health",
                      headers={"Authorization": f"Bearer {bob}"}).json()
    assert body["workspaces"] == ["beta"], body
    assert "alpha" not in body["workspaces"]


def test_with_auth_off_the_roster_is_unchanged(server):
    client, mgr = server
    body = client.get("/api/health").json()
    assert body["auth_enabled"] is False
    assert body["workspaces"], "the single-operator case still reports its workspaces"


def test_an_admin_of_one_workspace_does_not_get_the_whole_roster(server):
    """`token.role` is the legacy scalar view and _resync sets it to admin when the token
    is admin in ANY workspace, so admin of alpha listed every tenant on the instance."""
    client, mgr = server
    mgr.enable_auth(); mgr.bootstrap_admin()
    alice = mgr.create_token("alice", ["alpha"], role="admin")
    mgr.create_token("bob", ["beta"], role="user")
    mgr.get_workspace("alpha"); mgr.get_workspace("beta")
    body = client.get("/api/v1/admin/workspaces",
                      headers={"Authorization": f"Bearer {alice}"}).json()
    assert body["workspaces"] == ["alpha"], body
    assert "beta" not in body["workspaces"]


def test_an_instance_admin_still_gets_the_whole_roster(server):
    client, mgr = server
    mgr.enable_auth()
    admin = mgr.bootstrap_admin()
    mgr.create_token("bob", ["beta"], role="user")
    mgr.get_workspace("beta")
    body = client.get("/api/v1/admin/workspaces",
                      headers={"Authorization": f"Bearer {admin}"}).json()
    assert "beta" in body["workspaces"], body
