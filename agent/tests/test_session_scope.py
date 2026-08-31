"""Sessions belong to a workspace, the same way records do.

`SessionStore` had no workspace anywhere in it and the session routes declare no
authorization, so any tenant listed, read and DELETED any other tenant's sessions. The
export route resolved a workspace and then never consulted it for the lookup. This is
the hole `ScopedStore` closes for the record store, left open next door.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def tenants(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"file://{tmp_path}/rcdb")
    monkeypatch.setenv("REXGRAPH_SESSION_DIR", str(tmp_path / "sessions"))
    from agent.rcdb import reset_default_store
    from agent.server import audit, auth
    auth.reset_auth_manager(); audit.reset_cache(); reset_default_store()
    from agent.server import app as appmod
    appmod._store = None if hasattr(appmod, "_store") else None
    from agent.server.app import app
    mgr = auth.get_auth_manager(); mgr.enable_auth()
    admin = mgr.bootstrap_admin()
    bob = mgr.create_token("bob", ["beta"], role="user")
    yield (TestClient(app), {"Authorization": f"Bearer {admin}"},
           {"Authorization": f"Bearer {bob}", "X-Workspace": "beta"})
    auth.reset_auth_manager(); audit.reset_cache(); reset_default_store()


def _make_session(client, headers):
    r = client.post("/api/upload", headers=headers,
                    files={"file": ("d.csv", b"a,b\n1,2\n3,4\n", "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def test_a_tenant_does_not_see_another_tenants_sessions(tenants):
    client, ah, bh = tenants
    bob_sid = _make_session(client, bh)
    seen = [s["session_id"] for s in client.get("/api/sessions", headers=ah).json()]
    assert bob_sid not in seen, "alice listed bob's session"


def test_another_tenants_session_reads_as_absent(tenants):
    """404 rather than 403: saying it exists but is not yours turns a guessable id
    into a way to enumerate what other tenants hold."""
    client, ah, bh = tenants
    bob_sid = _make_session(client, bh)
    assert client.get(f"/api/sessions/{bob_sid}", headers=ah).status_code == 404
    assert client.get(f"/api/sessions/{bob_sid}", headers=bh).status_code == 200


def test_a_tenant_cannot_delete_another_tenants_session(tenants):
    """Deletion is the one operation an owner cannot undo."""
    client, ah, bh = tenants
    bob_sid = _make_session(client, bh)
    client.delete(f"/api/sessions/{bob_sid}", headers=ah)
    assert client.get(f"/api/sessions/{bob_sid}", headers=bh).status_code == 200, \
        "bob's session was destroyed by another tenant"


def test_export_consults_the_workspace_it_resolves(tenants):
    """It declared Depends(require_workspace), then looked the session up without it."""
    client, ah, bh = tenants
    bob_sid = _make_session(client, bh)
    assert client.get(f"/api/v1/export/session/{bob_sid}", headers=ah).status_code == 404


def test_an_owner_keeps_their_own(tenants):
    """The half worth checking as hard as the leak: a filter that hides everything
    from everyone also passes an isolation test."""
    client, ah, bh = tenants
    bob_sid = _make_session(client, bh)
    assert bob_sid in [s["session_id"] for s in client.get("/api/sessions", headers=bh).json()]
    assert client.get(f"/api/sessions/{bob_sid}", headers=bh).status_code == 200
    assert client.get(f"/api/v1/export/session/{bob_sid}", headers=bh).status_code == 200


def test_nothing_is_filtered_outside_a_request(tmp_path, monkeypatch):
    """The CLI and the suite are not serving a request and are not being scoped."""
    monkeypatch.setenv("REXGRAPH_SESSION_DIR", str(tmp_path / "s"))
    from agent.server.state import SessionStore
    st = SessionStore()
    s = st.create(name="local")
    assert st.get(s.session_id) is not None
    assert "workspace" not in s._metadata
