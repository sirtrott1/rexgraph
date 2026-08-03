"""Disabling auth is gated: admin token + direct loopback + a step-up passphrase.

A leaked or cached API token alone must not be enough to turn authentication off,
and the toggle must not be reachable from any non-local client. Also covers the
secure-by-default bootstrap helper.
"""
import pytest
from agent.server import auth
from agent.server.auth import get_auth_manager
from fastapi.testclient import TestClient

# manager-level

def test_passphrase_set_verify_roundtrip():
    mgr = get_auth_manager()
    assert mgr.has_disable_passphrase is False
    assert mgr.verify_disable_passphrase("anything") is False   # unset -> never matches
    mgr.set_disable_passphrase("correct horse battery")
    assert mgr.has_disable_passphrase is True
    assert mgr.verify_disable_passphrase("correct horse battery") is True
    assert mgr.verify_disable_passphrase("wrong") is False


def test_passphrase_min_length_enforced():
    mgr = get_auth_manager()
    with pytest.raises(ValueError):
        mgr.set_disable_passphrase("short")


def test_bootstrap_admin_mints_first_token():
    mgr = get_auth_manager()
    mgr.enable_auth()
    raw = mgr.bootstrap_admin()
    assert raw and mgr.verify(raw).role == "admin"
    assert mgr.bootstrap_admin() is None            # idempotent once a token exists


def test_bootstrap_admin_uses_supplied_token():
    mgr = get_auth_manager()
    mgr.enable_auth()
    assert mgr.bootstrap_admin("known-admin-token") == "known-admin-token"
    assert mgr.verify("known-admin-token").role == "admin"


def test_is_fresh_reflects_config(tmp_path, monkeypatch):
    assert get_auth_manager().is_fresh is True       # temp config dir, no auth.json yet
    (auth._CONFIG_DIR / "auth.json").write_text('{"enabled": true, "tokens": []}')
    auth.reset_auth_manager()
    assert get_auth_manager().is_fresh is False


# route-level

def _setup(*, local=True, passphrase="supersecret123", set_pass=True):
    from agent.server.app import app
    mgr = get_auth_manager()
    mgr.enable_auth()
    if set_pass:
        mgr.set_disable_passphrase(passphrase)
    tok = mgr.create_token("admin", ["default"], role="admin")
    host = "127.0.0.1" if local else "203.0.113.9"
    client = TestClient(app, client=(host, 40000))
    return client, tok


def _hdr(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_disable_requires_passphrase():
    client, tok = _setup(set_pass=False)
    r = client.post("/api/v1/admin/auth/disable", json={}, headers=_hdr(tok))
    assert r.status_code == 403 and "passphrase" in r.json()["detail"].lower()
    assert get_auth_manager().auth_enabled is True   # unchanged


def test_disable_rejects_wrong_passphrase():
    client, tok = _setup()
    r = client.post("/api/v1/admin/auth/disable",
                    json={"passphrase": "nope"}, headers=_hdr(tok))
    assert r.status_code == 403
    assert get_auth_manager().auth_enabled is True


def test_disable_rejected_from_non_local_client():
    client, tok = _setup(local=False)
    r = client.post("/api/v1/admin/auth/disable",
                    json={"passphrase": "supersecret123"}, headers=_hdr(tok))
    assert r.status_code == 403 and "host" in r.json()["detail"].lower()
    assert get_auth_manager().auth_enabled is True


def test_disable_rejected_when_proxied_even_from_loopback():
    client, tok = _setup()
    r = client.post("/api/v1/admin/auth/disable",
                    json={"passphrase": "supersecret123"},
                    headers={**_hdr(tok), "X-Forwarded-For": "203.0.113.9"})
    assert r.status_code == 403                       # forwarding header => treated as remote
    assert get_auth_manager().auth_enabled is True


def test_disable_succeeds_local_with_passphrase():
    client, tok = _setup()
    r = client.post("/api/v1/admin/auth/disable",
                    json={"passphrase": "supersecret123"}, headers=_hdr(tok))
    assert r.status_code == 200 and r.json()["auth_enabled"] is False
    assert get_auth_manager().auth_enabled is False


def test_disable_rejected_without_admin_token():
    client, _ = _setup()
    r = client.post("/api/v1/admin/auth/disable", json={"passphrase": "supersecret123"})
    assert r.status_code == 401                       # no bearer token at all
    assert get_auth_manager().auth_enabled is True


def test_enable_rejected_from_non_local_client():
    from agent.server.app import app
    # auth starts off; a remote caller is synthetic-admin but still host-gated
    client = TestClient(app, client=("203.0.113.9", 40000))
    r = client.post("/api/v1/admin/auth/enable", json={})
    assert r.status_code == 403
