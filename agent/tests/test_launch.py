"""Tests for the shared launch helper (agent.server.launch)."""

from __future__ import annotations

import sys
import types

import pytest

from agent.server import launch


def test_resolve_tls_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("REXGRAPH_TLS_CERT", raising=False)
    monkeypatch.delenv("REXGRAPH_TLS_KEY", raising=False)

    # explicit args are trusted -> https
    k, scheme = launch.resolve_tls(ssl_cert="/c.pem", ssl_key="/k.pem")
    assert scheme == "https" and k == {"ssl_certfile": "/c.pem", "ssl_keyfile": "/k.pem"}

    # nothing configured, no https flag -> http
    assert launch.resolve_tls() == ({}, "http")

    # configured certs activate only when the files exist
    cert = tmp_path / "c.pem"; key = tmp_path / "k.pem"
    cert.write_text("x"); key.write_text("y")
    monkeypatch.setenv("REXGRAPH_TLS_CERT", str(cert))
    monkeypatch.setenv("REXGRAPH_TLS_KEY", str(key))
    k, scheme = launch.resolve_tls()
    assert scheme == "https" and k["ssl_certfile"] == str(cert)


@pytest.fixture
def fake_uvicorn():
    captured = {}
    mod = types.ModuleType("uvicorn")
    mod.run = lambda app, **k: captured.update(app=app, **k)
    sys.modules["uvicorn"] = mod
    yield captured
    del sys.modules["uvicorn"]


def test_serve_passes_proxy_and_workers(fake_uvicorn, monkeypatch):
    # Binding 0.0.0.0 needs the explicit insecure override (auth is off in tests);
    # this test exercises host/worker/proxy pass-through, not the bind guard.
    monkeypatch.setenv("RCF_ALLOW_INSECURE", "1")
    launch.serve(host="0.0.0.0", port=9001, workers=4, open_browser=False)
    assert fake_uvicorn["app"] == "agent.server.app:app"
    assert fake_uvicorn["host"] == "0.0.0.0" and fake_uvicorn["port"] == 9001
    assert fake_uvicorn["proxy_headers"] is True
    assert fake_uvicorn["workers"] == 4
    assert "ssl_certfile" not in fake_uvicorn


def test_serve_secure_by_default_public_bind(fake_uvicorn, monkeypatch):
    """A fresh server binding a public host comes up authenticated, not refused."""
    monkeypatch.delenv("RCF_ALLOW_INSECURE", raising=False)
    from agent.server.auth import get_auth_manager, reset_auth_manager
    reset_auth_manager()                                  # fresh (no auth.json yet)
    launch.serve(host="0.0.0.0", port=9001, open_browser=False)
    assert fake_uvicorn["host"] == "0.0.0.0"              # bind proceeded
    assert get_auth_manager().auth_enabled is True        # secure-by-default kicked in


def test_serve_refuses_when_auth_explicitly_disabled(fake_uvicorn, monkeypatch):
    """Fail-closed: an operator who persisted auth=off cannot public-bind without override."""
    monkeypatch.delenv("RCF_ALLOW_INSECURE", raising=False)
    from agent.server import auth
    # Persist an explicit auth-disabled config, then reload so it is not "fresh".
    (auth._CONFIG_DIR / "auth.json").write_text('{"enabled": false, "tokens": []}')
    auth.reset_auth_manager()
    with pytest.raises(RuntimeError, match="auth"):
        launch.serve(host="0.0.0.0", port=9001, open_browser=False)
    assert fake_uvicorn == {}                              # uvicorn.run never reached
    # loopback is always allowed
    launch.serve(host="127.0.0.1", port=9002, open_browser=False)
    assert fake_uvicorn["host"] == "127.0.0.1"


def test_serve_https_and_reload_guard(fake_uvicorn):
    # explicit certs -> https; reload present -> workers must be dropped
    launch.serve(port=8443, ssl_cert="/c.pem", ssl_key="/k.pem",
                 reload=True, workers=4, open_browser=False)
    assert fake_uvicorn["ssl_certfile"] == "/c.pem"
    assert fake_uvicorn["reload"] is True
    assert "workers" not in fake_uvicorn
