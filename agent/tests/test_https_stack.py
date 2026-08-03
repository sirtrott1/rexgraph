"""Tests for the HTTPS stack: HSTS hardening + the built-in TLS adapters."""

from __future__ import annotations

import pytest
from agent.server import security
from agent.server.app import app
from fastapi.testclient import TestClient


def test_hsts_only_on_tls():
    https = TestClient(app, base_url="https://testserver")
    http = TestClient(app, base_url="http://testserver")
    assert https.get("/api/health").headers.get("strict-transport-security")
    # no HSTS leaked onto plain-HTTP responses
    assert http.get("/api/health").headers.get("strict-transport-security") is None


def test_https_config_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("REXGRAPH_TLS_CERT", raising=False)
    monkeypatch.delenv("REXGRAPH_TLS_KEY", raising=False)
    assert security.get_https_config() == {}          # nothing configured yet

    res = security.generate_self_signed_cert()
    if "error" in res:
        pytest.skip("cryptography not installed")
    cfg = security.get_https_config()                 # config-dir certs discovered
    assert cfg.get("ssl_certfile") and cfg.get("ssl_keyfile")

    # explicit env certs take precedence and are returned verbatim
    monkeypatch.setenv("REXGRAPH_TLS_CERT", res["cert_path"])
    monkeypatch.setenv("REXGRAPH_TLS_KEY", res["key_path"])
    assert security.get_https_config() == {
        "ssl_certfile": res["cert_path"], "ssl_keyfile": res["key_path"]}
