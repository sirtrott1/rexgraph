"""Tests for the DB-URI allow-list (dbguard) and the error-response sanitizer."""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from agent.server.dbguard import check_db_uri
from agent.server.security import add_error_sanitizer


# dbguard: opt-in DB-URI allow-list

def test_dbguard_off_by_default_allows_anything(monkeypatch):
    for k in ("REXGRAPH_DB_SAFE", "REXGRAPH_DB_BLOCK_LOCAL",
              "REXGRAPH_ALLOWED_DB_SCHEMES", "REXGRAPH_ALLOWED_DB_HOSTS"):
        monkeypatch.delenv(k, raising=False)
    # No policy set -> everything passes, including local files and private hosts.
    check_db_uri("sqlite:////etc/passwd")
    check_db_uri("postgresql://user:pw@10.0.0.5/db")
    check_db_uri("edgelist")            # bare scheme, no '://'


def test_dbguard_safe_blocks_file_and_private(monkeypatch):
    monkeypatch.setenv("REXGRAPH_DB_SAFE", "1")
    monkeypatch.delenv("REXGRAPH_ALLOWED_DB_SCHEMES", raising=False)
    monkeypatch.delenv("REXGRAPH_ALLOWED_DB_HOSTS", raising=False)
    with pytest.raises(HTTPException):
        check_db_uri("sqlite:////etc/passwd")           # file scheme
    with pytest.raises(HTTPException):
        check_db_uri("postgresql://localhost/db")        # loopback host
    with pytest.raises(HTTPException):
        check_db_uri("mysql://user:pw@192.168.1.9/db")   # private host
    # a public DB server is allowed
    check_db_uri("postgresql://user:pw@db.example.com:5432/prod")


def test_dbguard_scheme_and_host_allowlists(monkeypatch):
    monkeypatch.delenv("REXGRAPH_DB_SAFE", raising=False)
    monkeypatch.setenv("REXGRAPH_ALLOWED_DB_SCHEMES", "postgresql,mysql")
    with pytest.raises(HTTPException):
        check_db_uri("mongodb://db.example.com/x")       # scheme not allowed
    check_db_uri("postgresql+psycopg2://db.example.com/x")  # +driver stripped, ok

    monkeypatch.delenv("REXGRAPH_ALLOWED_DB_SCHEMES", raising=False)
    monkeypatch.setenv("REXGRAPH_ALLOWED_DB_HOSTS", "db1.corp,db2.corp")
    with pytest.raises(HTTPException):
        check_db_uri("postgresql://evil.example.com/x")  # host not allowed
    check_db_uri("postgresql://db1.corp/x")


# error sanitizer

def _sanitized_app(monkeypatch):
    monkeypatch.delenv("REXGRAPH_DEBUG_ERRORS", raising=False)
    app = FastAPI()
    add_error_sanitizer(app)

    @app.get("/boom500")
    def boom500():
        raise HTTPException(500, "postgres://user:secretpw@internal-host/db")

    @app.get("/boom400")
    def boom400():
        raise HTTPException(400, "you passed a bad value")

    @app.get("/crash")
    def crash():
        raise RuntimeError("internal secret token abc123")

    return TestClient(app, raise_server_exceptions=False)


def test_sanitizer_hides_5xx_detail(monkeypatch):
    client = _sanitized_app(monkeypatch)
    r = client.get("/boom500")
    assert r.status_code == 500
    body = r.json()
    assert body["detail"] == "Internal server error"
    assert "error_id" in body and len(body["error_id"]) >= 8
    assert "secretpw" not in r.text and "internal-host" not in r.text


def test_sanitizer_passes_4xx_detail(monkeypatch):
    client = _sanitized_app(monkeypatch)
    r = client.get("/boom400")
    assert r.status_code == 400
    assert r.json()["detail"] == "you passed a bad value"


def test_sanitizer_catches_unhandled(monkeypatch):
    client = _sanitized_app(monkeypatch)
    r = client.get("/crash")
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error"
    assert "abc123" not in r.text and "RuntimeError" not in r.text


def test_debug_errors_restores_detail(monkeypatch):
    monkeypatch.setenv("REXGRAPH_DEBUG_ERRORS", "1")
    app = FastAPI()
    add_error_sanitizer(app)

    @app.get("/boom500")
    def boom500():
        raise HTTPException(500, "verbose detail here")

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/boom500")
    assert r.status_code == 500
    assert r.json()["detail"] == "verbose detail here"


# rate limiting

def test_rate_limiter_enforces_and_disables(monkeypatch):
    from agent.server.security import setup_rate_limiter

    # disabled -> no limiter, unlimited requests
    monkeypatch.setenv("RCF_RATE_LIMIT", "0")
    off_app = FastAPI()
    assert setup_rate_limiter(off_app) is None

    @off_app.get("/ping")
    def ping_off():
        return {"ok": True}

    off = TestClient(off_app)
    assert all(off.get("/ping").status_code == 200 for _ in range(20))

    # enabled with a low limit -> the (N+1)th request in the window is 429
    monkeypatch.setenv("RCF_RATE_LIMIT", "3/minute")
    on_app = FastAPI()
    assert setup_rate_limiter(on_app) is not None

    @on_app.get("/ping")
    def ping_on():
        return {"ok": True}

    on = TestClient(on_app)
    codes = [on.get("/ping").status_code for _ in range(6)]
    assert codes[:3] == [200, 200, 200]
    assert 429 in codes[3:]


def test_rate_limiter_tiers_and_exemptions(monkeypatch):
    from agent.server.security import setup_rate_limiter
    monkeypatch.setenv("RCF_RATE_LIMIT", "100/minute")      # general: generous
    monkeypatch.setenv("RCF_RATE_LIMIT_HEAVY", "2/minute")  # heavy: tight
    app = FastAPI()
    setup_rate_limiter(app)

    @app.get("/api/v1/pipeline/go")   # heavy tier
    def heavy():
        return {"ok": True}

    @app.get("/api/misc")             # general tier
    def general():
        return {"ok": True}

    @app.get("/api/health")           # exempt
    def health():
        return {"ok": True}

    c = TestClient(app)
    heavy_codes = [c.get("/api/v1/pipeline/go").status_code for _ in range(4)]
    assert heavy_codes[:2] == [200, 200] and 429 in heavy_codes[2:]
    # general tier has its own (generous) bucket, unaffected by heavy exhaustion
    assert c.get("/api/misc").status_code == 200
    # health is never rate limited
    assert all(c.get("/api/health").status_code == 200 for _ in range(5))


def test_appjsx_served_as_executable_javascript():
    """Regression: app.jsx is loaded via <script src>. Under our
    X-Content-Type-Options: nosniff header the browser refuses to execute a script
    unless its MIME type is JavaScript - so StaticFiles must NOT serve it as
    application/octet-stream, or the whole UI silently fails to render."""
    from agent.server.app import app
    c = TestClient(app)
    r = c.get("/static/app.jsx")
    if r.status_code == 404:
        import pytest as _pytest
        _pytest.skip("frontend/ not present (non-editable install)")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
    assert r.headers.get("x-content-type-options") == "nosniff"


def test_dbguard_resolves_hostname_to_private(monkeypatch):
    import agent.server.dbguard as dbg
    monkeypatch.setenv("REXGRAPH_DB_SAFE", "1")
    monkeypatch.delenv("REXGRAPH_ALLOWED_DB_SCHEMES", raising=False)
    monkeypatch.delenv("REXGRAPH_ALLOWED_DB_HOSTS", raising=False)

    # hostname resolving to a private IP -> blocked (SSRF via DNS)
    monkeypatch.setattr(dbg.socket, "getaddrinfo",
                        lambda host, *a, **k: [(2, 1, 6, "", ("10.1.2.3", 0))])
    with pytest.raises(HTTPException):
        dbg.check_db_uri("postgresql://sneaky.example.com/db")

    # hostname resolving to a public IP -> allowed
    monkeypatch.setattr(dbg.socket, "getaddrinfo",
                        lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    dbg.check_db_uri("postgresql://legit.example.com/db")

    # unresolvable hostname falls through (the real connection would just fail)
    def _boom(*a, **k):
        raise OSError("name resolution failed")
    monkeypatch.setattr(dbg.socket, "getaddrinfo", _boom)
    dbg.check_db_uri("postgresql://does-not-exist.invalid/db")
