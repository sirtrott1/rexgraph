"""
agent.server.security - security utilities for production deployment.

    HTTPS configuration
    Temp file lifecycle management
    Request/response sanitization
    Rate limiting setup
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Temp file safety

@contextmanager
def secure_tempfile(suffix=".bin", prefix="rexgraph_"):
    """Context manager that guarantees temp file deletion.

    Usage:
        with secure_tempfile(suffix=".pdf") as path:
            save_upload(path)
            process(path)
        # file is deleted here, even on exception
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def cleanup_stale_tempfiles(max_age_hours=24):
    """Remove old rexgraph temp files from /tmp."""
    import time
    tmp = Path(tempfile.gettempdir())
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for f in tmp.glob("rexgraph_*"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    return removed


# Response sanitization

def sanitize_model_response(text: str) -> str:
    """Strip potentially dangerous content from model responses.

    Removes:
    - Embedded scripts/HTML tags
    - Data URIs
    - Attempts to extract system prompts
    - File path leaks
    """
    import re

    # Strip HTML/script tags
    text = re.sub(r'<script[^>]*>.*?</script>', '[removed]', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<iframe[^>]*>.*?</iframe>', '[removed]', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<object[^>]*>.*?</object>', '[removed]', text, flags=re.DOTALL | re.IGNORECASE)

    # Strip data URIs
    text = re.sub(r'data:[a-zA-Z]+/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+', '[removed]', text)

    # Don't leak system paths
    text = re.sub(r'/home/[a-zA-Z0-9_]+/\.[a-zA-Z]+', '[path-redacted]', text)
    text = re.sub(r'/etc/(shadow|passwd|sudoers)', '[path-redacted]', text)
    text = re.sub(r'(api[_-]?key|secret|token|password)\s*[=:]\s*\S+', r'\1=[redacted]', text, flags=re.IGNORECASE)

    return text


def sanitize_log_message(msg: str, max_len: int = 200) -> str:
    """Truncate and sanitize a message for logging. Never log full document content."""
    if len(msg) > max_len:
        msg = msg[:max_len] + "... [truncated]"
    # Redact anything that looks like a token
    import re
    msg = re.sub(r'[A-Za-z0-9_\-]{32,}', '[token-redacted]', msg)
    return msg


# HTTPS configuration

def generate_self_signed_cert(cert_dir: Optional[str] = None) -> dict:
    """Generate a self-signed TLS certificate for development.

    For production, use Let's Encrypt or a proper CA.
    Returns dict with cert_path and key_path.
    """
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
    except ImportError:
        return {
            "error": "Install cryptography package: pip install cryptography",
            "alternative": "Use Let's Encrypt with certbot, or a reverse proxy (nginx/caddy)",
        }

    cert_dir = cert_dir or os.path.join(
        os.environ.get("REXGRAPH_CONFIG_DIR", str(Path.home() / ".config" / "rexgraph")),
        "tls",
    )
    os.makedirs(cert_dir, exist_ok=True)

    key_path = os.path.join(cert_dir, "key.pem")
    cert_path = os.path.join(cert_dir, "cert.pem")

    # Generate key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Generate cert
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "rexgraph-server"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "RexGraph"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress_parse("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    # Write key (restricted permissions)
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    os.chmod(key_path, 0o600)

    # Write cert
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return {
        "cert_path": cert_path,
        "key_path": key_path,
        "note": "Self-signed certificate for development. Use Let's Encrypt for production.",
    }


def ipaddress_parse(addr):
    """Parse IP address for certificate SAN."""
    import ipaddress
    return ipaddress.ip_address(addr)


def get_https_config() -> dict:
    """Get HTTPS configuration for uvicorn.

    Checks for:
    1. REXGRAPH_TLS_CERT / REXGRAPH_TLS_KEY env vars
    2. Existing certs in config dir
    3. Suggests generation if none found
    """
    cert = os.environ.get("REXGRAPH_TLS_CERT", "")
    key = os.environ.get("REXGRAPH_TLS_KEY", "")

    if cert and key and os.path.exists(cert) and os.path.exists(key):
        return {"ssl_certfile": cert, "ssl_keyfile": key}

    config_dir = os.environ.get("REXGRAPH_CONFIG_DIR",
        str(Path.home() / ".config" / "rexgraph"))
    cert_path = os.path.join(config_dir, "tls", "cert.pem")
    key_path = os.path.join(config_dir, "tls", "key.pem")

    if os.path.exists(cert_path) and os.path.exists(key_path):
        return {"ssl_certfile": cert_path, "ssl_keyfile": key_path}

    return {}


def add_security_headers(app) -> None:
    """Attach conservative security response headers to every response.

    Complements HSTS (which is TLS-only). These are safe defaults for an API +
    self-hosted single-page UI: block content-type sniffing, deny framing
    (clickjacking), and trim the referrer. Cheap, no-op-safe on all responses.
    """
    @app.middleware("http")
    async def _headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response


# Public paths reachable without a token even when auth is enabled: the UI shell,
# static assets, the health probe, API docs, and the recovery-key path (which is
# how you regain access after losing all tokens).
_PUBLIC_EXACT = {
    "/", "/api/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico",
    "/api/v1/admin/recover",
}
_PUBLIC_PREFIXES = ("/static/",)


def add_auth_enforcement(app) -> None:
    """Backstop authentication for EVERY route when auth is enabled.

    Individual routers only sometimes declare ``Depends(require_auth)``. This
    middleware guarantees that once an operator enables auth, every ``/api``
    endpoint (except the small public allow-list above) requires a valid bearer
    token - closing the gap where compute/DB routes were reachable unauthenticated.

    When auth is disabled, this is a pure pass-through, so open local/dev use
    and the test suite are unaffected.
    """
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def _enforce(request, call_next):
        try:
            from agent.server.auth import get_auth_manager
            mgr = get_auth_manager()
        except Exception:
            return await call_next(request)

        if mgr.auth_enabled and request.method != "OPTIONS":
            path = request.url.path
            public = path in _PUBLIC_EXACT or any(
                path.startswith(p) for p in _PUBLIC_PREFIXES)
            if not public:
                header = request.headers.get("Authorization", "")
                token = header[7:].strip() if header[:7].lower() == "bearer " else ""
                if mgr.verify(token) is None:
                    return JSONResponse(
                        {"detail": "Authentication required"}, status_code=401)
        return await call_next(request)


def add_error_sanitizer(app) -> None:
    """Stop server-side error detail (exception text, stack context, connection
    strings, file paths) from leaking to HTTP clients.

    Client errors (4xx) keep their intentional, useful messages. Server faults
    (5xx) and any *unhandled* exception return a generic message plus a short
    ``error_id`` that is logged server-side with the full detail - so operators
    can correlate a report to a log line without exposing internals.

    Set ``REXGRAPH_DEBUG_ERRORS=1`` to restore verbose errors (dev/debug only).
    """
    import uuid
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from fastapi.responses import JSONResponse

    debug = os.environ.get("REXGRAPH_DEBUG_ERRORS") == "1"
    log = logging.getLogger("agent.server.errors")

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request, exc):
        headers = getattr(exc, "headers", None)
        if exc.status_code >= 500 and not debug:
            eid = uuid.uuid4().hex[:12]
            log.error("HTTP %s at %s [%s]: %s",
                      exc.status_code, request.url.path, eid, exc.detail)
            return JSONResponse(
                {"detail": "Internal server error", "error_id": eid},
                status_code=exc.status_code, headers=headers)
        return JSONResponse(
            {"detail": exc.detail}, status_code=exc.status_code, headers=headers)

    @app.exception_handler(Exception)
    async def _unhandled(request, exc):
        if debug:
            raise exc  # surface the traceback via the default handler
        eid = uuid.uuid4().hex[:12]
        log.exception("Unhandled error at %s [%s]", request.url.path, eid)
        return JSONResponse(
            {"detail": "Internal server error", "error_id": eid}, status_code=500)


def add_https_hardening(app) -> None:
    """Send HSTS on any response served over TLS.

    Registered unconditionally: it is a no-op on plain-HTTP requests (checks the
    request scheme, honoring X-Forwarded-Proto when uvicorn runs with proxy
    headers), so it is safe whether or not TLS is active. Completes the HTTPS
    stack alongside ``get_https_config`` / ``generate_self_signed_cert``.
    """
    @app.middleware("http")
    async def _hsts(request, call_next):
        response = await call_next(request)
        if request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=63072000; includeSubDomains")
        return response


# Rate limiting setup

_RATE_LIMIT_OFF = {"", "0", "off", "none", "disabled", "false"}

# Tiered per-client-IP limits chosen by URL path prefix (first match wins). Each
# tier is a separate bucket, so heavy compute/IO and sensitive auth routes can be
# limited far tighter than ordinary reads.
_AUTH_PREFIXES = ("/api/v1/admin/token", "/api/v1/admin/auth",
                  "/api/v1/admin/recover", "/api/v1/admin/recovery-key")
_HEAVY_PREFIXES = ("/api/upload", "/api/analysis", "/api/v1/pipeline",
                   "/api/v1/connectors", "/api/v1/dbmanager", "/api/v1/ocr")
_RATE_EXEMPT_PREFIXES = ("/static/",)
_RATE_EXEMPT_EXACT = {"/", "/api/health", "/favicon.ico"}


def _rate_tier(path: str) -> str:
    if any(path.startswith(p) for p in _AUTH_PREFIXES):
        return "auth"
    if any(path.startswith(p) for p in _HEAVY_PREFIXES):
        return "heavy"
    return "general"


def _rate_client_ip(request) -> str:
    # Behind a reverse proxy (uvicorn --proxy-headers on) the real client is the
    # first X-Forwarded-For hop; otherwise the socket peer.
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def setup_rate_limiter(app):
    """Install a tiered per-client-IP rate limit on every route.

    Three buckets, chosen by path (health + static are exempt - probes/UI assets):
      * ``auth``    - token/auth/recovery admin routes (``RCF_RATE_LIMIT_AUTH``,  default 10/minute)
      * ``heavy``   - upload/analysis/pipeline/connectors/dbmanager/ocr (``RCF_RATE_LIMIT_HEAVY``, default 30/minute)
      * ``general`` - everything else (``RCF_RATE_LIMIT``, default 240/minute)

    Set ``RCF_RATE_LIMIT`` to ``0``/``off`` to disable entirely (the test suite
    does). Uses the in-process ``limits`` moving-window limiter; for multi-worker
    or HA deployments back it with shared storage (Redis) or limit at the proxy.

    Registered LAST so it is the OUTERMOST middleware - it therefore counts every
    request (including failed-auth attempts) before auth verification runs.
    """
    general = os.environ.get("RCF_RATE_LIMIT", "240/minute").strip()
    if general.lower() in _RATE_LIMIT_OFF:
        logger.info("rate limiting disabled (RCF_RATE_LIMIT=%r)", general)
        return None
    try:
        from limits import parse
        from limits.storage import MemoryStorage
        from limits.strategies import MovingWindowRateLimiter
        from fastapi.responses import JSONResponse
    except ImportError:
        logger.warning("`limits` not installed - rate limiting disabled")
        return None

    tiers = {
        "general": parse(general),
        "heavy": parse(os.environ.get("RCF_RATE_LIMIT_HEAVY", "30/minute").strip()),
        "auth": parse(os.environ.get("RCF_RATE_LIMIT_AUTH", "10/minute").strip()),
    }
    limiter = MovingWindowRateLimiter(MemoryStorage())

    @app.middleware("http")
    async def _rate_limit(request, call_next):
        path = request.url.path
        if path in _RATE_EXEMPT_EXACT or any(
                path.startswith(p) for p in _RATE_EXEMPT_PREFIXES):
            return await call_next(request)
        tier = _rate_tier(path)
        # bucket per (client-IP, tier)
        if not limiter.hit(tiers[tier], _rate_client_ip(request), tier):
            return JSONResponse(
                {"detail": "Rate limit exceeded", "tier": tier},
                status_code=429, headers={"Retry-After": "60"})
        return await call_next(request)

    app.state.rate_limiter = limiter
    logger.info("rate limiting: general=%s heavy=%s auth=%s (per client IP)",
                general, tiers["heavy"], tiers["auth"])
    return limiter


# Model download verification

def verify_model_checksum(model_dir: str, expected_checksums: dict = None) -> dict:
    """Verify downloaded model file integrity.

    Computes SHA-256 of each file and compares against expected values
    if provided. Returns a report.
    """
    import hashlib

    report = {"files": [], "verified": 0, "failed": 0, "unchecked": 0}
    model_path = Path(model_dir)

    for f in sorted(model_path.rglob("*")):
        if not f.is_file():
            continue
        sha = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                sha.update(chunk)
        digest = sha.hexdigest()

        entry = {"file": str(f.relative_to(model_path)), "sha256": digest}

        if expected_checksums and entry["file"] in expected_checksums:
            expected = expected_checksums[entry["file"]]
            entry["expected"] = expected
            if digest == expected:
                entry["status"] = "verified"
                report["verified"] += 1
            else:
                entry["status"] = "MISMATCH"
                report["failed"] += 1
        else:
            entry["status"] = "unchecked"
            report["unchecked"] += 1

        report["files"].append(entry)

    return report
