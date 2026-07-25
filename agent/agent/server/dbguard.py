"""
agent.server.dbguard - opt-in allow-list for database connection URIs.

The dbmanager/connectors HTTP routes accept a connection URI and hand it to
SQLAlchemy ``create_engine`` / schema reflection. On a network-exposed server that
is an SSRF + local-file-read surface (``sqlite:////etc/passwd``,
``postgresql://internal-host/…``). This module gates it.

**Off by default** (nothing set -> every URI is allowed, preserving local/dev use).
Turn it on per deployment with any of:

    REXGRAPH_DB_SAFE=1              # preset: block file-based DBs (sqlite/duckdb)
                                    # and loopback/private-network hosts (anti-SSRF)
    REXGRAPH_ALLOWED_DB_SCHEMES=postgresql,mysql,snowflake   # scheme allow-list
    REXGRAPH_ALLOWED_DB_HOSTS=db1.corp,db2.corp              # host allow-list

Enforcement runs at the HTTP boundary only; the local `rexgraph-connect` CLI (an
operator on the box) is intentionally unrestricted.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

from fastapi import HTTPException

# Schemes that read a LOCAL FILE rather than talk to a server - blocked by the
# REXGRAPH_DB_SAFE preset because they can exfiltrate arbitrary files.
_FILE_SCHEMES = {"sqlite", "duckdb", "access", "csv"}


def _csv_env(name: str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def _policy_active() -> bool:
    return bool(
        os.environ.get("REXGRAPH_DB_SAFE") == "1"
        or os.environ.get("REXGRAPH_DB_BLOCK_LOCAL") == "1"
        or _csv_env("REXGRAPH_ALLOWED_DB_SCHEMES")
        or _csv_env("REXGRAPH_ALLOWED_DB_HOSTS")
    )


def _ip_blocked(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return (ip.is_loopback or ip.is_private or ip.is_link_local
            or ip.is_reserved or ip.is_unspecified or ip.is_multicast)


def _host_is_local_or_private(host: str) -> bool:
    if not host:
        return False
    if host.lower() in ("localhost", "localhost.localdomain", "ip6-localhost"):
        return True
    # Literal IP -> check directly.
    try:
        ipaddress.ip_address(host)
        return _ip_blocked(host)
    except ValueError:
        pass
    # Hostname -> resolve and block if ANY resolved address is loopback/private.
    # This closes the hostname->private-IP SSRF case (e.g. a name that resolves to
    # 169.254.169.254). Unresolvable names fall through - the real connection just
    # fails. Full DNS-rebinding defense needs connect-time IP pinning (out of scope).
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return any(_ip_blocked(info[4][0]) for info in infos)


def check_db_uri(uri: str) -> None:
    """Raise HTTPException(400) if ``uri`` violates the configured DB policy.

    A no-op when no policy env var is set, and for non-URI values (bare in-memory
    scheme names like ``edgelist`` that carry no ``://``).
    """
    if not uri or "://" not in uri or not _policy_active():
        return

    parsed = urlparse(uri)
    base_scheme = (parsed.scheme or "").split("+", 1)[0].lower()
    host = parsed.hostname or ""

    allowed_schemes = _csv_env("REXGRAPH_ALLOWED_DB_SCHEMES")
    allowed_hosts = _csv_env("REXGRAPH_ALLOWED_DB_HOSTS")
    safe = (os.environ.get("REXGRAPH_DB_SAFE") == "1"
            or os.environ.get("REXGRAPH_DB_BLOCK_LOCAL") == "1")

    if allowed_schemes is not None and base_scheme not in allowed_schemes:
        raise HTTPException(
            400, f"database scheme '{base_scheme}' is not permitted "
                 f"(REXGRAPH_ALLOWED_DB_SCHEMES)")

    if safe and base_scheme in _FILE_SCHEMES:
        raise HTTPException(
            400, f"file-based database scheme '{base_scheme}' is blocked "
                 f"(REXGRAPH_DB_SAFE); connect to a database server instead")

    if safe and _host_is_local_or_private(host):
        raise HTTPException(
            400, f"connection to loopback/private host '{host}' is blocked "
                 f"(REXGRAPH_DB_SAFE, anti-SSRF)")

    if allowed_hosts is not None and host.lower() not in allowed_hosts:
        raise HTTPException(
            400, f"database host '{host or '(none)'}' is not permitted "
                 f"(REXGRAPH_ALLOWED_DB_HOSTS)")
