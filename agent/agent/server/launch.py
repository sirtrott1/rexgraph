"""
agent.server.launch: the one room behind both front doors.

`run.py` (flags) and `rcf-server` / `app.main()` (env) are thin wrappers that map
their inputs onto :func:`serve`. All the launch logic (TLS resolution via the
built in adapters, the banner, optional browser open, proxy header handling, and the
single ``uvicorn.run`` call) lives here, so the two paths cannot diverge and new
options (workers, etc.) are added in exactly one place.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def _agent_version() -> str:
    """The agent version, read the same way as the core's so the two cannot drift.

    `agent.__version__` first: installed metadata is fixed at install time, so an
    editable checkout kept reporting the version it was installed at while the core,
    read from its module, moved on. Metadata is the fallback for the case where the
    package is installed but its source is not on the path.
    """
    try:
        import agent
        v = getattr(agent, "__version__", None)
        if v:
            return v
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("rexgraph-agent")
    except Exception:
        return "unknown"


def _core_version() -> str:
    """The rexgraph core version. Reported alongside the agent version because the
    two ship on their own cadences and a bug report needs both."""
    try:
        import rexgraph
        return getattr(rexgraph, "__version__", "unknown")
    except Exception:
        return "unknown"


VERSION = _agent_version()
CORE_VERSION = _core_version()
_APP = "agent.server.app:app"


def _check_rexgraph() -> str:
    try:
        import rexgraph
        return getattr(rexgraph, "__version__", "installed")
    except ImportError:
        return "not installed"


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _enforce_bind_safety(host: str) -> None:
    """Fail closed when publishing the server on a non loopback interface with
    authentication DISABLED.

    Binding to 0.0.0.0 (or any non loopback host) while auth is off exposes an
    unauthenticated *admin* API to the network: anyone who can reach the port
    gets a local admin token. The default bind (127.0.0.1) is unaffected. Set
    RCF_ALLOW_INSECURE=1 to override deliberately (e.g. auth is terminated at an
    upstream reverse proxy / the port is firewalled to a private network)."""
    if host in _LOOPBACK_HOSTS:
        return
    if os.environ.get("RCF_ALLOW_INSECURE") == "1":
        logger.warning(
            "Binding %s with auth disabled (RCF_ALLOW_INSECURE=1). Ensure auth "
            "is enforced upstream or the port is firewalled.", host)
        return
    try:
        from agent.server.auth import get_auth_manager
        if get_auth_manager().auth_enabled:
            return
    except Exception:
        pass
    raise RuntimeError(
        f"Refusing to bind host {host!r} with authentication DISABLED - this would "
        "expose an unauthenticated admin API to the network.\n"
        "  Fix one of:\n"
        "    * enable auth:   rexgraph-auth enable   (create a token first: "
        "rexgraph-auth create --name admin --role admin --save)\n"
        "    * bind local:    RCF_HOST=127.0.0.1     (the default)\n"
        "    * override:      RCF_ALLOW_INSECURE=1   (only if auth is enforced "
        "upstream, e.g. a reverse proxy, or the port is firewalled)"
    )


def _secure_by_default() -> None:
    """A fresh server (no auth.json) starts with auth ON and a bootstrap admin token.

    Only touches a fresh install; once auth.json exists the operator's choice is
    respected. Set RCF_ALLOW_INSECURE=1 to keep the open default for local dev,
    or REXGRAPH_ADMIN_TOKEN to choose the bootstrap token instead of a random one.
    """
    if os.environ.get("RCF_ALLOW_INSECURE") == "1":
        return
    try:
        from agent.server.auth import get_auth_manager
        mgr = get_auth_manager()
    except Exception:
        return
    if not mgr.is_fresh:
        return
    if not mgr.auth_enabled:
        mgr.enable_auth()
    supplied = os.environ.get("REXGRAPH_ADMIN_TOKEN") or None
    raw = mgr.bootstrap_admin(supplied)
    if raw and not supplied:
        print("\n  Authentication is enabled (secure default for a fresh server).")
        print("  Admin token, shown once, save it now:")
        print(f"    {raw}")
        print("  Store it:  rexgraph-auth login --url <url> --token <token>")
        print("  Prefer open local dev? start with RCF_ALLOW_INSECURE=1\n")


def _ensure_ui_assets() -> None:
    """Validate packaged runtime code offline, without writing the installation."""
    from agent.ui_assets import validate_ui_assets
    validate_ui_assets()


def _open_browser(url: str) -> None:
    """Open a URL in the default browser via a fully detached process, so a
    browser crash/hang never propagates back to the server."""
    try:
        kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                  "stderr": subprocess.DEVNULL}
        if sys.platform == "darwin":
            subprocess.Popen(["open", url], start_new_session=True, **kwargs)
        elif sys.platform == "win32":
            os.startfile(url)  # noqa: S606, the only reliable option on Windows
        else:
            subprocess.Popen(["xdg-open", url], start_new_session=True, **kwargs)
    except Exception:
        pass  # best effort; never crash the server for it


def resolve_tls(https: bool = False, ssl_cert: str | None = None,
                ssl_key: str | None = None) -> tuple[dict[str, str], str]:
    """Resolve uvicorn SSL kwargs via the built in TLS adapters.

    One precedence, shared by both launchers:
      1. explicit ``ssl_cert`` + ``ssl_key`` (flags / args)
      2. configured certs: env ``REXGRAPH_TLS_CERT``/``KEY`` or the config dir
         (``get_https_config``); when present, HTTPS is used automatically
      3. ``https=True`` with none of the above -> generate a self signed cert
      4. otherwise -> plain HTTP

    Returns ``(ssl_kwargs, scheme)`` where scheme is "http" or "https".
    """
    from .security import generate_self_signed_cert, get_https_config

    if bool(ssl_cert) != bool(ssl_key):
        raise ValueError("explicit TLS requires both a certificate and private key")
    if ssl_cert and ssl_key:
        return {"ssl_certfile": ssl_cert, "ssl_keyfile": ssl_key}, "https"

    cfg = get_https_config()
    if cfg:
        return cfg, "https"

    if https:
        print("HTTPS requested but no TLS certs configured - generating a self-signed cert…")
        result = generate_self_signed_cert()
        if "error" in result:
            raise RuntimeError(f"HTTPS certificate generation failed: {result['error']}")
        print(f"  Cert: {result['cert_path']}")
        print(f"  Key:  {result['key_path']}")
        if result.get("note"):
            print(f"  {result['note']}")
        return {"ssl_certfile": result["cert_path"], "ssl_keyfile": result["key_path"]}, "https"

    return {}, "http"


def serve(host: str = "127.0.0.1", port: int = 8000, *, reload: bool = False,
          https: bool = False, ssl_cert: str | None = None,
          ssl_key: str | None = None, workers: int | None = None,
          open_browser: bool = False,
          forwarded_allow_ips: str | None = None) -> None:
    """Start the FastAPI app over uvicorn. The single launch path."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is not installed. Install the server extras:\n"
            "    pip install 'rexgraph-agent[server]'") from exc

    # A fresh install comes up authenticated with a one time admin token.
    _secure_by_default()
    # Acquire the vendored UI libraries if a bare install skipped install.sh.
    _ensure_ui_assets()
    # Fail closed if this would publish an unauthenticated admin API to the network.
    _enforce_bind_safety(host)

    ssl_kwargs, scheme = resolve_tls(https=https, ssl_cert=ssl_cert, ssl_key=ssl_key)
    url = f"{scheme}://{host}:{port}"

    if open_browser:
        import threading
        threading.Timer(1.5, lambda: _open_browser(url)).start()

    print(f"\n    rexgraph agent v{VERSION}\n    {url}\n    API docs: {url}/docs")
    print(f"    rexgraph backend: {_check_rexgraph()}\n")

    run_kwargs = dict(
        host=host, port=port, reload=reload,
        proxy_headers=True,
        forwarded_allow_ips=forwarded_allow_ips
        or os.environ.get("RCF_FORWARDED_ALLOW_IPS", "127.0.0.1"),
        **ssl_kwargs,
    )
    # workers>1 requires the import string app (used here) and is incompatible
    # with --reload; only pass it when it actually applies.
    if workers and workers > 1 and not reload:
        run_kwargs["workers"] = workers

    uvicorn.run(_APP, **run_kwargs)
