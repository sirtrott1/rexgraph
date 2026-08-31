"""
RexGraph Agent: FastAPI application.

Serves the API endpoints and the frontend static files.
One command starts everything: python run.py
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse as _StarletteJSONResponse

from .launch import CORE_VERSION, VERSION
from .budget import add_compute_budget
from .scope import add_workspace_scope
from .security import (
    add_auth_enforcement,
    add_error_sanitizer,
    add_https_hardening,
    add_security_headers,
    cleanup_stale_tempfiles,
    setup_rate_limiter,
)
from .state import SessionStore

# Application

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """On startup: warm-load the activity journal (history across restarts) and tail it, so events
    recorded by any other local process - a `rexgraph-*` CLI, a worker - fold into this server's log
    and stream live to the UI. The journal is the cross-process event bus; this is the server's end."""
    try:
        from agent import activity
        activity.get_log().enable_journal(warm=True, tail=True)
    except Exception:
        pass
    yield


class FiniteJSONResponse(_StarletteJSONResponse):
    """JSON with non-finite floats rendered as null.

    A measurement over nothing is NaN and a mixing time with no cycle to mix through
    is infinity. Both are honest values and neither is JSON: the default encoder
    raises `Out of range float values are not JSON compliant`, so one empty complex
    takes the whole response down rather than reporting that a number is absent.

    This is the same encoder the SSE path uses, applied as the app's default so every
    route gets it instead of each one guarding separately.
    """

    def render(self, content) -> bytes:
        from rexgraph.io._compat import dumps
        return dumps(content, nan="null").encode("utf-8")


app = FastAPI(
    title="RexGraph Agent",
    description="Mathematical agent engine built on the Relational Complex Framework",
    version=VERSION,
    lifespan=_lifespan,
    default_response_class=FiniteJSONResponse,
)

import os

_cors_origins = os.environ.get("REXGRAPH_CORS_ORIGINS", "").split(",")
_cors_origins = [o.strip() for o in _cors_origins if o.strip()]
if not _cors_origins:
    _cors_origins = ["http://localhost:8000", "http://localhost:3000", "http://127.0.0.1:8000",
                     "https://localhost:8000", "https://localhost:3000", "https://127.0.0.1:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Workspace"],
)

# Starlette runs the LAST-registered middleware FIRST, so these three are registered
# in reverse of the order they run: a request is authenticated, then scoped to its
# workspace, then metered, and only then reaches a route.
add_compute_budget(app)
add_workspace_scope(app)
add_auth_enforcement(app)

# Conservative security headers on every response.
add_security_headers(app)

# Sanitize error responses (generic 5xx + error_id; verbose with REXGRAPH_DEBUG_ERRORS=1).
add_error_sanitizer(app)

# HTTPS hardening (HSTS on TLS responses; no-op on plain HTTP)
add_https_hardening(app)

# Rate limiting LAST -> outermost middleware, so it counts every request
# (including failed-auth attempts) before auth verification runs.
_limiter = setup_rate_limiter(app)

# Clean stale temp files on startup
try:
    removed = cleanup_stale_tempfiles()
    if removed:
        import logging
        logging.getLogger(__name__).info("Cleaned %d stale temp files", removed)
except Exception:
    pass

# Session store (singleton)

_store: SessionStore | None = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


# Routes

from agent.session import SnapshotUnreadable
from fastapi.responses import JSONResponse

from .routes import (
    admin,
    agents,
    analysis,
    builder,
    chat,
    connectors,
    corpus,
    courier,
    dbmanager,
    deploy,
    enrichment,
    explore,
    export,
    graph,
    hive,
    integrations,
    knowledge,
    mcp,
    ml,
    model,
    models,
    ocr,
    ontology,
    ops,
    pipeline,
    rcdb,
    releases,
    rex,
    schema,
    session,
    upload,
)

@app.exception_handler(SnapshotUnreadable)
async def _snapshot_unreadable(request, exc):
    """A stored session that cannot be read is the request's answer, not a crash.

    Seven routes reach a session's complex; handling it here means none of them
    has to, and none of them can forget to.
    """
    return JSONResponse(status_code=422, content={"detail": str(exc),
                                                  "error": "snapshot_unreadable"})


app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(analysis.router, prefix="/api", tags=["analysis"])
app.include_router(explore.router, prefix="/api", tags=["explore"])
app.include_router(session.router, prefix="/api", tags=["sessions"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(ocr.router, prefix="/api", tags=["ocr"])
app.include_router(model.router, prefix="/api", tags=["model"])
app.include_router(corpus.router, prefix="/api", tags=["corpus"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(pipeline.router, prefix="/api", tags=["pipeline"])
app.include_router(admin.router, prefix="/api", tags=["admin"])
app.include_router(export.router, prefix="/api", tags=["export"])
app.include_router(deploy.router, prefix="/api", tags=["deploy"])
app.include_router(rcdb.router, prefix="/api", tags=["rcdb"])
app.include_router(graph.router, prefix="/api", tags=["graph"])
app.include_router(schema.router, prefix="/api", tags=["schema"])
app.include_router(dbmanager.router, prefix="/api", tags=["dbmanager"])
app.include_router(ontology.router, prefix="/api", tags=["ontology"])
app.include_router(knowledge.router, prefix="/api", tags=["knowledge"])
app.include_router(enrichment.router, prefix="/api", tags=["enrichment"])
app.include_router(releases.router, prefix="/api", tags=["releases"])
app.include_router(mcp.router, prefix="/api", tags=["mcp"])
app.include_router(connectors.router, prefix="/api", tags=["connectors"])
app.include_router(integrations.router, prefix="/api", tags=["integrations"])
app.include_router(agents.router, prefix="/api", tags=["agents"])
app.include_router(hive.router, prefix="/api", tags=["hive"])
app.include_router(courier.router, prefix="/api", tags=["courier"])
app.include_router(ops.router, prefix="/api", tags=["ops"])
app.include_router(ml.router, prefix="/api", tags=["ml"])
app.include_router(builder.router, prefix="/api", tags=["builder"])
# mounted at the root rather than under /api: the native surface is the contract other
# rexgraph software speaks, not one more route group behind the UI's prefix
app.include_router(rex.router, tags=["rex"])

# Frontend

# Frontend static files live at the sibling root (agent/frontend/), not inside
# the Python package. From agent/agent/server/app.py, go up 3 levels to reach
# agent/, then down into frontend/.
_FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    index_path = _FRONTEND_DIR / "index.html"
    if index_path.exists():
        # the UI reports the package version, not a build or commit id
        return HTMLResponse(
            index_path.read_text()
            .replace("__CORE_VERSION__", CORE_VERSION)
            .replace("__VERSION__", VERSION)
        )
    return HTMLResponse(
        "<html><body><h1>RexGraph Agent</h1>"
        "<p>Frontend not built yet. API is live at <a href='/docs'>/docs</a></p>"
        "</body></html>"
    )


# Mount static files (frontend JS, CSS, components).
# app.jsx is loaded via <script src="/static/app.jsx">; StaticFiles would otherwise
# guess its type as application/octet-stream, which the browser REFUSES to execute
# as a script because we send X-Content-Type-Options: nosniff. Register .jsx as
# JavaScript so the UI actually runs (keeping the nosniff hardening intact).
import mimetypes

mimetypes.add_type("text/javascript", ".jsx")
mimetypes.add_type("text/javascript", ".mjs")

if _FRONTEND_DIR.exists():
    class _NoCacheStatic(StaticFiles):
        # frontend assets (app.jsx / theme.css) are edited in place; without this the browser
        # heuristic-caches them and edits don't show. 'no-cache' still uses the etag (cheap 304s)
        # but revalidates every load, so the UI is never stale.
        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            resp.headers["Cache-Control"] = "no-cache"
            return resp
    app.mount("/static", _NoCacheStatic(directory=str(_FRONTEND_DIR)), name="static")


# Health check

@app.get("/api/health")
async def health(request: Request):
    """Liveness, and the workspace roster only for a caller entitled to it.

    The probe stays unauthenticated so a load balancer can reach it, but the roster
    does not travel with it. A workspace name is exactly what `X-Workspace` takes, so
    listing every tenant to an anonymous caller hands them the namespace to aim at and
    names usually carry customer identity. With auth off there is one operator and no
    tenant boundary to cross, so the list is what it always was.
    """
    from agent.server.auth import get_auth_manager

    from .launch import _check_rexgraph
    mgr = get_auth_manager()
    out = {
        "status": "ok",
        "version": VERSION,
        "core_version": CORE_VERSION,
        "rexgraph": _check_rexgraph(),
        "auth_enabled": mgr.auth_enabled,
    }
    if not mgr.auth_enabled:
        out["workspaces"] = mgr.list_workspaces() or ["default"]
        return out
    header = request.headers.get("Authorization", "")
    raw = header[7:].strip() if header[:7].lower() == "bearer " else ""
    entry = mgr.verify(raw) if raw else None
    if entry is not None:
        # A wildcard token holds a role in every workspace, so the whole roster IS its
        # access list. Anyone else sees only what they can actually reach.
        out["workspaces"] = (mgr.list_workspaces() if entry.can_access("*")
                             else list(entry.workspaces))
    return out


# Console-script entry point (rcf-server)


def main() -> None:
    """Launch the FastAPI server via uvicorn (invoked by the `rcf-server`
    console script). A thin env->:func:`agent.server.launch.serve` wrapper; all
    launch logic lives in that one module.

    Environment variables:
        RCF_HOST (default 127.0.0.1)
        RCF_PORT (default 8000)
        RCF_RELOAD ("1" for dev auto-reload)
        RCF_HTTPS ("1" to serve HTTPS, auto-generating a self-signed cert if none
                   is configured)
        REXGRAPH_TLS_CERT / REXGRAPH_TLS_KEY (paths to your own cert/key; when
                   set, HTTPS is used automatically)
        RCF_WORKERS (worker process count; ignored with RCF_RELOAD)
        RCF_FORWARDED_ALLOW_IPS (trusted proxy IPs for X-Forwarded-Proto)
    """
    import os

    from .launch import serve

    serve(
        host=os.environ.get("RCF_HOST", "127.0.0.1"),
        port=int(os.environ.get("RCF_PORT", "8000")),
        reload=os.environ.get("RCF_RELOAD") == "1",
        https=os.environ.get("RCF_HTTPS") == "1",
        workers=int(os.environ["RCF_WORKERS"]) if os.environ.get("RCF_WORKERS") else None,
    )


if __name__ == "__main__":
    main()
