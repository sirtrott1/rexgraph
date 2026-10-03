"""RexGraph System FastAPI application."""
from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
from rcql import Executor, call, parse
from rcql import query as rcql_query
from rcql import source as rcql_source
from rcql.executor import format_expr

from system.panels import panel_query
from system.inspection import source_description, source_row
from system.serialize import json_value
from system.state import sources
from system import __version__

app = FastAPI(title="RexGraph System", version=__version__)

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


class QueryRequest(BaseModel):
    exactness: Literal["declared", "exact"] = "declared"
    query: str
    params: dict = Field(default_factory=dict)
    result_format: Literal["preview", "native"] = "preview"


def _result_payload(result):
    return {
        "values": [json_value(value) for value in result.values],
        "aliases": list(result.aliases),
        "plan": list(result.plan),
        "exactness": [None if item is None else item.value for item in result.exactness],
        "native_plan": json_value(result.native_plan),
        "provenance": json_value(result.provenance),
        "execution": json_value(result.execution),
        "rewrites": [
            {
                "before": format_expr(item.before),
                "after": format_expr(item.after),
                "reason": item.reason,
            }
            for item in result.rewrites
        ],
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    path = _FRONTEND_DIR / "index.html"
    if not path.exists():
        return HTMLResponse("<h1>RexGraph System</h1><p>Frontend is not installed.</p>")
    return HTMLResponse(path.read_text())


@app.get("/api/health")
async def health():
    return {"status": "ok", "sources": len(sources.snapshot())}


@app.get("/api/sources")
async def list_sources():
    return {"version": __version__,
            "sources": [source_row(name, value) for name, value in sources.snapshot().items()]}


@app.get("/api/source")
@app.get("/api/sources/{name:path}")
async def source_detail(name: str):
    try:
        value = sources.get(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return source_description(value, detail=True)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/catalog")
@app.get("/api/catalogs/{name}")
async def catalog_entries(name: str, q: str = "", limit: int = 100, offset: int = 0):
    """Return bounded catalog metadata without exposing filesystem paths."""
    try:
        expr = call("SEARCH", q, limit) if q else call("FILES", limit, offset)
        result = Executor(sources=sources.snapshot()).execute(
            rcql_query(rcql_source(name), expr))
        rows = result.values[0]
    except (KeyError, TypeError, ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"name": name, "entries": [json_value(row) for row in rows]}


@app.get("/api/catalogs/{name}/tensors")
async def catalog_tensors(name: str, entry: str, q: str = "", limit: int = 100):
    """Return bounded safetensors metadata without loading tensor payloads."""
    try:
        expr = call("SEARCH_TENSORS", entry, q, limit) if q else call("TENSORS", entry, limit)
        result = Executor(sources=sources.snapshot()).execute(
            rcql_query(rcql_source(name), expr))
        rows = result.values[0]
    except (KeyError, TypeError, ValueError, ImportError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"name": entry, "tensors": rows}


@app.get("/api/catalogs/{name}/hash")
async def catalog_hash(name: str, entry: str):
    """Return the current byte hash for one catalog entry."""
    try:
        result = Executor(sources=sources.snapshot()).execute(
            rcql_query(rcql_source(name), call("FILE_HASH", entry)))
        return {"name": entry, "sha256": result.values[0]}
    except (KeyError, TypeError, ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/panels/{panel}")
async def panel(panel: str, source: str):
    try:
        bindings = sources.snapshot()
        value = bindings[source]
        planned = panel_query(panel, source, value)
        result = Executor(sources=bindings).execute(planned)
    except (KeyError, TypeError, ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _result_payload(result)


@app.get("/api/rcdb/{name}/security")
async def rcdb_security(name: str):
    """Return bounded security configuration through the RCQL admin capability."""
    try:
        result = Executor(sources=sources.snapshot()).execute(
            rcql_query(rcql_source(name), call("RCDB_SECURITY")))
    except (KeyError, TypeError, ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return json_value(result.values[0])


@app.get("/api/rcdb/{name}/state-hash")
async def rcdb_state_hash(name: str):
    """Return the canonical logical RCDB state digest through RCQL."""
    try:
        result = Executor(sources=sources.snapshot()).execute(
            rcql_query(rcql_source(name), call("RCDB_STATE_HASH")))
    except (KeyError, TypeError, ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"digest": str(result.values[0])}


@app.get("/api/rcdb/{name}/{record_id}/commits")
async def rcdb_commits(name: str, record_id: str, limit: int = 100):
    """Return mutation lineage through the same RCQL capability boundary as queries."""
    try:
        result = Executor(sources=sources.snapshot()).execute(
            rcql_query(rcql_source(name), call("RCDB_COMMITS", record_id, limit),
                       call("RCDB_VERIFY", record_id)))
    except (KeyError, TypeError, ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"commits": json_value(result.values[0]), "verified": bool(result.values[1])}


@app.post("/api/query")
async def query(req: QueryRequest):
    try:
        parsed = parse(req.query)
        executor = Executor(sources=sources.snapshot(), params=req.params, exactness=req.exactness)
        result = executor.execute(parsed)
        if req.result_format == "native":
            return Response(content=result.to_bytes(), media_type="application/vnd.rexgraph.rcql-result",
                            headers={"Content-Disposition": 'attachment; filename="result.rgqr"'})
    except (KeyError, TypeError, ValueError, PermissionError, SyntaxError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _result_payload(result)


mimetypes.add_type("text/javascript", ".jsx")
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
