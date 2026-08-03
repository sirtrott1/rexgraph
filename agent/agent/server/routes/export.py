"""
agent.server.routes.export: session and workspace export.

    GET  /api/v1/export/session/{id}   export a session's analysis as JSON
    GET  /api/v1/export/workspace      export entire workspace
    GET  /api/v1/export/queries        export query history
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from agent.server.auth import TokenEntry, WorkspaceState, require_auth, require_workspace

router = APIRouter(prefix="/v1/export")


@router.get("/session/{session_id}")
async def export_session(
    session_id: str,
    format: str = Query("json"),
    token: TokenEntry = Depends(require_auth),
    ws: WorkspaceState = Depends(require_workspace),
):
    """Export a session's analysis results."""
    from agent.server.app import get_store
    store = get_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    snap = session.snapshots[session.current_step] if session.snapshots else None
    results = snap.results if snap else {}
    rex = session.current()

    if format == "json":
        export = {
            "session_id": session_id,
            "workspace": ws.name,
            "results": results,
        }
        if rex:
            export["topology"] = {
                "nV": rex.nV, "nE": rex.nE, "nF": rex.nF,
                "betti": rex.betti,
            }
        return JSONResponse(export)

    if rex is not None and format in ("safetensors", "hdf5", "h5"):
        import os
        import tempfile
        suffix = ".safetensors" if format == "safetensors" else ".h5"
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            if format == "safetensors":
                from rexgraph.io.safetensors_bridge import rex_to_safetensors
                rex_to_safetensors(rex, tmp)
            else:
                from rexgraph.io import save_hdf5
                save_hdf5(tmp, rex)
        except ImportError as e:
            raise HTTPException(
                400, f"'{format}' export needs an optional dependency: {e}")
        except Exception as e:
            raise HTTPException(500, f"Export failed: {e}")
        return FileResponse(tmp, filename=f"{session_id}{suffix}",
                            media_type="application/octet-stream")

    if format == "rex" and rex:
        from agent.server.persistence import save_document_rex
        path = save_document_rex(ws.name, session_id, rex)
        return {"path": path, "format": "rex"}

    raise HTTPException(400, f"Unsupported format: {format}")


@router.get("/workspace")
async def export_workspace(
    format: str = Query("json"),
    token: TokenEntry = Depends(require_auth),
    ws: WorkspaceState = Depends(require_workspace),
):
    """Export the entire workspace."""
    from agent.server.persistence import export_workspace as _export

    if format == "json":
        from agent.server.persistence import (
            list_document_bundles,
            load_query_history,
        )
        return {
            "workspace": ws.name,
            "documents": list_document_bundles(ws.name),
            "activity": ws.activity_summary(),
            "queries": load_query_history(ws.name),
        }

    from agent.server.security import secure_tempfile
    with secure_tempfile(suffix=f".{format}") as tmp:
        _export(ws.name, tmp, fmt=format)
        if os.path.exists(tmp):
            return FileResponse(tmp, filename=f"{ws.name}.{format}")
    raise HTTPException(500, "Export failed")


@router.get("/queries")
async def export_queries(
    limit: int = Query(50),
    token: TokenEntry = Depends(require_auth),
    ws: WorkspaceState = Depends(require_workspace),
):
    """Export query history for the workspace."""
    from agent.server.persistence import load_query_history
    return {"workspace": ws.name, "queries": load_query_history(ws.name, limit=limit)}
