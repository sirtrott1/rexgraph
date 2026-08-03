"""
Upload route: file upload -> auto_rex -> new session -> quick results.

After the initial quick results are returned, the full analysis continues
in background and streams via SSE on the /api/analysis/{id}/stream endpoint.
"""

from __future__ import annotations

import json
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from agent.auto import auto_rex
from agent.pipeline import AnalysisPipeline

router = APIRouter()


@router.post("/upload")
async def upload_data(
    file: UploadFile = File(...),
    options: str = Form("{}"),
):
    """Upload a dataset and auto-construct the relational complex.

    Returns session ID and quick topology results immediately.
    Full analysis streams via /api/analysis/{session_id}/stream.

    Parameters
    ----------
    file : uploaded file (CSV, JSON, or rexgraph format)
    options : JSON string with optional overrides:
        threshold, typing, sign, face_selection
    """
    from agent.server.app import get_store

    store = get_store()

    # Parse options
    try:
        opts = json.loads(options)
    except (json.JSONDecodeError, TypeError):
        opts = {}

    # Save uploaded file to temp location, streaming with a hard size cap so a
    # large (or malicious) upload can't exhaust memory/disk. Override the limit
    # with REXGRAPH_MAX_UPLOAD_MB (default 100 MB).
    try:
        max_mb = float(os.environ.get("REXGRAPH_MAX_UPLOAD_MB", "100"))
    except ValueError:
        max_mb = 100.0
    max_bytes = int(max_mb * 1024 * 1024)
    suffix = os.path.splitext(file.filename or "data.csv")[1]
    written = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=tempfile.gettempdir()) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                tmp.close()
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {max_mb:g} MB limit "
                           f"(set REXGRAPH_MAX_UPLOAD_MB to change).")
            tmp.write(chunk)

    try:
        # Auto-construct the rex
        rex = auto_rex(tmp_path, **opts)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Failed to build relational complex: {e}")
    finally:
        # Clean up temp file (rex data is now in memory)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Create session
    name = file.filename or "upload"
    session = store.create(name=name)
    meta = getattr(rex, "_agent_meta", {})

    # Run quick analysis (stages 1-3, < 1 second)
    pipeline = AnalysisPipeline(rex)
    quick_results = pipeline.run(depth="quick")

    # Save snapshot
    session.add_snapshot(
        rex=rex,
        action="upload",
        params=opts,
        results=quick_results,
        summary=f"Uploaded {name}: {rex.nV}V {rex.nE}E {rex.nF}F",
    )

    return {
        "session_id": session.session_id,
        "filename": file.filename,
        "input_type": meta.get("input_type", "unknown"),
        "nV": rex.nV,
        "nE": rex.nE,
        "nF": rex.nF,
        "n_types": meta.get("n_types", 0),
        "type_names": meta.get("type_names", []),
        "quick_results": quick_results,
    }
