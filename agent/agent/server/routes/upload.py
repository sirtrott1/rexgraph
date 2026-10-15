"""
Upload route: file upload -> auto_rex -> new session -> quick results.

After the initial quick results are returned, the full analysis continues
in background and streams via SSE on the /api/analysis/{id}/stream endpoint.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from agent.auto import auto_rex
from agent.pipeline import AnalysisPipeline
from agent.server.upload_files import temporary_upload

router = APIRouter()


@router.post("/upload")
async def upload_data(
    file: UploadFile = File(...),
    options: str = Form("{}"),
):
    """Upload a dataset and auto construct the relational complex.

    Returns session ID and quick topology results immediately.
    Full analysis streams via /api/analysis/{session_id}/stream.

    Parameters

    file : uploaded file (CSV, JSON, or rexgraph format)
    options : JSON string with optional overrides:
        threshold, typing, sign, face_selection
    """
    from agent.server.app import get_store

    # Parse options
    try:
        opts = json.loads(options)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(400, "Upload options must be a JSON object") from exc
    if not isinstance(opts, dict):
        raise HTTPException(400, "Upload options must be a JSON object")

    async with temporary_upload(file) as tmp_path:
        try:
            rex = auto_rex(tmp_path, **opts)
        except Exception as exc:
            raise HTTPException(400, f"Failed to build relational complex: {exc}") from exc
    if rex is None:
        raise HTTPException(400, "Upload produced no relational complex")

    # Create session
    name = file.filename or "upload"
    meta = getattr(rex, "_agent_meta", {})

    # Run quick analysis (stages 1-3, < 1 second)
    pipeline = AnalysisPipeline(rex)
    quick_results = pipeline.run(depth="quick")

    # Save snapshot
    session = get_store().create(name=name)
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
