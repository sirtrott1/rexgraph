"""
Analysis routes: retrieve computed results and stream progressive updates.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from agent.pipeline import AnalysisPipeline

router = APIRouter()


@router.get("/analysis/{session_id}")
async def get_analysis(session_id: str, depth: str = "standard"):
    """Return the full analysis for a session.

    If the analysis hasn't been computed at the requested depth yet,
    runs it synchronously and returns the results.
    """
    from agent.server.app import get_store

    store = get_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    rex = session.current()
    if rex is None:
        raise HTTPException(status_code=400, detail="No data in session")

    # Check if we already have results at this depth
    current_snapshot = session.snapshots[session.current_step] if session.snapshots else None
    if current_snapshot and current_snapshot.results:
        # If we have results and they include relational data, return them
        if depth == "quick" or "relational" in current_snapshot.results:
            return current_snapshot.results

    # Run analysis
    pipeline = AnalysisPipeline(rex)
    results = pipeline.run(depth=depth)

    # Cache the results on the current snapshot
    if current_snapshot:
        current_snapshot.results = results

    return results


@router.get("/analysis/{session_id}/stream")
async def stream_analysis(session_id: str, depth: str = "standard"):
    """SSE endpoint: stream analysis stages as they complete.

    The frontend connects to this after upload to receive progressive
    results. Each stage fires an SSE event with the stage name and data.
    """
    from agent.server.app import get_store
    from agent.server.stream import stream_pipeline

    store = get_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    rex = session.current()
    if rex is None:
        raise HTTPException(status_code=400, detail="No data in session")

    pipeline = AnalysisPipeline(rex)

    return StreamingResponse(
        stream_pipeline(pipeline, depth=depth),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/analysis/{session_id}/table")
async def analysis_table(session_id: str, kind: str = "character"):
    """A per-cell analysis table, through the canonical writer for its kind.

    `character` is the per-edge structural character; `persistence` is the diagram of
    the dimension filtration. Both come back as parquet, which is what the SQL bridge
    and the warehouse already read, rather than as a JSON list nothing else consumes.
    """
    from agent.server.app import get_store
    from agent.server.artifacts import character_file, persistence_file

    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    rex = session.current()
    if rex is None:
        raise HTTPException(400, "No data in session")

    kind = str(kind).strip().lower()
    if kind == "character":
        return character_file(rex, session_id)
    if kind == "persistence":
        filt_v, filt_e, filt_f = rex.filtration("dimension")
        return persistence_file(rex.persistence(filt_v, filt_e, filt_f), session_id)
    raise HTTPException(400, f"unknown table {kind!r}. Available: character, "
                             "persistence")
