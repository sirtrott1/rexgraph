"""
Session routes: list, inspect, navigate history, delete.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/sessions")
async def list_sessions(limit: int = 200):
    """Saved sessions, NEWEST FIRST, bounded.

    Unbounded and id-sorted before, which put a freshly recorded session at index 1177 of
    5278 on a real install: in the list, and unfindable in a control.
    """
    from agent.server.app import get_store
    return get_store().list_all(limit=(None if limit <= 0 else int(limit)))


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Return session metadata and history."""
    from agent.server.app import get_store
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(404, f"Session not found: {session_id}")
    return session.info()


@router.post("/sessions/{session_id}/goto/{step}")
async def goto_step(session_id: str, step: int):
    """Return to a previous analysis state.

    Loads the RexGraph snapshot at the given step. All cached
    properties from that point are immediately available.
    """
    from agent.server.app import get_store
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(404, f"Session not found: {session_id}")

    try:
        rex = session.at(step)
        return {
            "status": "ok",
            "step": step,
            "nV": rex.nV,
            "nE": rex.nE,
            "nF": rex.nF,
        }
    except (IndexError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its data."""
    from agent.server.app import get_store
    store = get_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session not found: {session_id}")
    store.delete(session_id)
    return {"status": "deleted", "session_id": session_id}
