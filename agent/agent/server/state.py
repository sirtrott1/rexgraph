"""
Server-side session store.

Manages active sessions in memory with disk persistence via .rex bundles.
Sessions are created on upload and persist across server restarts.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent.session import Session, create_session
from agent.session import list_sessions as _list_sessions


class SessionStore:
    """Manages all active sessions for the server."""

    #: where sessions live unless told otherwise. Overridable by REXGRAPH_SESSION_DIR
    #: so a test can hold sessions somewhere of its own: the suite had written 1065 of
    #: them into a real install in a day, 397 from one fixture, which buried the user's
    #: own work in a list it shared.
    DEFAULT_DIR = "~/.rexgraph-agent/sessions"

    def __init__(self, storage_dir: str | None = None):
        storage_dir = storage_dir or os.environ.get(
            "REXGRAPH_SESSION_DIR", self.DEFAULT_DIR)
        self.storage_dir = str(Path(storage_dir).expanduser())
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        self._active: dict[str, Session] = {}

    def create(self, name: str = "") -> Session:
        """Create a new session."""
        session = create_session(self.storage_dir)
        if name:
            session._metadata["name"] = name
        self._active[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Get a session by ID. Loads from disk if not in memory."""
        if session_id in self._active:
            return self._active[session_id]
        # Try loading from disk
        try:
            session = Session.load(session_id, self.storage_dir)
            if session.snapshots:
                self._active[session_id] = session
                return session
        except Exception:
            pass
        return None

    def list_all(self, *, limit: int | None = None) -> list[dict]:
        """Sessions with metadata, newest first. `limit` bounds what a control is handed."""
        return _list_sessions(self.storage_dir, limit=limit)

    def delete(self, session_id: str):
        """Delete a session."""
        session = self.get(session_id)
        if session:
            session.delete()
            self._active.pop(session_id, None)
