"""
Server-side session store.

Manages active sessions in memory with disk persistence via .rex bundles.
Sessions are created on upload and persist across server restarts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from agent.session import Session, create_session, list_sessions as _list_sessions


class SessionStore:
    """Manages all active sessions for the server."""

    def __init__(self, storage_dir: str = "~/.rexgraph-agent/sessions"):
        self.storage_dir = str(Path(storage_dir).expanduser())
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        self._active: Dict[str, Session] = {}

    def create(self, name: str = "") -> Session:
        """Create a new session."""
        session = create_session(self.storage_dir)
        if name:
            session._metadata["name"] = name
        self._active[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
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

    def list_all(self) -> List[Dict]:
        """List all sessions with metadata."""
        return _list_sessions(self.storage_dir)

    def delete(self, session_id: str):
        """Delete a session."""
        session = self.get(session_id)
        if session:
            session.delete()
            self._active.pop(session_id, None)
