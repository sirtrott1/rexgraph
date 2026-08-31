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

    @staticmethod
    def _current_workspace():
        """The workspace serving this request, or None when nothing is being scoped."""
        try:
            from agent.server.scope import current_workspace, scoping_active
            return current_workspace() if scoping_active() else None
        except Exception:
            return None

    @classmethod
    def _owned_here(cls, meta) -> bool:
        """Whether a session belongs to the workspace asking for it.

        The same rule the record store uses: an unstamped session predates ownership
        and stays visible, because stamping it retroactively would mean guessing whose
        it was. Outside a scoped request nothing is filtered at all, which is what the
        CLI and the test suite want.
        """
        ws = cls._current_workspace()
        if ws is None:
            return True
        owner = (meta or {}).get("workspace")
        return owner is None or owner == ws

    def create(self, name: str = "") -> Session:
        """Create a new session, stamped with the workspace creating it."""
        session = create_session(self.storage_dir)
        if name:
            session._metadata["name"] = name
        ws = self._current_workspace()
        if ws is not None:
            session._metadata["workspace"] = ws
        self._active[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Get a session by ID. Loads from disk if not in memory.

        A session belonging to another workspace reads as ABSENT rather than refused,
        the same answer `/rex/v1/fetch` gives for a record: saying it exists but is not
        yours turns a guessable id into a way to enumerate what other tenants hold.
        """
        if session_id in self._active:
            s = self._active[session_id]
            return s if self._owned_here(s._metadata) else None
        # Try loading from disk
        try:
            session = Session.load(session_id, self.storage_dir)
            if session.snapshots:
                if not self._owned_here(session._metadata):
                    return None
                self._active[session_id] = session
                return session
        except Exception:
            pass
        return None

    def list_all(self, *, limit: int | None = None) -> list[dict]:
        """Sessions with metadata, newest first. `limit` bounds what a control is handed."""
        rows = _list_sessions(self.storage_dir, limit=limit)
        return [r for r in rows if self._owned_here(r.get("metadata", r))]

    def delete(self, session_id: str):
        """Delete a session, if it is this workspace's to delete.

        `get` already refuses another tenant's session, so a delete of one is a no-op
        rather than a destruction. That is the whole fix: this route let any caller
        remove any session, and deletion is the one operation an owner cannot undo.
        """
        session = self.get(session_id)
        if session:
            session.delete()
            self._active.pop(session_id, None)
