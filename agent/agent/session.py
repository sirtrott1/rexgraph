"""
Session state as a temporal trajectory of analysis states.

Each user interaction (upload, query, parameter change) creates a new
timestep. Returning to a previous state is loading that snapshot.
Sessions persist as .rex bundles on disk.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SnapshotUnreadable(RuntimeError):
    """A session snapshot exists but its bundle cannot be loaded.

    Usually a bundle written by an older version of the format. It is a property of
    the stored session, so callers report it rather than crash on it.
    """


@dataclass
class Snapshot:
    """One timestep in a session."""
    step: int
    timestamp: float
    action: str                    # 'upload', 'analyze', 'reconfig', 'explore'
    params: dict[str, Any]         # parameters used at this step
    rex_path: str | None        # path to serialized .rex bundle (if saved)
    results: dict | None        # cached analysis results (in memory)
    summary: str = ""              # one-line summary of this step


class Session:
    """A user's analysis session, stored as a sequence of snapshots.

    Each interaction creates a new snapshot. The RexGraph at each
    snapshot is the complete analysis state. Previous states are
    recoverable by loading the corresponding .rex bundle.
    """

    def __init__(self, session_id: str, storage_dir: str):
        self.session_id = session_id
        self.storage_dir = Path(storage_dir)
        self.session_dir = self.storage_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.snapshots: list[Snapshot] = []
        self.current_step: int = -1
        self._current_rex = None
        self._metadata: dict[str, Any] = {
            "session_id": session_id,
            "created": time.time(),
            "name": "",
        }

    @property
    def current_rex(self):
        return self._current_rex

    @current_rex.setter
    def current_rex(self, rex):
        self._current_rex = rex

    def add_snapshot(
        self,
        rex,
        action: str,
        params: dict | None = None,
        results: dict | None = None,
        summary: str = "",
    ):
        """Record a new analysis state.

        Parameters
        ----------
        rex : RexGraph
            The current analysis state.
        action : str
            What the user did: 'upload', 'analyze', 'reconfig', 'explore'.
        params : dict, optional
            Parameters used (threshold, typing, etc.).
        results : dict, optional
            Analysis results to cache with this snapshot.
        summary : str
            One-line description.
        """
        step = len(self.snapshots)

        # Serialize the rex to disk
        rex_filename = f"snapshot_{step:04d}.rex"
        rex_path = str(self.session_dir / rex_filename)
        try:
            from rexgraph.io import save_rex
            save_rex(rex_path, rex)
        except Exception:
            rex_path = None

        snapshot = Snapshot(
            step=step,
            timestamp=time.time(),
            action=action,
            params=params or {},
            rex_path=rex_path,
            results=results,
            summary=summary or f"{action} (step {step})",
        )
        self.snapshots.append(snapshot)
        self.current_step = step
        self._current_rex = rex
        self._save_index()

    def at(self, step: int):
        """Return the RexGraph at a specific timestep.

        Loads from the serialized bundle on disk.
        """
        if step < 0 or step >= len(self.snapshots):
            raise IndexError(f"Step {step} out of range [0, {len(self.snapshots)})")

        snapshot = self.snapshots[step]
        if snapshot.rex_path is None:
            raise ValueError(f"No serialized rex at step {step}")

        from rexgraph.io import load_rex
        try:
            rex = load_rex(snapshot.rex_path)
        except Exception as e:
            # A snapshot that will not load is this session's problem, not a fault
            # in whatever is asking for it. Naming the session and step turns an
            # opaque 500 several frames away into something actionable.
            raise SnapshotUnreadable(
                f"session {self.session_id!r} step {step} cannot be read from "
                f"{snapshot.rex_path}: {e}") from e
        self.current_step = step
        self._current_rex = rex
        return rex

    def current(self):
        """Return the current RexGraph."""
        if self._current_rex is None and self.snapshots:
            return self.at(self.current_step)
        return self._current_rex

    def history(self) -> list[dict]:
        """Return the session history as a list of step summaries."""
        return [
            {
                "step": s.step,
                "timestamp": s.timestamp,
                "action": s.action,
                "summary": s.summary,
                "has_rex": s.rex_path is not None,
            }
            for s in self.snapshots
        ]

    def info(self) -> dict:
        """Return session metadata."""
        return {
            **self._metadata,
            "n_steps": len(self.snapshots),
            "current_step": self.current_step,
            "history": self.history(),
        }

    # Persistence

    def _save_index(self):
        """Persist the session index (metadata + snapshot list)."""
        index = {
            "metadata": self._metadata,
            "snapshots": [
                {
                    "step": s.step,
                    "timestamp": s.timestamp,
                    "action": s.action,
                    "params": s.params,
                    "rex_path": s.rex_path,
                    "summary": s.summary,
                }
                for s in self.snapshots
            ],
            "current_step": self.current_step,
        }
        index_path = self.session_dir / "session_index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, default=str)

    @classmethod
    def load(cls, session_id: str, storage_dir: str) -> Session:
        """Load a session from disk."""
        session = cls(session_id, storage_dir)
        index_path = session.session_dir / "session_index.json"
        if not index_path.exists():
            return session

        with open(index_path) as f:
            index = json.load(f)

        session._metadata = index.get("metadata", session._metadata)
        session.current_step = index.get("current_step", -1)

        for s in index.get("snapshots", []):
            snapshot = Snapshot(
                step=s["step"],
                timestamp=s["timestamp"],
                action=s["action"],
                params=s.get("params", {}),
                rex_path=s.get("rex_path"),
                results=None,
                summary=s.get("summary", ""),
            )
            session.snapshots.append(snapshot)

        return session

    def delete(self):
        """Delete this session and all its data from disk."""
        import shutil
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir)


def create_session(storage_dir: str = "~/.rexgraph-agent/sessions") -> Session:
    """Create a new session with a unique ID."""
    storage = str(Path(storage_dir).expanduser())
    session_id = str(uuid.uuid4())[:8]
    return Session(session_id, storage)


def list_sessions(storage_dir: str = "~/.rexgraph-agent/sessions") -> list[dict]:
    """List all saved sessions."""
    storage = Path(storage_dir).expanduser()
    if not storage.exists():
        return []

    sessions = []
    for d in sorted(storage.iterdir()):
        if d.is_dir() and (d / "session_index.json").exists():
            try:
                s = Session.load(d.name, str(storage))
                sessions.append(s.info())
            except Exception:
                pass
    return sessions
