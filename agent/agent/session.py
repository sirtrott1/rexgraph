"""
Session state as a temporal trajectory of analysis states.

Each user interaction (upload, query, parameter change) creates a new
timestep. Returning to a previous state is loading that snapshot.
Sessions persist as .rcbd bundles on disk. Bundles written before the rename
carry a .rex suffix and are still read.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .formats import BUNDLE_SUFFIX
from .data_paths import sessions_root


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
    rex_path: str | None        # path to serialized bundle (if saved)
    results: dict | None        # cached analysis results (in memory)
    summary: str = ""              # one line summary of this step


class Session:
    """A user's analysis session, stored as a sequence of snapshots.

    Each interaction creates a new snapshot. The RexGraph at each
    snapshot is the complete analysis state. Previous states are
    recoverable by loading the corresponding .rcbd bundle.
    """

    def __init__(self, session_id: str, storage_dir: str, *, _create: bool = True):
        if (not isinstance(session_id, str) or not session_id or session_id in {".", ".."}
                or any(char in session_id for char in ("/", "\\", ":", "\x00"))):
            raise ValueError("session ID must be a single directory name")
        self.session_id = session_id
        self.storage_dir = Path(storage_dir).expanduser()
        self.session_dir = self.storage_dir / session_id
        if self.session_dir.is_symlink():
            raise ValueError("session directories cannot be symbolic links")
        if _create:
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
        from rexgraph.io import save_rcbd
        from rexgraph.io.publication import publication_lock

        with publication_lock(self.session_dir):
            self._assert_saved_history()
            step = len(self.snapshots)
            rex_filename = f"snapshot_{step:04d}_{uuid.uuid4().hex}{BUNDLE_SUFFIX}"
            rex_path = str(self.session_dir / rex_filename)
            save_rcbd(rex_path, rex)
            snapshot = Snapshot(
                step=step,
                timestamp=time.time(),
                action=action,
                params=params or {},
                rex_path=rex_path,
                results=results,
                summary=summary or f"{action} (step {step})",
            )
            # Publish history before exposing the new state to readers. A failed
            # index write retains the graph bundle for recovery and leaves this
            # object's previous history intact.
            self._save_index(snapshots=[*self.snapshots, snapshot], current_step=step)
            self.snapshots.append(snapshot)
            self.current_step = step
            self._current_rex = rex

    def at(self, step: int):
        """Return the RexGraph at a specific timestep.

        Loads from the serialized bundle on disk.
        """
        if step < 0 or step >= len(self.snapshots):
            raise IndexError(f"Step {step} out of range [0, {len(self.snapshots)})")

        snapshot = self.snapshots[step]
        if snapshot.rex_path is None:
            raise ValueError(f"No serialized rex at step {step}")

        from rexgraph.io import load_rcbd
        path = Path(snapshot.rex_path)
        local = self.session_dir / path.name
        # Absolute paths in older indices remain readable after a session is moved.
        if local.exists():
            path = local
        elif not path.is_absolute():
            path = self.session_dir / path
        try:
            rex = load_rcbd(path)
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

    def _assert_saved_history(self):
        """Refuse to overwrite a history changed by another loaded session."""
        path = self.session_dir / "session_index.json"
        try:
            with path.open(encoding="utf-8") as stream:
                index = json.load(stream)
        except FileNotFoundError:
            saved = []
        else:
            saved = [(row.get("step"), row.get("timestamp"), row.get("rex_path"))
                     for row in index["snapshots"]]
        current = [(row.step, row.timestamp, row.rex_path) for row in self.snapshots]
        if saved != current:
            raise RuntimeError("session history changed on disk; reload the session before writing")

    def _save_index(self, *, snapshots=None, current_step=None):
        """Persist the session index (metadata + snapshot list)."""
        from rexgraph.io.publication import publication_lock, staged_publication
        from rexgraph.io._compat import dumps
        snapshots = self.snapshots if snapshots is None else snapshots
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
                for s in snapshots
            ],
            "current_step": self.current_step if current_step is None else current_step,
        }
        index_path = self.session_dir / "session_index.json"
        with publication_lock(self.session_dir):
            self._assert_saved_history()
            with staged_publication(index_path) as staged:
                with staged.open("w", encoding="utf-8") as stream:
                    stream.write(dumps(index, nan="null", indent=2))

    @classmethod
    def load(cls, session_id: str, storage_dir: str) -> Session:
        """Load a session from disk."""
        session = cls(session_id, storage_dir, _create=False)
        index_path = session.session_dir / "session_index.json"
        with open(index_path, encoding="utf-8") as f:
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


def create_session(storage_dir: str | None = None) -> Session:
    """Create a new session with a unique ID."""
    storage = str(Path(storage_dir).expanduser() if storage_dir else sessions_root())
    Path(storage).mkdir(parents=True, exist_ok=True)
    for _ in range(32):
        session_id = str(uuid.uuid4())[:8]
        try:
            (Path(storage) / session_id).mkdir()
        except FileExistsError:
            continue
        return Session(session_id, storage)
    raise FileExistsError("could not allocate a unique session directory")


def list_sessions(storage_dir: str | None = None, *,
                  limit: int | None = None) -> list[dict]:
    """List saved sessions by descending timestamp, optionally limiting the count.

    Read timestamps from session indexes, sort their paths, then load the
    selected sessions. A missing storage directory returns an empty list.
    """
    storage = Path(storage_dir).expanduser() if storage_dir else sessions_root()
    if not storage.exists():
        return []

    dated = []
    for d in storage.iterdir():
        index = d / "session_index.json"
        if not (d.is_dir() and index.exists()):
            continue
        try:
            meta = json.loads(index.read_text()).get("metadata", {})
            created = float(meta.get("created") or 0.0)
        except Exception:
            created = 0.0
        dated.append((created, d.name))
    dated.sort(reverse=True)
    if limit is not None:
        dated = dated[:int(limit)]

    sessions = []
    for _created, name in dated:
        try:
            sessions.append(Session.load(name, str(storage)).info())
        except Exception:
            pass
    return sessions
