"""
agent.server.audit: an append-only trail where an edited entry stops verifying.

A log that records what happened answers "what happened" only if nothing can quietly
edit it afterwards. A plain journal cannot tell a line that was always there from one
written later, so the interesting case, someone covering a change, is the case it
misses.

Each entry carries the digest of the entry before it. The digests form a chain, so
altering entry k changes its digest, which is recorded in k+1, and every entry after it
disagrees. Removing an entry breaks the same link. An attacker who can write the file
can still rewrite the whole tail, which is why the head digest is worth reading off the
box: verification is local, but anchoring is not.

That is tamper EVIDENCE, not tamper prevention, and the distinction is the point. The
trail proves a record was not edited after it was written; keeping the file writable
only by the service is what stops the write in the first place.

What is recorded is who, what, which workspace, what it touched, and what came back,
never the content itself. A trail holding payloads is a second copy of the data with
different access rules, which is a leak with an audit trail's name on it.

Journal: $REXGRAPH_AUDIT_JOURNAL, else $REXGRAPH_CONFIG_DIR/audit.jsonl, else
~/.config/rexgraph/audit.jsonl. One JSON object per line, opened O_APPEND so
concurrent writers interleave whole lines rather than corrupting each other.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

#: the digest recorded by the first entry, which has nothing before it
GENESIS = "0" * 64

_lock = threading.Lock()
_head: str | None = None


def journal_path() -> Path:
    explicit = os.environ.get("REXGRAPH_AUDIT_JOURNAL")
    if explicit:
        return Path(explicit)
    base = Path(os.environ.get("REXGRAPH_CONFIG_DIR",
                               Path.home() / ".config" / "rexgraph"))
    return base / "audit.jsonl"


def _digest(entry: dict) -> str:
    """The digest of one entry, over its content and the chain it extends.

    Sorted keys and separators fixed, so the same entry digests the same way whatever
    order the dict was built in.
    """
    body = json.dumps(entry, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _read_head(path: Path) -> str:
    """The digest of the last entry on disk, or GENESIS for an empty trail."""
    if not path.is_file():
        return GENESIS
    last = ""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = line
    if not last:
        return GENESIS
    try:
        return str(json.loads(last).get("digest") or GENESIS)
    except json.JSONDecodeError:
        return GENESIS


def record(action: str, *, user: str = "", workspace: str = "default",
           target: str = "", outcome: str = "ok", detail: dict | None = None,
           path: Path | None = None) -> dict:
    """Append one entry and return it, digest included.

    Never raises: an operation that succeeded should not be reported as failed because
    its trail could not be written. A trail that cannot be written is itself worth
    noticing, so the failure goes to the logger.
    """
    global _head
    p = path or journal_path()
    entry = {
        "ts": time.time(),
        "action": str(action),
        "user": str(user or "local"),
        "workspace": str(workspace or "default"),
        "target": str(target or "")[:512],
        "outcome": str(outcome),
        "detail": detail or {},
        "pid": os.getpid(),
    }
    try:
        with _lock:
            if _head is None or path is not None:
                _head = _read_head(p)
            entry["prev"] = _head
            entry["digest"] = _digest(entry)
            p.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry, sort_keys=True, separators=(",", ":"),
                              default=str) + "\n"
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
            _head = entry["digest"]
    except OSError:
        import logging
        logging.getLogger(__name__).warning(
            "audit entry for %s could not be written", action, exc_info=True)
    return entry


def read(path: Path | None = None, *, workspace: str = "", limit: int = 0) -> list[dict]:
    """The trail, oldest first, optionally narrowed to one workspace."""
    p = path or journal_path()
    if not p.is_file():
        return []
    out = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if workspace and entry.get("workspace") != workspace:
                continue
            out.append(entry)
    return out[-limit:] if limit else out


def verify(path: Path | None = None) -> dict:
    """Walk the chain and report the first entry that disagrees with it.

    Two ways to fail, reported apart because they mean different things: a digest that
    does not match the entry means the entry was edited, and a `prev` that does not
    match the entry before it means one was removed or spliced in.
    """
    entries = read(path)
    prev = GENESIS
    for i, entry in enumerate(entries):
        recorded = entry.get("digest")
        if entry.get("prev") != prev:
            return {"valid": False, "n_entries": len(entries), "broken_at": i,
                    "reason": "an entry was removed or inserted",
                    "head": prev}
        body = {k: v for k, v in entry.items() if k != "digest"}
        if _digest(body) != recorded:
            return {"valid": False, "n_entries": len(entries), "broken_at": i,
                    "reason": "an entry was edited after it was written",
                    "head": prev}
        prev = str(recorded)
    return {"valid": True, "n_entries": len(entries), "broken_at": None,
            "reason": "", "head": prev}


def head(path: Path | None = None) -> str:
    """The current head digest, the value to anchor somewhere off the box."""
    return _read_head(path or journal_path())


def reset_cache() -> None:
    """Drop the cached head so the next append re-reads the file. For tests."""
    global _head
    _head = None
