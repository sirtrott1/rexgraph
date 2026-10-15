"""Append audit entries with a hash chain and optional signed anchors.

Each entry records the actor, workspace, operation, affected resources and
result, together with the digest of the previous entry. Payload contents
are excluded. Verification detects changes that leave later hashes
inconsistent. An actor who can rewrite the file and recompute the remaining
hashes can create another valid chain, so the journal also supports anchors.

An anchor records the journal length and head digest in a separate sink.
verify_against_anchors compares the journal with those earlier records.
Set REXGRAPH_AUDIT_ANCHORS to a sink the service cannot rewrite and
REXGRAPH_ANCHOR_KEY to a signing key reference. Anchors stored beside a
writable journal do not independently protect against a complete rewrite.
File permissions control writes; hashes and anchors support later checks.

The journal uses REXGRAPH_AUDIT_JOURNAL when set, otherwise audit.jsonl
under REXGRAPH_CONFIG_DIR or ~/.config/rexgraph. Entries are serialized as
JSON lines and appended with O_APPEND.
"""

from __future__ import annotations

from agent.data_paths import config_root

import contextlib
import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path

try:
    import fcntl
except ImportError:                                  # non POSIX: one process only
    fcntl = None

#: the digest recorded by the first entry, which has nothing before it
GENESIS = "0" * 64

_lock = threading.Lock()


def journal_path() -> Path:
    explicit = os.environ.get("REXGRAPH_AUDIT_JOURNAL")
    if explicit:
        return Path(explicit).expanduser()
    return config_root() / "audit.jsonl"


def _line(obj: dict) -> bytes:
    """One canonical JSON line. Sorted keys, so the same object serializes the same way."""
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       default=str) + "\n").encode("utf-8")


@contextlib.contextmanager
def _locked_append(p: Path):
    """Hold `p` exclusively for the duration of the block, yielding an append mode fd.

    Both writers in this module read a file and then append to it based on what they
    read, so the lock has to span both halves. Closing the descriptor releases it.
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock:                                      # threads inside this process
        fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if fcntl is not None:                    # and every other process
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield fd
        finally:
            os.close(fd)


def _digest(entry: dict) -> str:
    """The digest of one entry, over its content and the chain it extends.

    Sorted keys and separators fixed, so the same entry digests the same way whatever
    order the dict was built in.
    """
    body = json.dumps(entry, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _last_line(path: Path, block: int = 4096) -> str:
    """The last complete line, read from the end of the file.

    Read backwards because this runs under the lock on every append: walking the whole
    journal to find its own tail would make writing the trail quadratic in its length.
    """
    if not path.is_file():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        pos = fh.tell()
        data = b""
        while pos > 0:
            step = min(block, pos)
            pos -= step
            fh.seek(pos)
            data = fh.read(step) + data
            trimmed = data.rstrip(b"\n")
            cut = trimmed.rfind(b"\n")
            if cut != -1:
                return trimmed[cut + 1:].decode("utf-8", "replace")
        return data.rstrip(b"\n").decode("utf-8", "replace")


def _read_head(path: Path) -> str:
    """The digest of the last entry on disk, or GENESIS for an empty trail."""
    last = _last_line(path)
    if not last.strip():
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
        with _locked_append(p) as fd:
            # The head is read from the file while the lock is held, never from a
            # process local cache. Two processes stamping `prev` from their own cached
            # head both extended the same entry, so the chain forked and `verify`
            # reported a break with nothing tampered, which makes a real break
            # indistinguishable from ordinary concurrency.
            entry["prev"] = _read_head(p)
            entry["digest"] = _digest(entry)
            os.write(fd, _line(entry))
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
    """No head is cached any more; every append reads it from the file under the lock.

    Kept because it is the reset seam ten test modules already call, and because it
    stays correct if this module gains process state again.
    """
    return None


#: names the reference to the anchor key, not the key. Absent means anchors are unsigned.
ANCHOR_KEY_REF_ENV = "REXGRAPH_ANCHOR_KEY"


def anchor_path() -> Path:
    """Return the configured anchor sink or a path beside the journal.

    REXGRAPH_AUDIT_ANCHORS overrides the default audit.anchors.jsonl path.
    For independent detection of journal rewrites, configure a sink that
    the service cannot rewrite.
    """
    explicit = os.environ.get("REXGRAPH_AUDIT_ANCHORS")
    if explicit:
        return Path(explicit).expanduser()
    return journal_path().with_name("audit.anchors.jsonl")


def _anchor_key() -> str:
    """The anchor key, resolved from the reference the environment names."""
    from agent.secrets import resolve_ref
    return resolve_ref(os.environ.get(ANCHOR_KEY_REF_ENV, ""))


def _anchor_mac(body: dict, key: str) -> str:
    if not key:
        return ""
    return hmac.new(key.encode("utf-8"), _line(body), hashlib.sha256).hexdigest()


def anchor(path: Path | None = None, sink: Path | None = None) -> dict:
    """Record the verified journal length and head digest in an anchor.

    The anchor can later detect a journal that no longer contains the
    recorded prefix. Independent storage is required to detect a rewrite
    by an actor who can change both the journal and its local anchors.

    If REXGRAPH_ANCHOR_KEY resolves to a key, the anchor includes an HMAC.
    The returned signed field records whether a key was used.
    """
    p = path or journal_path()
    state = verify(p)
    body = {"ts": time.time(), "journal": str(p),
            "n_entries": state["n_entries"], "head": state["head"]}
    key = _anchor_key()
    out = dict(body, mac=_anchor_mac(body, key), signed=bool(key),
               valid_when_taken=state["valid"])
    with _locked_append(sink or anchor_path()) as fd:
        os.write(fd, _line(out))
    return out


def read_anchors(sink: Path | None = None) -> list[dict]:
    """Every anchor taken, oldest first. An unreadable line is skipped, not fatal."""
    src = sink or anchor_path()
    if not src.is_file():
        return []
    out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        with contextlib.suppress(json.JSONDecodeError):
            out.append(json.loads(line))
    return out


def verify_against_anchors(path: Path | None = None,
                           sink: Path | None = None) -> dict:
    """Check the trail against every anchor taken of it.

    Reports the OLDEST anchor the journal disagrees with, because that is where the
    record stopped matching what was witnessed and everything after it is already
    suspect. Three ways to fail, apart because they mean different things: an anchor
    whose MAC does not verify was forged or the key changed; a trail shorter than an
    anchor witnessed was truncated; a head that differs at a length that was witnessed
    means the trail was rewritten behind the anchor.

    With no anchors this reduces to `verify`, and says so, because a chain that has
    never been witnessed cannot detect its own wholesale rewrite.
    """
    p = path or journal_path()
    state = verify(p)
    anchors = sorted(read_anchors(sink), key=lambda a: int(a.get("n_entries", 0)))
    key = _anchor_key()
    heads = [GENESIS]
    for entry in read(p):
        heads.append(str(entry.get("digest") or ""))

    for a in anchors:
        body = {k: a[k] for k in ("ts", "journal", "n_entries", "head") if k in a}
        expected = _anchor_mac(body, key)
        if expected and not hmac.compare_digest(expected, str(a.get("mac") or "")):
            return {"valid": False, "reason": "an anchor was forged or the key changed",
                    "anchor": a, "n_anchors": len(anchors), "chain": state}
        n = int(a.get("n_entries", 0))
        if n >= len(heads):
            return {"valid": False,
                    "reason": "the trail is shorter than an anchor witnessed",
                    "anchor": a, "n_anchors": len(anchors), "chain": state}
        if heads[n] != a.get("head"):
            return {"valid": False,
                    "reason": "the trail was rewritten behind an anchor",
                    "anchor": a, "n_anchors": len(anchors), "chain": state}

    return {"valid": bool(state["valid"]), "reason": state["reason"],
            "anchor": None, "n_anchors": len(anchors), "chain": state}
