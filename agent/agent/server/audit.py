"""
agent.server.audit: an append-only trail where an edited entry stops verifying.

A log that records what happened answers "what happened" only if nothing can quietly
edit it afterwards. A plain journal cannot tell a line that was always there from one
written later, so the interesting case, someone covering a change, is the case it
misses.

Each entry carries the digest of the entry before it. The digests form a chain, so
altering entry k changes its digest, which is recorded in k+1, and every entry after it
disagrees. Removing an entry breaks the same link. An attacker who can write the file
can still rewrite the whole tail and recompute every digest, and `verify` then reports a
clean chain, so the chain alone cannot detect that.

`anchor` is the other half. It witnesses how long the trail is and what its head is into
a separate sink, so a rewrite has to also change a record the rewriter does not hold, and
`verify_against_anchors` reports the oldest anchor the trail stopped agreeing with. The
sink is the whole point: anchors beside the journal are rewritten with it. Sign them by
naming a key reference in REXGRAPH_ANCHOR_KEY, and put them somewhere this service
cannot write, via REXGRAPH_AUDIT_ANCHORS.

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
except ImportError:                                  # non-POSIX: one process only
    fcntl = None

#: the digest recorded by the first entry, which has nothing before it
GENESIS = "0" * 64

_lock = threading.Lock()


def journal_path() -> Path:
    explicit = os.environ.get("REXGRAPH_AUDIT_JOURNAL")
    if explicit:
        return Path(explicit)
    base = Path(os.environ.get("REXGRAPH_CONFIG_DIR",
                               Path.home() / ".config" / "rexgraph"))
    return base / "audit.jsonl"


def _line(obj: dict) -> bytes:
    """One canonical JSON line. Sorted keys, so the same object serializes the same way."""
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       default=str) + "\n").encode("utf-8")


@contextlib.contextmanager
def _locked_append(p: Path):
    """Hold `p` exclusively for the duration of the block, yielding an append-mode fd.

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
            # process-local cache. Two processes stamping `prev` from their own cached
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
    """Where anchors are appended.

    The default sits beside the journal, which is convenient and weak: whoever can
    rewrite the trail can rewrite anchors in the same directory. Point
    REXGRAPH_AUDIT_ANCHORS at a sink this service cannot rewrite, another host, an
    append-only mount, object storage under a retention lock, or an anchor proves
    nothing the journal does not already prove on its own.
    """
    explicit = os.environ.get("REXGRAPH_AUDIT_ANCHORS")
    if explicit:
        return Path(explicit)
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
    """Witness how long the trail is and what its head is, so a later rewrite shows.

    The chain proves no entry was edited in place. It cannot prove the tail was not
    rewritten wholesale, because whoever rewrites it recomputes every digest from the
    point they changed. An anchor is the missing half: a statement kept somewhere the
    service cannot reach about where the trail stood at one moment, so rewriting history
    now also requires changing a record the rewriter does not hold.

    Signing is what stops an anchor from being rewritten alongside the journal when the
    sink turns out to be reachable after all. Unsigned anchors still catch an accident
    and a careless attacker, and the return value says which kind was written.
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
