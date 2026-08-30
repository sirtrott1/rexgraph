"""agent.activity: the activity log + model-usage registry, backed by a local journal.

Every action by every entity, across granularities (network / hive / team / worker / model), is
recorded here as a timestamped event. And every time a model is used, a use is opened and closed, so
the registry knows when a model was instantiated, what it is being used for, how long it has run, and
and, critically, how many things are using it CONCURRENTLY right now. This is the real data the logs,
the runtime readouts, and the usage portal read from; nothing in the UI is faked on top of it.

The log is process-local memory PLUS a write-through append-only journal on disk (JSONL). That file
is the event bus: any local process (a CLI, a worker, another agent) that records an event appends a
line: no server, no HTTP, no token needed to WRITE. A process that wants to OBSERVE the whole machine
(the web server) warm-loads the journal tail on startup and then tails it, folding every other
process's events into its own log and pushing them to the live (SSE) UI. So a `rexgraph-*` command in
one terminal shows up live in the GUI running in another: they share the file, not a socket.

    activity.record("worker:coder", "dispatch", detail={"query": "..."})
    activity.record("worker:mule", "deliver", on="hive:beta", flow="write")   # an oriented act
    h = activity.open_use("qwen-7b", "collaborate", by="hive:alpha"); ...; activity.close_use(h)
    activity.get_log().events(scope="worker")     # the log, filtered (this process + tailed peers)
    activity.get_log().usage()                    # per-model: instantiated / runtime / concurrent uses

Journal location: $REXGRAPH_ACTIVITY_JOURNAL, else $REXGRAPH_CONFIG_DIR/activity.jsonl, else
~/.config/rexgraph/activity.jsonl. Set the env to "off" (or empty) to disable journaling for a process.
"""
from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCOPES = ("network", "hive", "team", "worker", "model")

# A per-process id stamped on every journal line, so a tailer can tell its OWN writes (already in its
# in-memory log) from a peer process's writes (which it must fold in). Random per process start.
_SRC = uuid.uuid4().hex[:8]

_OFF_VALUES = {"", "off", "none", "0", "false", "no"}


def _scope_of(entity: str) -> str:
    """Infer the granularity from an entity id like 'hive:alpha' / 'worker:coder' / 'model:x'."""
    head = (entity or "").split(":", 1)[0]
    return head if head in _SCOPES else ("network" if entity == "network" else "worker")


def _journal_default() -> Path | None:
    """Where the journal lives, honoring the opt-out. Mirrors the config-dir convention used by auth."""
    env = os.environ.get("REXGRAPH_ACTIVITY_JOURNAL")
    if env is not None:
        if env.strip().lower() in _OFF_VALUES:
            return None
        return Path(env).expanduser()
    if os.environ.get("REXGRAPH_NO_JOURNAL", "").strip().lower() in ("1", "true", "yes"):
        return None
    cfg = os.environ.get("REXGRAPH_CONFIG_DIR")
    base = Path(cfg).expanduser() if cfg else (Path.home() / ".config" / "rexgraph")
    return base / "activity.jsonl"


#: How an act is oriented against the thing it acted on. `write` sends the actor's
#: sign into the object, `read` sends the object's into the actor. It is the only
#: content the boundary of an act has: an entity and a verb say that something
#: happened, and cannot say which way it went, so nothing composes and no cycle can
#: be read as consistent or not.
FLOWS = ("read", "write")


@dataclass
class Event:
    ts: float
    entity: str          # 'network' | 'hive:<n>' | 'team:<n>' | 'worker:<n>' | 'model:<n>'
    scope: str           # network | hive | team | worker | model
    action: str
    detail: dict[str, Any] = field(default_factory=dict)
    #: WHAT was acted on, and WHICH WAY. Both empty is an unoriented event, which is
    #: every event recorded before this and every one where the pair is not known.
    #: `on` is one name, or SEVERAL: an act over k participants is one k-ary relation
    #: and not k acts, so a carrier writing to three destinations says so in one event.
    on: str | list = ""
    flow: str = ""       # read | write

    @property
    def oriented(self) -> bool:
        return bool(self.on) and self.flow in FLOWS

    def public(self) -> dict:
        out = {"ts": round(self.ts, 3), "entity": self.entity, "scope": self.scope,
               "action": self.action, "detail": self.detail}
        # only when known, so a journal line for an unoriented event is byte-identical
        # to the ones already on disk and an old reader is unaffected
        if self.on:
            out["on"] = self.on
        if self.flow:
            out["flow"] = self.flow
        return out


class ActivityLog:
    """Append-only event log + open/closed model uses, thread- AND process-safe.

    In-process concurrency (workers run on threads) is guarded by a lock; cross-process concurrency
    (many `rexgraph-*` processes writing the one journal) rides on POSIX O_APPEND, which makes each
    single small write() atomic, so lines never interleave.
    """

    def __init__(self, cap: int = 8000):
        self._events: deque = deque(maxlen=cap)
        self._uses: dict[Any, dict] = {}          # int handle (own live uses) | (src, handle) (folded)
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._subscribers: list = []              # callables(event_public_dict) - the live push channel
        # journal / tailer
        self._journal: Path | None = None
        self._jfd: int | None = None
        self._wlock = threading.Lock()            # serialize journal writes (separate from the data lock)
        self._tail_thread: threading.Thread | None = None
        self._tail_stop: threading.Event | None = None
        self._tail_from: int = 0
        # append-only journaling is on by default so ANY process's events reach a watching server;
        # warm-load + tailing (observing peers) stay opt-in (the server turns them on). Never fatal.
        try:
            self.enable_journal(warm=False, tail=False)
        except Exception:
            self._journal = None

    #### live push channel
    def subscribe(self, fn) -> None:
        with self._lock:
            self._subscribers.append(fn)

    def unsubscribe(self, fn) -> None:
        with self._lock:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

    def _publish(self, pub: dict) -> None:
        """Fan one event out to live subscribers, OUTSIDE the data lock (a subscriber must not block
        the recorder). Used by both local records and folded-in peer events."""
        with self._lock:
            subs = list(self._subscribers)
        for fn in subs:
            with contextlib.suppress(Exception):
                fn(pub)

    #### recording
    def record(self, entity: str, action: str, *, scope: str = "", detail: dict | None = None,
               on: str | list = "", flow: str = "") -> Event:
        """Append one event, oriented when the caller knows the pair.

        `on` names what was acted on and `flow` says which way, so an act becomes a
        relation with a boundary rather than a label with a timestamp. Both are optional
        and default to the unoriented event this always recorded. An unrecognised `flow`
        is dropped rather than raised: the log must never be the reason a caller fails,
        and a bad orientation is better absent than believed."""
        if flow and flow not in FLOWS:
            logger.debug("dropping unknown flow %r on %s/%s", flow, entity, action)
            flow = ""
        ev = Event(time.time(), entity, scope or _scope_of(entity), action, detail or {},
                   on=on, flow=flow)
        pub = ev.public()
        line = None
        with self._lock:
            self._events.append(ev)
            if self._jfd is not None:
                line = json.dumps(dict(pub, src=_SRC), separators=(",", ":"))
        if line is not None:
            self._write_journal(line)
        self._publish(pub)
        return ev

    def _write_journal(self, line: str) -> None:
        fd = self._jfd
        if fd is None:
            return
        try:
            with self._wlock:                      # one write() per line -> atomic append under O_APPEND
                os.write(fd, (line + "\n").encode("utf-8"))
        except Exception:
            pass                                   # journaling must never break the caller

    def events(self, *, entity: str | None = None, scope: str | None = None,
               action: str | None = None, since: float | None = None,
               limit: int = 200) -> list[dict]:
        """The log, newest first. `entity` matches exactly or as a prefix (a hive covers its team/
        workers if you pass 'hive:alpha')."""
        with self._lock:
            evs = list(self._events)
        out = []
        for e in evs:
            if entity is not None and not (e.entity == entity or e.entity.startswith(entity + ":")):
                continue
            if scope is not None and e.scope != scope:
                continue
            if action is not None and e.action != action:
                continue
            if since is not None and e.ts < since:
                continue
            out.append(e.public())
        return out[-limit:][::-1]

    #### model usage (concurrency-safe)
    def open_use(self, model: str, purpose: str, *, by: str = "") -> int:
        """Mark a model as in use for `purpose` (by an entity). Returns a handle to close later.
        Multiple open uses of the same model = concurrent use, tracked as such."""
        h = next(self._counter)
        with self._lock:
            self._uses[h] = {"model": model, "purpose": purpose, "by": by,
                             "opened": time.time(), "closed": None}
        self.record("model:" + model, "use.open", detail={"purpose": purpose, "by": by, "handle": h})
        return h

    def close_use(self, handle: int) -> None:
        with self._lock:
            u = self._uses.get(handle)
            if u and u["closed"] is None:
                u["closed"] = time.time()
                secs = round(u["closed"] - u["opened"], 3)
            else:
                u = None
        if u:
            self.record("model:" + u["model"], "use.close", detail={"handle": handle, "seconds": secs})

    def usage(self) -> dict[str, dict]:
        """Per-model: when first seen (instantiated), how long it has run, its ACTIVE concurrent uses
        (what it is doing right now), and how many uses total across the session. Folds in peer
        processes' uses (rebuilt from their journaled use.open/use.close events)."""
        now = time.time()
        with self._lock:
            uses = [dict(u) for u in self._uses.values()]
            first: dict[str, float] = {}
            for e in self._events:
                if e.scope == "model":
                    m = e.entity.split(":", 1)[1]
                    if m not in first:
                        first[m] = e.ts
        agg: dict[str, dict] = defaultdict(lambda: {"active": [], "total": 0})
        for u in uses:
            m = u["model"]
            agg[m]["total"] += 1
            if u["closed"] is None:
                agg[m]["active"].append({"purpose": u["purpose"], "by": u["by"],
                                         "runtime_s": round(now - u["opened"], 1)})
        out = {}
        for m in set(list(agg) + list(first)):
            inst = first.get(m)
            out[m] = {"instantiated": round(inst, 3) if inst else None,
                      "runtime_s": round(now - inst, 1) if inst else 0.0,
                      "active_uses": agg[m]["active"], "concurrent": len(agg[m]["active"]),
                      "total_uses": agg[m]["total"]}
        return out

    #### journal: enable / warm-load / tail
    def enable_journal(self, path: str | None = None, *, warm: bool = True, tail: bool = False,
                       cap_lines: int = 40000) -> Path | None:
        """Point this log at the journal. `warm` folds the file's tail into memory (history across
        restarts); `tail` starts watching the file for OTHER processes' events (the server does both).
        Idempotent enough to call twice (server: once implicitly in __init__, once at startup)."""
        p = Path(path).expanduser() if path else _journal_default()
        if p is None:
            self._journal = None
            return None
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_big(p, cap_lines)
            if self._jfd is not None and self._journal != p:   # re-pointed at a new path: switch fds
                with contextlib.suppress(Exception):
                    os.close(self._jfd)
                self._jfd = None
            if self._jfd is None:
                self._jfd = os.open(str(p), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            self._journal = p
        except Exception:
            self._journal, self._jfd = None, None
            return None
        if warm:
            self._warm_load(p)
        try:
            self._tail_from = p.stat().st_size          # tail only what peers write AFTER we warm-loaded
        except OSError:
            self._tail_from = 0
        if tail:
            self.start_tailer()
        return p

    def _rotate_if_big(self, p: Path, cap_lines: int) -> None:
        """Keep the journal bounded: if it has grown well past the in-memory cap, rewrite it with only
        the most recent `cap_lines` lines. Operational file hygiene, not a decision threshold."""
        try:
            if not p.exists() or p.stat().st_size < 2_000_000:
                return
            lines = p.read_text(errors="replace").splitlines()
            if len(lines) <= cap_lines:
                return
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text("\n".join(lines[-cap_lines:]) + "\n")
            os.replace(str(tmp), str(p))
        except Exception:
            pass

    def _absorb_use(self, obj: dict) -> None:
        """Rebuild model-use state from a journaled use.open/use.close event (a peer's, or history).
        Called under self._lock. Keyed by (src, handle) so peers' handles never collide with ours."""
        act = obj.get("action")
        if act not in ("use.open", "use.close"):
            return
        det = obj.get("detail") or {}
        h = det.get("handle")
        if h is None:
            return
        key = (obj.get("src", "?"), h)
        if act == "use.open":
            ent = obj.get("entity", "")
            model = ent.split(":", 1)[1] if ":" in ent else ent
            self._uses[key] = {"model": model, "purpose": det.get("purpose", ""),
                               "by": det.get("by", ""), "opened": obj.get("ts", time.time()),
                               "closed": None}
        else:
            u = self._uses.get(key)
            if u and u["closed"] is None:
                u["closed"] = obj.get("ts", time.time())

    def _warm_load(self, p: Path) -> None:
        """Fold the journal's tail into memory so a freshly started observer already shows history."""
        try:
            lines = p.read_text(errors="replace").splitlines()
        except Exception:
            return
        keep = lines[-self._events.maxlen:]
        with self._lock:
            for ln in keep:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                self._events.append(Event(obj.get("ts", 0.0), obj.get("entity", ""),
                                          obj.get("scope", ""), obj.get("action", ""),
                                          obj.get("detail") or {}))
                self._absorb_use(obj)
            # a use held open by a process that has since exited is not really running now
            for u in self._uses.values():
                if u["closed"] is None:
                    u["closed"] = u["opened"]

    def start_tailer(self, interval: float = 0.1) -> None:
        """Watch the journal for events written by OTHER processes and fold each into this log +
        push it live. Runs on a daemon thread. No-op if already tailing or journaling is off."""
        if self._journal is None or (self._tail_thread and self._tail_thread.is_alive()):
            return
        self._tail_stop = threading.Event()
        self._tail_thread = threading.Thread(target=self._tail_loop, args=(interval,),
                                              name="activity-tailer", daemon=True)
        self._tail_thread.start()

    def stop_tailer(self) -> None:
        if self._tail_stop is not None:
            self._tail_stop.set()

    def _tail_loop(self, interval: float) -> None:
        path = self._journal
        stop = self._tail_stop
        if path is None or stop is None:
            return
        try:
            f = open(str(path), errors="replace")
        except Exception:
            return
        try:
            f.seek(self._tail_from)                # skip what warm-load already ingested
            pending = ""
            while not stop.is_set():
                chunk = f.read()
                if chunk:
                    pending += chunk
                    parts = pending.split("\n")
                    pending = parts.pop()          # trailing partial line (mid-write) waits for more
                    for ln in parts:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            obj = json.loads(ln)
                        except Exception:
                            continue
                        if obj.get("src") == _SRC:  # our own write - already in memory + already pushed
                            continue
                        self._fold(obj)
                    continue                       # keep draining until read() returns empty
                # EOF: detect truncation/rotation, else wait for more
                try:
                    if os.path.getsize(str(path)) < f.tell():
                        f.seek(0)
                        pending = ""
                except OSError:
                    pass
                stop.wait(interval)
        finally:
            with contextlib.suppress(Exception):
                f.close()

    def _fold(self, obj: dict) -> None:
        """Ingest one peer event: into the queryable log, into usage state, and out to live subscribers."""
        ev = Event(obj.get("ts", 0.0), obj.get("entity", ""), obj.get("scope", ""),
                   obj.get("action", ""), obj.get("detail") or {})
        with self._lock:
            self._events.append(ev)
            self._absorb_use(obj)
        self._publish(ev.public())

    def close(self) -> None:
        """Stop tailing and close the journal fd (used when the singleton is reset)."""
        self.stop_tailer()
        fd = self._jfd
        self._jfd = None
        if fd is not None:
            with contextlib.suppress(Exception):
                os.close(fd)


# process-wide singleton (like agent_complex.get_live)
_LOG: ActivityLog | None = None


def get_log() -> ActivityLog:
    global _LOG
    if _LOG is None:
        _LOG = ActivityLog()
    return _LOG


def record(entity: str, action: str, **kw) -> Event:
    return get_log().record(entity, action, **kw)


def open_use(model: str, purpose: str, **kw) -> int:
    return get_log().open_use(model, purpose, **kw)


def close_use(handle: int) -> None:
    get_log().close_use(handle)


def reset() -> None:
    global _LOG
    if _LOG is not None:
        _LOG.close()
    _LOG = ActivityLog()
