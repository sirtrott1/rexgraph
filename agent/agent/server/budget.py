"""
agent.server.budget: a ceiling on the work one caller can ask for at once.

The rate limiter in `security.py` counts requests, which bounds how OFTEN a caller can
ask and says nothing about how much each ask costs. These requests are not uniform: a
few hundred bytes can name a complex with millions of cells, and the exact-arithmetic
paths trade unbounded CPU for an answer with no rounding in it. A caller well inside
the rate limit can still occupy every core.

So this bounds the work itself, on three axes that fail differently:

    size          cells admitted per request, refused before anything is built
    concurrency   requests in flight per identity, so one caller cannot take every
                  worker and starve the rest
    time          a deadline the operation checks, so a request that turns out to be
                  expensive ends rather than running until the client gives up

Refusal is immediate and says which ceiling was hit. A caller that has to wait to find
out it was refused is a caller holding a connection, which is the thing being defended
against.

In-process, like the rate limiter: it bounds one server. Several workers behind a proxy
each get their own ceiling, so set the per-worker number to the share you actually want
each to serve.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

#: cells admitted in one request, across every grade
DEFAULT_MAX_CELLS = 5_000_000

#: requests in flight per identity
DEFAULT_MAX_INFLIGHT = 4

#: seconds one request may run before its deadline passes
DEFAULT_DEADLINE = 300.0


class BudgetExceeded(RuntimeError):
    """A request that costs more than the caller is allowed to spend."""

    def __init__(self, message: str, *, axis: str = ""):
        super().__init__(message)
        self.axis = axis


def _limit(name: str, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return type(default)(raw)
    except ValueError:
        return default


def max_cells() -> int:
    return _limit("REXGRAPH_MAX_CELLS", DEFAULT_MAX_CELLS)


def max_inflight() -> int:
    return _limit("REXGRAPH_MAX_INFLIGHT", DEFAULT_MAX_INFLIGHT)


def deadline_seconds() -> float:
    return _limit("REXGRAPH_DEADLINE", DEFAULT_DEADLINE)


class Deadline:
    """When a request runs out of time.

    Passed to the work rather than enforced around it: killing a thread mid-computation
    leaves shared state torn, so the operation is asked to stop at a point it chooses.
    Loops that can run long check `expired` and raise.
    """

    def __init__(self, seconds: float | None = None):
        self.seconds = float(seconds if seconds is not None else deadline_seconds())
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.seconds - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.remaining <= 0.0

    def check(self, what: str = "operation") -> None:
        if self.expired:
            raise BudgetExceeded(
                f"{what} passed its {self.seconds:g}s deadline", axis="time")


class _Inflight:
    """How many requests each identity has running."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    @contextmanager
    def slot(self, identity: str, limit: int):
        key = str(identity or "local")
        with self._lock:
            current = self._counts.get(key, 0)
            if current >= limit:
                raise BudgetExceeded(
                    f"{current} requests already running for this identity, "
                    f"limit is {limit}", axis="concurrency")
            self._counts[key] = current + 1
        try:
            yield
        finally:
            with self._lock:
                n = self._counts.get(key, 1) - 1
                if n > 0:
                    self._counts[key] = n
                else:
                    self._counts.pop(key, None)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._counts)


_inflight = _Inflight()


def check_size(counts, *, limit: int | None = None) -> None:
    """Refuse an oversized request before it is built.

    Takes cell counts rather than an object, so it can run against a frame header while
    the payload is still bytes.
    """
    cap = int(limit if limit is not None else max_cells())
    if isinstance(counts, dict):
        items = [(k, v) for k, v in counts.items() if isinstance(v, (int, float))]
    else:
        items = list(enumerate(counts))
    for name, n in items:
        if n is not None and int(n) > cap:
            raise BudgetExceeded(
                f"{name}={int(n)} is over the {cap}-cell limit for one request",
                axis="size")


@contextmanager
def guard(identity: str = "local", *, counts=None, seconds: float | None = None,
          inflight_limit: int | None = None):
    """Admit one request under every ceiling, or refuse it.

    Yields a `Deadline` the work can check. Size is tested first because it is free and
    a refusal there costs nothing; the concurrency slot is taken only once the request
    is known to be worth admitting.
    """
    if counts is not None:
        check_size(counts)
    limit = int(inflight_limit if inflight_limit is not None else max_inflight())
    with _inflight.slot(identity, limit):
        yield Deadline(seconds)


def inflight() -> dict:
    """Requests currently running, per identity. For the operator's view."""
    return _inflight.snapshot()


#: paths that must answer while the server is saturated, so they are never metered:
#: the health probe (a load balancer pulls the node otherwise), the UI shell and its
#: assets, and the operator's own view of what is currently running.
_EXEMPT_EXACT = {"/", "/api/health", "/docs", "/redoc", "/openapi.json",
                 "/favicon.ico", "/rex/v1/hello"}
_EXEMPT_PREFIXES = ("/static/",)


def add_compute_budget(app) -> None:
    """Hold every request to the concurrency ceiling, not just the ones that ask.

    The rate limiter counts requests per IP, which bounds how OFTEN a caller can ask
    and says nothing about how much each ask costs. A guard written into one route
    covers that route; a caller reaches the expensive work through whichever route did
    not remember. So the slot is taken here, where every route inherits it.

    Metered per IDENTITY rather than per connection, because one tenant holding several
    connections is the case worth bounding. Starlette runs the last-registered
    middleware first, so this is registered BEFORE the auth enforcement it must run
    behind: an unauthenticated request is rejected by auth and never reaches a slot,
    and the rate limiter (outermost) is what absorbs a flood of those.

    A deadline is bound for the request so long-running work can check it, the same
    shape `scope` uses for the workspace.
    """
    from fastapi.responses import JSONResponse

    from agent.server.auth import identity_and_workspace

    @app.middleware("http")
    async def _meter(request, call_next):
        path = request.url.path
        if path in _EXEMPT_EXACT or any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            return await call_next(request)
        identity, _ws = identity_and_workspace(request)
        try:
            with guard(identity or "anonymous") as deadline:
                request.state.rex_deadline = deadline
                return await call_next(request)
        except BudgetExceeded as e:
            return JSONResponse(
                {"detail": str(e), "axis": e.axis}, status_code=429,
                headers={"Retry-After": "1"})

    return _meter
