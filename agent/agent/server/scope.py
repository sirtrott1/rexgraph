"""
agent.server.scope: which workspace the request in hand belongs to, and a store that
cannot see past it.

Authentication is settled globally by `security.add_auth_enforcement`: with auth on,
every route needs a valid token. That answers whether a caller is someone. It does not
answer which records are theirs, and the record store is one namespace shared by every
workspace, so a valid token was enough to list and read what another tenant put there.

Fixing that route by route means remembering it on each of them, including the next one
someone writes. Instead the restriction goes where the records are reached: the
workspace travels with the request in a context variable, and the store accessor hands
back a view filtered by it. A route keeps calling `default_store()` and gets a store
that does not contain other people's records, so there is no check to forget.

Unset context means unrestricted, which is what the CLI, the test suite and any
in-process caller want: they are not serving a request and are not being scoped. The
filter also only engages when auth is on, so single-operator local use is untouched.

A record with no workspace recorded is treated as belonging to everyone. Those are the
ones written before this existed, and hiding them would make an upgrade look like data
loss. New records get stamped on the way in.
"""

from __future__ import annotations

from contextvars import ContextVar

#: the workspace serving the current request, or None outside one
_current: ContextVar[str | None] = ContextVar("rexgraph_workspace", default=None)


def current_workspace() -> str | None:
    return _current.get()


def set_workspace(name: str | None):
    """Bind the workspace for this request. Returns the token to reset with."""
    return _current.set(name)


def reset_workspace(token) -> None:
    _current.reset(token)


def scoping_active() -> bool:
    """Whether records should be filtered at all.

    Both conditions have to hold: auth is on, so there is more than one tenant, and a
    workspace is bound, so we are inside a request that belongs to one.
    """
    if _current.get() is None:
        return False
    try:
        from agent.server.auth import get_auth_manager
        return bool(get_auth_manager().auth_enabled)
    except Exception:                            # noqa: BLE001 - no auth, no scoping
        return False


def owns(meta, workspace: str | None) -> bool:
    """Whether a record belongs to this workspace.

    An unstamped record predates workspace ownership and stays visible; stamping it
    retroactively would mean guessing whose it was.
    """
    owner = (meta or {}).get("workspace")
    return owner is None or owner == workspace


class ScopedStore:
    """A record store that only shows one workspace's records.

    Wraps rather than subclasses, because the store has several backends and this has
    to apply to all of them identically. Anything not named here passes through, so a
    backend-specific method still works; the five that reach records by id or return
    them in bulk are the ones that need an opinion.
    """

    def __init__(self, inner, workspace: str | None):
        self._inner = inner
        self._workspace = workspace

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _visible(self, record) -> bool:
        return record is not None and owns(getattr(record, "meta", None),
                                           self._workspace)

    def get_record(self, id, **kw):
        record = self._inner.get_record(id, **kw)
        return record if self._visible(record) else None

    def get(self, id, **kw):
        # asked through get_record first, so a record belonging to someone else reads
        # as absent rather than as a permission error that confirms it exists
        if self.get_record(id) is None:
            return None
        return self._inner.get(id, **kw)

    def list(self, *a, **kw):
        return [r for r in self._inner.list(*a, **kw) if self._visible(r)]

    def query(self, *a, **kw):
        return [r for r in self._inner.query(*a, **kw) if self._visible(r)]

    def _record(self, action: str, target: str, outcome: str = "ok", **detail) -> None:
        """Put one write in the trail.

        Here rather than in the routes for the same reason the filtering is: a record
        that changed without a trail entry is the case the trail exists to rule out,
        and a route added later would not know to write one.
        """
        from agent.server import audit
        audit.record(action, workspace=self._workspace or "default",
                     target=str(target), outcome=outcome, detail=detail)

    def delete(self, id, **kw):
        if self.get_record(id) is None:
            self._record("db.delete", id, outcome="not_found")
            return False
        out = self._inner.delete(id, **kw)
        self._record("db.delete", id)
        return out

    def put(self, id, rex, meta=None, tags=None, **kw):
        """Store a record stamped with the workspace that is storing it.

        The stamp is applied here rather than by the caller so that every write goes in
        owned, including the ones written by routes that know nothing about workspaces.
        """
        meta = dict(meta or {})
        meta.setdefault("workspace", self._workspace)
        out = self._inner.put(id, rex, meta=meta, tags=tags, **kw)
        self._record("db.put", id, nE=int(getattr(rex, "nE", 0) or 0))
        return out


def scoped(store):
    """`store` as this request may see it, or unchanged outside a scoped request."""
    if not scoping_active():
        return store
    return ScopedStore(store, _current.get())


def add_workspace_scope(app) -> None:
    """Bind each request's workspace for the duration of that request.

    Middleware, so it covers every route including ones added later. It only records
    which workspace was asked for; whether the caller may use it is
    `auth.require_workspace`'s decision, and the routes that never declared that are
    exactly the ones that were reaching the shared store unfiltered.

    Starlette runs the LAST-registered middleware first, so this is registered AFTER
    the auth enforcement it must run behind. A caller that sends no `X-Workspace` still
    gets scoped to what their token grants, rather than to everything.
    """
    from agent.server.auth import identity_and_workspace

    @app.middleware("http")
    async def _bind(request, call_next):
        _identity, name = identity_and_workspace(request)
        token = set_workspace(name or "default")
        try:
            return await call_next(request)
        finally:
            reset_workspace(token)
