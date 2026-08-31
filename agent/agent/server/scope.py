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

#: WHO is serving it. Bound beside the workspace because the middleware already
#: resolved both and only kept one, which left every record and every trail entry
#: written during a request saying "local" while the request itself knew better.
_caller: ContextVar[str | None] = ContextVar("rexgraph_caller", default=None)


def current_workspace() -> str | None:
    return _current.get()


def current_caller() -> str | None:
    return _caller.get()


def set_caller(name: str | None):
    return _caller.set(name)


def reset_caller(token) -> None:
    _caller.reset(token)


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

    def __init__(self, inner, workspace: str | None, caller: str | None = None):
        self._inner = inner
        self._workspace = workspace
        self._caller = caller

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

    def _refuse_if_owned_elsewhere(self, id) -> None:
        """Refuse a write onto an id another workspace owns.

        Filtering reads was only half of it. A caller chooses the id, and a store
        appends a version rather than rejecting a collision, so writing onto someone
        else's id made the newest version theirs and the owner's own record then read
        as absent to them: not a leak but a deletion, performed by a plain user of an
        unrelated workspace.

        `owns` is the same rule the read side uses, so an unstamped record stays
        writable exactly as it stays visible. Those predate ownership and refusing them
        would strand data nobody can claim.
        """
        try:
            existing = self._inner.get_record(id)
        except Exception:
            return                                  # a store that cannot say is not a denial
        if existing is None or self._visible(existing):
            return
        self._record("db.put", id, outcome="refused",
                     reason="belongs to another workspace")
        raise PermissionError(f"record {id!r} belongs to another workspace")

    def _record(self, action: str, target: str, outcome: str = "ok", **detail) -> None:
        """Put one write in the trail.

        Here rather than in the routes for the same reason the filtering is: a record
        that changed without a trail entry is the case the trail exists to rule out,
        and a route added later would not know to write one.
        """
        from agent.server import audit
        audit.record(action, user=self._caller or "", workspace=self._workspace or "default",
                     target=str(target), outcome=outcome, detail=detail)

    def delete(self, id, **kw):
        if self.get_record(id) is None:
            self._record("db.delete", id, outcome="not_found")
            return False
        out = self._inner.delete(id, **kw)
        self._record("db.delete", id)
        return out

    def put(self, id, rex, meta=None, tags=None, **kw):
        """Store a record stamped with the workspace and the caller storing it.

        The stamps are applied here rather than by the caller so that every write goes
        in owned and attributed, including the ones written by routes that know nothing
        about workspaces.

        Assignment, not setdefault. Deferring to a value the caller already set reads as
        letting a route that knows better keep it, but inside a request the value does
        not come from the route: /api/v1/db/record-work took the workspace from the
        request BODY and work_recorder stamped a literal "default", and either one beat
        the scope and landed the record in another tenant. A route that genuinely knows
        better is one running outside a request, and outside a request this wrapper does
        not exist.
        """
        self._refuse_if_owned_elsewhere(id)
        meta = dict(meta or {})
        meta["workspace"] = self._workspace
        if self._caller:
            meta["stored_by"] = self._caller
        out = self._inner.put(id, rex, meta=meta, tags=tags, **kw)
        self._record("db.put", id, nE=int(getattr(rex, "nE", 0) or 0))
        return out


def scoped(store):
    """`store` as this request may see it, or unchanged outside a scoped request."""
    if not scoping_active():
        return store
    return ScopedStore(store, _current.get(), _caller.get())


class ScopedSecrets:
    """A secret store that only shows one workspace's saved connections.

    Wraps rather than subclasses, for the reason `ScopedStore` does: there are three
    backends and this has to apply to all of them identically. The workspace goes into
    the KEY rather than onto the record, because `SecretStore.put` carries no metadata
    field to stamp and adding one would change all three backends and their on-disk
    formats.

    A workspace name cannot contain "/" (`handles.WORKSPACE_RE`), so the first "/"
    splits a stored key unambiguously and a name a tenant chooses cannot forge another
    workspace's prefix: saving "alpha/prod" from workspace beta stores "beta/alpha/prod".
    """

    def __init__(self, inner, workspace: str):
        self._inner = inner
        self._workspace = workspace
        self._prefix = f"{workspace}/"

    def get(self, name: str) -> str:
        try:
            return self._inner.get(self._prefix + name)
        except KeyError:
            # An unprefixed entry predates workspace ownership and stays readable, the
            # same rule `owns` applies to an unstamped record. A name that is already
            # qualified is not retried, or one tenant could read another's by asking
            # for their key directly.
            if "/" in name:
                raise
            return self._inner.get(name)

    def put(self, name: str, uri: str, kind: str = "sql") -> None:
        self._inner.put(self._prefix + name, uri, kind)

    def list(self):
        mine, legacy = [], []
        for rec in self._inner.list():
            n = rec.get("name", "")
            if n.startswith(self._prefix):
                mine.append({**rec, "name": n[len(self._prefix):]})
            elif "/" not in n:
                legacy.append(rec)
        owned = {r["name"] for r in mine}
        return mine + [r for r in legacy if r.get("name") not in owned]

    def delete(self, name: str) -> bool:
        # Only what this workspace owns. A legacy unprefixed entry reads as everyone's,
        # so letting any one tenant delete it would remove it for all of them; that
        # stays an operator action outside a request.
        return self._inner.delete(self._prefix + name)

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


def scoped_secrets(store):
    """`store` as this request may see it, or unchanged outside a scoped request."""
    if not scoping_active():
        return store
    return ScopedSecrets(store, _current.get())


def bound_workspace() -> str:
    """The workspace this call belongs to, or "default" outside a request.

    The same question `agent_complex` and `hive_network` each answered with their own
    copy, each wrapped in a defensive try/except. This module holds the ContextVar and
    imports nothing but contextvars, so it is reachable from anywhere and there was never
    a reason for a second copy or for the guard around it.
    """
    return _current.get() or "default"


def effective_workspace(requested: str = "") -> str:
    """The workspace this request may write to.

    Inside a scoped request the bound workspace wins over anything the caller named,
    which is the rule `ScopedStore.put` applies to a record's stamp. Three routes took a
    workspace from a form field or a request body and used it to choose a directory or
    to stamp a lineage record, so naming another tenant was enough to write into them.

    Outside a request there is no bound workspace and the caller's own value stands,
    which is how the CLI and the recorder keep working.
    """
    if scoping_active():
        return bound_workspace()
    return requested or "default"


_SECRET_STORE = None


def secret_store():
    """The configured secret store as this request may see it.

    The opened backend is shared, the view of it is per request. This lives here rather
    than in a route because more than one route module resolves a saved connection name,
    and one of them reaching the raw store is enough to hand a tenant another tenant's
    credentials: routes/connectors.py called open_secret_store() directly and did exactly
    that, for read, validate and ingest.
    """
    global _SECRET_STORE
    if _SECRET_STORE is None:
        from agent.secrets import open_secret_store
        _SECRET_STORE = open_secret_store()
    return scoped_secrets(_SECRET_STORE)


def reset_secret_store() -> None:
    """Drop the opened backend so the next call re-reads the configuration. For tests."""
    global _SECRET_STORE
    _SECRET_STORE = None


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
        identity, name = identity_and_workspace(request)
        token = set_workspace(name or "default")
        who = set_caller(identity or None)
        try:
            return await call_next(request)
        finally:
            reset_caller(who)
            reset_workspace(token)
