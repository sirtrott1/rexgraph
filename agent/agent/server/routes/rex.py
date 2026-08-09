"""
agent.server.routes.rex: the rexgraph-native surface, over HTTP with binary bodies.

HTTP is the transport because it already has the parts worth not rebuilding: proxies,
TLS, auth headers, status codes. What it carries is the layered binary a complex is
stored as, so an object crosses the wire in the form it already has and no second
encoding exists to drift from the first. JSON appears only where the answer is a
scalar reading about a complex rather than the complex itself.

Every inbound frame is checked twice before anything runs on it, because two different
things can be wrong. The header's digest says whether these are the bytes that were
sent. The chain condition, `B_d B_{d+1} = 0`, exact over the integers, says whether
those bytes describe a complex: faces that do not close fail arithmetic rather than a
validator. The second check relates two grades, so it says nothing about a complex with
no faces, and the digest is what covers a plain graph in transit. Neither substitutes
for the other and the route runs both.

The boundary between local and network is one question, asked once: is auth on. With it
off this is an operator on their own machine, so files are named by path and every tool
is theirs to run. With it on, callers name files by handle, are admitted against their
workspace, and every call lands in the audit trail whether it succeeded or not.

    POST /rex/v1/verify     frame in, its fingerprint out
    POST /rex/v1/store      frame in, kept in the caller's workspace
    GET  /rex/v1/fetch/…    a stored complex back out, as a frame
    POST /rex/v1/upload     bytes in, a handle out
    GET  /rex/v1/files      the handles this workspace holds
    POST /rex/v1/call       run one tool
    GET  /rex/v1/hello      what this server speaks, and its ceilings
    GET  /rex/v1/audit      the trail, and whether it still verifies
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import Response

from rexgraph.protocol import CONTENT_TYPE, WIRE_VERSION, ProtocolError

from ..auth import TokenEntry, get_auth_manager, is_admin, require_auth

router = APIRouter(prefix="/rex/v1")


def _context(request: Request, token: TokenEntry):
    """Who this request is, and what it may reach."""
    from agent.mcp_tools import Context
    mgr = get_auth_manager()
    ws = (request.headers.get("X-Workspace")
          or request.query_params.get("workspace")
          or (token.workspaces[0] if token.workspaces else "default"))
    if mgr.auth_enabled and not token.can_access(ws):
        raise HTTPException(403, f"No access to workspace '{ws}'")
    return Context(workspace=ws, identity=token.user_id,
                   is_admin=is_admin(token, ws),
                   auth_enabled=bool(mgr.auth_enabled))


def frame_key() -> bytes | None:
    """The shared key frames are signed with, if this deployment sets one.

    `REXGRAPH_FRAME_KEY`. Unset means unsigned frames are accepted, which is right for
    a local server and for one behind TLS that terminates where it is trusted. Set it
    where the path between client and server is not trusted end to end: the digest in
    a frame's header is recomputed by anyone who rewrites the payload, so it catches
    accidents and not an attacker sitting in the path. An HMAC catches both.
    """
    import os
    raw = os.environ.get("REXGRAPH_FRAME_KEY", "")
    return raw.encode("utf-8") if raw else None


async def _frame(request: Request):
    """Read one frame off the request, refusing it before it is built.

    The declared length is checked against the ceiling before the body is read, so an
    oversized frame is refused rather than buffered, and the signature is checked
    before the payload is parsed: a frame that cannot be authenticated should not
    reach the decoder at all.
    """
    from rexgraph.protocol import DEFAULT_MAX_FRAME, decode, verify_signature

    from ..budget import max_cells

    declared = request.headers.get("content-length")
    if declared and int(declared) > DEFAULT_MAX_FRAME:
        raise HTTPException(413, "frame is over the size limit")
    body = await request.body()

    key = frame_key()
    if key is not None and not verify_signature(
            body, request.headers.get("X-Rex-Signature", ""), key):
        raise HTTPException(401, "frame signature missing or invalid")

    try:
        return decode(body, max_cells=max_cells())
    except ProtocolError as e:
        raise HTTPException(400, str(e)) from e


def _rebuild(frame):
    """The complex a frame carries, verified.

    A failure here is a refusal, not a repair: a body whose boundary data does not
    satisfy the chain condition is not a complex, and guessing what was meant is how a
    malformed payload becomes a stored one.
    """
    from rexgraph.protocol import to_complex
    try:
        return to_complex(frame, verify=True)
    except ProtocolError as e:
        raise HTTPException(422, str(e)) from e


def _binary(payload: bytes) -> Response:
    """A frame on the way out, signed if this deployment signs.

    Both directions or neither: a client that has to authenticate what it sends and
    cannot authenticate what it receives is still talking to whoever is in the path.
    """
    from rexgraph.protocol import sign
    headers = {}
    key = frame_key()
    if key is not None:
        headers["X-Rex-Signature"] = sign(payload, key)
    return Response(content=payload, media_type=CONTENT_TYPE, headers=headers)


@router.get("/hello")
async def hello(request: Request, token: TokenEntry = Depends(require_auth)):
    """What this server speaks and what it will not exceed.

    Read before sending anything: the ceilings are the server's, not the protocol's, so
    a client that checks here does not discover them by being refused.
    """
    from ..budget import deadline_seconds, max_cells, max_inflight
    from ..handles import paths_allowed
    ctx = _context(request, token)
    return {
        "wire_version": WIRE_VERSION,
        "content_type": CONTENT_TYPE,
        "workspace": ctx.workspace,
        "identity": ctx.identity,
        "auth_enabled": ctx.auth_enabled,
        "paths_allowed": paths_allowed(ctx.auth_enabled),
        "signed_frames": frame_key() is not None,
        "limits": {"max_cells": max_cells(), "max_inflight": max_inflight(),
                   "deadline_seconds": deadline_seconds()},
    }


@router.post("/verify")
async def verify(request: Request, token: TokenEntry = Depends(require_auth)):
    """Check a frame and report what it carries, without keeping it.

    The cheapest useful call: a client that wants to know whether what it holds is a
    well-formed complex, and which one, before deciding to store it.
    """
    from rexgraph.protocol import fingerprint

    from .. import audit
    from ..budget import BudgetExceeded, check_size
    ctx = _context(request, token)
    frame = await _frame(request)
    try:
        # the concurrency slot and the deadline are the budget middleware's, taken for
        # every route. What is frame-specific is the SIZE, which is read off the header
        # before the complex is built, so it is checked here.
        check_size(frame.header)
        rex = _rebuild(frame)
        out = fingerprint(rex)
    except BudgetExceeded as e:
        audit.record("rex.verify", user=ctx.identity, workspace=ctx.workspace,
                     outcome="refused", detail={"axis": e.axis})
        raise HTTPException(429, str(e)) from e
    audit.record("rex.verify", user=ctx.identity, workspace=ctx.workspace,
                 detail={"nV": out["nV"], "nE": out["nE"]})
    return {"bytes": frame.n_bytes, "fingerprint": out}


@router.post("/store")
async def store(request: Request, token: TokenEntry = Depends(require_auth)):
    """Keep a frame's complex in the caller's workspace.

    Stamped with the workspace that sent it, which is what makes a later fetch a
    question with an answer rather than a lookup in a shared namespace.
    """
    import secrets

    from agent.rcdb import default_store

    from .. import audit
    from ..budget import BudgetExceeded, check_size
    ctx = _context(request, token)
    frame = await _frame(request)
    try:
        check_size(frame.header)
        rex = _rebuild(frame)
        meta = dict(frame.header.get("meta") or {})
        meta["workspace"] = ctx.workspace
        meta["stored_by"] = ctx.identity
        # random rather than sequential or clock-derived: an id that can be guessed
        # from the one before it makes the ownership check the only thing between a
        # caller and every record in the store
        record_id = f"rx_{secrets.token_hex(8)}"
        default_store().put(record_id, rex, meta=meta)
    except BudgetExceeded as e:
        audit.record("rex.store", user=ctx.identity, workspace=ctx.workspace,
                     outcome="refused", detail={"axis": e.axis})
        raise HTTPException(429, str(e)) from e
    audit.record("rex.store", user=ctx.identity, workspace=ctx.workspace,
                 target=str(record_id), detail={"nE": int(rex.nE)})
    return {"record_id": record_id, "workspace": ctx.workspace}


@router.get("/fetch/{record_id}")
async def fetch(record_id: str, request: Request,
                token: TokenEntry = Depends(require_auth)):
    """A stored complex back out, as a frame.

    A record belonging to another workspace reads as absent. Saying it exists but is
    not yours turns a guessable id into a way to enumerate what other tenants hold.
    """
    from agent.rcdb import default_store
    from rexgraph.protocol import encode

    from .. import audit
    ctx = _context(request, token)
    store_ = default_store()
    record = store_.get_record(record_id)
    owner = ((record.meta or {}).get("workspace") if record is not None else None)
    if record is None or (ctx.auth_enabled and owner is not None
                          and owner != ctx.workspace):
        audit.record("rex.fetch", user=ctx.identity, workspace=ctx.workspace,
                     target=record_id, outcome="not_found")
        raise HTTPException(404, "no such record in this workspace")
    rex = store_.get(record_id)
    if rex is None:
        raise HTTPException(404, "no such record in this workspace")
    audit.record("rex.fetch", user=ctx.identity, workspace=ctx.workspace,
                 target=record_id)
    return _binary(encode(rex, meta={"record_id": record_id}))


@router.post("/upload")
async def upload(request: Request, token: TokenEntry = Depends(require_auth)):
    """Take bytes and return the handle that names them.

    The way a file enters a workspace. Content-addressed, so the same file uploaded
    twice is one copy under one handle.
    """
    from .. import audit
    from ..handles import HandleError, mint
    ctx = _context(request, token)
    body = await request.body()
    if not body:
        raise HTTPException(400, "no content")
    name = request.headers.get("X-Filename", "")
    try:
        out = mint(ctx.workspace, body, name=name)
    except HandleError as e:
        raise HTTPException(400, str(e)) from e
    audit.record("rex.upload", user=ctx.identity, workspace=ctx.workspace,
                 target=out["handle"], detail={"bytes": out["bytes"], "name": name})
    return out


@router.get("/files")
async def files(request: Request, token: TokenEntry = Depends(require_auth)):
    """The handles this workspace holds."""
    from ..handles import listing
    ctx = _context(request, token)
    return {"workspace": ctx.workspace, "files": listing(ctx.workspace)}


@router.delete("/files/{handle}")
async def drop_file(handle: str, request: Request,
                    token: TokenEntry = Depends(require_auth)):
    """Forget one handle's content."""
    from .. import audit
    from ..handles import forget
    ctx = _context(request, token)
    ok = forget(ctx.workspace, handle)
    audit.record("rex.forget", user=ctx.identity, workspace=ctx.workspace,
                 target=handle, outcome="ok" if ok else "not_found")
    if not ok:
        raise HTTPException(404, "no such handle in this workspace")
    return {"forgotten": handle}


@router.get("/tools")
async def tools(request: Request, token: TokenEntry = Depends(require_auth)):
    """Every capability this caller may run, with its schema."""
    from agent.mcp_tools import definitions
    ctx = _context(request, token)
    return {"tools": definitions(ctx)}


@router.post("/call")
async def call_tool(request: Request, body: dict = Body(...),
                    token: TokenEntry = Depends(require_auth)):
    """Run one tool as this caller.

    Level, arguments and file access are all settled by the registry's own gate, so
    this route decides nothing about them: it establishes who is asking, holds them to
    the ceilings, and records the outcome either way.
    """
    from agent.mcp_tools import call
    from agent.server.artifacts import plain

    from .. import audit
    from ..budget import BudgetExceeded
    from ..handles import HandleError

    ctx = _context(request, token)
    name = body.get("name")
    if not name:
        raise HTTPException(400, "Provide 'name'")
    arguments = body.get("arguments") or {}

    def _fail(status, message, outcome):
        audit.record("rex.call", user=ctx.identity, workspace=ctx.workspace,
                     target=str(name), outcome=outcome, detail={"error": message[:200]})
        return HTTPException(status, message)

    try:
        result = call(name, context=ctx, **arguments)
    except KeyError as e:
        raise _fail(404, str(e), "unknown_tool") from e
    except PermissionError as e:
        raise _fail(403, str(e), "denied") from e
    except HandleError as e:
        raise _fail(400, str(e), "bad_input") from e
    except TypeError as e:
        raise _fail(400, str(e), "bad_arguments") from e
    except BudgetExceeded as e:
        raise _fail(429, str(e), "refused") from e
    except Exception as e:                       # noqa: BLE001 - the tool's own fault
        raise _fail(400, f"{name} failed: {e}", "error") from e

    audit.record("rex.call", user=ctx.identity, workspace=ctx.workspace,
                 target=str(name))
    return {"tool": name, "result": plain(result)}


@router.get("/audit")
async def trail(request: Request, limit: int = 200,
                token: TokenEntry = Depends(require_auth)):
    """This workspace's trail, and whether the chain still verifies.

    Verification runs over the whole file rather than the slice returned, because a
    chain checked only where someone looked is not checked.
    """
    from .. import audit
    ctx = _context(request, token)
    return {
        "workspace": ctx.workspace,
        "entries": audit.read(workspace=ctx.workspace, limit=max(1, min(limit, 1000))),
        "integrity": audit.verify(),
    }
