"""
agent.server.routes.courier: route surface for carrying stored complexes between stores.

The courier itself is an ordinary hive worker; this is the door that lets something other
than a Python import reach it. Trips still land in the network complex, so what moves
between two stores is visible to `/api/v1/agents/monitor` as inter-hive traffic rather
than as a side channel beside it.

Admin, not user, for everything that binds a destination or moves a record. The rule is
the one the tool registry states next to its own handlers: anything that reaches beyond
the caller's own request needs admin, and a trip reaches another store, or another
machine.

Two consequences of that worth stating rather than discovering. A route bound with no
explicit store takes `rcdb.default_store()` AS THE CALLER SEES IT, so the binding carries
the workspace that made it and a courier configured by one tenant cannot later carry
another's records. And a peer is named by `api_key_ref`, never by a key: the reference is
resolved when the peer is built and the credential is not accepted over the wire, echoed
back, or held on the peer, which is the same contract `/api/v1/hive/attach` keeps.
"""
from fastapi import APIRouter, Body, Depends, HTTPException

from ..auth import TokenEntry, require_admin

router = APIRouter(prefix="/v1")


def _courier():
    from agent.courier import get_courier
    return get_courier()


def _spec(body: dict):
    from agent.courier import CarrySpec
    return CarrySpec.from_dict(body) if (body.get("tags") or body.get("ids")
                                         or body.get("limit")) else None


@router.get("/courier/status")
async def courier_status(_t: TokenEntry = Depends(require_admin)):
    """Which stores and peers this courier routes for, and what it has carried.

    Binding a store or a peer is already an admin operation, so reading which ones are
    bound is too: the courier is a process-wide singleton holding store views bound by
    whoever bound them, and a survey lists records through those views rather than
    through the caller's own.
    """
    return _courier().status()


@router.post("/courier/routes")
async def courier_route(body: dict = Body(...), _t: TokenEntry = Depends(require_admin)):
    """Bind a store to a hive name. body: {hive, store?}.

    `store` is an RCDB uri; omit it to bind this server's own store as the caller sees it.
    """
    hive = (body.get("hive") or "").strip()
    if not hive:
        raise HTTPException(400, "need 'hive'")
    store = body.get("store")
    if not store:
        from agent.rcdb import default_store
        store = default_store()
    c = _courier()
    try:
        c.attach_store(hive, store)
    except Exception as e:
        raise HTTPException(400, f"could not open that store: {e}") from e
    return {"ok": True, "hive": hive, "hives": c.hives()}


@router.post("/courier/peers")
async def courier_peer(body: dict = Body(...), _t: TokenEntry = Depends(require_admin)):
    """Register a remote server as a destination. body: {name, url, api_key_ref?, confirm?}.

    `api_key_ref` names an env var or secret-store entry holding the peer's bearer token.
    The API never accepts or returns the token itself.
    """
    name, url = (body.get("name") or "").strip(), (body.get("url") or "").strip()
    if not (name and url):
        raise HTTPException(400, "need 'name' and 'url'")
    if "api_key" in body:
        raise HTTPException(400, "pass 'api_key_ref', a reference; the API takes no keys")
    from agent.client import RexClient
    from agent.courier_remote import Ledger, Peer
    from agent.secrets import resolve_request_ref
    ref = body.get("api_key_ref", "")
    try:
        key = resolve_request_ref(ref)          # admin is not a licence to name any variable
    except PermissionError as e:
        raise HTTPException(400, str(e)) from e
    peer = Peer(name, RexClient(url, api_key=key or None),
                ledger=Ledger(body.get("ledger") or None),
                confirm=bool(body.get("confirm", False)))
    c = _courier()
    c.attach_peer(peer)
    return {"ok": True, "peer": name, "peers": c.peers(), "has_api_key": bool(key)}


@router.get("/courier/survey")
async def courier_survey(hive: str, tags: str = "", limit: int = 100,
                         _t: TokenEntry = Depends(require_admin)):
    """What a trip out of this hive would consider, carrying nothing."""
    from agent.courier import CarrySpec
    c = _courier()
    spec = CarrySpec(tags=[t for t in tags.split(",") if t], limit=limit)
    try:
        return {"hive": hive, "records": c.survey(hive, carry=spec)}
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/courier/deliver")
async def courier_deliver(body: dict = Body(...), _t: TokenEntry = Depends(require_admin)):
    """One trip. body: {source, dest, tags?, ids?, limit?}.

    `dest` names a bound store or a registered peer; a destination this courier does not
    already route for is refused rather than built from the request, so a caller cannot
    name a machine the operator never approved.
    """
    source, dest = (body.get("source") or "").strip(), (body.get("dest") or "").strip()
    if not (source and dest):
        raise HTTPException(400, "need 'source' and 'dest'")
    c = _courier()
    if dest not in c.destinations():
        raise HTTPException(404, f"no destination {dest!r}; register it first")
    try:
        return c.deliver(source, dest, carry=_spec(body))
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/courier/broadcast")
async def courier_broadcast(body: dict = Body(...), _t: TokenEntry = Depends(require_admin)):
    """One trip per destination. body: {source, dests?, tags?, ids?, limit?}."""
    source = (body.get("source") or "").strip()
    if not source:
        raise HTTPException(400, "need 'source'")
    c = _courier()
    dests = body.get("dests")
    if dests is not None:
        unknown = [d for d in dests if d not in c.destinations()]
        if unknown:
            raise HTTPException(404, f"no destination(s): {', '.join(unknown)}")
    try:
        return c.broadcast(source, dests, carry=_spec(body))
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
