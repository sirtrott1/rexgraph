"""
agent.server.routes.courier: route surface for carrying stored complexes between stores.

The courier itself is an ordinary hive worker; this is the door that lets something other
than a Python import reach it. Trips still land in the network complex, so what moves
between two stores is visible to `/api/v1/agents/monitor` as inter hive traffic rather
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
from fastapi import APIRouter, Body, Depends, HTTPException, Query

from ..auth import TokenEntry, require_admin

router = APIRouter(prefix="/v1")


def _courier():
    from agent.courier import get_courier
    return get_courier()


def _spec(body: dict):
    from agent.courier import CarrySpec
    try:
        return CarrySpec.from_dict(body) if any(k in body for k in ("tags", "ids", "limit")) else None
    except ValueError as exc:
        raise HTTPException(400, "invalid courier selection") from exc


def _name(body, key):
    from agent.courier import _name as literal_name
    try:
        return literal_name(body.get(key))
    except ValueError as exc:
        raise HTTPException(400, f"need bounded literal '{key}'") from exc


@router.get("/courier/status")
def courier_status(_t: TokenEntry = Depends(require_admin)):
    """Which stores and peers this courier routes for, and what it has carried.

    Binding a store or a peer is already an admin operation, so reading which ones are
    bound is too: the courier is a process wide singleton holding store views bound by
    whoever bound them, and a survey lists records through those views rather than
    through the caller's own.
    """
    return _courier().status()


@router.post("/courier/routes")
def courier_route(body: dict = Body(...), _t: TokenEntry = Depends(require_admin)):
    """Bind a store to a hive name. body: {hive, store?}.

    `store` is an RCDB uri; omit it to bind this server's own store as the caller sees it.
    """
    hive = _name(body, "hive")
    store = body.get("store")
    if store is None:
        from agent.rcdb import default_store
        store = default_store()
    elif type(store) is not str or not store:
        raise HTTPException(400, "store requires an RCDB URI")
    c = _courier()
    try:
        c.attach_store(hive, store)
    except Exception as e:
        raise HTTPException(400, "could not register that store") from e
    return {"ok": True, "hive": hive, "hives": c.hives()}


@router.post("/courier/peers")
def courier_peer(body: dict = Body(...), _t: TokenEntry = Depends(require_admin)):
    """Register a remote server as a destination. body: {name, url, api_key_ref?, confirm?}.

    `api_key_ref` names an env var or secret store entry holding the peer's bearer token.
    The API never accepts or returns the token itself.
    """
    name, url = _name(body, "name"), _name(body, "url")
    if "api_key" in body:
        raise HTTPException(400, "pass 'api_key_ref', a reference; the API takes no keys")
    from agent.client import RexClient
    from agent.courier_remote import Ledger, Peer
    from agent.secrets import resolve_request_ref
    ref = body.get("api_key_ref", "")
    if type(ref) is not str or type(body.get("confirm", False)) is not bool:
        raise HTTPException(400, "invalid peer reference or confirmation flag")
    ledger = body.get("ledger")
    if ledger is not None and (type(ledger) is not str or not ledger):
        raise HTTPException(400, "ledger requires a nonempty path")
    try:
        key = resolve_request_ref(ref)          # admin is not a licence to name any variable
    except PermissionError as e:
        raise HTTPException(400, str(e)) from e
    try:
        peer = Peer(name, RexClient(url, api_key=key or None),
                    ledger=Ledger(body.get("ledger") or None),
                    confirm=bool(body.get("confirm", False)))
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(400, "could not open the courier ledger") from exc
    c = _courier()
    try:
        c.attach_peer(peer)
    except ValueError as exc:
        raise HTTPException(400, "could not register that peer") from exc
    return {"ok": True, "peer": name, "peers": c.peers(), "has_api_key": bool(key)}


@router.post("/courier/reconcile")
def courier_reconcile(body: dict = Body(...), _t: TokenEntry = Depends(require_admin)):
    """Verify a known copy receipt against registered source and peer, without storing again."""
    from rcdb import CopyReceipt
    source, dest, record_id = (_name(body, key) for key in ("source", "dest", "record_id"))
    c = _courier()
    if source not in c.hives() or dest not in c.peers():
        raise HTTPException(404, "register the source store and destination peer first")
    try:
        receipt = CopyReceipt.from_record(body.get("receipt"))
        result = c.reconcile(source, dest, record_id, receipt)
    except Exception as exc:
        raise HTTPException(400, "could not verify or record the courier receipt") from exc
    from .. import audit
    from ..scope import current_workspace
    audit.record("courier.reconcile", detail={"source": source, "dest": dest, "record_id": record_id},
                 user=_t.user_id, workspace=current_workspace() or "default")
    return result.public()


@router.get("/courier/survey")
def courier_survey(hive: str, tags: str = "", limit: int = Query(100, ge=0, lt=2**63),
                         _t: TokenEntry = Depends(require_admin)):
    """What a trip out of this hive would consider, carrying nothing."""
    c = _courier()
    if hive not in c.hives():
        raise HTTPException(404, "register the source store first")
    spec = _spec({"tags": [t for t in tags.split(",") if t], "limit": limit})
    try:
        return {"hive": hive, "records": c.survey(hive, carry=spec)}
    except Exception as e:
        raise HTTPException(400, "could not survey the registered store") from e


@router.post("/courier/deliver")
def courier_deliver(body: dict = Body(...), _t: TokenEntry = Depends(require_admin)):
    """One trip. body: {source, dest, tags?, ids?, limit?}.

    `dest` names a bound store or a registered peer; a destination this courier does not
    already route for is refused rather than built from the request, so a caller cannot
    name a machine the operator never approved.
    """
    source, dest = _name(body, "source"), _name(body, "dest")
    spec = _spec(body)
    c = _courier()
    if dest not in c.destinations():
        raise HTTPException(404, f"no destination {dest!r}; register it first")
    if source not in c.hives():
        raise HTTPException(404, "register the source store first")
    try:
        return c.deliver(source, dest, carry=spec)
    except Exception as e:
        raise HTTPException(400, "could not select the registered source records") from e


@router.post("/courier/broadcast")
def courier_broadcast(body: dict = Body(...), _t: TokenEntry = Depends(require_admin)):
    """One trip per destination. body: {source, dests?, tags?, ids?, limit?}."""
    source = _name(body, "source")
    spec = _spec(body)
    c = _courier()
    dests = body.get("dests")
    if dests is not None:
        if type(dests) is not list:
            raise HTTPException(400, "dests requires a list of literal names")
        dests = [_name({"dest": d}, "dest") for d in dests]
        unknown = [d for d in dests if d not in c.destinations()]
        if unknown:
            raise HTTPException(404, f"no destination(s): {', '.join(unknown)}")
    if source not in c.hives():
        raise HTTPException(404, "register the source store first")
    try:
        return c.broadcast(source, dests, carry=spec)
    except Exception as e:
        raise HTTPException(400, "could not select the registered source records") from e
