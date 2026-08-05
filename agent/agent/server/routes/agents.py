"""
agent.server.routes.agents: route surface for the agentic relational complex over the swarm.

Feed inter-agent messages, then read the monitor (load-bearing agents,
interaction Hodge, alignment, divergence) and the router (query to most relevant
agent). The complex uses the RCF machinery. `agent.agent_complex.record` is
called wherever agents or models message, so this route reads live traffic.
"""
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from starlette.responses import StreamingResponse

from agent.server.auth import TokenEntry, WorkspaceState, is_admin, require_auth, require_workspace

# Verbs that change the world irreversibly. Only an admin may actually execute them (with confirm).
# Everyone with a valid token may still run read/build verbs and may PROPOSE these (confirm=False).
_CONSEQUENTIAL = {"kill"}

router = APIRouter(prefix="/v1")

_MAX_STREAMS = 8            # cap concurrent live streams so a client can't exhaust connections
_streams = {"n": 0}


@router.post("/agents/message")
async def agent_message(body: dict = Body(...)):
    """Append one inter-agent interaction to the live complex. body: {from, to, text}."""
    from agent import agent_complex
    s = body.get("from") or body.get("sender")
    r = body.get("to") or body.get("recipient")
    t = body.get("text")
    if not (s and r and t):
        raise HTTPException(400, "need 'from', 'to', 'text'")
    agent_complex.record(s, r, t)
    return {"ok": True, "n_messages": len(agent_complex.get_live()._msgs)}


@router.get("/agents/monitor")
async def agent_monitor(embed: bool = False):
    """Monitor the live swarm: load-bearing agents (effective resistance), interaction Hodge (coherent/circulating/persistent), deadlock cycles, cross-agent alignment, divergence flags. ?embed=true uses the running model's semantic embeddings (separates hallucination from a topically distinct specialist); otherwise a lexical fallback."""
    from agent import agent_complex
    ef = agent_complex.model_embed_fn() if embed else None
    return agent_complex.get_live().monitor(embed_fn=ef)


@router.post("/agents/route")
async def agent_route(body: dict = Body(...)):
    """Rank which agent(s) a query surfaces, by reweighting. body: {query, top_k?}."""
    from agent import agent_complex
    q = body.get("query")
    if not q:
        raise HTTPException(400, "need 'query'")
    return {"query": q, "agents": agent_complex.get_live().route(q, top_k=int(body.get("top_k", 3)))}


@router.post("/agents/reset")
async def agent_reset():
    """Clear the live agentic complex."""
    from agent import agent_complex
    agent_complex.reset_live()
    return {"ok": True}


@router.get("/agents/activity")
async def agents_activity(scope: str = None, entity: str = None, action: str = None, limit: int = 200):
    """The activity log: every action by every entity, newest first. Filter by scope
    (network|hive|team|worker|model), entity id (exact or prefix, e.g. 'worker:coder'), or action."""
    from agent import activity
    return {"events": activity.get_log().events(scope=scope, entity=entity, action=action, limit=limit)}


@router.get("/agents/events")
async def agents_events(request: Request):
    """Live event stream (SSE). Pushes each activity event the instant it happens: worker
    deploy/remove, model use.open/use.close, hive create/remove - so the UI reflects CLI/API actions
    with no polling. One-way and read-only (the same auth middleware gates it); 15s heartbeat;
    concurrent streams are capped so a client cannot exhaust connections."""
    import asyncio
    import json

    from agent import activity
    if _streams["n"] >= _MAX_STREAMS:
        raise HTTPException(503, "too many live streams")

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=2000)

    def on_event(ev):                                  # called from the (sync) recorder thread
        try:
            loop.call_soon_threadsafe(q.put_nowait, ev)
        except Exception:
            pass                                       # queue full / loop gone -> drop (client refetches)

    activity.get_log().subscribe(on_event)
    _streams["n"] += 1

    async def gen():
        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield "data: " + json.dumps(ev) + "\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"                 # heartbeat: keep-alive + detect a dead client
        finally:
            activity.get_log().unsubscribe(on_event)
            _streams["n"] -= 1

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


@router.get("/agents/usage")
async def agents_usage():
    """Model-usage portal. Per model: when it was instantiated, how long it has run, and its ACTIVE
    concurrent uses (what it is doing right now), plus total uses this session."""
    from agent import activity
    return {"usage": activity.get_log().usage()}


@router.get("/agents/dashboard")
async def agents_dashboard():
    """Full hive-network dashboard snapshot: roster, coordination health, information flow (who ->
    whom + the Hodge draining/circulating read), per-worker load/coherence/alignment, and the NNs
    the LMs forged. Read-only."""
    from agent import hive as hivemod
    from agent.dashboard import hive_dashboard
    return hive_dashboard(hivemod.get_hive())


@router.post("/agents/command")
async def agents_command(body: dict = Body(...), caller: TokenEntry = Depends(require_auth),
                         ws: WorkspaceState = Depends(require_workspace)):
    """Command the hive from the console. body: {command, scope?, confirm?}. Read/inspect verbs run
    freely; CONSEQUENTIAL verbs (kill) return a proposal unless confirm=true - the caller is the
    governor, nothing destructive happens without an explicit confirm. Executing a consequential verb
    (confirm=true) additionally requires admin of the current workspace; a user may propose it but not
    carry it out."""
    cmd = body.get("command")
    if not cmd:
        raise HTTPException(400, "need 'command'")
    verb = cmd.strip().split()[0].lower() if cmd.strip() else ""
    if verb in _CONSEQUENTIAL and bool(body.get("confirm", False)) and not is_admin(caller, ws.name):
        raise HTTPException(403, f"Only an admin of workspace '{ws.name}' may execute '{verb}'. Ask an admin, or "
                                 "omit confirm to get a proposal.")
    # audit: who ran what, in which workspace, with what role - stamped into the live feed + journal
    from agent import activity as _activity
    _activity.record("user:" + (caller.user_id or "local"), "command",
                     detail={"verb": verb, "workspace": ws.name, "role": caller.role_in(ws.name) or "-",
                             "confirm": bool(body.get("confirm", False))})
    from agent import hive as hivemod
    from agent.console import CommandConsole
    from agent.foundry import ModelFoundry
    from agent.reactive_hive import ReactiveHive
    scope = body.get("scope", "hive")
    net = hivemod.get_network()
    h = net.hive(scope.split(":", 1)[1]) if scope.startswith("hive:") else net.hive("default")
    console = CommandConsole(h, reactive=ReactiveHive(h), foundry=ModelFoundry(h))
    return console.command(cmd, scope=scope, confirm=bool(body.get("confirm", False)))


@router.get("/agents/network")
async def agents_network():
    """The hive network registry: every named hive with its roster, plus network totals."""
    from agent import hive as hivemod
    return hivemod.get_network().status()


@router.post("/agents/network/hives")
async def agents_network_create(body: dict = Body(...)):
    """Create a named hive. body: {name}."""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "need 'name'")
    from agent import hive as hivemod
    hivemod.get_network().hive(name)
    return {"ok": True, "created": name, "hives": hivemod.get_network().names()}
