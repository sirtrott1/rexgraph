"""
agent.server.routes.hive: route surface for the agent swarm (hive).

Endpoints to bring bees up (spawn managed llama.cpp servers or attach live
endpoints), inspect the swarm, route or dispatch a query to a bee, and read the
relational-complex monitor over swarm traffic. Every bee interaction is recorded
into the live complex, so this monitor and `/api/v1/agents/monitor` read the same
flow.
"""
from fastapi import APIRouter, Body, HTTPException

router = APIRouter(prefix="/v1")


@router.get("/hive/status")
async def hive_status(health: bool = False):
    """Return every bee (role/url/model/specialties), the queen and embedder; ?health=true adds a per-bee reachability probe."""
    from agent import hive
    return hive.get_hive().status(check_health=health)


@router.get("/hive/monitor")
async def hive_monitor(embed: bool = False):
    """Return the relational-complex monitor over swarm traffic: load-bearing bees, Hodge disagreement, deadlock cycles, alignment, divergence. ?embed=true uses the embedder bee."""
    from agent import hive
    return hive.get_hive().monitor(embed=embed)


@router.post("/hive/attach")
async def hive_attach(body: dict = Body(...)):
    """Attach an already-running endpoint as a bee. body: {name, url, role?, model?, specialties?}."""
    from agent import hive
    name, url = body.get("name"), body.get("url")
    if not (name and url):
        raise HTTPException(400, "need 'name' and 'url'")
    try:
        b = hive.get_hive().attach(name, url, role=body.get("role", "worker"),
                                   model=body.get("model", ""),
                                   specialties=body.get("specialties") or [])
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "bee": b.public()}


@router.post("/hive/attach-live")
async def hive_attach_live():
    """Discover running inference servers (ollama/vLLM/llama.cpp/…) and attach any new ones."""
    from agent import hive
    added = hive.get_hive().attach_live()
    return {"attached": [b.public() for b in added], "status": hive.get_hive().status()}


@router.get("/hive/plan")
async def hive_plan(budget: float = None):
    """Dry-run auto-composition: from the models on disk and the memory budget, return the queen, workers, and embedder that would fit. Spawns nothing. ?budget overrides the GB budget."""
    from agent import hive
    return hive.get_hive().auto_plan(budget)


@router.post("/hive/auto")
async def hive_auto(body: dict = Body(default={})):
    """Plan and stand up the hive that fits this machine, from the models on disk. body: {budget?}. Blocking; model loads take time. Use /hive/plan to preview first."""
    from agent import hive
    return hive.get_hive().auto((body or {}).get("budget"))


@router.post("/hive/spawn")
async def hive_spawn(body: dict = Body(...)):
    """Bring a bee up as a managed llama.cpp subprocess. body: {name, model_path, role?, specialties?, port?, ctx_size?}. Needs a built llama.cpp binary and the GGUF on disk."""
    from agent import hive
    name, mp = body.get("name"), body.get("model_path")
    if not (name and mp):
        raise HTTPException(400, "need 'name' and 'model_path'")
    try:
        b = hive.get_hive().spawn(name, mp, role=body.get("role", "worker"),
                                  specialties=body.get("specialties") or [],
                                  port=body.get("port"), ctx_size=body.get("ctx_size"))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "bee": b.public(), "status": hive.get_hive().status()}


@router.post("/hive/remove")
async def hive_remove(body: dict = Body(...)):
    """Stop (if managed) and unregister a bee. body: {name}."""
    from agent import hive
    name = body.get("name")
    if not name:
        raise HTTPException(400, "need 'name'")
    ok = hive.get_hive().remove(name)
    return {"ok": ok, "status": hive.get_hive().status()}


@router.post("/hive/route")
async def hive_route(body: dict = Body(...)):
    """Rank bees for a query by specialty and interaction-history reweighting. body: {query, top_k?}."""
    from agent import hive
    q = body.get("query")
    if not q:
        raise HTTPException(400, "need 'query'")
    return {"query": q, "routed": hive.get_hive().route(q, top_k=int(body.get("top_k", 3)))}


@router.post("/hive/dispatch")
async def hive_dispatch(body: dict = Body(...)):
    """Route a query to a bee and ask it. body: {query, sender?, system?}. Records both directions of the exchange into the complex."""
    from agent import hive
    q = body.get("query")
    if not q:
        raise HTTPException(400, "need 'query'")
    return hive.get_hive().dispatch(q, sender=body.get("sender", "user"), system=body.get("system"))


@router.post("/hive/ask")
async def hive_ask(body: dict = Body(...)):
    """Ask one specific bee. body: {name, prompt, sender?, system?}."""
    from agent import hive
    name, prompt = body.get("name"), body.get("prompt")
    if not (name and prompt):
        raise HTTPException(400, "need 'name' and 'prompt'")
    try:
        reply = hive.get_hive().ask(name, prompt, sender=body.get("sender", "user"),
                                    system=body.get("system"))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return {"bee": name, "reply": reply}


@router.post("/hive/down")
async def hive_down():
    """Stop all managed bees and clear the hive."""
    from agent import hive
    hive.get_hive().stop_all()
    return {"ok": True, "status": hive.get_hive().status()}


# profiles: named, switchable, editable hive setups

@router.get("/hive/profiles")
async def hive_profiles():
    """List all hive setups (built-in presets and saved profiles) and which one is active."""
    from agent import hive_config
    s = hive_config.get_store()
    return {"profiles": [p.to_dict() for p in s.list()], "active": s.active_id()}


@router.get("/hive/profiles/{pid}")
async def hive_profile_get(pid: str):
    from agent import hive_config
    p = hive_config.get_store().get(pid)
    if p is None:
        raise HTTPException(404, f"no profile {pid!r}")
    return p.to_dict()


@router.post("/hive/profiles")
async def hive_profile_save(body: dict = Body(...)):
    """Create or update a user profile. body: a profile dict, or {name, base?, ...overrides} to clone an existing preset. Built-ins are not mutated in place; this shadows them."""
    from agent import hive_config
    s = hive_config.get_store()
    if body.get("base") is not None or ("id" not in body and "name" in body and not body.get("compose")):
        p = s.create(body.get("name", "My setup"), base=body.get("base"),
                     **{k: v for k, v in body.items() if k not in ("name", "base")})
    else:
        p = s.save(hive_config.HiveProfile.from_dict(body))
    return {"ok": True, "profile": p.to_dict()}


@router.delete("/hive/profiles/{pid}")
async def hive_profile_delete(pid: str):
    """Delete a user profile (or a user override of a preset). Presets themselves persist."""
    from agent import hive_config
    return {"ok": hive_config.get_store().delete(pid)}


@router.post("/hive/profiles/{pid}/apply")
async def hive_profile_apply(pid: str, body: dict = Body(default={})):
    """Switch to a setup: stop the current swarm and bring the hive up per this profile, and set it active. Blocking; model loads and managed spawns take time."""
    from agent import hive_config
    try:
        return hive_config.get_store().apply(pid, reset=(body or {}).get("reset", True))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/hive/profiles/active")
async def hive_profile_set_active(body: dict = Body(...)):
    """Mark a profile active without applying it (pointer only). body: {id}."""
    from agent import hive_config
    hive_config.get_store().set_active(body.get("id"))
    return {"ok": True, "active": hive_config.get_store().active_id()}
