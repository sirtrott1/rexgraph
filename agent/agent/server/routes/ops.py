"""
agent.server.routes.ops: route surface for lifecycle operations.

One endpoint set covers every phase (serve/train/build/deploy/test). Each phase
reads the active hive setup and is recorded as a persistent run. The Operations
tab in the UI and the `rexgraph-ops` CLI both call these endpoints.
"""
from fastapi import APIRouter, Body, HTTPException

router = APIRouter(prefix="/v1")


@router.get("/ops/phases")
async def ops_phases():
    """List the available lifecycle phases (built-in and registered extensions)."""
    from agent import lifecycle
    return {"phases": lifecycle.phases()}


@router.get("/ops/inventory")
async def ops_inventory():
    """List the wired, runnable components: attention (relational/standard), optimizer (hodge/adam/...), model, each with native/default/available. Drives the choices shown in Setups."""
    from rexgraph import nn
    return nn.inventory()


@router.get("/ops/compute")
async def ops_compute():
    """The execution layer: available backends, current thread width + preferred backend, registered
    ops, and the active setup's compute config. Drives the compute controls shown in Setups."""
    from agent import hive_config
    from rexgraph import compute
    prof = hive_config.get_store().active()
    return {"inventory": compute.inventory(),
            "active_setup": (prof.id if prof else None),
            "setup_compute": (prof.compute.__dict__ if prof else None)}


@router.post("/ops/compute")
async def ops_compute_set(body: dict = Body(...)):
    """Tune the execution layer and persist it into a setup. body: {threads?, backend?, profile?}.
    Writes the compute config into the setup (shadowing a built-in) and applies it now. Every
    subsequent operation honors it."""
    from agent import hive_config
    from rexgraph import compute
    store = hive_config.get_store()
    base = store.get(body["profile"]) if body.get("profile") else store.active()
    if base is None:
        raise HTTPException(404, "no setup to tune")
    d = base.to_dict(); comp = dict(d.get("compute") or {})
    if "threads" in body: comp["threads"] = body["threads"]
    if "backend" in body: comp["backend"] = body["backend"]
    d["compute"] = comp
    prof = store.save(hive_config.HiveProfile.from_dict(d))
    eff = compute.apply_config(comp)
    return {"setup": prof.id, "compute": comp, "effective": eff}


@router.post("/ops/run")
async def ops_run(body: dict = Body(...)):
    """Run a phase. body: {phase, profile?, params?}. Reads the active setup unless `profile` is given. Blocking; serve/train take time. The run is persisted throughout."""
    from agent import lifecycle
    phase = body.get("phase")
    if not phase:
        raise HTTPException(400, "need 'phase'")
    try:
        rl = lifecycle.run(phase, profile_id=body.get("profile"),
                           background=bool(body.get("background", False)),
                           **(body.get("params") or {}))
    except KeyError as e:
        raise HTTPException(404, str(e))
    return rl.to_dict()


@router.get("/ops/runs")
async def ops_runs(limit: int = 30, phase: str = None):
    """List recent runs (most recent first), optionally filtered by phase."""
    from agent import lifecycle
    return {"runs": [r.to_dict() for r in lifecycle.get_store().list(limit=limit, phase=phase)]}


@router.get("/ops/runs/{run_id}")
async def ops_run_get(run_id: str):
    """Return one run's full record: status, params, timestamped step log, result/error."""
    from agent import lifecycle
    rl = lifecycle.get_store().get(run_id)
    if rl is None:
        raise HTTPException(404, f"no run {run_id!r}")
    return rl.to_dict()
