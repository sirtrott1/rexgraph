"""
agent.server.routes.ml: route surface for model building.

Select an archetype, set its parameters, point it at data (files, parquet, .rex,
or a TrustGraph knowledge core), and train it in single, multistep, or fusion
mode with any rexgraph.nn optimizer. Results persist through the rexgraph IO
layer and RCDB. Archetypes and optimizers come from the registries, so the UI
and CLI reflect whatever is wired.
"""
from fastapi import APIRouter, Body, HTTPException

router = APIRouter(prefix="/v1")


@router.get("/ml/archetypes")
async def ml_archetypes():
    """List every model archetype with its use-case, data kind, and tunable params."""
    from agent import models
    return {"archetypes": models.list_archetypes()}


@router.get("/ml/components")
async def ml_components():
    """List the swappable rexgraph.nn components (optimizers, attention) the models build on."""
    import rexgraph.nn as nn
    return nn.inventory()


@router.post("/ml/run")
async def ml_run(body: dict = Body(...)):
    """Build and train a model. body: {archetype, params?, data?(path), mode?(single|multistep|fusion), optimizer?, steps?, save_to?, stages?, specs?, fusion?, device?}. Blocking; keep steps modest from the UI, or use the `train` lifecycle phase for a streamed/background run."""
    from agent import models
    arch = body.get("archetype")
    if not arch:
        raise HTTPException(400, "need 'archetype'")
    # `data` is read and `save_to` is written as the server user, so both are held to
    # the same allow-list every other caller-supplied path is: save_to reached
    # Path(expanduser(p)).mkdir(parents=True) and then wrote weights there.
    from ..handles import path_allowed
    for field in ("data", "save_to"):
        value = body.get(field)
        if value is None or value == "":
            continue
        # A non-string skipped the check entirely and then crashed at the sink, which
        # reported a 500 for what is a bad request.
        if not isinstance(value, str):
            raise HTTPException(400, f"'{field}' must be a path")
        if not path_allowed(value):
            raise HTTPException(403, f"'{field}' is outside the allowed directories")
    specs = None
    if body.get("specs"):                       # [[name, {params}], ...] -> tuples
        specs = [(s[0], s[1] if len(s) > 1 else {}) for s in body["specs"]]
    try:
        result = models.run(arch, params=body.get("params"), data=body.get("data"),
                            mode=body.get("mode", "single"), optimizer=body.get("optimizer", "auto"),
                            steps=int(body.get("steps", 150)), lr=body.get("lr"),
                            seed=int(body.get("seed", 0)), stages=body.get("stages"), specs=specs,
                            fusion=body.get("fusion", "ensemble"), device=body.get("device", "cpu"),
                            save_to=body.get("save_to"))
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    # structural training diagnosis: the trajectory is an eval METRIC (higher is better); negate it
    # into a loss-proxy so the monitor reads a descent and names any issue + cause.
    traj = result.get("trajectory") or []
    if traj:
        try:
            from agent.training_monitor import diagnose
            result["diagnosis"] = diagnose([-float(m) for m in traj])
        except Exception:
            pass
    return result


@router.post("/ml/ingest")
async def ml_ingest(body: dict = Body(...)):
    """Ingest a TrustGraph knowledge core into a relational complex and a trainable bundle, and optionally train on it and/or catalogue it in the RCDB. body: {triples?([[s,p,o],..]) | url+flow, labels? ({entity:class}), train?(bool), archetype?, steps?, rcdb_uri?, name?}."""
    from agent import models
    triples = body.get("triples")
    if not (triples or body.get("flow")):
        raise HTTPException(400, "need 'triples' or 'url'+'flow'")
    if body.get("url"):
        from agent.server.dbguard import check_outbound_url
        check_outbound_url(str(body["url"]))
    try:
        bundle = models.bundle_from_core(triples, url=body.get("url"), flow=body.get("flow"),
                                         labels=body.get("labels"))
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}") from e
    out = {"n_nodes": bundle.meta["n_nodes"], "n_classes": bundle.meta["n_classes"],
           "entities": bundle.meta.get("entity_names", [])[:20]}
    if body.get("train"):
        out["train"] = models.run(body.get("archetype", "hgnn"), data=bundle,
                                   optimizer=body.get("optimizer", "auto"),
                                   steps=int(body.get("steps", 150)))
    if body.get("rcdb_uri"):
        # A caller-named store URI writes outside the workspace-scoped view that
        # default_store() provides, and trustgraph opened it directly. Inside a request
        # the workspace store is used and the named one is ignored; outside a request
        # the operator's own choice stands.
        from agent.server.scope import scoping_active
        store = None
        if scoping_active():
            from agent.rcdb import default_store
            store = default_store()
        out["rcdb"] = models.core_to_rcdb(triples, url=body.get("url"), flow=body.get("flow"),
                                          uri=body["rcdb_uri"], store=store,
                                          name=body.get("name", "knowledge_core"))
    return out
