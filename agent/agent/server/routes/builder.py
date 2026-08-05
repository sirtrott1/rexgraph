"""
Agent Builder routes: compose a pipeline from named steps, then execute it.

`AgentBuilder` has run configs since it was written, but only the CLI could execute
one. The Builder screen could compose a config and export it, and its Run button
printed instructions to go use the CLI. These routes give the screen the executor.

  GET  /api/v1/builder/steps      registered step types with their descriptions
  GET  /api/v1/builder/templates  starter configs
  POST /api/v1/builder/run        execute a config against uploaded files

The step list comes from the registry rather than a literal, so a step registered
through `register_step` appears in the UI without editing the frontend.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import asdict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/builder")

TEMPLATES = ("default", "rag", "training", "langgraph")


@router.get("/steps")
async def builder_steps():
    """Registered step types, each with the description its function carries."""
    from agent.builder import AgentBuilder
    steps = []
    for name in AgentBuilder.available_steps():
        doc = (AgentBuilder.step_help(name) or "").strip()
        steps.append({"type": name, "help": doc.split("\n")[0] if doc else ""})
    return {"steps": steps}


@router.get("/templates")
async def builder_templates():
    """Starter configs, so a first pipeline is a pick rather than a blank page."""
    from agent.builder import AgentBuilder
    out = {}
    for name in TEMPLATES:
        with contextlib.suppress(Exception):
            out[name] = AgentBuilder.template(name)
    return {"templates": out}


@router.post("/run")
async def builder_run(
    config: str = Form(...),
    query: str = Form(None),
    workspace: str = Form("default"),
    files: list[UploadFile] = File(None),
):
    """Execute a builder config.

    `config` is the JSON the Builder screen composes and exports, so what runs here
    is the same document the CLI takes. Files are optional: a config whose steps do
    not read documents runs without them.
    """
    try:
        cfg = json.loads(config)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"config is not valid JSON: {e}") from e
    if not isinstance(cfg, dict) or not cfg.get("steps"):
        raise HTTPException(400, "config needs a non-empty 'steps' list")

    from agent.builder import AgentBuilder
    known = set(AgentBuilder.available_steps())
    unknown = [s.get("type") for s in cfg["steps"] if s.get("type") not in known]
    if unknown:
        raise HTTPException(
            400, f"unknown step type(s): {', '.join(map(str, unknown))}. "
                 f"Registered: {', '.join(sorted(known))}")

    paths = []
    try:
        for f in files or []:
            suffix = os.path.splitext(f.filename or "")[1] or ".bin"
            fd, path = tempfile.mkstemp(suffix=suffix, prefix="rexgraph_builder_")
            os.close(fd)
            with open(path, "wb") as fh:
                fh.write(await f.read())
            paths.append(path)
        try:
            result = AgentBuilder(cfg).run(files=paths or None, query=query or None)
        except Exception as e:
            logger.exception("builder run failed")
            raise HTTPException(500, f"run failed: {e}") from e
    finally:
        for p in paths:
            with contextlib.suppress(OSError):
                os.unlink(p)

    out = asdict(result)
    # The accumulated documents and chunks are the pipeline's working state, not a
    # result to ship: they are large and already summarised per step.
    out["n_documents"] = len(out.pop("documents", []) or [])
    out["n_chunks"] = len(out.pop("chunks", []) or [])

    # One step of this pipeline's temporal rex per run, when the workspace asks for
    # it. Stages that errored are labelled as such, so the recorded shape is what
    # actually happened rather than what was composed.
    try:
        from agent import work_recorder as wr
        stages = [s["step_type"] + ("" if s["status"] == "ok" else f"!{s['status']}")
                  for s in out.get("steps", [])]
        rec = wr.record("pipeline-run", stages,
                        lineage_id="pipeline:" + (cfg.get("name") or "rexgraph-agent"),
                        workspace=workspace)
        if rec:
            out["recorded"] = rec
    except Exception:
        logger.debug("pipeline run not recorded", exc_info=True)
    return out
