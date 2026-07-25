"""
agent.lifecycle - shared operations spine for the agent platform.

One interface for every phase of an agent's life: serve (bring the swarm up), train (optimizer,
device, data export), build (assemble an agent pipeline), deploy (generate a container bundle),
test (smoke-verify the stack). Each phase reads the active hive profile (`hive_config`) for its
configuration, runs the underlying function, and records a persistent RunLog, so serve/train/
build/deploy/test are driven the same way from the CLI, the API, and the UI, with logging.

`@register_phase("name")` adds a phase (a custom deploy target, an eval suite, a fine-tune loop)
that works everywhere `run()` is exposed. The spine owns dispatch, logging, and provenance; the
handler owns the work.
"""
from __future__ import annotations

import json
import logging
import os
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("rexgraph.lifecycle")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _runs_dir() -> Path:
    base = Path(os.environ.get("REXGRAPH_CONFIG_DIR", Path.home() / ".config" / "rexgraph"))
    return base / "runs"


# the run record

@dataclass
class RunLog:
    id: str
    phase: str
    profile: Optional[str] = None
    status: str = "running"                 # running | ok | error
    started: str = ""
    ended: Optional[str] = None
    params: dict = field(default_factory=dict)
    steps: List[dict] = field(default_factory=list)     # [{t, msg}]
    result: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class RunContext:
    """Handed to a phase handler: the resolved profile, the call params, and a `log()` that both
    emits to the logger and appends a timestamped step to the RunLog (the audit trail)."""

    def __init__(self, profile, params: dict, run: RunLog):
        self.profile = profile
        self.params = params
        self.run = run

    def log(self, msg: str, *, persist: bool = True):
        self.run.steps.append({"t": _now(), "msg": str(msg)})
        logger.info("[%s] %s", self.run.phase, msg)
        if persist:                                  # so a polling client streams progress live
            try:
                get_store().save(self.run)
            except Exception:
                pass


PHASES: Dict[str, Dict] = {}


def register_phase(name: str, description: str = ""):
    """Register a lifecycle phase handler `fn(ctx: RunContext) -> dict`. The returned dict is the
    run result. Registering shadows a built-in of the same name (custom deploy/eval/train)."""
    def deco(fn: Callable):
        PHASES[name] = {"fn": fn, "description": description or (fn.__doc__ or "").strip()}
        return fn
    return deco


def phases() -> List[dict]:
    return [{"name": n, "description": p["description"]} for n, p in sorted(PHASES.items())]


# run store

class RunStore:
    def __init__(self, directory: Optional[Path] = None, keep: int = 200):
        self.dir = directory or _runs_dir()
        self.keep = keep

    def _path(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.json"

    def save(self, run: RunLog):
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path(run.id).write_text(json.dumps(run.to_dict(), indent=2))
        self._prune()

    def get(self, run_id: str) -> Optional[RunLog]:
        f = self._path(run_id)
        if not f.exists():
            return None
        try:
            return RunLog(**json.loads(f.read_text()))
        except Exception:
            return None

    def list(self, limit: int = 50, phase: Optional[str] = None) -> List[RunLog]:
        if not self.dir.exists():
            return []
        runs = []
        for f in self.dir.glob("*.json"):
            try:
                runs.append(RunLog(**json.loads(f.read_text())))
            except Exception:
                continue
        runs.sort(key=lambda r: r.started, reverse=True)
        if phase:
            runs = [r for r in runs if r.phase == phase]
        return runs[:limit]

    def _prune(self):
        files = sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files[self.keep:]:
            try:
                f.unlink()
            except OSError:
                pass


_STORE: Optional[RunStore] = None


def get_store() -> RunStore:
    global _STORE
    if _STORE is None:
        _STORE = RunStore()
    return _STORE


def reset_store():
    global _STORE
    _STORE = None


# the one entry point

_counter = 0


def _run_id(phase: str) -> str:
    global _counter
    _counter += 1
    # time-ordered id (sortable) + phase + counter; no randomness needed
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{phase}-{_counter:03d}"


def _apply_compute(ctx: "RunContext") -> None:
    """Apply the active setup's execution-layer config (CPU thread width + preferred backend) before
    the phase runs, so every operation on every surface honors it. Process-global and run-logged."""
    spec = getattr(ctx.profile, "compute", None) if ctx.profile else None
    if spec is None:
        return
    try:
        from rexgraph import compute as _compute
        eff = _compute.apply_config({"threads": getattr(spec, "threads", None),
                                     "backend": getattr(spec, "backend", "auto")})
        ctx.log(f"compute: threads={eff['threads'] or 'all'} backend={eff['backend']} "
                f"(available: {', '.join(eff['available'])})")
    except Exception as e:                       # never let tuning break an operation
        ctx.log(f"compute config skipped ({type(e).__name__}: {e})")


def _phase_device(ctx: "RunContext") -> str:
    """Resolve the training/inference device for a phase, bridging the setup's execution layer to
    the model lifecycle: an explicit ``device`` param wins; otherwise the active setup's
    ``ComputeSpec.backend`` (default 'auto') is resolved through ``rexgraph.nn.pick_device`` - so
    'auto' rides the compute stack's recommended backend (GPU when usable), 'cpu' forces CPU, and a
    GPU request on a CPU-only host degrades cleanly. Returns a device string (never None)."""
    dev = ctx.params.get("device")
    if dev is None:
        spec = getattr(ctx.profile, "compute", None) if ctx.profile else None
        dev = getattr(spec, "backend", None) if spec else None       # 'auto'/'cpu'/'cuda'/... or None
    try:
        from rexgraph.nn import optim
        return optim.pick_device(dev)
    except Exception:                            # torch/compute absent: safe CPU fallback
        return dev if isinstance(dev, str) and dev not in ("auto", None) else "cpu"


def _execute(ctx: "RunContext"):
    rl, phase = ctx.run, ctx.run.phase
    try:
        ctx.log(f"phase '{phase}' start (profile={rl.profile or 'none'})")
        _apply_compute(ctx)                      # honor the setup's execution-layer tuning
        rl.result = PHASES[phase]["fn"](ctx) or {}
        rl.status = "ok"
        ctx.log(f"phase '{phase}' ok")
    except Exception as e:
        rl.status = "error"
        rl.error = f"{type(e).__name__}: {e}"
        ctx.log(f"phase '{phase}' FAILED - {rl.error}")
        logger.debug("lifecycle %s failed\n%s", phase, traceback.format_exc())
    finally:
        rl.ended = _now()
        get_store().save(rl)


def run(phase: str, *, profile_id: Optional[str] = None, background: bool = False,
        **params) -> RunLog:
    """Execute a lifecycle phase for the active (or named) profile, with run-logging. The single
    call every surface (CLI/API/UI) goes through: dispatch, provenance, and audit in one place.
    With `background=True` it returns immediately (status 'running') and the phase runs in a daemon
    thread, persisting each logged step so a client can poll `get(id)` and stream progress. Returns
    the RunLog (inspect `.status`, `.result`, `.error`, `.steps`)."""
    if phase not in PHASES:
        raise KeyError(f"unknown phase {phase!r} (have: {', '.join(sorted(PHASES))})")
    from agent import hive_config
    store = hive_config.get_store()
    profile = store.get(profile_id) if profile_id else store.active()
    rl = RunLog(id=_run_id(phase), phase=phase, profile=(profile.id if profile else None),
                started=_now(), params=dict(params))
    ctx = RunContext(profile, params, rl)
    get_store().save(rl)                    # persist as running (so long jobs are visible)
    if background:
        import threading
        threading.Thread(target=_execute, args=(ctx,), daemon=True).start()
        return rl
    _execute(ctx)
    return rl


# built-in phases (wired to the real underlying functions)

@register_phase("serve", "Bring the hive up per the active setup (compose/attach the swarm).")
def _serve(ctx: RunContext) -> dict:
    if ctx.profile is None:
        raise RuntimeError("no active setup - pick one in Setups (or pass profile_id)")
    from agent import hive_config
    ctx.log(f"applying setup '{ctx.profile.id}' (compose={ctx.profile.compose})")
    res = hive_config.get_store().apply(ctx.profile.id)
    n = res.get("status", {}).get("n_bees", 0)
    ctx.log(f"hive up: {n} bee(s)")
    return {"n_bees": n, "spawned": res.get("spawned"), "attached": res.get("attached"),
            "status": res.get("status")}


@register_phase("train", "Build + train a model archetype (mlp/cnn/lm/hgnn) with your optimizer.")
def _train(ctx: RunContext) -> dict:
    """Build and train any model archetype on your data (file/parquet/.rex/synthetic), with the
    active setup's optimizer (HodgeAdam by default). mode ∈ {single, multistep, fusion}. Streams the
    loss into the run log and can checkpoint through the rexgraph IO layer (`save_to`)."""
    from agent import models
    p = ctx.profile
    arch = ctx.params.get("archetype", "mlp")
    optimizer = ctx.params.get("optimizer") or (p.optimizer if p else None) or "hodge"
    steps = int(ctx.params.get("steps", 200))
    reserved = {"archetype", "optimizer", "steps", "data", "mode", "save_to", "device", "lr", "seed"}
    params = {k: v for k, v in ctx.params.items() if k not in reserved} or None

    def on_step(i, loss, total):
        if i % max(1, total // 10) == 0 or i == total - 1:
            ctx.log(f"step {i+1}/{total}  loss {loss:.4f}")

    device = _phase_device(ctx)                  # setup ComputeSpec.backend -> resolved torch device
    ctx.log(f"train archetype={arch} optimizer={optimizer} steps={steps} device={device} "
            f"data={ctx.params.get('data') or 'synthetic'}")
    res = models.run(arch, params=params, data=ctx.params.get("data"),
                     mode=ctx.params.get("mode", "single"), optimizer=optimizer, steps=steps,
                     lr=ctx.params.get("lr"), device=device,
                     save_to=ctx.params.get("save_to"), on_step=on_step)
    ctx.log(f"{arch}: {res.get('metric_name')} {res.get('metric')}"
            + (f" · saved {res['saved']}" if res.get("saved") else ""))
    return res


@register_phase("ingest", "TrustGraph knowledge core -> relational complex -> trainable bundle + RCDB.")
def _ingest(ctx: RunContext) -> dict:
    """Ingest a TrustGraph knowledge core (triples, or a live flow) into a relational complex, make
    it a trainable bundle, and optionally train on it and catalogue it in the RCDB. Runs the
    DB -> core -> complex -> {train, store} pipeline as one operation."""
    from agent import models
    triples = ctx.params.get("triples")
    labels = ctx.params.get("labels")
    if not (triples or ctx.params.get("flow")):
        return {"skipped": "need 'triples' or 'url'+'flow'"}
    ctx.log("ingesting knowledge core -> relational complex")
    bundle = models.bundle_from_core(triples, url=ctx.params.get("url"),
                                     flow=ctx.params.get("flow"), labels=labels)
    ctx.log(f"complex: {bundle.meta['n_nodes']} entities, {bundle.meta['n_classes']} classes")
    out = {"n_nodes": bundle.meta["n_nodes"], "n_classes": bundle.meta["n_classes"]}
    if ctx.params.get("train"):
        out["train"] = models.run(ctx.params.get("archetype", "hgnn"), data=bundle,
                                  steps=int(ctx.params.get("steps", 150)))
        ctx.log(f"trained {ctx.params.get('archetype','hgnn')} -> {out['train'].get('metric')}")
    if ctx.params.get("rcdb_uri"):
        out["rcdb"] = models.core_to_rcdb(triples, uri=ctx.params["rcdb_uri"],
                                          name=ctx.params.get("name", "knowledge_core"))
        ctx.log(f"catalogued in RCDB: {out['rcdb']}")
    return out


@register_phase("pipeline", "End-to-end: source -> complex -> RCDB -> train -> predict -> hive worker -> SQL sink.")
def _pipeline(ctx: RunContext) -> dict:
    """Thread the data-to-agent flow as one operation. Every stage is optional and driven by params:

        source | triples/url+flow   -> a relational-complex DataBundle (rexgraph.io or TrustGraph)
        rcdb_uri                     -> catalogue the complex in the RCDB
        archetype (+steps/optimizer) -> train a model, checkpointed through the IO layer
        predict                      -> run the trained model and evaluate
        worker                       -> register the model as a capability worker in the hive
        sink                         -> write predictions back to a SQL database (rexgraph.io)

    Each configured stage streams into the run log. Returns a summary of what ran."""
    import os
    from agent import models
    p = ctx.profile
    out: dict = {"stages": []}

    # 1-3: source -> relational-complex bundle (a rexgraph.io/SQL source, or a TrustGraph core)
    if ctx.params.get("triples") or ctx.params.get("flow"):
        ctx.log("stage: TrustGraph core -> relational complex")
        bundle = models.bundle_from_core(ctx.params.get("triples"), url=ctx.params.get("url"),
                                         flow=ctx.params.get("flow"), labels=ctx.params.get("labels"))
        out["stages"].append("trustgraph")
    elif ctx.params.get("source"):
        ctx.log(f"stage: load {ctx.params['source']} -> relational-complex bundle")
        bundle = models.load_bundle(ctx.params["source"])
        out["stages"].append("source")
    else:
        return {"skipped": "need 'source' (a rexgraph.io/SQL source) or 'triples'/'url'+'flow'"}
    out["n"] = int(len(bundle.y)) if bundle.y is not None else 0
    ctx.log(f"complex bundle: kind={bundle.kind} n={out['n']}")

    # 4: catalogue the complex in the RCDB
    if ctx.params.get("rcdb_uri") and bundle.kind == "hypergraph":
        out["rcdb"] = models.to_rcdb(bundle, uri=ctx.params["rcdb_uri"],
                                     name=ctx.params.get("name", "pipeline_complex"))
        out["stages"].append("rcdb")
        ctx.log(f"stage: catalogued complex in RCDB -> {out['rcdb']}")

    # 5: train a model, checkpointed through the IO layer so predict/worker can reuse it
    ckpt = ctx.params.get("save_to")
    if ctx.params.get("archetype"):
        arch = ctx.params["archetype"]
        opt = ctx.params.get("optimizer") or (p.optimizer if p else None) or "hodge"
        if (ctx.params.get("worker") or ctx.params.get("predict")) and not ckpt:
            base = os.environ.get("REXGRAPH_CONFIG_DIR") or os.path.expanduser("~/.config/rexgraph")
            ckpt = os.path.join(base, "pipeline_models", ctx.params.get("name", "model"))
        device = _phase_device(ctx)              # setup ComputeSpec.backend -> resolved torch device
        ctx.log(f"stage: train {arch} (optimizer={opt}, steps={ctx.params.get('steps', 150)}, "
                f"device={device})")
        res = models.run(arch, data=bundle, optimizer=opt, steps=int(ctx.params.get("steps", 150)),
                         device=device, save_to=ckpt)
        out["train"] = {"metric": res.get("metric"), "saved": res.get("saved")}
        out["stages"].append("train")
        ctx.log(f"trained {arch}: metric {res.get('metric')}")

    # 6: predict / evaluate on the bundle (or a separate predict_source)
    preds = None
    if ctx.params.get("predict") and ckpt:
        pr = models.predict(ckpt, ctx.params.get("predict_source") or bundle,
                            split=ctx.params.get("split"))
        preds = pr["predictions"]
        out["predict"] = {"n": pr["n"], "metric": pr["metric"]}
        out["stages"].append("predict")
        ctx.log(f"stage: predict n={pr['n']} metric={pr['metric']}")

    # 7: register the trained model as a capability worker in the hive
    if ctx.params.get("worker") and ckpt:
        from agent import hive
        hive.get_hive().add_model(ctx.params["worker"], ckpt, capability="predict",
                                  specialties=ctx.params.get("specialties"))
        out["worker"] = ctx.params["worker"]
        out["stages"].append("worker")
        ctx.log(f"stage: registered model as hive worker '{ctx.params['worker']}'")

    # 8: write predictions back to a SQL database through rexgraph.io
    if ctx.params.get("sink") and preds is not None:
        try:
            import pandas as pd
            import rexgraph.io as rio
            eng = rio.get_engine(ctx.params["sink"])
            col = preds.reshape(len(preds), -1)[:, 0]
            pd.DataFrame({"row": range(len(preds)), "prediction": col}).to_sql(
                ctx.params.get("sink_table", "predictions"), eng, if_exists="replace", index=False)
            out["sink"] = ctx.params.get("sink_table", "predictions")
            out["stages"].append("sink")
            ctx.log(f"stage: wrote {len(preds)} predictions -> {ctx.params['sink']}")
        except Exception as e:
            ctx.log(f"sink skipped ({e})")

    ctx.log(f"pipeline complete: {' -> '.join(out['stages'])}")
    return out


@register_phase("bench", "Benchmark the optimizer (HodgeAdam vs Adam/AdamW/SGD) on a recognized task.")
def _bench(ctx: RunContext) -> dict:
    """Run an optimizer benchmark, or a fair lr-tuned A/B. params: benchmark (ill-cond / mnist /
    fashion-mnist / cifar10 / matrix-completion), optimizer, steps, ab (bool), optimizers (for A/B).
    Streams progress into the run log."""
    from agent import benchmarks
    name = ctx.params.get("benchmark", "ill-cond")
    steps = int(ctx.params.get("steps", 200))

    def on_step(label, i, loss, total):
        if i % max(1, total // 10) == 0 or i == total - 1:
            ctx.log(f"{label} step {i+1}/{total} loss {loss:.4f}")

    if ctx.params.get("ab"):
        opts = tuple(ctx.params.get("optimizers") or ("hodge", "adam"))
        ctx.log(f"A/B {name}: {' vs '.join(opts)} (steps={steps})")
        res = benchmarks.benchmark_ab(name, optimizers=opts, steps=steps, on_step=on_step)
        ctx.log(f"verdict: {res.get('verdict')}")
        return res
    p = ctx.profile
    opt = ctx.params.get("optimizer") or (p.optimizer if p else None) or "hodge"
    ctx.log(f"benchmark {name} optimizer={opt} steps={steps}")
    res = benchmarks.run_benchmark(name, optimizer=opt, steps=steps, on_step=on_step)
    ctx.log(f"{name}: eval_final={res.get('eval_final')}")
    return res


@register_phase("finetune", "LoRA-fine-tune a real HF model with your optimizer, A/B vs Adam.")
def _finetune(ctx: RunContext) -> dict:
    """Fine-tune a Hugging Face model (default Qwen2.5-0.5B-Instruct) with the setup's optimizer
    (HodgeAdam by default) against Adam on a held-out eval split, streaming both loss curves.
    Produces a loadable LoRA adapter. Needs the [finetune] extra; returns a skip message if it is
    absent."""
    from agent import finetune
    dep = finetune.deps_available()
    if not dep["ready"]:
        ctx.log(f"fine-tune deps missing ({', '.join(dep['missing'])}). Install: {dep['need']}")
        return {"skipped": dep["need"], "deps": dep}
    p = ctx.profile
    optimizer = ctx.params.get("optimizer") or (p.optimizer if p else None) or "hodge"
    model_id = ctx.params.get("model_id", finetune.DEFAULT_MODEL)
    steps = int(ctx.params.get("steps", 60))
    ab = str(ctx.params.get("ab", "true")).lower() not in ("false", "0", "no")
    hp = {}
    for k in ("lora_r", "seq_len", "batch", "seed", "device", "dataset", "text_field",
              "instruction_field", "response_field", "split", "data_limit", "full", "lr"):
        if k in ctx.params:
            v = ctx.params[k]
            if k in ("lora_r", "seq_len", "batch", "seed", "data_limit"):
                v = int(v)
            elif k == "lr":
                v = float(v)
            elif k == "full":
                v = str(v).lower() not in ("false", "0", "no")
            hp[k] = v
    hp.setdefault("device", _phase_device(ctx))  # setup ComputeSpec.backend -> resolved torch device

    def on_step(lbl, i, loss, total):
        if i % 5 == 0 or i == total - 1:
            ctx.log(f"[{lbl}] step {i+1}/{total}  loss {loss:.4f}")

    if ab:
        ctx.log(f"A/B fine-tune {model_id}: {optimizer} vs adam ({steps} steps each)")
        res = finetune.finetune_ab(model_id=model_id, optimizers=(optimizer, "adam"),
                                   steps=steps, on_step=on_step, **hp)
        if res.get("ab"):
            ctx.log(f"verdict (held-out eval): {res['verdict']}  "
                    f"| eval {res.get('eval_losses')} | train {res.get('train_losses')}")
        return res
    ctx.log(f"fine-tune {model_id} with {optimizer} ({steps} steps)")
    return finetune.finetune(model_id=model_id, optimizer=optimizer, steps=steps,
                             on_step=on_step, **hp)


@register_phase("build", "Assemble an agent pipeline (builder steps) - returns the plan/result.")
def _build(ctx: RunContext) -> dict:
    from agent import builder
    steps = ctx.params.get("steps")
    available = sorted(getattr(builder, "_STEPS", {}).keys())
    if not steps:
        ctx.log("no steps given - returning the available builder steps (dry plan)")
        return {"available_steps": available, "planned": False}
    ctx.log(f"building pipeline: {' -> '.join(steps)}")
    return {"planned_steps": steps, "available_steps": available, "planned": True}


@register_phase("deploy", "Generate a deployable container bundle seeded from the active setup.")
def _deploy(ctx: RunContext) -> dict:
    from agent import deploy
    p = ctx.profile
    # seed the deployment from the setup: talk to the queen, carry the auth posture
    model_url = ctx.params.get("model_url", "")
    if not model_url:
        try:
            from agent import hive
            q = hive.get_hive().queen
            model_url = q.url if q else ""
        except Exception:
            pass
    spec = deploy.DeploymentSpec(
        name=ctx.params.get("name", f"rexgraph-{p.id}" if p else "rexgraph-agent"),
        mode=ctx.params.get("mode", "service"),
        model_url=model_url,
        insecure=bool(ctx.params.get("insecure", False)),
    ).normalized()
    ctx.log(f"generating bundle '{spec.name}' (mode={spec.mode}, model_url={'set' if model_url else 'none'})")
    bundle = deploy.generate_bundle(spec)
    ctx.log(f"bundle: {len(bundle)} file(s) - {', '.join(sorted(bundle)[:6])}")
    return {"name": spec.name, "files": sorted(bundle.keys()), "n_files": len(bundle)}


@register_phase("test", "Smoke-verify the stack: core, optimizer, hive, and the active setup.")
def _test(ctx: RunContext) -> dict:
    checks = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        ctx.log(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")

    check("core import", lambda: (__import__("rexgraph") and (True, "rexgraph core loaded")))
    check("optimizer", lambda: (True, ", ".join(__import__("rexgraph.nn.optim", fromlist=["training_backends"]).training_backends()["devices"])))
    def _hive():
        from agent import hive
        s = hive.get_hive().status()
        return True, f"{s['n_bees']} bee(s), queen={s['queen']}"
    check("hive", _hive)
    check("active setup", lambda: ((ctx.profile is not None), (ctx.profile.id if ctx.profile else "none selected")))
    passed = sum(1 for c in checks if c["ok"])
    return {"passed": passed, "total": len(checks), "checks": checks,
            "ok": passed == len(checks)}


def main(argv=None):
    """CLI: `python -m agent.lifecycle <phase|phases|runs|show ID> [--profile ID] [--k v ...]`."""
    import argparse
    ap = argparse.ArgumentParser(prog="rexgraph-ops", description=(
        "The agent lifecycle - serve/train/build/deploy/test, one interface, logged."))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("phases", help="list available phases")
    r = sub.add_parser("run", help="run a phase"); r.add_argument("phase")
    r.add_argument("--profile", default=None)
    r.add_argument("--set", action="append", default=[], metavar="k=v", help="phase param, repeatable")
    rs = sub.add_parser("runs", help="recent runs"); rs.add_argument("--phase", default=None)
    rs.add_argument("--limit", type=int, default=20)
    sh = sub.add_parser("show", help="show one run"); sh.add_argument("id")
    cc = sub.add_parser("compute", help="show or tune the execution layer (threads/backend)")
    cc.add_argument("--threads", type=int, default=None, help="CPU parallel width (omit = all cores)")
    cc.add_argument("--backend", default=None, help="auto|cpu|openmp|cuda|mps")
    cc.add_argument("--profile", default=None, help="setup to tune (default: active)")
    # phase shortcuts: `lifecycle serve`, `lifecycle test`, ...
    for ph in ("serve", "train", "ingest", "pipeline", "bench", "finetune", "build", "deploy", "test"):
        sp = sub.add_parser(ph, help=f"run the '{ph}' phase")
        sp.add_argument("--profile", default=None)
        sp.add_argument("--set", action="append", default=[], metavar="k=v")
    a = ap.parse_args(argv)

    def _params(pairs):
        out = {}
        for kv in pairs or []:
            k, _, v = kv.partition("=")
            out[k] = v
        return out

    if a.cmd == "phases":
        for p in phases():
            print(f"  {p['name']:8s} {p['description']}")
        return
    if a.cmd == "runs":
        for rl in get_store().list(limit=a.limit, phase=a.phase):
            print(f"  {rl.id:28s} {rl.status:7s} profile={rl.profile}")
        return
    if a.cmd == "show":
        rl = get_store().get(a.id)
        if rl is None:
            print("no such run"); return
        print(json.dumps(rl.to_dict(), indent=2)); return
    if a.cmd == "compute":
        from rexgraph import compute as _compute
        from agent import hive_config
        store = hive_config.get_store()
        if a.threads is None and a.backend is None:                      # info
            prof = store.active()
            print(json.dumps({"inventory": _compute.inventory(),
                              "active_setup": (prof.id if prof else None),
                              "setup_compute": (prof.compute.__dict__ if prof else None)}, indent=2))
            return
        base = store.get(a.profile) if a.profile else store.active()     # tune -> persist into setup
        if base is None:
            print("no setup to tune (create one first)"); return
        d = base.to_dict(); comp = dict(d.get("compute") or {})
        if a.threads is not None: comp["threads"] = a.threads
        if a.backend is not None: comp["backend"] = a.backend
        d["compute"] = comp
        prof = store.save(hive_config.HiveProfile.from_dict(d))          # shadows a built-in
        eff = _compute.apply_config(comp)
        print(f"setup '{prof.id}' compute -> threads={eff['threads'] or 'all'} backend={eff['backend']}")
        return
    phase = a.phase if a.cmd == "run" else a.cmd
    rl = run(phase, profile_id=a.profile, **_params(a.set))
    print(f"{rl.id}  ->  {rl.status}")
    for s in rl.steps:
        print(f"  {s['msg']}")
    if rl.result:
        print(json.dumps(rl.result, indent=2))


if __name__ == "__main__":
    main()
