"""agent.foundry: language models that forge neural networks into a hive they control.

A ModelFoundry lets the LM bees BUILD neural networks live on data and register each trained model
as a worker bee. That makes a hive HIERARCHY: language models on top, the networks they built as
workers beneath. The LMs then drive the NNs with hive.invoke(). Every NN bee is placement-aware:
it trains and serves on a chosen device, resolved through rexgraph.compute (a live-probed GPU/iGPU
when available, else a CPU core), so a multi-core worker team and an iGPU can be filled at once on a
shared-memory machine.

    foundry = ModelFoundry(hive)
    foundry.forge("classifier", "mlp", data=my_data, device="auto")   # trains + registers a bee
    hive.invoke("classifier", new_data)                                # an LM drives the NN

Placement of the LM bees themselves (an iGPU queen + CPU-core workers) is `place_llm()`, a thin wrap
over hive.spawn's n_gpu_layers.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

# archetypes whose conv path is CPU-only here; everything else can ride the GPU/iGPU
_CPU_ONLY = {"cnn"}
_GPU_BACKENDS = {"cuda", "rocm", "hip", "gpu"}

# which archetype a task/data implies, by the words that name each one's domain (structural word
# presence, not a scored match). The order matters: the first archetype whose words appear wins.
_ARCH_WORDS = [
    ("cnn", ["image", "vision", "pixel", "photo", "convolut"]),
    ("hgnn", ["graph", "hypergraph", "node", "edge", "relation", "network", "mesh", "topolog"]),
    ("lm", ["text", "language", "token", "sequence", "sentence", "nlp", "next-token", "chat"]),
    ("mlp", ["tabular", "vector", "feature", "column", "regress", "classif", "predict"]),
]
_KIND_ARCH = {"image": "cnn", "graph": "hgnn", "rex": "hgnn", "hypergraph": "hgnn",
              "text": "lm", "sequence": "lm", "tabular": "mlp", "vector": "mlp"}


def choose_archetype(task: str = "", data=None) -> str:
    """Pick an archetype from the data's kind or the task's words: structural (exact kind lookup,
    then word presence), defaulting to mlp. This is the deterministic backbone; an LM can override."""
    from agent import models
    kind = str(getattr(data, "kind", "") or "").lower()
    for k, arch in _KIND_ARCH.items():
        if k in kind and arch in models.ARCHETYPES:
            return arch
    t = (task or "").lower()
    for arch, words in _ARCH_WORDS:
        if arch in models.ARCHETYPES and any(w in t for w in words):
            return arch
    return "mlp"


def resolve_device(archetype: str, requested: str = "auto") -> str:
    """Pick a device for an archetype. An explicit request wins. 'auto' sends conv archetypes to the
    CPU and everything else to the GPU/iGPU when rexgraph.compute reports a usable one (live probe),
    else the CPU."""
    if requested and requested != "auto":
        return requested
    if archetype in _CPU_ONLY:
        return "cpu"
    try:
        from rexgraph import compute
        return "cuda" if (compute.recommended_backend() in _GPU_BACKENDS) else "cpu"
    except Exception:
        return "cpu"


def place_llm(hive, name: str, model_path: str, *, on: str = "igpu", role: str = "worker",
              ctx_size: int = 4096, wait: float = 90.0, **kw):
    """Bring up a managed llama.cpp bee placed on the iGPU ('igpu'/'gpu' -> all layers offloaded) or
    across the CPU cores ('cpu' -> 0 gpu layers). The iGPU-queen + CPU-worker pattern: spawn the
    queen on the iGPU and the optimized workers on the cores, all sharing unified memory."""
    ngl = 99 if on in ("igpu", "gpu") else 0
    return hive.spawn(name, model_path, role=role, ctx_size=ctx_size,
                      n_gpu_layers=ngl, wait=wait, **kw)


def bundle_from_rows(rows: List[dict], *, target: str, features: Optional[List[str]] = None):
    """Turn database/query rows into a vector DataBundle for training: numeric feature columns become
    X (non-numeric features are label-encoded), the target column becomes integer classes y. This is
    what lets a forged NN learn on the ACTUAL data instead of a synthetic set."""
    import numpy as np
    from agent.models import data as _data
    if not rows:
        raise ValueError("no rows to build a training bundle from")
    feats = features or [c for c in rows[0].keys() if c != target]

    def encode(values):
        try:
            return np.array([float(v) for v in values], dtype="float32")
        except (TypeError, ValueError):
            keys = {v: i for i, v in enumerate(sorted(set(map(str, values))))}
            return np.array([keys[str(v)] for v in values], dtype="float32")

    X = (np.stack([encode([r.get(f) for r in rows]) for f in feats], axis=1)
         if feats else np.zeros((len(rows), 1), dtype="float32"))
    raw_y = [r.get(target) for r in rows]
    classes = {v: i for i, v in enumerate(sorted(set(map(str, raw_y))))}   # label-encode -> 0..k-1
    y = np.array([classes[str(v)] for v in raw_y], dtype="int64")
    return _data._vector_bundle(X, y)


class ModelFoundry:
    """Forge trained neural networks into worker bees the LMs control."""

    def __init__(self, hive=None, *, store_dir: Optional[str] = None):
        if hive is None:
            from . import hive as hivemod
            hive = hivemod.get_hive()
        self.hive = hive
        self.store_dir = store_dir or tempfile.mkdtemp(prefix="foundry-")
        self.forged: List[Dict[str, Any]] = []          # the NN sub-hive this foundry built

    def forge(self, name: str, archetype: str, *, data=None, params=None, steps: int = 100,
              device: str = "auto", capability: str = "predict", specialties=None,
              optimizer: str = "auto", seed: int = 0) -> Dict[str, Any]:
        """Train an NN on `data` (a bundle / a data source / None for the archetype's synthetic set)
        and register it as a worker bee. Returns the model card. Falls back to CPU once if a chosen
        GPU turns out unusable."""
        from agent import models
        dev = resolve_device(archetype, device)
        path = os.path.join(self.store_dir, f"{name}.pt")
        try:
            res = models.run(archetype, params=params, data=data, steps=steps, device=dev,
                             optimizer=optimizer, seed=seed, save_to=path)
        except Exception:
            if dev == "cpu":
                raise
            dev = "cpu"                                  # visible-but-unusable GPU -> degrade once
            res = models.run(archetype, params=params, data=data, steps=steps, device=dev,
                             optimizer=optimizer, seed=seed, save_to=path)
        wtype = f"model:{archetype}"
        self.hive.add_model(name, res["saved"], capability=capability, device=dev,
                            specialties=specialties or [archetype, "predict", "model"],
                            worker_type=wtype)
        from . import activity as _act
        _act.record("model:" + name, "forge",
                    detail={"archetype": archetype, "device": dev, "metric": res.get("metric")})
        card = {"name": name, "archetype": archetype, "device": dev,
                "metric": res.get("metric"), "saved": res["saved"], "worker_type": wtype,
                "capability": capability}
        self.forged.append(card)
        return card

    def _ask_coder_for_spec(self, coder, task: str, data) -> Optional[dict]:
        """Let an LM choose the archetype + params. `coder` is a callable prompt->reply (e.g.
        lambda p: hive.ask('architect', p)). Returns the parsed spec, or None on any failure."""
        from agent import models
        names = ", ".join(sorted(models.ARCHETYPES.keys()))
        kind = getattr(data, "kind", "unknown")
        prompt = ("Choose a neural network archetype for this task and reply with ONLY JSON "
                  '{"archetype": <name>, "params": {}}.\n'
                  f"Archetypes: {names}\nTask: {task}\nData kind: {kind}")
        try:
            reply = coder(prompt)
            m = re.search(r"\{.*\}", reply or "", re.S)
            spec = json.loads(m.group(0)) if m else None
            return spec if isinstance(spec, dict) else None
        except Exception:
            return None

    def forge_from_task(self, name: str, task: str, *, data=None, coder=None, params=None,
                        device: str = "auto", steps: int = 100, capability: str = "predict",
                        seed: int = 0) -> Dict[str, Any]:
        """Forge an NN whose ARCHETYPE is chosen from the task. If a `coder` (an LM) is given it
        picks (and may set params); otherwise, or if its pick is invalid, the structural heuristic
        (`choose_archetype`) does. The coder decides WHAT to build; the foundry builds and places it.
        """
        from agent import models
        spec = self._ask_coder_for_spec(coder, task, data) if coder is not None else None
        archetype = (spec or {}).get("archetype")
        chosen_by = "coder"
        if archetype not in models.ARCHETYPES:
            archetype = choose_archetype(task, data)
            chosen_by = "heuristic"
        p = {**((spec or {}).get("params") or {}), **(params or {})}
        card = self.forge(name, archetype, data=data, params=p or None, device=device,
                          steps=steps, capability=capability, seed=seed)
        card.update({"task": task, "chosen_by": chosen_by})
        return card

    def forge_on_rows(self, name: str, rows: List[dict], *, target: str, features=None,
                      archetype: str = "auto", task: str = "", coder=None, device: str = "auto",
                      steps: int = 100, seed: int = 0) -> Dict[str, Any]:
        """Forge an NN that trains on ACTUAL rows (from a query / a DB table). Builds a bundle from
        the data, then forges. Archetype defaults to what the tabular data implies (mlp) unless a
        coder/explicit choice overrides."""
        bundle = bundle_from_rows(rows, target=target, features=features)
        if archetype == "auto":
            spec = self._ask_coder_for_spec(coder, task or f"predict {target}", bundle) if coder else None
            from agent import models
            archetype = (spec or {}).get("archetype")
            if archetype not in models.ARCHETYPES:
                archetype = choose_archetype(task, bundle)
        card = self.forge(name, archetype, data=bundle, device=device, steps=steps, seed=seed)
        card.update({"trained_on": "rows", "n_rows": len(rows), "target": target,
                     "features": features or [c for c in rows[0].keys() if c != target]})
        return card

    def forge_many(self, specs: List[dict]) -> List[Dict[str, Any]]:
        """Forge several NNs (each spec: {name, archetype, ...}) into the sub-hive."""
        return [self.forge(s.pop("name"), s.pop("archetype"), **s) for s in (dict(x) for x in specs)]

    def invoke(self, name: str, data=None, **kw):
        """Drive one forged NN (what an LM does when it uses a network it built)."""
        return self.hive.invoke(name, data, **kw)

    def roster(self) -> List[Dict[str, Any]]:
        """The NN sub-hive: each network, its archetype, and its device placement."""
        return [dict(c) for c in self.forged]


def hierarchy(hive) -> Dict[str, Any]:
    """The hive as a control hierarchy: the language models (chat/generate bees) that CONTROL the
    neural networks (model:* worker bees) beneath them, plus the device each network runs on."""
    controllers, networks = [], []
    for b in hive.bees():
        if b.capability == "generate":
            controllers.append({"name": b.name, "role": b.role})
        elif (b.worker_type or "").startswith("model:"):
            networks.append({"name": b.name, "type": b.worker_type})
    return {"controllers": controllers, "networks": networks}
