"""
models - model-builder framework over the rexgraph.nn substrate.

Select an archetype, override its parameters, point it at data, and train it in one of three
modes: single run, multistep (staged), or multi-model fusion. The archetypes (LM / CNN / MLP /
HGNN) are example models built from rexgraph.nn; they are not part of the library.

    from models import list_archetypes, run
    list_archetypes()                                  # names, use-cases, params
    run("cnn", params={"norm": False}, steps=300)       # synthetic data, optimizer="auto"
    run("mlp", data="mydata.csv")                       # file data
    run("lm",  params={"attention": "standard"})
    run("mlp", data="mydata.csv", optimizer="adamw")    # a named optimizer when you want one
    # multistep (staged): warm up, then refine at a lower lr
    run("mlp", mode="multistep", stages=[{"steps":100}, {"steps":200,"lr":5e-4}])
    # multi-model fusion (ensemble / split / stack)
    run("mlp", mode="fusion", specs=[("mlp",{}), ("mlp",{"d_hid":64})], fusion="ensemble")
"""
from . import (  # noqa: F401
    archetypes,
    data,
    store,  # noqa: F401
    train,
    trustgraph,  # noqa: F401
)
from .archetypes import ARCHETYPES, get, merged_cfg, register_archetype  # noqa: F401
from .store import (  # noqa: F401  rexgraph IO bridge
    load_bundle,
    load_checkpoint,
    save_checkpoint,
    save_complex_rex,
    to_rcdb,
)
from .trustgraph import (  # noqa: F401  TrustGraph ingestion (DB -> knowledge core -> complex)
    bundle_from_core,
    core_to_rcdb,
    core_to_rex,
    core_to_rex_file,
)


def list_archetypes() -> list:
    """Return every archetype with its use-case, the data kind it consumes, and its
    customizable parameters (with defaults)."""
    return [{"name": a["name"], "use_case": a["use_case"], "data_kind": a["data_kind"],
             "params": a["defaults"]} for a in sorted(ARCHETYPES.values(), key=lambda x: x["name"])]


def _load(archetype, source, params, seed):
    """Return a DataBundle from a rexgraph.io source (parquet / vectors / .rex / sql / csv/jsonl/txt),
    an already-built DataBundle, or the archetype's synthetic generator (source=None)."""
    if source is None:
        return get(archetype)["synth"](merged_cfg(archetype, params), seed)
    if hasattr(source, "kind"):
        return source
    return store.load_bundle(source)


def build(archetype, *, params=None, data=None, seed=0):
    """Build (don't train) a model for an archetype on given/synthetic data. Returns
    (model, cfg, bundle)."""
    bundle = data if hasattr(data, "kind") else _load(archetype, data, params, seed)
    cfg = merged_cfg(archetype, params)
    model = get(archetype)["build"](cfg, bundle)
    return model, cfg, bundle


def run(archetype, *, params=None, data=None, mode="single", optimizer="auto", steps=200,
        lr=None, seed=0, stages=None, specs=None, fusion="ensemble", device="cpu",
        save_to=None, on_step=None, amp=False, schedule=None, warmup=0, grad_accum=1,
        resume=None) -> dict:
    """Build and train an archetype. `mode` is one of {single, multistep, fusion}. Returns the run
    result (metric trajectory / stages / fused metric). `data` may be a path or a DataBundle.
    `device` defaults to 'cpu'; use 'cuda' for the non-conv archetypes (mlp/lm/hgnn). Conv fails on
    this box's ROCm build, so cnn stays on cpu."""
    if mode == "fusion":
        bundle = data if hasattr(data, "kind") else _load(archetype, data, params, seed)
        return train.train_fusion(specs or [(archetype, {}), (archetype, {})], bundle,
                                  mode=fusion, steps=steps, optimizer=optimizer, device=device, seed=seed)
    model, cfg, bundle = build(archetype, params=params, data=data, seed=seed)
    if mode == "multistep":
        res = train.train_multistep(model, bundle, list(stages or []), device=device, seed=seed)
    else:
        res = train.train_one(model, bundle, optimizer=optimizer, steps=steps, lr=lr,
                              device=device, seed=seed, on_step=on_step, amp=amp,
                              schedule=schedule, warmup=warmup, grad_accum=grad_accum, resume=resume)
    if save_to:                                   # checkpoint through the rexgraph IO layer (store.py)
        store.save_checkpoint(save_to, model, archetype, cfg, bundle=bundle, result=res)
    return {"archetype": archetype, "cfg": cfg, "saved": save_to, **res}


def predict(checkpoint, data=None, *, split=None, device="cpu", save_to=None) -> dict:
    """Run a trained model on new data. `checkpoint` is a saved-checkpoint path (or a
    (model, config) pair from load_checkpoint). `data` is a DataBundle, any rexgraph.io
    source (parquet / .rex / sql / csv / jsonl / safetensors), or None for the archetype's
    synthetic data. Returns {archetype, n, predictions, metric, split}. When `save_to` is a
    .safetensors path, the predictions are written through rexgraph.io.save_vectors."""
    import rexgraph.nn as R
    dev = R.pick_device(device)                   # 'auto' rides the compute stack; 'cpu' forces CPU
    model, conf = (checkpoint if isinstance(checkpoint, tuple)
                   else store.load_checkpoint(checkpoint, device=dev))   # map_location -> picked device
    arch = conf["archetype"]
    kind = get(arch)["data_kind"]
    bundle = data if hasattr(data, "kind") else _load(arch, data, None, 0)
    bundle.to(dev)
    model = model.to(dev)
    preds, metric = train.predict_on(model, bundle, kind, split=split)
    out = {"archetype": arch, "n": int(len(preds)), "predictions": preds,
           "metric": metric, "split": split or "all"}
    if save_to:                                   # predictions back through the rexgraph IO layer
        import rexgraph.io as rio
        vecs = preds.reshape(len(preds), -1).astype("float32")
        rio.save_vectors(vecs, [str(i) for i in range(len(preds))], str(save_to))
        out["saved_to"] = str(save_to)
    return out
