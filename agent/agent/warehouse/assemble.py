"""The self-assembling forge-and-persist loop (Slice 1): ingest any weighted edge list as an
edge-primal relational complex, dispatch a per-tier HGNN sweep through the Hive Coordinator (CPU
proc + iGPU threads), keep the per-tier best, deploy it as a bee, infer per edge, and persist a
model-agnostic record to the RCDB. Fully programmatic and idempotent."""
from __future__ import annotations

import logging
import os
import tempfile

import numpy as np

from . import source as S
from .foundry_tasks import train_one

logger = logging.getLogger(__name__)

_DEFAULT_SWEEP = [
    {"archetype": "hgnn", "params": {"d_hid": 16, "n_layers": 2}, "seed": 0},
    {"archetype": "hgnn", "params": {"d_hid": 32, "n_layers": 2}, "seed": 1},
    {"archetype": "hgnn", "params": {"d_hid": 32, "n_layers": 3}, "seed": 2},
]


def assemble(path, *, store, hive=None, source=None, target=None, weight=None, usecols=None,
             n_tiers=3, sweep=None, steps=80, save_dir=None) -> dict:
    from rexgraph.coordinator import Coordinator, LanePools

    from ..coordinator_adapter import work_units
    from ..foundry import _CPU_ONLY
    if hive is None:
        from .. import hive as hivemod
        hive = hivemod.get_hive()
    sweep = sweep or _DEFAULT_SWEEP
    save_dir = save_dir or tempfile.mkdtemp(prefix="warehouse-")

    ed = S.load_edges(path, source=source, target=target, weight=weight, usecols=usecols)
    rex = S.edge_complex(ed)
    tiers = S.tier_split(ed, n_tiers)

    pools = LanePools("warehouse")
    coord = Coordinator(pools=pools)
    tasks, tier_ctx = [], {}
    results = {}
    try:
        # build the whole T x S wave of picklable training specs
        for ti, mask in enumerate(tiers):
            if mask.shape[0] < 4:                 # too few edges to train/split
                continue
            X, names = S.edge_features(rex, ed, mask)
            y = S.labels(ed, mask)
            b = S.hypergraph_bundle(ed, mask, X, y)
            tier_ctx[ti] = {"mask": mask, "X": X, "y": y, "names": names, "bundle": b}
            for si, cfg in enumerate(sweep):
                cid = f"t{ti}-{cfg['archetype']}#{si}"
                arche = cfg["archetype"]
                device = "cpu" if arche in _CPU_ONLY else "cuda"
                spec = {"archetype": arche, "params": cfg.get("params"), "device": device,
                        "save_path": os.path.join(save_dir, cid + ".pt"), "steps": steps,
                        "he_ptr": b.extra["he_ptr"], "he_idx": b.extra["he_idx"],
                        "X": X, "y": y, "feat_dim": int(X.shape[1]), "n_classes": 2,
                        "tier": ti, "config_id": cid, "seed": int(cfg.get("seed", si))}
                tasks.append({"id": cid, "kind": f"train:{arche}",
                              "fn": _thunk(train_one, spec)})

        units = work_units(tasks)
        placement = coord.plan(units)
        results = coord.pools.run(units, placement, cost=coord.cost)   # per-task isolation drops failures
    finally:
        pools.shutdown()

    # select best per tier, deploy, infer, persist
    report = {"tiers": [], "store_ids": []}
    for ti, ctx in tier_ctx.items():
        cands = [r for cid, r in results.items()
                 if isinstance(r, dict) and r.get("tier") == ti and r.get("saved")
                 and np.isfinite(r.get("metric", float("nan")))]
        if not cands:
            logger.warning("tier %d: no surviving model", ti)
            report["tiers"].append({"tier": ti, "best": None,
                                    "n_edges": int(ctx["mask"].shape[0])})
            continue
        best = max(cands, key=lambda r: r["metric"])
        bee_name = f"tier-{ti}-best"
        try:
            hive.add_model(bee_name, best["saved"], capability="predict", device=best.get("device"),
                           specialties=[best["archetype"], "edge", "predict"],
                           worker_type=f"model:{best['archetype']}")
            best["bee"] = bee_name
        except Exception as ex:
            logger.warning("tier %d deploy failed: %s", ti, ex)
            best["bee"] = None

        # persist a model-agnostic RCDB record: the tier complex + typed/tensor context + model card
        tier_rex = _subcomplex(ed, ctx["mask"])
        rid = f"tier-{ti}"
        meta = {"tier": ti, "n_edges": int(ctx["mask"].shape[0]),
                "feature_channels": ctx["names"], "col_types": ed.col_types,
                "winner": {k: best.get(k) for k in ("config_id", "archetype", "metric", "device")},
                "sweep": [{"config_id": r.get("config_id"), "metric": r.get("metric")}
                          for r in results.values() if isinstance(r, dict) and r.get("tier") == ti]}
        store.put(rid, tier_rex, meta=meta, tags=[f"tier-{ti}", best["archetype"]])
        report["store_ids"].append(rid)
        report["tiers"].append({"tier": ti, "best": best, "rcdb_id": rid,
                                "n_edges": int(ctx["mask"].shape[0])})

    # prune losing checkpoints; keep only what each tier's winner references
    winners = {t["best"]["saved"] for t in report["tiers"]
               if t.get("best") and t["best"].get("saved")}
    try:
        for r in results.values():
            if isinstance(r, dict):
                sp = r.get("saved")
                if sp and sp not in winners and os.path.exists(sp):
                    os.remove(sp)
    except OSError as ex:
        logger.warning("checkpoint cleanup skipped: %s", ex)
    return report


def _subcomplex(ed, mask):
    """The tier's own edge complex (edges in `mask`), reindexed, as the RCDB blob."""
    from rexgraph.graph import RexGraph
    ss = ed.src_idx[mask]; ds = ed.dst_idx[mask]
    verts = np.unique(np.concatenate([ss, ds]))
    remap = {int(v): i for i, v in enumerate(verts)}
    src = np.array([remap[int(x)] for x in ss], np.int32)
    tgt = np.array([remap[int(x)] for x in ds], np.int32)
    return RexGraph(sources=src, targets=tgt)


def _thunk(fn, spec):
    import functools
    return functools.partial(fn, spec)
