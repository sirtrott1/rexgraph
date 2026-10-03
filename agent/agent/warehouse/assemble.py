"""The self assembling forge and persist loop: ingest any weighted edge list as an
edge primal relational complex, dispatch a per tier HGNN sweep through the Hive Coordinator (CPU
proc + iGPU threads), keep the per tier best, deploy it as a bee, infer per edge, and persist a
model agnostic record to the RCDB. Fully programmatic and idempotent."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import stat
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

    ed = S.load_edges(path, source=source, target=target, weight=weight, usecols=usecols)
    rex = S.edge_complex(ed)
    tiers = S.tier_split(ed, n_tiers)

    pools = LanePools("warehouse")
    coord = Coordinator(pools=pools)
    tasks, tier_ctx = [], {}
    results = {}
    run_dir, run_identity = None, None
    checkpoint_paths = []
    feature_table, feature_names = None, None
    try:
        # build the whole T x S wave of picklable training specs
        for ti, mask in enumerate(tiers):
            if mask.shape[0] < 4:                 # too few edges to train/split
                continue
            if feature_table is None:
                feature_table, feature_names = S.edge_features(rex, ed, np.arange(rex.nE))
            X, names = feature_table[mask], list(feature_names)
            y = S.labels(ed, mask)
            b = S.hypergraph_bundle(ed, mask, X, y)
            tier_ctx[ti] = {"mask": mask, "X": X, "y": y, "names": names, "bundle": b}
            for si, cfg in enumerate(sweep):
                if run_dir is None:
                    parent = Path(save_dir).expanduser().resolve() if save_dir else None
                    if parent is not None:
                        parent.mkdir(parents=True, exist_ok=True)
                    run_dir = Path(tempfile.mkdtemp(prefix="warehouse-", dir=parent)).resolve()
                    created = run_dir.stat()
                    run_identity = (created.st_dev, created.st_ino)
                cid = f"t{ti}-{cfg['archetype']}#{si}"
                arche = cfg["archetype"]
                device = "cpu" if arche in _CPU_ONLY else "cuda"
                checkpoint = run_dir / f"t{ti}-candidate-{si}.pt"
                checkpoint_paths.append(checkpoint)
                spec = {"archetype": arche, "params": cfg.get("params"), "device": device,
                        "save_path": str(checkpoint), "steps": steps,
                        "he_ptr": b.extra["he_ptr"], "he_idx": b.extra["he_idx"],
                        "X": X, "y": y, "feat_dim": int(X.shape[1]), "n_classes": 2,
                        "tier": ti, "config_id": cid, "seed": int(cfg.get("seed", si))}
                tasks.append({"id": cid, "kind": f"train:{arche}",
                              "fn": _thunk(train_one, spec)})

        units = work_units(tasks)
        placement = coord.plan(units)
        results = coord.pools.run(units, placement, cost=coord.cost)   # per task isolation drops failures
    finally:
        pools.shutdown()

    # select best per tier, deploy, infer, persist
    report = {"tiers": [], "store_ids": []}
    for ti, ctx in tier_ctx.items():
        cands = [r for cid, r in results.items()
                 if isinstance(r, dict) and r.get("tier") == ti and r.get("saved")
                 and r.get("metric") is not None and np.isfinite(r["metric"])]
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

        # persist a model agnostic RCDB record: the tier complex + typed/tensor context + model card
        tier_rex = _subcomplex(ed, ctx["mask"], rex=rex)
        rid = f"tier-{ti}"
        meta = {"tier": ti, "n_edges": int(ctx["mask"].shape[0]),
                "feature_channels": ctx["names"], "col_types": ed.col_types,
                "feature_contract": {"version": 2, "numeric": "float32",
                    "hodge": "absolute-amplitude", "heat": "down-sector",
                    "dirac_heat": "in-grade"},
                "winner": {k: best.get(k) for k in ("config_id", "archetype", "metric", "device")},
                "sweep": [{"config_id": r.get("config_id"), "metric": r.get("metric"),
                           "error": r.get("error")}
                          for r in results.values() if isinstance(r, dict) and r.get("tier") == ti]}
        store.put(rid, tier_rex, meta=meta, tags=[f"tier-{ti}", best["archetype"]])
        report["store_ids"].append(rid)
        report["tiers"].append({"tier": ti, "best": best, "rcdb_id": rid,
                                "n_edges": int(ctx["mask"].shape[0])})

    # Retire only planned outputs in this run, including partial failed writes.
    # Publication failures above preserve the entire run for recovery.
    winners = {t["best"]["saved"] for t in report["tiers"]
               if t.get("best") and t["best"].get("saved")}
    _cleanup_checkpoints(run_dir, run_identity, checkpoint_paths, winners)
    return report


def _cleanup_checkpoints(run_dir, identity, paths, winners):
    """Retire losing planned outputs; worker reported paths confer no ownership."""
    if run_dir is None:
        return
    protected = {os.path.abspath(os.fspath(path)) for path in winners}
    for checkpoint in paths:
        # Refuse a replaced run root rather than traversing someone else's tree.
        try:
            current = run_dir.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("checkpoint cleanup skipped for %s: %s", run_dir, exc)
            return
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            logger.warning("checkpoint cleanup skipped: run directory changed: %s", run_dir)
            return
        if checkpoint.parent != run_dir or str(checkpoint) in protected:
            continue
        try:
            if checkpoint.is_symlink():
                checkpoint.unlink()  # remove only the link, never its target
            elif checkpoint.is_dir():
                shutil.rmtree(checkpoint)
            else:
                checkpoint.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("checkpoint cleanup skipped for %s: %s", checkpoint, exc)
    try:
        current = run_dir.lstat()
        if stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == identity:
            run_dir.rmdir()  # only an empty run; retained checkpoints keep it alive
    except OSError:
        pass


def _subcomplex(ed, mask, *, rex=None):
    """The tier's own edge complex (edges in `mask`), reindexed, as the RCDB blob."""
    source = S.edge_complex(ed) if rex is None else rex
    indices = S._tier_indices(mask, source.nE)
    if source.nE != len(ed.src_idx):
        raise ValueError("warehouse tier source does not match its edge data")
    keep = np.zeros(source.nE, dtype=bool)
    keep[indices] = True
    return source.subgraph(keep)[0]


def _thunk(fn, spec):
    import functools
    return functools.partial(fn, spec)
