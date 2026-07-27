"""Module-level, picklable NN-training task for the coordinator proc/igpu lanes. Rebuilds the
hypergraph DataBundle from picklable arrays inside the worker and calls models.run. No hive, no
closures, so it is forkserver-safe (proc lane) and thread-safe (igpu lane)."""
from __future__ import annotations

import numpy as np


def train_one(spec: dict) -> dict:
    from agent.agent.models import run as models_run
    from agent.agent.models.data import DataBundle, make_splits
    import torch
    b = DataBundle("hypergraph",
                   torch.as_tensor(np.asarray(spec["X"], np.float32)),
                   torch.as_tensor(np.asarray(spec["y"], np.int64)),
                   meta={"feat_dim": int(spec["feat_dim"]), "n_classes": int(spec["n_classes"]),
                         "n_nodes": int(np.asarray(spec["y"]).shape[0])})
    b.extra = {"he_ptr": np.asarray(spec["he_ptr"], np.int32),
               "he_idx": np.asarray(spec["he_idx"], np.int32)}
    b.splits = make_splits(int(np.asarray(spec["y"]).shape[0]), seed=int(spec.get("seed", 0)))
    try:
        res = models_run(spec["archetype"], params=spec.get("params"), data=b,
                         optimizer="hodge", steps=int(spec.get("steps", 80)),
                         device=spec.get("device", "cpu"), save_to=spec.get("save_path"),
                         seed=int(spec.get("seed", 0)))
        metric = _final_metric(res)
    except Exception as ex:
        return {"tier": spec["tier"], "config_id": spec["config_id"],
                "archetype": spec["archetype"], "device": spec.get("device"),
                "metric": float("nan"), "saved": None, "error": repr(ex)}
    return {"tier": spec["tier"], "config_id": spec["config_id"], "archetype": spec["archetype"],
            "device": spec.get("device"), "metric": metric, "saved": res.get("saved")}


def _final_metric(res: dict) -> float:
    """Extract a scalar accuracy/score from models.run's result (metric trajectory or scalar)."""
    m = res.get("metric")
    if isinstance(m, (list, tuple)) and m:
        last = m[-1]
        return float(last[-1] if isinstance(last, (list, tuple)) else last)
    if isinstance(m, dict):
        return float(m.get("acc", m.get("val_acc", m.get("score", 0.0))))
    try:
        return float(m)
    except Exception:
        return 0.0
