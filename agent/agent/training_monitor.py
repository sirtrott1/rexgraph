"""agent.training_monitor: watch model training live and diagnose/fix what's wrong, structurally.

Training is a signal on a complex: the per-step loss is a 1-D trajectory whose SHAPE says what is
happening. A healthy run descends (the loss "drains"); a broken one shows a structural signature:
it never moves (no learning signal), it climbs or goes non-finite (diverging), or the validation
turns up while training falls (overfitting). `diagnose()` reads those signatures with exact/relative
signals (finiteness, sign of the trend, a numerical-zero flatness test), never a tuned cutoff, and
names the likely CAUSE and a FIX. `train_watched()` runs a training with the live loss hook, applies
the fix, and retries: the reactive layer aimed at the training loop instead of the swarm.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# machine-precision "is this change essentially zero vs the loss scale": a numerical zero, not a
# policy threshold.
_ZERO = 1e-9


def diagnose(losses, val=None) -> dict[str, Any]:
    """Structural diagnosis of a loss trajectory. Returns {status, issue, cause, fix}."""
    xs = np.asarray([float(l) for l in losses if l is not None], dtype=float)
    if xs.size < 3:
        return {"status": "unknown", "issue": None, "cause": None, "fix": None}

    if not np.all(np.isfinite(xs)):
        return {"status": "diverging", "issue": "loss became NaN or infinite",
                "cause": "exploding gradients - learning rate too high, or unscaled / mis-encoded inputs",
                "fix": "lower_lr"}

    scale = max(abs(float(xs[0])), _ZERO)
    # trend vs wandering: efficiency = net change / total movement, in [-1, 1]. -1 is a clean descent,
    # +1 a clean ascent, ~0 is movement that cancels out (no learning signal). The split is a
    # majority (more than half the movement is net trend): a natural boundary, not a tuned value.
    tv = float(np.sum(np.abs(np.diff(xs))))
    net = float(xs[-1] - xs[0])
    if tv <= _ZERO * scale:                              # never moved at all
        return {"status": "not_learning", "issue": "loss is flat",
                "cause": "no learning signal - likely the wrong archetype, a learning rate near "
                         "zero, or mis-encoded inputs", "fix": "raise_lr_then_switch_archetype"}
    frac = net / tv
    if frac >= 0.5:                                      # net ascent dominates the movement
        return {"status": "diverging", "issue": "loss is climbing",
                "cause": "the optimizer is diverging - learning rate too high, or a bad target setup",
                "fix": "lower_lr"}
    if frac > -0.5:                                      # movement cancels out - no net progress
        return {"status": "not_learning", "issue": "loss wanders without net progress",
                "cause": "no consistent learning signal - likely the wrong archetype for this data, "
                         "a learning rate near zero, or mis-encoded inputs",
                "fix": "raise_lr_then_switch_archetype"}

    # validation turning up while training falls -> overfitting
    if val is not None:
        vs = np.asarray([float(v) for v in val if v is not None], dtype=float)
        if vs.size >= 3 and np.all(np.isfinite(vs)):
            vfloor = float(vs[:max(2, vs.size // 5)].min())
            if vs[-1] > vfloor + _ZERO * max(abs(vfloor), _ZERO):
                return {"status": "overfitting", "issue": "training improves but validation worsens",
                        "cause": "the model is memorizing - too much capacity or too little data",
                        "fix": "reduce_capacity"}

    # decreased overall; is it still moving, or has it settled?
    k = max(2, xs.size // 5)
    recent_drop = float(xs[-k] - xs[-1])
    if abs(recent_drop) <= _ZERO * scale:
        return {"status": "converged", "issue": None,
                "cause": "loss decreased then flattened (settled)", "fix": None}
    return {"status": "healthy", "issue": None, "cause": None, "fix": None}


_FIXABLE = {"diverging", "not_learning", "overfitting"}


class TrainingMonitor:
    """Runs a training with the live loss hook, diagnoses it, and (optionally) applies the fix and
    retries. Registers the final model as a bee. Governed: `autofix` is off by default; it proposes
    a diagnosis and stops; turn it on for a bounded self-healing loop."""

    def __init__(self, hive=None):
        if hive is None:
            from . import hive as hivemod
            hive = hivemod.get_hive()
        self.hive = hive

    def _apply_fix(self, fix, *, archetype, params, lr, steps, tried):
        """Map a fix name to the next attempt's settings. Structural choices, not tuned values."""
        from agent import models
        base = lr if lr else 1e-3
        if fix == "lower_lr":
            return archetype, params, base / 10.0, steps
        if fix == "raise_lr_then_switch_archetype":
            if lr is None or lr < 1e-2:                  # first try a usable learning rate
                return archetype, params, 1e-3, steps
            others = [a for a in models.ARCHETYPES if a != archetype and a not in tried]
            return (others[0] if others else archetype), params, 1e-3, steps
        if fix == "reduce_capacity":
            return archetype, params, lr, max(2, steps // 2)
        return archetype, params, lr, steps

    def train_watched(self, name, archetype, *, data=None, params=None, steps: int = 100,
                      device: str = "auto", optimizer: str = "auto", lr=None, seed: int = 0,
                      autofix: bool = False, attempts: int = 3, register: bool = True) -> dict[str, Any]:
        import os
        import tempfile

        from agent import models
        from agent.foundry import resolve_device
        from agent.models import store as _store
        from agent.models import train as _train

        arch, p = archetype, dict(params or {})
        tried, log = set(), []
        best = None
        for attempt in range(1, attempts + 1):
            tried.add(arch)
            dev = resolve_device(arch, device)
            losses: list[float] = []
            try:
                model, cfg, bundle = models.build(arch, params=p, data=data, seed=seed)
                res = _train.train_one(model, bundle, optimizer=optimizer, steps=steps, lr=lr,
                                       device=dev, seed=seed,
                                       on_step=lambda i, loss, total: losses.append(loss))
                # the eval trajectory is a METRIC (higher is better); negate it into a loss-proxy so
                # "validation worsening" == the proxy rising, matching diagnose's convention.
                traj = res.get("trajectory") or []
                val_proxy = [-float(m) for m in traj] if traj else None
                diag = diagnose(losses, val=val_proxy)
                metric = res.get("metric")
            except Exception as e:
                diag = {"status": "error", "issue": str(e),
                        "cause": "training raised - likely a data-format / shape mismatch",
                        "fix": "raise_lr_then_switch_archetype"}
                model, cfg, bundle, metric = None, None, None, None
            log.append({"attempt": attempt, "archetype": arch, "lr": lr, "steps": steps,
                        "device": dev, "metric": metric, "diagnosis": diag})
            if model is not None and diag["status"] in ("healthy", "converged", "unknown"):
                best = (model, cfg, arch, bundle, metric); break
            if model is not None:
                best = (model, cfg, arch, bundle, metric)     # keep the last usable model
            if not autofix or attempt == attempts or not diag.get("fix"):
                break
            arch, p, lr, steps = self._apply_fix(diag["fix"], archetype=arch, params=p, lr=lr,
                                                 steps=steps, tried=tried)

        card = {"name": name, "attempts": log, "final": log[-1] if log else None}
        if best and register:
            model, cfg, arch, bundle, metric = best
            path = os.path.join(tempfile.mkdtemp(prefix="trainmon-"), f"{name}.pt")
            _store.save_checkpoint(path, model, arch, cfg, bundle=bundle)
            self.hive.add_model(name, path, capability="predict", device=log[-1]["device"],
                                specialties=[arch, "predict", "model"], worker_type=f"model:{arch}")
            card.update({"archetype": arch, "metric": metric, "saved": path, "registered": True})
        return card
