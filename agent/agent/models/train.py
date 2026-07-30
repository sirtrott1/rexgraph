"""
train: training loops for a built model, single run, multistep (staged) training, and multi-model
fusion (ensemble / data-split / stacking). The optimizer is any rexgraph.nn optimizer (HodgeAdam by
default). The loop dispatches on the DataBundle's `kind`, so one interface trains every archetype.
"""
from __future__ import annotations

from typing import List

import torch as _t
import torch.nn.functional as _F

import rexgraph.nn as R


def _device(d):
    return R.pick_device(d)


def _forward_loss(model, bundle, idx, kind):
    """Return (loss, logits_or_none) for a batch of indices on the given data kind."""
    X, y = bundle.X, bundle.y
    if kind in ("vector", "image"):
        out = model(X[idx])
        if bundle.meta.get("task") == "regression":
            return _F.mse_loss(out.squeeze(-1), y[idx].float()), out
        return _F.cross_entropy(out, y[idx]), out
    if kind == "sequence":
        out = model(X[idx])                                  # (b,T,V)
        V = out.shape[-1]
        return _F.cross_entropy(out.reshape(-1, V), y[idx].reshape(-1)), out
    if kind == "hypergraph":
        out = model(X)                                       # full-graph
        return _F.cross_entropy(out[idx], y[idx]), out
    raise ValueError(f"unknown data kind {kind!r}")


@_t.no_grad()
def _evaluate(model, bundle, kind):
    model.eval()
    te = bundle.splits["test"]
    if kind == "hypergraph":
        pred = model(bundle.X)[te].argmax(1); acc = float((pred == bundle.y[te]).float().mean())
    elif kind == "sequence":
        out = model(bundle.X[te]); acc = float((out.argmax(-1) == bundle.y[te]).float().mean())
    elif bundle.meta.get("task") == "regression":
        acc = -float(_F.mse_loss(model(bundle.X[te]).squeeze(-1), bundle.y[te].float()))
    else:
        acc = float((model(bundle.X[te]).argmax(1) == bundle.y[te]).float().mean())
    model.train()
    return round(acc, 4)


@_t.no_grad()
def predict_on(model, bundle, kind, split=None):
    """Run a trained model on a bundle (a named split, or all rows when split is None).
    Returns (predictions ndarray, metric-or-None). Metric is accuracy, or -MSE for
    regression, and is None when the bundle carries no labels (pure inference)."""
    model.eval()
    sel = bundle.splits[split] if split else None
    take = (lambda z: z) if sel is None else (lambda z: z[sel])
    if kind == "hypergraph":
        out = model(bundle.X)
        preds = (out if sel is None else out[sel]).argmax(1)
    elif kind == "sequence":
        preds = model(take(bundle.X)).argmax(-1)
    elif bundle.meta.get("task") == "regression":
        preds = model(take(bundle.X)).squeeze(-1)
    else:
        preds = model(take(bundle.X)).argmax(1)
    metric = None
    if bundle.y is not None:
        y = take(bundle.y)
        if bundle.meta.get("task") == "regression":
            metric = round(-float(_F.mse_loss(preds.float(), y.float())), 4)
        else:
            metric = round(float((preds == y).float().mean()), 4)
    model.train()
    return preds.detach().cpu().numpy(), metric


def _lr_at(step, total, base, schedule, warmup):
    """Per-step learning rate: linear warmup then a cosine or linear decay, or flat (schedule=None)."""
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    if schedule == "cosine":
        import math
        return base * 0.5 * (1.0 + math.cos(math.pi * p))
    if schedule == "linear":
        return base * (1.0 - p)
    return base


def _set_lr(opt, lr):
    for grp in opt.param_groups:
        grp["lr"] = lr


def train_one(model, bundle, *, optimizer="auto", steps=200, lr=None, batch=64,
              n_heads=1, device=None, seed=0, on_step=None,
              amp=False, schedule=None, warmup=0, grad_accum=1, resume=None) -> dict:
    """Train `model` on `bundle` with a rexgraph.nn optimizer. Returns the eval-metric trajectory
    and which optimizer ran (metric = test accuracy, or -MSE for regression).

    `steps` counts optimizer updates. `amp=True` runs bf16 autocast on CUDA (ignored on CPU).
    `schedule` in {None, 'cosine', 'linear'} with `warmup` steps sets the lr per step. `grad_accum`
    averages that many micro-batches per update. `resume` (a checkpoint path) loads weights first."""
    _t.manual_seed(seed)
    dev = _device(device)
    if resume:                                                # continue from a saved checkpoint
        from . import store
        model.load_state_dict(store.load_checkpoint(resume, device=dev)[0].state_dict())
    model = model.to(dev); bundle.to(dev)
    kind = bundle.kind
    tr = bundle.splits["train"]
    opt, opt_class = R.make_optimizer(optimizer, model,
                                      [p for p in model.parameters() if p.requires_grad],
                                      n_heads=n_heads, lr=lr)
    base_lr = opt.param_groups[0]["lr"]
    use_amp = bool(amp) and str(dev).startswith("cuda")
    accum = max(1, int(grad_accum))
    g = _t.Generator().manual_seed(seed)
    traj = []
    full = (kind == "hypergraph")
    for i in range(steps):
        model.train()
        if schedule or warmup:
            _set_lr(opt, _lr_at(i, steps, base_lr, schedule, warmup))
        opt.zero_grad()
        step_loss = 0.0
        for _a in range(accum):
            idx = tr if full else tr[_t.randint(0, len(tr), (min(batch, len(tr)),), generator=g)]
            if use_amp:
                with _t.autocast(device_type="cuda", dtype=_t.bfloat16):
                    loss, _ = _forward_loss(model, bundle, idx, kind)
            else:
                loss, _ = _forward_loss(model, bundle, idx, kind)
            (loss / accum).backward()
            step_loss += float(loss.item()) / accum
        opt.step()
        if i % max(1, steps // 20) == 0 or i == steps - 1:
            traj.append(_evaluate(model, bundle, kind))
        if on_step:
            on_step(i, step_loss, steps)
    return {"optimizer": optimizer, "optimizer_class": opt_class, "steps": steps,
            "metric": (max if not bundle.meta.get("task") == "regression" else max)(traj) if traj else None,
            "metric_name": ("test accuracy" if not bundle.meta.get("task") == "regression"
                            else "-test MSE"),
            "final": traj[-1] if traj else None, "trajectory": traj}


def train_multistep(model, bundle, stages: List[dict], *, device=None, seed=0) -> dict:
    """Train one model through a sequence of stages on the same (or per-stage) data. Each stage is
    a dict of train_one overrides: a curriculum, an optimizer schedule (hodge to adam), or a
    warmup-to-refine lr schedule. Returns per-stage results."""
    results = []
    for s, stage in enumerate(stages):
        b = stage.pop("bundle", bundle) if isinstance(stage, dict) else bundle
        r = train_one(model, b, device=device, seed=seed, **{k: v for k, v in stage.items()})
        r["stage"] = s
        results.append(r)
    return {"mode": "multistep", "n_stages": len(stages), "stages": results,
            "final": results[-1]["final"] if results else None}


def train_fusion(specs, bundle, *, mode="ensemble", steps=200, optimizer="auto",
                 device=None, seed=0) -> dict:
    """Train multiple models and fuse them. `specs` is a list of (archetype_name, cfg_overrides).
      - mode='ensemble'  : each model trains on the full data; predictions are averaged.
      - mode='split'     : the training set is partitioned across models, then ensembled.
      - mode='stack'     : base models train, their logits are concatenated, and a linear meta-head
                           is trained on top.
    Returns the fused test metric and each base model's metric."""
    from . import archetypes as A
    dev = _device(device); bundle.to(dev)
    kind = bundle.kind
    models, per = [], []
    tr = bundle.splits["train"]
    parts = _split_indices(tr, len(specs), seed) if mode == "split" else [tr] * len(specs)
    for k, (name, over) in enumerate(specs):
        cfg = A.merged_cfg(name, over)
        m = A.get(name)["build"](cfg, bundle).to(dev)
        sub = _sub_bundle(bundle, parts[k])
        r = train_one(m, sub, optimizer=optimizer, steps=steps, device=device, seed=seed + k)
        models.append(m); per.append({"archetype": name, **{k2: r[k2] for k2 in ("final", "optimizer_class")}})

    @_t.no_grad()
    def _probs(m):
        m.eval()
        te = bundle.splits["test"]
        out = m(bundle.X)[te] if kind == "hypergraph" else m(bundle.X[te])
        return _F.softmax(out, dim=-1)

    te = bundle.splits["test"]
    if mode == "stack":
        # meta-head on concatenated base logits (fit on train, eval on test)
        fused_acc = _stack(models, bundle, kind, dev, seed)
    else:
        avg = sum(_probs(m) for m in models) / len(models)
        fused_acc = round(float((avg.argmax(-1) == bundle.y[te]).float().mean()), 4)
    return {"mode": mode, "n_models": len(specs), "fused_metric": fused_acc,
            "base_models": per, "metric_name": "test accuracy"}


def _split_indices(idx, k, seed):
    perm = idx[_t.randperm(len(idx), generator=_t.Generator().manual_seed(seed))]
    return [perm[i::k] for i in range(k)]


def _sub_bundle(bundle, train_idx):
    from .data import DataBundle
    b = DataBundle(bundle.kind, bundle.X, bundle.y, dict(bundle.meta),
                   {"train": train_idx, "val": bundle.splits["val"], "test": bundle.splits["test"]},
                   dict(bundle.extra))
    return b


def _stack(models, bundle, kind, dev, seed):
    import torch.nn as nn
    te, tr = bundle.splits["test"], bundle.splits["train"]

    @_t.no_grad()
    def feats(idx):
        xs = []
        for m in models:
            m.eval()
            o = m(bundle.X)[idx] if kind == "hypergraph" else m(bundle.X[idx])
            xs.append(_F.softmax(o, -1))
        return _t.cat(xs, -1)

    Ftr, Fte = feats(tr), feats(te)
    meta = nn.Linear(Ftr.shape[1], int(bundle.y.max()) + 1).to(dev)
    opt = _t.optim.Adam(meta.parameters(), lr=1e-2)
    for _ in range(200):
        loss = _F.cross_entropy(meta(Ftr), bundle.y[tr]); opt.zero_grad(); loss.backward(); opt.step()
    return round(float((meta(Fte).argmax(1) == bundle.y[te]).float().mean()), 4)
