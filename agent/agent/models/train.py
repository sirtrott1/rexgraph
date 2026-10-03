"""
train: training loops for a built model, single run, multistep (staged) training, and multi model
fusion (ensemble / data split / stacking). The optimizer is built by `R.make_optimizer(optimizer,
...)` with `optimizer="auto"` by default, which routes per model: GreensCochain for a model whose
parameters are a cochain on a complex (it exposes `greens_groups()`), plain Adam otherwise. Every
archetype here is feature-space, so in practice that is Adam; any other optimizer is opt-in by name.
The loop dispatches on the DataBundle's `kind`, so one interface trains every archetype.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch as _t
import torch.nn.functional as _F

import rexgraph.nn as R


def _fetch(bundle, indices):
    if hasattr(bundle, "fetch"):
        return bundle.fetch(indices)
    return bundle.X[indices], None if bundle.y is None else bundle.y[indices]


def _batches(bundle, indices):
    size = getattr(bundle, "eval_batch_size", max(1, len(indices)))
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def _sample(indices, count, generator):
    positions = _t.randint(0, len(indices), (min(count, len(indices)),), generator=generator)
    if isinstance(indices, range):
        return indices.start + positions * indices.step
    return indices[positions]


def _metric_batch(predictions, targets, regression):
    if regression:
        return -float(((predictions.double() - targets.double()) ** 2).sum()), targets.numel()
    return int((predictions == targets).sum()), targets.numel()


def _device(d):
    return R.pick_device(d)


def _task(model, bundle):
    return getattr(model, "task", bundle.meta.get("task", "classification"))


@contextmanager
def _evaluating(model):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def _forward_loss(model, bundle, idx, kind):
    """Return (loss, logits_or_none) for a batch of indices on the given data kind."""
    X, y = bundle.X, bundle.y
    if kind in ("vector", "image"):
        xb, yb = _fetch(bundle, idx)
        out = model(xb)
        if _task(model, bundle) == "regression":
            return _F.mse_loss(out.squeeze(-1), yb.float()), out
        return _F.cross_entropy(out, yb), out
    if kind == "sequence":
        out = model(X[idx])                                  # (b,T,V)
        V = out.shape[-1]
        return _F.cross_entropy(out.reshape(-1, V), y[idx].reshape(-1)), out
    if kind == "hypergraph":
        out = model(X)                                       # full graph
        return _F.cross_entropy(out[idx], y[idx]), out
    raise ValueError(f"unknown data kind {kind!r}")


@_t.no_grad()
def _evaluate(model, bundle, kind):
    with _evaluating(model):
        te = bundle.splits["test"]
        if hasattr(bundle, "eval_batch_size"):
            total, count = 0.0, 0
            regression = _task(model, bundle) == "regression"
            for idx in _batches(bundle, te):
                xb, yb = _fetch(bundle, idx)
                out = model(xb)
                pred = out.squeeze(-1) if regression else out.argmax(1)
                score, n = _metric_batch(pred, yb, regression)
                total += score; count += n
            if not count:
                raise ValueError("evaluation requires a nonempty test split")
            return round(total / count, 4)
        if kind == "hypergraph":
            pred = model(bundle.X)[te].argmax(1); acc = float((pred == bundle.y[te]).float().mean())
        elif kind == "sequence":
            out = model(bundle.X[te]); acc = float((out.argmax(-1) == bundle.y[te]).float().mean())
        elif _task(model, bundle) == "regression":
            acc = -float(_F.mse_loss(model(bundle.X[te]).squeeze(-1), bundle.y[te].float()))
        else:
            acc = float((model(bundle.X[te]).argmax(1) == bundle.y[te]).float().mean())
        return round(acc, 4)


@_t.no_grad()
def predict_on(model, bundle, kind, split=None):
    """Run a trained model on a bundle (a named split, or all rows when split is None).
    Returns (predictions ndarray, metric or None). Metric is accuracy, or -MSE for
    regression, and is None when the bundle carries no labels (pure inference)."""
    with _evaluating(model):
        sel = bundle.splits[split] if split else None
        if hasattr(bundle, "eval_batch_size"):
            import numpy as np
            indices = range(len(bundle.X)) if sel is None else sel
            regression = _task(model, bundle) == "regression"
            predictions, total, count = [], 0.0, 0
            for idx in _batches(bundle, indices):
                xb, yb = _fetch(bundle, idx)
                out = model(xb)
                pred = out.squeeze(-1) if regression else out.argmax(1)
                predictions.append(pred.cpu().numpy())
                if yb is not None:
                    score, n = _metric_batch(pred, yb, regression)
                    total += score; count += n
            # Returning all predictions necessarily costs O(number of rows) host
            # memory; feature batches and GPU memory remain bounded.
            result = np.concatenate(predictions) if predictions else np.empty(0, dtype=float if regression else int)
            return result, round(total / count, 4) if count else None
        take = (lambda z: z) if sel is None else (lambda z: z[sel])
        if kind == "hypergraph":
            out = model(bundle.X)
            preds = (out if sel is None else out[sel]).argmax(1)
        elif kind == "sequence":
            preds = model(take(bundle.X)).argmax(-1)
        elif _task(model, bundle) == "regression":
            preds = model(take(bundle.X)).squeeze(-1)
        else:
            preds = model(take(bundle.X)).argmax(1)
        metric = None
        if bundle.y is not None:
            y = take(bundle.y)
            if _task(model, bundle) == "regression":
                metric = round(-float(_F.mse_loss(preds.float(), y.float())), 4)
            else:
                metric = round(float((preds == y).float().mean()), 4)
        return preds.detach().cpu().numpy(), metric


def _lr_at(step, total, base, schedule, warmup):
    """Per step learning rate: linear warmup then a cosine or linear decay, or flat (schedule=None)."""
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
    """Train `model` on `bundle` with a rexgraph.nn optimizer. Returns the eval metric trajectory
    and which optimizer ran (metric = test accuracy, or -MSE for regression).

    `steps` counts optimizer updates. `amp=True` runs bf16 autocast on CUDA (ignored on CPU).
    `schedule` in {None, 'cosine', 'linear'} with `warmup` steps sets the lr per step. `grad_accum`
    averages that many micro batches per update. `resume` (a checkpoint path) loads weights first."""
    _t.manual_seed(seed)
    dev = _device(device)
    if resume:                                                # continue from a saved checkpoint
        from . import store
        model.load_state_dict(store.load_checkpoint(resume, device=dev)[0].state_dict())
    model = model.to(dev); bundle.to(dev)
    kind = bundle.kind
    if bundle.y is None:
        raise ValueError("training requires targets; this bundle is for inference")
    tr = bundle.splits["train"]
    if not len(tr) or (kind != "hypergraph" and batch < 1):
        raise ValueError("training requires a nonempty split and a positive batch size")
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
            idx = tr if full else _sample(tr, batch, g)
            if use_amp:
                with _t.autocast(device_type="cuda", dtype=_t.bfloat16):
                    loss, _ = _forward_loss(model, bundle, idx, kind)
            else:
                loss, _ = _forward_loss(model, bundle, idx, kind)
            if not _t.isfinite(loss):
                raise ArithmeticError("nonfinite training objective")
            (loss / accum).backward()
            step_loss += float(loss.item()) / accum
        if any(p.grad is not None and not _t.isfinite(p.grad).all() for p in model.parameters()):
            raise ArithmeticError("nonfinite training gradient")
        opt.step()
        if i % max(1, steps // 20) == 0 or i == steps - 1:
            traj.append(_evaluate(model, bundle, kind))
        if on_step:
            on_step(i, step_loss, steps)
    return {"optimizer": optimizer, "optimizer_class": opt_class, "steps": steps,
            "metric": max(traj) if traj else None,
            "metric_name": ("test accuracy" if _task(model, bundle) != "regression"
                            else "-test MSE"),
            "final": traj[-1] if traj else None, "trajectory": traj}


def train_multistep(model, bundle, stages: list[dict], *, device=None, seed=0) -> dict:
    """Train one model through a sequence of stages on the same (or per stage) data. Each stage is
    a dict of train_one overrides: a curriculum, an optimizer schedule, or a warmup to refine lr
    schedule. A stage that names no optimizer gets the "auto" route. Returns per stage results."""
    results = []
    for s, stage in enumerate(stages):
        options = dict(stage)
        b = options.pop("bundle", bundle)
        r = train_one(model, b, device=device, seed=seed, **options)
        r["stage"] = s
        results.append(r)
    return {"mode": "multistep", "n_stages": len(stages), "stages": results,
            "final": results[-1]["final"] if results else None}


def train_fusion(specs, bundle, *, mode="ensemble", steps=200, optimizer="auto",
                 device=None, seed=0) -> dict:
    """Train multiple models and fuse them. `specs` is a list of (archetype_name, cfg_overrides).
      - mode='ensemble'  : each model trains on the full data; predictions are averaged.
      - mode='split'     : the training set is partitioned across models, then ensembled.
      - mode='stack'     : base models train, their logits are concatenated, and a linear meta head
                           is trained on top.
    Returns the fused test metric and each base model's metric."""
    from . import archetypes as A
    if mode not in ("ensemble", "split", "stack") or not specs:
        raise ValueError("fusion needs models and an ensemble, split or stack mode")
    dev = _device(device); bundle.to(dev)
    kind = bundle.kind
    models, per = [], []
    regression = None
    tr = bundle.splits["train"]
    parts = _split_indices(tr, len(specs), seed) if mode == "split" else [tr] * len(specs)
    for k, (name, over) in enumerate(specs):
        cfg = A.merged_cfg(name, over)
        m = A.get(name)["build"](cfg, bundle).to(dev)
        current_task = _task(m, bundle) == "regression"
        if regression is not None and regression != current_task:
            raise ValueError("fusion models must use the same prediction task")
        regression = current_task
        sub = _sub_bundle(bundle, parts[k])
        r = train_one(m, sub, optimizer=optimizer, steps=steps, device=device, seed=seed + k)
        models.append(m); per.append({"archetype": name, **{k2: r[k2] for k2 in ("final", "optimizer_class")}})

    @_t.no_grad()
    def _probs(m):
        with _evaluating(m):
            te = bundle.splits["test"]
            out = m(bundle.X)[te] if kind == "hypergraph" else m(bundle.X[te])
            return out if regression else _F.softmax(out, dim=-1)

    te = bundle.splits["test"]
    if mode == "stack":
        # meta head on concatenated base logits (fit on train, eval on test)
        fused_acc = _stack(models, bundle, kind, dev, seed)
    elif hasattr(bundle, "eval_batch_size"):
        total, count = 0.0, 0
        for idx in _batches(bundle, te):
            xb, yb = _fetch(bundle, idx)
            with _t.no_grad():
                outputs = []
                for m in models:
                    with _evaluating(m):
                        out = m(xb)
                        outputs.append(out if regression else _F.softmax(out, -1))
                avg = sum(outputs) / len(outputs)
                pred = avg.squeeze(-1) if regression else avg.argmax(-1)
                score, n = _metric_batch(pred, yb, regression)
                total += score; count += n
        if not count:
            raise ValueError("fusion requires a nonempty test split")
        fused_acc = round(total / count, 4)
    else:
        avg = sum(_probs(m) for m in models) / len(models)
        fused_acc = (round(-float(_F.mse_loss(avg.squeeze(-1), bundle.y[te].float())), 4) if regression
                     else round(float((avg.argmax(-1) == bundle.y[te]).float().mean()), 4))
    return {"mode": mode, "n_models": len(specs), "fused_metric": fused_acc,
            "base_models": per, "metric_name": "-test MSE" if regression else "test accuracy"}


def _split_indices(idx, k, seed):
    if isinstance(idx, range):
        # Keep a compact partition for mapped data, rather than allocating a
        # dataset sized random permutation. Every training row belongs to one part.
        return [idx[i::k] for i in range(k)]
    perm = idx[_t.randperm(len(idx), generator=_t.Generator().manual_seed(seed))]
    return [perm[i::k] for i in range(k)]


def _sub_bundle(bundle, train_idx):
    from copy import copy
    b = copy(bundle)
    b.meta, b.extra = dict(bundle.meta), dict(bundle.extra)
    b.splits = {"train": train_idx, "val": bundle.splits["val"], "test": bundle.splits["test"]}
    return b


def _stack(models, bundle, kind, dev, seed):
    import torch.nn as nn
    te, tr = bundle.splits["test"], bundle.splits["train"]
    regression = _task(models[0], bundle) == "regression"

    if hasattr(bundle, "eval_batch_size"):
        return _stack_mapped(models, bundle, dev, regression)

    @_t.no_grad()
    def feats(idx):
        xs = []
        for m in models:
            with _evaluating(m):
                o = m(bundle.X)[idx] if kind == "hypergraph" else m(bundle.X[idx])
                xs.append(o if regression else _F.softmax(o, -1))
        return _t.cat(xs, -1)

    Ftr, Fte = feats(tr), feats(te)
    n_out = 1 if regression else int(bundle.y.max()) + 1
    meta = nn.Linear(Ftr.shape[-1], n_out).to(dev)
    opt = _t.optim.Adam(meta.parameters(), lr=1e-2)
    for _ in range(200):
        output = meta(Ftr)
        loss = (_F.mse_loss(output.squeeze(-1), bundle.y[tr].float()) if regression
                else _F.cross_entropy(output.reshape(-1, n_out), bundle.y[tr].reshape(-1)))
        opt.zero_grad(); loss.backward(); opt.step()
    with _t.no_grad():
        output = meta(Fte)
        return (round(-float(_F.mse_loss(output.squeeze(-1), bundle.y[te].float())), 4) if regression
                else round(float((output.argmax(-1) == bundle.y[te]).float().mean()), 4))


def _stack_mapped(models, bundle, dev, regression):
    """Full batch meta head updates without dataset sized feature caches.

    Base models stay frozen. Recomputing their outputs each epoch trades compute
    for bounded memory while retaining the existing full batch objective.
    """
    import torch.nn as nn
    tr, te = bundle.splits["train"], bundle.splits["test"]
    n_out = 1 if regression else bundle.meta["n_classes"]
    meta = nn.Linear(len(models) * n_out, n_out).to(dev)
    opt = _t.optim.Adam(meta.parameters(), lr=1e-2)
    @_t.no_grad()
    def features(xb):
        outputs = []
        for m in models:
            with _evaluating(m):
                out = m(xb)
                outputs.append(out if regression else _F.softmax(out, -1))
        return _t.cat(outputs, -1)
    for _ in range(200):
        opt.zero_grad()
        for idx in _batches(bundle, tr):
            xb, yb = _fetch(bundle, idx)
            output = meta(features(xb))
            loss = (_F.mse_loss(output.squeeze(-1), yb.float()) if regression
                    else _F.cross_entropy(output, yb))
            if not _t.isfinite(loss):
                raise ArithmeticError("nonfinite stacking objective")
            (loss * (len(idx) / len(tr))).backward()
        if any(p.grad is not None and not _t.isfinite(p.grad).all() for p in meta.parameters()):
            raise ArithmeticError("nonfinite stacking gradient")
        opt.step()
    total, count = 0.0, 0
    with _t.no_grad():
        for idx in _batches(bundle, te):
            xb, yb = _fetch(bundle, idx)
            out = meta(features(xb))
            pred = out.squeeze(-1) if regression else out.argmax(-1)
            score, n = _metric_batch(pred, yb, regression)
            total += score; count += n
    if not count:
        raise ValueError("stacking requires a nonempty test split")
    return round(total / count, 4)
