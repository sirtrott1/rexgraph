"""
benchmarks: standard ML benchmarks for comparing optimizers on real, recognized tasks.

The point: settle optimizer claims (hodge / hodge-arch vs Adam/AdamW/SGD) where the community can
compare, not on toy data. Each benchmark builds a real model + data + eval metric and runs with ANY
registered optimizer (`nn.make_optimizer`), streaming loss, so `benchmark_ab` gives a fair,
held-out comparison and a metric-aware verdict. Every model here is feature-space, so the routing
default resolves to plain Adam; the hodge arms are named by the caller because they are what is
under test.

Two kinds:
  - `ill-cond`: a CONTROLLED ill-conditioned matrix-regression (tunable condition number κ). No
    download, runs anywhere, and it is exactly the regime where per-Hodge-component preconditioning
    is supposed to help: the diagnostic that actually tests the claim.
  - `mnist` / `fashion-mnist` / `cifar10`: the recognized image-classification benchmarks (loaded
    via the `datasets` extra), an MLP trained to test accuracy.

Everything degrades cleanly without torch/datasets.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger("rexgraph.benchmarks")

try:
    import torch as _t
    import torch.nn as _nn
    import torch.nn.functional as _F
    _HAS_TORCH = True
except Exception:                                    # pragma: no cover
    _HAS_TORCH = False


_BENCH: dict[str, dict] = {}


def register_benchmark(name: str, fn: Callable, *, description: str = "",
                       higher_better: bool = False, needs=("torch",)):
    _BENCH[name] = {"fn": fn, "description": description,
                    "higher_better": higher_better, "needs": tuple(needs)}


def benchmarks() -> list:
    def _ok(needs):
        for m in needs:
            try:
                __import__(m)
            except Exception:
                return False
        return True
    return [{"name": n, "description": b["description"], "metric_higher_better": b["higher_better"],
             "available": _ok(b["needs"])} for n, b in sorted(_BENCH.items())]


def run_benchmark(name: str, *, optimizer: str = "auto", steps: int = 200,
                  lr: float | None = None, device: str | None = None, seed: int = 0,
                  on_step: Callable = None, label: str | None = None, **kw) -> dict:
    """Run one benchmark with one optimizer. `optimizer` defaults to "auto" (the router), so an
    unnamed run measures what a model of this shape actually trains with; name an optimizer to test
    a specific one. Returns train + held-out-eval trajectories and the task metric. Graceful:
    missing torch/datasets → a clear skip, not a crash."""
    if name not in _BENCH:
        return {"skipped": f"unknown benchmark {name!r} (have: {', '.join(sorted(_BENCH))})"}
    b = _BENCH[name]
    for m in b["needs"]:
        try:
            __import__(m)
        except Exception:
            return {"skipped": f"benchmark {name!r} needs '{m}' (pip install -e '.[finetune]')",
                    "optimizer": optimizer}
    return b["fn"](optimizer=optimizer, steps=steps, lr=lr, device=device, seed=seed,
                   on_step=on_step, label=label or optimizer, **kw)


# default lr grid for the fair (tuned) comparison. Each optimizer keeps its best lr
_LR_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def benchmark_ab(name: str, *, optimizers=("hodge", "adam"), steps: int = 200,
                 lrs=_LR_GRID, on_step: Callable = None, **kw) -> dict:
    """A/B the same benchmark under each optimizer (same data/seed). FAIR BY DEFAULT: each
    optimizer is tuned over an lr grid and keeps its best. A fixed-lr comparison only measures
    lr-sensitivity, not optimizer quality. Verdict on the held-out metric; a sub-1% gap = tie.
    Pass `lrs=None` (or a single-element list) to skip the sweep."""
    if name not in _BENCH:
        return {"skipped": f"unknown benchmark {name!r}", "ab": []}
    higher = _BENCH[name]["higher_better"]
    grid = list(lrs) if lrs else [None]

    def _better(a, b):
        return (a > b) if higher else (a < b)

    runs = []
    for opt in optimizers:
        best = None
        for lr in grid:
            r = run_benchmark(name, optimizer=opt, steps=steps,
                              lr=lr, on_step=(on_step if len(grid) == 1 else None),
                              label=opt, **kw)
            if "skipped" in r:
                return r
            if r.get("metric") is not None and (best is None or _better(r["metric"], best["metric"])):
                best = r; best["lr_used"] = lr
        runs.append(best if best is not None else r)
    scores = {r["optimizer"]: r.get("metric") for r in runs if r.get("metric") is not None}
    best, margin, verdict = None, None, "inconclusive"
    if len(scores) == 2:
        best = (max if higher else min)(scores, key=scores.get)
        vals = sorted(scores.values())
        margin = round(vals[1] - vals[0], 5)
        rel = abs(margin) / (max(abs(v) for v in vals) or 1)
        mname = runs[0].get("metric_name", "metric")
        verdict = (f"{best} won on {mname} (gap {margin})" if rel > 0.01
                   else f"tie: {mname} gap {margin} is within noise")
    return {"benchmark": name, "ab": runs, "metrics": scores, "best": best, "margin": margin,
            "lrs_used": {r["optimizer"]: r.get("lr_used") for r in runs},
            "tuned": bool(lrs and len(grid) > 1),
            "metric_name": runs[0].get("metric_name") if runs else None,
            "higher_better": higher, "verdict": verdict}


#### the controlled ill-conditioned benchmark (the claim test, runs anywhere)
def _ill_conditioned(*, optimizer, steps, lr, device, seed, on_step, label,
                     d_in: int = 64, d_out: int = 16, n: int = 512, kappa: float = 1000.0,
                     noise: float = 0.1, **kw):
    """Multi-output linear regression Y = X·W (+ noise) with X's spectrum set so cond(XᵀX) = κ. The
    weight W is a real matrix, so the hodge arm has something to decompose; the Hessian condition
    number is κ, the regime a per-component preconditioner is claimed to help. `noise` gives a
    non-trivial optimum (not exactly 0), so the metric measures real generalization. Metric =
    held-out MSE (lower better)."""
    import rexgraph.nn as nn
    from rexgraph.nn import optim
    _t.manual_seed(seed)
    dev = optim.pick_device(device)
    # X = Q·diag(s), s geometric in [1, sqrt(kappa)] ⇒ cond(XᵀX) = kappa (exactly)
    q, _ = _t.linalg.qr(_t.randn(n, d_in))
    s = _t.logspace(0, 0.5 * _t.log10(_t.tensor(kappa)).item(), d_in)
    X = (q * s).to(dev)
    Wt = _t.randn(d_in, d_out, device=dev)
    Y = X @ Wt + noise * _t.randn(n, d_out, device=dev)
    qe, _ = _t.linalg.qr(_t.randn(n, d_in))
    Xe = (qe * s).to(dev); Ye = Xe @ Wt + noise * _t.randn(n, d_out, device=dev)

    model = _nn.Linear(d_in, d_out, bias=False).to(dev)
    opt, opt_class = nn.make_optimizer(optimizer, model,
                                       [p for p in model.parameters() if p.requires_grad],
                                       n_heads=1, lr=lr or 1e-2)

    def _eval():
        with _t.no_grad():
            return round(float(_F.mse_loss(model(Xe), Ye).item()), 6)

    traj, evals = [], []
    for i in range(steps):
        loss = _F.mse_loss(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
        traj.append(round(float(loss.item()), 6))
        if i % 5 == 0 or i == steps - 1:
            evals.append(_eval())
        if on_step:
            on_step(label, i, float(loss.item()), steps)
    return {"optimizer": optimizer, "optimizer_class": opt_class, "device": dev,
            "benchmark": "ill-cond", "kappa": kappa, "steps": len(traj),
            "loss_start": traj[0] if traj else None, "loss_final": traj[-1] if traj else None,
            "eval_start": evals[0] if evals else None, "eval_final": evals[-1] if evals else None,
            "eval_trajectory": evals, "trajectory": traj,
            "metric": evals[-1] if evals else None, "metric_name": "eval MSE",
            "improved": bool(evals and evals[-1] < evals[0])}


#### recognized image-classification benchmarks (need the `datasets` extra)
def _image_clf(dataset_name, image_key, *, optimizer, steps, lr, device, seed, on_step, label,
               hidden: int = 256, batch: int = 128, train_n: int = 4000, eval_n: int = 1000,
               conv: bool = False, **kw):
    """Standard image dataset → test accuracy, with an MLP or (conv=True) a small CNN whose 4-tensor
    kernels exercise the general k-tensor Hodge split. Loads via HF `datasets`."""
    import numpy as np
    from datasets import load_dataset

    import rexgraph.nn as nn
    from rexgraph.nn import optim
    _t.manual_seed(seed)
    dev = optim.pick_device(device)
    keep_spatial = bool(conv)

    def _xy(split, limit):
        ds = load_dataset(dataset_name, split=f"{split}[:{limit}]")
        imgs = np.stack([np.asarray(im, dtype="float32") / 255.0 for im in ds[image_key]])
        x = _t.tensor(imgs)
        if keep_spatial:
            if x.dim() == 3:                          # grayscale H,W → 1,H,W
                x = x.unsqueeze(1)
            elif x.dim() == 4:                        # H,W,C → C,H,W
                x = x.permute(0, 3, 1, 2)
        else:
            x = x.reshape(len(imgs), -1)
        return x.to(dev), _t.tensor(ds["label"]).to(dev)

    xtr, ytr = _xy("train", train_n)
    xte, yte = _xy("test", eval_n)
    n_cls = int(max(ytr.max().item(), yte.max().item())) + 1
    if keep_spatial:
        ch = xtr.shape[1]
        depth = int(kw.get("depth", 2))              # number of conv blocks
        norm = str(kw.get("norm", True)).lower() not in ("false", "0", "no")
        layers = []; c = ch; w = 32
        for _ in range(max(1, depth)):
            blk = [_nn.Conv2d(c, w, 3, padding=1)]
            if norm:
                blk.append(_nn.BatchNorm2d(w))       # normalization fixes conditioning (confound)
            blk += [_nn.ReLU(), _nn.MaxPool2d(2)]
            layers += blk
            c = w; w = min(w * 2, 256)
        layers += [_nn.AdaptiveAvgPool2d(2), _nn.Flatten(), _nn.Linear(c * 4, n_cls)]
        model = _nn.Sequential(*layers).to(dev)
    else:
        d_in = xtr.shape[1]
        model = _nn.Sequential(_nn.Linear(d_in, hidden), _nn.ReLU(), _nn.Linear(hidden, n_cls)).to(dev)
    opt, opt_class = nn.make_optimizer(optimizer, model,
                                       [p for p in model.parameters() if p.requires_grad],
                                       n_heads=1, lr=lr or 1e-3)

    def _acc():
        with _t.no_grad():
            return round(float((model(xte).argmax(1) == yte).float().mean().item()), 4)

    g = _t.Generator().manual_seed(seed)
    traj, evals = [], []
    for i in range(steps):
        idx = _t.randint(0, len(xtr), (batch,), generator=g)
        loss = _F.cross_entropy(model(xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        traj.append(round(float(loss.item()), 4))
        if i % 10 == 0 or i == steps - 1:
            evals.append(_acc())
        if on_step:
            on_step(label, i, float(loss.item()), steps)
    return {"optimizer": optimizer, "optimizer_class": opt_class, "device": dev,
            "benchmark": dataset_name, "steps": len(traj),
            "loss_start": traj[0] if traj else None, "loss_final": traj[-1] if traj else None,
            "eval_start": evals[0] if evals else None, "eval_final": evals[-1] if evals else None,
            "eval_trajectory": evals, "trajectory": traj,
            "metric": evals[-1] if evals else None, "metric_name": "test accuracy",
            "improved": bool(evals and evals[-1] > evals[0])}


def _matrix_completion(*, optimizer, steps, lr, device, seed, on_step, label,
                       m: int = 40, n: int = 40, rank: int = 3, obs: float = 0.4, **kw):
    """Low-rank matrix completion: recover M = A·Bᵀ (rank r) from a random `obs` fraction of its
    entries by fitting U·Vᵀ. NON-CONVEX, and U/V are real matrices, so the hodge arm decomposes them.
    Metric = MSE on the HELD-OUT (unobserved) entries: generalization on a non-convex problem."""
    import rexgraph.nn as nn
    from rexgraph.nn import optim
    _t.manual_seed(seed)
    dev = optim.pick_device(device)
    A = _t.randn(m, rank, device=dev); Bm = _t.randn(n, rank, device=dev)
    M = A @ Bm.T
    mask = (_t.rand(m, n, device=dev) < obs).float()
    U = _nn.Parameter(0.1 * _t.randn(m, rank, device=dev))
    V = _nn.Parameter(0.1 * _t.randn(n, rank, device=dev))
    opt, opt_class = nn.make_optimizer(optimizer, _nn.ParameterList([U, V]), [U, V],
                                       n_heads=1, lr=lr or 1e-2)

    def _held_out():
        with _t.no_grad():
            pred = U @ V.T
            inv = 1.0 - mask
            return round(float(((pred - M) ** 2 * inv).sum() / inv.sum().clamp_min(1)), 6)

    traj, evals = [], []
    for i in range(steps):
        pred = U @ V.T
        loss = ((pred - M) ** 2 * mask).sum() / mask.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step()
        traj.append(round(float(loss.item()), 6))
        if i % 5 == 0 or i == steps - 1:
            evals.append(_held_out())
        if on_step:
            on_step(label, i, float(loss.item()), steps)
    return {"optimizer": optimizer, "optimizer_class": opt_class, "device": dev,
            "benchmark": "matrix-completion", "steps": len(traj),
            "loss_start": traj[0] if traj else None, "loss_final": traj[-1] if traj else None,
            "eval_start": evals[0] if evals else None, "eval_final": evals[-1] if evals else None,
            "eval_trajectory": evals, "trajectory": traj,
            "metric": evals[-1] if evals else None, "metric_name": "held-out MSE",
            "improved": bool(evals and evals[-1] < evals[0])}


def _bilinear_game(*, optimizer, steps, lr, device, seed, on_step, label,
                   d: int = 16, **kw):
    """Bilinear min–max game: minₓ max_Y ⟨X, C·Y⟩ with equilibrium at X=Y=0. Its gradient field is
    purely ROTATIONAL: simultaneous descent/ascent orbits and naive methods don't converge. This
    is the regime where damping the rotational component (the hodge arm's `gamma_curl`) is claimed
    to matter. Metric = distance to equilibrium ‖X‖²+‖Y‖² at the end (lower = converged)."""
    import rexgraph.nn as nn
    from rexgraph.nn import optim
    _t.manual_seed(seed)
    dev = optim.pick_device(device)
    C = _t.randn(d, d, device=dev)
    X = _nn.Parameter(_t.randn(d, d, device=dev))
    Y = _nn.Parameter(_t.randn(d, d, device=dev))
    opt, opt_class = nn.make_optimizer(optimizer, _nn.ParameterList([X, Y]), [X, Y],
                                       n_heads=1, lr=lr or 1e-2)

    def _dist():
        with _t.no_grad():
            return round(float((X ** 2).sum() + (Y ** 2).sum()), 6)

    traj, evals = [], []
    for i in range(steps):
        f = (X * (C @ Y)).sum()                 # ⟨X, C·Y⟩
        opt.zero_grad(); f.backward()
        if Y.grad is not None:
            Y.grad.neg_()                       # ascend on Y (min–max)
        opt.step()
        dist = _dist()
        traj.append(dist)
        if i % 5 == 0 or i == steps - 1:
            evals.append(dist)
        if on_step:
            on_step(label, i, dist, steps)
    return {"optimizer": optimizer, "optimizer_class": opt_class, "device": dev,
            "benchmark": "bilinear-game", "steps": len(traj),
            "loss_start": traj[0] if traj else None, "loss_final": traj[-1] if traj else None,
            "eval_start": evals[0] if evals else None, "eval_final": evals[-1] if evals else None,
            "eval_trajectory": evals, "trajectory": traj,
            "metric": evals[-1] if evals else None, "metric_name": "dist to equilibrium",
            "improved": bool(evals and evals[-1] < evals[0])}


register_benchmark("ill-cond", _ill_conditioned, higher_better=False, needs=("torch",),
                   description="Controlled ill-conditioned matrix regression (tune kappa), where "
                               "per-Hodge-component preconditioning should beat Adam. Metric: eval MSE.")
register_benchmark("matrix-completion", _matrix_completion, higher_better=False, needs=("torch",),
                   description="Non-convex low-rank matrix completion (tune obs/rank). Metric: held-out MSE.")
register_benchmark("mnist", lambda **kw: _image_clf("ylecun/mnist", "image", **kw),
                   higher_better=True, needs=("torch", "datasets"),
                   description="MNIST digits (MLP, or conv=True). Metric: test accuracy.")
register_benchmark("fashion-mnist", lambda **kw: _image_clf("zalando-datasets/fashion_mnist", "image", **kw),
                   higher_better=True, needs=("torch", "datasets"),
                   description="Fashion-MNIST (MLP, or conv=True). Metric: test accuracy.")
register_benchmark("cifar10", lambda **kw: _image_clf("uoft-cs/cifar10", "img", **kw),
                   higher_better=True, needs=("torch", "datasets"),
                   description="CIFAR-10 (MLP, or conv=True). Metric: test accuracy.")
