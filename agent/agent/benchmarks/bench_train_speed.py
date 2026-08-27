"""What the relational optimizer costs per step, against Adam.

Step time is not the whole story, since an optimizer that costs more per step can still
win if it needs fewer of them, but it is the half that is a speed question, and it
is the half that was never written down. Convergence is a separate measurement and
this file does not claim anything about it.

`make_optimizer("auto")` is the honest router: it picks GreensCochain only for a
model exposing `greens_groups()`, and plain Adam otherwise. The MLP here has no
such groups, so "auto" routing to Adam is the router working, not a fallback.

Run:  python -m agent.benchmarks.bench_train_speed
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from rexgraph.nn.factory import make_optimizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def step_cost(name, *, d=512, batch=256, iters=40, warm=8, lr=1e-3):
    """Wall-clock of one full forward + backward + step."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d),
                          nn.GELU(), nn.Linear(d, 10)).to(DEV)
    opt, label = make_optimizer(name, model, list(model.parameters()), lr=lr)
    x = torch.randn(batch, d, device=DEV)
    y = torch.randint(0, 10, (batch,), device=DEV)
    lossf = nn.CrossEntropyLoss()

    def one():
        opt.zero_grad(set_to_none=True)
        lossf(model(x), y).backward()
        opt.step()

    for _ in range(warm):
        one()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    _sync()
    return (time.perf_counter() - t0) / iters * 1e3, label


def main():
    print(f"device={DEV}  torch {torch.__version__}")
    print("a plain MLP, so 'auto' is expected to route to Adam\n")
    print(f"  {'requested':14s} {'resolved':26s} {'step ms':>9s} {'vs adam':>8s}")
    base = None
    for name in ("adam", "auto", "greens", "hodge"):
        try:
            ms, label = step_cost(name)
        except Exception as e:                       # a retired path may not build
            print(f"  {name:14s} {type(e).__name__}: {str(e)[:50]}")
            continue
        if base is None:
            base = ms
        print(f"  {name:14s} {label:26s} {ms:9.3f} {ms / base:7.2f}x")


if __name__ == "__main__":
    main()
