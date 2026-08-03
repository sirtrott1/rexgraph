"""
The intrinsic model: both relational organs baked into one small transformer, trained
end-to-end on associative recall (retrieve the value bound to a distant key — importance,
not likelihood). A 2×2 ablation isolates each organ:

    attention ∈ {standard softmax, propagator (relational)}
    optimizer ∈ {Adam, HodgeAdam (vector Hodge decomposition)}

so we see the conventional baseline (standard+Adam), each organ alone, and both together.
Bidirectional encoder (the symmetric propagator).

HodgeAdam is named explicitly here because it is one arm of the ablation, not because it is a
recommended default: this model is feature-space, and the routing default (make_optimizer("auto"))
gives it plain Adam. The cell measures what HodgeAdam does to a standard transformer; that is the
question, so naming it is the point.

Run:  python -m agent.benchmarks.bench_intrinsic_model [--quick]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from agent.benchmarks.bench_associative_recall import (
    Encoder,
    StandardAttention,
    make_batch,
)

# straight from _experimental: the A/B arm names the demoted optimizer on purpose, so it reads
# from where it lives rather than through optim's back-compat re-export.
from rexgraph.nn._experimental import HodgeAdam
from rexgraph.nn.optim import pick_device
from rexgraph.nn.relational_attention import PropagatorAttention


def run(attn_kind, opt_kind, *, seed, steps, device, lr=3e-3,
        n_pairs=6, n_keys=8, n_vals=16, d=64, n_head=4, n_layer=2, bs=256):
    torch.manual_seed(seed)
    vocab = n_keys + n_vals + 1
    T = 2 * n_pairs + 2

    def mk_attn():
        if attn_kind == "standard":
            return StandardAttention(d, n_head)
        return PropagatorAttention(d, n_head, channels=("heat", "curl"), cheb_order=16)

    model = Encoder(vocab, T, d, n_head, n_layer, mk_attn).to(device)
    if opt_kind == "hodge":
        opt = HodgeAdam(model.parameters(), lr=lr)        # the ablation arm, named on purpose
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        seq, tgt, _, _ = make_batch(bs, n_pairs, n_keys, n_vals, device)
        loss = F.cross_entropy(model(seq)[:, -1, :], tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        seq, tgt, _, _ = make_batch(2048, n_pairs, n_keys, n_vals, device)
        acc = (model(seq)[:, -1, :].argmax(-1) == tgt).float().mean().item()
    return acc, dt


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    device = a.device or pick_device()
    steps = 1000 if a.quick else 2000
    seeds = [0, 1] if a.quick else [0, 1, 2]
    print("device=%s steps=%d seeds=%s  (associative recall, 2×2 ablation)\n" % (device, steps, seeds))
    print("  %-24s %-10s %-10s" % ("", "Adam", "HodgeAdam(vector)"))
    for attn in ("standard", "propagator"):
        cells = []
        for opt in ("adam", "hodge"):
            accs = [run(attn, opt, seed=s, steps=steps, device=device)[0] for s in seeds]
            cells.append(f"{float(np.mean(accs)):.3f}±{float(np.std(accs)):.3f}")
        tag = "standard attn" if attn == "standard" else "propagator attn"
        print("  %-24s %-10s %-10s" % (tag, cells[0], cells[1]))
    print("\n(top-left = conventional baseline; bottom-right = both relational organs)")


if __name__ == "__main__":
    main()
