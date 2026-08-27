"""What the relational attention costs, against the thing it has to beat.

The claim in `relational_attention` is that the mixing operator is matrix-free and
so the n x n object is "never formed at scale". `CausalPropagatorAttention` as
written does form it: `scores = q @ k.transpose(-2,-1)` is [B,H,T,T] before the
mask, and the K propagator matvecs are added ON TOP of that. So it is strictly more
work than standard attention, and windowing changes which entries are masked rather
than how many are computed.

Measured here rather than argued. Two regimes, because they answer different
questions and only one of them is about generation:

  prefill   a full forward over T tokens. This is where the T x T cost lives and
            where the current class loses.
  decode    one token against a KV cache of length T. This is what token/s means,
            and where a bounded window is worth a great deal.

Run:  python -m agent.benchmarks.bench_attention_speed
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from rexgraph.nn.relational_attention import CausalPropagatorAttention

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def timed(fn, iters=20, warm=6):
    for _ in range(warm):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1e3


class FusedCausal(nn.Module):
    """torch's own fused causal attention: the baseline that has to be beaten."""

    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (z.view(B, T, self.h, self.dk).transpose(1, 2) for z in (q, k, v))
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, d))


def prefill(d=256, h=4, b=4, lengths=(512, 1024, 2048, 4096, 8192)):
    """A full forward over T tokens. The current class loses here, and windowing
    does not rescue it: the [B,H,T,T] is built either way."""
    rows = []
    for T in lengths:
        x = torch.randn(b, T, d, device=DEV)
        std = FusedCausal(d, h).to(DEV)
        prop = CausalPropagatorAttention(d, h, hops=4).to(DEV)
        band = CausalPropagatorAttention(d, h, hops=4, window=64, sparse=True).to(DEV)
        dense = CausalPropagatorAttention(d, h, hops=4, window=64, sparse=False).to(DEV)
        with torch.no_grad():
            rows.append((T,
                         timed(lambda m=std, v=x: m(v)),
                         timed(lambda m=prop, v=x: m(v)),
                         timed(lambda m=dense, v=x: m(v)),
                         timed(lambda m=band, v=x: m(v))))
    return rows


def decode(d=512, h=8, b=8, lengths=(256, 1024, 4096, 8192), window=64, hops=(1, 4, 8)):
    """One token against a KV cache of length T, which is what token/s measures.

    The window is what buys the speed; the hops buy back the reach the window gave
    up, at a cost proportional to K. Note this recomputes the w x w neighbour block
    every step, so it is PESSIMISTIC for the propagator: a real decoder caches it.
    """
    dk = d // h
    rows = []
    for T in lengths:
        Kc = torch.randn(b, h, T, dk, device=DEV)
        Vc = torch.randn(b, h, T, dk, device=DEV)
        q = torch.randn(b, h, 1, dk, device=DEV)

        def full(q=q, Kc=Kc, Vc=Vc):
            with torch.no_grad():
                s = (q @ Kc.transpose(-2, -1)) / math.sqrt(dk)
                return s.softmax(-1) @ Vc

        def prop(K, q=q, Kc=Kc, Vc=Vc):
            def f(q=q, Kc=Kc, Vc=Vc):
                with torch.no_grad():
                    Kw, Vw = Kc[:, :, -window:], Vc[:, :, -window:]
                    A = ((q @ Kw.transpose(-2, -1)) / math.sqrt(dk)).softmax(-1)
                    Aw = ((Kw @ Kw.transpose(-2, -1)) / math.sqrt(dk)).softmax(-1)
                    y, Vk = A @ Vw, Vw
                    for _ in range(K):
                        Vk = Aw @ Vk
                        y = y + A @ Vk
                    return y
            return f

        rows.append((T, timed(full), *[timed(prop(k)) for k in hops]))
    return rows


def main():
    print(f"device={DEV}  torch {torch.__version__}\n")
    print("PREFILL  full forward over T tokens (B=4 d=256 h=4)")
    print(f"  {'T':>6s} {'fused ms':>10s} {'dense ms':>10s} {'BANDED ms':>11s} "
          f"{'band/fused':>11s} {'band/dense':>11s}")
    for T, f, _p, dn, bd in prefill():
        print(f"  {T:6d} {f:10.3f} {dn:10.3f} {bd:11.3f} {bd / f:10.2f}x {bd / dn:10.2f}x")

    print("\nDECODE  one token against a KV cache (B=8 d=512 h=8, window=64)")
    print(f"  {'cache T':>8s} {'full ms':>9s} {'K=1 ms':>8s} {'K=4 ms':>8s} "
          f"{'K=8 ms':>8s} {'full/K=4':>9s}")
    for T, f, k1, k4, k8 in decode():
        print(f"  {T:8d} {f:9.4f} {k1:8.4f} {k4:8.4f} {k8:8.4f} {f / k4:8.2f}x")


if __name__ == "__main__":
    main()
