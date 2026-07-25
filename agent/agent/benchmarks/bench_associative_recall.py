"""
In-model test of the corrected attention on ASSOCIATIVE RECALL — the task that punishes the
copy/likelihood reflex. A sequence of (key,value) pairs, then a query key; predict the value
BOUND TO THAT KEY (seen earlier), not the locally-recent token. The load-bearing token is
distant and not content-obvious, so importance-gated relational routing should help where
plain content attention drifts.

Compares (bidirectional encoder, Adam, to isolate the attention organ):
  * standard softmax attention
  * propagator attention (heat+curl), importance OFF
  * propagator attention (heat+curl), importance ON  ← the corrected organ

Run:  python -m agent.benchmarks.bench_associative_recall [--quick]
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rexgraph.nn.optim import pick_device
from rexgraph.nn.relational_attention import PropagatorAttention


def make_batch(bs, n_pairs, n_keys, n_vals, device):
    """[k1,v1,...,kn,vn, QUERY, kq] ; target at last pos = value bound to kq. Vectorized."""
    QUERY = n_keys + n_vals
    T = 2 * n_pairs + 2
    keys = torch.rand(bs, n_keys, device=device).argsort(dim=1)[:, :n_pairs]   # per-row perm
    vals = torch.randint(0, n_vals, (bs, n_pairs), device=device)
    seq = torch.empty(bs, T, dtype=torch.long, device=device)
    seq[:, 0:2 * n_pairs:2] = keys
    seq[:, 1:2 * n_pairs:2] = n_keys + vals
    qi = torch.randint(0, n_pairs, (bs,), device=device)
    ar = torch.arange(bs, device=device)
    seq[:, -2] = QUERY
    seq[:, -1] = keys[ar, qi]
    tgt = n_keys + vals[ar, qi]
    return seq, tgt, T, QUERY


class StandardAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.h = n_head; self.dk = d // n_head
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)

    def forward(self, x, return_diag=False):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1) / math.sqrt(self.dk)).softmax(-1)
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, d)), None


class Block(nn.Module):
    def __init__(self, d, n_head, attn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.attn = attn
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        a, _ = self.attn(self.ln1(x))
        x = x + a
        return x + self.mlp(self.ln2(x))


class Encoder(nn.Module):
    def __init__(self, vocab, T, d, n_head, n_layer, mk_attn):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, n_head, mk_attn()) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))


def run(mk_attn, *, seed, steps, device, n_pairs=6, n_keys=8, n_vals=16,
        d=64, n_head=4, n_layer=2, bs=256):
    torch.manual_seed(seed)
    vocab = n_keys + n_vals + 1
    T = 2 * n_pairs + 2
    model = Encoder(vocab, T, d, n_head, n_layer, mk_attn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    if device == "cuda":
        torch.cuda.synchronize(); t0 = time.time()
    else:
        t0 = time.time()
    for _ in range(steps):
        seq, tgt, _, _ = make_batch(bs, n_pairs, n_keys, n_vals, device)
        logits = model(seq)[:, -1, :]                # predict at the query position
        loss = F.cross_entropy(logits, tgt)
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
    steps = 800 if a.quick else 2500
    seeds = [0, 1] if a.quick else [0, 1, 2]
    print("device=%s steps=%d seeds=%s  (associative recall)\n" % (device, steps, seeds))

    configs = [
        ("standard", lambda: StandardAttention(64, 4)),
        ("propagator (imp OFF)", lambda: PropagatorAttention(64, 4, channels=("heat", "curl"),
                                                             cheb_order=16, importance=False)),
        ("propagator (imp ON)", lambda: PropagatorAttention(64, 4, channels=("heat", "curl"),
                                                            cheb_order=16, importance=True)),
    ]
    for name, mk in configs:
        accs, ts = [], []
        for s in seeds:
            acc, dt = run(mk, seed=s, steps=steps, device=device)
            accs.append(acc); ts.append(dt)
        print("  %-22s acc=%.3f±%.3f  %.1fs" % (name, float(np.mean(accs)),
                                                float(np.std(accs)), float(np.mean(ts))))


if __name__ == "__main__":
    main()
