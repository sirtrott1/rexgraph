"""
archetypes - the model types, each built on the rexgraph.nn substrate.

Every archetype exposes: a `use_case`, a `defaults` param dict, the `data_kind` it consumes, a
`synth` generator, and `build(cfg, bundle)` returning an nn.Module. Select an archetype by name and
override its defaults. Register new archetypes with `register_archetype(...)`.

Components come from rexgraph.nn (HodgeAdam optimizer, PropagatorAttention, build_attention). These
archetypes assemble models from that substrate; they are not part of the library.
"""
from __future__ import annotations


import numpy as np
import torch as _t
import torch.nn as _nn
import torch.nn.functional as _F

import rexgraph.nn as R
from . import data as _data

ARCHETYPES: dict = {}


def register_archetype(name, *, use_case, data_kind, defaults, build, synth):
    ARCHETYPES[name] = {"name": name, "use_case": use_case, "data_kind": data_kind,
                        "defaults": dict(defaults), "build": build, "synth": synth}


def get(name):
    if name not in ARCHETYPES:
        raise KeyError(f"unknown archetype {name!r} (have: {', '.join(sorted(ARCHETYPES))})")
    return ARCHETYPES[name]


def merged_cfg(name, overrides=None):
    cfg = dict(get(name)["defaults"])
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


# MLP - tabular / vector

class MLP(_nn.Module):
    def __init__(self, d_in, n_out, d_hid=128, n_layers=2, task="classification"):
        super().__init__()
        dims = [d_in] + [d_hid] * n_layers
        self.body = _nn.ModuleList([_nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.head = _nn.Linear(dims[-1], n_out)
        self.task = task

    def forward(self, x):
        for lin in self.body:
            x = _F.gelu(lin(x))
        return self.head(x)


def _build_mlp(cfg, bundle):
    d_in = bundle.meta.get("feat_dim", cfg["feat_dim"])
    n_out = 1 if cfg["task"] == "regression" else bundle.meta.get("n_classes", cfg["n_classes"])
    return MLP(d_in, n_out, cfg["d_hid"], cfg["n_layers"], cfg["task"])


register_archetype(
    "mlp", use_case="Tabular / vector data - classification or regression.",
    data_kind="vector",
    defaults={"feat_dim": 16, "n_classes": 4, "d_hid": 128, "n_layers": 2, "task": "classification"},
    build=_build_mlp, synth=lambda cfg, seed: _data.synth_vectors(
        feat_dim=cfg["feat_dim"], n_classes=cfg["n_classes"], seed=seed))


# CNN - images

class CNN(_nn.Module):
    def __init__(self, in_ch, n_classes, depth=2, width=32, norm=True):
        super().__init__()
        layers, c, w = [], in_ch, width
        for _ in range(depth):
            layers += [_nn.Conv2d(c, w, 3, padding=1)]
            if norm:
                layers.append(_nn.BatchNorm2d(w))
            layers += [_nn.ReLU(), _nn.MaxPool2d(2)]
            c, w = w, min(w * 2, 256)
        layers += [_nn.AdaptiveAvgPool2d(2), _nn.Flatten(), _nn.Linear(c * 4, n_classes)]
        self.net = _nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _build_cnn(cfg, bundle):
    return CNN(bundle.meta.get("in_channels", cfg["in_channels"]),
               bundle.meta.get("n_classes", cfg["n_classes"]),
               cfg["depth"], cfg["width"], cfg["norm"])


register_archetype(
    "cnn", use_case="Image classification. norm=False is where HodgeAdam's conditioning edge shows.",
    data_kind="image",
    defaults={"in_channels": 3, "n_classes": 4, "depth": 2, "width": 32, "norm": True},
    build=_build_cnn, synth=lambda cfg, seed: _data.synth_images(
        c=cfg["in_channels"], n_classes=cfg["n_classes"], seed=seed))


# LM - sequences (relational or standard attention)

class LM(_nn.Module):
    def __init__(self, vocab, d=64, n_head=4, n_layer=2, seq_len=64, attention="relational"):
        super().__init__()
        self.tok = _nn.Embedding(vocab, d); self.pos = _nn.Embedding(seq_len, d)
        self.blocks = _nn.ModuleList()
        self.attn_used = None
        for _ in range(n_layer):
            attn, used = R.build_attention(attention, d, n_head)   # from rexgraph.nn
            self.attn_used = used
            self.blocks.append(_nn.ModuleDict({
                "ln1": _nn.LayerNorm(d), "attn": attn, "ln2": _nn.LayerNorm(d),
                "mlp": _nn.Sequential(_nn.Linear(d, 4 * d), _nn.GELU(), _nn.Linear(4 * d, d))}))
        self.lnf = _nn.LayerNorm(d); self.head = _nn.Linear(d, vocab)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(_t.arange(T, device=idx.device))[None]
        for b in self.blocks:
            a, _ = b["attn"](b["ln1"](x)); x = x + a
            x = x + b["mlp"](b["ln2"](x))
        return self.head(self.lnf(x))


def _build_lm(cfg, bundle):
    return LM(bundle.meta.get("vocab", cfg["vocab"]), cfg["d"], cfg["n_head"], cfg["n_layer"],
              bundle.meta.get("seq_len", cfg["seq_len"]), cfg["attention"])


register_archetype(
    "lm", use_case="Sequence / language modeling (next-token). attention: 'relational' (propagator) "
                   "or 'standard'.",
    data_kind="sequence",
    defaults={"vocab": 24, "d": 64, "n_head": 4, "n_layer": 2, "seq_len": 24, "attention": "relational"},
    build=_build_lm, synth=lambda cfg, seed: _data.synth_sequences(
        vocab=cfg["vocab"], seq_len=cfg["seq_len"], seed=seed))


# HGNN - relational-complex hypergraph net (advection + diffusion)

class _FlowLayer(_nn.Module):
    """Advection + diffusion: heat propagator on L0 (topology from |B1|) + a directed
    gradient->disperse->divergence flow through the signed incidence B1 (orientation)."""
    def __init__(self, d, K=10):
        super().__init__()
        self.Wd = _nn.Linear(d, d, bias=False); self.Wg = _nn.Linear(d, d, bias=False)
        self.Wc = _nn.Linear(d, d, bias=False); self.Wv = _nn.Linear(d, d, bias=False)
        self.log_t0 = _nn.Parameter(_t.tensor(0.0)); self.log_t1 = _nn.Parameter(_t.tensor(0.0))
        self.K = K

    def forward(self, h0, h1, L0, L1, B1, lam0, lam1):
        diff = R.rcf_torch.heat_apply(L0, self.Wd(h0), self.log_t0.exp(), K=self.K, lam_max=lam0)
        cell = B1.t() @ self.Wg(h0) + self.Wc(h1)
        cell = R.rcf_torch.heat_apply(L1, cell, self.log_t1.exp(), K=self.K, lam_max=lam1)
        return _F.gelu(diff + B1 @ self.Wv(cell)), _F.gelu(cell)


class HGNN(_nn.Module):
    def __init__(self, d_in, n_classes, he_ptr, he_idx, d_hid=32, n_layers=2, flow=True):
        super().__init__()
        from rexgraph.graph import RexGraph
        g = RexGraph.from_hypergraph(np.asarray(he_ptr, "int32"), np.asarray(he_idx, "int32"))
        self.register_buffer("B1", _t.as_tensor(np.asarray(g.B1_dense, "float32")))
        self.register_buffer("L0", _t.as_tensor(np.asarray(g.L0, "float32")))
        self.register_buffer("L1", _t.as_tensor(np.asarray(g.L1, "float32")))
        self.lam0 = R.rcf_torch.spectral_bound(self.L0)
        self.lam1 = R.rcf_torch.spectral_bound(self.L1) if self.L1.numel() else 1.0
        self.enc = _nn.Linear(d_in, d_hid); self.he0 = _nn.Parameter(_t.zeros(1, d_hid))
        self.layers = _nn.ModuleList([_FlowLayer(d_hid) for _ in range(n_layers)])
        self.head = _nn.Linear(d_hid, n_classes); self.flow = flow

    def forward(self, X):
        h0 = self.enc(X); h1 = self.he0.expand(self.B1.shape[1], -1).contiguous()
        for layer in self.layers:
            h0, h1 = layer(h0, h1, self.L0, self.L1, self.B1, self.lam0, self.lam1)
        return self.head(h0)


def _build_hgnn(cfg, bundle):
    return HGNN(bundle.meta.get("feat_dim", cfg["feat_dim"]),
                bundle.meta.get("n_classes", cfg["n_classes"]),
                bundle.extra["he_ptr"], bundle.extra["he_idx"],
                cfg["d_hid"], cfg["n_layers"], cfg["flow"])


register_archetype(
    "hgnn", use_case="Node classification on hypergraphs / higher-order relational data. "
                     "Fiber-bundle advection+diffusion; uses the complex's signed orientation.",
    data_kind="hypergraph",
    defaults={"feat_dim": 16, "n_classes": 4, "d_hid": 32, "n_layers": 2, "flow": True,
              "oriented": False},
    build=_build_hgnn, synth=lambda cfg, seed: _data.synth_hypergraph(
        feat_dim=cfg["feat_dim"], n_classes=cfg["n_classes"], oriented=cfg.get("oriented", False),
        seed=seed))
