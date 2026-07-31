"""
data - load, synthesize, and split data for the model archetypes.

A DataBundle carries one training set. `kind` tells the trainer how to feed the model
(vector / image / sequence / hypergraph), `X`/`y` are the tensors, `meta` carries shapes
(feat_dim, n_classes, vocab, ...), `splits` holds train/val/test index tensors, and `extra`
holds structure (e.g. a hypergraph's CSR incidence). Build a bundle from files, a HF dataset,
or a per-archetype synthetic generator.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np

try:
    import torch as _t
    _HAS_TORCH = True
except Exception:                                    # pragma: no cover
    _HAS_TORCH = False


@dataclass
class DataBundle:
    kind: str                              # vector | image | sequence | hypergraph
    X: object = None
    y: object = None
    meta: dict = field(default_factory=dict)
    splits: dict = field(default_factory=dict)      # {"train": idx, "val": idx, "test": idx}
    extra: dict = field(default_factory=dict)

    def to(self, device):
        if _HAS_TORCH and hasattr(self.X, "to"):
            self.X = self.X.to(device)
        if _HAS_TORCH and hasattr(self.y, "to"):
            self.y = self.y.to(device)
        return self


def make_splits(n: int, ratios=(0.6, 0.2, 0.2), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a = int(ratios[0] * n); b = a + int(ratios[1] * n)
    idx = lambda s: (_t.as_tensor(s) if _HAS_TORCH else s)
    return {"train": idx(perm[:a]), "val": idx(perm[a:b]), "test": idx(perm[b:])}


# file loaders (files / HF) -> vectors or text

def load_table(source, *, x_cols=None, y_col="label", limit=None):
    """Load a numeric table (.csv/.jsonl/.npz) into a vector DataBundle. `x_cols` selects feature
    columns (default: all numeric except `y_col`); `y_col` is the label."""
    p = os.path.expanduser(str(source))
    rows = []
    if p.endswith(".jsonl"):
        with open(p) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    elif p.endswith(".csv"):
        with open(p, newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    elif p.endswith(".npz"):
        d = np.load(p)
        X, y = d["X"].astype("float32"), d["y"].astype("int64")
        return _vector_bundle(X, y)
    else:
        raise ValueError(f"unsupported table format: {source}")
    if limit:
        rows = rows[:int(limit)]
    keys = x_cols or [k for k in rows[0] if k != y_col]
    X = np.array([[float(r[k]) for k in keys] for r in rows], dtype="float32")
    y = np.array([int(float(r[y_col])) for r in rows], dtype="int64")
    return _vector_bundle(X, y)


def load_text(source, *, vocab_size=256, seq_len=64, limit=None):
    """Load a text file into a byte-level sequence DataBundle for LM training."""
    p = os.path.expanduser(str(source))
    with open(p, "rb") as f:
        data = f.read()
    if limit:
        data = data[:int(limit)]
    ids = np.frombuffer(data, dtype=np.uint8).astype("int64") % vocab_size
    n = len(ids) // (seq_len + 1)
    ids = ids[:n * (seq_len + 1)].reshape(n, seq_len + 1)
    X = _as(ids[:, :seq_len]); y = _as(ids[:, 1:seq_len + 1])
    b = DataBundle("sequence", X, y, meta={"vocab": vocab_size, "seq_len": seq_len})
    b.splits = make_splits(n)
    return b


def _as(a):
    return _t.as_tensor(np.ascontiguousarray(a)) if _HAS_TORCH else a


def _vector_bundle(X, y):
    n = len(X)
    b = DataBundle("vector", _as(X), _as(y),
                   meta={"feat_dim": X.shape[1], "n_classes": int(y.max()) + 1})
    b.splits = make_splits(n)
    return b


# synthetic generators (one per archetype; run without external data)

def synth_vectors(n=800, feat_dim=16, n_classes=4, sep=1.5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    centers = rng.normal(0, sep, (n_classes, feat_dim))
    X = (centers[y] + rng.normal(0, 1, (n, feat_dim))).astype("float32")
    return _vector_bundle(X, y.astype("int64"))


def synth_images(n=800, c=3, hw=16, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    base = rng.normal(0, 1, (n_classes, c, hw, hw))
    X = (base[y] + rng.normal(0, 0.6, (n, c, hw, hw))).astype("float32")
    b = DataBundle("image", _as(X), _as(y.astype("int64")),
                   meta={"in_channels": c, "hw": hw, "n_classes": n_classes})
    b.splits = make_splits(n)
    return b


def synth_sequences(n=1024, vocab=24, seq_len=24, period=6, seed=0):
    """Periodic-copy task: the token at t equals the token `period` steps back. Routing
    information a fixed hop distance is what the propagator is built for."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, vocab, (n, period))
    full = np.tile(base, (1, seq_len // period + 2))[:, :seq_len + 1]
    X = _as(full[:, :seq_len].astype("int64")); y = _as(full[:, 1:seq_len + 1].astype("int64"))
    b = DataBundle("sequence", X, y, meta={"vocab": vocab, "seq_len": seq_len})
    b.splits = make_splits(n)
    return b


def synth_hypergraph(n_nodes=500, n_hyperedges=600, edge_size=5, n_classes=4,
                     feat_dim=16, homophily=0.75, oriented=False, feat_noise=1.4, seed=0):
    """Contextual hypergraph SBM (homophily) or, with oriented=True, a potential-gradient task
    where the hyperedge orientation carries the signal (features near-noise)."""
    rng = np.random.default_rng(seed)
    if oriented:
        potential = rng.normal(size=n_nodes)
        order = np.argsort(potential)
        y = np.zeros(n_nodes, "int64")
        for c, ch in enumerate(np.array_split(order, n_classes)):
            y[ch] = c
        idx, ptr = [], [0]
        for _ in range(n_hyperedges):
            m = rng.choice(n_nodes, edge_size, replace=False)
            m = m[np.argsort(potential[m])]
            idx.extend(int(x) for x in m); ptr.append(len(idx))
        X = rng.normal(0, feat_noise, (n_nodes, feat_dim)).astype("float32")
    else:
        y = rng.integers(0, n_classes, n_nodes)
        byc = [np.where(y == c)[0] for c in range(n_classes)]
        idx, ptr = [], [0]
        for _ in range(n_hyperedges):
            c = rng.integers(0, n_classes); mem = []
            for _ in range(edge_size):
                mem.append(int(rng.choice(byc[c])) if rng.random() < homophily and len(byc[c])
                           else int(rng.integers(0, n_nodes)))
            mem = list(dict.fromkeys(mem))
            if len(mem) >= 2:
                idx.extend(mem); ptr.append(len(idx))
        proto = rng.normal(0, 2, (n_classes, feat_dim))
        X = (proto[y] + rng.normal(0, feat_noise, (n_nodes, feat_dim))).astype("float32")
    b = DataBundle("hypergraph", _as(X), _as(np.asarray(y, "int64")),
                   meta={"feat_dim": feat_dim, "n_classes": n_classes, "n_nodes": n_nodes},
                   extra={"he_ptr": np.array(ptr, "int32"), "he_idx": np.array(idx, "int32")})
    b.splits = make_splits(n_nodes)
    return b
