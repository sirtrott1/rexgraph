"""
data: load, synthesize, and split data for the model archetypes.

A DataBundle carries one training set. `kind` tells the trainer how to feed the model
(vector / image / sequence / hypergraph), `X`/`y` are the tensors, `meta` carries shapes
(feat_dim, n_classes, vocab, ...), `splits` holds train/val/test index tensors, and `extra`
holds structure (e.g. a hypergraph's CSR incidence). Build a bundle from files, a HF dataset,
or a per archetype synthetic generator.
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

    def fetch(self, indices):
        """Return one feature/target batch on this bundle's selected device."""
        return self.X[indices], None if self.y is None else self.y[indices]


class MappedDataBundle(DataBundle):
    """Read only .npy tables with bounded feature transfers and stable class IDs.

    Splits are ordered ranges; only sampling within the training split is random.
    Shuffle the files beforehand when a random split is appropriate. Values are
    validated when read, so corrupt rows outside a fetched batch are not scanned.
    """
    def to(self, device):
        if not _HAS_TORCH:
            raise ImportError("mapped training requires torch")
        self.device = _t.device(device)
        return self

    def fetch(self, indices):
        if isinstance(indices, range):
            indices = slice(indices.start, indices.stop, indices.step)
        elif _HAS_TORCH and isinstance(indices, _t.Tensor):
            indices = indices.detach().cpu().numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            X = np.array(self.X[indices], dtype=np.float32, copy=True)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("mapped features must be finite and representable in float32")
        y = None
        if self.y is not None:
            raw = self.y[indices]
            if self.meta["task"] == "classification":
                if np.any(raw < 0) or np.any(raw >= self.meta["n_classes"]):
                    raise ValueError("mapped class IDs must be within the declared n_classes")
                y = np.array(raw, dtype=np.int64, copy=True)
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    y = np.array(raw, dtype=np.float32, copy=True)
                if not np.isfinite(y).all():
                    raise ValueError("mapped regression targets must be finite float32 values")
        def move(a):
            t = _t.from_numpy(a)
            if self.device.type == "cuda" and self.pin_memory:
                t = t.pin_memory()
            return t.to(self.device, non_blocking=self.pin_memory)
        return move(X), None if y is None else move(y)


def load_mapped_table(features, targets=None, *, task="classification", n_classes=None,
                      ratios=(0.6, 0.2, 0.2), eval_batch_size=1024, pin_memory=True):
    """Map separate feature/target .npy files without loading the whole dataset.

    Classification requires an explicit class count and integer target storage;
    class IDs are never inferred or remapped per batch. Regression uses one scalar
    target per row. Evaluation/fusion read at most ``eval_batch_size`` rows at once.
    ``to(device)`` selects the batch destination; the full files remain on disk.
    Pinning enables nonblocking CUDA/ROCm copies on Torch's current stream; it does
    not implement prefetch or promise overlap with the next CPU batch.
    """
    if not _HAS_TORCH:
        raise ImportError("mapped training requires torch")
    if task not in ("classification", "regression"):
        raise ValueError("vector task must be classification or regression")
    if (isinstance(eval_batch_size, (bool, np.bool_)) or
            not isinstance(eval_batch_size, (int, np.integer)) or eval_batch_size < 1):
        raise ValueError("eval_batch_size must be a positive integer")
    ratios = np.asarray(ratios, dtype=float)
    if (ratios.shape != (3,) or not np.isfinite(ratios).all() or
            np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("split ratios must be three nonnegative values summing to one")
    def mapped(path):
        p = os.path.expanduser(os.fspath(path))
        if not p.endswith(".npy"):
            raise ValueError("mapped tables require separate .npy files")
        return np.load(p, mmap_mode="r", allow_pickle=False)
    X = mapped(features)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or X.dtype.kind not in "iuf":
        raise ValueError("mapped features must be a nonempty real matrix")
    y = None if targets is None else mapped(targets)
    meta = {"feat_dim": X.shape[1], "task": task}
    if y is not None:
        if y.shape != (len(X),) or y.dtype.kind not in "iuf":
            raise ValueError("mapped targets need one real value per row")
        if task == "classification" and y.dtype.kind not in "iu":
            raise ValueError("mapped classification targets require integer storage")
    if task == "classification":
        if (isinstance(n_classes, (bool, np.bool_)) or
                not isinstance(n_classes, (int, np.integer)) or not 1 <= n_classes < 2**63):
            raise ValueError("mapped classification requires a positive integer n_classes")
        meta["n_classes"] = int(n_classes)
    a, b = int(ratios[0] * len(X)), int((ratios[0] + ratios[1]) * len(X))
    bundle = MappedDataBundle("vector", X, y, meta,
                              {"train": range(a), "val": range(a, b), "test": range(b, len(X))})
    bundle.eval_batch_size = int(eval_batch_size)
    bundle.pin_memory = bool(pin_memory)
    return bundle.to("cpu")


def make_splits(n: int, ratios=(0.6, 0.2, 0.2), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a = int(ratios[0] * n); b = a + int(ratios[1] * n)
    idx = lambda s: (_t.as_tensor(s) if _HAS_TORCH else s)
    return {"train": idx(perm[:a]), "val": idx(perm[a:b]), "test": idx(perm[b:])}


# file loaders (files / HF) -> vectors or text

def load_table(source, *, x_cols=None, y_col="label", limit=None, task="classification"):
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
        with np.load(p, allow_pickle=False) as d:
            X, y = d["X"], d["y"]
        if limit:
            X, y = X[:int(limit)], y[:int(limit)]
        return _vector_bundle(X, y, task=task)
    else:
        raise ValueError(f"unsupported table format: {source}")
    if limit:
        rows = rows[:int(limit)]
    keys = x_cols or [k for k in rows[0] if k != y_col]
    X = np.array([[float(r[k]) for k in keys] for r in rows], dtype="float32")
    y = np.array([float(r[y_col]) for r in rows], dtype="float64")
    return _vector_bundle(X, y, task=task)


def load_text(source, *, vocab_size=256, seq_len=64, limit=None):
    """Load a text file into a byte level sequence DataBundle for LM training."""
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


def _vector_bundle(X, y, *, task="classification"):
    if task not in ("classification", "regression"):
        raise ValueError("vector task must be classification or regression")
    if any(np.ma.isMaskedArray(a) and np.ma.getmaskarray(a).any() for a in (X, y)):
        raise ValueError("model data requires explicit unmasked values")
    X = np.asarray(X)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or X.dtype.kind not in "iuf":
        raise ValueError("model features must be a nonempty real matrix")
    if y is not None:
        y = np.asarray(y)
        if y.dtype.kind in "US" or (y.dtype.kind == "O" and
                all(isinstance(v, (str, bytes)) for v in y.flat)):
            y = y.astype("float64")
        if y.shape != (len(X),) or y.dtype.kind not in "iuf" or not np.isfinite(y).all():
            raise ValueError("model targets must be one finite real value per feature row")
        if task == "classification":
            if np.any(y < 0) or np.any(y >= 2**63) or np.any(y != np.floor(y)):
                raise ValueError("classification targets must be nonnegative integer class indices")
            y = y.astype("int64")
        else:
            y = y.astype("float64")
    with np.errstate(over="ignore", invalid="ignore"):
        X = X.astype("float32")
    if not np.isfinite(X).all():
        raise ValueError("model features must be finite and representable in float32")
    n = len(X)
    meta = {"feat_dim": X.shape[1], "task": task}
    if task == "classification" and y is not None:
        meta["n_classes"] = int(y.max()) + 1
    b = DataBundle("vector", _as(X), None if y is None else _as(y),
                   meta=meta)
    b.splits = make_splits(n)
    return b


# synthetic generators (one per archetype; run without external data)

def synth_vectors(n=800, feat_dim=16, n_classes=4, sep=1.5, seed=0, task="classification"):
    rng = np.random.default_rng(seed)
    if task == "regression":
        X = rng.normal(size=(n, feat_dim)).astype("float32")
        y = X @ rng.normal(size=feat_dim) + rng.normal(0, .1, n)
        return _vector_bundle(X, y, task=task)
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
    """Periodic copy task: the token at t equals the token `period` steps back. Routing
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
    """Contextual hypergraph SBM (homophily) or, with oriented=True, a potential gradient task
    where the hyperedge orientation carries the signal (features near noise)."""
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
