"""Bridge Agent models to rexgraph.io and RCDB.

load_bundle reads supported tables, vector corpora, RCBD graphs and text into a
DataBundle. save_checkpoint writes weights.safetensors, config.json and a vector
trajectory to a local directory. save_complex_rex writes the bundle's graph to
RCBD. to_rcdb publishes that graph through the selected RCDB backend.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import rexgraph.io as rio

from ..formats import BUNDLE_SUFFIXES
from . import archetypes as A
from . import data as D

# load: any rexgraph.io source -> a DataBundle

def load_bundle(source, *, y_col="label", x_cols=None, table=None, limit=None,
                task="classification") -> D.DataBundle:
    """Load training data from any rexgraph.io-supported source into a DataBundle.
      .parquet          -> vector bundle (feature columns + `y_col`)
      .safetensors      -> vector corpus written by save_vectors, or an embedding corpus
      .rcbd              -> relational complex (hypergraph bundle for hgnn)
      sqlalchemy URI    -> pass `table=`; reads a numeric table
      .csv/.jsonl/.npz  -> table;  .txt -> byte sequence (for lm)
    """
    s = str(source)
    if s.endswith(".parquet"):
        cols = rio.read_parquet(s, columns=(list(x_cols) + [y_col]) if x_cols else None)
        y = cols.pop(y_col)
        keys = x_cols or list(cols)
        X = np.stack([np.asarray(cols[k], "float32") for k in keys], axis=1)
        return D._vector_bundle(X, y, task=task)
    if s.endswith(".safetensors"):
        X, labels, feat_names, meta = rio.load_vectors(s)
        return D._vector_bundle(np.asarray(X, "float32"), labels, task=task)
    if s.endswith(BUNDLE_SUFFIXES):
        return _bundle_from_rex(rio.load_rcbd(s))
    if table is not None:                                   # a database URI + table
        eng = rio.get_engine(s)
        # The ENGINE is cached for the life of the process, which is what makes asking
        # for one per load correct. A checked out CONNECTION is not: it has to go back
        # to the pool when this read is done, or every load holds one open.
        def read(conn):
            batches = list(rio.read_sql_batches(conn, table))
            if not batches:
                raise ValueError("model training table is empty")
            return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
        if hasattr(eng, "connect"):
            with eng.connect() as conn:
                rows = read(conn)
        else:
            rows = read(eng)
        y = rows.pop(y_col)
        keys = x_cols or list(rows)
        X = np.stack([np.asarray(rows[k], "float32") for k in keys], axis=1)
        return D._vector_bundle(X, y, task=task)
    if s.endswith(".txt"):
        return D.load_text(s, limit=limit)
    return D.load_table(s, x_cols=x_cols, y_col=y_col, limit=limit, task=task)


def _bundle_from_rex(rex):
    """Convert a RexGraph to a hypergraph DataBundle.

    Read boundary_ptr and boundary_idx directly to preserve participant order and
    declared vertex count, including isolated vertices.
    """
    ptr = np.asarray(rex.boundary_ptr, dtype=np.int64)
    idx = np.asarray(rex.boundary_idx, dtype=np.int64)
    meta = {"feat_dim": 0, "n_classes": 0, "n_nodes": int(rex.nV)}
    return D.DataBundle("hypergraph", None, None, meta,
                        extra={"he_ptr": ptr.astype("int32"), "he_idx": idx.astype("int32")})


# save: a checkpoint on the rexgraph IO stack

def save_checkpoint(path, model, archetype, cfg, *, bundle=None, result=None) -> str:
    """Persist a trained model as a directory: weights.safetensors + config.json + (if a training
    `result` is given) trajectory.safetensors written through rexgraph.io.save_vectors."""
    from safetensors.torch import save_file
    d = Path(os.path.expanduser(path)); d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()},
              str(d / "weights.safetensors"))
    meta = dict(bundle.meta) if bundle is not None else {}
    extra = {}
    if bundle is not None and bundle.extra:                 # hgnn: keep the complex for rebuild
        extra = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in bundle.extra.items()}
    (d / "config.json").write_text(json.dumps(
        {"archetype": archetype, "cfg": cfg, "meta": meta, "extra": extra}, indent=2))
    if result and result.get("trajectory"):
        traj = np.asarray(result["trajectory"], "float32").reshape(-1, 1)
        labels = np.array([f"step_{i}" for i in range(len(traj))])
        rio.save_vectors(traj, labels, str(d / "trajectory.safetensors"),
                         feature_names=[result.get("metric_name", "metric")],
                         metadata={"kind": "training_trajectory", "archetype": archetype,
                                   "optimizer": str(result.get("optimizer_class"))})
    return str(d)


def load_checkpoint(path, *, device=None):
    """Rebuild a model from a checkpoint and load its weights onto the resolved device. Returns
    (model, config).

    Device agnostic: a checkpoint saved on any backend (weights are written CPU contiguous by
    ``save_checkpoint``) loads onto whatever ``device`` resolves to through ``rexgraph.nn.pick_device``
    - None/'auto' rides the compute stack's recommended backend, 'cpu' forces CPU. The weights are
    map located to that device (safetensors ``device=`` is torch's ``map_location`` equivalent, with a
    CPU load fallback), and the model is moved there, so save here / load there always works."""
    from safetensors.torch import load_file

    import rexgraph.nn as R
    dev = R.pick_device(device)
    d = Path(os.path.expanduser(path))
    conf = json.loads((d / "config.json").read_text())
    stub = D.DataBundle(A.get(conf["archetype"])["data_kind"], None, None,
                        dict(conf["meta"]),
                        extra={k: np.asarray(v, "int32") if k in ("he_ptr", "he_idx") else v
                               for k, v in conf.get("extra", {}).items()})
    model = A.get(conf["archetype"])["build"](conf["cfg"], stub)
    weights = str(d / "weights.safetensors")
    try:                                            # map_location straight onto the picked device
        state = load_file(weights, device=str(dev))
    except Exception:                               # backend can't map on load (e.g. mps): via cpu
        state = load_file(weights)
    model.load_state_dict(state)
    model = model.to(dev)                           # guarantee placement on the resolved device
    return model, conf


# complex: a hypergraph's relational complex -> .rcbd / RCDB

def _bundle_node_count(bundle):
    n_nodes = bundle.meta.get("n_nodes")
    if bundle.X is not None and n_nodes is None:
        n_nodes = int(bundle.X.shape[0])
    if (n_nodes is not None and type(n_nodes) is not int
            or bundle.X is not None and n_nodes != bundle.X.shape[0]):
        raise ValueError("HGNN declared node count must match its feature rows")
    return n_nodes


def _complex_from_bundle(bundle):
    from rexgraph.graph import RexGraph
    from rexgraph.relations import Relations
    return RexGraph.from_relations(Relations.from_arrays(
        bundle.extra["he_ptr"], bundle.extra["he_idx"], n_vertices=_bundle_node_count(bundle)))


def save_complex_rex(bundle, path) -> str:
    """Serialize a hypergraph bundle's relational complex as a .rcbd bundle (rexgraph.io)."""
    g = _complex_from_bundle(bundle)
    rio.save_rcbd(str(os.path.expanduser(path)), g)
    return str(path)


def to_rcdb(bundle, uri="memory://", *, name="hypergraph", tags=None):
    """Catalogue a hypergraph's complex in the RCDB (agent.rcdb), stored by its structural
    signature (Betti/coherence) with the blob via rexgraph.io. Requires the agent installed.
    Returns the id."""
    from agent.rcdb import open_store
    g = _complex_from_bundle(bundle)
    store = open_store(uri)
    store.put(name, g, meta={"source": "model-complex"}, tags=tags or ["model-complex"])
    return name
