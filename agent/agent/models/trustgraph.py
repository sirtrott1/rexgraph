"""
trustgraph: ingestion bridge from TrustGraph knowledge cores to rexgraph.

TrustGraph structures database inputs into knowledge cores (entities and defined relations). The
agent's TrustGraphAdapter represents each core as a relational complex. This module converts that
into the forms the rest of the stack consumes:

    core_to_rex(...)        knowledge core -> RexGraph complex (standalone triples or a live flow)
    bundle_from_core(...)   knowledge core -> trainable DataBundle (entities=nodes, relations=
                            structure) for the model archetypes (hgnn/etc.)
    store.to_rcdb / save_complex_rex   persist or catalogue the complex by topology

The path is DB -> TrustGraph core -> complex -> train (models) / store (RCDB) / persist
(rexgraph.io). Requires the agent installed; a live flow also needs a TrustGraph endpoint.
"""
from __future__ import annotations

import numpy as np

from . import data as D
from . import store as _store


def core_to_rex(triples=None, *, url: str | None = None, flow: str | None = None):
    """Convert a knowledge core into (RexGraph, meta). `triples` is a list of (s,p,o) tuples or
    SimpleTriple (standalone); or pass `url`+`flow` to pull a live TrustGraph flow."""
    from agent.integrations.trustgraph_adapter import SimpleTriple, TrustGraphAdapter
    adapter = TrustGraphAdapter(url=url) if url else TrustGraphAdapter()
    if flow:
        return adapter.from_flow(flow)
    tris = [(t if not isinstance(t, (tuple, list)) else SimpleTriple(*t)) for t in (triples or [])]
    return adapter.from_triples(tris)


def bundle_from_core(triples=None, *, url=None, flow=None, labels=None, feat_dim=16,
                     seed=0) -> D.DataBundle:
    """Convert a knowledge core into a hypergraph DataBundle for the archetypes. Entities are nodes;
    the core's relations are the (signed) complex structure. `labels` is an optional
    {entity_name: class} map for node classification; without it the bundle is unlabeled (structure
    only, for link-pred / unsupervised). Node features are placeholder random projections; replace
    them with real entity embeddings."""
    rex, meta = core_to_rex(triples, url=url, flow=flow)
    b = _store._bundle_from_rex(rex)                     # he_ptr/he_idx from the complex
    names = list(meta.get("vertex_labels") or [str(i) for i in range(b.meta["n_nodes"])])
    n = b.meta["n_nodes"]
    rng = np.random.default_rng(seed)
    if labels:
        classes = sorted(set(labels.values()))
        cmap = {c: i for i, c in enumerate(classes)}
        y = np.array([cmap.get(labels.get(nm), 0) for nm in names], "int64")
        n_classes = len(classes)
        X = rng.normal(0, 1.0, (n, feat_dim)).astype("float32")
        X += 0.4 * (np.eye(n_classes)[y] @ rng.normal(0, 1, (n_classes, feat_dim))).astype("float32")
    else:
        y = np.zeros(n, "int64"); n_classes = 1
        X = rng.normal(0, 1.0, (n, feat_dim)).astype("float32")
    b.X, b.y = D._as(X), D._as(y)
    b.meta.update({"feat_dim": feat_dim, "n_classes": n_classes, "entity_names": names})
    b.splits = D.make_splits(n, seed=seed)
    return b


def core_to_rcdb(triples=None, *, url=None, flow=None, uri="memory://", name="knowledge_core",
                 tags=None, store=None):
    """Ingest a knowledge core and catalogue its complex in the RCDB (queryable by Betti/coherence).

    `store` takes an already-opened store and `uri` is then ignored. A route passes the
    workspace-scoped store that way, because opening a caller-named URI here writes
    outside the scoped view entirely.
    """
    rex, _ = core_to_rex(triples, url=url, flow=flow)
    if store is None:
        from agent.rcdb import open_store
        store = open_store(uri)
    store.put(name, rex, meta={"source": "trustgraph"}, tags=tags or ["trustgraph"])
    return name


def core_to_rex_file(triples=None, *, url=None, flow=None, path="core.rex"):
    """Ingest a knowledge core and persist its complex as a .rex bundle (rexgraph.io)."""
    import rexgraph.io as rio
    rex, _ = core_to_rex(triples, url=url, flow=flow)
    rio.save_rex(str(path), rex)
    return str(path)
