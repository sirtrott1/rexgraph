"""Edge primal generic source: any weighted edge list -> relational complex -> per edge tensor field
features -> a co participation hypergraph bundle for HGNN. Pandas free; all IO via rexgraph.io. The
original edge complex stays PRIMARY (for tensor fields, the RCDB record, and the future new model
type); the hypergraph is an HGNN specific view where each EDGE is a node."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from numbers import Real

import numpy as np


@dataclass
class EdgeData:
    src_idx: np.ndarray        # int32[nE]  source vertex id (0..n_src-1)
    dst_idx: np.ndarray        # int32[nE]  destination vertex id (n_src..n_src+n_dst-1)
    weight: np.ndarray         # float64[nE]
    n_src: int
    n_dst: int
    col_types: dict = field(default_factory=dict)


def load_edges(path, *, source=None, target=None, weight=None, usecols=None) -> EdgeData:
    """Load a weighted edge list (any tabular schema) as an edge primal dataset. `source`/`target`
    name the two node id columns; `weight` names a numeric weight column; `usecols` optionally
    restricts which columns are read (for wide files with many unrelated columns). When a name is
    omitted, the csv_loader name/position heuristic is used. Dedups (source,target) keeping the
    first row, indexes source column node ids first (0..n_src-1) then target column node ids
    (n_src..n_src+n_dst-1), so an edge runs source node -> destination node. Pandas free."""
    from rexgraph.io.csv_loader import load_edge_csv
    gd = load_edge_csv(path, source=source, target=target, weight=weight, usecols=usecols)
    w = np.asarray(gd.w_E, dtype=np.float64)
    su = np.asarray(gd.sources)      # source column node names
    dv = np.asarray(gd.targets)      # target column node names
    ok = np.isfinite(w)
    su, dv, w = su[ok], dv[ok], w[ok]
    # dedup (source, destination), first occurrence
    seen = {}
    keep = np.zeros(len(su), dtype=bool)
    for i in range(len(su)):
        k = (su[i], dv[i])
        if k not in seen:
            seen[k] = i; keep[i] = True
    su, dv, w = su[keep], dv[keep], w[keep]
    src_names = sorted(set(su.tolist()))
    dst_names = sorted(set(dv.tolist()))
    smap = {s: i for i, s in enumerate(src_names)}
    dmap = {d: i + len(src_names) for i, d in enumerate(dst_names)}
    src_idx = np.array([smap[s] for s in su], dtype=np.int32)
    dst_idx = np.array([dmap[d] for d in dv], dtype=np.int32)
    types = {name: p.role for name, p in gd.profiles.items()}
    return EdgeData(src_idx, dst_idx, w, len(src_names), len(dst_names), types)


def edge_data_from_knowledge(knowledge, *, weight_by: str = "uniform") -> EdgeData:
    """A joined complex as an `EdgeData`, so the warehouse pipeline takes it unchanged.

    `load_edges` reads a two column table and indexes the source column and the target
    column into disjoint ranges, which is what makes an edge run source node ->
    destination node. A joined complex has one entity space, so an entity appearing on
    both sides would otherwise be two nodes; the same range is used for both and the
    entity keeps one identity.

    `weight_by` selects the edge signal the tier split and the labels read:

        uniform   every relation weighs 1
        degree    a relation weighs the joint degree of its endpoints, so the tiers
                  separate densely connected regions from sparse ones
    """
    entities = list(knowledge.entities)
    index = {c: i for i, c in enumerate(entities)}
    src, dst = [], []
    for a, _rel, b, _origin in knowledge.edges:
        if a in index and b in index and a != b:
            src.append(index[a])
            dst.append(index[b])
    if not src:
        raise ValueError("the joined complex has no relation between distinct entities")
    src = np.asarray(src, dtype=np.int32)
    dst = np.asarray(dst, dtype=np.int32)

    if weight_by == "degree":
        degree = np.zeros(len(entities), dtype=np.float64)
        np.add.at(degree, src, 1.0)
        np.add.at(degree, dst, 1.0)
        w = degree[src] + degree[dst]
    elif weight_by == "uniform":
        w = np.ones(src.shape[0], dtype=np.float64)
    else:
        raise ValueError(
            f"unknown weight_by {weight_by!r}. Available: uniform, degree")

    return EdgeData(src, dst, w, len(entities), 0,
                    {"source": "knowledge", "n_sources": len(knowledge.parts)})


def edge_complex(ed: EdgeData):
    """The PRIMARY source destination complex: one edge per record (source node -> destination node)."""
    from rexgraph.graph import RexGraph
    from rexgraph.relations import Relations
    if (type(ed.n_src) is not int or type(ed.n_dst) is not int or ed.n_src < 0 or ed.n_dst < 0
            or np.asarray(ed.src_idx).ndim != 1 or np.asarray(ed.dst_idx).shape != np.asarray(ed.src_idx).shape):
        raise ValueError("warehouse edge data requires aligned indices and declared vertex counts")
    support = np.column_stack((ed.src_idx, ed.dst_idx)).reshape(-1)
    relations = Relations.from_arrays(np.arange(0, len(support)+1, 2, dtype=np.int64), support,
                                     n_vertices=ed.n_src+ed.n_dst, weights=ed.weight)
    return RexGraph.from_relations(relations)


def tier_split(ed: EdgeData, n_tiers: int = 3):
    """Partition source nodes into tiers by mean incident edge weight; an edge belongs to its source
    node's tier. Returns a list of edge index arrays."""
    tmean = np.zeros(ed.n_src, dtype=np.float64)
    cnt = np.zeros(ed.n_src, dtype=np.float64)
    np.add.at(tmean, ed.src_idx, ed.weight)
    np.add.at(cnt, ed.src_idx, 1.0)
    tmean = tmean / np.maximum(cnt, 1.0)
    qs = np.percentile(tmean, np.linspace(0, 100, n_tiers + 1)[1:-1]) if n_tiers > 1 else np.array([])
    tier_of_src = np.digitize(tmean, qs)               # 0..n_tiers-1
    tier_of_edge = tier_of_src[ed.src_idx]
    return [np.where(tier_of_edge == k)[0] for k in range(n_tiers)]


def labels(ed: EdgeData, mask: np.ndarray) -> np.ndarray:
    mask = _tier_indices(mask, len(ed.src_idx))
    pk = _numeric_signal(ed.weight, len(ed.src_idx))[mask]
    if not len(pk):
        return np.empty(0, dtype=np.int64)
    return (pk >= np.median(pk)).astype(np.int64)


def _numeric_signal(flow, n_edges):
    """Approximate ML signal; this never replaces the exact primary relation state."""
    if np.ma.isMaskedArray(flow) and np.any(np.ma.getmaskarray(flow)):
        raise ValueError("warehouse signal cannot contain masked values")
    raw = np.asarray(flow)
    if raw.shape != (n_edges,) or raw.dtype.kind not in "iufO":
        raise ValueError("warehouse signal requires one real numeric value per edge")
    if raw.dtype.kind == "O" and any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw):
        raise ValueError("warehouse signal requires explicit real numeric values without absence")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("warehouse signal requires finite binary64 values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("warehouse signal requires finite binary64 values")
    return values


def _tier_indices(mask, n_edges):
    if np.ma.isMaskedArray(mask) and np.any(np.ma.getmaskarray(mask)):
        raise ValueError("warehouse tier indices cannot contain masked addresses")
    indices = np.asarray(mask)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu"
            or indices.size and (int(indices[0]) < 0 or int(indices[-1]) >= n_edges
                                 or np.any(indices[1:] <= indices[:-1]))):
        raise ValueError("warehouse tier indices must be unique, ascending and within the source")
    return indices.astype(np.intp, copy=False)


def _hodge_amplitudes(rex, flow):
    """Absolute gradient/curl/harmonic amplitudes, using Core's declared Hodge split."""
    signal = _numeric_signal(flow, rex.nE)
    parts = tuple(rex.hodge(signal))
    if len(parts) != 3:
        raise ValueError("warehouse Hodge split requires exactly three edge signals")
    return tuple(np.abs(_numeric_signal(part, rex.nE)) for part in parts)


def _diffused(rex, flow, t_scales):
    """Down sector heat at each scale, plus full in grade Dirac squared heat.

    Heat does not cross grades. The Dirac seed covers every declared grade;
    an edge seed's returned feature is its grade one slice.
    """
    from rexgraph.native_sparse import as_native
    import rexgraph.scale_propagator as spg
    signal = _numeric_signal(flow, rex.nE)
    try:
        scales = tuple(t_scales)
    except TypeError as exc:
        raise ValueError("warehouse heat scales require a nonempty sequence") from exc
    if (not scales or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, Real) for t in scales)):
        raise ValueError("warehouse heat scales require a nonempty real numeric sequence")
    try:
        times = np.asarray(scales, dtype=np.float64)
    except (ValueError, OverflowError) as exc:
        raise ValueError("warehouse heat scales must be finite and nonnegative") from exc
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("warehouse heat scales must be finite and nonnegative")
    B1 = as_native(rex.B1_sparse)
    L1_down = B1.T.product(B1)
    # Core shares the Chebyshev vectors across all scales rather than repeating
    # the sparse polynomial walk independently for every feature column.
    # The current Core heat API accepts SciPy carriers. Use its explicit sparse
    # bridge after native product construction; no dense array or local solver.
    down = np.asarray(spg.heat_trajectory(L1_down.as_scipy(), signal, times), dtype=np.float64)
    if down.shape != (len(times), rex.nE) or not np.all(np.isfinite(down)):
        raise ValueError("warehouse down-sector heat returned an invalid signal")
    propagator = rex.sparse_dirac()
    if propagator.sizes[1] != rex.nE:
        raise ValueError("warehouse Dirac grade-one basis differs from the primary edges")
    edge_slice = propagator.grade_slice(1)
    psi0 = np.zeros(propagator.N, dtype=np.float64)
    psi0[edge_slice] = signal
    heat = np.asarray(propagator.heat_squared(psi0, float(times.max())), dtype=np.float64)
    if heat.shape != (propagator.N,) or not np.all(np.isfinite(heat)):
        raise ValueError("warehouse Dirac heat returned an invalid graded signal")
    names = [f"heat_diffus_t{t}" for t in times] + ["dirac_diffus"]
    return np.column_stack((down.T, heat[edge_slice])), names


#: the relational Laplacian's channels, in their canonical order
CHANNELS = ("L1_down", "L_O", "L_SG", "L_C")


def _chi_canonical(rex):
    """Per edge character in a FIXED four column layout.

    `nhats` is adaptive: a channel that is identically zero for a complex is not
    carried, so two disjoint edges report two channels and a complex with shared
    vertices reports four. That is right for reading one complex and wrong for
    learning across many, where column 2 has to mean the same thing in every row of
    a batch. Each active channel is placed in its own slot and an inactive one is the
    zero it already is.
    """
    chi = np.asarray(rex.structural_character, dtype=np.float64)
    names = list(getattr(rex, "hat_names", []) or [])
    out = np.zeros((chi.shape[0], len(CHANNELS)), dtype=np.float64)
    for j, name in enumerate(names[:chi.shape[1]]):
        if name in CHANNELS:
            out[:, CHANNELS.index(name)] = chi[:, j]
    return out


def edge_features(rex, ed, mask: np.ndarray, t_scales=(0.5, 2.0)):
    """Per edge tensor field feature matrix for the edges in `mask`, with channel names. The
    complex is PRIMARY; each edge reads its slice of the tensor fields, absolute Hodge amplitudes, and the
    diffused edge weight signal.

    `ed` is an `EdgeData` or the edge signal itself. Any complex has a signal on its
    edges, and only the signal is used here, so a caller with a complex and no
    EdgeData (an ontology, a joined knowledge complex) reads the same features rather
    than a second implementation of them.
    """
    mask = _tier_indices(mask, rex.nE)
    flow = _numeric_signal(getattr(ed, "weight", ed), rex.nE)
    chi = _chi_canonical(rex)                                          # (nE, 4), fixed slots
    curv = np.asarray(rex.rcfe_curvature, dtype=np.float64).reshape(-1, 1)   # (nE, 1)
    g, c, h = _hodge_amplitudes(rex, flow)
    hodge = np.stack([g, c, h], axis=1)                                 # (nE, 3)
    diff, dnames = _diffused(rex, flow, t_scales)                       # (nE, k)
    feats = np.concatenate([chi, curv, hodge, diff], axis=1)           # (nE, F)
    char_names = [f"char_{n}" for n in CHANNELS]
    names = (char_names + ["rcfe_curv", "hodge_grad_abs", "hodge_curl_abs", "hodge_harm_abs"] + dnames)
    if not np.all(np.isfinite(feats)):
        raise ValueError("warehouse features contain nonfinite values")
    try:
        with np.errstate(over="raise", invalid="raise"):
            X = feats[mask].astype(np.float32)
    except FloatingPointError as exc:
        raise ValueError("warehouse features exceed finite float32 training values") from exc
    return X, names


def hypergraph_bundle(ed: EdgeData, mask: np.ndarray, X, y, *, n_classes=2):
    """Co participation hypergraph over the edges in `mask`: each edge is a NODE; a hyperedge
    groups edges that share a declared vertex. Bipartite source/target domains are already
    disjoint in EdgeData; a knowledge entity has one identity on both sides. Feature and label
    rows follow the unique ascending tier indices. `n_classes` declares the output domain even
    when some classes are unobserved. The original complex remains primary elsewhere."""
    mask = _tier_indices(mask, len(ed.src_idx))
    if type(n_classes) is not int or not 0 < n_classes < 2**31:
        raise ValueError("warehouse class count must be a positive native integer")
    if any(np.ma.isMaskedArray(a) and np.any(np.ma.getmaskarray(a)) for a in (X, y)):
        raise ValueError("warehouse training rows cannot contain masked values")
    features, targets = np.asarray(X), np.asarray(y)
    if features.ndim != 2 or features.shape[0] != len(mask) or features.dtype.kind not in "iufO":
        raise ValueError("warehouse features require one real numeric row per tier edge")
    if features.dtype.kind == "O":
        features = _numeric_signal(features.reshape(-1), features.size).reshape(features.shape)
    try:
        with np.errstate(over="raise", invalid="raise"):
            features = features.astype(np.float32)
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError("warehouse features require finite float32 training values") from exc
    if not np.all(np.isfinite(features)):
        raise ValueError("warehouse features require finite float32 training values")
    if (targets.shape != (len(mask),) or targets.dtype.kind not in "iu"
            or targets.size and (int(targets.min()) < 0 or int(targets.max()) >= n_classes)):
        raise ValueError("warehouse labels require one integer class in the declared domain per tier edge")
    from ..models.data import DataBundle
    groups = defaultdict(list)
    for local, b in enumerate(mask):
        src, dst = int(ed.src_idx[b]), int(ed.dst_idx[b])
        groups[src].append(local)
        if dst != src:
            groups[dst].append(local)
    he = [nodes for nodes in groups.values() if len(nodes) >= 2]        # non trivial hyperedges only
    he_ptr = np.zeros(len(he) + 1, dtype=np.int32)
    idx = []
    for i, nodes in enumerate(he):
        he_ptr[i + 1] = he_ptr[i] + len(nodes)
        idx.extend(nodes)
    he_idx = np.asarray(idx, dtype=np.int32)
    import torch
    b = DataBundle("hypergraph",
                   torch.as_tensor(features),
                   torch.as_tensor(targets.astype(np.int64)),
                   meta={"feat_dim": int(features.shape[1]), "n_classes": n_classes, "n_nodes": len(mask)})
    b.extra = {"he_ptr": he_ptr, "he_idx": he_idx}
    return b


def knowledge_bundle(knowledge, *, weight_by: str = "degree", target: str = "relation",
                     t_scales=(0.5, 2.0)):
    """A joined complex as a `DataBundle` the model factory can train on.

    The chain a knowledge complex takes to a model: entities and relations become an
    `EdgeData`, the complex reads its own tensor fields for the feature matrix, and the
    co participation hypergraph over the relations is the structure an HGNN consumes.
    Relations are the nodes of that hypergraph, which is the edge primal view: a
    hyperedge groups the relations sharing an endpoint.

    `target` chooses what is learned:

        relation   the relation's own type, so the model learns to tell an `is_a` from
                   an annotation from a genomic overlap out of structure alone
        weight     above or below the median edge signal, the warehouse's default

    Returns the bundle with `meta['classes']` naming the target values.
    """
    ed = edge_data_from_knowledge(knowledge, weight_by=weight_by)
    rex = edge_complex(ed)
    mask = np.arange(ed.src_idx.shape[0], dtype=np.int64)
    X, names = edge_features(rex, ed, mask, t_scales=t_scales)

    if target == "relation":
        construction = knowledge.edge_construction()
        y = np.asarray(construction.type_labels, dtype=np.int64)
        classes = list(construction.type_names)
    elif target == "weight":
        y = labels(ed, mask)
        classes = ["below_median", "at_or_above_median"]
    else:
        raise ValueError(f"unknown target {target!r}. Available: relation, weight")

    bundle = hypergraph_bundle(ed, mask, X, y, n_classes=len(classes))
    bundle.meta.update({
        "feature_names": names, "classes": classes,
        "n_entities": knowledge.nV, "n_relations": knowledge.nE,
    })
    from ..models.data import make_splits
    bundle.splits = make_splits(int(X.shape[0]))
    return bundle
