"""Edge-primal generic source: any weighted edge list -> relational complex -> per-edge tensor-field
features -> a co-participation hypergraph bundle for HGNN. Pandas-free; all IO via rexgraph.io. The
original edge complex stays PRIMARY (for tensor fields, the RCDB record, and the future new model
type); the hypergraph is an HGNN-specific view where each EDGE is a node."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

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
    """Load a weighted edge list (any tabular schema) as an edge-primal dataset. `source`/`target`
    name the two node-id columns; `weight` names a numeric weight column; `usecols` optionally
    restricts which columns are read (for wide files with many unrelated columns). When a name is
    omitted, the csv_loader name/position heuristic is used. Dedups (source,target) keeping the
    first row, indexes source-column node ids first (0..n_src-1) then target-column node ids
    (n_src..n_src+n_dst-1), so an edge runs source-node -> destination-node. Pandas-free."""
    from rexgraph.io.csv_loader import load_edge_csv
    gd = load_edge_csv(path, source=source, target=target, weight=weight, usecols=usecols)
    w = np.asarray(gd.w_E, dtype=np.float64)
    su = np.asarray(gd.sources)      # source-column node names
    dv = np.asarray(gd.targets)      # target-column node names
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

    `load_edges` reads a two-column table and indexes the source column and the target
    column into disjoint ranges, which is what makes an edge run source-node ->
    destination-node. A joined complex has one entity space, so an entity appearing on
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
    """The PRIMARY source-destination complex: one edge per record (source node -> destination node)."""
    from rexgraph.graph import RexGraph
    return RexGraph(sources=ed.src_idx.astype(np.int32), targets=ed.dst_idx.astype(np.int32))


def tier_split(ed: EdgeData, n_tiers: int = 3):
    """Partition source nodes into tiers by mean incident edge weight; an edge belongs to its source
    node's tier. Returns a list of edge-index arrays."""
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
    pk = ed.weight[mask]
    return (pk >= np.median(pk)).astype(np.int64)


def _hodge_energies(rex, flow):
    """Per-edge gradient / curl / harmonic ENERGY (abs value) of an edge flow via rex.hodge."""
    grad, curl, harm = None, None, None
    try:
        parts = rex.hodge(np.asarray(flow, dtype=np.float64))
        arrs = [np.asarray(p, dtype=np.float64) for p in parts]
        edge_parts = [a for a in arrs if a.shape[0] == rex.nE]
        # rex.hodge returns (gradient, curl, harmonic) each length nE
        while len(edge_parts) < 3:
            edge_parts.append(np.zeros(rex.nE))
        grad, curl, harm = edge_parts[0], edge_parts[1], edge_parts[2]
    except Exception:
        z = np.zeros(rex.nE)
        grad, curl, harm = z, z, z
    return np.abs(grad), np.abs(curl), np.abs(harm)


def _diffused(rex, flow, t_scales):
    """Signal diffusion in the tensor field: heat_apply on L1 at each t, plus the graded-Dirac heat
    (cross-grade). Returns a (nE, len(t_scales)+1) array of per-edge diffused values, plus names."""
    from rexgraph.core._sparse import to_scipy_csr

    import rexgraph.scale_propagator as spg
    B1 = to_scipy_csr(rex.B1_sparse).astype(np.float64)
    L1 = (B1.T @ B1).tocsr()
    f = np.asarray(flow, dtype=np.float64).reshape(-1, 1)
    cols, names = [], []
    for t in t_scales:
        hv = np.asarray(spg.heat_apply(L1, f, float(t))).reshape(-1)
        cols.append(hv); names.append(f"heat_diffus_t{t}")
    # graded Dirac cross-grade heat on a graded state seeded on the edge grade
    try:
        psi0 = np.zeros(rex.nV + rex.nE + rex.nF, dtype=np.float64)
        psi0[rex.nV:rex.nV + rex.nE] = np.asarray(flow, dtype=np.float64)
        dh = np.asarray(rex.dirac_heat(float(max(t_scales)), psi0))
        cols.append(dh[rex.nV:rex.nV + rex.nE]); names.append("dirac_diffus")
    except Exception:
        cols.append(np.zeros(rex.nE)); names.append("dirac_diffus")
    return np.stack(cols, axis=1), names


#: the relational Laplacian's channels, in their canonical order
CHANNELS = ("L1_down", "L_O", "L_SG", "L_C")


def _chi_canonical(rex):
    """Per-edge character in a FIXED four-column layout.

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
    """Per-edge tensor-field feature matrix for the edges in `mask`, with channel names. The
    complex is PRIMARY; each edge reads its slice of the tensor fields, Hodge energies, and the
    diffused edge weight signal.

    `ed` is an `EdgeData` or the edge signal itself. Any complex has a signal on its
    edges, and only the signal is used here, so a caller with a complex and no
    EdgeData (an ontology, a joined knowledge complex) reads the same features rather
    than a second implementation of them.
    """
    chi = _chi_canonical(rex)                                          # (nE, 4), fixed slots
    curv = np.asarray(rex.rcfe_curvature, dtype=np.float64).reshape(-1, 1)   # (nE, 1)
    flow = getattr(ed, "weight", ed)                     # EdgeData, or the signal itself
    g, c, h = _hodge_energies(rex, flow)
    hodge = np.stack([g, c, h], axis=1)                                 # (nE, 3)
    diff, dnames = _diffused(rex, flow, t_scales)                       # (nE, k)
    feats = np.concatenate([chi, curv, hodge, diff], axis=1)           # (nE, F)
    char_names = [f"char_{n}" for n in CHANNELS]
    names = (char_names + ["rcfe_curv", "hodge_grad_E", "hodge_curl_E", "hodge_harm_E"] + dnames)
    X = feats[mask].astype(np.float32)
    return X, names


def hypergraph_bundle(ed: EdgeData, mask: np.ndarray, X, y):
    """Co-participation hypergraph over the edges in `mask`: each edge is a NODE; a hyperedge
    groups edges that share a source node, and another groups edges that share a destination node.
    This is the HGNN-specific edge-primal view; the original complex remains primary elsewhere."""
    from ..models.data import DataBundle
    local = {int(b): i for i, b in enumerate(mask)}         # edge index -> node id
    groups = defaultdict(list)
    for b in mask:
        groups[("s", int(ed.src_idx[b]))].append(local[int(b)])
        groups[("d", int(ed.dst_idx[b]))].append(local[int(b)])
    he = [nodes for nodes in groups.values() if len(nodes) >= 2]        # non-trivial hyperedges only
    he_ptr = np.zeros(len(he) + 1, dtype=np.int32)
    idx = []
    for i, nodes in enumerate(he):
        he_ptr[i + 1] = he_ptr[i] + len(nodes)
        idx.extend(nodes)
    he_idx = np.asarray(idx, dtype=np.int32)
    import torch
    b = DataBundle("hypergraph",
                   torch.as_tensor(np.asarray(X, np.float32)),
                   torch.as_tensor(np.asarray(y, np.int64)),
                   meta={"feat_dim": int(X.shape[1]), "n_classes": 2, "n_nodes": int(mask.shape[0])})
    b.extra = {"he_ptr": he_ptr, "he_idx": he_idx}
    return b


def knowledge_bundle(knowledge, *, weight_by: str = "degree", target: str = "relation",
                     t_scales=(0.5, 2.0)):
    """A joined complex as a `DataBundle` the model factory can train on.

    The chain a knowledge complex takes to a model: entities and relations become an
    `EdgeData`, the complex reads its own tensor fields for the feature matrix, and the
    co-participation hypergraph over the relations is the structure an HGNN consumes.
    Relations are the nodes of that hypergraph, which is the edge-primal view: a
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

    bundle = hypergraph_bundle(ed, mask, X, y)
    bundle.meta.update({
        "feat_dim": int(X.shape[1]), "n_classes": len(set(y.tolist())),
        "feature_names": names, "classes": classes,
        "n_entities": knowledge.nV, "n_relations": knowledge.nE,
    })
    from ..models.data import make_splits
    bundle.splits = make_splits(int(X.shape[0]))
    return bundle
