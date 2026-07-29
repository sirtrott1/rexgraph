"""rexgraph.mesh_health: topological health of any coordination graph.

A distributed system (microservices, a job DAG, a settlement network, a hive of
model workers) is a relational complex: components are vertices, calls/messages
are directed edges, live traffic is an edge flow. The Hodge decomposition of that
flow splits it into two physically meaningful parts:

    gradient  = load that DRAINS:      flows downhill and clears (healthy)
    curl+harmonic = load that CIRCULATES: trapped in a loop, building
                    (a retry storm, a circular dependency, a distributed deadlock)

`mesh_health(edges, flow)` returns that split plus the loops the stuck load lives
on and the structural bottlenecks: a JSON-friendly report an adapter can emit as
SLIs on top of existing telemetry (OpenTelemetry spans, a service map, or the
live agent complex). Unlike DFS cycle detection it is load-weighted (ranks
severity, ignores benign cycles), localizing, and an early signal: the
circulating fraction rises before absolute traffic saturates.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .graph import RexGraph

__all__ = ["mesh_health", "harmonic_health"]

# a machine-precision "is this quantity nonzero" test: reads the SUPPORT of the harmonic
# field (which edges actually carry circulation), not as a tunable policy threshold.
_ZERO = 1e-9


def harmonic_health(rex, flow=None) -> dict:
    """The exact structural character of a complex's circulation, eigen-free.

    The harmonic (oscillatory) part of ``flow`` is the circulation no potential can
    explain; this decomposes it, via the structural character, into the topological
    FRUSTRATION channel and the geometric COPARTICIPATION channel:

        health_ratio = frustration_total / coparticipation_total

    ``> 1`` means the circulation is irreducible topological tension (a genuine
    deadlock no face can fill); ``< 1`` means it is geometric overlap that a
    co-participation could close. All quantities are exact-structural: dim_H is the
    integer beta_1, the per-edge harmonic magnitude is the support of the stuck
    loops, and the channel split comes from the character. (The same computation the
    AnalysisPipeline's hodge stage runs, promoted to a reusable call.)
    """
    out = {"dim_H": int(rex.betti[1]), "harm_per_edge": np.zeros(int(rex.nE)),
           "frustration_per_edge": None, "coparticipation_per_edge": None,
           "frustration_total": 0.0, "coparticipation_total": 0.0, "health_ratio": None}
    if out["dim_H"] == 0:
        return out
    if flow is None:
        flow = np.ones(int(rex.nE), dtype=np.float64)
    flow = np.asarray(flow, dtype=np.float64)
    try:
        from . import harmonic_sparse as _hsp
        H = _hsp.harmonic_basis(rex)
        harm = np.asarray(_hsp.harmonic_projection(H, flow), dtype=np.float64)
    except Exception:
        return out
    out["harm_per_edge"] = np.abs(harm)
    try:
        if int(getattr(rex, "nhats", 0)) >= 3:
            chi = (np.asarray(rex.structural_character)
                   * np.asarray(rex._rl4_sparse.diagonal())[:, None])
            frustration = np.abs(harm) * chi[:, 0]              # topological channel
            coparticipation = np.abs(harm) * chi[:, 1]          # geometric/overlap channel
            fsum, csum = float(frustration.sum()), float(coparticipation.sum())
            out["frustration_per_edge"] = frustration
            out["coparticipation_per_edge"] = coparticipation
            out["frustration_total"] = round(fsum, 6)
            out["coparticipation_total"] = round(csum, 6)
            out["health_ratio"] = round(fsum / csum, 6) if csum > _ZERO else None
    except Exception:
        pass
    return out


def _normalize(edges, flow):
    """Map arbitrary node labels to ids, drop self-loops, aggregate duplicate
    directed edges (summing flow). Returns (labels, src, tgt, w, id_of)."""
    ids: dict = {}

    def nid(x):
        if x not in ids:
            ids[x] = len(ids)
        return ids[x]

    agg: dict = {}
    order = []
    edges = list(edges)
    if flow is None:
        flow = np.ones(len(edges), dtype=np.float64)
    flow = np.asarray(flow, dtype=np.float64).ravel()
    if flow.shape[0] != len(edges):
        raise ValueError(f"flow has {flow.shape[0]} entries but there are {len(edges)} edges")
    for (a, b), f in zip(edges, flow):
        if a == b:
            continue                                   # self-loop carries no coordination
        key = (nid(a), nid(b))
        if key not in agg:
            agg[key] = 0.0
            order.append(key)
        agg[key] += float(f)
    labels = [None] * len(ids)
    for lbl, i in ids.items():
        labels[i] = lbl
    src = np.array([s for s, _ in order], dtype=np.int32)
    tgt = np.array([t for _, t in order], dtype=np.int32)
    w = np.array([agg[k] for k in order], dtype=np.float64)
    return labels, src, tgt, w, ids


def _align_to_graph(rex, src, tgt, w):
    """Reorder the flow to the graph's stored edge orientation, flipping sign where the
    stored edge runs opposite to ours. A no-op when from_graph preserves input order."""
    gs, gt = getattr(rex, "sources", None), getattr(rex, "targets", None)
    if gs is None or gt is None:
        return w
    want = {}
    for (s, t, val) in zip(src.tolist(), tgt.tolist(), w.tolist()):
        want[(s, t)] = val
    out = np.zeros(len(gs), dtype=np.float64)
    for e, (s, t) in enumerate(zip(gs.tolist(), gt.tolist())):
        if (s, t) in want:
            out[e] = want[(s, t)]
        elif (t, s) in want:
            out[e] = -want[(t, s)]                      # stored opposite: flip the sign
    return out


def _void_groups(rex, labels, gs, gt):
    """Node groups the structure implies but does not yet recognize (the void complex): a set of
    mutually related nodes with no coparticipation face: a structural completion candidate (a
    missing junction / normalization hint / an implied coordination group). `closes_a_hole` says
    whether recognizing it removes a harmonic cycle (fills_beta); `affinity` is the void's eta."""
    try:
        vc = rex.void_complex
    except Exception:
        return []
    tri = vc.get("tri_edges")
    if tri is None:
        return []
    tri = np.asarray(tri)
    if tri.ndim != 2 or tri.shape[0] == 0:
        return []
    fills = np.asarray(vc.get("fills_beta")) if vc.get("fills_beta") is not None else None
    eta = np.asarray(vc.get("eta")) if vc.get("eta") is not None else None
    ne = len(gs)
    out = []
    for k in range(tri.shape[0]):
        eids = [int(e) for e in tri[k] if 0 <= int(e) < ne]
        nodes = sorted({int(gs[e]) for e in eids} | {int(gt[e]) for e in eids})
        out.append({
            "services": [labels[i] for i in nodes],
            "closes_a_hole": bool(fills[k]) if fills is not None and k < len(fills) else None,
            "affinity": round(float(eta[k]), 4) if eta is not None and k < len(eta) else None,
        })
    out.sort(key=lambda d: -(d["affinity"] or 0.0))
    return out


def _components(nodes, adj):
    """Union-find connected components over a set of nodes and an adjacency dict."""
    parent = {n: n for n in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, nbrs in adj.items():
        for b in nbrs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)
    return list(groups.values())


def mesh_health(edges: Iterable[Tuple], flow: Optional[Sequence[float]] = None) -> dict:
    """Topological health of a coordination graph.

    Parameters
    ----------
    edges : iterable of (source, target)
        Directed calls/messages. Node labels may be any hashable value.
    flow : sequence of float, optional
        Per-edge load (request rate, message count, in-flight count). Aligned to
        ``edges``. Defaults to uniform 1.0 (pure structure).

    Returns
    -------
    dict with: n_nodes, n_edges, n_cycles (beta_1), draining, circulating,
    status, stuck_loops (each: services, circulating, edges), bottlenecks
    (each: node, criticality).

    ``status`` is structural, not a tuned band: ``acyclic`` when beta_1 == 0 (no
    cycle can trap load), ``draining`` when cycles exist but the harmonic field
    vanishes on this flow, ``circulating`` when the harmonic field is nonzero. The
    ``circulating`` fraction is the reported magnitude; the caller applies its own
    policy to it. The only tolerance used is a machine-precision numerical zero
    (is the harmonic component nonzero here), not a policy threshold.
    """
    labels, src, tgt, w, _ = _normalize(edges, flow)
    n_edges = int(src.shape[0])
    if n_edges == 0:
        return {"n_nodes": len(labels), "n_edges": 0, "n_cycles": 0,
                "draining": 1.0, "circulating": 0.0, "status": "acyclic",
                "stuck_loops": [], "bottlenecks": []}

    rex = RexGraph.from_graph(src, tgt)
    flow_g = _align_to_graph(rex, src, tgt, w)
    gs = getattr(rex, "sources", None)
    gt = getattr(rex, "targets", None)
    if gs is None or gt is None:                        # fall back to our own order
        gs, gt = src, tgt

    n_cycles = int(rex.betti[1])                        # exact integer invariant
    grad, curl, harm = rex.hodge(flow_g)
    circ_edge = curl + harm                             # the non-gradient (circulating) part
    fn = float(np.linalg.norm(flow_g))
    circulating = float(np.linalg.norm(circ_edge) / fn) if fn > 0 else 0.0
    draining = max(0.0, 1.0 - circulating)
    # status from structure: no cycles -> nothing can circulate; cycles but the harmonic field is
    # (numerically) zero -> draining; a nonzero harmonic field -> circulating. _ZERO is a
    # machine-precision "is it nonzero" test, not a tuned severity band.
    if n_cycles == 0:
        status = "acyclic"
    elif circulating <= _ZERO:
        status = "draining"
    else:
        status = "circulating"

    # the exact character of the circulation: frustration (irreducible topological tension) vs
    # coparticipation (geometric overlap a face could fill). health_ratio > 1 => the stuck load is
    # a genuine structural deadlock; < 1 => it is fillable overlap.
    hh = harmonic_health(rex, flow_g)
    fpe = hh.get("frustration_per_edge")
    cpe = hh.get("coparticipation_per_edge")

    # localize: an edge is part of a stuck loop iff its circulating component is nonzero (in the
    # support of the harmonic field), measured against the peak by the same numerical zero.
    # Only localize when the flow actually circulates: a draining or acyclic flow has no stuck
    # loops by definition. This also keeps the localization from tripping on the tiny per-edge
    # residual an iterative (matrix-free) solver leaves behind while the global circulating
    # fraction is still (correctly) negligible.
    stuck_loops = []
    if status == "circulating":
        mag = np.abs(circ_edge)
        peak = float(mag.max()) if mag.size else 0.0
        hot = [e for e in range(len(mag)) if peak > 0 and mag[e] > _ZERO * peak]
        adj = defaultdict(set)
        hot_nodes = set()
        for e in hot:
            a, b = int(gs[e]), int(gt[e])
            adj[a].add(b); adj[b].add(a)
            hot_nodes.add(a); hot_nodes.add(b)
        for comp in _components(hot_nodes, adj):
            comp_edges = [e for e in hot if int(gs[e]) in comp and int(gt[e]) in comp]
            if len(comp_edges) < len(comp):
                continue                                # a stuck loop must actually be cyclic
            # classify the loop by which character channel dominates its harmonic content (a
            # comparison of exact channel sums, not a tuned cutoff)
            kind = None
            if fpe is not None and cpe is not None:
                f = float(sum(fpe[e] for e in comp_edges))
                c = float(sum(cpe[e] for e in comp_edges))
                kind = "irreducible" if f >= c else "fillable"
            stuck_loops.append({
                "services": [labels[i] for i in comp],
                "circulating": round(float(sum(mag[e] for e in comp_edges)), 4),
                "kind": kind,                           # irreducible tension vs fillable overlap
                "edges": [{"from": labels[int(gs[e])], "to": labels[int(gt[e])],
                           "circulating": round(float(mag[e]), 4)} for e in comp_edges],
            })
        stuck_loops.sort(key=lambda d: -d["circulating"])

    # structural bottlenecks: effective-resistance centrality (a failure here
    # fragments the graph the most)
    bottlenecks = []
    try:
        er = np.asarray(rex._effective_resistance_batch(np.arange(rex.nE)), dtype=np.float64)
        load = np.zeros(rex.nV)
        for e in range(len(er)):
            load[int(gs[e])] += er[e]; load[int(gt[e])] += er[e]
        for i in np.argsort(-load)[:5]:
            if load[i] > 0:
                bottlenecks.append({"node": labels[int(i)], "criticality": round(float(load[i]), 4)})
    except Exception:
        pass

    # per-node structural coherence kappa: how integrated each node is in the complex (a node with
    # low kappa is structurally peripheral / fragile). Sorted least-coherent first.
    coherence = []
    try:
        kap = np.asarray(rex.coherence, dtype=np.float64)
        coherence = sorted(({"node": labels[i], "kappa": round(float(kap[i]), 4)}
                            for i in range(min(len(kap), len(labels)))),
                           key=lambda d: d["kappa"])
    except Exception:
        pass

    # implied structure the graph does not yet recognize (void complex): completion candidates
    implied_groups = _void_groups(rex, labels, gs, gt)

    return {
        "n_nodes": len(labels),
        "n_edges": n_edges,
        "n_cycles": n_cycles,
        "draining": round(draining, 4),
        "circulating": round(circulating, 4),
        "status": status,
        "health_ratio": hh.get("health_ratio"),          # frustration/coparticipation (kind of tension)
        "frustration": hh.get("frustration_total"),
        "coparticipation": hh.get("coparticipation_total"),
        "stuck_loops": stuck_loops,
        "bottlenecks": bottlenecks,
        "coherence": coherence,                          # per-node kappa (structural centrality)
        "implied_groups": implied_groups,                # void-complex completion candidates
    }
