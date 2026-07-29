"""rexgraph.flow.navigator: the FieldNavigator, a lazy loop over a changing
relational complex (TemporalRex).

The navigator walks a TemporalRex stream one snapshot at a time and stays
idle by default: each step it asks the MalaughGate whether the change in
H_T at this step is a surprise. Only when the gate fires does the navigator
do any further work, and even then the work is localized: it diffs the
current snapshot's edges against the previous snapshot's edges to find the
region that actually changed, then hands that region (not the whole
complex) to a flow step.

Edge source for changed_edges: it diffs canonical cell keys computed off
the boundary CSR (rexgraph.core._temporal.cell_keys_of), the same arrays
TemporalRex.at() builds each snapshot's RexGraph from. That keeps
changed_edges agnostic to how a RexGraph was constructed (snapshot tuple,
boundary arrays, or otherwise) rather than reaching into TemporalRex's
private _snapshots, and works for any cell arity (simple edges, witness
edges, branching hyperedges).

flow_step resolves a unit seed on the localized region into the real flow
response of the complex: a Hodge decomposition of the seed splits it into a
draining part (gradient, along the tree-like acyclic direction) and a
circulating part (curl plus harmonic, along actual cycles), and the signed
boundary B1 carries the seed across grades to the vertices it touches.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from rexgraph.core._hodge import _hodge_sparse
from rexgraph.core._sparse import to_scipy_csr
from rexgraph.flow.gate import MalaughGate

__all__ = ["FieldNavigator", "flow_step", "changed_edges", "EdgeChange", "removed_region_for"]


EdgeChange = namedtuple("EdgeChange", "added removed")
"""`added`: int64 indices into curr's edges whose canonical key is not in
prev. `removed`: int64 canonical keys that were in prev but are absent from
curr (a removed cell has no curr index to point at, so it is reported by
key, not by position)."""


def changed_edges(prev_rex, curr_rex) -> EdgeChange:
    """Diff prev_rex and curr_rex by canonical cell key, both directions.

    `added` is indices into curr_rex's edges whose key is not present in
    prev (today's additions semantics, preserved). `removed` is prev's keys
    that are no longer present in curr.

    Compares canonical cell keys (rexgraph.core._temporal.cell_keys_of), not
    raw counts or positions: an edge that shifted index but kept its
    identity is not "changed", and a cell whose key is new (even if some
    other cell kept that index) is. cell_keys_of hashes the boundary CSR
    column directly, so this works for ANY arity: ordered pair encoding for
    simple 2-arity edges, an order-independent hash of the boundary vertices
    for witness (arity 1) and branching (arity > 2) hyperedges alike. There
    is no longer a 2-arity precondition.

    Edge identity is directedness aware, using each rex's own
    `_directed` flag, the same scheme the delta store keys edges with, so
    `added`/`removed` line up with the delta encoders. In an undirected
    complex, reversing an edge's source and target is the same undirected
    edge, so that reversal alone is not reported as a change. In a directed
    complex, a reversed edge is a different key, so it is reported as one
    removal plus one addition.
    """
    from rexgraph.core._temporal import cell_keys_of

    prev_rex._ensure_clean()
    curr_rex._ensure_clean()
    prev_keys = cell_keys_of(prev_rex._boundary_ptr, prev_rex._boundary_idx, prev_rex._directed)
    curr_keys = cell_keys_of(curr_rex._boundary_ptr, curr_rex._boundary_idx, curr_rex._directed)

    prev_key_set = set(prev_keys.tolist())
    curr_key_set = set(curr_keys.tolist())

    added = np.asarray(
        [j for j in range(curr_keys.shape[0]) if int(curr_keys[j]) not in prev_key_set],
        dtype=np.int64,
    )
    removed = np.asarray(
        [k for k in prev_keys.tolist() if int(k) not in curr_key_set],
        dtype=np.int64,
    )
    return EdgeChange(added=added, removed=removed)


def flow_step(rex, region: NDArray) -> Dict[str, object]:
    """Resolve a unit flow seeded on `region` into the complex's real response.

    A unit seed is placed on the region's edges and run through the Hodge
    decomposition of the (signed) boundary structure: the gradient part is
    the draining component (flow that terminates, tree-like), and the curl
    plus harmonic parts are the circulating component (flow that follows
    actual cycles, closed or trapped). Separately, the signed vertex
    boundary B1 carries the seed across grades to the vertices it touches
    (the across-grade response), independent of the in-grade decomposition.

    Calls _hodge_sparse directly instead of the generic hodge_decomposition
    dispatcher, which falls to a dense LAPACK lstsq/SVD solve below its size
    cutoff (every realistic Slice-1 input is below it). Direct call keeps
    the flow path matrix-free at every scale.
    """
    seed = np.zeros(rex.nE)
    seed[region] = 1.0

    # use the chain-filtered B2 (self-loop/chain-violating faces removed) so that
    # B1 B2 = 0 holds and the draining/circulating split stays orthogonal. This is
    # what every other hodge_decomposition call site in the codebase passes.
    b1 = rex._B1_dual
    b2 = getattr(rex, "_B2_hodge_dual", None)

    sp_b1 = to_scipy_csr(b1)
    L0 = sp_b1 @ sp_b1.T
    L2 = None
    if b2 is not None:
        sp_b2 = to_scipy_csr(b2)
        L2 = sp_b2.T @ sp_b2

    gradient, curl, harmonic = _hodge_sparse(b1, b2, seed, L0, L2)
    draining = gradient
    circulating = curl + harmonic

    vertex_response = np.asarray(sp_b1 @ seed).ravel()

    return {
        "draining": draining,
        "circulating": circulating,
        "vertex_response": vertex_response,
        "region": np.asarray(region),
    }


def removed_region_for(prev_rex, curr_rex, removed_keys) -> NDArray:
    """Turn removed canonical keys into a disturbance region on `curr_rex`.

    A removed cell has no index in `curr_rex`, so map each removed key back to
    its endpoint vertices in `prev_rex` (via cell_keys_of), then collect the
    current edges incident to any of those vertices. Keyed by canonical key so
    an index shift never mis-locates the removed cell."""
    from rexgraph.core._temporal import cell_keys_of

    removed_keys = np.asarray(removed_keys, dtype=np.int64)
    if removed_keys.size == 0:
        return np.zeros(0, dtype=np.int64)
    prev_rex._ensure_clean()
    curr_rex._ensure_clean()
    prev_keys = cell_keys_of(prev_rex._boundary_ptr, prev_rex._boundary_idx, prev_rex._directed)
    key_to_col = {int(k): j for j, k in enumerate(prev_keys.tolist())}
    pptr = np.asarray(prev_rex._boundary_ptr)
    pidx = np.asarray(prev_rex._boundary_idx)
    verts = set()
    for k in removed_keys.tolist():
        j = key_to_col.get(int(k))
        if j is None:
            continue
        for v in pidx[pptr[j]:pptr[j + 1]].tolist():
            verts.add(int(v))
    if not verts:
        return np.zeros(0, dtype=np.int64)
    cptr = np.asarray(curr_rex._boundary_ptr)
    cidx = np.asarray(curr_rex._boundary_idx)
    region = [j for j in range(curr_rex.nE)
              if any(int(v) in verts for v in cidx[cptr[j]:cptr[j + 1]].tolist())]
    return np.asarray(sorted(set(region)), dtype=np.int64)


class FieldNavigator:
    """Lazy loop over a TemporalRex stream: idle unless the gate fires.

    Each step observes the gate on the current snapshot. If the gate does
    not fire, the step is recorded as idle and no flow work happens at all.
    If the gate fires, the navigator localizes to the changed edges (via
    changed_edges against the previous snapshot) and runs flow_step only
    over that region, counting the call in flow_calls.

    Single-use instance: a FieldNavigator (and the MalaughGate it wraps)
    carries gate state (_prev, _hist) across calls to run(). It is meant
    to be run once per stream; reusing one instance across separate
    streams blends the baseline from the first stream into the second
    instead of starting fresh.
    """

    def __init__(self, gate: Optional[MalaughGate] = None):
        self.gate = gate if gate is not None else MalaughGate()
        self.flow_calls = 0

    def step(self, rex, change=None, removed_region=None) -> Dict[str, object]:
        """Advance the field ONE snapshot. `change` is an EdgeChange(added, removed);
        None means all-added (first step). `removed_region` is the caller-resolved
        int index array (into `rex`) of edges disturbed by removals (see
        removed_region_for). Returns {event} when idle, else {event, region, flow};
        `run` wraps this with the snapshot index `t`."""
        o = self.gate.observe(rex)
        if not o["event"]:
            return {"event": False}
        if change is None:
            region = np.arange(rex.nE, dtype=np.int64)
        else:
            added = np.asarray(change.added, dtype=np.int64)
            if removed_region is None:
                rr = np.zeros(0, dtype=np.int64)
            else:
                rr = np.asarray(removed_region, dtype=np.int64)
            region = np.unique(np.concatenate([added, rr]).astype(np.int64))
        res = flow_step(rex, region)
        self.flow_calls += 1
        return {"event": True, "region": region, "flow": res}

    def run(self, trex) -> List[Dict[str, object]]:
        """Walk trex snapshot by snapshot, running flow_step only on gate events.

        Delegates each snapshot to step(), which consumes both
        changed_edges().added and the removed cells (mapped to their prior
        endpoints via removed_region_for). Idle steps are {t, event: False};
        events are {t, event: True, region, flow}. On growth-only streams
        (no removals), region == added."""
        log: List[Dict[str, object]] = []
        for i in range(trex.T):
            rex_i = trex.at(i)
            if i > 0:
                prev = trex.at(i - 1)
                change = changed_edges(prev, rex_i)
                removed_region = removed_region_for(prev, rex_i, change.removed)
            else:
                change = None
                removed_region = None
            out = self.step(rex_i, change, removed_region)
            log.append({"t": i, **out})
        return log
