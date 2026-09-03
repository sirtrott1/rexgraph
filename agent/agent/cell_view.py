"""
agent.cell_view: one row per cell, carrying what that cell actually is.

The old dashboard's per-cell table was the useful part of it: a vertex or an edge with
its readings attached, so a question about one cell had one place to look. What it
carried was partly wrong, and the shape assumed things a relational complex does not.

Two changes of substance.

An edge row named a `source` and a `target`. That is the arity-2 coordinate of a
relation, not the relation, so a branching column of arity k had k-2 of its boundary
vertices nowhere in the row. A row here carries its whole BOUNDARY and its arity, so a
4-ary relation reads as one relation over four cells rather than as a pair with
something missing.

Channels are named, not positional. `L1_down` and `L_O` share a diagonal on an
unweighted complex, so reading "channel 0" and "channel 1" off a chi row gives two
numbers that are equal for a structural reason and look like a coincidence. Every share
here is keyed by the channel it belongs to.

What is deliberately absent: the Fiedler entry and the partitions derived from it, which
report where a linear cut fell rather than what a cell is, and the standard baselines
(PageRank, betweenness, clustering, community), which are the comparison column and not
a reading of this structure. `analyze(..., standard_metrics=True)` still produces them
where the point IS the comparison.
"""

from __future__ import annotations

import numpy as np

__all__ = ["vertex_rows", "edge_rows", "cells"]


def _channels(rex) -> list:
    return list(getattr(rex, "hat_names", None) or [])


def _named(values, names) -> dict:
    """A character row keyed by channel name, rounded for reading."""
    row = np.asarray(values, dtype=float).ravel()
    if not names or len(names) != row.shape[0]:
        names = [f"channel_{i}" for i in range(row.shape[0])]
    return {str(n): round(float(v), 6) for n, v in zip(names, row, strict=True)}


def _boundaries(rex) -> list:
    """Each relation's boundary vertices, whatever its arity."""
    rex._ensure_clean()
    bp, bi = rex.boundary_ptr, rex.boundary_idx
    if bp is None:
        src, tgt = rex._ensure_src_tgt()
        return [[int(s), int(t)] for s, t in zip(src, tgt, strict=True)]
    bp, bi = np.asarray(bp), np.asarray(bi)
    return [[int(v) for v in bi[bp[e]:bp[e + 1]]] for e in range(int(rex.nE))]


def vertex_rows(rex, *, labels=None, signal=None, limit: int = 0,
                positions: bool = True) -> list:
    """One row per vertex: what it participates in, and how consistently.

    `phi` is the vertex's share of each channel and `at` is that same reading as a
    coordinate, so the table and the picture are the same numbers. `coherence` is the
    exact per-vertex kappa against the global Green's function; `local_coherence` is the
    O(nnz) companion that reads only the incident characters, and the two disagreeing is
    a fact about the vertex rather than an error.
    """
    nV = int(rex.nV)
    names = _channels(rex)
    phi = np.asarray(rex.vertex_character, dtype=float)
    star = np.asarray(rex.star_character, dtype=float)
    kappa = np.asarray(rex.coherence, dtype=float).ravel()
    kappa_local = np.asarray(rex.local_coherence, dtype=float).ravel()
    degree = np.asarray(rex.degree).ravel()
    in_deg = np.asarray(rex.in_degree).ravel()
    out_deg = np.asarray(rex.out_degree).ravel()

    at = None
    if positions:
        from agent.graph_view import character_positions
        at = character_positions(rex, grade="vertex", dim=3)["positions"]

    divergence = None
    if signal is not None:
        sig = np.asarray(signal, dtype=float).ravel()
        if sig.shape[0] == int(rex.nE):
            divergence = np.asarray(rex.B1 @ sig).ravel()

    n = nV if not limit else min(nV, int(limit))
    rows = []
    for v in range(n):
        row = {
            "index": v,
            "label": str(labels[v]) if labels is not None and v < len(labels)
                     else f"v{v}",
            "degree": int(degree[v]) if v < degree.shape[0] else 0,
            "in_degree": int(in_deg[v]) if v < in_deg.shape[0] else 0,
            "out_degree": int(out_deg[v]) if v < out_deg.shape[0] else 0,
            "phi": _named(phi[v], names) if v < phi.shape[0] else {},
            "star_character": _named(star[v], names) if v < star.shape[0] else {},
            "coherence": round(float(kappa[v]), 6) if v < kappa.shape[0] else None,
            "local_coherence": (round(float(kappa_local[v]), 6)
                                if v < kappa_local.shape[0] else None),
        }
        if at is not None and v < at.shape[0]:
            row["at"] = [round(float(x), 6) for x in at[v]]
        if divergence is not None and v < divergence.shape[0]:
            row["divergence"] = round(float(divergence[v]), 6)
        rows.append(row)
    return rows


def edge_rows(rex, *, labels=None, signal=None, limit: int = 0,
              positions: bool = True) -> list:
    """One row per relation, carrying its whole boundary.

    `boundary` is every vertex the relation touches and `arity` is how many, so a
    branching relation is one row rather than a pair with the rest missing. When a
    signal is given, its Hodge parts are split per relation: how much of it is explained
    by a potential, how much circulates through a face, and how much is neither.
    """
    nE = int(rex.nE)
    names = _channels(rex)
    chi = np.asarray(rex.structural_character, dtype=float)
    bounds = _boundaries(rex)
    types = np.asarray(rex.edge_types).ravel() if nE else np.zeros(0)
    weights = (np.asarray(rex.w_E, dtype=float).ravel()
               if getattr(rex, "w_E", None) is not None else None)
    try:
        curvature = np.asarray(rex.rcfe_curvature, dtype=float).ravel()
    except Exception:                            # noqa: BLE001 - no faces, no curvature
        curvature = np.zeros(nE, dtype=float)

    at = None
    if positions:
        from agent.graph_view import character_positions
        at = character_positions(rex, grade="edge", dim=3)["positions"]

    parts = None
    if signal is not None:
        sig = np.asarray(signal, dtype=float).ravel()
        if sig.shape[0] == nE:
            grad, curl, harm = rex.hodge(np.ascontiguousarray(sig))
            parts = (np.asarray(grad).ravel(), np.asarray(curl).ravel(),
                     np.asarray(harm).ravel())

    def _label(v):
        return str(labels[v]) if labels is not None and v < len(labels) else f"v{v}"

    n = nE if not limit else min(nE, int(limit))
    rows = []
    for e in range(n):
        support = bounds[e] if e < len(bounds) else []
        row = {
            "index": e,
            "boundary": [_label(v) for v in support],
            "boundary_index": list(support),
            "arity": len(support),
            "type": int(types[e]) if e < types.shape[0] else 0,
            "chi": _named(chi[e], names) if e < chi.shape[0] else {},
            "curvature": round(float(curvature[e]), 6) if e < curvature.shape[0] else 0.0,
        }
        if weights is not None and e < weights.shape[0]:
            row["weight"] = round(float(weights[e]), 6)
        if at is not None and e < at.shape[0]:
            row["at"] = [round(float(x), 6) for x in at[e]]
        if parts is not None:
            row["hodge"] = {"gradient": round(float(parts[0][e]), 6),
                            "curl": round(float(parts[1][e]), 6),
                            "harmonic": round(float(parts[2][e]), 6)}
        rows.append(row)
    return rows


def cells(rex, *, grade: str = "both", labels=None, signal=None,
          limit: int = 0, positions: bool = True) -> dict:
    """The per-cell view, with the channel names its readings are keyed by."""
    if grade not in ("vertex", "edge", "both"):
        raise ValueError(f"grade must be vertex, edge or both, got {grade!r}")
    out = {
        "channels": _channels(rex),
        "nV": int(rex.nV), "nE": int(rex.nE), "nF": int(rex.nF_hodge),
        "has_branching": bool(getattr(rex, "has_branching", False)),
    }
    if grade in ("vertex", "both"):
        out["vertices"] = vertex_rows(rex, labels=labels, signal=signal,
                                      limit=limit, positions=positions)
    if grade in ("edge", "both"):
        out["relations"] = edge_rows(rex, labels=labels, signal=signal,
                                     limit=limit, positions=positions)
    return out
