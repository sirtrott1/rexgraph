"""
Explore routes: drill into specific RexGraph properties and operations.

These map directly to RexGraph @cached_property accessors and methods,
giving the frontend interactive access to any part of the analysis.
"""

from __future__ import annotations


import numpy as np
from fastapi import APIRouter, HTTPException, Body

router = APIRouter()


#: the one encoder (rexgraph.io._compat). Non-finite floats go out as null:
#: a bare NaN token is not JSON and every browser JSON.parse rejects it.
from rexgraph.io._compat import json_sanitize


def _sanitize(obj):
    return json_sanitize(obj, nan="null")


def _get_rex(session_id: str):
    from agent.server.app import get_store
    store = get_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session not found: {session_id}")
    rex = session.current()
    if rex is None:
        raise HTTPException(400, "No data in session")
    return rex, session


@router.get("/explore/{session_id}/property/{name}")
async def get_property(session_id: str, name: str):
    """Retrieve a specific @cached_property from the current rex.

    Supported names include: betti, euler_characteristic, chain_valid,
    structural_character, vertex_character, coherence, nhats,
    coupling_constants, eigenvalues_L0, per_channel_mixing_times,
    dimension, nV, nE, nF, and many more.
    """
    rex, _ = _get_rex(session_id)

    if name.startswith("_"):
        raise HTTPException(400, "Cannot access private properties")

    if not hasattr(rex, name):
        raise HTTPException(404, f"Property not found: {name}")

    try:
        value = getattr(rex, name)
        # Convert to JSON-safe format
        result = _sanitize({"property": name, "value": value})
        return result
    except Exception as e:
        raise HTTPException(500, f"Error computing {name}: {e}")


@router.get("/explore/{session_id}/explain/{dim}/{idx}")
async def explain_cell(session_id: str, dim: int, idx: int):
    """Full diagnostic for a single cell via rex.explain(dim, idx).

    dim=0 for vertices, dim=1 for edges. Face explanation (dim=2) is
    not currently supported by rexgraph.
    """
    rex, _ = _get_rex(session_id)

    if dim not in (0, 1):
        raise HTTPException(400, f"explain supports dim=0 (vertex) and dim=1 (edge), got dim={dim}")

    max_idx = rex.nV if dim == 0 else rex.nE
    if idx < 0 or idx >= max_idx:
        raise HTTPException(400, f"Index {idx} out of range [0, {max_idx}) for dim={dim}")

    try:
        result = rex.explain(dim, idx)
        return _sanitize(result)
    except Exception as e:
        raise HTTPException(500, f"Error explaining cell ({dim}, {idx}): {e}")


def _edge_endpoints(rex, e: int):
    from rexgraph.core._sparse import to_scipy_csr
    B1c = to_scipy_csr(rex._B1_dual).tocsc()
    return [int(v) for v in B1c.indices[B1c.indptr[e]:B1c.indptr[e + 1]]]


def _find_edge(rex, a: int, b: int):
    """Edge index joining vertices a,b (via a's star), or None."""
    v2e_ptr, v2e_idx = rex._v2e
    v2e_ptr = np.asarray(v2e_ptr); v2e_idx = np.asarray(v2e_idx)
    for j in range(int(v2e_ptr[a]), int(v2e_ptr[a + 1])):
        e = int(v2e_idx[j])
        if set(_edge_endpoints(rex, e)) == {a, b}:
            return e
    return None


def _edge_label(rex, e: int, labels):
    eps = _edge_endpoints(rex, e)
    nm = [labels[v] if v < len(labels) else str(v) for v in eps]
    return " - ".join(nm)


@router.post("/explore/{session_id}/context")
async def local_context(session_id: str, body: dict = Body(...)):
    """The forged contextual picture around query ENTITIES (vertices) and RELATIONS
    (edges) - per-seed diagnostics plus the bounded relevant sub-complex reached by one
    heat diffusion seeded across both grades, so the LLM acts on the relevant structure
    instead of enumerating the whole graph.

    body: {"vertices": ["T cells", 42, ...], "edges": [7, ["T cells","tumor"], ...],
           "t": 1.0, "max_cells": 50}
      vertices : entity labels (matched against vertex_labels) and/or indices.
                 ('seeds' is accepted as an alias.)
      edges    : relation edge indices, and/or [endpoint, endpoint] pairs (labels or
                 indices) resolved to the joining edge.
      t        : heat-diffusion scale (small = tight, larger = wider).
      max_cells: optional cap on the returned neighborhood per grade.

    Returns {seed_vertices:[explain_vertex...], seed_edges:[explain_edge...],
    neighborhood:{vertices, vertex_labels, vertex_coherence, edges, edge_labels,
    edge_character, ...}} - computed only where the signal lands.
    """
    rex, _ = _get_rex(session_id)
    meta = getattr(rex, "_agent_meta", {}) or {}
    labels = list(meta.get("vertex_labels", []) or [])
    lab2idx = {l: i for i, l in enumerate(labels)}

    def _resolve_v(s):
        if isinstance(s, bool):
            return None
        if isinstance(s, int) and 0 <= s < rex.nV:
            return s
        if isinstance(s, str) and s in lab2idx:
            return lab2idx[s]
        return None

    vin = body.get("vertices", body.get("seeds", [])) or []
    vidx = [v for v in (_resolve_v(s) for s in vin) if v is not None]
    eidx = []
    for s in body.get("edges", []) or []:
        if isinstance(s, bool):
            continue
        if isinstance(s, int) and 0 <= s < rex.nE:
            eidx.append(s)
        elif isinstance(s, (list, tuple)) and len(s) == 2:
            a, b = _resolve_v(s[0]), _resolve_v(s[1])
            if a is not None and b is not None:
                e = _find_edge(rex, a, b)
                if e is not None:
                    eidx.append(e)
    if not vidx and not eidx:
        raise HTTPException(400, "provide at least one resolvable 'vertices' (entities) "
                                 "or 'edges' (relations)")

    try:
        t = float(body.get("t", 1.0))
        max_cells = body.get("max_cells")
        max_cells = int(max_cells) if max_cells is not None else None
        ctx = rex.explain_context(vertices=vidx, edges=eidx, t=t, max_cells=max_cells)
        nb = ctx["neighborhood"]
        nb["vertex_labels"] = [labels[i] if i < len(labels) else str(i)
                               for i in nb["vertices"]]
        nb["edge_labels"] = [_edge_label(rex, e, labels) for e in nb["edges"]]
        ctx["seed_vertex_indices"] = vidx
        ctx["seed_edge_indices"] = eidx
        return _sanitize(ctx)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error building context: {e}")


@router.post("/explore/{session_id}/hodge")
async def hodge_decompose(session_id: str, body: dict = Body(...)):
    """Hodge decomposition of a signal on the edge space.

    body: {"signal": [1.0, 0.5, ...]} or {"signal": "uniform"}
    Returns gradient, curl, harmonic components with energy fractions.
    """
    rex, _ = _get_rex(session_id)

    signal_spec = body.get("signal", "uniform")
    if signal_spec == "uniform":
        signal = np.ones(rex.nE, dtype=np.float64)
    elif isinstance(signal_spec, list):
        signal = np.array(signal_spec, dtype=np.float64)
        if len(signal) != rex.nE:
            raise HTTPException(400, f"Signal length {len(signal)} != nE={rex.nE}")
    else:
        raise HTTPException(400, "signal must be 'uniform' or a list of floats")

    try:
        result = rex.hodge_full(signal)
        return _sanitize(result)
    except Exception as e:
        raise HTTPException(500, f"Error in Hodge decomposition: {e}")


@router.post("/explore/{session_id}/interfacing")
async def interfacing_vector(session_id: str, body: dict = Body(...)):
    """Compute the interfacing vector for given targets.

    body: {
        "target_indices": [0, 1, 2],
        "target_weights": [1.0, 1.0, 1.0],
        "signal": "uniform" or [...]
    }
    """
    rex, _ = _get_rex(session_id)

    target_indices = np.array(body["target_indices"], dtype=np.int32)
    target_weights = np.array(body.get("target_weights", np.ones(len(target_indices))))

    signal_spec = body.get("signal", "uniform")
    if signal_spec == "uniform":
        signal = np.ones(rex.nE, dtype=np.float64)
    else:
        signal = np.array(signal_spec, dtype=np.float64)

    try:
        result = rex.interfacing_vector(
            target_indices=target_indices,
            target_weights=target_weights,
            target_signal=signal,
        )
        return _sanitize(result)
    except Exception as e:
        raise HTTPException(500, f"Error computing interfacing vector: {e}")


@router.post("/explore/{session_id}/reconfig")
async def reconfigure(session_id: str, body: dict = Body(...)):
    """Reconfigure the rex with new parameters and create a new snapshot.

    body: {
        "threshold": 0.3,
        "typing": "spectral",
        "face_selection": "typed",
        ...
    }

    Rebuilds the rex from the original data with the new parameters.
    Creates a new timestep in the session's history.
    """
    rex, session = _get_rex(session_id)

    # Get the original data path from the session metadata
    # For now, rebuild from the current rex's metadata
    meta = getattr(rex, "_agent_meta", {})

    return {
        "status": "reconfig",
        "message": "Reconfiguration requires the original data. "
                   "Re-upload with new parameters for now.",
        "current_params": meta,
    }
