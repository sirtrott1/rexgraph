"""
agent.connectors.build: turn a connector's ``(rex, meta)`` into a storable
``RexGraph`` the one canonical way, so every connector (and the validation
harness) agrees on how topology, faces, and labels become a complex.

This mirrors the engine's own construction path (``RexGraph(sources, targets,
B2_col_ptr, B2_row_idx, B2_vals)`` with ``_agent_meta`` attached) - a connector
never has to know the CSC face encoding; it may emit faces as a dense
``B₂ ∈ {-1,0,+1}^{nE×nF}`` and this module converts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def faces_to_csc(b2_dense: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a dense face boundary ``B₂`` (shape ``nE × nF``) to the CSC
    ``(col_ptr, row_idx, vals)`` triple the engine expects (one column per
    face, nonzero rows = the edges bounding it). No validity check here - the
    harness verifies ``∂²=0`` after construction via ``chain_valid``."""
    b2 = np.asarray(b2_dense)
    if b2.ndim != 2:
        raise ValueError(f"B2 must be 2-D (nE × nF); got shape {b2.shape}")
    nE, nF = b2.shape
    col_ptr = [0]
    row_idx: list = []
    vals: list = []
    for f in range(nF):
        rows = np.nonzero(b2[:, f])[0]
        for r in rows:
            row_idx.append(int(r))
            vals.append(float(b2[r, f]))
        col_ptr.append(len(row_idx))
    return (np.asarray(col_ptr, dtype=np.int32),
            np.asarray(row_idx, dtype=np.int32),
            np.asarray(vals, dtype=np.float64))


def to_rexgraph(rex: Any, meta: Dict[str, Any]):
    """Normalise a connector's ``rex`` into a built ``RexGraph`` carrying
    ``_agent_meta`` (so it stores and round-trips).

    ``rex`` may be:
      * an already-built ``RexGraph`` (returned as-is, meta attached), or
      * a ``(sources, targets)`` pair of edge-endpoint arrays.

    If ``meta['faces']`` is a dense ``B₂`` array it is attached as the face
    selection. Vertex labels come from ``meta['vertex_labels']``.
    """
    from rexgraph.graph import RexGraph

    labels = list(meta.get("vertex_labels") or [])
    agent_meta = {
        "vertex_labels": labels,
        "source": meta.get("source"),
        "input_type": meta.get("input_type", "connector"),
    }

    # already a RexGraph?  (duck-typed: has nV / nE)
    if hasattr(rex, "nV") and hasattr(rex, "nE"):
        rex._agent_meta = {**agent_meta, **getattr(rex, "_agent_meta", {})}
        return rex

    try:
        sources, targets = rex
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "connector rex must be a RexGraph or a (sources, targets) pair"
        ) from exc

    src = np.asarray(sources, dtype=np.int32)
    tgt = np.asarray(targets, dtype=np.int32)

    faces = meta.get("faces")
    if faces is not None:
        cp, rp, vp = faces_to_csc(np.asarray(faces))
        g = RexGraph(sources=src, targets=tgt,
                     B2_col_ptr=cp, B2_row_idx=rp, B2_vals=vp)
    else:
        g = RexGraph(sources=src, targets=tgt)
    g._agent_meta = agent_meta
    return g
