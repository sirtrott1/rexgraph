"""
agent.scoring: one ranking, using the reads the library provides for exactly this.

An earlier version of this module called `interfacing_vector` with `target=None`.
That was a misreading. In `_interfacing` the target is a TARGET EDGE VECTOR and the
channel score is a bilinear form I_X = target^T S_X psi between the source's induced
flow and a target pattern; passing None scores psi against itself, which is a self
energy and interfaces with nothing. It also built the whole bundle per document,
paying O(nV . solve) for an answer needed at a handful of vertices.

RexGraph already answers "what does this query touch in this complex" directly, and
demand-driven:

    coherence_response(seed)  kappa at just the query's vertices, by diffusion --
                              O(|seed| . nhats . diffusion), and identical to
                              coherence[seed] rather than an approximation of it
    agentic_reading(seed)     the decision-ready reading the agent layer is meant to
                              consume: the bounded neighborhood, relations ranked by
                              effective resistance (the bridges), entities whose
                              coherence is a low outlier under a data-adaptive Tukey
                              fence, and context_size -- what a correct answer costs

So relevance here is the query's footprint measured by the DOCUMENT's own coherence
field: sum of kappa over the matched vertices. It grows with how much of the query
the document carries and with how coherent that footprint is inside it, and it needs
no mixing constant, because it is one field summed over one seed.

Lexical overlap remains a candidate prefilter only. It decides what to look at.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

#: fewer matched vertices than this and there is no footprint to read: a single
#: vertex has no neighbourhood for the diffusion to describe.
MIN_SHARED = 2


def _zero(n_shared: int = 0) -> dict[str, Any]:
    return {"score": 0.0, "kappa": [], "kappa_mean": 0.0, "context_size": 0,
            "n_load_bearing": 0, "n_frustrated": 0, "n_shared": int(n_shared)}


def shared_indices(doc_labels: Sequence[str],
                   query_labels: Sequence[str]) -> list[int]:
    """Vertex indices in the document matched by the query's vocabulary."""
    if not doc_labels or not query_labels:
        return []
    pos: dict[str, int] = {}
    for i, lab in enumerate(doc_labels):
        pos.setdefault(str(lab).lower(), i)
    out, seen = [], set()
    for w in query_labels:
        i = pos.get(str(w).lower())
        if i is not None and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def interfacing_score(rex, doc_labels: Sequence[str], query_labels: Sequence[str],
                      *, reading: bool = True) -> dict[str, Any]:
    """Score a complex against a query vocabulary, by demand-driven read.

    `reading=False` skips `agentic_reading` and returns the coherence score alone,
    for callers ranking a large candidate set who want the diagnostics only on what
    survives.
    """
    if rex is None or int(getattr(rex, "nE", 0) or 0) == 0:
        return _zero()
    idx = shared_indices(doc_labels, query_labels)
    if len(idx) < MIN_SHARED:
        return _zero(len(idx))

    seed = np.asarray(idx, dtype=np.int32)
    try:
        kappa = np.asarray(rex.coherence_response(seed), dtype=np.float64).ravel()
    except Exception:
        return _zero(len(idx))
    kappa = kappa[np.isfinite(kappa)]
    if kappa.size == 0:
        return _zero(len(idx))

    out = {
        # the query's footprint under the document's own coherence field
        "score": float(kappa.sum()),
        "kappa": [float(x) for x in kappa],
        "kappa_mean": float(kappa.mean()),
        "n_shared": len(idx),
        "context_size": 0,
        "n_load_bearing": 0,
        "n_frustrated": 0,
    }
    if reading:
        try:
            ar = rex.agentic_reading(vertices=seed)
            out["context_size"] = int(ar.get("context_size", 0) or 0)
            out["n_load_bearing"] = len(ar.get("load_bearing", []) or [])
            out["n_frustrated"] = len(ar.get("frustrated", []) or [])
        except Exception:
            pass
    return out
