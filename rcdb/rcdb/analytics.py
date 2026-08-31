"""
rcdb.analytics: the measurements a stored record's signature is built from.

These live here rather than in the agent because `structural_signature` needs them on
every put, and a store that had to import the application to describe a complex could not
be installed or reasoned about on its own. The agent re-exports them, so its callers keep
the import path they had.

`interfacing_score` is also reachable as a HOOK, so an application can supply a richer
similarity without this package knowing anything about it. What is here is the default
that makes the store work alone.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["coherence_greens", "coherence_greens_mean", "coherence_kappa",
           "coherence_mean", "greens_budget", "interfacing_score",
           "shared_indices", "structural_metrics"]

_GREENS_BUDGET_ENV = "REXGRAPH_VERTEX_CHARACTER_MAX_NODES"
_GREENS_BUDGET_DEFAULT = 1500

#: Below this many shared terms a score is not evidence, it is coincidence: a single
#: matched vertex has no neighbourhood for the diffusion to describe.
MIN_SHARED = 2



def greens_budget() -> int:
    """Vertex budget for the global Green's coherence, from the environment
    (0 = no budget, run it at any size). One reader so every caller gates alike."""
    try:
        return int(os.environ.get(_GREENS_BUDGET_ENV, str(_GREENS_BUDGET_DEFAULT)))
    except ValueError:
        return _GREENS_BUDGET_DEFAULT


def coherence_kappa(rex) -> np.ndarray:
    """Per-vertex coherence kappa, shape (nV,), at any scale. THE default read.

    This is `local_coherence`: kappa against the star-average character, O(nnz), so
    it answers on a complex of any size. Its companion `rex.coherence` is a
    different moment of the same propagator (kappa against the global Green's phi),
    not a more accurate version of this one: on real complexes the two correlate
    anywhere from -0.30 to +0.99, and the global read costs one block-CG solve per
    vertex because its sandwiched two-inverse numerator resists selected inversion.
    Reach for that one through `coherence_greens`, which gates it, and report it
    under its own key rather than mixing the two in one field."""
    with contextlib.suppress(Exception):
        k = np.asarray(rex.local_coherence, dtype=float).ravel()
        if k.size:
            return np.where(np.isfinite(k), k, 0.0)
    return np.zeros(int(getattr(rex, "nV", 0) or 0), dtype=float)


def coherence_mean(rex, default: float = 0.0) -> float:
    """Mean of `coherence_kappa`. The value stored as a signature's `kappa_mean`:
    one quantity at every scale, so records stay comparable under `avg(kappa_mean)`."""
    k = coherence_kappa(rex)
    return float(k.mean()) if k.size else float(default)


def coherence_greens(rex, budget: int | None = None) -> np.ndarray | None:
    """Per-vertex GLOBAL Green's coherence, or None when the complex is over budget.

    The exact global moment, at one solve per vertex. None means "not computed at
    this size", never "zero": store it under its own key so an absent value stays
    distinguishable from a low one."""
    b = greens_budget() if budget is None else int(budget)
    nV = int(getattr(rex, "nV", 0) or 0)
    if b > 0 and nV > b:
        return None
    with contextlib.suppress(Exception):
        k = np.asarray(rex.coherence, dtype=float).ravel()
        if k.size:
            return np.where(np.isfinite(k), k, 0.0)
    return None


def coherence_greens_mean(rex, budget: int | None = None) -> float | None:
    """Mean of `coherence_greens`, or None when over budget."""
    k = coherence_greens(rex, budget)
    return float(k.mean()) if k is not None and k.size else None


def structural_metrics(rex) -> dict:
    """The relational complex's OWN information metrics from the RL4 spectrum, all
    eigen-free: `structural_entropy_H2` = the harmonic-log (Rényi-2);
    `structural_perplexity` = exp(H₂) = the effective mode count (how many degrees of
    freedom the relation graph carries); `varentropy_gap` = the H₂-H₃ reliability
    certificate (small -> the H₂ summary is trustworthy). The structural analog of an
    LLM's perplexity/varentropy - computed with the same calculus as token_metrics."""
    H2 = float(rex.harmonic_entropy)
    # The H3 half needs tr(RL4^3), which forms RL4^2, and that product's FILL is
    # data-dependent: on a complex with wide branching groups the co-participation
    # matrix squares into something enormous. Measured on a lexical complex, 31.4s and
    # 22.3 GB at nE 553,021, and still climbing through 94 GB at nE 1,626,490 before it
    # was killed. The bound nnz(X^2) <= sum_i sum_{j in row i} nnz(row j) is one matvec,
    # so the cost is knowable BEFORE paying it rather than after.
    ve = {"H2": round(H2, 6), "H3": None, "gap": None, "declined": None}
    try:
        X = rex._rl4_sparse
        rownnz = np.diff(X.indptr).astype(np.float64)
        # sum over nonzeros (i,j) of nnz(row j): the exact upper bound on nnz(X^2)
        fill = (float(np.dot(np.bincount(X.indices, minlength=X.shape[0])
                             .astype(np.float64), rownnz)) if X.nnz else 0.0)
        from rexgraph.core._common import check_dense_allocation
        check_dense_allocation("character_varentropy RL4^2 fill",
                               int(max(fill, 1)), 1)
        ve = rex.character_varentropy
    except Exception as exc:
        ve = {"H2": round(H2, 6), "H3": None, "gap": None,
              "declined": f"{type(exc).__name__}"}
    return {
        "structural_entropy_H2": round(H2, 6),
        "structural_perplexity": round(float(np.exp(H2)), 4),
        "effective_modes": round(float(np.exp(H2)), 4),
        "varentropy_gap": ve.get("gap"),
        # gap is None when the H3 moment was declined; that is "not certified", which is
        # a different statement from "certified unreliable"
        # `reliability_gap` certifies that the CHEAP H2 is exact, and its own docstring
        # says when: "~0 on flat/unweighted spectra (the cheap H2 is exact); grows with
        # weight-induced non-uniformity". So this is an exactness test, not a policy
        # band, and measured the values are 13 orders apart with nothing in between:
        # 5.6e-16 and 1.1e-15 where H2 is exact, 4.3e-02 where it is not. The old 0.05
        # sat ABOVE the inexact case and certified it.
        "reliable": (None if ve.get("gap") is None
                     else bool(abs(ve["gap"]) <= 1e-9 * max(abs(ve.get("H2") or 1.0), 1.0))),
    }


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
