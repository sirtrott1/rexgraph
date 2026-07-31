"""
agent.scoring: one ranking, built on the interfacing vector.

Three rankings had grown in the tree, none of them using the canonical mechanism:
`corpus.score_document` mixed a label Jaccard, a cosine between MEAN structural
characters, and a hand-rolled spectral term under fixed 0.3/0.35/0.35 weights;
`rcdb.find_similar` used a kappa correlation times a square-rooted overlap; and
`RexGraph.interfacing_vector` -- the Poisson-lift -> typed-channel -> bilinear-score
map the whole design is built around -- was reachable only through an HTTP route.

This is that mechanism used as the ranking:

    rho  = weighted vertex source from the query tokens present in the document
    psi  = B1^T L0^+ rho                     the query's induced edge flow
    iv   = [I_T, I_G, I_F, schrodinger]      psi read through the typed channels
    score     = ||iv||                       how strongly the query engages the doc
    character = iv / ||iv||                  which channels it engages through

That split is the point, and it is not the obvious choice. sphere_pos alone ranks
badly: it is a DIRECTION, so normalizing divides out engagement strength. A
three-sentence stub puts nearly all of its (tiny) interfacing energy in the T
channel and scores 0.99, beating a document that actually answers the query at
0.19 -- ranking on it rewards structural poverty. Measured on a 5-document set:
sphere_pos[0] 5/6, ||iv|| 6/6. Magnitude ranks; direction explains.

Lexical overlap is no longer a ranking term. It survives as a candidate prefilter
(`query_engine._signature_affinity`), which is what a token match is good for:
deciding what to look at, not deciding what is relevant.

No fixed mixing constants. Magnitude and direction are just the raw interfacing
vector in polar form, so nothing is discarded and nothing is invented; coverage /
efficiency / confidence come back as diagnostics for the caller's policy rather
than folded into the number.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


#: fewer shared vertices than this and the Poisson source is a point, not a
#: footprint: L0^+ rho carries no interfacing structure to read.
MIN_SHARED = 2


def _zero(n_shared: int = 0) -> Dict[str, Any]:
    return {"score": 0.0, "character": [0.0, 0.0, 0.0, 0.0],
            "channels": [0.0, 0.0, 0.0, 0.0], "magnitude": 0.0,
            "coverage": 0.0, "efficiency": 0.0, "confidence": {},
            "n_shared": int(n_shared)}


def shared_indices(doc_labels: Sequence[str],
                   query_labels: Sequence[str]) -> List[int]:
    """Vertex indices in the document matched by the query's vocabulary."""
    if not doc_labels or not query_labels:
        return []
    pos: Dict[str, int] = {}
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
                      *, target_signal: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Score a complex against a query vocabulary through the interfacing vector.

    `target_signal` is the edge vector psi is read against. It defaults to psi
    itself, which makes the channel readings quadratic forms of the induced flow --
    orientation-invariant, unlike any signal built from raw incidence, and the
    reading that says how strongly the query engages this document rather than
    which way its edges happen to point.

    Returns the score plus the diagnostics the bundle already computes; nothing is
    folded into the scalar that a caller might want to see separately.
    """
    if rex is None or int(getattr(rex, "nE", 0) or 0) == 0:
        return _zero()
    idx = shared_indices(doc_labels, query_labels)
    if len(idx) < MIN_SHARED:
        return _zero(len(idx))

    ti = np.asarray(idx, dtype=np.int32)
    tw = np.ones(len(idx), dtype=np.float64)
    try:
        # target_signal=None asks the bundle for the self-interfacing reading, which
        # it resolves from the psi it computes anyway: one L0^+ solve, not the two
        # a caller pays to obtain psi and then hand it back.
        iv = rex.interfacing_vector(
            ti, tw,
            None if target_signal is None
            else np.ascontiguousarray(target_signal, dtype=np.float64))
        mag = float(np.linalg.norm(np.asarray(iv.get("psi"), dtype=np.float64)))
        if not np.isfinite(mag) or mag <= 0.0:
            return _zero(len(idx))
    except Exception:
        return _zero(len(idx))

    raw = np.asarray(iv.get("iv"), dtype=np.float64).ravel()
    sp = np.asarray(iv.get("sphere_pos"), dtype=np.float64).ravel()
    if raw.size == 0 or not np.all(np.isfinite(raw)):
        return _zero(len(idx))
    cov = float(iv.get("coverage", 0.0) or 0.0)
    score = float(np.linalg.norm(raw))
    if not np.isfinite(score) or cov <= 0.0:
        # no spectral support: the reading exists but nothing backs it.
        score = 0.0
    return {
        "score": max(score, 0.0),
        "character": [float(x) for x in sp] if np.all(np.isfinite(sp)) else [0.0] * 4,
        "channels": [float(x) for x in raw],
        "magnitude": mag,
        "coverage": cov,
        "efficiency": float(iv.get("efficiency", 0.0) or 0.0),
        "confidence": iv.get("confidence", {}),
        "n_shared": len(idx),
    }
