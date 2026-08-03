"""
agent.temporal: time as a retrieval signal.

The RCDB was already bitemporal and retrieval already threaded as_of/valid_at, so a
corpus could be read AS IT STOOD at a time. What was missing is the other use of
time: once candidates are structurally scored, preferring evidence that has settled
over evidence still in dispute, or recent over stale.

Everything here reads STORED SIGNATURES ONLY. `store.history` returns records without
touching a blob, and a signature already carries nV/nE/betti1/kappa_mean per version,
so a candidate's temporal features cost dict arithmetic. `rcdb.trajectory` answers a
richer question -- it reconstructs a complex per version and runs a cross-complex
bridge per step -- which is the right tool for inspecting one lineage and the wrong
one to run per candidate inside a query.

No decay constants. A half-life would be a magic number with no defensible value, so
recency is the ORDERING of the candidates actually in hand: same information,
scale-free, and it cannot be wrong about a unit. Stability is likewise a relative
quantity, built from each step's change measured against its own magnitude.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: structural quantities a signature always carries. Drift and stability are read
#: from these rather than from the payload.
QUANTITIES = ("nV", "nE", "nF", "betti1", "kappa_mean")

# How a caller may combine temporal signal with the structural score. A registry,
# not a fixed tuple: a domain-specific policy -- a pseudotime ordering, a
# batch-corrected recency -- should not mean editing this module. A policy is
# `fn(features, recency_weights, doc_id) -> weight`, and the structural score gates
# it whatever it returns.
from rexgraph.registry import Registry

_POLICIES = Registry("temporal policy")


def register_policy(name: str, fn) -> None:
    """Register a rerank policy. `fn(features, recency, doc_id) -> float weight`."""
    _POLICIES.register(name, fn)


def unregister_policy(name: str):
    return _POLICIES.unregister(name)


def available_policies() -> list[str]:
    return _POLICIES.available()


register_policy("off", lambda f, r, d: 1.0)
register_policy("stability", lambda f, r, d: f[d]["stability"])
register_policy("recency", lambda f, r, d: r.get(d, 1.0))
#: recent AND undisputed
register_policy("settled", lambda f, r, d: f[d]["stability"] * r.get(d, 1.0))

#: kept as a name for callers that enumerated the old tuple
MODES = tuple(_POLICIES.available())


def _num(x) -> float:
    """Coerce a signature entry to a float. `betti` arrives as a list; take b1."""
    if isinstance(x, (list, tuple)):
        return float(x[1]) if len(x) > 1 else (float(x[0]) if x else 0.0)
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _relative_change(a: float, b: float) -> float:
    """|b - a| scaled by the pair's own magnitude, so the result is unitless and in
    [0, 1] whatever the quantity measures. Two zeros are unchanged, not undefined."""
    denom = abs(a) + abs(b)
    if denom <= 0.0:
        return 0.0
    return min(1.0, abs(b - a) / denom)


def temporal_features(store, id: str) -> dict[str, Any]:
    """Per-record temporal features from the stored signatures. Opens no blob.

    stability : 1.0 means every revision left the structure where it was; 0.0 means
                each revision replaced it. A record with one version is fully stable
                -- nothing about it has ever been in dispute.
    drift     : net signed change per quantity across the whole lineage.
    direction : whether the later half of the lineage settled ("converging"), came
                apart ("diverging"), or neither.
    """
    neutral = {"id": id, "n_versions": 0, "stability": 1.0, "drift": {},
               "direction": "level", "tx_from": None, "valid_from": None,
               "version": None}
    try:
        hist = list(store.history(id) or [])
    except Exception:
        return neutral
    if not hist:
        return neutral

    sigs = [(r.signature or {}) for r in hist]
    last = hist[-1]
    out = {
        "id": id,
        "n_versions": len(hist),
        "version": getattr(last, "version", None),
        "tx_from": getattr(last, "tx_from", None),
        "valid_from": getattr(last, "valid_from", None),
    }

    steps: list[float] = []
    for i in range(1, len(sigs)):
        a, b = sigs[i - 1], sigs[i]
        shared = [k for k in QUANTITIES if a.get(k) is not None and b.get(k) is not None]
        if not shared:
            continue
        steps.append(sum(_relative_change(_num(a[k]), _num(b[k])) for k in shared)
                     / len(shared))

    out["stability"] = 1.0 - (sum(steps) / len(steps)) if steps else 1.0
    out["drift"] = {k: _num(sigs[-1].get(k)) - _num(sigs[0].get(k))
                    for k in QUANTITIES
                    if sigs[0].get(k) is not None and sigs[-1].get(k) is not None}

    # direction: did the lineage settle down or come apart? Compare the mean change
    # of its later half against its earlier half. Relative, so no threshold.
    if len(steps) >= 2:
        mid = len(steps) // 2
        early = sum(steps[:mid]) / max(mid, 1)
        late = sum(steps[mid:]) / max(len(steps) - mid, 1)
        out["direction"] = ("converging" if late < early
                            else "diverging" if late > early else "level")
    else:
        out["direction"] = "level"
    return out


def recency_weights(items: Sequence[dict[str, Any]], *,
                    key: str = "tx_from") -> dict[str, float]:
    """Map each candidate to a [0, 1] recency weight by its ORDER among the others.

    Deliberately not a decay: exp(-t/tau) needs a tau, and no value of tau is
    defensible across a corpus of documents, a lineage of schema versions and a
    stream of measurements at once. Ranking the candidates in hand carries the same
    preference with nothing to tune, and is invariant to the units of the clock.

    The weight is (i + 1) / n, so it is in (0, 1] and never reaches zero. A rank is
    ORDINAL: the oldest candidate of a set is not "0% recent", and a multiplicative
    zero would erase a strongly relevant document for no reason but its position in
    the ordering. Recency prefers; it does not annihilate.
    """
    rows = [(it.get(key), str(it.get("doc_id") or it.get("id"))) for it in items]
    known = sorted({t for t, _ in rows if t is not None})
    if not known:
        return {d: 1.0 for _, d in rows}
    # an unknown timestamp is its OWN lowest bucket, not a share of the oldest known
    # one: absence of evidence must not tie with evidence, let alone outrank it.
    buckets: list[Any] = ([None] if any(t is None for t, _ in rows) else []) + known
    n = len(buckets)
    weight = {b: (i + 1) / n for i, b in enumerate(buckets)}
    return {d: weight[t] for t, d in rows}


def rerank(sections: list[dict[str, Any]], store, *, mode: str = "stability",
           ) -> list[dict[str, Any]]:
    """Reorder structurally-scored sections by a temporal policy.

    The structural score is a gate, never a summand: a candidate that matched nothing
    stays at zero whatever its history looks like. Temporal signal reorders relevant
    results; it does not manufacture relevance.

    Each section keeps `structural_score` and gains a `temporal` block, so the
    reordering is auditable and a caller can re-derive it under another policy.
    """
    if mode not in _POLICIES:
        raise ValueError(f"unknown temporal mode {mode!r}, expected one of "
                         f"{available_policies()}")
    if not sections:
        return list(sections)

    feats = {}
    for s in sections:
        did = s.get("doc_id")
        f = temporal_features(store, did)
        feats[did] = f
        s.setdefault("structural_score", s.get("score", 0.0))
        s["temporal"] = f
        s.setdefault("tx_from", f.get("tx_from"))

    if mode == "off":
        return sections

    rec = recency_weights([{"doc_id": s.get("doc_id"),
                            "tx_from": feats[s.get("doc_id")].get("tx_from")}
                           for s in sections])

    policy = _POLICIES.require(mode)
    for s in sections:
        did = s.get("doc_id")
        base = float(s.get("structural_score", 0.0) or 0.0)
        w = policy(feats, rec, did)
        s["temporal"] = {**feats[did], "weight": round(float(w), 6), "mode": mode}
        s["score"] = round(base * float(w), 6)

    sections.sort(key=lambda s: (-float(s.get("score", 0.0)), str(s.get("doc_id"))))
    return sections
