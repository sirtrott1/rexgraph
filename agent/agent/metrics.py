"""
Information metrics - perplexity, entropy, varentropy - as ONE calculus over two
carriers: the relational spectrum (structural) and an LLM token distribution (the
standard LLM metrics).

Perplexity = exp(entropy); entropy has an eigen-free Rényi form (RCF Part D /
scripts 18-19); varentropy = the spread of surprisal - a known LLM uncertainty
signal AND the RCF H₂-H₃ reliability gap. Same math, two carriers: a nonnegative
spectrum (RL4 channels) or a token-probability vector. That is why "these LLM
metrics work here" - they are the relational entropy calculus applied to tokens.
"""
from __future__ import annotations

import contextlib

import numpy as np


def _norm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    s = p.sum()
    return p / s if s > 0 else p


def entropy(p, order: float = 1.0) -> float:
    """Rényi entropy of order `order` (nats) of a distribution or nonneg spectrum.
    order=1 -> Shannon; order=2 -> collision (the RCF harmonic-log H₂). Eigen-free for
    integer order (only Σpᵃ)."""
    p = _norm(p)
    if p.size == 0:
        return 0.0
    if abs(order - 1.0) < 1e-9:
        return float(-(p * np.log(p)).sum())
    return float(np.log((p ** order).sum()) / (1.0 - order))


def perplexity(p, order: float = 1.0) -> float:
    """exp(entropy) - the effective number of states (Hill number / effective support
    size). For tokens this is the classic perplexity; for a spectrum it is the
    effective mode count."""
    return float(np.exp(entropy(p, order)))


def varentropy(p) -> float:
    """Var(-log p) under p - the spread of surprisal ("uncertainty of the
    uncertainty"). ~0 on a flat distribution, grows with heavy tails; equals ½·the
    Shannon-collision gap to leading order (RCF Part D.4 / script 19)."""
    p = _norm(p)
    if p.size == 0:
        return 0.0
    s = -np.log(p)
    mean = float((p * s).sum())
    return float((p * (s - mean) ** 2).sum())


# LLM token metrics (from per-token logprobs)
def token_metrics(logprobs) -> dict:
    """Standard LLM metrics from per-token logprobs (natural log): `perplexity` =
    exp(-mean logprob) (the cross-entropy PPL), `mean_surprisal` (nats/token), and
    the token `varentropy` (variance of surprisal across tokens) - the same varentropy
    the RCF reliability gap uses, here on the token distribution. High varentropy at
    low perplexity flags a confident-but-branchy step (a good place to look twice)."""
    lp = np.asarray(logprobs, dtype=float).ravel()
    lp = lp[np.isfinite(lp)]
    if lp.size == 0:
        return {"perplexity": float("nan"), "mean_surprisal": float("nan"),
                "varentropy": float("nan"), "n_tokens": 0}
    surprisal = -lp
    mean_s = float(surprisal.mean())
    return {
        "perplexity": float(np.exp(mean_s)),
        "mean_surprisal": mean_s,
        "varentropy": float(surprisal.var()),
        "n_tokens": int(lp.size),
    }


# Structural metrics (RCF-native, from a rex; no LLM needed)
def structural_metrics(rex) -> dict:
    """The relational complex's OWN information metrics from the RL4 spectrum, all
    eigen-free (Part D): `structural_entropy_H2` = the harmonic-log (Rényi-2);
    `structural_perplexity` = exp(H₂) = the effective mode count (how many degrees of
    freedom the relation graph carries); `varentropy_gap` = the H₂-H₃ reliability
    certificate (small -> the H₂ summary is trustworthy). The structural analog of an
    LLM's perplexity/varentropy - computed with the same calculus as token_metrics."""
    H2 = float(rex.harmonic_entropy)
    ve = rex.character_varentropy
    return {
        "structural_entropy_H2": round(H2, 6),
        "structural_perplexity": round(float(np.exp(H2)), 4),
        "effective_modes": round(float(np.exp(H2)), 4),
        "varentropy_gap": ve["gap"],
        "reliable": bool(ve["gap"] < 0.05),
    }


def _summ(values) -> dict:
    """Distribution summary (mean/std/min/max/n) of a list of numbers."""
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {}
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "n": int(a.size)}


def _trend(vals) -> str:
    """Direction of the last few values vs their start (±5% band). 'rising' /
    'falling' / 'stable' / 'insufficient'."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if len(v) < 3:
        return "insufficient"
    a, b = v[-3], v[-1]
    if b > a * 1.05:
        return "rising"
    if b < a * 0.95:
        return "falling"
    return "stable"


def session_metrics(coherence_per_turn, perplexity_per_turn=None) -> dict:
    """Per-SESSION information metrics over a conversation's turns: the trend of
    structural coherence (is the conversation losing structure?) and - when token
    metrics were captured per reply - of perplexity (is the model getting more
    uncertain?), plus per-metric summaries. Trend is over the last 3 turns."""
    out: dict = {"n_turns": len(list(coherence_per_turn))}
    coh = list(coherence_per_turn)
    if any(c is not None for c in coh):
        out["coherence"] = _summ(coh)
        out["coherence_trend"] = _trend(coh)
    if perplexity_per_turn is not None and any(p is not None for p in perplexity_per_turn):
        out["perplexity"] = _summ(perplexity_per_turn)
        out["perplexity_trend"] = _trend(perplexity_per_turn)
    return out


def corpus_metrics(rexes) -> dict:
    """Per-CORPUS information metrics: the DISTRIBUTION of each document's structural
    metrics (structural perplexity, effective modes, coherence, varentropy gap) across
    the collection, plus a corpus diversity = the effective number of coherence-distinct
    documents (exp of the Shannon entropy of the normalized per-document coherence).
    The 'how varied / how coherent is this collection' reading."""
    per, cohs = [], []
    for rex in rexes:
        if rex is None:
            continue
        try:
            sm = structural_metrics(rex)
            k = float(np.asarray(rex.coherence).mean())
            sm["coherence"] = round(k, 4)
            cohs.append(k)
            per.append(sm)
        except Exception:
            continue
    if not per:
        return {"n_documents": 0}
    out = {
        "n_documents": len(per),
        "structural_perplexity": _summ([p["structural_perplexity"] for p in per]),
        "effective_modes": _summ([p["effective_modes"] for p in per]),
        "coherence": _summ(cohs),
        "varentropy_gap": _summ([p["varentropy_gap"] for p in per]),
    }
    if len(cohs) > 1:
        # effective number of coherence-distinct documents (Hill number of the
        # normalized per-document coherence) - corpus structural diversity.
        out["corpus_diversity"] = round(perplexity(cohs), 3)
    return out


def reply_metrics(text: str, logprobs=None, token: dict = None,
                  structural: bool = False) -> dict:
    """Metrics for a generated reply. TWO COST TIERS:
      token (always, ~free): perplexity/varentropy from the reply's logprobs - the
        model already produced these, so extracting them costs ~0.02 ms.
      structural (only if `structural=True`, ~250 ms): builds the reply's OWN
        relational complex (auto_rex) for structural_perplexity/effective_modes/
        response_coherence + the fluent-but-hollow advisory. This is the expensive
        tier - computed on demand (when the interface asks), never eagerly on every
        reply. Best-effort; never raises. Shared by /model/generate and /chat."""
    out: dict = {}
    if token is not None:
        out["token"] = token
    elif logprobs:
        out["token"] = token_metrics(logprobs)
    if structural:
        try:
            if text and 20 < len(text) < 20000:
                from agent.auto import auto_rex
                rex = auto_rex(text)
                out["structural"] = structural_metrics(rex)
                kappa = float(np.asarray(rex.coherence).mean())
                out["response_coherence"] = round(kappa, 4)
                tok = out.get("token") or {}
                if tok.get("perplexity") and tok["perplexity"] < 10.0 and kappa < 0.5:
                    out["advisory"] = ("fluent_but_hollow: reads fluently but the "
                                       "relations it asserts are weakly coherent - verify")
        except Exception:
            pass
    return out


def response_metrics(rex=None, logprobs=None) -> dict:
    """Unified reading over whatever is available: the structural metrics of a
    response's relational complex and/or the token metrics of its logprobs. When both
    are present the agent can compare them - e.g. a low token perplexity but high
    structural perplexity means the text reads fluently yet the relations it asserts
    are diffuse/unsupported (a fluent-but-hollow answer)."""
    out: dict = {}
    if rex is not None:
        with contextlib.suppress(Exception):
            out["structural"] = structural_metrics(rex)
    if logprobs is not None:
        out["token"] = token_metrics(logprobs)
    return out
