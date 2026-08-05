"""
agent.conversation: conversation memory as temporal topology.

Each exchange in a conversation produces an ExchangeResult.
The sequence of exchanges is a temporal relational complex.
BIOES tags track which relational edges persist, are born,
or die across exchanges. Persistent edges ARE the memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from agent.exchange import ExchangeResult, analyze_exchange_sequence, build_exchange_complex

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Tracked state across a conversation."""
    exchanges: list[ExchangeResult] = field(default_factory=list)
    exchange_rexes: list = field(default_factory=list)
    texts: list[dict] = field(default_factory=list)  # {input, output}
    drift_report: dict = field(default_factory=dict)


class ConversationTracker:
    """Maintains a temporal relational complex across chat exchanges.

    Each exchange is recorded as an ExchangeResult. The tracker
    monitors drift (kappa changing), context loss (shared entities
    dropping), and structural shifts (Hodge profile changing).
    """

    def __init__(self):
        self._state = ConversationState()

    @property
    def n_exchanges(self):
        return len(self._state.exchanges)

    @property
    def exchanges(self):
        return self._state.exchanges

    def record_exchange(self, input_text: str, output_text: str) -> ExchangeResult:
        """Record a new input/output exchange.

        Builds the exchange complex, computes invariants,
        updates drift tracking.
        """
        rex, result, ec = build_exchange_complex(input_text, output_text)

        self._state.exchanges.append(result)
        self._state.exchange_rexes.append(rex)
        self._state.texts.append({"input": input_text, "output": output_text})

        # Update drift analysis
        if len(self._state.exchanges) >= 2:
            self._state.drift_report = analyze_exchange_sequence(
                self._state.exchanges,
            )

        return result

    def get_drift_report(self) -> dict:
        """Get the current drift analysis across all exchanges."""
        if len(self._state.exchanges) < 2:
            return {"n_exchanges": len(self._state.exchanges), "status": "too few exchanges"}

        report = dict(self._state.drift_report)

        # Add trend assessments
        kappas = report.get("kappas", [])
        if len(kappas) >= 3:
            recent = kappas[-3:]
            if all(recent[i] < recent[i-1] for i in range(1, len(recent))):
                report["kappa_trend"] = "declining"
            elif all(recent[i] > recent[i-1] for i in range(1, len(recent))):
                report["kappa_trend"] = "improving"
            else:
                report["kappa_trend"] = "stable"

        shared = report.get("shared_counts", [])
        if len(shared) >= 2:
            if shared[-1] < shared[0] * 0.5:
                report["context_status"] = "significant context loss"
            elif shared[-1] < shared[0] * 0.8:
                report["context_status"] = "minor context loss"
            else:
                report["context_status"] = "context preserved"

        return report

    def note_reply_perplexity(self, perplexity: float) -> None:
        """Record a reply's token perplexity (from the generate metrics) so the
        session metrics can trend model uncertainty alongside structural coherence."""
        if not hasattr(self, "_reply_ppls"):
            self._reply_ppls: list = []
        self._reply_ppls.append(
            float(perplexity) if perplexity is not None else None)

    def note_exchange_metrics(self, metrics: dict, text: str = "") -> None:
        """Store a reply's TOKEN metrics (free) for this turn, per MESSAGE, so they
        are retrievable when the user returns to any point in the conversation, and
        feed its perplexity into the session trend. The reply text is kept so the
        expensive structural tier can be computed LAZILY on demand, never eagerly."""
        if not hasattr(self, "_exchange_metrics"):
            self._exchange_metrics: list = []
        self._exchange_metrics.append({"metrics": metrics or {}, "text": text or ""})
        ppl = ((metrics or {}).get("token") or {}).get("perplexity")
        self.note_reply_perplexity(ppl)

    def exchange_metrics(self, structural: bool = False) -> list[dict]:
        """Per-message metrics for every recorded turn (what the UI reads on navigating
        back to a message). Token metrics are stored/free. When `structural=True`, the
        expensive tier is computed lazily from each stored reply text and CACHED on the
        record, so it is paid once, only if the interface actually drills in."""
        out = []
        for rec in getattr(self, "_exchange_metrics", []):
            m = dict(rec.get("metrics") or {})
            if structural and "structural" not in m and rec.get("text"):
                try:
                    from agent.metrics import reply_metrics
                    s = reply_metrics(rec["text"], token=m.get("token"), structural=True)
                    m.update({k: v for k, v in s.items() if k != "token"})
                    rec["metrics"] = m   # cache the computed structural tier
                except Exception:
                    pass
            out.append(m)
        return out

    def session_metrics(self) -> dict:
        """Per-SESSION information metrics: the trend of structural coherence over
        turns (exchange κ: is the conversation losing structure?) and, when reply
        perplexities were noted, of model uncertainty, with per-metric summaries.
        Same Rényi/varentropy calculus as the per-reply/document/corpus metrics
        (agent.metrics.session_metrics)."""
        from agent.metrics import session_metrics as _session_metrics
        # kappa_mean is the exchange complex's overall coherence per turn; exchange_kappa
        # is κ restricted to cross-exchange edges (0 when Q/A share no structure), so it
        # is not the right trend signal.
        cohs = [getattr(ex, "kappa_mean", None) for ex in self._state.exchanges]
        ppls = getattr(self, "_reply_ppls", None)
        return _session_metrics(cohs, ppls)

    def get_memory_edges(self) -> list[str]:
        """Get the entities that persist across all exchanges.

        Uses entity_bioes_matrix for per-entity lifecycle tracking
        when available, falls back to set intersection.
        """
        if not self._state.texts:
            return []

        from agent.adapters.text import TextAdapter
        ta = TextAdapter()

        all_entity_sets = []
        for t in self._state.texts:
            combined = t["input"] + " " + t["output"]
            ec = ta.build(combined, min_count=1, max_vocab=300)
            all_entity_sets.append(set(w.lower() for w in ec.vertex_labels))

        if not all_entity_sets:
            return []

        # Try per-entity lifecycle via compiled kernel
        try:
            from rexgraph.core._temporal_entity import entity_bioes_matrix
            n_entities = len(all_entity_sets[0])
            T = len(all_entity_sets)
            if T >= 2 and n_entities > 0:
                # Build birth/death arrays from entity presence
                all_labels = sorted(all_entity_sets[0])
                birth = np.zeros(len(all_labels), dtype=np.int32)
                death = np.full(len(all_labels), T, dtype=np.int32)
                for i, label in enumerate(all_labels):
                    for t_idx, eset in enumerate(all_entity_sets):
                        if label in eset:
                            if birth[i] == 0 and t_idx > 0:
                                birth[i] = t_idx
                            death[i] = t_idx + 1
                bioes = entity_bioes_matrix(birth, death, T)
                # Persistent = entities tagged "I" (inside) across all time steps
                if isinstance(bioes, np.ndarray) and bioes.ndim == 2:
                    persistent = []
                    for i, label in enumerate(all_labels):
                        if i < bioes.shape[0]:
                            # Entity is persistent if it's present in first and last
                            if label in all_entity_sets[0] and label in all_entity_sets[-1]:
                                persistent.append(label)
                    if persistent:
                        return sorted(persistent)
        except Exception:
            pass

        # Fallback: set intersection
        persistent = all_entity_sets[0]
        for s in all_entity_sets[1:]:
            persistent = persistent & s

        return sorted(persistent)

    def suggest_context(self, query: str) -> str:
        """Suggest which previous exchanges to include as context.

        Uses exchange kappa to identify the most structurally
        coherent previous exchanges to reference.
        """
        if not self._state.exchanges:
            return ""

        # Rank previous exchanges by exchange kappa
        ranked = sorted(
            enumerate(self._state.exchanges),
            key=lambda x: x[1].exchange_kappa if not np.isnan(x[1].exchange_kappa) else 0,
            reverse=True,
        )

        # Include the top exchanges as context
        context_parts = []
        for idx, _ex in ranked[:3]:
            if idx < len(self._state.texts):
                t = self._state.texts[idx]
                context_parts.append(t["output"])

        return "\n\n".join(context_parts)

    def reset(self):
        """Clear conversation state."""
        self._state = ConversationState()
