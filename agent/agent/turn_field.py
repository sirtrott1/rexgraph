"""A conversation as a complex: turns are relations over the terms they name.

A turn's terms are one branching relation, so a term shared across turns is one
vertex and the path a conversation takes is readable from the boundary. Seeds for
the next retrieval come from the whole path or the latest turn, decided by the gate.
"""
from __future__ import annotations

import numpy as np

__all__ = ["TurnField"]


class TurnField:
    """One conversation. Not reusable across sessions: the gate carries a baseline."""

    __slots__ = ("_turns", "_vocab", "_gate", "_rex", "_last")

    def __init__(self, *, fence_k: float = 3.0, warmup: int = 3):
        from rexgraph.flow.gate import MalaughGate
        self._turns: list[list[int]] = []      # each turn, as vertex ids
        self._vocab: dict[str, int] = {}
        self._gate = MalaughGate(fence_k=fence_k, warmup=warmup)
        self._rex = None
        self._last: dict = {}

    @property
    def n_turns(self) -> int:
        return len(self._turns)

    @property
    def rex(self):
        """The conversation complex as it stands: one branching column per turn."""
        return self._rex

    def _terms(self, text: str, profile=None) -> list[str]:
        from rexgraph.corpus_profile import TEXT, tokenize
        prof = profile if profile is not None else TEXT
        out, seen = [], set()
        for w, _a, _b in tokenize(text or "", prof):
            if w not in seen and any(ch.isalnum() for ch in w):
                seen.add(w)
                out.append(w)
        return out

    def _rebuild(self):
        from rexgraph.graph import RexGraph
        ptr = np.zeros(len(self._turns) + 1, dtype=np.int64)
        flat: list[int] = []
        for i, t in enumerate(self._turns):
            flat.extend(t)
            ptr[i + 1] = len(flat)
        self._rex = RexGraph.from_hypergraph(ptr, np.asarray(flat, dtype=np.int64))

    def observe(self, text: str, *, profile=None) -> dict:
        """Add a turn and return what the retrieval should seed with.

        Keys: `terms` (this turn's), `seeds` (what to retrieve with, this turn's terms
        when the path turned, the whole path's otherwise), `weights` aligned to `seeds`
        as `1/deg` in the conversation, `event` (the gate fired), `carried` (how many
        terms came from earlier turns), `n_turns`, `H_T`.
        """
        terms = self._terms(text, profile)
        if not terms:
            return {"terms": [], "seeds": [], "weights": [], "event": False,
                    "carried": 0, "n_turns": self.n_turns, "H_T": None}

        ids = []
        for w in terms:
            j = self._vocab.get(w)
            if j is None:
                j = len(self._vocab)
                self._vocab[w] = j
            ids.append(j)
        self._turns.append(ids)
        self._rebuild()

        event, h_t = False, None
        try:
            obs = self._gate.observe(self._rex)
            event, h_t = bool(obs["event"]), float(obs["H_T"])
        except Exception:
            pass                                # a degenerate first turn is not an event

        # a turn that LEFT the path stands alone; one that extends it carries the path
        if event or self.n_turns == 1:
            seeds = list(terms)
        else:
            back = {}
            for t in self._turns:
                for j in t:
                    back[j] = True
            inv = {j: w for w, j in self._vocab.items()}
            seeds = [inv[j] for j in back]

        deg = np.asarray(self._rex.degree, dtype=np.float64)
        weights = [1.0 / max(float(deg[self._vocab[w]]), 1.0) if self._vocab[w] < deg.size
                   else 1.0 for w in seeds]
        self._last = {"terms": terms, "seeds": seeds, "weights": weights,
                      "event": event, "carried": len(seeds) - len(terms),
                      "n_turns": self.n_turns, "H_T": h_t}
        return dict(self._last)
