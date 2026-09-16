"""Conversation turns as primary relations, with numerical path observations."""
from copy import deepcopy
from math import isfinite, isnan
from threading import RLock

import numpy as np

from .gate import MalaughGate

__all__ = ["TurnField"]


class TurnField:
    """One live conversation. Capture and preview never append to it."""

    def __init__(self, *, fence_k=3.0, warmup=3):
        if isinstance(fence_k, bool) or not isinstance(fence_k, (int, float)) or not isfinite(fence_k) or fence_k < 0:
            raise ValueError("fence_k must be a finite nonnegative number")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise ValueError("warmup must be a positive integer")
        self._lock = RLock()
        self._turns = []
        self._vocab = {}
        self._gate = MalaughGate(fence_k=fence_k, warmup=warmup)
        self._rex = None
        self._last = {}

    @property
    def n_turns(self):
        with self._lock:
            return len(self._turns)

    @property
    def rex(self):
        return self._rex

    def _terms(self, text, profile=None):
        from rexgraph.corpus_profile import TEXT, tokenize
        if not isinstance(text, str):
            raise TypeError("a conversation turn must be text")
        return list(dict.fromkeys(w for w, _, _ in tokenize(text, TEXT if profile is None else profile)
                                  if any(ch.isalnum() for ch in w)))

    @staticmethod
    def _complex(turns, vocab):
        from rexgraph.graph import RexGraph
        ptr = np.concatenate(([0], np.cumsum([len(t) for t in turns], dtype=np.int64)))
        flat = np.asarray([v for turn in turns for v in turn], dtype=np.int64)
        rex = RexGraph.from_hypergraph(ptr, flat, relation_ids=np.arange(len(turns), dtype=np.int64))
        inverse = {v: word for word, v in vocab.items()}
        rex._agent_meta = {"vertex_labels": [inverse[v] for v in range(rex.nV)]}
        return rex

    def _advance(self, text, profile, *, publish):
        from rexgraph.scale_propagator import malaugh_quantities
        terms = self._terms(text, profile)
        n = len(self._turns)
        if not terms:
            return {"terms": [], "seeds": [], "weights": [], "event": False,
                    "carried": 0, "n_turns": n, "H_T": None,
                    "baseline_turns": n, "status": "no_terms"}
        vocab = dict(self._vocab)
        ids = [vocab.setdefault(term, len(vocab)) for term in terms]
        turns = self._turns + [ids]
        rex = self._complex(turns, vocab)
        gate = deepcopy(self._gate)
        entropy = float(malaugh_quantities(rex)["H_T"])
        if isnan(entropy):
            event, entropy, status = False, None, "undefined_entropy"
        else:
            event = bool(gate.observe_entropy(entropy)["event"])
            status = "observed"
        seeds = terms if event or not n else list(vocab)
        degree = np.asarray(rex.degree)
        result = {"terms": list(terms), "seeds": list(seeds),
                  "weights": [1.0 / max(int(degree[vocab[word]]), 1) for word in seeds],
                  "event": event, "carried": len(seeds) - len(terms),
                  "n_turns": n + 1, "H_T": entropy, "baseline_turns": n, "status": status}
        if publish:
            self._turns, self._vocab, self._gate, self._rex = turns, vocab, gate, rex
            self._last = deepcopy(result)
        return result

    def observe(self, text, *, profile=None):
        """Append one tokenized turn after its construction and observation succeed."""
        with self._lock:
            return self._advance(text, profile, publish=True)

    def preview(self, text, *, profile=None):
        """Observe a candidate with a copied baseline, without changing live state."""
        with self._lock:
            return self._advance(text, profile, publish=False)

    def validate_interval(self, start=0, stop=None):
        n = self.n_turns
        stop = n if stop is None else stop
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, stop)):
            raise TypeError("turn interval bounds must be integers")
        if not 0 <= start <= stop <= n:
            raise ValueError("turn interval must satisfy 0 <= start <= stop <= n_turns")
        return start, stop

    def snapshot(self, start=0, stop=None):
        """Capture a half open interval of complete conversation prefixes."""
        from rexgraph.graph import TemporalRex
        with self._lock:
            start, stop = self.validate_interval(start, stop)
            result = TemporalRex([], general=True)
            for step in range(start, stop):
                result.append_snapshot(self._complex(self._turns[:step + 1], self._vocab), at=step)
            return result
