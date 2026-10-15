"""Word sense as a restriction on the incidence (word, relation).

A polysemous word is one vertex carrying several senses, so the sense sits on the
incidence: which relation a word is approached through selects its sense. A sense
admits its own members plus, one hop out, the members of everything it relates to.
Where several senses tie, `disambiguate` returns all of them.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

__all__ = ["SenseModel", "extents"]


def extents(synsets: dict, relations: dict, senses, *, hops: int = 1) -> dict:
    """sense -> the lemmas it admits: its own members plus, `hops` out, its neighbours'.

    `relations` maps a synset to an iterable of neighbour synsets.
    """
    out = {}
    for s in senses:
        seen, frontier = set(synsets.get(s, ())), {s}
        for _ in range(max(0, int(hops))):
            nxt = set()
            for u in frontier:
                for t in relations.get(u, ()):
                    if t not in nxt:
                        nxt.add(t)
                        seen |= set(synsets.get(t, ()))
            frontier = nxt
        out[s] = seen
    return out


class SenseModel:
    """The candidate senses of one word, and which lemmas each admits.

    Accepts an inventory shaped as `{sense: [lemma, ...]}` and a neighbour map.
    WordNet records can be loaded through `inventory`.
    """

    def __init__(self, word: str, senses, extent: dict):
        self.word = str(word).lower()
        self.senses = list(senses)
        self.extent = extent
        self.pos = {s: i for i, s in enumerate(self.senses)}
        self.d = len(self.senses)

    @classmethod
    def for_word(cls, word, synsets, by_lemma, relations, *, hops: int = 1,
                 min_members: int = 2):
        senses = [s for s in by_lemma.get(str(word).lower(), ())
                  if len(synsets.get(s, ())) >= min_members]
        return cls(word, senses, extents(synsets, relations, senses, hops=hops))

    def mask(self, lemma) -> np.ndarray:
        """Which senses admit this lemma: the indicator a restriction carries."""
        L = str(lemma).lower()
        return np.asarray([1.0 if L in self.extent[s] else 0.0 for s in self.senses])

    def disambiguate(self, context) -> dict:
        """Which sense(s) the context selects.

        THE HUB IS EXCLUDED. The target word is shared with every sense by construction,
        so gluing through it decides nothing; only the other lemmas carry information.

        Returns every sense tied for the most support, because a tie is the honest answer
        when the structure does not distinguish them. `senses` is empty when nothing in the
        context reaches any sense, which is an abstention rather than a guess.
        """
        ctx = {str(c).lower() for c in context} - {self.word}
        if not self.senses or not ctx:
            return {"senses": [], "support": {}, "tied": False, "abstain": True}
        support = {s: sum(1 for L in ctx if L in self.extent[s]) for s in self.senses}
        best = max(support.values())
        if best == 0:
            return {"senses": [], "support": support, "tied": False, "abstain": True}
        win = [s for s, v in support.items() if v == best]
        return {"senses": win, "support": support, "tied": len(win) > 1,
                "abstain": False}


# the query path
_INVENTORY = None
_INVENTORY_KEY = None
_INVENTORY_RAW = None


def inventory(path=None):
    """Return `(synsets, by_lemma, relations)` for the selected WordNet source.

    Without a path, reuse the registered lexical answerer's loaded records.
    Explicit files are cached by resolved path, size and modification time.
    Changing sources replaces the cached inventory.
    """
    global _INVENTORY, _INVENTORY_KEY, _INVENTORY_RAW
    raw = None
    if path is None:
        try:
            from agent.answerers import _default_registry
            lex = _default_registry().get("lexical")
            if lex is not None:
                lex[0]._lex()
                raw = lex[0]._wn
        except Exception:
            raw = None
    if raw is None:
        from agent.answerers.lexical import DEFAULT_WORDNET
        source = Path(path or DEFAULT_WORDNET).expanduser().resolve()
        stat = source.stat()
        key = (str(source), stat.st_size, stat.st_mtime_ns)
    else:
        key = ("loaded", id(raw))
    if _INVENTORY is None or key != _INVENTORY_KEY:
        if raw is None:
            from agent.adapters.lexical import load_wordnet
            raw = load_wordnet(source)
        lemma = {e: str(w).lower() for e, w in (raw.get("lemma_of") or {}).items()}
        syn = {}
        for sid, members in (raw.get("synsets") or {}).items():
            got = sorted({lemma[m] for m in members if m in lemma})
            if got:
                syn[sid] = got
        by_lemma = {}
        for sid, words in syn.items():
            for w in words:
                by_lemma.setdefault(w, []).append(sid)
        rel = {}
        for src, _kind, tgt in (raw.get("relations") or ()):
            if src in syn and tgt in syn:
                rel.setdefault(src, []).append(tgt)
        _INVENTORY = (syn, by_lemma, rel)
        _INVENTORY_KEY = key
        # Retain the loaded object so its identity cannot be reused by a new source.
        _INVENTORY_RAW = raw
    return _INVENTORY


def sense_expansion(terms, *, hops: int = 1, blind: bool = False, cap: int = 64):
    """Terms a query's senses admit, weighted by the share of the group crossed."""
    syn, by_lemma, rel = inventory()
    seeds = [str(t).lower() for t in terms]
    out: dict[str, tuple] = {}

    def add(term, w, prov):
        cur = out.get(term)
        if cur is None or w > cur[0]:
            out[term] = (w, prov)

    for t in seeds:
        add(t, 1.0, ("query", "asked", t))

    for t in seeds:
        senses = by_lemma.get(t) or []
        if not senses:
            continue
        model = SenseModel.for_word(t, syn, by_lemma, rel, hops=hops)
        if model.d == 0:
            continue
        if blind or model.d == 1:
            chosen = model.senses
        else:
            got = model.disambiguate([s for s in seeds if s != t])
            chosen = got["senses"] or model.senses    # abstain -> no basis to filter
        for s in chosen:
            members = syn.get(s, ())
            k = len(members)
            if k < 2:
                continue
            share = 1.0 / (k - 1)
            for m in list(members)[:cap]:
                m = str(m).lower()
                if m != t:
                    add(m, share, ("wordnet", s, t))
    return out
