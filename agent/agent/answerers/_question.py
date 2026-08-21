"""Reading a question: which relation it asks for, and which term it asks about.

Shared by every answerer. A word naming a relation wins over a frame word; the
interrogative decides whether a frame word counts at all; a MENTION frame names its own
subject. Where a structural fact is available it is used instead, in `subject_by_degree`.
"""
from __future__ import annotations

import os

__all__ = ["tokens", "relation_asked", "subject_by_degree", "function_words",
           "interface_words",
           "FRAME", "NON_PREDICATIVE", "MENTION"]

#: words that open a question without naming a relation.
FRAME = ("is", "are", "was", "were", "what", "which", "does", "do", "did")

#: interrogatives that ask for something predication cannot supply.
NON_PREDICATIVE = ("where", "when", "who", "whom", "whose", "how")

#: frames that quote the term they are about.
MENTION = ("word", "term", "phrase", "spelling")


_FUNCTION: frozenset | None = None


def function_words() -> frozenset:
    """The function words, as the lexicon's own recorded parts of speech define them.

    Empty when no lexicon is on disk, and an empty gate excludes nothing.
    """
    global _FUNCTION
    if _FUNCTION is None:
        _FUNCTION = frozenset()
        try:
            from agent.adapters import wiktionary as WK
            from agent.answerers.linkage import DEFAULT_WIKTIONARY
            if os.path.exists(DEFAULT_WIKTIONARY):
                idx = WK.read_index(DEFAULT_WIKTIONARY, verify=False)
                _FUNCTION = WK.function_words(idx, exclusive=False)
        except Exception:
            # The gate is an optional refinement: with no lexicon on disk it stays
            # empty, and an empty gate excludes nothing.
            pass
    return _FUNCTION


_INTERFACE_WORDS: frozenset | None = None


def interface_words() -> frozenset:
    """Every word that names a relation to some structure, from the registered interfaces."""
    global _INTERFACE_WORDS
    if _INTERFACE_WORDS is None:
        import importlib
        words = set(FRAME) | set(NON_PREDICATIVE) | set(MENTION)
        for mod in ("lexical", "linkage", "ontology"):
            try:
                m = importlib.import_module(f"agent.answerers.{mod}")
            except Exception:
                # An answerer whose optional dependency is absent contributes no
                # interface words. The frame vocabulary above does not depend on it.
                continue
            words |= set(getattr(m, "INTERFACE", {}) or {})
        _INTERFACE_WORDS = frozenset(words)
    return _INTERFACE_WORDS


def tokens(query: str) -> list[str]:
    from rexgraph.corpus_profile import TEXT, tokenize
    return [w for w, _a, _b in tokenize(str(query or ""), TEXT)]


def relation_asked(toks, interface: dict, *, frame=FRAME,
                   non_predicative=NON_PREDICATIVE):
    """Which relation of `interface` this question asks for, or None.

    A named relation wins. If only a frame word fired, the interrogative decides whether
    the question is of the kind this interface can answer at all.
    """
    named = next((interface[t] for t in toks if t in interface and t not in frame), None)
    if named is not None:
        return named
    if any(t in non_predicative for t in toks):
        return None
    return next((interface[t] for t in toks if t in interface), None)


def subject_by_degree(toks, interface: dict, holds, degree, *, exclude=()):
    """The query term with the fewest groups in the source, which is its subject.

    Degree means different things in different sources, so `degree` is supplied by
    the caller rather than assumed here.
    """
    for i, t in enumerate(toks[:-1]):
        if t in MENTION:
            nxt = toks[i + 1]
            return (nxt if holds(nxt) else None), True
    cands = [t for t in toks
             if t not in interface and t not in MENTION and t not in exclude
             and holds(t)]
    if not cands:
        return None, False
    return min(cands, key=lambda t: degree(t)), False
