"""Roget and Wiktionary answerer.

Both sources are labelled groups over words, so one class serves both. Roget records
undirected thematic categories; Wiktionary records ten directed link kinds, kept separate
because merging synonyms with antonyms asserts what neither says.

Reads the source files rather than the stored complexes: `build_lexical_store` drops the
per-group label, so a stored column carries no relation type.
"""
from __future__ import annotations

import os

from agent.answerers import _question as Q

__all__ = ["LinkageAnswerer", "DEFAULT_ROGET", "DEFAULT_WIKTIONARY"]

_ROOT = "~/projects/rexgraph/data/lexical"
DEFAULT_ROGET = os.path.expanduser(f"{_ROOT}/thesaurus/rogets-pg22.txt")
DEFAULT_WIKTIONARY = os.path.expanduser(f"{_ROOT}/wiktionary/wiktionary-en.rexidx")

#: question head -> the kind of link it asks for, in the sources' own kind names. Declared
#: interface. A source keeps only the kinds it actually records, so asking Roget for
#: antonyms declines rather than inventing.
INTERFACE = {
    "synonym": "synonyms", "synonyms": "synonyms", "same": "synonyms",
    "means": "synonyms", "mean": "synonyms",
    "opposite": "antonyms", "antonym": "antonyms", "antonyms": "antonyms",
    "kind": "hyponyms", "kinds": "hyponyms", "type": "hyponyms",
    "types": "hyponyms", "example": "hyponyms", "examples": "hyponyms",
    "is": "hypernyms", "what": "hypernyms",
    "part": "meronyms", "parts": "meronyms",
    "belongs": "holonyms", "whole": "holonyms",
    "derived": "derived", "from": "derived",
    "related": "related", "relate": "related", "associated": "related",
    "about": "related", "like": "coordinate_terms",
}


class LinkageAnswerer:
    """One source of typed word groups, answered in its own kind names."""

    capability = "relate"

    def __init__(self, *, name, kinds, loader, max_terms: int = 12):
        self._name = name
        self._kinds = tuple(kinds)
        self._loader = loader
        self._max = int(max_terms)
        self._by_word: dict | None = None       # word -> [(kind, label, members)]

    #### the two sources ########################################################
    @classmethod
    def roget(cls, path: str | None = None):
        def load():
            from agent.adapters import lexical as L
            out: dict = {}
            for _num, cat in L.load_roget(path or DEFAULT_ROGET).items():
                terms = list(cat["terms"])
                if len(terms) < 2:
                    continue
                entry = ("category", str(cat["name"]), terms)
                for t in terms:
                    out.setdefault(t.lower(), []).append(entry)
            return out
        # Roget records ONE relation: thematic co-membership. It is reported as
        # `related` and never as a hierarchy, because a category asserts no direction.
        return cls(name="roget", kinds=("related",), loader=load)

    @classmethod
    def wiktionary(cls, path: str | None = None):
        def load():
            from agent.adapters import wiktionary as WK
            idx = WK.read_index(path or DEFAULT_WIKTIONARY, verify=False)
            groups, labels = WK.wiktionary_groups(idx)
            out: dict = {}
            for g, (kind, head) in zip(groups, labels, strict=True):
                entry = (str(kind), str(head), list(g))
                for t in g:
                    out.setdefault(str(t).lower(), []).append(entry)
            return out
        from agent.adapters.wiktionary import LINK_KINDS
        return cls(name="wiktionary", kinds=LINK_KINDS, loader=load)

    #### the structure, loaded once #############################################
    def _index(self):
        if self._by_word is None:
            self._by_word = self._loader()
        return self._by_word

    def holds(self, term: str) -> bool:
        return str(term).lower() in self._index()

    def _degree(self, term: str) -> int:
        """How many groups this word participates in: its degree in this source. A word
        in many groups is a general one, which is what picks the question's subject."""
        return len(self._index().get(str(term).lower(), ()))

    #### the reading ############################################################
    def answer(self, query: str) -> dict:
        toks = Q.tokens(query)
        # the interface check needs no source, so a query naming no linkage never pays
        # to load one.
        if not any(t in INTERFACE for t in toks):
            return {"answered": False, "reason": "no linkage is asked for",
                    "capability": self.capability}
        want = Q.relation_asked(toks, INTERFACE)
        if want is None:
            return {"answered": False, "reason": "no linkage is asked for",
                    "capability": self.capability}
        if want not in self._kinds:
            return {"answered": False,
                    "reason": f"{self._name} records no {want}",
                    "asked": want, "capability": self.capability}
        self._index()
        subject, mentioned = Q.subject_by_degree(toks, INTERFACE, self.holds,
                                                 self._degree,
                                                 exclude=Q.function_words() | Q.interface_words())
        if subject is None:
            unknown = [t for t in toks if t not in INTERFACE and t.isalpha()]
            why = (f"{self._name} does not hold the term you named"
                   if mentioned else f"{self._name} holds none of {unknown[:6]}")
            return {"answered": False, "reason": why, "asked": want,
                    "capability": self.capability}

        groups = [(k, lab, mem) for k, lab, mem in self._index()[subject]
                  if k == want or (want == "related" and k == "category")]
        if not groups:
            have = sorted({k for k, _l, _m in self._index()[subject]})
            return {"answered": False,
                    "reason": f"{self._name} records no {want} for {subject!r}"
                              + (f"; it records {have}" if have else ""),
                    "asked": want, "subject": subject,
                    "capability": self.capability}

        # ORIENTATION IS THE CONTENT. `wiktionary_groups` puts the head at position 0
        # because "synonyms of x" is a statement ABOUT x, so a group whose head is the
        # subject answers the question asked, and a group that merely LISTS the subject
        # answers its converse. Reporting the second as the first is what made "synonyms
        # of grief" reply with the synonyms of every head word that mentions grief.
        # A Roget category has no head and asserts no direction, so it is neither.
        out = []
        for kind, label, members in groups:
            others = [m for m in members if str(m).lower() != subject]
            if not others:
                continue
            direction = ("undirected" if kind == "category"
                         else "of" if str(label).lower() == subject else "converse")
            out.append({"kind": kind, "label": label, "direction": direction,
                        "terms": others[:self._max], "n_terms": len(others)})
        # the direct reading first: a question about x is answered by the groups that are
        # about x, and the converse is context beneath it, never in place of it.
        out.sort(key=lambda g: {"of": 0, "undirected": 1, "converse": 2}[g["direction"]])
        if not out:
            return {"answered": False,
                    "reason": f"{subject!r} is alone in every {want} group",
                    "asked": want, "subject": subject,
                    "capability": self.capability}
        return {"answered": True, "asked": want, "subject": subject,
                "groups": out, "source": self._name,
                "capability": self.capability}

    def as_worker(self):
        def handler(data):
            q = data.get("query") if isinstance(data, dict) else data
            return self.answer(str(q))
        return handler, self.capability, f"answerer:{self._name}"


def render(result: dict) -> str:
    """The answer as text. Each line is one recorded group, named by its own type."""
    if not result.get("answered"):
        return ""
    subj = result["subject"]
    lines = []
    for g in result["groups"]:
        more = f" (+{g['n_terms'] - len(g['terms'])} more)" if g["n_terms"] > len(g["terms"]) else ""
        if g["direction"] == "undirected":
            # a Roget category asserts co-membership, with no direction to report.
            lines.append(f"{subj} is grouped under {g['label']!r} with: "
                         f"{', '.join(g['terms'])}{more}")
        elif g["direction"] == "of":
            lines.append(f"{g['kind']} of {subj}: {', '.join(g['terms'])}{more}")
        else:
            # the converse: the source records the subject among ANOTHER word's links,
            # which is a different statement and is said as one.
            lines.append(f"{subj} is listed among the {g['kind']} of "
                         f"{g['label']}{more}")
    return "\n".join(lines)
