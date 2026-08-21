"""WordNet answerer: definitions and typed relations for one term.

`INTERFACE` maps a question head to a WordNet relation name. A query naming no lexical
relation is declined without loading the lexicon.
"""
from __future__ import annotations

import os

from agent.answerers import _question as Q

__all__ = ["LexicalAnswerer", "DEFAULT_WORDNET"]

DEFAULT_WORDNET = os.path.expanduser(
    "~/projects/rexgraph/data/lexical/wordnet/english-wordnet-2024.xml.gz")

#: what a question's head asks for, in this lexicon's own relation names. Declared,
#: because it states what the structure exposes; nothing here is inferred or tuned.
INTERFACE = {
    "mean": "definition", "means": "definition", "meaning": "definition",
    "define": "definition", "definition": "definition", "is": "definition",
    "kind": "hyponym", "kinds": "hyponym", "type": "hyponym", "types": "hyponym",
    "example": "hyponym", "examples": "hyponym",
    "part": "holo_member", "parts": "holo_member",
    "cause": "causes", "causes": "causes",
    "entail": "entails", "entails": "entails",
    "opposite": "antonym", "antonym": "antonym",
}



class LexicalAnswerer:
    """One structure, one question kind, exact or declined.

    `answer(query)` returns `{"answered": bool, ...}`. When it declines it says WHY, which
    is the useful half: "no lexical relation was asked for" and "the lexicon does not hold
    this term" are different facts and a caller routing to another answerer needs both.
    """

    capability = "define"

    def __init__(self, path: str | None = None, *, wn=None):
        self._path = path or DEFAULT_WORDNET
        self._wn = wn
        self._by_lemma: dict | None = None
        self._senses: dict | None = None
        self._verb_forms: set | None = None

    #### the lexicon, loaded once ##############################################
    def _lex(self):
        if self._wn is None:
            from agent.adapters.lexical import load_wordnet
            self._wn = load_wordnet(self._path)
        if self._by_lemma is None:
            by = {}
            for eid, written in (self._wn.get("lemma_of") or {}).items():
                by.setdefault(str(written).lower(), []).append(eid)
            self._by_lemma = by
            # polysemy is the lexicon's own degree: the incidence count of a term in
            # the complex. Picking the lowest is the same 1/deg rule the retrieval
            # uses, rather than a curated list.
            deg = {}
            for members in (self._wn.get("synsets") or {}).values():
                for e in members:
                    deg[e] = deg.get(e, 0) + 1
            self._senses = {lem: sum(deg.get(e, 0) for e in eids)
                            for lem, eids in by.items()}
            # RECORDED INFLECTIONS settle what polysemy alone cannot. WordNet holds `are`
            # as a noun (a unit of area, 100 square metres) with ONE sense, so it beats
            # `whale` on specificity and a question about whales resolves to `are`. But the
            # lexicon also records `are` as an inflected form of a verb entry, and `whale`
            # as a form of nothing. A token appearing as a recorded verb form is being used
            # as a verb, which is a fact the lexicon states rather than a rule about English.
            vforms = set()
            for eid, forms in (self._wn.get("forms_of") or {}).items():
                if (self._wn.get("pos_of") or {}).get(eid) == "v":
                    vforms.update(str(f).lower() for f in forms if f)
            self._verb_forms = vforms
        return self._wn

    def holds(self, term: str) -> bool:
        """Is this term IN the lexicon. A caller routing between structures needs to ask
        this without paying for an answer."""
        self._lex()
        return str(term).lower() in self._by_lemma

    #### the reading ###########################################################
    def _asked(self, tokens):
        """(relation, subject) read off the query, or (None, None).

        The relation comes from the query's HEAD and the subject is the lexicon entry it
        asks about; both readings are shared with every other answerer in `_question`,
        because they are statements about how a question is built rather than about this
        lexicon. What IS specific here is the exclusion: WordNet records inflected forms,
        so a token appearing as a recorded verb form is being used as a verb, a fact the
        lexicon states rather than a rule about English.
        """
        rel = Q.relation_asked(tokens, INTERFACE)
        if rel is None:
            return None, None
        subject, _mentioned = Q.subject_by_degree(
            tokens, INTERFACE, self.holds,
            lambda t: self._senses.get(t, 1 << 30),
            exclude=self._verb_forms | Q.function_words() | Q.interface_words())
        return rel, subject

    def answer(self, query: str) -> dict:
        toks = Q.tokens(query)
        # The INTERFACE check needs no lexicon, and most queries are not lexical, so a
        # non-lexical query must not pay 2.6 s to load one before being declined.
        if not any(t in INTERFACE for t in toks):
            return {"answered": False, "reason": "no lexical relation is asked for",
                    "capability": self.capability}
        self._lex()
        rel, subject = self._asked(toks)
        if rel is None:
            return {"answered": False, "reason": "no lexical relation is asked for",
                    "capability": self.capability}
        if subject is None:
            unknown = [t for t in toks if t not in INTERFACE and t.isalpha()]
            return {"answered": False,
                    "reason": f"the lexicon holds none of {unknown[:6]}",
                    "relation": rel, "capability": self.capability}

        wn = self._wn
        eids = self._by_lemma[subject]
        senses = []
        for sid, members in (wn.get("synsets") or {}).items():
            mine = [e for e in eids if e in members]
            if not mine:
                continue
            entry = {"synset": sid,
                     "pos": sorted({wn["pos_of"].get(e) for e in mine} - {None}),
                     "definition": (wn.get("definitions") or {}).get(sid, ""),
                     "related": []}
            if rel != "definition":
                for src, rtype, tgt in wn.get("relations") or ():
                    if src == sid and rtype == rel:
                        gloss = (wn.get("definitions") or {}).get(tgt)
                        if gloss:
                            entry["related"].append({"synset": tgt, "definition": gloss})
            senses.append(entry)
        if not senses:
            return {"answered": False, "reason": f"{subject!r} has no synset",
                    "relation": rel, "capability": self.capability}
        if rel != "definition" and not any(s["related"] for s in senses):
            # the term is held and the relation is real, but this lexicon records none
            # for it. That is an answer about the lexicon, not a failure to look.
            return {"answered": False,
                    "reason": f"the lexicon records no {rel} for {subject!r}",
                    "relation": rel, "subject": subject,
                    "capability": self.capability}
        return {"answered": True, "relation": rel, "subject": subject,
                "senses": senses, "source": "wordnet",
                "capability": self.capability}

    #### the worker interface the hive already has #############################
    def as_worker(self):
        """`(handler, capability, worker_type)` for `Hive.add_worker`, which is the
        existing primitive for a non-HTTP member invoked like any other bee."""
        def handler(data):
            q = data.get("query") if isinstance(data, dict) else data
            return self.answer(str(q))
        return handler, self.capability, "answerer:lexical"


def render(result: dict) -> str:
    """The answer as text. Extractive and exact: every clause is a lexicon row."""
    if not result.get("answered"):
        return ""
    subj, rel = result["subject"], result["relation"]
    lines = []
    for s in result["senses"]:
        pos = "/".join(s["pos"]) or "?"
        if rel == "definition":
            lines.append(f"{subj} ({pos}): {s['definition']}")
        else:
            for r in s["related"]:
                lines.append(f"{subj} ({pos}) {rel}: {r['definition']}")
        for r in (s["related"] if rel == "definition" else []):
            lines.append(f"    {rel}: {r['definition']}")
    return "\n".join(lines)
