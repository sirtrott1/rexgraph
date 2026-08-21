"""Ontology answerer over a `ParsedOntology`.

Subsumption is answered as the transitive closure with the axiom chain that reaches each
class. Other predicates are answered in their own name and direction. Predicate roles come
from `ontology_reasoning.classify_predicate`.
"""
from __future__ import annotations

import numpy as np

from agent.answerers import _question as Q

__all__ = ["OntologyAnswerer"]

#: what a question's head asks of an ontology, in role terms. Declared interface: it
#: states which questions this structure can be asked, not how to guess an answer.
INTERFACE = {
    "is": "parents", "what": "parents", "define": "parents", "definition": "parents",
    "ancestor": "parents", "ancestors": "parents", "parent": "parents",
    "parents": "parents", "superclass": "parents",
    "kind": "children", "kinds": "children", "subclass": "children",
    "subclasses": "children", "child": "children", "children": "children",
    "descendant": "children", "descendants": "children", "type": "children",
    "types": "children",
    "part": "relations", "parts": "relations", "related": "relations",
    "relation": "relations", "relations": "relations",
}



class OntologyAnswerer:
    """One ontology, the questions its own predicates support, exact or declined."""

    capability = "classify"

    def __init__(self, parsed=None, *, name: str = "ontology"):
        self._parsed = parsed
        self._name = name
        self._rc = None
        self._by_label: dict | None = None

    #### the ontology, built once ##############################################
    @classmethod
    def from_file(cls, path, *, name: str | None = None):
        """Read any format the adapter layer supports and answer over it."""
        from agent.adapters import ontology_formats as fmt
        import os
        ext = os.path.splitext(str(path))[1].lower()
        reader = {".obo": fmt.parse_obo, ".json": fmt.parse_obograph,
                  ".nt": fmt.parse_ntriples, ".ttl": fmt.parse_turtle,
                  ".owl": fmt.parse_rdfxml, ".rdf": fmt.parse_rdfxml,
                  ".gaf": fmt.parse_gaf}.get(ext, fmt.parse_obo)
        return cls(reader(path), name=name or os.path.basename(str(path)))

    def _built(self):
        if self._rc is None:
            if self._parsed is None:
                return None
            from agent import ontology_reasoning as orx
            self._rc = orx.build(self._parsed.triples)
            labels = getattr(self._parsed, "labels", {}) or {}
            by = {}
            for v, term in enumerate(self._rc.labels):
                by.setdefault(str(term).lower(), v)
                human = labels.get(term)
                if human:
                    by.setdefault(str(human).lower(), v)
            self._by_label = by
        return self._rc

    def holds(self, term: str) -> bool:
        rc = self._built()
        return bool(rc) and str(term).lower() in self._by_label

    def _display(self, v: int) -> str:
        """The human name where the ontology gives one, so an answer reads
        'apoptotic process' rather than GO:0006915."""
        rc = self._rc
        term = rc.labels[v] if 0 <= v < len(rc.labels) else str(v)
        human = (getattr(self._parsed, "labels", {}) or {}).get(term)
        return f"{human} ({term})" if human and human != term else str(term)

    #### the reading ###########################################################
    def _subject(self, toks):
        """The term this ontology holds. A multi-word term is matched whole first,
        because "apoptotic process" is one concept and its words separately are not."""
        low = " ".join(toks)
        best = None
        for label, v in self._by_label.items():
            if " " in label and label in low:
                if best is None or len(label) > len(best[0]):
                    best = (label, v)
        if best is not None:
            return best[1]
        for t in toks:
            if t not in INTERFACE and t in self._by_label:
                return self._by_label[t]
        return None

    def answer(self, query: str) -> dict:
        from agent import ontology_reasoning as orx

        toks = Q.tokens(query)
        # the INTERFACE check needs no ontology, so a non-ontological query is declined
        # without paying to build one.
        if not any(t in INTERFACE for t in toks):
            return {"answered": False, "reason": "no ontology relation is asked for",
                    "capability": self.capability}
        rc = self._built()
        if rc is None:
            return {"answered": False, "reason": "no ontology is loaded",
                    "capability": self.capability}
        want = Q.relation_asked(toks, INTERFACE)
        if want is None:
            return {"answered": False, "reason": "no ontology relation is asked for",
                    "capability": self.capability}
        subject = self._subject(toks)
        if subject is None:
            unknown = [t for t in toks if t not in INTERFACE and t.isalpha()]
            return {"answered": False,
                    "reason": f"this ontology holds none of {unknown[:6]}",
                    "asked": want, "capability": self.capability}

        results = []
        if want in ("parents", "children"):
            # the transitive closure, with the chain of axioms that reaches each class.
            # A hierarchy answer without its chain is not checkable, and the chain is
            # what the closure already returns.
            reach = orx.subsumption_closure(rc, subject, up=(want == "parents"))
            for v, chain in sorted(reach.items(), key=lambda kv: len(kv[1])):
                if v == subject:
                    continue
                results.append({"term": self._display(v), "steps": len(chain) - 1,
                                "via": [self._display(c) for c in chain[1:-1]]})
        else:
            S, T = np.asarray(rc.rex.sources), np.asarray(rc.rex.targets)
            for e, role in enumerate(rc.roles):
                if role == "subsumption":
                    continue
                s_, t_ = int(S[e]), int(T[e])
                other = t_ if s_ == subject else (s_ if t_ == subject else None)
                if other is not None:
                    results.append({"term": self._display(other),
                                    "predicate": rc.predicates[e],
                                    "role": role,
                                    "direction": "out" if s_ == subject else "in"})
        if not results:
            return {"answered": False,
                    "reason": f"the ontology asserts no {want} for "
                              f"{self._display(subject)!r}",
                    "asked": want, "subject": self._display(subject),
                    "capability": self.capability}
        return {"answered": True, "asked": want, "subject": self._display(subject),
                "results": results, "source": self._name,
                "capability": self.capability}

    def as_worker(self):
        def handler(data):
            q = data.get("query") if isinstance(data, dict) else data
            return self.answer(str(q))
        return handler, self.capability, "answerer:ontology"


def render(result: dict) -> str:
    """The answer as text. Every line is an asserted axiom or a chain of them."""
    if not result.get("answered"):
        return ""
    subj, asked = result["subject"], result["asked"]
    lines = []
    for r in result["results"]:
        if asked == "parents":
            via = "  via " + " < ".join(r["via"]) if r.get("via") else ""
            lines.append(f"{subj} is a {r['term']}{via}")
        elif asked == "children":
            via = "  via " + " < ".join(reversed(r["via"])) if r.get("via") else ""
            lines.append(f"{r['term']} is a kind of {subj}{via}")
        else:
            arrow = "->" if r["direction"] == "out" else "<-"
            lines.append(f"{subj} {arrow} {r['predicate']} {arrow} {r['term']}")
    return "\n".join(lines)
