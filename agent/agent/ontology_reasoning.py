"""
agent.ontology_reasoning: classification, consistency and module extraction, exactly.

A description-logic reasoner answers three questions: are these axioms consistent,
which classes are equivalent, and what does this fragment of the ontology contribute.
It answers them with a tableau, and when the answer is "inconsistent" what comes back
is a refutation a person cannot read.

Each question is an exact integer invariant of the relational complex, and the object
that carries the answer also carries where it came from.

**Disjointness is a sign, and orientation decides.** `disjointWith` says two classes
cannot share an instance, which is a negative relation and therefore a sign. The sign
alone does not settle consistency, and it is worth being exact about why: both of
these close a negative cycle, and only the first is a contradiction.

    A ⊑ B,  A ⊑ C,  B disjointWith C     A is a common SUBclass of two disjoint
                                         classes, so A can have no instance
    B ⊑ D,  C ⊑ D,  B disjointWith C     two disjoint siblings under one parent,
                                         the most ordinary shape there is

Undirected they are the same triangle. What separates them is that the subsumption
edges LEAVE the common vertex in the first and ENTER it in the second. So the sign
marks where to look and the orientation of the boundary decides: a class is
unsatisfiable exactly when it lies below two classes asserted disjoint. That is
descent in the subsumption order, read off the directed boundary, and it is exact.

Holonomy is still reported, because balance of the signed complex is a real
gauge-invariant quantity and a frustrated cycle localises where a disjointness meets
the hierarchy. It is simply not the same question as consistency.

**Classification is a quotient.** Two classes are equivalent when they are
indistinguishable once the subsumption subcomplex is collapsed, which is congruence
modulo that subcomplex. `quotient_analysis` computes the classes.

**Module extraction is relative homology.** H(R, I) is what the fragment contributes
that the rest does not, computed over the integers rather than by syntactic locality.

Nothing here approximates. Every number is an integer invariant or a sign product, so
a result can be read back to the axioms that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: predicates asserting that two classes cannot share an instance. These carry the
#: negative sign, and they are the entire source of inconsistency in this reading.
DISJOINT_PREDICATES = frozenset({
    "disjointwith", "disjoint_from", "disjointfrom", "disjointclasses",
    "complementof", "differentfrom", "not",
})

#: predicates that place one class under another. The hierarchy.
SUBSUMPTION_PREDICATES = frozenset({
    "subclassof", "is_a", "isa", "subsumes", "subpropertyof", "subclass",
})

#: predicates asserting two classes are the same thing
EQUIVALENCE_PREDICATES = frozenset({
    "equivalentclass", "equivalentto", "sameas", "equivalent",
})

#: restriction predicates: `C ⊑ ∃R.D` reaches D through a named relation R. In a
#: typed complex that is an edge of type R, which is what the parsers already emit,
#: so an existential needs no encoding of its own.
EXISTENTIAL_PREDICATES = frozenset({
    "somevaluesfrom", "allvaluesfrom", "hasvalue", "onproperty",
})


def _local(p: str) -> str:
    return str(p).split("#")[-1].split("/")[-1].split(":")[-1].strip().lower()


def classify_predicate(p: str) -> str:
    """Which role a predicate plays: subsumption, equivalence, disjoint, or relation."""
    pl = _local(p)
    if pl in SUBSUMPTION_PREDICATES:
        return "subsumption"
    if pl in EQUIVALENCE_PREDICATES:
        return "equivalence"
    if pl in DISJOINT_PREDICATES:
        return "disjoint"
    if pl in EXISTENTIAL_PREDICATES:
        return "existential"
    return "relation"


@dataclass
class ReasoningComplex:
    """An ontology as a signed, typed complex, with the masks the questions need."""

    rex: object
    labels: list[str]
    predicates: list[str]                       # per edge, the predicate as written
    roles: list[str]                            # per edge, classify_predicate result
    meta: dict = field(default_factory=dict)

    @property
    def subsumption_mask(self) -> np.ndarray:
        return np.array([r == "subsumption" for r in self.roles], np.uint8)

    @property
    def disjoint_mask(self) -> np.ndarray:
        return np.array([r == "disjoint" for r in self.roles], np.uint8)

    def name(self, v: int) -> str:
        return self.labels[v] if 0 <= v < len(self.labels) else str(v)


def build(triples) -> ReasoningComplex:
    """A signed typed complex from (subject, predicate, object) triples.

    A disjointness gets sign −1 and everything else +1. That single choice is what
    makes consistency a holonomy question rather than a search.
    """
    from rexgraph.graph import RexGraph

    labels: list[str] = []
    idx: dict[str, int] = {}

    def vid(x):
        if x not in idx:
            idx[x] = len(labels)
            labels.append(x)
        return idx[x]

    src, tgt, signs, preds, roles = [], [], [], [], []
    types: dict[str, int] = {}
    type_labels = []
    for s, p, o in triples:
        if s == o:
            continue
        role = classify_predicate(p)
        src.append(vid(s))
        tgt.append(vid(o))
        signs.append(-1.0 if role == "disjoint" else 1.0)
        preds.append(str(p))
        roles.append(role)
        key = _local(p)
        if key not in types:
            types[key] = len(types)
        type_labels.append(types[key])
    if not src:
        raise ValueError("no relations between distinct terms")

    rex = RexGraph(sources=np.asarray(src, np.int32),
                   targets=np.asarray(tgt, np.int32),
                   signs=np.asarray(signs, np.float64))
    # types are attached rather than constructed: RexGraph takes the boundary, and
    # the typing is a labelling of its edges.
    rex.type_labels = np.asarray(type_labels, np.int32)
    meta = {
        "vertex_labels": labels,
        "input_type": "ontology",
        "type_names": [k for k, _ in sorted(types.items(), key=lambda kv: kv[1])],
        "n_disjoint": int(sum(1 for r in roles if r == "disjoint")),
        "n_subsumption": int(sum(1 for r in roles if r == "subsumption")),
    }
    rex._agent_meta = meta
    return ReasoningComplex(rex, labels, preds, roles, meta)


#### consistency


def _descendants(rc: ReasoningComplex, root: int) -> dict[int, list[int]]:
    """Every class at or below `root`, with the chain that reaches it.

    An edge's source is the child and its target the parent, which is the direction
    `is_a` and `subClassOf` are written in. Returned as class -> the chain up to
    `root`, so a finding reads as the axioms that produced it rather than as a set.
    """
    children: dict[int, list[int]] = {}
    S = np.asarray(rc.rex.sources)
    T = np.asarray(rc.rex.targets)
    for e, role in enumerate(rc.roles):
        if role == "subsumption":
            children.setdefault(int(T[e]), []).append(int(S[e]))
    seen = {root: [root]}
    stack = [root]
    while stack:
        v = stack.pop()
        for c in children.get(v, ()):
            if c not in seen:
                seen[c] = [c, *seen[v]]
                stack.append(c)
    return seen


def consistency(rc: ReasoningComplex, *, limit: int = 50) -> dict:
    """Which classes cannot have an instance, and which axioms make it so.

    A class is unsatisfiable exactly when it descends from two classes asserted
    disjoint: everything below `B` is a `B`, everything below `C` is a `C`, and
    nothing is both. Disjointness edges are found by their sign; descent is read off
    the directed subsumption boundary. Both are exact.

    A disjointness between a class and its own ancestor is the degenerate case of the
    same rule and reports identically.

    The answer names the class and both chains, because "unsatisfiable" without the
    two paths that made it so is not something a curator can act on.
    """
    S, T = np.asarray(rc.rex.sources), np.asarray(rc.rex.targets)
    findings = []
    for e, role in enumerate(rc.roles):
        if role != "disjoint":
            continue
        b, c = int(S[e]), int(T[e])
        below_b, below_c = _descendants(rc, b), _descendants(rc, c)
        for x in sorted(set(below_b) & set(below_c)):
            findings.append({
                "unsatisfiable_class": rc.name(x),
                "disjoint_pair": [rc.name(b), rc.name(c)],
                "path_to_first": [rc.name(v) for v in below_b[x]],
                "path_to_second": [rc.name(v) for v in below_c[x]],
                "summary": (f"{rc.name(x)} is below both {rc.name(b)} and "
                            f"{rc.name(c)}, which are asserted disjoint, so it can "
                            f"have no instance."),
            })
    return {
        "consistent": not findings,
        "n_unsatisfiable": len(findings),
        "unsatisfiable": findings[:limit],
        "n_disjointness_axioms": rc.meta["n_disjoint"],
        "holonomy": frustration(rc),
        "method": ("a class is unsatisfiable exactly when it descends from two "
                   "classes asserted disjoint; disjointness by sign, descent by the "
                   "directed subsumption boundary"),
    }


def frustration(rc: ReasoningComplex) -> dict:
    """Sign holonomy over a cycle basis: where a disjointness meets the hierarchy.

    A cycle whose edge signs multiply to -1 is frustrated. Gauge-invariant and real,
    but NOT consistency: two disjoint siblings under a shared parent frustrate a
    cycle and assert nothing contradictory. Reported because it localises the
    interaction between the disjointness axioms and the hierarchy.
    """
    signs = np.asarray(rc.rex._edge_signs, dtype=np.float64)
    n_cycles = n_frustrated = 0
    for column in rc.rex.cycle_basis:
        col = np.asarray(column, dtype=np.float64).ravel()
        support = np.nonzero(np.abs(col) > 1e-12)[0]
        if support.size == 0:
            continue
        n_cycles += 1
        if float(np.prod(signs[support])) < 0:
            n_frustrated += 1
    return {"n_independent_cycles": n_cycles, "n_frustrated": n_frustrated,
            "balanced": n_frustrated == 0,
            "frustrated_fraction": (n_frustrated / n_cycles) if n_cycles else 0.0}


#### classification


def equivalence_classes(rc: ReasoningComplex) -> dict:
    """Classes indistinguishable once the subsumption hierarchy is collapsed.

    Congruence modulo the subsumption subcomplex: two cells are congruent when the
    quotient cannot tell them apart. That is classification, computed rather than
    searched.
    """
    rex = rc.rex
    mask = rc.subsumption_mask
    if not mask.any():
        return {"n_classes": 0, "classes": [],
                "note": "no subsumption axioms, so there is no hierarchy to collapse"}
    labels, n_classes = rex.congruence_classes(mask, 1)
    labels = np.asarray(labels)
    groups: dict[int, list[str]] = {}
    S, T = np.asarray(rex.sources), np.asarray(rex.targets)
    for e, c in enumerate(labels):
        if int(c) < 0:
            continue
        groups.setdefault(int(c), []).append(
            f"{rc.name(int(S[e]))} {rc.predicates[e]} {rc.name(int(T[e]))}")
    return {
        "n_classes": int(n_classes),
        "classes": [{"class": c, "members": m}
                    for c, m in sorted(groups.items()) if len(m) > 1],
        "n_collapsed": int(mask.sum()),
    }


def classification(rc: ReasoningComplex) -> dict:
    """The full relative reading of the ontology modulo its own hierarchy.

    `betti_rel` is the homology of the pair: what the non-hierarchical axioms
    contribute that the hierarchy does not already account for.
    """
    rex = rc.rex
    mask = rc.subsumption_mask
    if not mask.any():
        return {"note": "no subsumption axioms"}
    out = rex.quotient_analysis(mask)
    return {
        "betti_relative": [int(b) for b in out["betti_rel"]],
        "dims": [int(d) for d in out["dims"]],
        "n_congruence_classes": int(out["n_congruence_classes"]),
        "relative_cycle_dim": int(out["rel_cycle_dim"]),
        "hodge_full": _hodge_pcts(out["hodge_R"]),
        "hodge_quotient": _hodge_pcts(out["hodge_RI"]),
        "chain_valid": bool(out["chain_ok"]),
        "reading": ("betti_relative is H(R, I): what the non-hierarchical axioms add "
                    "beyond the subsumption hierarchy alone"),
    }


def _hodge_pcts(h) -> dict:
    """The gradient/curl/harmonic split, whichever shape the kernel returned."""
    if isinstance(h, dict):
        return {k: float(v) for k, v in h.items()
                if isinstance(v, (int, float, np.floating))}
    arr = np.asarray(h, dtype=np.float64).ravel()
    return dict(zip(("gradient", "curl", "harmonic"),
                    (float(x) for x in arr), strict=False))


def module_extraction(rc: ReasoningComplex, terms) -> dict:
    """What a set of terms contributes beyond the rest of the ontology.

    The signature-based module a reasoner extracts by syntactic locality is here the
    relative homology of the pair (whole, complement), which is exact and does not
    depend on a locality notion.
    """
    rex = rc.rex
    wanted = {str(t) for t in terms}
    keep = np.array([1 if (rc.name(int(s)) in wanted or rc.name(int(t)) in wanted)
                     else 0 for s, t in zip(np.asarray(rex.sources),
                                            np.asarray(rex.targets), strict=False)],
                    np.uint8)
    if not keep.any():
        return {"n_axioms": 0, "terms": sorted(wanted),
                "note": "no axiom mentions any of these terms"}
    complement = (1 - keep).astype(np.uint8)
    out = rex.quotient_analysis(complement)
    S, T = np.asarray(rex.sources), np.asarray(rex.targets)
    axioms = [f"{rc.name(int(S[e]))} {rc.predicates[e]} {rc.name(int(T[e]))}"
              for e in np.nonzero(keep)[0]]
    return {
        "terms": sorted(wanted),
        "n_axioms": int(keep.sum()),
        "axioms": axioms[:200],
        "betti_relative": [int(b) for b in out["betti_rel"]],
        "reading": ("betti_relative is what these terms contribute that the rest of "
                    "the ontology does not already imply"),
    }


#### cardinality, read off the typed star


def cardinality(rc: ReasoningComplex, *, limit: int = 100) -> dict:
    """Per-class relation counts, by predicate.

    A cardinality restriction is a statement about how many times a class stands in a
    named relation. The complex holds that directly as the typed degree of the class's
    star, so checking one is a count rather than a proof obligation.
    """
    rex = rc.rex
    S = np.asarray(rex.sources)
    per: dict[str, dict[str, int]] = {}
    for e in range(int(rex.nE)):
        per.setdefault(rc.name(int(S[e])), {})
        key = _local(rc.predicates[e])
        per[rc.name(int(S[e]))][key] = per[rc.name(int(S[e]))].get(key, 0) + 1
    rows = [{"class": c, "counts": v, "total": sum(v.values())}
            for c, v in per.items()]
    rows.sort(key=lambda r: -r["total"])
    return {"n_classes": len(rows), "classes": rows[:limit]}


#### one call


def reason(triples, *, terms=None) -> dict:
    """Every question at once, over one complex."""
    rc = build(triples)
    out = {
        "n_terms": len(rc.labels),
        "n_axioms": int(rc.rex.nE),
        "betti": [int(b) for b in rc.rex.betti],
        "consistency": consistency(rc),
        "equivalence": equivalence_classes(rc),
        "classification": classification(rc),
        "cardinality": cardinality(rc),
    }
    if terms:
        out["module"] = module_extraction(rc, terms)
    return out
