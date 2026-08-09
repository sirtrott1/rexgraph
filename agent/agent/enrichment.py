"""
agent.enrichment: which terms a set of entities is concentrated in.

The standard question. A gene set comes out of an experiment, and the ontology is
asked which of its terms the set favours. Every tool answers it the same way: apply
the true-path rule so that annotating a term annotates its ancestors, count, run a
hypergeometric test per term, correct for multiplicity, sort by p.

That answer is computed here, exactly, because it is the number people know how to
read and refusing to produce it would not be an improvement. It is computed on the
same complex that carries everything else, so the counts come from the joined
structure rather than from a separate annotation table.

Then the same input is read a second way. A p-value is one threshold; a filtration is
every threshold at once. Ordering the complex by how strongly each relation is
implicated and taking persistence gives the structure of the answer: which groupings
appear, how long they survive as the threshold moves, and whether the set is
concentrated in one region or scattered across several. `bottleneck_distance` then
compares two experiments with a stability theorem behind it, which a list of p-values
does not have.

Both readings are returned. The first is comparable to what a reviewer expects; the
second is what the extra structure buys, and keeping them side by side is the only
way to tell whether it bought anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: relation names that place a term under another. Annotations propagate along these.
HIERARCHY = frozenset({"is_a", "isa", "subclassof", "part_of", "partof"})

#: relation names that attach an entity to a term
ANNOTATION = frozenset({"involved_in", "enables", "located_in", "acts_upstream_of",
                        "contributes_to", "colocalizes_with", "part_of",
                        "annotated_with", "is_active_in"})


@dataclass
class AnnotationModel:
    """Terms, entities, and which entity is annotated to which term."""

    terms: list[str]
    entities: list[str]
    direct: dict[str, set] = field(default_factory=dict)      # term -> entities
    parents: dict[str, set] = field(default_factory=dict)     # term -> parent terms
    closed: dict[str, set] = field(default_factory=dict)      # term -> entities, propagated

    @property
    def universe(self) -> set:
        return set(self.entities)


def _relation_name(full: str) -> str:
    """`goa.gaf:involved_in` -> `involved_in`. Provenance is kept on the edge type."""
    return full.split(":")[-1].strip().lower()


def build_annotation_model(knowledge) -> AnnotationModel:
    """Split a joined complex into hierarchy and annotations.

    Both are edges of the same complex and are told apart by their relation, which is
    what the type channel carries. Nothing is re-read from a file.
    """
    terms, entities = set(), set()
    direct: dict[str, set] = {}
    parents: dict[str, set] = {}

    for s, rel, o, _origin in knowledge.edges:
        name = _relation_name(rel)
        subj, obj = knowledge.display(s), knowledge.display(o)
        if name in HIERARCHY:
            parents.setdefault(subj, set()).add(obj)
            terms.add(subj)
            terms.add(obj)
        elif name in ANNOTATION:
            direct.setdefault(obj, set()).add(subj)
            terms.add(obj)
            entities.add(subj)
    return AnnotationModel(sorted(terms), sorted(entities), direct, parents)


def apply_true_path(model: AnnotationModel) -> AnnotationModel:
    """Annotating a term annotates every term above it.

    The true-path rule, and it is not optional: without it a term's count is only its
    direct annotations and every ancestor reads as empty, so the hierarchy the
    ontology exists to provide contributes nothing to the answer.

    Ancestors are accumulated by walking up the parent relation, memoised per term, so
    a deep hierarchy costs one pass rather than one per annotation.
    """
    memo: dict[str, set] = {}

    def ancestors(t: str, seen: frozenset = frozenset()) -> set:
        if t in memo:
            return memo[t]
        if t in seen:                       # a cycle in the hierarchy: stop, do not hang
            return set()
        out = set()
        for p in model.parents.get(t, ()):
            out.add(p)
            out |= ancestors(p, seen | {t})
        memo[t] = out
        return out

    closed: dict[str, set] = {t: set(e) for t, e in model.direct.items()}
    for term, ents in model.direct.items():
        for anc in ancestors(term):
            closed.setdefault(anc, set()).update(ents)
    model.closed = closed
    return model


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def hypergeometric_sf(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(N, K, n). Exact, in log space.

    N is the universe, K the entities annotated to the term, n the study set size, k
    the overlap. Summed from the tail so a tiny probability does not vanish.
    """
    if k <= 0:
        return 1.0
    hi = min(K, n)
    if k > hi:
        return 0.0
    denom = _log_choose(N, n)
    total = 0.0
    for i in range(int(k), int(hi) + 1):
        lp = _log_choose(K, i) + _log_choose(N - K, n - i) - denom
        total += math.exp(lp)
    return min(1.0, max(0.0, total))


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """FDR-adjusted p-values, in the order given."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = m - rank + 1
        val = min(prev, pvalues[idx] * m / i)
        adjusted[idx] = val
        prev = val
    return adjusted


def enrich(knowledge, study_set, *, universe=None, min_term_size: int = 1) -> dict:
    """Which terms the study set is concentrated in.

    Returns the classical answer (counts, hypergeometric p, BH-adjusted q) and the
    structural one (the persistence of the annotation complex restricted to the set)
    over the same complex, so the two can be compared directly.
    """
    model = apply_true_path(build_annotation_model(knowledge))
    study = {str(s) for s in study_set}
    known = model.universe
    background = set(universe) if universe is not None else known
    study_in = study & background
    N, n = len(background), len(study_in)

    rows = []
    for term, ents in model.closed.items():
        annotated = ents & background
        K = len(annotated)
        if K < min_term_size:            # noqa: SIM300 - K is the hypergeometric parameter, not a constant
            continue
        overlap = annotated & study_in
        k = len(overlap)
        if k == 0:
            continue
        p = hypergeometric_sf(k, N, K, n) if N and n else 1.0
        rows.append({
            "term": term, "n_study": k, "n_term": K,
            "expected": (K * n / N) if N else 0.0,
            "fold_enrichment": ((k / n) / (K / N)) if n and K and N else 0.0,
            "p_value": p, "entities": sorted(overlap)[:25],
        })
    for row, q in zip(rows, benjamini_hochberg([r["p_value"] for r in rows]),
                      strict=False):
        row["q_value"] = q
    rows.sort(key=lambda r: (r["p_value"], -r["n_study"]))

    return {
        "n_universe": N,
        "n_study": n,
        "n_study_unmapped": len(study - background),
        "n_terms_tested": len(rows),
        "terms": rows,
        "structure": structural_reading(knowledge, study_in),
        "note": ("p_value is the exact hypergeometric tail after the true-path rule; "
                 "q_value is Benjamini-Hochberg. `structure` reads the same complex "
                 "as a filtration instead of a threshold."),
    }


def structural_reading(knowledge, study_set) -> dict:
    """The same question as a filtration rather than a threshold.

    Each relation is ordered by how much of the study set it touches, and persistence
    over that order records which groupings appear and how long they survive. The
    barcode is comparable between experiments under `bottleneck_distance`, which has
    a stability theorem: a small change in the input moves the diagram by a small
    amount. A ranked p-value list has no such guarantee.
    """
    try:
        rex = knowledge.rex(face_selection="none")
    except ValueError:
        return {"available": False, "reason": "the complex has no relations"}
    study = {str(s) for s in study_set}
    labels = [knowledge.display(c) for c in knowledge.entities]
    in_study = np.array([1.0 if lb in study else 0.0 for lb in labels])
    if not in_study.any():
        return {"available": False,
                "reason": "no member of the study set is in the complex"}

    S, T = np.asarray(rex.sources), np.asarray(rex.targets)
    # a relation is reached early when it touches the study set at both ends
    filt_e = 2.0 - (in_study[S] + in_study[T])
    filt_v = 1.0 - in_study
    filt_f = np.zeros(int(rex.nF)) if int(rex.nF) else np.zeros(0)
    try:
        result = rex.persistence(filt_v.astype(np.float64),
                                 filt_e.astype(np.float64),
                                 filt_f.astype(np.float64))
        barcodes = np.asarray(rex.persistence_barcodes(result))
        entropy = float(rex.persistence_entropy(barcodes))
    except Exception as e:                       # noqa: BLE001 - reported, not raised
        return {"available": False, "reason": str(e)[:200]}

    # The kernel classifies bars itself: `pairs` are the ones that die, `essential`
    # are the ones that do not, and an essential death is written as 1e308 rather than
    # numpy's inf. Reading the classification is exact; sniffing the magnitude is a
    # guess about a sentinel, and it let an essential bar report a lifetime of 1e308.
    pairs = np.asarray(result.get("pairs"), dtype=np.float64).reshape(-1, 5)
    essential = np.asarray(result.get("essential"), dtype=np.float64).reshape(-1, 3)
    lifetimes = (pairs[:, 3] - pairs[:, 1]) if pairs.size else np.zeros(0)
    return {
        "available": True,
        "n_features": int(pairs.shape[0] + essential.shape[0]),
        "n_essential": int(essential.shape[0]),
        "persistence_entropy": entropy,
        "longest_lifetime": float(lifetimes.max()) if lifetimes.size else 0.0,
        "barcodes": [[float(b), float(dth)] for b, dth in
                     zip(pairs[:200, 1], pairs[:200, 3], strict=False)],
        "reading": ("features appearing early and surviving long are groupings the "
                    "study set holds together across every threshold, not only at a "
                    "chosen cut"),
    }


def compare(reading_a: dict, reading_b: dict) -> dict:
    """Distance between two structural readings.

    Bottleneck distance between the two diagrams. Stable: a bounded perturbation of
    the input moves this by a bounded amount, which is what makes comparing two
    experiments meaningful rather than anecdotal.
    """
    from rexgraph.graph import RexGraph

    a = np.asarray(reading_a.get("barcodes") or [], dtype=np.float64)
    b = np.asarray(reading_b.get("barcodes") or [], dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {"available": False, "reason": "one reading has no finite features"}
    return {
        "available": True,
        "bottleneck": float(RexGraph.persistence_distance(a, b, metric="bottleneck")),
        "wasserstein": float(RexGraph.persistence_distance(a, b,
                                                           metric="wasserstein")),
    }
