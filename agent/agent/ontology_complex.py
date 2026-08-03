"""
agent.ontology_complex: RDFS/OWL ontologies as typed relational complexes.

The mapping follows arity ≠ grade:
  * ``rdfs:subClassOf`` (binary A ⊑ B) -> a **gradient edge** (the class
    hierarchy is the gradient / order of operations).
  * an ``owl:ObjectProperty`` (domain->range) -> a typed **edge**.
  * ``owl:equivalentClass`` / ``owl:sameAs`` (a mutual definition) -> a
    **bigon face** (bounded).
  * ``owl:inverseOf`` / ``owl:SymmetricProperty`` -> a **bigon face** (an
    intentional 2-cycle, bounded - not an inconsistency).
  * a definition over several relations (intersectionOf: C ≡ A ⊓ B) -> a
    **k-gon face** over the participants.

Diagnosis is the same descriptive Hodge readout as schemas: subsumption is the
gradient hierarchy, definitions are bounded (curl), and a subsumption cycle with
no defining face is harmonic - an inconsistency, reported as present, not judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_SUBCLASS = {"subclassof", "is_a", "isa", "subsumes"}
_EQUIV = {"equivalentclass", "sameas", "equivalent"}
_SYMMETRIC = {"inverseof", "symmetric", "sym"}


def _local(p: str) -> str:
    """Local name of a predicate (strip namespace / prefix)."""
    return p.split("#")[-1].split("/")[-1].split(":")[-1].strip().lower()


@dataclass
class OntologyModel:
    classes: list[str]
    edges: list[tuple[str, str, str, str]]      # (from, to, predicate, kind)
    definitions: list[list[str]] = field(default_factory=list)  # face vertex sets

    def class_names(self) -> list[str]:
        return list(self.classes)


def parse_rdf(triples: list[tuple[str, str, str]]) -> OntologyModel:
    """Parse (subject, predicate, object) triples into an OntologyModel."""
    classes: list[str] = []
    seen = set()
    edges, definitions = [], []

    def add_class(c):
        if c not in seen:
            seen.add(c)
            classes.append(c)

    for s, p, o in triples:
        add_class(s)
        add_class(o)
        pl = _local(p)
        if pl in _SUBCLASS:
            edges.append((s, o, pl, "gradient"))            # child -> parent
        elif pl in _EQUIV:
            edges.append((s, o, pl, "definition"))
            edges.append((o, s, pl, "definition"))          # mutual -> bigon
            definitions.append([s, o])
        elif pl in _SYMMETRIC:
            edges.append((s, o, pl, "symmetric"))
            edges.append((o, s, pl, "symmetric"))
            definitions.append([s, o])
        else:
            edges.append((s, o, pl, "object"))              # typed object property
    return OntologyModel(classes, edges, definitions)


def ontology_to_rex(model: OntologyModel):
    """Build the typed complex: subsumption/object edges + definition faces
    (bigons for equivalent/symmetric, k-gons for multi-relation definitions).
    Returns ``(rex_or_None, meta)``."""
    names = model.class_names()
    idx = {n: i for i, n in enumerate(names)}
    src, tgt, etypes = [], [], []
    directed = {}
    for (a, b, _pred, kind) in model.edges:
        if a == b:
            continue
        e = len(src)
        src.append(idx[a])
        tgt.append(idx[b])
        etypes.append(kind)
        directed[(idx[a], idx[b])] = e
    meta = {"vertex_labels": names, "input_type": "ontology", "source": "ontology",
            "n_classes": len(names), "n_relations": len(src),
            "definitions": [list(d) for d in model.definitions]}
    if not src:
        return None, meta
    from rexgraph.graph import RexGraph

    from .auto import check_analysis_size
    check_analysis_size(len(names), len(src))   # cap oversized ontologies
    S = np.asarray(src, dtype=np.int32)
    T = np.asarray(tgt, dtype=np.int32)

    # build definition faces (signed loops) as B₂ columns, verify ∂₁∂₂ = 0
    def _b1_times(col):
        acc = {}
        for e, sgn in col:
            acc[int(S[e])] = acc.get(int(S[e]), 0.0) - sgn
            acc[int(T[e])] = acc.get(int(T[e]), 0.0) + sgn
        return all(abs(v) < 1e-9 for v in acc.values())

    cols, seen = [], set()
    for d in model.definitions:
        vs = [idx[x] for x in d if x in idx]
        if len(vs) == 2:                                   # bigon
            a, b = vs
            ea = directed.get((a, b))
            eb = directed.get((b, a))
            if ea is None or eb is None:
                continue
            face = [(ea, 1.0), (eb, 1.0)]                  # traverse a->b->a
        elif len(vs) >= 3:                                 # k-gon over the loop
            face = []
            ok = True
            for i in range(len(vs)):
                a, b = vs[i], vs[(i + 1) % len(vs)]
                if (a, b) in directed:
                    face.append((directed[(a, b)], 1.0))
                elif (b, a) in directed:
                    face.append((directed[(b, a)], -1.0))
                else:
                    ok = False
                    break
            if not ok:
                continue
        else:
            continue
        key = tuple(sorted(e for e, _ in face))
        if key in seen or not _b1_times(face):
            continue
        seen.add(key)
        cols.append(face)
    if cols:
        col_ptr, row_idx, vals = [0], [], []
        for face in cols:
            for e, sgn in face:
                row_idx.append(e)
                vals.append(sgn)
            col_ptr.append(len(row_idx))
        try:
            rex = RexGraph(sources=S, targets=T,
                           B2_col_ptr=np.asarray(col_ptr, dtype=np.int32),
                           B2_row_idx=np.asarray(row_idx, dtype=np.int32),
                           B2_vals=np.asarray(vals, dtype=np.float64))
        except Exception:
            rex = RexGraph(sources=S, targets=T)
    else:
        rex = RexGraph(sources=S, targets=T)
    rex._agent_meta = meta
    return rex, meta


def diagnose_ontology(model: OntologyModel) -> dict[str, Any]:
    """Descriptive readout: subsumption hierarchy (gradient), bounded
    definitions (curl), inconsistencies (harmonic subsumption cycles)."""
    rex, meta = ontology_to_rex(model)
    report: dict[str, Any] = {
        "n_classes": meta["n_classes"], "n_relations": meta["n_relations"],
        "definitions": meta["definitions"], "findings": [],
    }
    if rex is None:
        report["state"] = "no_relations"
        report["summary"] = "No subsumption or object relations found."
        return report
    try:
        report["betti"] = [int(b) for b in rex.betti]
    except Exception:
        report["betti"] = None
    try:
        h = rex.hodge_full(np.ones(rex.nE))
        grad, curl, harm = h["pct_grad"], h["pct_curl"], h["pct_harm"]
        report["hodge"] = {
            "subsumption_hierarchy": round(float(grad), 4),
            "bounded_definitions": round(float(curl), 4),
            "inconsistencies": round(float(harm), 4)}
        # EXACT integer invariants drive the state (β₁ = harmonic dim = unbounded
        # subsumption cycles; rank(B₂) = defining relations that close a loop) -
        # the fractions above are informative magnitudes, not the decision.
        _b = report.get("betti")
        harmonic_dim = int(_b[1]) if _b else 0
        try:
            curl_dim = int(rex.nF_hodge) - (int(_b[2]) if _b else 0)   # rank(B₂)
        except Exception:
            curl_dim = 0
        report["hodge"]["inconsistency_dimension"] = harmonic_dim
        report["hodge"]["definition_dimension"] = curl_dim
        if harmonic_dim > 0:
            report["state"] = "inconsistent"
            report["summary"] = ("Subsumption cycle(s) with no defining relation - "
                                 "an inconsistency (a class transitively subsumes "
                                 "itself). Present, not necessarily fatal; review.")
            report["findings"].append({
                "severity": "high",
                "issue": "Unbounded subsumption cycle (harmonic) - a class subsumes "
                         "itself with nothing defining the loop.", "type": "cycle"})
        elif curl_dim > 0:
            report["state"] = "bounded_definitions"
            report["summary"] = ("Definitions present (equivalent/symmetric/"
                                 "intersection) - bounded, as intended.")
        else:
            report["state"] = "acyclic_hierarchy"
            report["summary"] = "A clean subsumption hierarchy (acyclic gradient)."
    except Exception as e:
        report["state"] = "unknown"
        report["summary"] = f"Diagnosis error: {e}"
    if meta["definitions"]:
        report["findings"].append({
            "severity": "info",
            "issue": f"{len(meta['definitions'])} definition face(s) "
                     "(equivalent/symmetric/intersection) - bounded, not cycles.",
            "type": "definition"})
    return report
