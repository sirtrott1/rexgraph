"""What an answer rests on, exactly.

A retrieval currently reports how MANY records it opened. That is a count, and it cannot
say whether the answer would survive losing any one of them. The tower already answers
that: the leverage is the diagonal of the projector onto the row space of `B1`, so a
relation is irreplaceable to the degree that nothing else reaches where it reaches, and
the degree is exact.

This is semantic significance in the sense the term is used when asking whether a finding
may be ACTED ON. Statistical significance normalises against an assumed distribution;
this normalises against `rank(B1)` and the cycle space, which the structure fixes itself.
So there is no threshold here and no null model, and none should be added: the readings
are magnitudes, and the policy over them belongs to the caller.

A query is a SECTION of the corpus complex, so its provenance is the section reading:

    irreplaceable   relations at R_eff = 1: nothing else reaches there. Losing one
                    loses what it carried, with no alternative route.
    corroborated    R_eff < 1, and 1 - R_eff is how much of the cycle space it shares
    gap             what the REST of the corpus closes for this section. Zero means the
                    answer stands on the retrieved relations alone.
    coupling        how strongly the supporting relations couple through the complex.
                    Measured to track functional relatedness with overlap held fixed;
                    it is NOT a conflict or disagreement reading. OPT-IN: it is the only
                    reading here that costs a solve, and it was 1.33s of a 1.35s query
    unaccounted     the harmonic share of the response: what no higher-order structure
                    in the corpus explains

The last one is the honest part of an answer. A response whose harmonic share is large is
one the corpus has no structure to account for, and reporting it is the difference
between a retrieval and a claim.
"""
from __future__ import annotations

import numpy as np

__all__ = ["query_provenance", "store_provenance", "index_leverage", "format_provenance"]


def index_leverage(index):
    """The leverage over a stored index, computed once and cached against its digest.

    The index is the complex, so this is a property OF THE STORE and not of a query. It
    is invalidated by a write, which is what the digest tracks, so a read-heavy corpus
    pays the solve once and every subsequent query is a lookup.
    """
    from agent import rcdb_index as ix

    digest = index.get("state_digest")
    hit = index.get("_leverage")
    if hit is not None and hit[0] == digest:
        return hit[1], hit[2]
    rex = ix.complex_of(index)
    lev = np.asarray(rex._effective_resistance_batch(np.arange(int(rex.nE))))
    index["_leverage"] = (digest, rex, lev)
    index["_corpus_rank"] = int(rex.rank_tower()["ranks"][0])
    return rex, lev


def store_provenance(index, doc_ids, *, response=None, coupling=False, field=None):
    """Provenance for a retrieval, resolving record ids through the stored index.

    The retrieval returns records; the readings are over RELATIONS. A record owns a run
    of them (`relations_of`), so the section is the union of those runs, which is why the
    answer's support is a section of the corpus complex rather than a list of documents.
    """
    from agent import rcdb_index as ix

    rex, lev = index_leverage(index)
    want = [str(d) for d in doc_ids]
    # rows_for is the index's own lookup: one hash over the id table rather than a scan
    # of every row per query. Hand-rolling the scan made a retrieval O(records) in
    # Python on a structure that answers it directly.
    rows = [int(r) for r in ix.rows_for(index, ids=want)]
    owner = ix.rel_owner(index)
    rels = [e for r in rows for e in ix.relations_of(index, r)]
    labels = {e: str(index["ids"][owner[e]]) for e in rels}
    p = query_provenance(rex, rels, response=response, leverage=lev, labels=labels,
                         coupling=coupling, field=field,
                         corpus_rank=index.get("_corpus_rank"))
    p["n_records"] = len(rows)
    p["missing"] = sorted(set(want) - {str(index["ids"][r]) for r in rows})
    return p


def query_provenance(rex, retrieved, *, response=None, leverage=None, labels=None,
                     coupling=False, field=None, corpus_rank=None):
    """The exact structural provenance of one answer.

    `retrieved` is the relation indices the answer rests on. `response` is an optional
    1-cochain over the relations (the field the answer was read from), which adds the
    Hodge split; without it the structural readings are still returned.

    `coupling` is OFF by default because it is the only reading here that costs a solve.
    Measured on a real store it was 1.33s of a 1.35s query while every structural reading
    together came to 0.02s, so computing it unasked made every retrieval pay for a
    coordinate most callers never read. Pass `field` to reuse a solve across queries.

    `corpus_rank` is a property of the store rather than of the query, so a caller
    holding it passes it in instead of having it recomputed per retrieval.

    Nothing is thresholded. Every value is a magnitude for the caller to act on.
    """
    from rexgraph.partition import section_readings

    ids = np.asarray(sorted({int(i) for i in retrieved}), dtype=np.int64)
    nE = int(rex.nE)
    if ids.size == 0:
        return {"n": 0, "note": "no relation supports this answer"}
    if ids.min() < 0 or ids.max() >= nE:
        raise IndexError(f"retrieved indexes outside 0..{nE - 1}")

    lev = (np.asarray(leverage) if leverage is not None
           else np.asarray(rex._effective_resistance_batch(np.arange(nE))))
    sec = section_readings(rex, {"answer": ids}, leverage=lev)["answer"]

    r = lev[ids]
    irreplaceable = ids[r > 1.0 - 1e-9]
    order = np.argsort(-r)
    carriers = [{"relation": int(ids[i]),
                 "irreplaceability": float(r[i]),
                 "label": (labels[int(ids[i])] if labels is not None else None)}
                for i in order[:10]]

    out = {
        "n": int(ids.size),
        "irreplaceable": [int(x) for x in irreplaceable],
        "n_irreplaceable": int(irreplaceable.size),
        "carriers": carriers,
        "mass": sec["mass"],
        "own_rank": sec["own_rank"],
        "efficiency": sec["efficiency"],
        "own_cycles": sec["own_cycles"],
        "share": sec["share"],
        "gap": sec["gap"],
        "corpus_rank": int(corpus_rank if corpus_rank is not None
                           else rex.rank_tower()["ranks"][0]),
    }
    out["share_of_corpus"] = sec["mass"] / max(out["corpus_rank"], 1)
    if coupling:
        from rexgraph.partition import coupling_fraction
        # a failure here is a defect and is reported as one. Swallowing it into a NaN
        # made a broken field solve read the same as a section too small to have pairs.
        try:
            out["coupling"] = coupling_fraction(rex, {"answer": ids},
                                                field=field)["answer"]
        except Exception as exc:
            out["coupling"] = float("nan")
            out["coupling_error"] = f"{type(exc).__name__}: {exc}"

    if response is not None:
        f = np.asarray(response, dtype=float).ravel()
        if f.size != nE:
            raise ValueError(f"response is {f.size} long for {nE} relations")
        g, c, h = rex.hodge(f)
        tot = float(np.linalg.norm(f)) or 1.0
        out["hodge"] = {
            "gradient": float(np.linalg.norm(g)) / tot,
            "curl": float(np.linalg.norm(c)) / tot,
            "unaccounted": float(np.linalg.norm(h)) / tot,
        }
    return out


def format_provenance(p) -> str:
    """The provenance as plain sentences, stating magnitudes and no verdicts."""
    if not p.get("n"):
        return "No relation in the corpus supports this answer."
    L = []
    L.append(f"The answer rests on {p['n']} relations, which hold {p['mass']:.2f} of the "
             f"corpus rank {p['corpus_rank']} ({p['share_of_corpus']*100:.2f}%).")
    if p["n_irreplaceable"]:
        L.append(f"{p['n_irreplaceable']} of them are irreplaceable: nothing else in the "
                 f"corpus reaches where they reach, so losing one loses what it carried.")
    else:
        L.append("None of them is irreplaceable: every one is corroborated by an "
                 "alternative route through the corpus.")
    if abs(p["gap"]) < 1e-9:
        L.append("The answer stands on the retrieved relations alone: nothing outside "
                 "them closes any of its structure.")
    else:
        L.append(f"{p['gap']:.2f} of its cycle content is closed by the REST of the "
                 f"corpus rather than by the retrieved relations, so the answer depends "
                 f"on structure it did not return.")
    if "coupling" in p and np.isfinite(p.get("coupling", float("nan"))):
        L.append(f"Its supporting relations couple at {p['coupling']*100:.0f}% "
                 f"(higher means more structurally entangled through the corpus).")
    hz = p.get("hodge")
    if hz:
        L.append(f"Of the response, {hz['gradient']*100:.0f}% is additive and could have "
                 f"come from a per-entity model, {hz['curl']*100:.0f}% is explained by a "
                 f"higher-order relation already present, and {hz['unaccounted']*100:.0f}%"
                 f" is accounted for by nothing in the corpus.")
    return " ".join(L)
