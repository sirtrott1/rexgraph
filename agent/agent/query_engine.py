"""
agent.query_engine: structural, relational-complex-aware question answering.

Every query becomes its own relational complex. That complex is aligned
against the document (or corpus) complex to find which concepts and
sections structurally resonate with the question: retrieval by shared
topology, not just string match. The retrieved sections ground an answer
that is synthesized by a language model when one is configured
(``agent.chat_model``), and by a structural summary otherwise. Results
are content-addressed cached.

This is the piece that integrates pipeline output (document/corpus
complexes + chunks + analysis) with chat.
"""

from __future__ import annotations

import logging

import contextlib
import os
import re
from typing import Any

import numpy as np

from agent.metrics import coherence_kappa, coherence_mean

# query complex

def build_query_rex(query: str, max_vocab: int = 200, *,
                    relation_mode: str = "branching"):
    """Build a relational complex from the query text, the way a DOCUMENT is built.

    Returns ``(rex_or_None, edge_construction_or_None)``. A single-word
    query has no edges -> ``rex`` is None but the ec (vocabulary) is
    still returned for label-level alignment.

    `relation_mode="branching"` is not a tuning knob, it is the condition under which
    the score means anything. `interfacing_score` compares this complex against a stored
    document, and `rexgraph.document.build_document` carries each sentence as ONE k-ary
    relation with no pairs enumerated. Built pairwise, a query is a windowed
    co-occurrence graph and the document is a field of branching relations: the two are
    different objects and the comparison is between constructions rather than texts. The
    tokenizer is shared through `from_text`, so the vocabulary they align on is also the
    same code.
    """
    from agent.adapters.text import TextAdapter
    ec = TextAdapter().build(query, min_count=1, max_vocab=max_vocab,
                             relation_mode=relation_mode)
    if not getattr(ec, "vertex_labels", None):
        return None, None
    if ec.nE == 0:
        return None, ec
    try:
        from agent.auto import FACE_RULE, build_rex_from_edges
        # Faces are asked for only on the pairwise construction. FACE_RULE fills cycles
        # in a pairwise complex; a branching document has no faces added either, so
        # requesting them here would reintroduce exactly the asymmetry this mode exists
        # to remove.
        rex = (build_rex_from_edges(ec) if relation_mode == "branching"
               else build_rex_from_edges(ec, face_selection=FACE_RULE))
    except Exception:
        rex = None
    return rex, ec


def query_signature(rex, ec) -> dict[str, Any]:
    """Compact structural signature of the query complex."""
    labels = list(getattr(ec, "vertex_labels", []) or [])
    sig: dict[str, Any] = {
        "n_concepts": len(labels),
        "n_relations": int(getattr(ec, "nE", 0) or 0),
        "concepts": labels[:24],
    }
    if rex is not None:
        with contextlib.suppress(Exception):
            sig["betti"] = [int(b) for b in rex.betti]
        with contextlib.suppress(Exception):
            sig["kappa_mean"] = round(coherence_mean(rex), 4)
    return sig


# query <-> document relation

def relate_query_to_doc(query_ec, doc_rex, doc_meta: dict) -> dict[str, Any]:
    """Align the query's concepts against the document complex.

    Uses the compiled ``align_by_labels`` to find shared concepts, then
    ranks them by the document's per-vertex coherence (κ) - i.e. which
    of the query's concepts are *structurally central* in the document.
    """
    doc_labels = list(doc_meta.get("vertex_labels", []) or [])
    q_labels = list(getattr(query_ec, "vertex_labels", []) or [])
    if not doc_labels or not q_labels:
        return {"n_shared": 0, "coverage": 0.0, "concepts": []}

    try:
        from rexgraph.core._cross_complex import align_by_labels
        shared, idx_q, idx_doc = align_by_labels(q_labels, doc_labels)
    except Exception:
        # pure-python fallback
        dset = {l: i for i, l in enumerate(doc_labels)}
        shared, idx_doc = [], []
        for l in q_labels:
            if l in dset:
                shared.append(l); idx_doc.append(dset[l])

    # Coherence κ at ONLY the shared query concepts, by demand-driven diffusion -
    # propagate from the relevant vertices instead of enumerating the whole document
    # complex's per-vertex coherence just to read a handful of entries.
    doc_idx = np.asarray([int(idx_doc[k]) for k in range(len(shared))], dtype=int)
    kvals = None
    try:
        kvals = np.asarray(doc_rex.coherence_response(doc_idx), dtype=float)
    except Exception:
        try:                                    # fallback: full enumeration
            full = coherence_kappa(doc_rex)
            kvals = np.array([full[i] if i < len(full) else 0.0 for i in doc_idx])
        except Exception:
            kvals = None

    scored: list[tuple[str, float]] = []
    for k, label in enumerate(shared):
        kv = float(kvals[k]) if (kvals is not None and k < len(kvals)) else 0.0
        scored.append((label, round(kv, 4)))
    scored.sort(key=lambda x: -x[1])

    return {
        "n_shared": len(shared),
        "coverage": round(len(shared) / max(len(q_labels), 1), 3),
        "concepts": [{"concept": c, "doc_coherence": k} for c, k in scored[:16]],
    }


# retrieval

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


#: Sentences kept per retrieved section. This is an OUTPUT BUDGET, not a decision
#: threshold: it bounds how much context reaches the model, and nothing about the
#: ranking depends on it. It was an inline 2, which capped the whole context at 4 to 10
#: sentences regardless of top_k and left the model to fill the gap with generic prose.
#: Override with REXGRAPH_SECTION_SENTENCES, or per call via `section_sentences`.
SECTION_SENTENCES = _env_int("REXGRAPH_SECTION_SENTENCES", 6)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def _best_sentences(text: str, query_tokens: set, k: int = 2) -> list[str]:
    """Top-k sentences of ``text`` by overlap with the query tokens."""
    scored = []
    for sent in _split_sentences(text):
        toks = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", sent.lower()))
        overlap = len(toks & query_tokens)
        if overlap:
            scored.append((overlap, sent))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:k]]


def retrieve_closure(rex, seeds, *, labels=None, max_depth: int = 8) -> dict:
    """The smallest subcomplex after which more context stops changing the answer.

    Ranking answers "which items are most like the query". This answers a different
    question: "what is the whole of what this complex says about these entities, and how
    do I know I have all of it". No top_k, because top_k is a number someone picked; the
    boundary here is where the reading stops moving, which is a property of the entities
    and the structure around them.

    Each seed's closure is taken separately and the union returned, with the per-seed
    depths kept. A seed that closes at depth 1 and one that needs depth 3 are different
    facts about those entities and averaging them away would lose the more interesting
    one: on real binding data a self-contained target closed at 1 while a target whose
    ligands are shared closed at 2, having acquired six independent cycles on the way.

    The audit trail is the point. `steps` carries the shape at every depth, so a caller
    can see what arrived when, and `betti` says whether the evidence CLOSES: a tree is
    facts hanging off the seed, a cycle is facts corroborating each other through a
    second path. That is an explicit, inspectable context structure rather than a
    similarity that cannot be interrogated.
    """
    from rexgraph.tower import semantic_closure

    seeds = [int(s) for s in seeds]
    if not seeds:
        return {"seeds": [], "relations": [], "n_relations": 0, "closures": [],
                "reason": "no seed entities were given"}

    relations, closures = set(), []
    for seed in seeds:
        closure = semantic_closure(rex, seed, max_depth=int(max_depth))
        relations.update(closure["relations"])
        closures.append({
            "seed": seed,
            "label": (str(labels[seed]) if labels is not None and seed < len(labels)
                      else None),
            "depth": closure["depth"],
            "converged": closure["converged"],
            "steps": closure["steps"],
            "n_relations": len(closure["relations"]),
        })

    supports = rex.relation_supports()
    covered = sorted({v for e in relations for v in supports[e]})
    unclosed = [c["label"] or c["seed"] for c in closures if not c["converged"]]
    return {
        "seeds": seeds,
        "relations": sorted(relations),
        "n_relations": len(relations),
        "vertices": covered,
        "n_vertices": len(covered),
        "closures": closures,
        "all_converged": not unclosed,
        "unclosed": unclosed,
        "reading": ("the boundary is where the reading stops changing, not a top_k. "
                    "`steps` is the audit trail and `betti` says whether the evidence "
                    "closes on itself or hangs off the seed"),
    }


def retrieve_sections(query: str, top_k: int, *, corpus=None,
                      doc_rex=None, doc_meta: dict | None = None,
                      query_ec=None,
                      section_sentences: int | None = None,
                      store=None, prefix: str = "", candidates: int | None = None,
                      as_of=None, valid_at=None, mode: str = "hybrid",
                      temporal: str | None = None,
                      ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (sections, relation).

    - With an RCStore: rank the persisted corpus (see `retrieve_from_store`).
    - With a non-empty CorpusBuilder: delegate to its structural retrieval
      (``corpus.query`` -> chi/spectral/hybrid ranking).
    - Single document (or empty corpus): rank sentences of the source text
      by the coherence-weighted mass of query concepts they contain.

    The three are tried in that order and each falls through to the next, so a
    store that holds nothing for this query still answers from whatever is local.
    """
    if store is not None:
        sections, relation = retrieve_from_store(
            query, top_k, store=store, prefix=prefix, candidates=candidates,
            as_of=as_of, valid_at=valid_at, mode=mode, temporal=temporal,
            section_sentences=section_sentences)
        if sections:
            return sections, relation

    if corpus is not None:
        try:
            has_docs = len(getattr(corpus, "documents", []) or []) > 0
        except Exception:
            has_docs = True
        if has_docs:
            try:
                qr = corpus.query(query, top_k=top_k)
                recmap = {}
                for rec in getattr(corpus, "documents", []) or []:
                    recmap[getattr(rec, "doc_id", None)] = rec
                q_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower()))
                sections = []
                for s in (qr.ranked_sections or []):
                    did = s.get("doc_id") or s.get("document")
                    rec = recmap.get(did)
                    text = ""
                    if rec is not None:
                        rmeta = getattr(getattr(rec, "rex", None), "_agent_meta", {}) or {}
                        src = rmeta.get("source_text", "") or getattr(rec, "text", "") or ""
                        n_sent = (SECTION_SENTENCES if section_sentences is None
                                  else max(1, int(section_sentences)))
                        best = _best_sentences(src, q_tokens, n_sent)
                        text = " … ".join(best) if best else src[:300]
                    sections.append({
                        "doc_id": did,
                        "text": text,
                        "score": round(float(s.get("score", s.get("relevance", 0.0))), 4),
                        "shared_entities": s.get("n_shared_entities"),
                    })
                if sections:
                    return sections[:top_k], {"mode": "corpus", "n_ranked": len(sections)}
            except Exception:
                pass  # fall through to single-doc

    return _single_doc_retrieve(query, top_k, doc_rex=doc_rex,
                                doc_meta=doc_meta, query_ec=query_ec,
                                section_sentences=section_sentences)




# Store-backed retrieval
#
# Two stages, because a signature is queryable without touching a blob: rank the
# signatures to pick candidates, then deserialize only those and score them with the
# corpus's own score_document. Scoring every blob would make persistence a pure cost.

#: how many blobs to open per query when the caller does not say. Enough headroom for
#: the signature prefilter to be wrong about a few, small enough that the store stays
#: cheaper than holding the corpus in memory.
logger = logging.getLogger(__name__)

STORE_CANDIDATES = _env_int("REXGRAPH_STORE_CANDIDATES", 24)

#: the store predicate is a token match, so it over-returns relative to the ranking.
#: Pull a multiple of the candidate budget and let the signature affinity order them,
#: rather than trusting the first `n` rows the store happens to hand back.
_PREFILTER_SLACK = _env_int("REXGRAPH_PREFILTER_SLACK", 4)


def _sections_by_field(doc, rec, qec, k, *, channels=False):
    """The top `k` sections of a document, found by DIFFUSION over its own partition.

    This is the lookup the layers exist for, and it is a different thing from what it
    replaces. `_best_sentences` re-split a stored blob with a punctuation regex and
    ranked the pieces by how many query words they contained; this seeds the document's
    field at the query's vertices, lets heat spread through the document's own relations,
    and reads the response back over the exact partition already stored with it. Nothing
    scans the text and nothing is re-segmented: the spans were computed once at ingest
    and the prose is fetched by seek.

    Returns a list of `{section_id, layer, span, response, mass, proof_len, text}`, or
    None when the record carries no sectioning to look up.
    """
    import numpy as np

    from rexgraph.partition import section_coverage, section_response
    from rexgraph.sectioning import sectionings_of

    rex = getattr(doc, "rex", None)
    if rex is None:
        return None
    store = sectionings_of(rex)
    if not store:
        return None
    layer = "sentence" if "sentence" in store else sorted(store)[0]
    sect = store[layer]
    if sect.is_derived:
        sect = sect.resolved(store)

    labels = [str(x).lower() for x in (getattr(doc, "vertex_labels", []) or [])]
    want = {str(x).lower() for x in (getattr(qec, "vertex_labels", []) or [])}
    seeds = [i for i, lb in enumerate(labels) if lb in want]
    if not seeds:
        return None
    try:
        resp, names = section_response(rex, sect, seeds)
    except Exception:
        # One candidate that cannot be read is dropped from the ranking rather than
        # failing the query, since a retrieval scores many and keeps few. None is the
        # sentinel the caller already handles for "no seeds matched".
        return None
    if not resp.size:
        return None

    # THE CHANNEL PROFILE, carried rather than summed away. `resp` is the profile summed
    # over (topology, geometry, frustration, coparticipation), and those channels move in
    # OPPOSITE directions between a section that answers and one that merely shares
    # vocabulary: measured, topology 0.2379 against 0.2161 and coparticipation 0.2053
    # against 0.2570, so the sum annihilates the difference exactly. A section that
    # answers responds through the document's own topology; one that only shares words
    # responds through co-participation. The scalar cannot say which.
    #
    # Held out, the direction classifies answerable from foreign at 54.0% against 50.6%
    # chance: it recovers signal the scalar destroyed without being a reliable typing.
    # It is reported, not acted on.
    #
    # Computed only when ASKED, because `rex.structural_character` is 0.37 s a document
    # and a retrieval scores 24 candidates to keep 3. That is the same split
    # `score_document(reading=False)` documents: diagnostics on what survives, not on
    # what is about to be discarded. Measured, doing it for every candidate took a
    # whole-corpus query from 5 s to 32 s.
    prof, chan = None, []
    if channels:
        try:
            prof, _pn, chan = section_response(rex, sect, seeds, channels=True)
        except Exception:
            # The profile is diagnostic and additive. Losing it costs the caller the
            # per-channel axes, never the ranking, which is `resp` above.
            prof, chan = None, []

    # TWO READINGS, because which one is right is a property of the QUERY and a caller
    # cannot know in advance which it has. Measured at n=149, top-1 on the section a
    # query was lifted from: magnitude 94.6% / 71.8% / 33.6% as the query goes from the
    # whole section to a half to a quarter, coverage 38.3% / 53.0% / 51.0%. They cross
    # over, and a real question is at the short end.
    #
    # This is ADDITIVE and deliberately so. Magnitude still orders the result, which
    # keeps the full-query case exactly as it was: there, taking magnitude's own top-2
    # beats consulting coverage (97.3% against 94.6%), so coverage must not displace
    # anything. It is appended as one extra candidate when it disagrees.
    #
    # Agreement between the two is a confidence signal with no threshold in it: when they
    # name the same section, magnitude is right 100% of the time on half- and
    # quarter-length queries, against 57.0% and 16.1% when they disagree.
    try:
        cov, _n2 = section_coverage(rex, sect, seeds)
    except Exception:
        # Coverage is the ADDITIVE second reading described above: magnitude still
        # orders the result, so its absence returns the full-query behaviour exactly.
        cov = None

    heap = (rec.meta or {}).get("heap") or ""
    order = list(np.argsort(resp)[::-1][:max(1, int(k))])
    top_cov = int(np.argmax(cov)) if cov is not None and cov.size and cov.max() > 0 else None
    agree = top_cov is not None and len(order) and int(order[0]) == top_cov
    if top_cov is not None and top_cov not in [int(j) for j in order]:
        order.append(top_cov)
    out = []
    for j in order:
        j = int(j)
        from_cov = top_cov is not None and j == top_cov
        if resp[j] <= 0 and not from_cov:
            continue
        span = (tuple(int(x) for x in sect.spans[j])
                if sect.spans is not None and j < len(sect.spans) else None)
        text = ""
        if span and heap:
            with contextlib.suppress(Exception):
                from rexgraph.document import section_text
                text = section_text(rex, layer, j, path=heap).strip()
        out.append({
            "section_id": names[j] if j < len(names) else str(j),
            "layer": layer, "span": span,
            "response": float(resp[j]),
            "coverage": float(cov[j]) if cov is not None else None,
            # the axes, and the direction separately: magnitude says how much, direction
            # says what KIND of response it is
            "channels": ([float(v) for v in prof[j]]
                         if prof is not None and j < len(prof) else None),
            "channel_names": list(chan) if chan else None,
            # which reading put this section here, and whether the two agreed at the top
            "reading": ("both" if (from_cov and j == int(np.argsort(resp)[::-1][0]))
                        else "coverage" if from_cov else "magnitude"),
            "agree": bool(agree),
            "text": text,
        })
    return out or None


def _field_candidates(store, q_tokens: set, limit: int, prefix: str = ""):
    """Candidate records, ordered by the index complex's own response to the query.

    Returns None when the store has no index complex to read (a MemoryStore, a store
    whose snapshot has not been compacted), so the caller falls back to the scan.

    This replaces `_signature_affinity`, which scored candidates off `labels_sample`,
    twelve entries, and therefore returned 0.0000 for essentially every record on the
    Gutenberg store, leaving the order arbitrary. A sample cannot rank a corpus. The
    field can, because the index carries every record's full accession.
    """
    snap = getattr(getattr(store, "_idx", None), "_snap", None)
    if not isinstance(snap, dict) or not snap.get("n"):
        return None
    try:
        from agent import rcdb_index as ix
        prof, ids, chan = ix.record_response(snap, q_tokens, channels=True)
        # THE SECOND TOWER. The share reading divides each term's contribution by the
        # record's width, so it answers "what fraction of this record is the query",
        # a DENSITY. Measured on the documents that all hold `221b baker street`, its
        # order agrees with the ordering by accession width at rank correlation +1.000:
        # a 3,206-term pamphlet holding a page number beat the Adventures of Sherlock
        # Holmes, which sat at 38. The existence tower reads the {0,1} incidence and so
        # answers the MASS question, which puts Holmes at 4.
        #
        # NEITHER IS THE DEFAULT, because measured on 12 queries with known answers they
        # trade: share takes top-1 6/12 to existence's 5/12, and existence takes Alice
        # from 5 to 941. Read together they recover what either alone drops: recall@20
        # 10/12 against 9/12 each. This is the same both-readings pattern
        # `_sections_by_field` already uses one grade down, where it is magnitude vs
        # coverage.
        mass, _ids2 = ix.record_response(snap, q_tokens, reading="existence")
    except Exception:
        # A silent None here means the caller quietly falls back to the scan, which is
        # 88 s against 1.2 s: an eighty-fold regression that looks like nothing. It
        # happened: a CSC/CSR mixup made `deg` per-relation instead of per-vertex and
        # this except swallowed the IndexError for an entire session. Say so.
        logger.warning("field prefilter unavailable, falling back to the scan",
                       exc_info=True)
        return None
    import numpy as _np
    # THE ORDER IS A REDUCTION AND IS NAMED AS ONE. Identification needs an ordering:
    # that is what identification IS, so the profile is summed over its channels to get
    # one, and the sum is exactly the scalar `record_response` used to return. What is
    # different is that the reduction happens HERE, visibly, and the profile travels with
    # the record instead of being discarded at the source. The axes matter downstream:
    # a record that answers responds through topology, one that merely shares vocabulary
    # responds through co-participation, and the sum cancels that distinction.
    scores = prof.sum(axis=1)
    order = _np.argsort(scores)[::-1]
    # candidates the mass reading ranks highly and the density reading buried, merged in
    # after the share order rather than interleaved: share wins top-1, so it leads, and
    # this is recall the other tower alone would have kept.
    # INTERLEAVED, not concatenated, and not blended by a ratio. Appending the mass
    # order after the share order does nothing: the share order fills the candidate quota
    # by itself, which is what happened first and left pg1661 out of a 24-candidate pool
    # entirely. Alternating gives each tower equal voice with no constant to pick.
    mass_order = _np.argsort(mass)[::-1]
    merged, seen = [], set()
    for a, b in zip(order, mass_order, strict=True):
        for j in (int(a), int(b)):
            if j not in seen and (scores[j] > 0 or mass[j] > 0):
                seen.add(j); merged.append(j)
        if len(merged) >= 4 * max(1, limit):
            break
    order = merged
    out = []
    for j in order:
        j = int(j)
        if scores[j] <= 0 and mass[j] <= 0:
            break                       # unreached: neither tower got there
        rid = str(ids[j])
        if prefix and not rid.startswith(prefix):
            continue
        rec = store.get_record(rid)
        if rec is not None:
            out.append((rec, float(scores[j]), [float(v) for v in prof[j]], list(chan)))
        if len(out) >= limit:
            break
    return out


class _StoreDoc:
    """The duck type score_document reads: a rex, its labels, an id."""

    __slots__ = ("doc_id", "rex", "vertex_labels", "text", "analysis", "source")

    def __init__(self, doc_id, rex, labels, text, source=""):
        self.doc_id = doc_id
        self.rex = rex
        self.vertex_labels = labels
        self.text = text
        self.source = source
        self.analysis = {}


def retrieve_from_store(query: str, top_k: int, *, store, prefix: str = "",
                        candidates: int | None = None, as_of=None, valid_at=None,
                        mode: str = "hybrid", temporal: str | None = None,
                        section_sentences: int | None = None,
                        provenance: bool | str = False,
                        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank a persisted corpus without holding it in memory.

    `as_of`/`valid_at` pass straight through to the store, so retrieval inherits the
    RCDB's bitemporal reads: the corpus can be queried as it stood at a time.
    """
    from agent.adapters.text import TextAdapter
    from agent.corpus import count_shared_entities, score_document
    from rexgraph.graph import RexGraph

    qec = TextAdapter().build(query, min_count=1, max_vocab=200)
    if not getattr(qec, "vertex_labels", None):
        return [], {"mode": "store", "n_ranked": 0}
    q_tokens = {w.lower() for w in qec.vertex_labels}

    # THE PREFILTER IS A FIELD READING, not a scan.
    #
    # The store's index IS a complex: records and one shared vocabulary are its vertices,
    # and a record's accession is a single branching relation with the record at position
    # 0 carrying the -1. `record_response` seeds the query's TERM vertices and applies
    # `L0 x = B1 (B1^T x)` matrix-free, so "which records answer these terms" is a matvec
    # over the operator the store already holds.
    #
    # What this replaces: `store.query(labels_any=...)`, which materialised every
    # record's meta (around 8,000 label strings each) and set-intersected in Python.
    # Measured on the 61,353-document Gutenberg store, 88 s for the prefilter alone, and
    # then `_signature_affinity` ordered the survivors off `labels_sample`, twelve
    # entries, scoring 0.0000 against every candidate, so the ordering was arbitrary and
    # the first two records opened were the corpus's two largest documents. A single
    # query did not finish in 23 minutes. The field reading is 1.2 s and ranks the right
    # book 1st, 1st, 2nd, 3rd and 7th on five title queries.
    #
    # as_of/valid_at still go to the per-candidate read. A bitemporal PREFILTER over the
    # index complex needs the index to be as-of too, which it is not, so a time-travelling
    # query falls back to the scan rather than silently reading today's vocabulary.
    n_cand = max(1, int(candidates if candidates is not None else STORE_CANDIDATES))
    n_sent_pre = (SECTION_SENTENCES if section_sentences is None
                  else max(1, int(section_sentences)))
    records = field_score = None
    field_profile = {}
    if as_of is None and valid_at is None:
        hits = _field_candidates(store, q_tokens, n_cand * _PREFILTER_SLACK, prefix)
        if hits is not None:
            records = [h[0] for h in hits]
            field_score = {h[0].id: h[1] for h in hits}
            field_profile = {h[0].id: (h[2], h[3]) for h in hits}
    if records is None:
        try:
            records = [r for r in store.query(labels_any=sorted(q_tokens),
                                              limit=n_cand * _PREFILTER_SLACK,
                                              as_of=as_of, valid_at=valid_at)
                       if not prefix or r.id.startswith(prefix)]
        except Exception:
            return [], {"mode": "store", "n_ranked": 0}
    if not records:
        return [], {"mode": "store", "n_ranked": 0}

    q_chi = None
    if qec.nE > 0:
        try:
            q_rex = RexGraph(sources=qec.sources, targets=qec.targets)
            if qec.n_types > 1:
                from agent.auto import FACE_RULE, attach_faces
                q_rex = attach_faces(q_rex, FACE_RULE, type_labels=qec.type_labels)
            q_chi = q_rex.structural_character
        except Exception:
            q_chi = None

    scored = []
    for rec in records[:n_cand]:
        try:
            # verify=False: this is a SPECULATIVE read of a candidate that may be
            # discarded, and the integrity rebuild costs a `_leaf_digests` pass: 37.6 s
            # on the largest record here. What survives into the answer is verified by
            # `answer_query` before it is committed.
            rex = store.get(rec.id, as_of=as_of, valid_at=valid_at, verify=False)
        except TypeError:
            rex = store.get(rec.id, as_of=as_of, valid_at=valid_at)
        except Exception:
            rex = None
        if rex is None:
            continue
        rmeta = getattr(rex, "_agent_meta", {}) or {}
        labels = list(rmeta.get("vertex_labels")
                      or (rec.meta or {}).get("vertex_labels") or [])
        doc = _StoreDoc(rec.id, rex, labels, rmeta.get("source_text", "") or "",
                        (rec.meta or {}).get("source", ""))
        # ONE field reading per candidate, serving both jobs. `_sections_by_field` seeds
        # the query's vertices, diffuses on the document's own relations and integrates
        # the response over its stored partition; the document's score IS the response of
        # its best-answering section, and those same sections are what comes back.
        # Scoring separately would be a second reading of the same field.
        #
        # This replaces `score_document` -> `interfacing_score` -> `coherence_response`,
        # which was the retrieval path's actual bottleneck: it builds the sparse character
        # channels and runs block-CG PER CANDIDATE, and a stack dump during a hung query
        # landed in `build_sparse_channels` or `_block_cg` every time. It was also the
        # wrong reading. Measured, the section field ranks the section a query was lifted
        # from first 86.5% of the time; a label-overlap score cannot see the construction
        # at all, because clique, spanning and branching share one vertex set.
        # EACH GRADE DOES ITS OWN JOB, and both halves are measured.
        #
        # The document's rank comes from the CORPUS complex: `_field_candidates`
        # already ordered the records by the index field's response, which put the right
        # book at rank 1, 1, 2, 3 and 7 on five title queries over 61,353 documents. Its
        # sections come from the DOCUMENT complex, where `_sections_by_field` ranks the
        # section a query was lifted from first 86.5% of the time.
        #
        # Re-deriving a document score inside the document is the mistake: the section
        # field is excellent at "where in this book" and weak at "which book", measured
        # at 24.5% top-1 cross-document against 86.5% within-document. A store with no
        # index complex has no corpus-grade reading available, so it keeps
        # `score_document`: that is a real fallback, not a legacy path.
        got = _sections_by_field(doc, rec, qec, n_sent_pre)
        rank_score = (field_score.get(rec.id) if field_score is not None
                      else score_document(doc, qec, q_chi, mode))
        scored.append((rank_score, doc, rec, got))

    if not scored:
        return [], {"mode": "store", "n_ranked": 0}

    # same deterministic tiebreak as the in-memory path: store enumeration order
    # must not decide which of two equally-scoring documents comes back.
    scored.sort(key=lambda t: (-t[0], str(t[1].doc_id)))
    sections = []
    for score, doc, rec, got in scored[:top_k]:
        # NOW resolve the channels, for the survivors only
        deeper = _sections_by_field(doc, rec, qec, n_sent_pre, channels=True)
        if deeper:
            rp = field_profile.get(rec.id)
            if rp:
                for part in deeper:
                    part["record_channels"], part["record_channel_names"] = rp[0], rp[1]
            got = deeper
        if got is None:
            # a record with no sectioning (a lexical complex, an older document) has no
            # partition to look up, so the text path stands for it rather than nothing
            best = _best_sentences(doc.text, q_tokens, n_sent_pre) if doc.text else []
            got = [{"text": " … ".join(best) if best else doc.text[:300]}]
        for part in got:
            sections.append({
                "doc_id": doc.doc_id,
                "score": round(float(score), 4),
                "shared_entities": count_shared_entities(qec.vertex_labels,
                                                        doc.vertex_labels),
                "version": rec.version,
                **part,
            })
    relation = {"mode": "store", "n_ranked": len(scored),
                "n_records": len(records), "n_opened": len(scored)}
    if provenance:
        # WHICH records, not how many. The index is the corpus complex, so the returned
        # sections are a section of it and the readings say what the answer rests on and
        # whether it would survive losing any one of them. Opt-in because the first call
        # solves for the leverage; it is then cached against the index digest.
        #
        # `provenance="full"` adds the coupling reading, which is the only one that costs
        # a solve per query: measured at 1.33s of a 1.35s retrieval against 0.02s for
        # every structural reading together. A plain True stays cheap.
        snap = getattr(getattr(store, "_idx", None), "_snap", None)
        if snap is None:
            # a Memory or SQL store need not hold a snapshot, and that is not an error
            relation["provenance"] = {"note": "this store holds no index snapshot"}
        else:
            from agent.provenance import store_provenance
            try:
                relation["provenance"] = store_provenance(
                    snap, [s["doc_id"] for s in sections],
                    coupling=(provenance == "full"))
            except Exception as exc:            # a reading must never fail a retrieval
                relation["provenance"] = {"error": f"{type(exc).__name__}: {exc}"}
    if temporal:
        # rerank the RETURNED sections, not the candidate set: temporal features are
        # cheap (signatures only) but there is no reason to compute them for
        # candidates the structural score already ruled out.
        from agent.temporal import rerank as _temporal_rerank
        sections = _temporal_rerank(sections, store, mode=temporal)
        relation["temporal"] = temporal
    return sections, relation


def _single_doc_retrieve(query: str, top_k: int, *, doc_rex=None,
                         doc_meta: dict | None = None, query_ec=None,
                         section_sentences: int | None = None):
    doc_meta = doc_meta or {}
    relation = relate_query_to_doc(query_ec, doc_rex, doc_meta) if doc_rex is not None else {"concepts": []}
    concept_weight = {c["concept"].lower(): (c["doc_coherence"] + 0.1)
                      for c in relation.get("concepts", [])}
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower()):
        concept_weight.setdefault(tok, 0.1)

    text = doc_meta.get("source_text", "") or ""
    sentences = _split_sentences(text)
    scored = []
    for sent in sentences:
        toks = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", sent.lower()))
        score = sum(concept_weight.get(t, 0.0) for t in toks)
        if score > 0:
            scored.append((score, sent))
    scored.sort(key=lambda x: -x[0])
    sections = [{"doc_id": doc_meta.get("doc_id", "current"),
                 "text": s, "score": round(float(sc), 4)}
                for sc, s in scored[:top_k]]
    return sections, {"mode": "single_doc", **relation}


# synthesis

_SYSTEM_PREAMBLE = (
    "You are RexGraph's analysis assistant. Answer the user's question using "
    "ONLY the structural analysis and document context provided below. The "
    "analysis comes from a relational-complex (topological) model: Betti "
    "numbers are independent cycles, kappa is coherence, the Hodge split is "
    "gradient/curl/harmonic, voids are unrealized structure. Be concise, cite "
    "concrete numbers, and do not invent facts not present in the context."
)


def _fallback_answer(query: str, doc_summary: str, sections: list[dict],
                     relation: dict) -> str:
    """The answer when no language model is involved.

    This is not a degraded narration. A document complex records CO-OCCURRENCE, so what
    it exactly supports is "these spans contain these of your terms, at these offsets",
    and that is a citation, which is more than a synthesis without one. The passage
    answerer states it; what follows here is the structural context around it.
    """
    parts = []
    from agent.answerers.passage import PassageAnswerer, render as _render_passage
    got = PassageAnswerer().answer(query, sections)
    if got.get("answered"):
        parts.append(_render_passage(got))
    elif sections:
        # retrieval found spans but none holds a query term: say that, rather than
        # printing them as though they answered something.
        parts.append(f"No retrieved passage contains {', '.join(got.get('terms', [])[:6])}. "
                     f"The nearest {min(3, len(sections))} by field response:")
        for i, s in enumerate(sections[:3], 1):
            snippet = (s.get("text") or "")[:280]
            parts.append(f"  {i}. {snippet}{'…' if len(s.get('text') or '') > 280 else ''}")

    if doc_summary:
        parts.append(("\n" if parts else "") + doc_summary)
    concepts = relation.get("concepts", [])
    if concepts:
        top = ", ".join(f"{c['concept']} (κ={c['doc_coherence']:.2f})"
                        for c in concepts[:6])
        parts.append(f"\nShared with the document's own concepts: {top}.")
    # the EVIDENTIAL provenance, when the caller asked for it to be computed. The
    # passages above say WHERE the answer is; this says what it RESTS ON: how much of
    # the corpus rank the supporting relations hold, and whether anything else reaches
    # where they reach. It was being computed and then dropped on this path.
    prov = relation.get("provenance")
    if isinstance(prov, dict) and prov.get("n"):
        from agent.provenance import format_provenance
        parts.append("\n" + format_provenance(prov))

    if not parts:
        parts.append("Nothing in the indexed structure bears on this question.")
    return "\n".join(parts)


def synthesize(query: str, doc_summary: str, sections: list[dict],
               relation: dict) -> tuple[str, bool, dict]:
    """Return (answer_text, model_used, token_metrics). token_metrics is the reply's
    perplexity/varentropy from the model's logprobs (empty for the structural fallback
    or backends without logprobs)."""
    context_lines = []
    if doc_summary:
        context_lines.append("Document analysis:\n" + doc_summary)
    if sections:
        context_lines.append("\nRelevant passages:")
        for i, s in enumerate(sections, 1):
            context_lines.append(f"[{i}] {s['text'][:600]}")
    concepts = relation.get("concepts", [])
    if concepts:
        context_lines.append("\nKey shared concepts (with document coherence): "
                             + ", ".join(f"{c['concept']}={c['doc_coherence']}"
                                         for c in concepts[:10]))
    context = "\n".join(context_lines)

    try:
        from agent import chat_model
        if chat_model.is_available():
            res = chat_model.generate_with_metrics(
                prompt=query,
                system=_SYSTEM_PREAMBLE + "\n\n" + context,
                max_tokens=512,
            )
            if res and res.get("text"):
                return res["text"], True, (res.get("metrics") or {})
    except Exception:
        pass
    return _fallback_answer(query, doc_summary, sections, relation), False, {}


# orchestration + cache

def _cache_key(doc_meta: dict, query: str, top_k: int, corpus_id: str = "") -> str | None:
    try:
        from agent import cache
        basis = (doc_meta or {}).get("source_text", "") or corpus_id
        if not basis:
            return None
        return cache.content_key(basis, depth="chat",
                                 extra=f"{query}|{top_k}|{corpus_id}")
    except Exception:
        return None


def answer_query(doc_rex, query: str, results: dict | None = None, *,
                 corpus=None, doc_meta: dict | None = None,
                 top_k: int = 5, use_cache: bool = True,
                 doc_summary: str = "",
                 section_sentences: int | None = None,
                 store=None) -> dict[str, Any]:
    """End-to-end structural answer for a chat query.

    Builds the query complex, retrieves resonant sections from the
    document/corpus, synthesizes an answer (model or structural), and
    caches the result.

    `store` is an RCStore to answer from, and `retrieve_sections` already ranks it ahead
    of a `CorpusBuilder` and a single document, falling through when it holds nothing for
    the query. Without it a caller can only reach what is in memory: the chat route had
    no way to pass one, so a persisted corpus of any size was unreachable from a
    conversation no matter what had been ingested into it.
    """
    doc_meta = doc_meta or (getattr(doc_rex, "_agent_meta", {}) if doc_rex is not None else {})
    corpus_id = getattr(corpus, "corpus_id", "") if corpus is not None else ""
    if store is not None and not corpus_id:
        # the cache key has to name the SOURCE, or an answer from the store and an answer
        # from a local document collide on the same query text
        corpus_id = f"store:{getattr(store, 'root', None) or getattr(store, 'backend', 'rc')}"

    key = _cache_key(doc_meta, query, top_k, corpus_id) if use_cache else None
    if key:
        try:
            from agent import cache
            hit = cache.get(key)
            if hit and "answer" in hit:
                hit["cached"] = True
                return hit
        except Exception:
            pass

    # THE EXACT ANSWERERS FIRST. An exact method is exact because it is specific to a
    # structure, so the question goes to the structure that makes it exact rather than to
    # one mechanism stretched over everything. The lexicon's relations ARE predications:
    # `hypernym` is is-a, so "what does X mean" is answerable there and is not answerable
    # from a corpus of co-occurrence, which is why it used to return whaling narratives.
    #
    # Each DECLINES anything it cannot support, and declining costs nothing: an answerer
    # checks its own interface before touching its structure, so a non-lexical query never
    # loads a lexicon. The passages below still run either way: the composition is a
    # union of exact answers, not a choice between them.
    exact = None
    with contextlib.suppress(Exception):
        from agent.answerers import exact_answers
        got = exact_answers(query)
        if got:
            # a UNION: every structure that can answer exactly does, and each keeps its
            # own provenance. Two answerers answering is two answers, not a merge.
            exact = got[0] | {"answers": got} if len(got) == 1 else {
                "kind": "+".join(g["kind"] or "?" for g in got),
                "subject": got[0]["subject"],
                "source": ", ".join(dict.fromkeys(g["source"] for g in got)),
                "text": "\n\n".join(g["text"] for g in got if g["text"]),
                "answers": got,
            }

    q_rex, q_ec = build_query_rex(query)
    q_sig = query_signature(q_rex, q_ec) if q_ec is not None else {"n_concepts": 0}

    sections, relation = retrieve_sections(
        query, top_k, corpus=corpus, doc_rex=doc_rex,
        doc_meta=doc_meta, query_ec=q_ec, section_sentences=section_sentences,
        store=store)

    answer, model_used, token_metrics = synthesize(query, doc_summary, sections, relation)

    # The exact answer LEADS, and the passages stay beneath it as what the corpus
    # separately supports. Union, not replacement: the lexicon answered a definitional
    # question exactly and the corpus can still say where the term is used.
    if exact is not None:
        answer = (f"{exact['text']}\n\n[{exact['source']}, {exact['kind']} of "
                  f"{exact['subject']!r}]"
                  + (f"\n\n{answer}" if answer else ""))

    payload: dict[str, Any] = {
        "answer": answer,
        "exact": exact,
        "query_complex": q_sig,
        "sections": sections,
        "relation": {k: v for k, v in relation.items() if k != "concepts"} | (
            {"concepts": relation.get("concepts", [])[:8]} if "concepts" in relation else {}),
        "model_used": model_used,
        "token_metrics": token_metrics,
        "method": relation.get("mode", "single_doc"),
        "cached": False,
    }

    if key and not model_used:
        # only cache deterministic (structural) answers; model answers may vary
        try:
            from agent import cache
            cache.set(key, payload)
        except Exception:
            pass
    return payload
