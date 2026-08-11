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

import contextlib
import os
import re
from typing import Any

import numpy as np

from agent.metrics import coherence_kappa, coherence_mean

# query complex

def build_query_rex(query: str, max_vocab: int = 200):
    """Build a relational complex from the query text.

    Returns ``(rex_or_None, edge_construction_or_None)``. A single-word
    query has no edges -> ``rex`` is None but the ec (vocabulary) is
    still returned for label-level alignment.
    """
    from agent.adapters.text import TextAdapter
    ec = TextAdapter().build(query, min_count=1, max_vocab=max_vocab)
    if not getattr(ec, "vertex_labels", None):
        return None, None
    if ec.nE == 0:
        return None, ec
    try:
        from agent.auto import FACE_RULE, build_rex_from_edges
        # The query complex is compared against document complexes, so it has to be
        # built under the same face rule they are; a faceless query scored against
        # faced documents compares two different objects.
        rex = build_rex_from_edges(ec, face_selection=FACE_RULE)
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
STORE_CANDIDATES = _env_int("REXGRAPH_STORE_CANDIDATES", 24)

#: the store predicate is a token match, so it over-returns relative to the ranking.
#: Pull a multiple of the candidate budget and let the signature affinity order them,
#: rather than trusting the first `n` rows the store happens to hand back.
_PREFILTER_SLACK = _env_int("REXGRAPH_PREFILTER_SLACK", 4)


def _signature_affinity(sig: dict[str, Any], q_tokens: set) -> float:
    """Cheap prefilter score from the stored signature alone.

    labels_sample is a sample, not the vocabulary, so this ORDERS candidates; it
    never decides the answer. A record with no labels stored still gets a small
    positive score so it can be opened rather than silently dropped.
    """
    labels = {str(x).lower() for x in (sig.get("labels_sample") or [])}
    if not labels:
        return 1e-6
    hit = len(labels & q_tokens)
    return hit / len(labels | q_tokens) if (labels | q_tokens) else 0.0


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

    # Ask the store which records share a token. This used to be
    # store.list(limit=10**6) plus a Python filter, i.e. every record in the store
    # crossed the process boundary on every query. SQLStore answers it from an
    # indexed label table; Memory/File match the vocabulary in record meta.
    #
    # as_of/valid_at go to the PREFILTER, not just to the per-candidate read. Matching
    # today's vocabulary and then opening yesterday's blob silently drops any document
    # whose terms have since been replaced: a time-travelling query that omits what
    # was relevant at the time.
    n_cand = max(1, int(candidates if candidates is not None else STORE_CANDIDATES))
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

    records.sort(key=lambda r: -_signature_affinity(r.signature or {}, q_tokens))

    scored = []
    for rec in records[:n_cand]:
        try:
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
        scored.append((score_document(doc, qec, q_chi, mode), doc, rec))

    if not scored:
        return [], {"mode": "store", "n_ranked": 0}

    # same deterministic tiebreak as the in-memory path: store enumeration order
    # must not decide which of two equally-scoring documents comes back.
    scored.sort(key=lambda t: (-t[0], str(t[1].doc_id)))
    n_sent = (SECTION_SENTENCES if section_sentences is None
              else max(1, int(section_sentences)))
    sections = []
    for score, doc, rec in scored[:top_k]:
        best = _best_sentences(doc.text, q_tokens, n_sent) if doc.text else []
        sections.append({
            "doc_id": doc.doc_id,
            "text": " … ".join(best) if best else doc.text[:300],
            "score": round(float(score), 4),
            "shared_entities": count_shared_entities(qec.vertex_labels,
                                                     doc.vertex_labels),
            "version": rec.version,
        })
    relation = {"mode": "store", "n_ranked": len(scored),
                "n_records": len(records), "n_opened": len(scored)}
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
    parts = []
    if doc_summary:
        parts.append(doc_summary)
    concepts = relation.get("concepts", [])
    if concepts:
        top = ", ".join(f"{c['concept']} (κ={c['doc_coherence']:.2f})"
                        for c in concepts[:6])
        parts.append(f"\nYour question resonates most with these document "
                     f"concepts: {top}.")
    if sections:
        parts.append("\nMost relevant passages:")
        for i, s in enumerate(sections[:3], 1):
            snippet = s["text"][:280] + ("…" if len(s["text"]) > 280 else "")
            parts.append(f"  {i}. {snippet}")
    if not parts:
        parts.append("No structural context is available for this question yet.")
    parts.append("\n(No language model is configured, so this is a structural "
                 "answer. Connect a model in the Models tab for narrative "
                 "synthesis.)")
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
                 section_sentences: int | None = None) -> dict[str, Any]:
    """End-to-end structural answer for a chat query.

    Builds the query complex, retrieves resonant sections from the
    document/corpus, synthesizes an answer (model or structural), and
    caches the result.
    """
    doc_meta = doc_meta or (getattr(doc_rex, "_agent_meta", {}) if doc_rex is not None else {})
    corpus_id = getattr(corpus, "corpus_id", "") if corpus is not None else ""

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

    q_rex, q_ec = build_query_rex(query)
    q_sig = query_signature(q_rex, q_ec) if q_ec is not None else {"n_concepts": 0}

    sections, relation = retrieve_sections(
        query, top_k, corpus=corpus, doc_rex=doc_rex,
        doc_meta=doc_meta, query_ec=q_ec, section_sentences=section_sentences)

    answer, model_used, token_metrics = synthesize(query, doc_summary, sections, relation)

    payload: dict[str, Any] = {
        "answer": answer,
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
