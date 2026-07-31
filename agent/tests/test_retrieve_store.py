"""Slice 2: retrieve_sections ranks over an RCStore.

The consumer half of the seam. retrieve_sections had two modes -- an in-memory
CorpusBuilder, or a single document -- so a persisted corpus was unreachable from
the query path even after slice 1 put it in the store. Store mode closes that, and
ranks with the same score_document the in-memory path uses rather than a second
copy of the ranking.

The prefilter is the point: a signature is queryable without touching a blob, so
only candidates get deserialized.
"""

import pytest

from agent import query_engine as qe
from agent import rcdb
from agent.corpus import CorpusBuilder


DOCS = {
    "boundary": "The boundary map sends an edge to its endpoints. Composing two "
                "boundary maps gives zero. Orientation decides the sign of every entry. "
                "A boundary column may carry any signed arity.",
    "channels": "Each typed channel weighs the edges differently. The overlap channel "
                "counts shared endpoints. The frustration channel tracks sign conflict "
                "between neighbouring edges.",
    "storage": "Records persist as binary blobs beside a queryable signature. "
               "A signature carries the structure without the payload. Blobs stay "
               "closed until a candidate needs scoring.",
}


@pytest.fixture
def store():
    c = CorpusBuilder()
    for did, text in DOCS.items():
        c.add_text(text, doc_id=did)
    c.build(depth="quick")
    st = rcdb.MemoryStore()
    c.persist(st)
    return st


def test_store_mode_returns_ranked_sections(store):
    sections, relation = qe.retrieve_sections(
        "boundary map orientation sign", top_k=3, store=store)
    assert sections, "store mode returned nothing"
    assert relation["mode"] == "store"
    assert sections[0]["doc_id"] in DOCS
    scores = [s["score"] for s in sections]
    assert scores == sorted(scores, reverse=True)


def test_store_mode_returns_text_not_just_ids(store):
    """A retrieval that cannot produce the passage is not RAG. The text rides in the
    blob's _agent_meta, so store mode has to read it back out."""
    sections, _ = qe.retrieve_sections("boundary map orientation", top_k=1, store=store)
    assert sections[0]["text"].strip()


def test_store_mode_ranks_the_relevant_document_first(store):
    for query, expect in [("boundary map orientation sign endpoints", "boundary"),
                          ("frustration channel sign conflict overlap", "channels"),
                          ("binary blobs queryable signature payload", "storage")]:
        sections, _ = qe.retrieve_sections(query, top_k=3, store=store)
        assert sections[0]["doc_id"] == expect, f"{query!r} -> {sections[0]['doc_id']}"


def test_top_k_is_honoured(store):
    for k in (1, 2, 3):
        sections, _ = qe.retrieve_sections("boundary channel signature", top_k=k,
                                           store=store)
        assert len(sections) <= k


def test_store_mode_agrees_with_the_in_memory_corpus(store):
    """Store mode must not be a second ranking: same relevant documents, same order.

    Compared over the RELEVANT results only. The in-memory path scores every document
    and so pads its tail with zero-scoring ones; the store path's label prefilter
    drops anything sharing no vocabulary with the query, which is the point of having
    a prefilter. Agreement on the nonzero prefix is the invariant that matters -- a
    zero-scored document is not a retrieval result.
    """
    back = CorpusBuilder.from_store(store)
    query = "boundary map orientation sign"
    mem = [s["doc_id"] for s in back.query(query, top_k=3).ranked_sections
           if s["score"] > 0]
    sections, _ = qe.retrieve_sections(query, top_k=3, store=store)
    got = [s["doc_id"] for s in sections if s["score"] > 0]
    assert got == mem and got, f"{got} vs {mem}"


def test_the_prefilter_opens_only_candidate_blobs(store):
    """Signature-first is the whole reason to keep structure beside the payload:
    scoring every blob in the store would make persistence a pure cost."""
    opened = []
    real_get = store.get

    def counting_get(id, **kw):
        opened.append(id)
        return real_get(id, **kw)

    store.get = counting_get
    qe.retrieve_sections("blobs signature payload", top_k=1, store=store,
                         candidates=1)
    assert len(opened) == 1, f"opened {opened} for a single candidate"


def test_a_prefix_scopes_retrieval_to_one_corpus(store):
    other = CorpusBuilder()
    other.add_text(DOCS["boundary"], doc_id="copy")
    other.build(depth="quick")
    other.persist(store, prefix="other/")

    sections, _ = qe.retrieve_sections("boundary map orientation", top_k=5,
                                       store=store, prefix="other/")
    assert sections and all(s["doc_id"].startswith("other/") for s in sections)


def test_an_empty_store_falls_back_rather_than_raising():
    sections, relation = qe.retrieve_sections("anything", top_k=3,
                                              store=rcdb.MemoryStore())
    assert sections == [] or relation["mode"] != "store"


def test_store_mode_reads_the_corpus_as_it_stood(store):
    """RCDB is bitemporal, so retrieval can be scoped to a transaction time. A store
    read as_of before anything was written has nothing to rank."""
    sections, relation = qe.retrieve_sections("boundary map", top_k=3, store=store,
                                              as_of=1.0)
    assert sections == [] or relation["mode"] != "store"
