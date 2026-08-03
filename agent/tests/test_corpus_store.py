"""Slice 1: a corpus persists into the RCDB and comes back out.

Retrieval could only see documents held in the current process: CorpusBuilder kept
everything in memory and had no path into the store, while the RCDB right next door
already had versioning, bitemporal reads and a queryable structural signature. This
is the producer half of the seam -- one record per document, so the store becomes
the corpus rather than a place a corpus gets copied to.
"""

import numpy as np
import pytest
from agent.corpus import CorpusBuilder

from agent import rcdb

DOCS = {
    "boundary": "The boundary map sends an edge to its endpoints. Composing two "
                "boundary maps gives zero. Orientation decides the sign of each entry.",
    "channels": "Each typed channel weighs the edges differently. The overlap channel "
                "counts shared endpoints. The frustration channel tracks sign conflict.",
    "storage": "Records persist as binary blobs beside a queryable signature. "
               "A signature carries the structure without the payload.",
}


@pytest.fixture
def corpus():
    c = CorpusBuilder()
    for did, text in DOCS.items():
        c.add_text(text, doc_id=did)
    c.build(depth="quick")
    return c


@pytest.fixture
def store():
    return rcdb.MemoryStore()


def test_persist_writes_one_record_per_document(corpus, store):
    ids = corpus.persist(store)
    assert set(ids) == set(DOCS)
    assert {r.id for r in store.list(limit=50)} == set(DOCS)


def test_persisted_records_keep_the_text_and_labels_retrieval_needs(corpus, store):
    corpus.persist(store)
    for doc in corpus.documents:
        rex = store.get(doc.doc_id)
        meta = rex._agent_meta or {}
        assert meta.get("vertex_labels") == doc.vertex_labels
        assert meta.get("source_text", "") == (doc.rex._agent_meta or {}).get("source_text", "")


def test_the_signature_stays_queryable_without_loading_a_blob(corpus, store):
    """put() with an explicit meta drops vertex_labels from the signature, so the
    cheap prefilter loses its labels. Persist has to pass them through."""
    corpus.persist(store)
    for rec in store.list(limit=50):
        assert rec.signature.get("n_labels"), f"{rec.id} lost its labels"
        assert rec.signature.get("labels_sample")
        assert rec.signature["nE"] > 0


def test_persist_records_the_document_identity_in_meta(corpus, store):
    corpus.persist(store)
    for rec in store.list(limit=50):
        assert rec.meta.get("doc_id") == rec.id
        assert rec.meta.get("source")


def test_a_prefix_namespaces_a_corpus_in_a_shared_store(corpus, store):
    ids = corpus.persist(store, prefix="setA/")
    assert set(ids) == {f"setA/{d}" for d in DOCS}
    assert store.get("setA/boundary") is not None


def test_repersisting_an_unchanged_corpus_does_not_mint_versions(corpus, store):
    corpus.persist(store)
    first = {r.id: r.version for r in store.list(limit=50)}
    corpus.persist(store)
    assert {r.id: r.version for r in store.list(limit=50)} == first


def test_tags_land_on_every_record(corpus, store):
    corpus.persist(store, tags=["corpus", "v1"])
    for rec in store.list(limit=50):
        assert "corpus" in rec.signature.get("tags", [])


def test_from_store_rehydrates_a_working_corpus(corpus, store):
    corpus.persist(store)
    back = CorpusBuilder.from_store(store)
    assert {d.doc_id for d in back.documents} == set(DOCS)
    for doc in back.documents:
        assert doc.rex is not None
        assert doc.vertex_labels
        assert doc.text


def test_a_rehydrated_corpus_answers_queries_like_the_original(corpus, store):
    corpus.persist(store)
    back = CorpusBuilder.from_store(store)
    q = "boundary map orientation sign"
    a = [s["doc_id"] for s in corpus.query(q, top_k=3).ranked_sections]
    b = [s["doc_id"] for s in back.query(q, top_k=3).ranked_sections]
    assert a and a == b, f"ranking diverged after a round-trip: {a} vs {b}"


def test_from_store_can_select_a_prefix(corpus, store):
    corpus.persist(store, prefix="setA/")
    other = CorpusBuilder()
    other.add_text(DOCS["storage"], doc_id="only")
    other.build(depth="quick")
    other.persist(store, prefix="setB/")

    back = CorpusBuilder.from_store(store, prefix="setA/")
    assert {d.doc_id for d in back.documents} == set(DOCS)


def test_scoring_is_reusable_outside_the_builder():
    """_score_document touches no instance state; it was a method by placement only.
    A store-backed retriever has to reuse it, not re-implement the ranking."""
    from agent.corpus import score_document

    assert callable(score_document)


def test_a_cached_document_still_carries_its_metadata_on_the_rex():
    """cache.get_rex_and_analysis returns meta as a third value, but the rebuilt
    RexGraph has no _agent_meta. Latent while everything read doc.meta; live once
    the rex blob is what carries labels and source text into the store, because a
    cached corpus then persists documents stripped of both."""

    from agent import cache
    from rexgraph.graph import RexGraph

    rex = RexGraph(sources=np.array([0, 1, 2], np.int32),
                   targets=np.array([1, 2, 0], np.int32))
    rex._agent_meta = {"vertex_labels": ["a", "b", "c"], "source_text": "hello"}
    key = "test-cache-agent-meta"
    assert cache.store_rex_and_analysis(key, rex, {"ok": 1}, dict(rex._agent_meta))
    try:
        back, analysis, meta = cache.get_rex_and_analysis(key)
        assert back is not None and meta.get("source_text") == "hello"
        assert (back._agent_meta or {}).get("vertex_labels") == ["a", "b", "c"]
        assert (back._agent_meta or {}).get("source_text") == "hello"
    finally:
        cache.delete(key) if hasattr(cache, "delete") else None
