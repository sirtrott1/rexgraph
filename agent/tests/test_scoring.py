"""One ranking, built on the reads RexGraph provides for exactly this.

Two earlier designs are buried under this file. The first mixed a label Jaccard, a
cosine of MEAN structural characters and a hand-rolled spectral term under fixed
0.3/0.35/0.35 weights. The second replaced that with ||iv|| from interfacing_vector
called with target=None -- which scores psi against itself rather than interfacing
with anything, and paid a whole-complex bundle for an answer wanted at a few
vertices.

What the library actually offers is demand-driven: coherence_response reads kappa at
just the query's vertices and is identical to coherence[seed] rather than an
approximation, and agentic_reading returns the bounded neighbourhood, the
load-bearing relations, the frustrated entities under a data-adaptive Tukey fence,
and context_size. Relevance is the query's footprint under the document's own
coherence field.
"""

import numpy as np
import pytest

from agent.adapters.text import TextAdapter
from agent.corpus import CorpusBuilder
from agent.scoring import interfacing_score, shared_indices


DOCS = {
    "boundary": "The boundary map sends an edge to its endpoints. Composing two "
                "boundary maps gives zero. Orientation decides the sign of every "
                "entry. A boundary column may carry any signed arity, so a witness "
                "column has one entry and a branching hyperedge has many.",
    "channels": "Each typed channel weighs the edges differently. The overlap channel "
                "counts shared endpoints. The frustration channel tracks sign conflict "
                "between neighbouring edges. The co-participation channel is a "
                "weighted line graph Laplacian.",
    "storage": "Records persist as binary blobs beside a queryable signature. A "
               "signature carries the structure without the payload. Blobs stay "
               "closed until a candidate needs scoring, so the store is cheaper than "
               "holding the corpus in memory.",
    "harmonic": "The harmonic part of a signal is what neither drains nor circulates. "
                "Hodge decomposition splits a flow into gradient, curl and harmonic "
                "pieces. The harmonic dimension counts independent cycles.",
    # deliberately tiny: a degenerate complex must not outrank a real one
    "stub": "An edge has a sign. A map has a boundary. A signature has structure.",
}

QUERIES = {
    "boundary map orientation sign endpoints": "boundary",
    "frustration channel sign conflict overlap": "channels",
    "binary blobs queryable signature payload": "storage",
    "harmonic hodge gradient curl circulates": "harmonic",
    "boundary column signed arity witness branching": "boundary",
    "line graph laplacian co-participation weighted": "channels",
}


@pytest.fixture(scope="module")
def corpus():
    c = CorpusBuilder()
    for did, text in DOCS.items():
        c.add_text(text, doc_id=did)
    c.build(depth="quick")
    return c


def _rank(corpus, query):
    qec = TextAdapter().build(query, min_count=1, max_vocab=200)
    rows = [(interfacing_score(d.rex, d.vertex_labels, qec.vertex_labels), d.doc_id)
            for d in corpus.documents]
    rows.sort(key=lambda t: -t[0]["score"])
    return rows


def test_shared_indices_maps_query_vocabulary_onto_document_vertices():
    assert shared_indices(["a", "b", "c"], ["C", "a", "zz"]) == [2, 0]
    assert shared_indices([], ["a"]) == []
    assert shared_indices(["a"], []) == []


@pytest.mark.parametrize("query,expected", list(QUERIES.items()))
def test_the_relevant_document_ranks_first(corpus, query, expected):
    rows = _rank(corpus, query)
    assert rows[0][1] == expected, [(r[1], round(r[0]["score"], 4)) for r in rows[:3]]


def test_score_is_the_query_footprint_under_the_document_coherence(corpus):
    """One field summed over one seed, so there is no mixing constant to justify:
    score = sum of kappa over the vertices the query matched."""
    import numpy as np

    from agent.scoring import shared_indices

    doc = next(d for d in corpus.documents if d.doc_id == "boundary")
    qec = TextAdapter().build("boundary map orientation", min_count=1, max_vocab=200)
    r = interfacing_score(doc.rex, doc.vertex_labels, qec.vertex_labels)
    seed = np.asarray(shared_indices(doc.vertex_labels, qec.vertex_labels), np.int32)
    assert np.isclose(r["score"], float(np.sum(doc.rex.coherence_response(seed))))


def test_the_demand_driven_read_equals_the_full_field_at_the_seed(corpus):
    """coherence_response is documented as identical to coherence[seed], computed
    only where asked -- not an approximation of it. If that stops holding, the
    cheap read has quietly become a different quantity."""
    import numpy as np

    from agent.scoring import shared_indices

    doc = next(d for d in corpus.documents if d.doc_id == "channels")
    qec = TextAdapter().build("frustration channel overlap", min_count=1, max_vocab=200)
    seed = np.asarray(shared_indices(doc.vertex_labels, qec.vertex_labels), np.int32)
    assert seed.size >= 2
    assert np.allclose(doc.rex.coherence_response(seed),
                       np.asarray(doc.rex.coherence)[seed])


def test_the_agentic_reading_diagnostics_come_back(corpus):
    """context_size is what a correct answer costs; load_bearing are the bridges;
    frustrated is a data-adaptive Tukey outlier, not a threshold I chose."""
    doc = next(d for d in corpus.documents if d.doc_id == "boundary")
    qec = TextAdapter().build("boundary map orientation sign", min_count=1, max_vocab=200)
    r = interfacing_score(doc.rex, doc.vertex_labels, qec.vertex_labels)
    for key in ("context_size", "n_load_bearing", "n_frustrated", "kappa", "kappa_mean"):
        assert key in r
    assert r["context_size"] > 0
    assert len(r["kappa"]) == r["n_shared"]


def test_the_reading_can_be_skipped_for_a_large_candidate_set(corpus):
    """Ranking thousands of candidates should not pay for diagnostics on all of
    them; the score itself is the cheap part."""
    doc = next(d for d in corpus.documents if d.doc_id == "boundary")
    qec = TextAdapter().build("boundary map orientation", min_count=1, max_vocab=200)
    full = interfacing_score(doc.rex, doc.vertex_labels, qec.vertex_labels)
    cheap = interfacing_score(doc.rex, doc.vertex_labels, qec.vertex_labels, reading=False)
    assert cheap["score"] == full["score"]
    assert cheap["context_size"] == 0 and full["context_size"] > 0


def test_no_shared_vocabulary_scores_zero(corpus):
    doc = corpus.documents[0]
    r = interfacing_score(doc.rex, doc.vertex_labels, ["zzzz", "qqqq"])
    assert r["score"] == 0.0 and r["n_shared"] == 0


def test_a_single_shared_token_is_not_a_footprint(corpus):
    """One vertex makes the Poisson source a point; L0^+ of a point carries no
    interfacing structure to read, so it scores zero rather than noise."""
    doc = next(d for d in corpus.documents if d.doc_id == "harmonic")
    one = [doc.vertex_labels[0]]
    assert interfacing_score(doc.rex, doc.vertex_labels, one)["score"] == 0.0




def test_an_empty_or_missing_complex_is_handled(corpus):
    assert interfacing_score(None, ["a"], ["a"])["score"] == 0.0


def test_corpus_scoring_delegates_to_the_shared_scorer(corpus):
    """score_document must not keep a second ranking alive behind the new one."""
    from agent import corpus as corpus_mod

    qec = TextAdapter().build("boundary map orientation", min_count=1, max_vocab=200)
    doc = next(d for d in corpus.documents if d.doc_id == "boundary")
    direct = interfacing_score(doc.rex, doc.vertex_labels, qec.vertex_labels)["score"]
    assert np.isclose(corpus_mod.score_document(doc, qec, None), direct, rtol=1e-9)


def test_find_similar_delegates_to_the_shared_scorer():
    """The third ranking. Same mechanism, so a UI percentage and a retrieval score
    stop disagreeing about what similar means."""
    from agent import rcdb

    c = CorpusBuilder()
    for did, text in DOCS.items():
        c.add_text(text, doc_id=did)
    c.build(depth="quick")
    store = rcdb.MemoryStore()
    c.persist(store)

    probe = next(d for d in c.documents if d.doc_id == "boundary")
    out = rcdb.find_similar(store, probe.rex, probe.vertex_labels,
                            top_k=5, exclude_id="boundary")
    assert out, "find_similar returned nothing"
    assert all(0.0 <= r["match"] for r in out)
    assert [r["match"] for r in out] == sorted((r["match"] for r in out), reverse=True)
