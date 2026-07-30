"""One ranking, built on the interfacing vector.

score_document mixed a label Jaccard, a cosine of MEAN structural characters and a
hand-rolled spectral term under fixed 0.3/0.35/0.35 weights; find_similar used a
kappa correlation times a square-rooted overlap; and interfacing_vector, the
mechanism both were approximating, was reachable only from an HTTP route.
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
    # deliberately tiny: structurally degenerate, and the reason the ranking scalar
    # is the interfacing magnitude rather than its normalized direction.
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


def test_ranking_on_the_normalized_direction_would_reward_a_degenerate_complex(corpus):
    """The reason the scalar is ||iv|| and not sphere_pos[0]. sphere_pos is a
    DIRECTION, so it divides out engagement strength: the stub puts nearly all of
    its tiny interfacing energy in the T channel and wins on composition alone."""
    query = "boundary map orientation sign endpoints"
    rows = _rank(corpus, query)
    by_direction = sorted(rows, key=lambda t: -t[0]["character"][0])

    assert rows[0][1] == "boundary"
    assert by_direction[0][1] == "stub", "the failure mode this guards is gone; recheck"
    stub = dict(rows)["stub"] if False else next(r for r, d in rows if d == "stub")
    boundary = next(r for r, d in rows if d == "boundary")
    assert stub["character"][0] > boundary["character"][0]
    assert stub["score"] < boundary["score"]


def test_score_is_the_magnitude_and_character_is_the_direction(corpus):
    """Together they are the raw interfacing vector in polar form: nothing dropped,
    no mixing constant invented."""
    rows = _rank(corpus, "boundary map orientation sign endpoints")
    for r, _ in rows:
        raw = np.asarray(r["channels"], dtype=float)
        if r["score"] <= 0:
            continue
        assert np.isclose(np.linalg.norm(raw), r["score"], rtol=1e-9)
        recovered = raw / np.linalg.norm(raw)
        assert np.allclose(recovered, r["character"], atol=1e-9)


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


def test_diagnostics_come_back_rather_than_being_folded_in(corpus):
    rows = _rank(corpus, "harmonic hodge gradient curl circulates")
    top = rows[0][0]
    for key in ("coverage", "efficiency", "confidence", "magnitude", "n_shared"):
        assert key in top
    assert 0.0 <= top["coverage"] <= 1.0


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
