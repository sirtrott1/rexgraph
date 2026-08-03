"""Slice 3: time as a retrieval signal.

The RCDB was already bitemporal and retrieval already passed as_of/valid_at through,
so a corpus could be read AS IT STOOD. What did not exist is time as a ranking
signal: nothing preferred settled evidence over churning evidence, or recent over
stale, once the candidates were structurally scored.

Everything here reads STORED SIGNATURES ONLY. store.history returns records without
touching a blob, and a signature already carries nV/nE/betti1/kappa_mean per version,
so per-candidate temporal features cost dict arithmetic rather than a reconstruction
per version. rcdb.trajectory does the expensive thing (a blob per version plus a
cross-complex bridge per step) and is the wrong tool inside a query.
"""

import pytest
from agent.corpus import CorpusBuilder

from agent import query_engine as qe
from agent import rcdb

SETTLED = ("The boundary map sends an edge to its endpoints. Composing two boundary "
           "maps gives zero. Orientation decides the sign of every entry.")
CHURN_1 = ("The boundary operator carries an edge to its endpoints. Composition "
           "vanishes. Orientation fixes each sign.")
CHURN_2 = ("Boundary columns hold signed arity across witness and branching cases, "
           "with endpoints, orientation, composition and vanishing all reconsidered "
           "together under a much wider vocabulary of structural terms and entries.")


def _put(store, doc_id, text):
    c = CorpusBuilder()
    c.add_text(text, doc_id=doc_id)
    c.build(depth="quick")
    c.persist(store)
    return c


@pytest.fixture
def store():
    st = rcdb.MemoryStore()
    # "settled": written twice with the same content, so version_if_changed keeps it
    # at one version -- nothing about it is in dispute.
    _put(st, "settled", SETTLED)
    _put(st, "settled", SETTLED)
    # "churning": three genuinely different structures under one id.
    _put(st, "churning", SETTLED)
    _put(st, "churning", CHURN_1)
    _put(st, "churning", CHURN_2)
    return st


def test_history_gives_versions_without_opening_a_blob(store):
    """The premise the whole slice rests on."""
    opened = []
    real_get = store.get
    store.get = lambda i, **kw: (opened.append(i), real_get(i, **kw))[1]
    from agent.temporal import temporal_features

    f = temporal_features(store, "churning")
    assert f["n_versions"] == 3
    assert opened == [], f"temporal features opened blobs: {opened}"


def test_a_repeatedly_confirmed_record_is_more_stable_than_a_churning_one(store):
    from agent.temporal import temporal_features

    settled = temporal_features(store, "settled")
    churning = temporal_features(store, "churning")
    assert settled["stability"] > churning["stability"]
    assert 0.0 <= churning["stability"] <= 1.0
    assert 0.0 <= settled["stability"] <= 1.0


def test_a_single_version_record_is_fully_stable(store):
    from agent.temporal import temporal_features

    assert temporal_features(store, "settled")["stability"] == 1.0
    assert temporal_features(store, "settled")["n_versions"] == 1


def test_features_report_drift_direction_and_magnitude(store):
    from agent.temporal import temporal_features

    f = temporal_features(store, "churning")
    assert set(f) >= {"n_versions", "stability", "drift", "direction",
                      "tx_from", "valid_from", "version"}
    assert f["direction"] in ("converging", "diverging", "level")
    assert isinstance(f["drift"], dict) and "nE" in f["drift"]


def test_a_missing_record_yields_neutral_features(store):
    from agent.temporal import temporal_features

    f = temporal_features(store, "nope")
    assert f["n_versions"] == 0 and f["stability"] == 1.0


def test_recency_is_a_rank_within_the_candidates_not_a_decay_constant(store):
    """An absolute half-life would be a magic number with no defensible value. The
    ordering among the actual candidates carries the same information and needs none."""
    from agent.temporal import recency_weights

    w = recency_weights([{"doc_id": "a", "tx_from": 10.0},
                         {"doc_id": "b", "tx_from": 30.0},
                         {"doc_id": "c", "tx_from": 20.0}])
    assert w["b"] > w["c"] > w["a"]
    assert all(0.0 <= v <= 1.0 for v in w.values())


def test_recency_weights_handle_a_single_candidate():
    from agent.temporal import recency_weights

    assert recency_weights([{"doc_id": "only", "tx_from": 5.0}]) == {"only": 1.0}


def test_rerank_off_preserves_the_structural_order(store):
    sections, rel = qe.retrieve_sections("boundary map orientation sign", top_k=5,
                                         store=store, temporal="off")
    base, _ = qe.retrieve_sections("boundary map orientation sign", top_k=5,
                                   store=store)
    assert [s["doc_id"] for s in sections] == [s["doc_id"] for s in base]


def test_rerank_attaches_the_features_it_used(store):
    sections, rel = qe.retrieve_sections("boundary map orientation sign", top_k=5,
                                         store=store, temporal="stability")
    assert sections
    assert rel.get("temporal") == "stability"
    for s in sections:
        assert "temporal" in s
        assert "stability" in s["temporal"]
        assert "structural_score" in s, "the pre-rerank score must stay visible"


def test_stability_rerank_prefers_the_settled_record(store):
    """Both documents answer the query; one has been revised three times under the
    same id. Structural score alone cannot tell them apart."""
    sections, _ = qe.retrieve_sections("boundary map orientation sign endpoints",
                                       top_k=5, store=store, temporal="stability")
    ids = [s["doc_id"] for s in sections]
    assert ids[0] == "settled", [(s["doc_id"], s["score"]) for s in sections]


def test_rerank_never_promotes_a_zero_structural_match(store):
    """Temporal signal reorders relevant results; it does not manufacture relevance."""
    sections, _ = qe.retrieve_sections("xylophone quasar bandwidth", top_k=5,
                                       store=store, temporal="stability")
    assert all(s["score"] == 0.0 for s in sections) or sections == []


def test_an_unknown_temporal_mode_is_an_error_not_a_silent_no_op():
    from agent.temporal import rerank

    with pytest.raises(ValueError):
        rerank([], rcdb.MemoryStore(), mode="vibes")


def test_recency_never_annihilates_a_relevant_document(store):
    """A rank is ordinal: the oldest of two candidates is not '0% recent'. Mapping
    the bottom rank to 0.0 multiplied a structurally strong document (1.48) to
    exactly zero for no reason but its position in the ordering."""
    from agent.temporal import recency_weights

    w = recency_weights([{"doc_id": "old", "tx_from": 1.0},
                         {"doc_id": "new", "tx_from": 2.0}])
    assert w["new"] > w["old"] > 0.0

    sections, _ = qe.retrieve_sections("boundary map orientation sign endpoints",
                                       top_k=5, store=store, temporal="recency")
    for s in sections:
        if s["structural_score"] > 0:
            assert s["score"] > 0, f"{s['doc_id']} was erased by recency alone"


def test_an_unknown_timestamp_ranks_oldest_but_is_not_erased():
    from agent.temporal import recency_weights

    w = recency_weights([{"doc_id": "a", "tx_from": None},
                         {"doc_id": "b", "tx_from": 5.0}])
    assert 0.0 < w["a"] < w["b"]
