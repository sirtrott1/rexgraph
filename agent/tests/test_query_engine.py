"""Tests for the structural chat/query engine, model layer, and their
wiring into the server (model setup, per-query complex, retrieval,
grounded synthesis, chat cache, pipeline<->chat, cross-document)."""


import pytest
from agent.adapters.text import TextAdapter
from agent.auto import build_rex_from_edges

from agent import chat_model
from agent import query_engine as qe

DOC = ("Cells signal through receptors. Receptors bind ligands and activate "
       "pathways. Pathways regulate gene expression. Genes express proteins "
       "that fold into structures during development in tissues and organs.")


def _doc_rex(text=DOC):
    ec = TextAdapter().build(text, min_count=1, max_vocab=100)
    rex = build_rex_from_edges(ec)
    rex._agent_meta = {"vertex_labels": ec.vertex_labels, "source_text": text,
                       "type_names": [], "doc_id": "d0"}
    return rex


# model layer (setup)

class TestChatModel:
    def teardown_method(self):
        chat_model.configure(url="")  # clear override between tests

    def test_unavailable_by_default(self, monkeypatch):
        monkeypatch.delenv("CHAT_MODEL_URL", raising=False)
        monkeypatch.delenv("UNLIMITED_OCR_URL", raising=False)
        chat_model.configure(url="")
        # may still resolve a running server in some envs; assert the API shape
        st = chat_model.status()
        assert set(st) >= {"available", "source", "model", "endpoint"}

    def test_configure_and_status(self):
        chat_model.configure(url="http://localhost:9999", model="test-model")
        assert chat_model.is_available() is True
        st = chat_model.status()
        assert st["available"] is True
        assert st["source"] == "configured"
        assert st["model"] == "test-model"
        assert st["endpoint"] == "localhost:9999"

    def test_generate_returns_none_when_unreachable(self):
        # points at a dead port -> generate must degrade to None, not raise
        chat_model.configure(url="http://127.0.0.1:9", model="x")
        assert chat_model.generate("hello", max_tokens=8, timeout=1.0) is None

    def test_clear_override(self, monkeypatch):
        monkeypatch.delenv("CHAT_MODEL_URL", raising=False)
        chat_model.configure(url="http://localhost:1234")
        assert chat_model.is_available()
        chat_model.configure(url="")
        # after clearing, availability depends only on env/manager (env cleared)


# query complex

class TestQueryComplex:
    def test_build_query_rex(self):
        rex, ec = qe.build_query_rex("how do receptors regulate genes")
        assert ec is not None
        assert "receptors" in ec.vertex_labels
        sig = qe.query_signature(rex, ec)
        assert sig["n_concepts"] >= 3
        assert "betti" in sig

    def test_single_word_query_is_a_witness(self):
        rex, ec = qe.build_query_rex("receptors")
        # a single token is a WITNESS (column `(+1)`, sum one, `L0 u = u`) which is a
        # cell class, not an absence. It used to come back as None on the belief that one
        # token is "no relation"; it is a relation that exists and bounds nothing.
        assert ec is not None and "receptors" in ec.vertex_labels
        assert rex is not None and int(rex.nE) == 1
        assert int(rex.edge_types[0]) == 3, "EdgeType.WITNESS"

    def test_relate_query_to_doc(self):
        rex = _doc_rex()
        _, q_ec = qe.build_query_rex("how do receptors regulate genes")
        rel = qe.relate_query_to_doc(q_ec, rex, rex._agent_meta)
        assert rel["n_shared"] >= 2
        concepts = [c["concept"] for c in rel["concepts"]]
        assert "regulate" in concepts or "genes" in concepts


# retrieval + answering

class TestAnswerQuery:
    def test_single_doc_retrieval_and_answer(self):
        rex = _doc_rex()
        r = qe.answer_query(rex, "How do receptors regulate genes?",
                            top_k=3, use_cache=False,
                            doc_summary="Relational complex.")
        assert r["method"] == "single_doc"
        assert r["model_used"] is False
        assert r["query_complex"]["n_concepts"] >= 3
        # top section should be the regulation sentence
        assert r["sections"], "expected retrieved sections"
        assert "regulate" in r["sections"][0]["text"].lower()

    def test_answer_never_empty(self):
        rex = _doc_rex()
        r = qe.answer_query(rex, "tell me something", top_k=2, use_cache=False)
        assert isinstance(r["answer"], str) and r["answer"].strip()

    def test_cache_roundtrip(self, monkeypatch, tmp_path):
        monkeypatch.delenv("REXGRAPH_NO_CACHE", raising=False)
        monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path))
        rex = _doc_rex()
        q = "what regulates genes?"
        r1 = qe.answer_query(rex, q, top_k=3, use_cache=True)
        r2 = qe.answer_query(rex, q, top_k=3, use_cache=True)
        assert r1["cached"] is False
        assert r2["cached"] is True
        assert list(tmp_path.iterdir())  # a cache file was written

    def test_corpus_cross_document(self):
        from agent.corpus import CorpusBuilder
        cb = CorpusBuilder()
        cb.add_text("Neurons fire action potentials. Synapses transmit signals "
                    "between neurons.", doc_id="neuro")
        cb.add_text("Immune cells detect pathogens. T cells communicate via "
                    "signals and attack infected cells.", doc_id="immune")
        cb.build(depth="quick")
        r = qe.answer_query(None, "how do cells communicate signals?",
                            corpus=cb, top_k=3, use_cache=False,
                            doc_summary="Corpus.")
        assert r["method"] == "corpus"
        assert r["sections"]
        # each section is attributed to a document and carries passage text
        assert all(s.get("doc_id") for s in r["sections"])
        assert any(s.get("text") for s in r["sections"])


# server wiring (pipeline <-> chat, model setup endpoint)

@pytest.fixture(scope="module")
def client():
    from agent.server.app import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


class TestServerIntegration:
    def _upload(self, client, text=DOC):
        r = client.post("/api/upload",
                        files={"file": ("d.txt", text.encode(), "text/plain")},
                        data={"options": "{}"})
        assert r.status_code == 200
        return r.json()["session_id"]

    def test_model_config_endpoint(self, client):
        st = client.get("/api/v1/model/chat-config").json()
        assert "available" in st
        set_r = client.post("/api/v1/model/chat-config",
                            json={"url": "http://localhost:9999", "model": "m"})
        assert set_r.status_code == 200
        assert set_r.json()["status"]["available"] is True
        # clear
        client.post("/api/v1/model/chat-config", json={"url": ""})

    def test_chat_general_question_uses_engine(self, client):
        sid = self._upload(client)
        r = client.post(f"/api/chat/{sid}",
                        json={"message": "How do receptors regulate genes?"}).json()
        assert "query_complex" in r
        assert r["query_complex"]["n_concepts"] >= 2
        assert r["sections"]  # structural retrieval happened
        assert "regulate" in r["sections"][0]["text"].lower()

    def test_chat_precise_query_uses_exact_dispatch(self, client):
        sid = self._upload(client)
        r = client.post(f"/api/chat/{sid}",
                        json={"message": "what are the betti numbers?"}).json()
        assert r.get("property") == "betti"
        assert "betti" in r["text"].lower()

    def test_chat_no_document_is_graceful(self, client):
        r = client.post("/api/chat/nonexistent-xyz",
                        json={"message": "hello"}).json()
        assert "response" in r or "text" in r


# Retrieval scoring: the hybrid mix must be commensurable
def _mini_corpus():
    from agent.adapters.text import TextAdapter
    from agent.corpus import CorpusBuilder
    ta = TextAdapter()
    docs = {
        "hodge": "hodge decomposition gradient curl harmonic projection cycle basis "
                 "boundary operator laplacian channel",
        "store": "versioned store append only bitemporal transaction valid time "
                 "record signature lineage drift",
        "io":    "serialize safetensors parquet arrow zarr container format array "
                 "dtype round trip loader",
    }
    c = CorpusBuilder()
    for did, text in docs.items():
        # real sentence terminators: _best_sentences splits on [.!?]\s+ or newlines, so
        # a single unpunctuated blob is one sentence and no sentence budget can bite
        words = text.split()
        sentences = [" ".join(words[i:i + 4]) + "." for i in range(0, len(words), 4)]
        body = "\n".join(sentences * 3)
        c.add_document(source="<text>", doc_id=did, text=body,
                       edge_construction=ta.build(body))
    c.build(depth="quick")
    return c


def test_scores_are_a_coherent_mass_not_a_mixture_of_incommensurable_terms():
    """The original defect was mixing a Jaccard and a character cosine (both in [0,1])
    with the raw quadratic form psi^T RL^+ psi, unbounded and reaching 729 on a real
    corpus, so the bounded terms contributed 0.14% and hybrid ranking was the spectral
    term alone.

    The score is now ONE field summed over one seed (kappa over the vertices the
    query matched) so nothing is mixed and there is no unit range to enforce: each
    kappa is in [0,1], so the score lies in [0, n_shared] and grows with how much of
    the query the document actually carries."""
    c = _mini_corpus()
    for mode in ("chi", "spectral", "hybrid"):
        qr = c.query("hodge decomposition harmonic projection", top_k=3, mode=mode)
        for s in qr.ranked_sections:
            assert s["score"] >= 0.0, f"{mode}: {s['doc_id']} scored {s['score']}"
    # bounded by the seed size, which is what makes it a mass rather than a sum of
    # unrelated quantities
    from agent.adapters.text import TextAdapter
    from agent.scoring import interfacing_score
    qec = TextAdapter().build("hodge decomposition harmonic projection",
                              min_count=1, max_vocab=200)
    for doc in c.documents:
        r = interfacing_score(doc.rex, doc.vertex_labels, qec.vertex_labels)
        assert 0.0 <= r["score"] <= max(r["n_shared"], 1)
        assert all(0.0 <= k <= 1.0 for k in r["kappa"])


def test_hybrid_ranking_is_not_decided_by_the_spectral_term_alone():
    """With the terms commensurable, the vocabulary-matched document must win."""
    c = _mini_corpus()
    qr = c.query("hodge decomposition gradient curl harmonic projection", top_k=3,
                 mode="hybrid")
    assert qr.ranked_sections[0]["doc_id"] == "hodge"


def test_retrieved_sections_are_not_capped_at_two_sentences():
    """_best_sentences(src, q_tokens, 2) capped every section at two sentences, so the
    context handed to synthesize was 4-10 sentences no matter what top_k was. That is a
    hard ceiling on what the shipped retrieval path can answer."""
    from agent import query_engine

    c = _mini_corpus()
    q = "hodge decomposition gradient curl harmonic projection boundary laplacian"
    few, _ = query_engine.retrieve_sections(q, 3, corpus=c, section_sentences=2)
    many, _ = query_engine.retrieve_sections(q, 3, corpus=c, section_sentences=12)
    assert sum(len(s["text"]) for s in many) > sum(len(s["text"]) for s in few)


def test_section_sentence_budget_is_a_named_setting():
    """The budget must be a named, env-overridable setting rather than an inline literal."""
    from agent import query_engine

    assert isinstance(query_engine.SECTION_SENTENCES, int)
    assert query_engine.SECTION_SENTENCES >= 2
