"""Tests for the ecosystem-integration routes: vLLM structural routing,
LangChain confidence/analyze, LangGraph state analysis, TrustGraph triple
analysis, HuggingFace axiom compliance. All run standalone (no external
services, no torch/langchain/langgraph install required)."""

import pytest


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from agent.server.app import app
    with TestClient(app) as c:
        yield c


TRIPLES = [["Alice", "knows", "Bob"], ["Bob", "knows", "Carol"],
           ["Carol", "worksAt", "Acme"], ["Alice", "worksAt", "Acme"],
           ["Bob", "manages", "Carol"], ["Carol", "knows", "Alice"]]


class TestVLLMRouter:
    def test_route_returns_capability_and_channel(self, client):
        r = client.post("/api/v1/vllm/route",
                        json={"text": "prove step by step that even plus even is even"})
        assert r.status_code == 200
        d = r.json()
        assert d["routed_to"] in ("reasoning", "creative", "analytical", "multi-hop")
        assert d["dominant_channel"] in ("T", "G", "F", "C")
        assert "character" in d and set(d["character"]) == {"T", "G", "F", "C"}

    def test_route_custom_models(self, client):
        r = client.post("/api/v1/vllm/route", json={
            "text": "analyze the tension between these contradictory claims",
            "models": {"reasoning": "mistral", "creative": "llama",
                       "analytical": "qwen", "multi-hop": "deepseek"}})
        assert r.status_code == 200
        assert r.json()["routed_to"] in ("mistral", "llama", "qwen", "deepseek")

    def test_empty_prompt_rejected(self, client):
        assert client.post("/api/v1/vllm/route", json={"text": ""}).status_code == 400


class TestLangChainRunnable:
    def _sid(self, client):
        doc = b"Cells signal through receptors. Pathways regulate gene expression. Genes express proteins."
        return client.post("/api/upload",
                           files={"file": ("d.txt", doc, "text/plain")},
                           data={"options": "{}"}).json()["session_id"]

    def test_confidence_on_text(self, client):
        r = client.post("/api/v1/langchain/confidence",
                        json={"text": "cells signal receptors genes express proteins pathways"})
        assert r.status_code == 200
        d = r.json()
        assert "verdict" in d and d["verdict"] in ("supported", "low_confidence")
        assert "kappa_mean" in d and "chain_valid" in d

    def test_confidence_on_session(self, client):
        sid = self._sid(client)
        r = client.post("/api/v1/langchain/confidence", json={"session_id": sid})
        assert r.status_code == 200
        assert r.json()["source"].startswith("session:")

    def test_analyze_returns_hodge_and_betti(self, client):
        r = client.post("/api/v1/langchain/analyze",
                        json={"text": "cells signal receptors genes express proteins pathways regulate"})
        assert r.status_code == 200
        d = r.json()
        assert len(d["betti"]) == 3
        assert "hodge" in d and set(d["hodge"]) == {"gradient", "curl", "harmonic"}

    def test_confidence_no_input_rejected(self, client):
        assert client.post("/api/v1/langchain/confidence", json={}).status_code == 400

    def test_tools_listing_still_works(self, client):
        r = client.post("/api/v1/langchain/tools", json={})
        assert r.status_code == 200
        assert len(r.json()["tools"]) == 4


class TestLangGraph:
    def test_default_graph_has_real_topology(self, client):
        """Regression: betti/hodge were None/0 due to wrong result keys."""
        r = client.post("/api/v1/langgraph/state", json={})
        assert r.status_code == 200
        d = r.json()
        assert d["betti"][0] is not None          # was None before the fix
        assert d["hodge"]["gradient"] is not None
        # some hodge energy must be present (not all zero)
        assert (d["hodge"]["gradient"] + d["hodge"]["curl"] + d["hodge"]["harmonic"]) > 0
        assert d["recommendation"] in ("continue", "caution", "stop", "halt")

    def test_pure_cycle_is_curl_dominated(self, client):
        r = client.post("/api/v1/langgraph/state", json={
            "states": ["a", "b", "c"],
            "transitions": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"},
                            {"from": "c", "to": "a"}]})
        assert r.status_code == 200
        d = r.json()
        # a 3-cycle is a pure circulation -> curl dominates
        assert d["hodge"]["curl"] >= d["hodge"]["gradient"]
        assert "channel_profile" in d


class TestTrustGraph:
    def test_analyze_triples_standalone(self, client):
        r = client.post("/api/v1/trustgraph/analyze", json={"triples": TRIPLES})
        assert r.status_code == 200
        d = r.json()
        assert d["n_entities"] == 4
        assert d["n_relations"] == len(TRIPLES)
        assert d["predicate_types"]
        assert len(d["betti"]) == 3

    def test_health_standalone(self, client):
        r = client.post("/api/v1/trustgraph/health", json={"triples": TRIPLES})
        assert r.status_code == 200
        assert r.json()["nV"] == 4

    def test_analyze_requires_triples(self, client):
        assert client.post("/api/v1/trustgraph/analyze", json={}).status_code == 400


class TestHuggingFace:
    def test_text_mode_is_labeled_and_reports_axiom(self, client):
        r = client.post("/api/v1/huggingface/analyze",
                        json={"text": "cells signal receptors genes express proteins pathways regulate"})
        assert r.status_code == 200
        d = r.json()
        assert d["mode"] == "text_cooccurrence"
        assert "chain_condition" in d and "satisfied" in d["chain_condition"]

    def test_requires_some_input(self, client):
        assert client.post("/api/v1/huggingface/analyze", json={}).status_code == 400
