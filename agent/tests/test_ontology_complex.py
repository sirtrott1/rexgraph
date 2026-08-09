"""Tests for ontology mode (RDFS/OWL as a typed complex)."""

import os

import pytest

from agent import ontology_complex as oc


class TestOntologyMapping:
    def test_subclass_chain_is_acyclic_gradient(self):
        d = oc.diagnose_ontology(oc.parse_rdf([
            ("Dog", "subClassOf", "Mammal"),
            ("Mammal", "subClassOf", "Animal"),
            ("Cat", "subClassOf", "Mammal")]))
        assert d["state"] == "acyclic_hierarchy"
        assert d["hodge"]["subsumption_hierarchy"] == 1.0
        assert d["hodge"]["inconsistencies"] == 0.0

    def test_equivalent_class_is_bounded_definition(self):
        d = oc.diagnose_ontology(oc.parse_rdf([
            ("Human", "subClassOf", "Mammal"),
            ("Human", "equivalentClass", "Person")]))
        assert d["state"] == "bounded_definitions"
        assert d["hodge"]["bounded_definitions"] > 0     # the bigon face
        assert ["Human", "Person"] in d["definitions"]

    def test_subsumption_cycle_is_inconsistent(self):
        d = oc.diagnose_ontology(oc.parse_rdf([
            ("A", "subClassOf", "B"),
            ("B", "subClassOf", "C"),
            ("C", "subClassOf", "A")]))
        assert d["state"] == "inconsistent"
        assert d["hodge"]["inconsistencies"] == 1.0

    def test_intersection_definition_is_a_kgon_face(self):
        # C ≡ A ⊓ B modeled as a definition over A, B, C (a triangle face)
        model = oc.parse_rdf([
            ("C", "subClassOf", "A"), ("C", "subClassOf", "B"),
            ("A", "subClassOf", "B")])
        model.definitions.append(["C", "A", "B"])
        d = oc.diagnose_ontology(model)
        assert d["state"] in ("bounded_definitions", "acyclic_hierarchy")
        assert d["hodge"]["inconsistencies"] == 0.0      # the face bounds the loop

    def test_object_property_is_an_edge(self):
        model = oc.parse_rdf([("Person", "hasPet", "Animal")])
        rex, meta = oc.ontology_to_rex(model)
        assert rex is not None and rex.nE == 1 and meta["n_classes"] == 2


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient
    cfg = tmp_path_factory.mktemp("ontcfg")
    os.environ["REXGRAPH_RCDB_URI"] = f"sqlite:///{cfg / 'rcdb.sqlite'}"
    import agent.server.routes.rcdb as R
    R._STORE = None
    from agent.server.app import app
    from agent.server.auth import get_auth_manager
    get_auth_manager().disable_auth(persist=False)
    with TestClient(app) as c:
        yield c
    R._STORE = None


class TestOntologyRoute:
    def test_analyze_and_store(self, client):
        r = client.post("/api/v1/ontology/analyze", json={
            "triples": [["Dog", "subClassOf", "Mammal"],
                        ["Mammal", "subClassOf", "Animal"]],
            "store_id": "animals"}).json()
        assert r["state"] == "acyclic_hierarchy"
        assert r["stored_as"] == "animals"
        # stored as an ontology-tagged complex, queryable like any other
        q = client.post("/api/v1/db/query", json={"tags_any": ["ontology"]}).json()
        assert "animals" in [rec["id"] for rec in q["records"]]

    def test_requires_triples(self, client):
        assert client.post("/api/v1/ontology/analyze", json={}).status_code == 400
