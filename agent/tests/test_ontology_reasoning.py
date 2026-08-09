"""Reasoning and enrichment, against answers that are known independently.

Both are questions with settled answers, so the tests are not shape checks. An
unsatisfiable class is unsatisfiable whatever computes it, and a hypergeometric tail
is an exact rational number that can be written down by hand. Where the structural
reading has no incumbent to check against, what is asserted is the property that
makes it worth having: that it is a metric, and that it is stable.
"""
from __future__ import annotations

import math

import pytest
from agent.enrichment import (
    apply_true_path,
    benjamini_hochberg,
    build_annotation_model,
    compare,
    enrich,
    hypergeometric_sf,
)
from agent.knowledge import join
from agent.ontology_reasoning import (
    build,
    cardinality,
    classification,
    consistency,
    equivalence_classes,
    frustration,
    module_extraction,
    reason,
)

#: a class below two disjoint classes cannot have an instance
INCONSISTENT = [
    ("Cat", "subClassOf", "Mammal"), ("Cat", "subClassOf", "Reptile"),
    ("Mammal", "disjointWith", "Reptile"),
    ("Mammal", "subClassOf", "Animal"), ("Reptile", "subClassOf", "Animal"),
]

#: two disjoint siblings under one parent, which is ordinary and consistent
SIBLINGS = [
    ("Cat", "subClassOf", "Mammal"), ("Dog", "subClassOf", "Mammal"),
    ("Mammal", "disjointWith", "Reptile"),
    ("Mammal", "subClassOf", "Animal"), ("Reptile", "subClassOf", "Animal"),
]

DEEP = [
    ("Tabby", "subClassOf", "Cat"), ("Cat", "subClassOf", "Mammal"),
    ("Tabby", "subClassOf", "Lizard"), ("Lizard", "subClassOf", "Reptile"),
    ("Mammal", "disjointWith", "Reptile"),
]


#### consistency


def test_a_class_below_two_disjoint_classes_is_unsatisfiable():
    out = consistency(build(INCONSISTENT))
    assert out["consistent"] is False
    assert out["n_unsatisfiable"] == 1
    assert out["unsatisfiable"][0]["unsatisfiable_class"] == "Cat"
    assert set(out["unsatisfiable"][0]["disjoint_pair"]) == {"Mammal", "Reptile"}


def test_disjoint_siblings_are_consistent():
    """`Mammal disjointWith Reptile` with both under `Animal` is the most ordinary
    shape in any ontology. A checker that flags it is useless."""
    assert consistency(build(SIBLINGS))["consistent"] is True


def test_the_negative_cycle_alone_does_not_decide():
    """Both cases close a cycle whose signs multiply to -1. Only one is a
    contradiction, which is why orientation and not the sign settles it."""
    assert frustration(build(SIBLINGS))["n_frustrated"] >= 1
    assert consistency(build(SIBLINGS))["consistent"] is True
    assert frustration(build(INCONSISTENT))["n_frustrated"] >= 1
    assert consistency(build(INCONSISTENT))["consistent"] is False


def test_descent_is_followed_through_several_levels():
    out = consistency(build(DEEP))
    assert out["n_unsatisfiable"] == 1
    f = out["unsatisfiable"][0]
    assert f["unsatisfiable_class"] == "Tabby"
    assert f["path_to_first"] == ["Tabby", "Cat", "Mammal"]
    assert f["path_to_second"] == ["Tabby", "Lizard", "Reptile"]


def test_a_class_disjoint_from_its_own_ancestor_is_unsatisfiable():
    out = consistency(build([("Cat", "subClassOf", "Mammal"),
                             ("Cat", "disjointWith", "Mammal")]))
    assert out["consistent"] is False
    assert out["unsatisfiable"][0]["unsatisfiable_class"] == "Cat"


def test_an_ontology_with_no_disjointness_is_consistent():
    out = consistency(build([("Cat", "subClassOf", "Mammal"),
                             ("Mammal", "subClassOf", "Animal")]))
    assert out["consistent"] is True
    assert out["n_disjointness_axioms"] == 0


def test_a_finding_carries_both_paths_that_produced_it():
    """"Unsatisfiable" with no reason is not something a curator can act on."""
    for f in consistency(build(INCONSISTENT))["unsatisfiable"]:
        assert f["path_to_first"] and f["path_to_second"]
        assert f["summary"].strip()
        assert f["path_to_first"][0] == f["unsatisfiable_class"]


def test_a_hierarchy_cycle_does_not_hang_the_descent():
    out = consistency(build([("A", "subClassOf", "B"), ("B", "subClassOf", "A"),
                             ("A", "disjointWith", "C")]))
    assert isinstance(out["consistent"], bool)


def test_holonomy_is_reported_separately_from_consistency():
    out = consistency(build(SIBLINGS))
    assert "holonomy" in out
    assert set(out["holonomy"]) >= {"n_frustrated", "balanced",
                                    "n_independent_cycles"}


#### classification and modules


def test_the_disjointness_edge_is_the_negative_one():
    rc = build(INCONSISTENT)
    import numpy as np
    signs = np.asarray(rc.rex._edge_signs)
    negative = set(np.nonzero(signs < 0)[0].tolist())
    disjoint = {i for i, r in enumerate(rc.roles) if r == "disjoint"}
    assert negative == disjoint


def test_predicate_roles_are_recognised():
    from agent.ontology_reasoning import classify_predicate
    assert classify_predicate("rdfs:subClassOf") == "subsumption"
    assert classify_predicate("is_a") == "subsumption"
    assert classify_predicate("owl:disjointWith") == "disjoint"
    assert classify_predicate("owl:equivalentClass") == "equivalence"
    assert classify_predicate("part_of") == "relation"


def test_classification_returns_relative_homology():
    out = classification(build(SIBLINGS))
    assert len(out["betti_relative"]) >= 2
    assert out["chain_valid"] is True
    assert isinstance(out["hodge_full"], dict)


def test_equivalence_classes_are_computed():
    out = equivalence_classes(build(SIBLINGS))
    assert out["n_classes"] >= 1
    assert out["n_collapsed"] == 4          # the four subsumption axioms


def test_an_ontology_with_no_hierarchy_says_so():
    out = equivalence_classes(build([("A", "relatedTo", "B")]))
    assert out["n_classes"] == 0 and "note" in out


def test_module_extraction_names_the_axioms_it_kept():
    out = module_extraction(build(SIBLINGS), ["Mammal"])
    assert out["n_axioms"] > 0
    assert all("Mammal" in a for a in out["axioms"])
    assert len(out["betti_relative"]) >= 2


def test_module_extraction_over_absent_terms_says_so():
    out = module_extraction(build(SIBLINGS), ["NotATerm"])
    assert out["n_axioms"] == 0 and "note" in out


def test_cardinality_counts_relations_per_class():
    out = cardinality(build(SIBLINGS))
    rows = {r["class"]: r["counts"] for r in out["classes"]}
    assert rows["Cat"]["subclassof"] == 1
    assert rows["Mammal"]["disjointwith"] == 1


def test_reason_answers_everything_at_once():
    out = reason(INCONSISTENT, terms=["Cat"])
    assert out["consistency"]["consistent"] is False
    assert "equivalence" in out and "classification" in out
    assert out["module"]["n_axioms"] > 0


def test_an_empty_axiom_set_is_refused():
    with pytest.raises(ValueError, match="no relations"):
        build([])


#### enrichment: the classical answer


OBO = """format-version: 1.2

[Term]
id: GO:0006281
name: DNA repair
is_a: GO:0006974

[Term]
id: GO:0006974
name: response to DNA damage
is_a: GO:0008150

[Term]
id: GO:0008150
name: biological_process

[Term]
id: GO:0006955
name: immune response
is_a: GO:0008150
"""

REPAIR = ["BRCA1", "BRCA2", "ATM", "RAD51", "PALB2", "CHEK2"]
IMMUNE = ["IL2", "IL6", "TNF", "IFNG", "CD4", "CD8A"]


def _gaf(pairs) -> str:
    rows = ["!gaf-version: 2.2"]
    for i, (sym, term) in enumerate(pairs):
        rows.append("\t".join([
            "UniProtKB", f"P{i:05d}", sym, "involved_in", term, "PMID:1", "IDA",
            "", "P", f"{sym} protein", sym, "protein", "taxon:9606", "2026",
            "U", "", ""]))
    return "\n".join(rows) + "\n"


@pytest.fixture
def study(tmp_path):
    """Twelve genes: six annotated to DNA repair, six to immune response."""
    obo = tmp_path / "go.obo"
    obo.write_text(OBO)
    gaf = tmp_path / "goa.gaf"
    gaf.write_text(_gaf([(g, "GO:0006281") for g in REPAIR]
                        + [(g, "GO:0006955") for g in IMMUNE]))
    return join(str(obo), str(gaf))


def test_the_hypergeometric_tail_is_exact():
    """P(X >= 4) with N=12, K=6, n=4 is C(6,4)/C(12,4) = 15/495."""
    assert hypergeometric_sf(4, 12, 6, 4) == pytest.approx(15 / 495)


def test_the_tail_of_a_certain_event_is_one():
    assert hypergeometric_sf(0, 10, 5, 3) == 1.0


def test_an_impossible_overlap_has_probability_zero():
    assert hypergeometric_sf(6, 12, 5, 4) == 0.0


def test_benjamini_hochberg_is_monotone_and_bounded():
    ps = [0.001, 0.01, 0.02, 0.5, 0.9]
    qs = benjamini_hochberg(ps)
    assert all(q <= 1.0 for q in qs)
    assert all(q >= p for q, p in zip(qs, ps, strict=False))
    assert qs == sorted(qs), "q must not decrease as p increases"


def test_the_true_path_rule_propagates_to_ancestors(study):
    """Without it every ancestor reads as empty and the hierarchy contributes
    nothing, which is the whole reason the ontology is there."""
    model = apply_true_path(build_annotation_model(study))
    assert len(model.closed["DNA repair"]) == 6
    assert len(model.closed["response to DNA damage"]) == 6, \
        "the parent term did not inherit its child's annotations"
    assert len(model.closed["biological_process"]) == 12


def test_the_enriched_term_is_the_right_one(study):
    out = enrich(study, ["BRCA1", "BRCA2", "ATM", "RAD51"])
    assert out["terms"][0]["term"] in ("DNA repair", "response to DNA damage")
    assert out["terms"][0]["p_value"] == pytest.approx(15 / 495)
    assert out["terms"][0]["fold_enrichment"] == pytest.approx(2.0)


def test_the_other_gene_set_enriches_the_other_term(study):
    out = enrich(study, IMMUNE[:4])
    assert out["terms"][0]["term"] == "immune response"


def test_the_root_term_is_never_enriched(study):
    """Everything is under `biological_process`, so it can carry no signal."""
    out = enrich(study, ["BRCA1", "BRCA2", "ATM", "RAD51"])
    root = [t for t in out["terms"] if t["term"] == "biological_process"]
    assert root and root[0]["p_value"] == pytest.approx(1.0)
    assert root[0]["fold_enrichment"] == pytest.approx(1.0)


def test_the_universe_is_the_annotated_entities(study):
    out = enrich(study, ["BRCA1"])
    assert out["n_universe"] == 12


def test_a_gene_outside_the_annotation_set_is_counted_as_unmapped(study):
    out = enrich(study, ["BRCA1", "NOT_A_GENE"])
    assert out["n_study"] == 1 and out["n_study_unmapped"] == 1


def test_an_explicit_universe_is_honoured(study):
    out = enrich(study, ["BRCA1", "BRCA2"], universe=REPAIR)
    assert out["n_universe"] == 6


def test_every_reported_term_carries_its_genes(study):
    for t in enrich(study, REPAIR[:4])["terms"]:
        assert t["entities"], f"{t['term']} reports no members"
        assert t["n_study"] == len(t["entities"]) or t["n_study"] > 25


#### enrichment: the structural reading


def test_the_structural_reading_is_produced(study):
    s = enrich(study, REPAIR[:4])["structure"]
    assert s["available"] is True
    assert s["n_features"] > 0


def test_no_lifetime_is_an_essential_bar_in_disguise(study):
    """The kernel writes an essential death as 1e308 rather than inf. Reading the
    magnitude instead of the classification reported a lifetime of 1e308."""
    s = enrich(study, REPAIR[:4])["structure"]
    assert math.isfinite(s["longest_lifetime"])
    assert s["longest_lifetime"] < 1e300
    assert all(math.isfinite(b) and math.isfinite(d) for b, d in s["barcodes"])


def test_comparing_a_reading_with_itself_is_zero(study):
    """The metric axiom. Without it the distance means nothing."""
    s = enrich(study, REPAIR[:4])["structure"]
    out = compare(s, s)
    assert out["bottleneck"] == pytest.approx(0.0)
    assert out["wasserstein"] == pytest.approx(0.0)


def test_two_different_studies_are_a_positive_distance_apart(study):
    a = enrich(study, REPAIR[:4])["structure"]
    b = enrich(study, IMMUNE[:4])["structure"]
    out = compare(a, b)
    assert out["available"] and out["bottleneck"] > 0


def test_a_study_set_that_reaches_nothing_says_so(study):
    s = enrich(study, ["NOT_A_GENE"])["structure"]
    assert s["available"] is False and s["reason"]


#### the routes


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "r.sqlite"))
    from agent.rcdb import reset_default_store
    reset_default_store()
    from agent.server.app import app
    from fastapi.testclient import TestClient
    yield TestClient(app)
    reset_default_store()


def _triples_text(triples):
    return "\n".join(" ".join(t) for t in triples)


def test_the_reason_route_finds_an_unsatisfiable_class(client):
    r = client.post("/api/v1/ontology/reason",
                    data={"text": _triples_text(INCONSISTENT), "format": "triples"})
    assert r.status_code == 200, r.text[:200]
    c = r.json()["consistency"]
    assert c["consistent"] is False
    assert c["unsatisfiable"][0]["unsatisfiable_class"] == "Cat"
    assert c["unsatisfiable"][0]["path_to_first"]


def test_the_reason_route_passes_a_consistent_ontology(client):
    r = client.post("/api/v1/ontology/reason",
                    data={"text": _triples_text(SIBLINGS), "format": "triples"})
    assert r.json()["consistency"]["consistent"] is True


def test_the_reason_route_reads_an_obo_file(client):
    r = client.post("/api/v1/ontology/reason",
                    files=[("files", ("go.obo", OBO.encode(), "text/plain"))])
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body["n_terms"] > 0 and body["source"]["format"] == "obo"


def test_the_reason_route_returns_a_module_when_terms_are_given(client):
    r = client.post("/api/v1/ontology/reason",
                    data={"text": _triples_text(SIBLINGS), "format": "triples",
                          "terms": "Mammal"})
    assert r.json()["module"]["n_axioms"] > 0


def test_the_reason_route_needs_an_input(client):
    assert client.post("/api/v1/ontology/reason", data={}).status_code == 400


def test_the_enrichment_route_ranks_the_right_term(client):
    gaf = _gaf([(g, "GO:0006281") for g in REPAIR]
               + [(g, "GO:0006955") for g in IMMUNE])
    r = client.post("/api/v1/enrichment/run",
                    files=[("files", ("go.obo", OBO.encode(), "text/plain")),
                           ("files", ("goa.gaf", gaf.encode(), "text/plain"))],
                    data={"study": "BRCA1,BRCA2,ATM,RAD51"})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["n_universe"] == 12 and body["n_study"] == 4
    assert body["terms"][0]["term"] in ("DNA repair", "response to DNA damage")
    assert body["terms"][0]["p_value"] == pytest.approx(15 / 495)
    assert body["structure"]["available"] is True


def test_the_enrichment_route_accepts_whitespace_separated_entities(client):
    gaf = _gaf([(g, "GO:0006281") for g in REPAIR])
    r = client.post("/api/v1/enrichment/run",
                    files=[("files", ("go.obo", OBO.encode(), "text/plain")),
                           ("files", ("goa.gaf", gaf.encode(), "text/plain"))],
                    data={"study": "BRCA1 BRCA2  ATM"})
    assert r.json()["n_study"] == 3


def test_the_enrichment_route_warns_when_no_annotations_were_supplied(client):
    """An ontology alone has nothing to enrich against, and silently returning an
    empty table would read as "no enrichment found"."""
    r = client.post("/api/v1/enrichment/run",
                    files=[("files", ("go.obo", OBO.encode(), "text/plain"))],
                    data={"study": "BRCA1"})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["n_universe"] == 0
    assert "annotation" in body["warning"]


def test_the_enrichment_route_needs_a_study_set(client):
    r = client.post("/api/v1/enrichment/run",
                    files=[("files", ("go.obo", OBO.encode(), "text/plain"))],
                    data={"study": "  "})
    assert r.status_code == 400


def test_the_reason_route_hands_back_the_signed_complex(client, tmp_path):
    """The report is a reading of an object; the object is what lets someone else
    reproduce or extend the reasoning."""
    r = client.post("/api/v1/ontology/reason",
                    data={"text": _triples_text(INCONSISTENT), "format": "triples",
                          "download": "rex"})
    assert r.status_code == 200, r.text[:200]
    assert r.headers["content-type"] == "application/octet-stream"

    import io
    import zipfile
    out = str(tmp_path / "reasoning.rex")
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(out)
    from rexgraph.io import load_rex
    rex = load_rex(out)
    assert int(rex.nE) == len(INCONSISTENT)


def test_the_downloaded_complex_carries_the_disjointness_as_a_sign(client, tmp_path):
    """The negative sign is the whole encoding of negation, so it has to survive."""
    import io
    import zipfile

    import numpy as np
    r = client.post("/api/v1/ontology/reason",
                    data={"text": _triples_text(INCONSISTENT), "format": "triples",
                          "download": "rex"})
    out = str(tmp_path / "reasoning.rex")
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(out)
    from rexgraph.io import load_rex
    signs = np.asarray(load_rex(out)._edge_signs)
    assert (signs < 0).sum() == 1, "the disjointness axiom lost its sign in transit"


def test_the_reason_route_refuses_an_unknown_container(client):
    r = client.post("/api/v1/ontology/reason",
                    data={"text": _triples_text(SIBLINGS), "format": "triples",
                          "download": "csv"})
    assert r.status_code == 400
