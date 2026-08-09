"""Ontology files, read as the formats ontologies actually ship as.

The Ontology screen used to accept one thing: whitespace-separated triples typed into
a box. GO, HPO, MONDO, ChEBI and UBERON ship `.obo` and `.owl`; GO's current
distribution is OBO Graphs JSON; and what connects any of them to biology is a
`.gaf`. None of that could be loaded.

The fixtures here are real-shaped fragments of those formats, small enough to assert
against exactly. Every parser is checked for what it extracts *and* for what it must
not invent, since a reader that silently drops half an ontology diagnoses a clean
hierarchy.
"""
from __future__ import annotations

import pytest
from agent.adapters import ontology_formats as OF
from agent.ontology_complex import (
    diagnose_ontology,
    parse_rdf,
    subsumption_cycles,
)

OBO = """format-version: 1.2
data-version: releases/2026-07-01
ontology: go

[Term]
id: GO:0006915
name: apoptotic process
namespace: biological_process
is_a: GO:0012501 ! programmed cell death
relationship: part_of GO:0008219

[Term]
id: GO:0012501
name: programmed cell death
namespace: biological_process
is_a: GO:0008219 ! cell death

[Term]
id: GO:0008219
name: cell death
namespace: biological_process

[Term]
id: GO:0000001
name: mitochondrion inheritance
namespace: biological_process
is_obsolete: true

[Typedef]
id: part_of
name: part of
is_transitive: true
"""

GAF = "\n".join([
    "!gaf-version: 2.2",
    "\t".join(["UniProtKB", "P04637", "TP53", "involved_in", "GO:0006915",
               "PMID:1", "IDA", "", "P", "Cellular tumor antigen p53", "TP53",
               "protein", "taxon:9606", "20260101", "UniProt", "", ""]),
    "\t".join(["UniProtKB", "Q07817", "BCL2L1", "NOT|involved_in", "GO:0006915",
               "PMID:2", "IMP", "", "P", "Bcl-2-like protein 1", "",
               "protein", "taxon:9606", "20260101", "UniProt", "", ""]),
    "\t".join(["UniProtKB", "P10415", "BCL2", "located_in", "GO:0005739",
               "PMID:3", "IDA", "", "C", "Apoptosis regulator Bcl-2", "",
               "protein", "taxon:9606", "20260101", "UniProt", "", ""]),
]) + "\n"

TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://ex.org/> .
ex:Dog a owl:Class ; rdfs:subClassOf ex:Mammal ; rdfs:label "Dog" .
ex:Cat a owl:Class ; rdfs:subClassOf ex:Mammal .
ex:Mammal rdfs:subClassOf ex:Animal .
ex:Human owl:equivalentClass ex:Person ; rdfs:subClassOf ex:Mammal .
"""

NT = """<http://ex.org/Dog> <http://www.w3.org/2000/01/rdf-schema#subClassOf> <http://ex.org/Mammal> .
<http://ex.org/Cat> <http://www.w3.org/2000/01/rdf-schema#subClassOf> <http://ex.org/Mammal> .
<http://ex.org/Mammal> <http://www.w3.org/2000/01/rdf-schema#label> "Mammal" .
"""

OWL = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://ex.org/Dog">
    <rdfs:label>Dog</rdfs:label>
    <rdfs:subClassOf rdf:resource="http://ex.org/Mammal"/>
  </owl:Class>
  <owl:Class rdf:about="http://ex.org/Mammal">
    <rdfs:subClassOf rdf:resource="http://ex.org/Animal"/>
  </owl:Class>
  <owl:Class rdf:about="http://ex.org/Human">
    <owl:equivalentClass rdf:resource="http://ex.org/Person"/>
  </owl:Class>
</rdf:RDF>
"""

OBOGRAPH = """{"graphs":[{"id":"http://purl.obolibrary.org/obo/go.owl",
"nodes":[{"id":"GO:0006915","lbl":"apoptotic process"},
         {"id":"GO:0012501","lbl":"programmed cell death"},
         {"id":"GO:0008219","lbl":"cell death"}],
"edges":[{"sub":"GO:0006915","pred":"is_a","obj":"GO:0012501"},
         {"sub":"GO:0012501","pred":"is_a","obj":"GO:0008219"},
         {"sub":"GO:0006915","pred":"http://purl.obolibrary.org/obo/BFO_0000050",
          "obj":"GO:0008219"}]}]}"""

SAMPLES = {"obo": OBO, "gaf": GAF, "turtle": TTL, "ntriples": NT,
           "rdfxml": OWL, "obograph": OBOGRAPH}


#### detection


@pytest.mark.parametrize("fmt,text", list(SAMPLES.items()))
def test_a_format_is_recognised_from_its_content(fmt, text):
    """A pasted ontology has no filename, and GO ships OBO Graphs as `go.json`, so
    the extension cannot be the only signal."""
    assert OF.sniff_format(text) == fmt


@pytest.mark.parametrize("name,ext", [
    ("obo", "go.obo"), ("rdfxml", "hp.owl"), ("turtle", "x.ttl"),
    ("ntriples", "x.nt"), ("gaf", "goa_human.gaf"), ("gpad", "x.gpad"),
])
def test_a_format_is_recognised_from_its_extension(name, ext):
    assert OF.format_for_extension(ext) == name


def test_a_gzipped_ontology_is_recognised_by_its_inner_extension():
    """Annotation sets ship compressed."""
    assert OF.format_for_extension("goa_human.gaf.gz") == "gaf"


def test_an_unrecognisable_input_is_refused_with_the_list():
    with pytest.raises(ValueError, match="obo"):
        OF.parse("\x00\x01\x02")


def test_an_unknown_format_name_is_refused():
    with pytest.raises(ValueError, match="unknown ontology format"):
        OF.parse(OBO, "sparql")


#### OBO


def test_obo_reads_the_hierarchy():
    p = OF.parse(OBO, "obo")
    assert ("GO:0006915", "is_a", "GO:0012501") in p.triples
    assert ("GO:0012501", "is_a", "GO:0008219") in p.triples


def test_obo_reads_a_relationship_as_its_own_relation():
    """`relationship: part_of X` is a part_of edge, not an is_a edge. Collapsing the
    two would make every partonomy read as subsumption."""
    p = OF.parse(OBO, "obo")
    assert ("GO:0006915", "part_of", "GO:0008219") in p.triples


def test_obo_keeps_the_names():
    p = OF.parse(OBO, "obo")
    assert p.labels["GO:0006915"] == "apoptotic process"
    assert p.labels["GO:0008219"] == "cell death"


def test_obo_drops_the_trailing_comment_from_a_reference():
    """`is_a: GO:0012501 ! programmed cell death` references an id; the text after
    `!` is a courtesy and is not part of it."""
    p = OF.parse(OBO, "obo")
    assert all("!" not in o and " " not in o for _s, _r, o in p.triples)


def test_obo_counts_obsolete_terms_without_relating_them():
    p = OF.parse(OBO, "obo")
    assert p.meta["n_obsolete"] == 1
    assert "GO:0000001" in p.meta["obsolete"]
    assert not [t for t in p.triples if "GO:0000001" in t]


def test_obo_reports_what_the_file_declared():
    p = OF.parse(OBO, "obo")
    assert p.meta["n_terms"] == 4
    assert p.meta["n_typedefs"] == 1
    assert p.meta["namespaces"] == {"biological_process": 4}


def test_obo_does_not_duplicate_an_intersection_conjunct():
    """`intersection_of: part_of GO:x` is one axiom. Writing both the relation and a
    membership edge would put two parallel edges on it, which reads as a 2-cycle the
    term does not have."""
    text = OBO + """
[Term]
id: GO:0097194
name: execution phase of apoptosis
intersection_of: part_of GO:0006915
"""
    p = OF.parse(text, "obo")
    got = [t for t in p.triples if t[0] == "GO:0097194"]
    assert len(got) == 1, f"one axiom produced {len(got)} edges: {got}"


#### OBO Graphs JSON


def test_obograph_reads_nodes_and_edges():
    p = OF.parse(OBOGRAPH, "obograph")
    assert ("GO:0006915", "is_a", "GO:0012501") in p.triples
    assert p.labels["GO:0006915"] == "apoptotic process"
    assert p.meta["n_terms"] == 3


def test_obograph_reduces_a_relation_iri_to_its_name():
    p = OF.parse(OBOGRAPH, "obograph")
    assert ("GO:0006915", "BFO_0000050", "GO:0008219") in p.triples


def test_json_that_is_not_obographs_is_refused():
    with pytest.raises(ValueError, match="OBO Graphs"):
        OF.parse('{"nodes": 1}', "obograph")


#### RDF serialisations


def test_turtle_reads_prefixed_names():
    p = OF.parse(TTL, "turtle")
    assert ("Dog", "subClassOf", "Mammal") in p.triples
    assert ("Human", "equivalentClass", "Person") in p.triples


def test_turtle_keeps_a_label_out_of_the_relations():
    p = OF.parse(TTL, "turtle")
    assert p.labels.get("Dog") == "Dog"
    assert not [t for t in p.triples if t[1] in ("label", "prefLabel")]


def test_a_type_declaration_is_not_a_relation_between_classes():
    """`ex:Dog a owl:Class` says what Dog is, not what it relates to. Emitting it as
    an edge puts every class on a star through one vertex called Class, which is a
    hub the ontology does not contain."""
    for fmt, text in (("turtle", TTL), ("rdfxml", OWL)):
        p = OF.parse(text, fmt)
        assert not [t for t in p.triples if t[2] == "Class"], \
            f"{fmt} emitted a type declaration as a relation"
        assert p.declarations.get("Dog") == "Class"


def test_rdfxml_reads_the_class_skeleton():
    p = OF.parse(OWL, "rdfxml")
    assert ("Dog", "subClassOf", "Mammal") in p.triples
    assert ("Mammal", "subClassOf", "Animal") in p.triples
    assert ("Human", "equivalentClass", "Person") in p.triples


def test_rdfxml_predicates_carry_no_namespace_residue():
    """ElementTree hands back `{ns}tag`. A predicate of `}subClassOf` matches none of
    the vocabularies and would silently become an untyped object relation."""
    p = OF.parse(OWL, "rdfxml")
    assert all(not r.startswith("}") and "{" not in r for _s, r, _o in p.triples)


def test_rdfxml_counts_the_classes_it_found():
    assert OF.parse(OWL, "rdfxml").meta["n_classes"] == 3


def test_malformed_xml_is_refused():
    with pytest.raises(ValueError, match="RDF/XML"):
        OF.parse("<rdf:RDF><unclosed>", "rdfxml")


def test_ntriples_reads_a_line_per_statement():
    p = OF.parse(NT, "ntriples")
    assert ("Dog", "subClassOf", "Mammal") in p.triples
    assert p.labels["Mammal"] == "Mammal"
    assert p.meta["unparsed"] == 0


#### annotations


def test_gaf_relates_a_gene_product_to_a_term():
    p = OF.parse(GAF, "gaf")
    assert ("TP53", "involved_in", "GO:0006915") in p.triples
    assert ("BCL2", "located_in", "GO:0005739") in p.triples


def test_gaf_does_not_assert_a_negative_annotation():
    """A `NOT|involved_in` row states the relation does NOT hold. Emitting it would
    record the opposite of what the file says."""
    p = OF.parse(GAF, "gaf")
    assert not [t for t in p.triples if t[0] == "BCL2L1"]
    assert p.meta["n_negative"] == 1


def test_gaf_keeps_the_gene_symbol_as_the_name():
    """Column 3 is the symbol and column 10 is a long product description. The
    symbol is what a reader knows the gene by."""
    p = OF.parse(GAF, "gaf")
    assert "TP53" in [s for s, _p, _o in p.triples]
    assert p.named_triples()[0][0] == "TP53"


def test_gaf_reports_its_evidence_and_taxa():
    p = OF.parse(GAF, "gaf")
    assert p.meta["evidence_codes"].get("IDA") == 2
    assert p.meta["taxa"].get("taxon:9606") == 2


def test_gpad_reads_its_own_column_layout():
    gpad = ("!gpad-version: 2.0\n"
            "UniProtKB\tP04637\tinvolved_in\tGO:0006915\tPMID:1\tIDA\n")
    p = OF.parse(gpad, "gpad")
    assert ("UniProtKB:P04637", "involved_in", "GO:0006915") in p.triples


#### combining: the point of reading annotations


def test_an_annotation_resolves_against_its_ontology():
    """A `.gaf` names GO:0006915 and says nothing about it. Loaded with the `.obo`,
    the gene reaches a named term in the hierarchy instead of a bare accession."""
    both = OF.combine(OF.parse(OBO, "obo"), OF.parse(GAF, "gaf"))
    named = both.named_triples()
    assert ("TP53", "involved_in", "apoptotic process") in named, \
        "the annotation did not resolve against the ontology"


def test_combining_keeps_both_files_relations():
    a, b = OF.parse(OBO, "obo"), OF.parse(GAF, "gaf")
    both = OF.combine(a, b)
    assert len(both.triples) == len(a.triples) + len(b.triples)
    assert both.meta["format"] == "obo+gaf"


def test_combining_drops_a_duplicate_assertion():
    """Two files asserting the same relation is one relation."""
    both = OF.combine(OF.parse(OBO, "obo"), OF.parse(OBO, "obo"))
    assert len(both.triples) == len(OF.parse(OBO, "obo").triples)


def test_combining_nothing_does_not_crash():
    assert len(OF.combine()) == 0


#### into a complex


@pytest.mark.parametrize("fmt,text", list(SAMPLES.items()))
def test_every_format_builds_an_edge_construction(fmt, text):
    """An ontology file has to work everywhere a document works."""
    ec = OF.to_edge_construction(OF.parse(text, fmt))
    assert ec.nE > 0
    assert len(ec.vertex_labels) > 0


def test_the_predicate_becomes_the_edge_type():
    ec = OF.to_edge_construction(OF.parse(OBO, "obo"))
    assert set(ec.type_names) >= {"is_a", "part_of"}
    assert ec.n_types >= 2


def test_an_ontology_with_no_relations_is_refused():
    with pytest.raises(ValueError, match="no relations"):
        OF.to_edge_construction(OF.ParsedOntology([], {}, {}, {}))


#### the diagnosis over real ontology shapes


def test_a_subsumption_cycle_is_an_inconsistency():
    model = parse_rdf([("A", "is_a", "B"), ("B", "is_a", "C"), ("C", "is_a", "A")])
    assert subsumption_cycles(model) == 1
    assert diagnose_ontology(model)["state"] == "inconsistent"


def test_a_diamond_through_two_relations_is_not_an_inconsistency():
    """GO does this constantly: `apoptotic process is_a programmed cell death is_a
    cell death` alongside `apoptotic process part_of cell death`. The loop closes
    through part_of, so the hierarchy itself is acyclic and nothing subsumes itself.
    Calling it inconsistent would make every real ontology inconsistent."""
    model = parse_rdf(OF.parse(OBO, "obo").triples)
    assert subsumption_cycles(model) == 0
    report = diagnose_ontology(model)
    assert report["state"] == "multiple_inheritance"
    assert report["hodge"]["inconsistency_dimension"] == 0
    assert report["hodge"]["multi_relation_cycle_dimension"] >= 1


def test_a_clean_hierarchy_reads_as_one():
    model = parse_rdf([("Dog", "is_a", "Mammal"), ("Cat", "is_a", "Mammal"),
                       ("Mammal", "is_a", "Animal")])
    assert diagnose_ontology(model)["state"] == "acyclic_hierarchy"


def test_an_equivalence_is_a_bounded_definition_not_a_cycle():
    model = parse_rdf([("Human", "equivalentClass", "Person"),
                       ("Human", "is_a", "Mammal")])
    r = diagnose_ontology(model)
    assert r["state"] == "bounded_definitions"
    assert r["hodge"]["inconsistency_dimension"] == 0


def test_a_repeated_subsumption_is_not_a_cycle():
    """Two files each asserting `A is_a B` is one relation, not a 2-cycle."""
    model = parse_rdf([("A", "is_a", "B"), ("A", "is_a", "B")])
    assert subsumption_cycles(model) == 0


def test_a_self_subsumption_is_not_counted_as_a_cycle_here():
    """`A is_a A` carries no boundary, so it is not an independent cycle. It is a
    degenerate axiom rather than a subsumption loop between two classes."""
    assert subsumption_cycles(parse_rdf([("A", "is_a", "A")])) == 0


@pytest.mark.parametrize("fmt,text", list(SAMPLES.items()))
def test_every_format_diagnoses_without_raising(fmt, text):
    r = diagnose_ontology(parse_rdf(OF.parse(text, fmt).named_triples()))
    assert r["state"], f"{fmt} produced no state"
    assert r["summary"]


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


def _upload(client, *files, **data):
    return client.post("/api/v1/ontology/upload",
                       files=[("files", (n, t.encode(), "text/plain"))
                              for n, t in files],
                       data=data)


def test_the_formats_route_lists_every_parser(client):
    """The screen builds its selector from this, so a parser registered in PARSERS
    reaches the UI without a frontend edit."""
    r = client.get("/api/v1/ontology/formats")
    assert r.status_code == 200
    body = r.json()
    assert set(body["formats"]) == set(OF.PARSERS)
    assert ".obo" in body["extensions"] and ".gaf" in body["extensions"]


@pytest.mark.parametrize("fmt,text", list(SAMPLES.items()))
def test_every_format_uploads(client, fmt, text):
    r = _upload(client, (f"sample.{fmt}", text), format=fmt)
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body["n_triples"] > 0
    assert body["state"]


def test_an_upload_detects_the_format_from_the_filename(client):
    r = _upload(client, ("go.obo", OBO))
    assert r.status_code == 200
    assert r.json()["source"]["format"] == "obo"


def test_an_upload_reports_what_the_file_said(client):
    body = _upload(client, ("go.obo", OBO)).json()
    assert body["source"]["n_terms"] == 4
    assert body["source"]["n_obsolete"] == 1
    assert body["predicates"]["is_a"] == 2


def test_two_files_upload_as_one_complex(client):
    """The annotation set and its ontology, together."""
    r = _upload(client, ("go.obo", OBO), ("goa.gaf", GAF))
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body["files"] == ["go.obo", "goa.gaf"]
    assert ["TP53", "involved_in", "apoptotic process"] in body["triples"], \
        "the annotation did not resolve against the ontology through the route"


def test_a_file_that_cannot_be_parsed_is_named_not_fatal(client):
    r = _upload(client, ("go.obo", OBO), ("junk.ttl", "\x00\x01"))
    assert r.status_code == 200, r.text[:200]
    assert r.json()["failed_files"][0]["file"] == "junk.ttl"


def test_an_upload_of_only_unparsable_files_is_refused(client):
    r = _upload(client, ("junk.owl", "<not xml"))
    assert r.status_code == 400
    assert "junk.owl" in r.json()["detail"]


def test_an_upload_stores_the_complex_when_asked(client):
    body = _upload(client, ("go.obo", OBO), store_id="go-demo").json()
    assert body.get("stored_as") == "go-demo", body.get("store_error")


def test_ids_are_reported_when_names_are_declined(client):
    body = _upload(client, ("go.obo", OBO), use_names="false").json()
    assert any(t[0].startswith("GO:") for t in body["triples"])


@pytest.mark.parametrize("fmt,text", list(SAMPLES.items()))
def test_pasted_text_analyses_with_no_format_given(client, fmt, text):
    r = client.post("/api/v1/ontology/analyze", json={"text": text})
    assert r.status_code == 200, r.text[:200]
    assert r.json()["source"]["format"] == fmt


def test_the_triple_box_still_works(client):
    """The format the screen accepted before this: one triple per line."""
    r = client.post("/api/v1/ontology/analyze",
                    json={"triples": [["Dog", "subClassOf", "Mammal"],
                                      ["Cat", "subClassOf", "Mammal"]]})
    assert r.status_code == 200
    assert r.json()["state"] == "acyclic_hierarchy"


def test_an_empty_body_says_what_to_provide(client):
    r = client.post("/api/v1/ontology/analyze", json={})
    assert r.status_code == 400
    assert "triples" in r.json()["detail"] and "text" in r.json()["detail"]


def test_text_that_parses_to_nothing_says_so(client):
    r = client.post("/api/v1/ontology/analyze",
                    json={"text": "format-version: 1.2\n", "format": "obo"})
    assert r.status_code == 400
    assert "no relations" in r.json()["detail"]


def test_an_unknown_format_is_refused_by_the_route(client):
    r = client.post("/api/v1/ontology/analyze",
                    json={"text": OBO, "format": "sparql"})
    assert r.status_code == 400


def test_a_large_ontology_is_truncated_in_the_response_not_in_the_complex(client):
    from agent.server.routes.ontology import PREVIEW_LIMIT
    n = PREVIEW_LIMIT + 50
    text = "\n".join(f"T{i} is_a Root" for i in range(n))
    body = client.post("/api/v1/ontology/analyze",
                       json={"text": text, "format": "triples"}).json()
    assert body["n_triples"] == n
    assert len(body["triples"]) == PREVIEW_LIMIT
    assert body["truncated"] is True
    assert body["n_relations"] == n, "the complex was truncated, not just the preview"
