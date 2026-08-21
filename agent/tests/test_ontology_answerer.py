"""Ontology answerer."""
import pytest

from agent.answerers.ontology import OntologyAnswerer, render

MINI = """format-version: 1.2
ontology: mini

[Term]
id: X:0001
name: organelle

[Term]
id: X:0002
name: membrane-bounded organelle
is_a: X:0001 ! organelle

[Term]
id: X:0003
name: mitochondrion
is_a: X:0002 ! membrane-bounded organelle
relationship: part_of X:0010 ! cytoplasm

[Term]
id: X:0004
name: chloroplast
is_a: X:0002 ! membrane-bounded organelle

[Term]
id: X:0010
name: cytoplasm
is_a: X:0001 ! organelle
"""


@pytest.fixture(scope="module")
def answerer(tmp_path_factory):
    p = tmp_path_factory.mktemp("onto") / "mini.obo"
    p.write_text(MINI)
    return OntologyAnswerer.from_file(str(p))


def test_parents_are_the_transitive_ancestors_with_their_chains(answerer):
    r = answerer.answer("what is a mitochondrion")
    assert r["answered"]
    assert r["asked"] == "parents"
    terms = [x["term"] for x in r["results"]]
    assert any("membrane-bounded organelle" in t for t in terms)
    assert any(t.endswith("(X:0001)") for t in terms)          # transitive, not just direct
    far = next(x for x in r["results"] if x["term"].endswith("(X:0001)"))
    assert far["steps"] == 2 and far["via"]                     # the chain is reported


def test_children_walk_the_other_direction(answerer):
    r = answerer.answer("what kinds of organelle are there")
    assert r["answered"] and r["asked"] == "children"
    terms = {x["term"].split(" (")[0] for x in r["results"]}
    assert {"membrane-bounded organelle", "mitochondrion", "chloroplast",
            "cytoplasm"} <= terms


def test_a_frame_word_never_beats_the_word_that_names_the_relation(answerer):
    # "what" opens both of these and means neither; `kinds` and `part` decide.
    assert answerer.answer("what kinds of organelle are there")["asked"] == "children"
    assert answerer.answer("what is mitochondrion part of")["asked"] == "relations"


def test_a_non_subsumption_relation_is_answered_in_its_own_predicate(answerer):
    r = answerer.answer("what is mitochondrion part of")
    assert r["answered"]
    assert [x["predicate"] for x in r["results"]] == ["part_of"]
    assert r["results"][0]["term"].startswith("cytoplasm")


def test_a_multiword_term_is_matched_whole(answerer):
    r = answerer.answer("what is a membrane-bounded organelle")
    assert r["answered"]
    assert r["subject"].startswith("membrane-bounded organelle")


def test_a_term_the_ontology_lacks_is_declined_not_guessed(answerer):
    r = answerer.answer("what is a ribosome")
    assert not r["answered"] and "holds none of" in r["reason"]


def test_a_non_ontological_question_is_declined_without_building_anything():
    fresh = OntologyAnswerer(None)
    r = fresh.answer("who lives at 221b baker street")
    assert not r["answered"]
    assert r["reason"] == "no ontology relation is asked for"
    assert fresh._rc is None                    # nothing was built to decline


def test_an_asserted_relation_that_does_not_exist_declines_about_the_ontology(answerer):
    r = answerer.answer("what is organelle part of")
    assert not r["answered"]
    assert "asserts no relations" in r["reason"]
    assert r["subject"].startswith("organelle")  # it knows the term, not the relation


def test_render_emits_one_line_per_axiom(answerer):
    r = answerer.answer("what is a mitochondrion")
    lines = render(r).splitlines()
    assert len(lines) == len(r["results"])
    assert all("is a" in ln for ln in lines)


def test_the_worker_interface_matches_the_hive_primitive(answerer):
    handler, capability, wtype = answerer.as_worker()
    assert capability == "classify" and wtype == "answerer:ontology"
    assert handler({"query": "what is a mitochondrion"})["answered"]
