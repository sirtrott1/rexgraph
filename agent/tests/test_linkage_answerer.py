"""Roget and Wiktionary answerer."""
import os

import pytest

from agent.answerers.linkage import (LinkageAnswerer, render,
                                     DEFAULT_ROGET, DEFAULT_WIKTIONARY)

has_roget = pytest.mark.skipif(not os.path.exists(DEFAULT_ROGET),
                               reason="Roget not on disk")
has_wik = pytest.mark.skipif(not os.path.exists(DEFAULT_WIKTIONARY),
                             reason="Wiktionary index not on disk")


@pytest.fixture(scope="module")
def rog():
    return LinkageAnswerer.roget()


@pytest.fixture(scope="module")
def wik():
    return LinkageAnswerer.wiktionary()


#### what each source can be asked #########################################
@has_roget
def test_roget_answers_thematic_grouping_with_the_category_name(rog):
    r = rog.answer("what is related to harpoon")
    assert r["answered"] and r["asked"] == "related"
    g = r["groups"][0]
    assert g["kind"] == "category" and g["direction"] == "undirected"
    assert g["label"]                                    # the category is NAMED
    assert "harpoon" not in [t.lower() for t in g["terms"]]


@has_roget
def test_roget_declines_a_kind_it_does_not_record(rog):
    # a category asserts co-membership and no direction, so it has no antonyms and
    # must say so rather than returning its category as if it were one.
    r = rog.answer("what is the opposite of joy")
    assert not r["answered"] and r["reason"] == "roget records no antonyms"


@has_wik
def test_wiktionary_answers_in_its_own_kind_names(wik):
    for q, kind in (("what are synonyms of grief", "synonyms"),
                    ("what is the opposite of joy", "antonyms"),
                    ("what kinds of whale are there", "hyponyms")):
        r = wik.answer(q)
        assert r["answered"], (q, r["reason"])
        assert r["asked"] == kind
        assert any(g["kind"] == kind for g in r["groups"])


#### the two bugs this had ##################################################
@has_wik
def test_the_direct_reading_comes_before_the_converse(wik):
    r = wik.answer("what are synonyms of grief")
    assert r["groups"][0]["direction"] == "of"
    assert r["groups"][0]["label"].lower() == "grief"
    text = render(r)
    assert text.splitlines()[0].startswith("synonyms of grief:")
    conv = [g for g in r["groups"] if g["direction"] == "converse"]
    if conv:
        assert "is listed among the" in text


@has_wik
def test_a_function_word_is_never_the_subject(wik):
    r = wik.answer("what are synonyms of grief")
    assert r["answered"] and r["subject"] == "grief"


@has_wik
def test_a_relation_naming_word_is_never_the_subject(wik):
    # `related` names a relation, so it is question vocabulary even here.
    r = wik.answer("what is related to harpoon")
    assert r["answered"] and r["subject"] == "harpoon"


#### declining ##############################################################
@has_wik
def test_a_term_the_source_lacks_is_declined(wik):
    r = wik.answer("what is related to zzzznotaword")
    assert not r["answered"] and "holds none of" in r["reason"]


@has_wik
def test_a_recorded_term_with_no_such_kind_says_what_it_does_record(wik):
    r = wik.answer("what are the parts of a ship")
    if not r["answered"]:
        assert "records no meronyms" in r["reason"]
        assert "it records" in r["reason"]           # names the kinds it has


def test_a_non_linkage_question_is_declined_without_loading_anything():
    fresh = LinkageAnswerer.roget()
    r = fresh.answer("who lives at 221b baker street")
    assert not r["answered"] and r["reason"] == "no linkage is asked for"
    assert fresh._by_word is None                    # nothing was read to decline


@has_roget
def test_the_worker_interface_matches_the_hive_primitive(rog):
    handler, capability, wtype = rog.as_worker()
    assert capability == "relate" and wtype == "answerer:roget"
    assert handler({"query": "what is related to harpoon"})["answered"]
