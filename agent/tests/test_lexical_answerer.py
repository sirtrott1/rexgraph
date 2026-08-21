"""WordNet answerer."""
from __future__ import annotations

import os

import pytest

from agent.answerers.lexical import DEFAULT_WORDNET, LexicalAnswerer, render

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_WORDNET),
                                reason="the WordNet lexicon is not present")


@pytest.fixture(scope="module")
def ans():
    a = LexicalAnswerer()
    a._lex()
    return a


def test_it_defines_the_term_the_question_names(ans):
    r = ans.answer("what does the word harpoon mean")
    assert r["answered"] and r["subject"] == "harpoon"
    text = render(r).lower()
    assert "spear" in text, text[:200]


def test_a_mention_frame_beats_a_commoner_entry(ans):
    r = ans.answer("what does the word harpoon mean")
    assert r["subject"] == "harpoon", "the mention frame names its own subject"


def test_a_mentioned_term_the_lexicon_lacks_is_declined_not_substituted(ans):
    r = ans.answer("what does the word zzzqx mean")
    assert not r["answered"]
    assert "holds none" in r["reason"]


def test_a_recorded_verb_form_is_not_the_subject(ans):
    r = ans.answer("what kinds of whale are there")
    assert r["answered"] and r["subject"] == "whale"
    assert r["relation"] == "hyponym"
    assert any(s["related"] for s in r["senses"])


def test_the_relation_asked_for_is_the_one_traversed(ans):
    r = ans.answer("what are the parts of a harpoon")
    assert r["answered"] and r["relation"] == "holo_member"
    assert r["subject"] == "harpoon"


def test_a_non_lexical_question_is_declined_without_loading_anything():
    fresh = LexicalAnswerer()
    r = fresh.answer("who lives at 221b baker street")
    assert not r["answered"]
    assert r["reason"] == "no lexical relation is asked for"
    assert fresh._wn is None, "declining must not have loaded the lexicon"


def test_a_relation_the_lexicon_does_not_record_is_declined_as_such(ans):
    r = ans.answer("what causes rain")
    assert not r["answered"]
    assert "records no" in r["reason"] and r.get("subject") == "rain"


def test_every_answer_carries_its_provenance(ans):
    r = ans.answer("what is a frigate")
    assert r["answered"] and r["source"] == "wordnet"
    assert all(s["synset"] for s in r["senses"]), "each sense names its synset id"


def test_it_registers_as_a_hive_worker(ans):
    handler, capability, worker_type = ans.as_worker()
    assert capability == "define" and worker_type == "answerer:lexical"
    out = handler({"query": "what does the word harpoon mean"})
    assert out["answered"] and out["subject"] == "harpoon"


def test_a_locational_question_is_not_a_definitional_one(ans):
    # "is" fires the definition interface, and `hunted` is a real WordNet adjective, so
    # this used to answer "hunted (a): reflecting the fear or terror of one who is
    # hunted" to a question about where whaling happens. The interrogative governs.
    r = ans.answer("where is the whale hunted")
    assert not r["answered"]
    assert r["reason"] == "no lexical relation is asked for"


def test_a_predicative_question_still_reaches_the_definition(ans):
    # the same copula, the other interrogative: this one IS definitional.
    r = ans.answer("what is a harpoon")
    assert r["answered"] and r["relation"] == "definition"


def test_a_named_relation_beats_the_copula(ans):
    # `kinds` names a relation; `is` only frames the question.
    r = ans.answer("what kinds of whale are there")
    assert r["answered"] and r["relation"] == "hyponym"
