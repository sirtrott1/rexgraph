"""Passage answerer."""
import pytest

from agent.answerers import _question as Q
from agent.answerers.passage import PassageAnswerer, render

CHAN = ["topology", "geometry", "frustration", "coparticipation"]

#: stated here rather than read from disk, so the gated cases read the same anywhere.
#: With nothing on disk the gate is empty, which the last test pins.
GATE = frozenset({"a", "an", "and", "at", "does", "in", "is", "it", "of", "the",
                  "was", "what", "where", "who", "why"})
INTERFACE = frozenset({"mean", "word", "related", "definition", "synonym"})


@pytest.fixture(autouse=True)
def _gate(monkeypatch):
    monkeypatch.setattr(Q, "function_words", lambda: GATE)
    monkeypatch.setattr(Q, "interface_words", lambda: INTERFACE)


def sections():
    return [
        {"section_id": "ch-12", "layer": "chapter", "span": (4120, 4680),
         "text": "The harpoon was darted at the whale, and the line ran out.",
         "reading": "both", "agree": True,
         "channels": [0.31, 0.22, 0.05, 0.42], "channel_names": CHAN},
        {"section_id": "ch-3", "layer": "chapter", "span": (900, 1200),
         "text": "A quiet morning in the village, nothing of note.",
         "reading": "magnitude", "agree": False,
         "channels": None, "channel_names": None},
    ]


def test_only_passages_that_contain_a_query_term_are_reported():
    r = PassageAnswerer().answer("where is the harpoon", sections())
    assert r["answered"]
    assert [p["section"] for p in r["passages"]] == ["ch-12"]   # ch-3 holds no term


def test_what_a_passage_contains_is_the_exact_intersection():
    r = PassageAnswerer().answer("what does the word harpoon mean", sections())
    assert r["terms"] == ["harpoon"]
    p = r["passages"][0]
    assert p["contains"] == ["harpoon"] and p["missing"] == []
    assert r["uncovered"] == []


def test_a_term_the_passage_lacks_is_reported_as_missing():
    r = PassageAnswerer().answer("harpoon aqueduct", sections())
    p = r["passages"][0]
    assert p["contains"] == ["harpoon"] and p["missing"] == ["aqueduct"]
    assert r["uncovered"] == ["aqueduct"]


def test_the_frame_carries_no_retrieval_content():
    r = PassageAnswerer().answer("what is it", sections())
    assert not r["answered"]
    assert r["reason"] == "the question names no term to locate"


def test_no_matching_passage_declines_rather_than_printing_the_nearest():
    r = PassageAnswerer().answer("where is the aqueduct", sections())
    assert not r["answered"]
    assert "no retrieved passage contains" in r["reason"]


def test_empty_retrieval_declines():
    r = PassageAnswerer().answer("where is the harpoon", [])
    assert not r["answered"] and r["reason"] == "retrieval returned no passage"


def test_provenance_survives_to_the_rendered_text():
    r = PassageAnswerer().answer("where is the harpoon", sections(), document="Moby-Dick")
    text = render(r)
    assert "Moby-Dick" in text and "ch-12" in text
    # a span is (offset, length), the address section_text seeks to, not start/end
    assert "bytes 4120+4680" in text
    assert "found by: both" in text


def test_the_channel_profile_is_reported_not_a_scalar():
    r = PassageAnswerer().answer("where is the harpoon", sections())
    p = r["passages"][0]
    assert p["channels"] == [0.31, 0.22, 0.05, 0.42]
    text = render(r)
    for name in CHAN:
        assert name in text
    # the sum is never what is shown: the channels move in opposite directions
    assert "1.0" not in text.split("found by")[0].split("coparticipation")[-1]


def test_the_answer_states_what_co_occurrence_cannot_do():
    text = render(PassageAnswerer().answer("where is the harpoon", sections()))
    assert "co-occurrence" in text
    assert "not assert what it means" in text
    assert "language model" not in text.lower()                 # no apology


def test_the_worker_interface_matches_the_hive_primitive():
    handler, capability, wtype = PassageAnswerer().as_worker()
    assert capability == "passage" and wtype == "answerer:passage"
    got = handler({"query": "where is the harpoon", "sections": sections(),
                   "document": "Moby-Dick"})
    assert got["answered"] and got["passages"][0]["document"] == "Moby-Dick"


def test_a_span_is_rendered_as_offset_and_length():
    # 10657+394 is a 394-byte section at offset 10657. Printed as a range it reads
    # "10657-394", an interval running backwards.
    r = PassageAnswerer().answer("where is the harpoon", [
        {"section_id": "s124", "layer": "sentence", "span": (10657, 394),
         "text": "the harpoon", "reading": "magnitude", "agree": False,
         "channels": None, "channel_names": None}])
    assert "bytes 10657+394" in render(r)


def test_with_no_lexicon_the_gate_excludes_nothing(monkeypatch):
    monkeypatch.setattr(Q, "function_words", frozenset)
    monkeypatch.setattr(Q, "interface_words", frozenset)
    r = PassageAnswerer().answer("where is the harpoon", sections())
    assert r["terms"] == ["where", "is", "the", "harpoon"]
    assert r["answered"], "an ungated query still locates the passage that holds it"
