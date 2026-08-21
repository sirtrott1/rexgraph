"""Grammar adapter."""
from __future__ import annotations

import pytest

from agent.adapters.grammar import FrameGrammar, wordnet_grammar


@pytest.fixture
def grammar():
    return FrameGrammar(
        frames={"via": ("Somebody ----s", 2),
                "vtai": ("Somebody ----s something", 3),
                "ditransitive": ("Somebody ----s somebody something", 4)},
        word_frames={"sleep": {"via"}, "chase": {"vtai"}, "give": {"ditransitive"},
                     "slept": {"via"}, "book": {"vtai"}},
    )


#### the head rule ##############################################################

def test_the_token_subcategorising_for_the_arity_heads_it(grammar):
    assert grammar.head_of(["dog", "sleep"]) == (1, "via")
    assert grammar.head_of(["dog", "chase", "cat"]) == (1, "vtai")
    assert grammar.head_of(["she", "give", "him", "it"]) == (1, "ditransitive")


def test_the_arity_is_DISTINCT_participants_matching_the_column(grammar):
    assert grammar.head_of(["the", "dog", "chase", "the", "cat"]) is None
    assert grammar.head_of(["dog", "chase", "cat"]) == (1, "vtai")


def test_several_candidates_make_no_claim(grammar):
    assert grammar.head_of(["chase", "book", "x"]) is None


def test_nothing_subcategorising_makes_no_claim(grammar):
    assert grammar.head_of(["the", "quiet", "room"]) is None


def test_a_witness_has_no_frame(grammar):
    assert grammar.head_of(["sleep"]) is None
    assert grammar.head_of([]) is None


def test_arity_is_read_off_the_frame(grammar):
    assert grammar.arity_of("ditransitive") == 4
    assert grammar.arity_of("nonexistent") == 0


#### orientation is reordering, because position carries it #####################

def test_orienting_moves_the_head_to_position_zero(grammar):
    from rexgraph.construct import _orient
    g, fid = _orient(["dog", "chase", "cat"], grammar)
    assert g[0] == "chase" and fid == "vtai"
    assert sorted(g) == sorted(["dog", "chase", "cat"]), "reordered, not rewritten"


def test_no_claim_leaves_the_order_the_text_gave(grammar):
    from rexgraph.construct import _orient
    g, fid = _orient(["the", "quiet", "room"], grammar)
    assert g == ["the", "quiet", "room"] and fid is None


def test_no_grammar_is_the_positional_reading(grammar):
    from rexgraph.construct import _orient
    g, fid = _orient(["dog", "chase", "cat"], None)
    assert g == ["dog", "chase", "cat"] and fid is None


#### the frame reaches the boundary column ######################################

def test_the_frame_is_attached_to_the_relation_it_oriented(grammar):
    from rexgraph.corpus_profile import ENGLISH_GUTENBERG
    from rexgraph.document import build_document
    text = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
            "Dog chase cat.\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n")
    # the markers are a PROFILE fact now; without one they are left in the body and
    # become tokens of the first relation, which changes its arity and its frame
    rex, info = build_document(text, profile=ENGLISH_GUTENBERG, grammar=grammar)
    assert info["n_oriented_by_frame"] >= 1
    idx = next(i for i, f in enumerate(info["frames"]) if f)
    assert rex.get_metadata(1, idx, "frame") == info["frames"][idx]


def test_a_relation_no_frame_governed_carries_no_frame_attribute(grammar):
    from rexgraph.document import build_document
    rex, info = build_document("The quiet room. Nothing at all here.\n")
    assert all(f is None for f in info["frames"])
    assert rex.get_metadata(1, 0, "frame") is None


#### what the lexicon actually supplies #########################################

def test_recorded_inflections_are_reachable():
    g = FrameGrammar(frames={"via": ("Somebody ----s", 2)},
                     word_frames={"sleep": {"via"}, "slept": {"via"}})
    assert g.head_of(["dog", "slept"]) == (1, "via")


def test_wordnet_grammar_builds_from_a_loaded_lexicon():
    wn = {"frames": {"via": "Somebody ----s"},
          "frames_of": {"s1": ["via"]},
          "sense_of": {"s1": ("e1", "ss1")},
          "lemma_of": {"e1": "Sleep"},
          "forms_of": {"e1": ["slept"]}}
    g = wordnet_grammar(wn)
    assert g.arity_of("via") == 2
    assert "sleep" in g.word_frames and "slept" in g.word_frames, "lemma AND its forms"
    assert g.head_of(["dog", "SLEPT"]) == (1, "via"), "the lookup is case-folded"
