"""Segmentation: the claim under test is that no ONE signal decides a boundary.

These pin the failures that motivated the design, not the successes. A period after an
abbreviation, an initial, a decimal or a slang particle is the same character as a period
ending a sentence, so anything that reads the character alone gets them wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.segment import (boundary_signals, document_layers, encoding_energy,
                              encoding_width, segment_chapters, segment_paragraphs,
                              segment_sentences, strip_markers)
from rexgraph.corpus_profile import ENGLISH_GUTENBERG


def _texts(text, **kw):
    """The English veto comes from the profile now, so a test about English abbreviations
    has to supply it: the segmenter no longer carries a language."""
    kw.setdefault("abbreviations", ENGLISH_GUTENBERG.veto)
    return [text[a:a + n] for a, n in segment_sentences(text, **kw)[0]]


#### the period is not the boundary ############################################

def test_a_title_period_does_not_end_a_sentence():
    assert _texts("Dr. Smith went home. He slept.") == [
        "Dr. Smith went home.", "He slept."]


def test_a_run_of_initials_does_not_end_a_sentence():
    out = _texts("Mr. J. R. R. Tolkien wrote it. It sold.")
    assert out == ["Mr. J. R. R. Tolkien wrote it.", "It sold."]


def test_the_lexicon_can_only_veto_never_create():
    """A supplied abbreviation pattern removes candidates; it cannot add one. Passing a
    pattern that matches nothing therefore never yields FEWER spans than the default."""
    import re as _re
    never = _re.compile(r"(?!x)x")
    with_veto = len(_texts("Dr. Smith went home. He slept."))
    without = len(segment_sentences("Dr. Smith went home. He slept.",
                                    abbreviations=never)[0])
    assert without >= with_veto


def test_a_decimal_point_is_not_a_terminator_at_all():
    """No space follows, so the candidate never opens: this is not a veto, it is not a
    candidate. Keeping the two apart matters, because a veto is corpus-specific and this
    is structural."""
    sig, _sup = boundary_signals("pi is 3.14159 and that is all")
    assert not sig["terminator"].any()


def test_the_terminator_gates_even_when_nothing_else_agrees():
    """Slang and dialect. The vote used to hold this whole because only one channel
    fired; the gate cuts, because a terminator that is not part of a token IS where the
    relation ends. Two utterances is the honest reading, and case is a consequence of a
    boundary rather than evidence for one."""
    assert len(_texts("aint no way she done that fr fr. we out")) == 2


def test_only_the_terminator_decides_and_the_rest_are_readings():
    """One channel gates. The others are reported at the same positions and weighed into
    nothing: a count of agreeing signals was a threshold sitting inside an existence
    decision, and the signals were not independent evidence anyway."""
    sig, _sup = boundary_signals("It ended. Then it began.")
    assert set(sig) == {"terminator", "spacing", "case", "quote", "energy"}
    assert sig["terminator"].any()
    spans, method = segment_sentences("It ended. Then it began.")
    assert method == "terminator_gate" and len(spans) == 2


def test_the_channels_all_fire_at_the_same_index():
    """Channels indexed at different positions can never agree, which silently accepted
    nothing. Every channel is indexed by the position the cut would happen at."""
    sig, _sup = boundary_signals("It ended. Then it began.")
    i = int(np.flatnonzero(sig["terminator"])[0])
    assert sig["case"][i] and sig["spacing"][i] is not None
    assert sig["case"][i], "case must be readable at the terminator's own index"


def test_end_of_text_closes_the_final_sentence():
    assert _texts("One thing happened. Another did too.")[-1] == "Another did too."


#### the encoding decides, exactly ##############################################

def test_a_script_change_is_exactly_a_change_of_byte_width():
    """No level and no statistic. In utf-8 Latin costs 1 byte and Greek 2, so notation
    set into prose is a run of width 2 inside a run of width 1 and its boundary is where
    the width changes."""
    t = "The coefficient αβγ appears here but the rest is plain english prose."
    w = encoding_width(t)
    assert all(w[i] == 2 for i, c in enumerate(t) if c in "αβγ")
    assert all(w[i] == 1 for i, c in enumerate(t) if c.isalpha() and c not in "αβγ")


def test_width_is_a_property_of_the_encoding_not_the_character():
    assert encoding_width("é", encoding="utf-8")[0] == 2
    assert encoding_width("é", encoding="latin-1")[0] == 1


def test_an_all_greek_text_needs_no_baseline_at_all():
    """The old median-normalised form had to ask what was 'ordinary for this text'. Byte
    width does not: every character here is exactly 2, and a run of uniform width has no
    internal boundary to find."""
    w = encoding_width("αβγδε ζηθικ λμνξο")
    assert set(w.tolist()) == {2, 1}, "Greek is 2, the spaces are 1"
    sig, _sup = boundary_signals("αβγδε ζηθικ λμνξο")
    assert not sig["terminator"].any()


def test_energy_is_still_a_property_of_the_encoding():
    a = encoding_energy("é", encoding="utf-8")[0]
    b = encoding_energy("é", encoding="latin-1")[0]
    assert a != b, "the same character costs different bytes in different encodings"


def test_empty_text_has_no_widths_and_raises_nothing():
    assert encoding_width("").size == 0


#### layers, and refusing to invent one ########################################

def test_paragraphs_split_on_the_blank_line_the_author_wrote():
    spans, how = segment_paragraphs("one one one\n\ntwo two two\n\nthree")
    assert len(spans) == 3 and how == "blank_line"


def test_a_text_with_no_blank_line_is_one_paragraph_and_says_so():
    spans, how = segment_paragraphs("a single unbroken block of text")
    assert len(spans) == 1 and how == "single_block"


def test_chapters_are_absent_rather_than_invented():
    """The load-bearing refusal. A book with no headings supports no chapter layer, and
    guessing one would produce divisions no reader could confirm."""
    spans, titles, how = segment_chapters("just prose\n\nmore prose\n\nand more",
                                          headings=ENGLISH_GUTENBERG.headings)
    assert spans == [] and titles == [] and how == "none"


def test_a_matched_chapter_convention_is_named():
    text = "front\n\nCHAPTER I\n\nbody one\n\nCHAPTER II\n\nbody two\n"
    spans, titles, how = segment_chapters(text, headings=ENGLISH_GUTENBERG.headings)
    assert how == "numbered_chapter" and len(spans) == 2
    assert titles[0].startswith("CHAPTER I")


def test_a_corpus_wrapper_is_stripped_and_the_offset_stays_addressable():
    """The markers come from the PROFILE: a Gutenberg wrapper is a fact about one corpus,
    not a property of text, so a segmenter carrying it cannot serve another."""
    raw = ("header junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
           "real body here\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\ntail")
    body, off, how = strip_markers(raw, ENGLISH_GUTENBERG.markers)
    assert "real body here" in body
    assert how == "start_marker+end_marker"
    assert raw[off:off + len(body)] == body, "the offset must address the ORIGINAL file"


def test_a_profileless_reading_claims_nothing():
    """No markers stripped, no heading convention tried. That is the honest default for
    an unknown corpus rather than English's conventions applied to it."""
    raw = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody here now.\n"
           "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n")
    assert document_layers(raw)["document"]["method"] == "no_markers"
    assert document_layers(raw, profile=ENGLISH_GUTENBERG)["document"]["method"] == (
        "start_marker+end_marker")


def test_paragraphs_split_on_crlf_too():
    """A CRLF file separates paragraphs with "\r\n\r\n", which an LF-only pattern cannot
    match: every such document read as ONE block. Two real books, 3,762 and 6,345 CRLF
    pairs, were doing exactly that."""
    spans, how = segment_paragraphs("one one one\r\n\r\ntwo two two\r\n\r\nthree")
    assert len(spans) == 3 and how == "blank_line"


def test_document_layers_omits_what_it_cannot_resolve():
    raw = "one thing here. two things here.\n\nthree things here.\n"
    layers = document_layers(raw)
    assert "chapter" not in layers, "no headings, so no chapter layer"
    assert {"document", "paragraph", "sentence"} <= set(layers)
    assert layers["document"]["method"] == "no_markers"


def test_every_span_addresses_the_original_text():
    raw = ("*** START OF THE PROJECT GUTENBERG EBOOK T ***\n"
           "One sentence here. Another one follows.\n\nA second paragraph.\n"
           "*** END OF THE PROJECT GUTENBERG EBOOK T ***\n")
    layers = document_layers(raw)
    for name, layer in layers.items():
        for a, n in layer["spans"]:
            assert 0 <= a and a + n <= len(raw), f"{name} span leaves the file"
            assert raw[a:a + n].strip(), f"{name} span is blank"


@pytest.mark.parametrize("bad", ["", "   \n\n  "])
def test_empty_input_yields_no_sentences_rather_than_raising(bad):
    spans, how = segment_sentences(bad)
    assert spans == [] and how == "empty"


#### the leverage share field ##################################################

def test_the_sketch_falls_back_to_exact_when_it_would_cost_more():
    """A projection wider than the thing it projects is not a saving, and reporting it
    as an approximation would misstate the reading."""
    from rexgraph.core._sparse import to_scipy_csr
    from rexgraph.fiedler import leverage_diagonal, leverage_sketch
    from rexgraph.graph import RexGraph
    r = RexGraph(sources=[0, 1, 2, 2, 3, 4], targets=[1, 2, 0, 3, 4, 2])
    B = to_scipy_csr(r._B1_dual).tocsc()
    a, _ra = leverage_sketch(B, epsilon=0.1)
    e, _re = leverage_diagonal(B)
    assert np.allclose(a, e), "tiny complexes take the exact path, not a sample"


def test_the_sketch_preserves_the_total_it_is_a_share_of():
    """Foster's identity holds in expectation, so the SUM survives a projection that
    individual entries only approximate. That is why section masses read better than
    single relations do."""
    from rexgraph.core._sparse import to_scipy_csr
    from rexgraph.fiedler import leverage_sketch
    from rexgraph.graph import RexGraph
    rng = np.random.default_rng(0)
    n = 400
    src = rng.integers(0, 120, n)
    tgt = (src + rng.integers(1, 30, n)) % 120
    keep = src != tgt
    r = RexGraph(sources=src[keep].tolist(), targets=tgt[keep].tolist())
    B = to_scipy_csr(r._B1_dual).tocsc()
    approx, rank = leverage_sketch(B, epsilon=0.5, seed=3)
    assert abs(approx.sum() - rank) / max(rank, 1) < 0.05
