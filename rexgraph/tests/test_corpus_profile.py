"""The profile: what a corpus contributes, separated from what the math does.

The tests that matter here are the ones that used to return NOTHING. `_TOKEN =
r"[a-z']+"` gave six tokens for English and zero for Greek, Russian, Chinese and Arabic,
so four corpora out of seven built an empty complex and nothing raised. A tokenizer that
fails silently is worse than one that fails, because every reading above it is then a
reading of nothing.
"""
from __future__ import annotations

import pytest

from rexgraph.corpus_profile import (ENGLISH_GUTENBERG, PYTHON_SOURCE, TEXT, CorpusProfile,
                              is_scriptio_continua, tokenize)


def _toks(text, profile=TEXT):
    return [t for t, _a, _b in tokenize(text, profile)]


#### the failure that forced the profile ########################################

@pytest.mark.parametrize("name,text,expect", [
    ("english", "the cat sat on the mat", 6),
    ("greek", "ο σκύλος έτρεξε γρήγορα", 4),
    ("russian", "кошка сидела на коврике", 4),
    ("arabic", "القطة جلست على السجادة", 4),
    ("hindi", "बिल्ली चटाई पर बैठी", 4),
    ("hebrew", "החתול ישב על המחצלת", 4),
])
def test_a_space_separated_script_tokenises_to_its_words(name, text, expect):
    """Every one of these except English returned ZERO tokens before."""
    assert len(_toks(text)) == expect, f"{name}: {_toks(text)}"


def test_indic_vowel_marks_do_not_break_the_word():
    """Indic scripts write vowels as combining MARKS, which a `\\w` class excludes, so a
    `\\w`-based rule split every matra: eight fragments for four Hindi words. The Unicode
    category is what has to be read."""
    assert _toks("बिल्ली चटाई पर बैठी") == ["बिल्ली", "चटाई", "पर", "बैठी"]


def test_a_mixed_script_run_survives():
    """The exact case the energy channel was demonstrated on. `αβγ` used to vanish.
    Case is folded because that is a PROSE fact carried by the profile: a
    sentence-initial capital is noise here and load-bearing in source."""
    assert _toks("The coefficient αβγ equals 3") == [
        "the", "coefficient", "αβγ", "equals", "3"]


#### scripts written without separators ##########################################

def test_a_continua_script_is_read_by_grapheme_not_by_run():
    """Chinese has no spaces, so a run rule would make ONE vertex of a whole sentence."""
    assert _toks("猫坐在垫子上") == ["猫", "坐", "在", "垫", "子", "上"]
    assert is_scriptio_continua("猫") and not is_scriptio_continua("a")


def test_a_grapheme_cluster_holds_its_marks_together():
    """Per-codepoint tore Thai's base+vowel+tone into separate vertices."""
    toks = _toks("แมวนั่งบนเสื่อ")
    assert "นั่" in toks, toks
    assert all(len(t) >= 1 for t in toks)


def test_no_word_boundary_is_invented_inside_a_continua_script():
    """Finding words in Thai needs a dictionary. Reading graphemes is the finest TRUE
    unit; asserting a word boundary that is not there would be the error."""
    assert len(_toks("แมวนั่งบนเสื่อ")) > 4, "graphemes, not guessed words"


#### source code #################################################################

def test_an_identifier_stays_whole():
    """Splitting `build_document` into two vertices dissolves the name for the same
    reason clique expansion dissolves a relation."""
    toks = _toks("def build_document(raw, *, min_terms=1): return raw", PYTHON_SOURCE)
    assert "build_document" in toks and "min_terms" in toks
    assert "build" not in toks and "terms" not in toks


def test_source_punctuation_is_a_token_because_it_gates():
    toks = _toks("f(x, y)", PYTHON_SOURCE)
    assert "(" in toks and "," in toks
    assert set(PYTHON_SOURCE.gate) & set(toks), "the gate is in the stream"


def test_the_two_rules_disagree_and_that_is_the_point():
    """The underscore is connector punctuation (Pc), not a letter. In source it joins an
    identifier; in prose it does not, and neither reading is a correction of the other,
    which is exactly why the rule belongs on the profile rather than in the library."""
    src = "min_terms = 1"
    assert "min_terms" in _toks(src, PYTHON_SOURCE), "source keeps the name whole"
    assert _toks(src, TEXT)[:3] == ["min", "_", "terms"], "prose does not join on it"


#### positions, because everything above addresses by span #######################

def test_every_token_carries_its_position():
    """The span must recover the token. Under a case-folding profile it recovers it up
    to that fold, which is why the comparison is against the folded source text and not
    against the raw slice."""
    text = "The coefficient αβγ equals 3"
    for tok, a, b in tokenize(text, TEXT):
        assert text[a:b].lower() == tok
    for tok, a, b in tokenize("Foo = foo", PYTHON_SOURCE):
        assert "Foo = foo"[a:b] == tok, "source folds nothing, so it is exact"


def test_positions_are_monotone_and_non_overlapping():
    prev = 0
    for _t, a, b in tokenize("don't split cat's tail, ok?", TEXT):
        assert a >= prev and b > a
        prev = b


#### the profile carries the corpus facts, not the library #######################

def test_english_and_gutenberg_facts_live_on_the_profile():
    assert ENGLISH_GUTENBERG.markers and ENGLISH_GUTENBERG.headings
    assert ENGLISH_GUTENBERG.veto is not None
    assert not TEXT.markers and not TEXT.headings, "the bare profile claims nothing"


def test_a_code_profile_declares_its_layers_rather_than_inferring_them():
    """A programming language's layers are stated by its grammar, so none is ever
    'absent because unresolvable' the way a chapter can be."""
    assert PYTHON_SOURCE.layers == ("statement", "block", "function", "module")
    assert PYTHON_SOURCE.token_rule == "identifier"


def test_a_profile_is_immutable():
    with pytest.raises(Exception):
        TEXT.name = "other"


def test_an_empty_text_tokenises_to_nothing_without_raising():
    assert tokenize("", TEXT) == []
    assert tokenize("   \n ", TEXT) == []


def test_a_custom_profile_needs_no_library_change():
    """The separation, stated as a test: a new corpus is a new record, not a new branch."""
    klingon = CorpusProfile(name="klingon", gate=frozenset({"'"}), casefold=False)
    assert _toks("nuqneH", klingon) == ["nuqneH"], "case survives when the profile says"
    assert klingon.layers == TEXT.layers


def test_this_module_does_not_shadow_a_stdlib_name():
    """It was called `profile.py` and that broke 17 unrelated torch tests.

    `cProfile` does `import profile` to reuse the stdlib profiler's `run`, and
    `torch._dynamo` imports `cProfile`. A top-level `profile.py` anywhere on `sys.path`
    therefore shadows it, and pytest puts the package directory on the path only when
    the whole suite runs, so the failure appeared in the suite and vanished in
    isolation, which is the worst shape for finding it.
    """
    import importlib
    import os
    import rexgraph.corpus_profile as _cp

    # the SOURCE directory, taken from a module's own file. Two wrong ways: the
    # substring "rexgraph" also matches the conda environment's path, and
    # `rexgraph.__path__` is synthesised by the editable-install loader and is not a
    # directory at all: both make the check pass while testing nothing.
    pkg = os.path.dirname(os.path.realpath(_cp.__file__))
    for name in ("profile", "cProfile", "code", "token", "types", "copy", "string",
                 "select", "signal", "keyword", "operator", "io", "abc", "enum"):
        mod = importlib.import_module(name)
        path = os.path.realpath(getattr(mod, "__file__", "") or "")
        assert not path.startswith(pkg + os.sep), (
            f"rexgraph shadows the stdlib module {name!r} ({path})")


def test_no_module_in_the_package_collides_with_a_stdlib_name():
    """The general form, so the next one is caught at the point it is added."""
    import os
    import sys

    import rexgraph.corpus_profile as _cp
    pkg = os.path.dirname(os.path.realpath(_cp.__file__))
    ours = {f[:-3] for f in os.listdir(pkg) if f.endswith(".py")}
    clash = ours & set(sys.stdlib_module_names)
    assert not clash, f"these shadow stdlib modules once the package is on sys.path: {sorted(clash)}"


def test_case_folding_is_a_profile_fact_not_a_default():
    """Prose folds because a sentence-initial capital is noise. Source does not, because
    `Foo` and `foo` are different identifiers and folding merges two vertices the
    language keeps apart."""
    assert _toks("The Cat sat", TEXT) == ["the", "cat", "sat"]
    assert _toks("Foo = foo", PYTHON_SOURCE) == ["Foo", "=", "foo"]
    assert TEXT.casefold and not PYTHON_SOURCE.casefold


#### one mark, three jobs #######################################################

_BURGERS = "Mr. Jim\'s burgers are great...but his wife\'s burgers are better."


def test_the_same_character_does_three_different_jobs_in_one_sentence():
    """`Mr.` is a suffix, `...` is an internal delimiter, and the last `.` ends the
    sentence. All three are periods, and reading the character alone gets every one of
    them wrong."""
    from rexgraph.segment import segment_sentences
    spans, _m = segment_sentences(_BURGERS, abbreviations=ENGLISH_GUTENBERG.veto)
    assert len(spans) == 1, [(_BURGERS[a:a + n]) for a, n in spans]
    assert _BURGERS[spans[0][0]:spans[0][0] + spans[0][1]].startswith("Mr.")


def test_without_the_veto_the_abbreviation_ends_a_sentence():
    """Which is what makes the veto a LEXICAL fact rather than a nicety."""
    from rexgraph.segment import segment_sentences
    assert len(segment_sentences(_BURGERS, abbreviations=None)[0]) == 2


def test_a_vetoed_period_belongs_to_its_token_at_every_layer():
    """It used to be a suffix at the sentence layer and a delimiter at the span layer,
    so the same period was two different things depending on who asked. `Mr.` is one
    token or it is one token nowhere."""
    toks = [t for t, _a, _b in tokenize(_BURGERS, ENGLISH_GUTENBERG)]
    assert "mr." in toks and "mr" not in toks


def test_a_possessive_apostrophe_stays_inside_the_word():
    toks = [t for t, _a, _b in tokenize(_BURGERS, ENGLISH_GUTENBERG)]
    assert "jim\'s" in toks and "wife\'s" in toks


def test_an_ellipsis_is_one_mark_not_three_periods():
    toks = [t for t, _a, _b in tokenize(_BURGERS, ENGLISH_GUTENBERG)]
    assert "..." in toks, toks
    assert toks.count(".") == 1, "only the sentence-final period is a lone period"


def test_a_punctuation_run_is_one_token():
    assert [t for t, _a, _b in tokenize("what?! yes... no.", TEXT)] == [
        "what", "?!", "yes", "...", "no", "."]


def test_a_token_of_joiners_only_terminates():
    """A joiner may not end a word, so `--` trims back to nothing and arrives at the
    punctuation run on a character that rule calls a word: the tokenizer hung. Progress
    is not optional."""
    assert [t for t, _a, _b in tokenize("-- \'\' - x", TEXT)][-1] == "x"


def test_the_ellipsis_divides_the_span_but_not_the_sentence():
    """The period that ends nothing still delimits something: `...` gates a span while
    the sentence runs on through it."""
    from rexgraph.document import build_document
    rex, info = build_document(_BURGERS + "\n", profile=ENGLISH_GUTENBERG)
    assert info["n_sentences"] == 1
    assert info["n_spans"] == 2
    assert info["base_layer"] == "span", "the layer is named for what it DOES"
    assert "mr." in info["vocab"], "the abbreviation is a vertex, not a fragment"
