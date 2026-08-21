"""A document as one field with layers: the canonical construction, once.

The load-bearing test here is the alignment one. `from_text` drops a sentence that fails
its filters, so group j is not sentence j, and aligning spans by position silently
misaddresses everything after the first drop: a section would then point at another
sentence's bytes and every proof and citation built on it would be confidently wrong.
"""
from __future__ import annotations

import re

import pytest

from rexgraph.document import build_document, document_sections, section_text
from rexgraph.merkle import build_merkle
from rexgraph.corpus_profile import ENGLISH_GUTENBERG, CorpusProfile
from rexgraph.sectioning import sectionings_of

_BOOK = (
    "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n\n"
    "CHAPTER I\n\n"
    "The morning arrived with considerable noise and confusion. "
    "Mercy. "                                   # 1 term: dropped by min_terms
    "Wilson considered the peculiar arrangement of the room carefully. "
    "Alas. "                                    # dropped ("No." is vetoed as an
                                                #  abbreviation, so it never splits)
    "Judge Driscoll examined the disputed fingerprints under lamplight.\n\n"
    "The twins departed before anybody noticed their curious absence. "
    "Roxy watched everything unfold from the kitchen doorway quietly.\n\n"
    "CHAPTER II\n\n"
    "Everybody assembled inside the crowded courtroom awaiting judgement. "
    "The verdict astonished every single person present that afternoon.\n\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
)


@pytest.fixture
def doc():
    # the profile supplies the markers, the heading conventions and the veto; without
    # one the reading claims nothing, which is the honest default and not what this
    # fixture is exercising
    return build_document(_BOOK, profile=ENGLISH_GUTENBERG)


def _words(s):
    return set(re.findall(r"[a-z']{3,}", s.lower()))


#### the alignment the whole citation layer rests on ###########################

def test_every_section_span_contains_that_section_s_own_words(doc):
    """The regression. A section's cells name words; those words must appear in the
    bytes the section's span points at. Off-by-one from a dropped sentence breaks this
    on every section after the first drop and on none before it."""
    rex, info = doc
    labels = list(sectionings_of(rex)["sentence"].labels)
    vocab = set(info["vocab"])
    for i, lab in enumerate(labels):
        span_words = _words(section_text(rex, "sentence", i, _BOOK))
        assert span_words, f"{lab} span is empty"
        # a section only exists because it cleared min_terms=3 distinct vocabulary
        # terms, so its OWN bytes must still carry at least that many. A span shifted
        # onto a neighbouring sentence fails this on the shifted ones.
        assert span_words & vocab, (
            f"{lab} span carries none of the vocabulary: it addresses the wrong bytes")


def test_the_kept_map_holds_even_when_nothing_is_dropped(doc):
    """`from_text` compacts whenever it drops, so group j is not sentence j and a
    positional map misaddresses everything after the first drop. At min_terms=1 nothing
    drops (a one-term span is a witness, not a failure), so the map is the identity here
    but the LAST section is still the one a smoke test would miss."""
    rex, info = doc
    assert info["n_dropped"] == 0, "witnesses are kept, so nothing is filtered"
    last = section_text(rex, "sentence", info["n_sentences"] - 1, _BOOK)
    assert "verdict" in last.lower(), (
        "the final section addresses the wrong bytes: this is the kept-index bug")
    first = section_text(rex, "sentence", 0, _BOOK)
    assert "morning" in first.lower()


def test_spans_address_the_original_text_not_a_cleaned_copy(doc):
    rex, _info = doc
    s = sectionings_of(rex)["sentence"]
    for a, n in s.spans:
        assert 0 <= int(a) and int(a) + int(n) <= len(_BOOK)
        assert _BOOK[int(a):int(a) + int(n)].strip()


def test_section_text_reads_from_a_path_by_seek(doc, tmp_path):
    rex, _info = doc
    p = tmp_path / "book.txt"
    p.write_text(_BOOK, encoding="utf-8")
    assert (section_text(rex, "sentence", 3, path=str(p))
            == section_text(rex, "sentence", 3, _BOOK))


def test_section_text_without_raw_or_path_refuses(doc):
    rex, _info = doc
    with pytest.raises(ValueError, match="live somewhere"):
        section_text(rex, "sentence", 0)


#### the three choices the builder makes ########################################

def test_the_sentence_layer_is_a_partition(doc):
    """Not the cover: only the partition closes exactly, and Merkle requires it."""
    rex, info = doc
    assert sectionings_of(rex)["sentence"].is_partition()
    assert info["orphan_cells"] == 0, "every relation must have an owner"


def test_coarsenings_store_parent_maps_not_memberships(doc):
    rex, _info = doc
    store = sectionings_of(rex)
    for name in ("paragraph", "chapter"):
        assert store[name].is_derived, f"{name} must be derived from the finer layer"
        assert store[name].indices.size == 0
        assert store[name].parent is not None


def test_the_layers_cover_the_same_cells(doc):
    rex, _info = doc
    store = sectionings_of(rex)
    base = sorted(store["sentence"].indices.tolist())
    for name in ("paragraph", "chapter"):
        assert sorted(store[name].resolved(store).indices.tolist()) == base


def test_no_pairs_are_enumerated_by_default(doc):
    """A sentence IS one relation over its words. The pairs are not in the text, and on
    a real book they were 11,421 invented columns against 1,469 real ones."""
    rex, info = doc
    assert info["pair_mode"] == "none"
    assert int(rex.nE) == info["n_sentences"], "one relation per sentence, and no more"


#### methods, and refusing to invent a layer ####################################

def test_each_layer_records_the_convention_it_matched(doc):
    _rex, info = doc
    assert info["methods"]["document"] == "start_marker+end_marker"
    assert info["methods"]["chapter"] == "numbered_chapter"
    assert info["methods"]["paragraph"] == "blank_line"
    assert info["methods"]["sentence"] == "terminator_gate"


def test_a_document_without_headings_gets_no_chapter_layer():
    plain = ("Nothing here announces a chapter heading at all. "
             "The prose simply continues without any structural marker.\n\n"
             "A second paragraph follows the first one without ceremony. "
             "Still no heading appears anywhere in this particular text.\n")
    rex, info = build_document(plain)
    assert "chapter" not in info["layers"]
    assert "chapter" not in sectionings_of(rex)
    assert "paragraph" in info["layers"]


def test_a_text_with_no_usable_sentence_refuses_rather_than_guessing():
    with pytest.raises(ValueError, match="nothing to construct"):
        build_document("   \n\n   \n")


#### it composes with what was built on top ####################################

def test_the_merkle_tree_builds_over_the_document_hierarchy(doc):
    rex, _info = doc
    m = build_merkle(rex)
    assert m.chain == ["sentence", "paragraph", "chapter"]
    from rexgraph.merkle import verify_proof
    assert all(verify_proof(m.leaves[i], m.proof(i), m.root)
               for i in range(len(m.leaves)))


def test_it_round_trips_through_the_state_with_every_layer(doc):
    from rexgraph.io.rex_state import from_state, to_state
    rex, info = doc
    st = to_state(rex)
    back = from_state(st, verify=True)
    got = sectionings_of(back)
    assert sorted(got) == sorted(info["layers"])
    assert build_merkle(back).root.hex() == st.header["merkle"]["root"]
    assert tuple(got["sentence"].spans[0]) == tuple(
        sectionings_of(rex)["sentence"].spans[0])


def test_document_sections_resolves_a_coarsening(doc):
    rex, _info = doc
    para = document_sections(rex, "paragraph")
    assert para and all(isinstance(v, list) for v in para.values())


def test_asking_for_a_layer_the_document_lacks_names_what_it_has(doc):
    rex, _info = doc
    with pytest.raises(ValueError, match="not a layer"):
        document_sections(rex, "stanza")


#### spans are BYTE offsets, and only a multi-byte text proves it ###############

_UNICODE = (
    "The narrator said “this is a curly quotation” quite deliberately here. "
    "Another sentence follows the first one and mentions naïve café façade. "
    "A third sentence closes the paragraph with ordinary ascii words only.\n\n"
    "The coefficient αβγ appears inside this second paragraph deliberately. "
    "Yet another sentence rounds the whole sample off with plain english.\n"
)


@pytest.fixture
def unicode_doc(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text(_UNICODE, encoding="utf-8")
    return build_document(_UNICODE), str(p)


def test_seeking_a_path_matches_slicing_the_text_on_multibyte_input(unicode_doc):
    """The bug this pins returned 'ng! No, it'll never' from a real book: spans were
    CHARACTER offsets, and a text handle's seek takes an opaque cookie, so every span
    after the first multi-byte character landed mid-codepoint. ASCII fixtures cannot
    catch it, so this one must contain curly quotes, accents and Greek."""
    (rex, info), path = unicode_doc
    assert info["span_units"] == "bytes"
    n = info["n_sentences"]
    for i in range(n):
        by_path = section_text(rex, "sentence", i, path=path)
        by_raw = section_text(rex, "sentence", i, _UNICODE)
        assert by_path == by_raw, f"section {i} disagrees between seek and slice"
        assert by_path.strip(), f"section {i} is empty"
        # a fragment starting mid-word is the signature of the old bug
        assert not by_path.lstrip().startswith(("ng", "”", "”")), by_path[:20]


def test_byte_spans_address_the_encoded_file_not_the_decoded_string(unicode_doc):
    (rex, _info), path = unicode_doc
    raw_bytes = open(path, "rb").read()
    s = sectionings_of(rex)["sentence"]
    for a, n in s.spans:
        chunk = raw_bytes[int(a):int(a) + int(n)]
        # a valid span decodes cleanly; a character-offset span would split a codepoint
        chunk.decode("utf-8")


def test_a_section_containing_multibyte_text_round_trips_exactly(unicode_doc):
    (rex, info), path = unicode_doc
    base = info["base_layer"]
    found = [section_text(rex, base, i, path=path)
             for i in range(info["n_spans"])]
    joined = " ".join(found)
    assert "curly quotation" in joined
    assert "αβγ" in joined, "the Greek run must survive the byte round-trip"
    assert "naïve café façade" in joined


def test_crlf_line_endings_do_not_shift_any_span(tmp_path):
    """The bug that survived the multi-byte fix. Python's TEXT mode translates CRLF to
    LF, so the decoded string is shorter than the file and every offset past the first
    line ending is wrong. On one real book: 174,311 bytes decoding to 163,950 chars,
    3,762 CRLF pairs, and all 1,469 sections misaddressed. An LF-only fixture cannot
    catch it, so this one must use CRLF.
    """
    from rexgraph.document import read_document
    body = _BOOK.replace("\n", "\r\n")
    p = tmp_path / "crlf.txt"
    p.write_bytes(body.encode("utf-8"))

    raw, exact = read_document(str(p))
    assert exact, "the text must re-encode to the file for spans to address it"
    assert "\r\n" in raw, "read_document must NOT translate newlines"

    rex, info = build_document(raw, profile=ENGLISH_GUTENBERG)
    for i in range(info["n_sentences"]):
        assert (section_text(rex, "sentence", i, path=str(p))
                == section_text(rex, "sentence", i, raw)), f"section {i} shifted by CRLF"
    assert "verdict" in section_text(
        rex, "sentence", info["n_sentences"] - 1, path=str(p)).lower()


def test_read_document_reports_when_spans_cannot_address_the_file(tmp_path):
    """Undecodable bytes make the decode lossy, so the text no longer re-encodes to the
    file and a span means nothing against it. That must be reported, not assumed."""
    from rexgraph.document import read_document
    p = tmp_path / "bad.txt"
    p.write_bytes(b"Valid ascii sentence here about things. \xff\xfe Another one follows.\n")
    _raw, exact = read_document(str(p))
    assert exact is False


def test_no_vertex_is_invented(doc):
    """`owner_vertex` would add a hub per sentence that the text does not contain (star
    expansion, the one thing the model is built not to need) and the sentence's identity
    is already the sectioning's label and span. Vertices are words, and only words."""
    import numpy as np
    from rexgraph.core._sparse import to_scipy_csr
    rex, info = doc
    assert int(rex.nV) == len(info["vocab"]), "vertices are words, and only words"
    B = to_scipy_csr(rex._B1_dual).tocsr()
    assert (np.diff(B.indptr) > 0).all(), "every vertex participates in some relation"


def test_every_column_is_a_zero_sum_relation_at_its_own_arity(doc):
    """The boundary condition: signed and summing to zero, at ANY arity. Nothing about
    the construction may reintroduce an unsigned or unbalanced column."""
    import numpy as np
    from rexgraph.core._sparse import to_scipy_csr
    rex, _info = doc
    B = to_scipy_csr(rex._B1_dual).tocsc()
    sums = np.asarray(B.sum(axis=0)).ravel()
    arity = np.diff(B.indptr)
    # The WITNESS is the one column that does not sum to zero: `(+1)`, sum 1, and
    # `L0 u = u`. Asserting every column sums to zero asserted the witness out of
    # existence, which is the same collapse `min_terms` used to perform.
    assert np.allclose(sums[arity >= 2], 0.0), "a k>=2 boundary column sums to zero"
    assert np.allclose(sums[arity == 1], 1.0), "a witness sums to one"
    for c in range(min(int(rex.nE), 40)):
        col = B.data[B.indptr[c]:B.indptr[c + 1]]
        k = len(col)
        if k == 1:
            assert np.isclose(col[0], 1.0), "a witness is (+1)"
            continue
        assert (col < 0).sum() == 1, "exactly one head carries the -1"
        assert np.isclose(col.min(), -1.0)
        assert np.allclose(col[col > 0], 1.0 / (k - 1)), "share is 1/(k-1)"


#### the lookup: field diffusion over the exact partition ######################

def test_section_response_finds_the_section_a_query_was_lifted_from():
    """The lookup the layers exist for. No text is scanned and nothing is re-segmented:
    the query's vertices seed the document's own field, heat spreads through its
    relations, and the response is read back over the partition stored at ingest."""
    import re

    import numpy as np

    from rexgraph.partition import section_response
    from rexgraph.sectioning import sectionings_of

    rex, info = build_document(_BOOK, profile=ENGLISH_GUTENBERG)
    sect = sectionings_of(rex)["sentence"]
    index_of = {w.lower(): i for i, w in enumerate(info["vocab"])}
    target = sect.n_sections - 1
    q = section_text(rex, "sentence", target, _BOOK).lower()
    seeds = [index_of[w] for w in dict.fromkeys(re.findall(r"[a-z']+", q))
             if w in index_of]
    scores, labels = section_response(rex, sect, seeds)
    assert len(scores) == sect.n_sections == len(labels)
    assert int(np.argmax(scores)) == target, "the source section must answer loudest"


def test_inverse_degree_seeding_is_what_makes_it_work():
    """A word in most sentences says nothing about WHERE. Weighting each seed by 1/deg
    is a structural statement, not a threshold, and it is the difference between a
    working lookup and one that ranks by length: measured on a book, median rank 2 with
    it and 38-42 without."""
    import re

    import numpy as np

    from rexgraph.partition import section_response
    from rexgraph.sectioning import sectionings_of

    rex, info = build_document(_BOOK, profile=ENGLISH_GUTENBERG)
    sect = sectionings_of(rex)["sentence"]
    index_of = {w.lower(): i for i, w in enumerate(info["vocab"])}
    target = sect.n_sections - 1
    q = section_text(rex, "sentence", target, _BOOK).lower()
    seeds = [index_of[w] for w in dict.fromkeys(re.findall(r"[a-z']+", q))
             if w in index_of]
    inv, _l = section_response(rex, sect, seeds, seed_weight="invdeg")
    flat, _l2 = section_response(rex, sect, seeds, seed_weight="flat")
    r_inv = int(np.where(np.argsort(inv)[::-1] == target)[0][0])
    r_flat = int(np.where(np.argsort(flat)[::-1] == target)[0][0])
    assert r_inv <= r_flat, "inverse-degree seeding must not rank the source worse"


def test_an_empty_seed_set_answers_zero_rather_than_raising():
    import numpy as np

    from rexgraph.partition import section_response
    from rexgraph.sectioning import sectionings_of
    rex, _info = build_document(_BOOK, profile=ENGLISH_GUTENBERG)
    sect = sectionings_of(rex)["sentence"]
    scores, _labels = section_response(rex, sect, [])
    assert scores.shape == (sect.n_sections,) and not np.any(scores)


#### the span layer: gating inside the sentence #################################

_GATED = CorpusProfile(
    name="en-gated", markers=ENGLISH_GUTENBERG.markers,
    headings=ENGLISH_GUTENBERG.headings, veto=ENGLISH_GUTENBERG.veto,
    gate=frozenset({"the", "a", "your", "on", "of", "and", ",", ".", "!"}))


def test_an_empty_gate_degenerates_to_the_sentence_layer():
    """One code path. A profile that gates nothing yields one span per sentence, which
    IS the previous construction: the span layer is not a second mechanism."""
    _rex, info = build_document(_BOOK, profile=ENGLISH_GUTENBERG)
    assert info["base_layer"] == "sentence"
    assert "span" not in info["layers"]
    assert info["n_spans"] == info["n_sentences"]


def test_gating_makes_the_span_the_base_and_the_sentence_a_coarsening():
    rex, info = build_document(_BOOK, profile=_GATED)
    assert info["base_layer"] == "span"
    assert info["layers"][:2] == ["span", "sentence"]
    assert info["n_spans"] > info["n_sentences"], "the gate must actually divide"
    store = sectionings_of(rex)
    assert not store["span"].is_derived, "the span layer owns the cells"
    assert store["sentence"].is_derived, "the sentence layer is a parent map over spans"


def test_a_layer_that_divides_nothing_is_not_named():
    """A layer identical to the one above it is that layer under another name. So the
    base is only called `span` when the gate genuinely splits a sentence."""
    never = CorpusProfile(name="no-gate", markers=ENGLISH_GUTENBERG.markers,
                          gate=frozenset({"\u00a7"}))
    _rex, info = build_document(_BOOK, profile=never)
    assert info["base_layer"] == "sentence"


def test_the_vocative_witness_appears_at_the_span_layer():
    """A vocative is exactly a participant that exists and bounds nothing. At the
    SENTENCE layer it is absorbed into one branching relation; the comma gates it off
    only at the span layer."""
    text = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
            "Take away your mother, Jerry.\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n")
    rex, info = build_document(text, profile=ENGLISH_GUTENBERG)
    assert 3 in set(map(int, rex.edge_types)), "a witness exists"
    assert info["n_spans"] > info["n_sentences"], "the comma divided one sentence"
    # punctuation gates without any profile listing it, so the vocative is found even
    # with no function-word gate: a mark is a delimiter by nature
    assert set(map(int, rex.edge_types)) == {2, 3}, "one branching relation, one witness"


def test_span_byte_ranges_are_the_tokens_own_extent():
    """A span's range must cover its own tokens, not the whole sentence, or the layer
    addresses the same bytes as the one above it."""
    rex, info = build_document(_BOOK, profile=_GATED)
    store = sectionings_of(rex)
    sp = store["span"].spans
    assert sp is not None and len(sp) == info["n_spans"]
    widths = [int(n) for _a, n in sp]
    assert min(widths) > 0
    assert sum(widths) < len(_BOOK.encode("utf-8")), "spans are narrower than the text"


def test_every_layer_still_covers_the_same_cells():
    rex, _info = build_document(_BOOK, profile=_GATED)
    store = sectionings_of(rex)
    base = sorted(store["span"].indices.tolist())
    for name in ("sentence", "paragraph"):
        if name in store:
            assert sorted(store[name].resolved(store).indices.tolist()) == base


def test_both_propagators_find_the_section_and_the_boundary_one_is_free():
    """`section_response` carries two readings, not one made faster: "rl4" runs
    S0 = B1 f(RL4) B1^T and "boundary" applies L0 = B1 B1^T matrix-free. Measured over 46
    identical queries on 10 Gutenberg documents they agree exactly (97.8% top-1, 100%
    top-5, median rank 1) at 115.4 s against 0.1 s, so "boundary" is the default. RL4 is
    not affordable at document scale because a common word puts most spans in contact with
    most others: 15 to 58 million nonzeros at nE 7,000 to 17,000."""
    import numpy as np

    from rexgraph.corpus_profile import tokenize
    from rexgraph.partition import section_response
    from rexgraph.sectioning import sectionings_of

    rex, info = build_document(_BOOK, profile=ENGLISH_GUTENBERG)
    base = info["base_layer"]
    sect = sectionings_of(rex)[base]
    vocab = {str(v).lower(): i for i, v in enumerate(info["vocab"])}
    target = 1
    q = section_text(rex, base, target, _BOOK)
    seeds = [vocab[w] for w, _a, _b in tokenize(q, ENGLISH_GUTENBERG) if w in vocab]
    assert seeds

    a, _ = section_response(rex, sect, seeds, propagator="boundary")
    b, _ = section_response(rex, sect, seeds, propagator="rl4")
    assert int(np.argmax(a)) == target
    assert int(np.argmax(b)) == target


def test_the_propagator_is_a_real_switch_and_boundary_is_the_default():
    """Both readings must be reachable and the default must be the measured-cheap one.

    Note what is NOT asserted here: that a second boundary step degrades localisation.
    That is real (97.8% top-1 to 0.0% over 46 queries on 10 Gutenberg documents, and the
    same collapse one grade up on the corpus index) but it needs thousands of sections
    to show. On a nine-section fixture two steps still lands on the target, so a test here
    would pass or fail for reasons that have nothing to do with the claim. The measurement
    lives in `section_response`'s docstring and in `bench_section_localization.py`.
    """
    import inspect

    from rexgraph.partition import section_response

    sig = inspect.signature(section_response)
    assert sig.parameters["propagator"].default == "boundary"
