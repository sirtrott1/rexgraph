"""Lexical source readers."""
from __future__ import annotations

import gzip

import pytest
from agent.adapters import lexical as L

_LMF = """<?xml version="1.0" encoding="UTF-8"?>
<LexicalResource>
  <Lexicon id="t" label="t" language="en" email="e" license="l" version="1">
    <LexicalEntry id="e-dog-n">
      <Lemma writtenForm="dog" partOfSpeech="n"/>
      <Sense id="s-dog-1" synset="ss-1-n"/>
    </LexicalEntry>
    <LexicalEntry id="e-hound-n">
      <Lemma writtenForm="hound" partOfSpeech="n"/>
      <Sense id="s-hound-1" synset="ss-1-n"/>
    </LexicalEntry>
    <LexicalEntry id="e-animal-n">
      <Lemma writtenForm="animal" partOfSpeech="n"/>
      <Sense id="s-animal-1" synset="ss-2-n"/>
    </LexicalEntry>
    <Synset id="ss-1-n" members="e-dog-n e-hound-n" partOfSpeech="n">
      <Definition>a domesticated canine</Definition>
      <SynsetRelation relType="hypernym" target="ss-2-n"/>
      <Example>the dog barked</Example>
    </Synset>
    <Synset id="ss-2-n" members="e-animal-n" partOfSpeech="n">
      <Definition>a living organism</Definition>
    </Synset>
  </Lexicon>
</LexicalResource>
"""


@pytest.fixture
def wn_file(tmp_path):
    p = tmp_path / "wn.xml"
    p.write_text(_LMF, encoding="utf-8")
    return p


def test_wordnet_reads_synsets_lemmas_and_relations(wn_file):
    wn = L.load_wordnet(wn_file, with_examples=True)
    assert wn["synsets"]["ss-1-n"] == ["e-dog-n", "e-hound-n"]
    assert wn["lemma_of"]["e-dog-n"] == "dog"
    assert wn["pos_of"]["e-dog-n"] == "n"
    assert ("ss-1-n", "hypernym", "ss-2-n") in wn["relations"]
    assert wn["definitions"]["ss-1-n"] == "a domesticated canine"
    assert wn["examples"]["ss-1-n"] == ["the dog barked"]
    assert wn["sense_of"]["s-dog-1"] == ("e-dog-n", "ss-1-n")


def test_wordnet_reads_a_gzipped_file_by_content_not_by_name(tmp_path):
    p = tmp_path / "wn.bin"                      # deliberately not .gz
    p.write_bytes(gzip.compress(_LMF.encode("utf-8")))
    wn = L.load_wordnet(p)
    assert len(wn["synsets"]) == 2


def test_a_synset_becomes_a_group_and_a_singleton_is_dropped(wn_file):
    wn = L.load_wordnet(wn_file)
    groups, labels = L.wordnet_groups(wn, include_relations=False)
    assert ["dog", "hound"] in groups, "the synset IS the group"
    assert ["animal"] not in groups, "a group of one bounds nothing"
    assert labels[0] == ("synset", "ss-1-n")


def test_by_lemma_merges_spellings_and_by_entry_keeps_them_apart(wn_file):
    wn = L.load_wordnet(wn_file)
    by_lemma, _ = L.wordnet_groups(wn, include_relations=False, by="lemma")
    by_entry, _ = L.wordnet_groups(wn, include_relations=False, by="entry")
    assert by_lemma == [["dog", "hound"]]
    assert by_entry == [["e-dog-n", "e-hound-n"]]


def test_a_synset_relation_is_two_representatives_not_a_clique(wn_file):
    wn = L.load_wordnet(wn_file)
    groups, labels = L.wordnet_groups(wn, include_relations=True)
    rel = [g for g, lab in zip(groups, labels, strict=True) if lab[0] == "hypernym"]
    assert rel == [["dog", "animal"]]


def test_the_groups_feed_the_constructor(wn_file):
    from rexgraph.construct import from_groups
    wn = L.load_wordnet(wn_file)
    groups, _ = L.wordnet_groups(wn)
    rex, info = from_groups(groups, verify=False)
    assert int(rex.nE) == info["n_wide"] + info["n_pairs"]
    assert set(info["vertex_of"]) == {"dog", "hound", "animal"}


_ROGET = """
#1. Existence.—N. existence, being, entity, _ens_, _esse_,
subsistence.

     reality, actuality; positiveness &c. adj.; fact, matter of fact &c. 494.

#2. Inexistence.—N. inexistence; nonexistence, nonsubsistence.
"""


def test_roget_strips_the_pos_heading_and_the_cross_reference_tail(tmp_path):
    p = tmp_path / "roget.txt"
    p.write_text(_ROGET, encoding="utf-8")
    cats = L.load_roget(p)
    t = cats[1]["terms"]
    assert cats[1]["name"] == "Existence"
    assert "existence" in t and "being" in t and "entity" in t
    assert "n existence" not in t, "the leading N. is a part-of-speech heading"
    assert "positiveness adj" not in t, "&c. adj. is a tail, not part of the term"
    assert "positiveness" in t
    # a sentence period ends a term list too
    assert "subsistence" in t and "reality" in t
    assert not any(x == "subsistence reality" for x in t)
    assert 2 in cats and "nonexistence" in cats[2]["terms"]


def test_nrc_vad_is_three_values_per_word(tmp_path):
    p = tmp_path / "vad.txt"
    p.write_text("happy\t1.000\t0.735\t0.772\ngrief\t0.070\t0.640\t0.474\n",
                 encoding="utf-8")
    vad = L.load_nrc_vad(p)
    assert vad["happy"] == (1.0, 0.735, 0.772)
    assert vad["grief"][0] < vad["happy"][0], "valence orders these the right way"
    assert len(vad) == 2


def test_nrc_vad_skips_a_header_rather_than_raising(tmp_path):
    p = tmp_path / "vad.txt"
    p.write_text("word\tvalence\tarousal\tdominance\nhappy\t1.0\t0.7\t0.8\n",
                 encoding="utf-8")
    assert set(L.load_nrc_vad(p)) == {"happy"}


def test_emolex_keeps_only_the_emotions_marked_present(tmp_path):
    p = tmp_path / "emo.txt"
    p.write_text("grief\tanger\t0\ngrief\tsadness\t1\ngrief\tnegative\t1\n"
                 "aback\tanger\t0\n", encoding="utf-8")
    emo = L.load_nrc_emolex(p)
    assert sorted(emo["grief"]) == ["negative", "sadness"]
    assert emo["aback"] == {}, "a word with no association is present but empty"


def test_the_connotation_lexicons_are_vertex_values_not_relations(tmp_path):
    p = tmp_path / "vad.txt"
    p.write_text("happy\t1.0\t0.7\t0.8\nsad\t0.1\t0.4\t0.3\n", encoding="utf-8")
    vad = L.load_nrc_vad(p)
    assert all(isinstance(v, tuple) and len(v) == 3 for v in vad.values())
    assert not any(isinstance(v, (list, set)) for v in vad.values())


#### the grammar layer: frames are oriented relations, recorded not inferred ####

_LMF_FRAMES = """<?xml version="1.0" encoding="UTF-8"?>
<LexicalResource>
  <Lexicon id="t" label="t" language="en" email="e" license="l" version="1">
    <LexicalEntry id="e-take-v">
      <Lemma writtenForm="take" partOfSpeech="v"/>
      <Sense id="s-take-1" synset="ss-1-v" subcat="vtai vtii"/>
    </LexicalEntry>
    <LexicalEntry id="e-take-n">
      <Lemma writtenForm="take" partOfSpeech="n"/>
      <Sense id="s-take-n1" synset="ss-2-n"/>
    </LexicalEntry>
    <Synset id="ss-1-v" members="e-take-v" partOfSpeech="v"/>
    <Synset id="ss-2-n" members="e-take-n" partOfSpeech="n"/>
    <SyntacticBehaviour id="vtai" subcategorizationFrame="Somebody ----s something"/>
    <SyntacticBehaviour id="vtii" subcategorizationFrame="Something ----s something"/>
    <SyntacticBehaviour id="ditransitive"
                        subcategorizationFrame="Somebody ----s somebody something"/>
    <SyntacticBehaviour id="vtaa-with"
                        subcategorizationFrame="Somebody ----s somebody with something"/>
  </Lexicon>
</LexicalResource>
"""


@pytest.fixture
def wn_frames(tmp_path):
    p = tmp_path / "wnf.xml"
    p.write_text(_LMF_FRAMES, encoding="utf-8")
    return L.load_wordnet(p)


def test_the_frames_are_read_as_recorded_fact(wn_frames):
    assert wn_frames["frames"]["vtai"] == "Somebody ----s something"
    assert wn_frames["frames"]["ditransitive"] == "Somebody ----s somebody something"


def test_a_sense_names_the_frames_it_admits(wn_frames):
    assert wn_frames["frames_of"]["s-take-1"] == ["vtai", "vtii"]


def test_a_noun_sense_admits_no_frame_and_that_is_the_pos_evidence(wn_frames):
    assert "s-take-n1" not in wn_frames["frames_of"]
    assert "s-take-1" in wn_frames["frames_of"]


def test_a_frame_fixes_the_arity_and_therefore_the_share(wn_frames):
    slots, unknown = L.frame_slots(wn_frames["frames"]["ditransitive"])
    assert slots == ["somebody", "somebody", "something"] and not unknown
    k = 1 + len(slots)
    assert k == 4, "agent, recipient, theme, plus the verb that heads them"


def test_slots_match_by_token_not_by_substring(wn_frames):
    slots, unknown = L.frame_slots(wn_frames["frames"]["vtaa-with"])
    assert slots == ["somebody", "somebody", "something"], slots
    assert not unknown
    assert 1 + len(slots) == 4


def test_an_unrecognised_filler_is_reported_not_dropped():
    slots, unknown = L.frame_slots("Somebody ----s wibble")
    assert slots == ["somebody"] and unknown == ["wibble"]


def test_a_frame_without_the_verb_marker_yields_nothing():
    assert L.frame_slots("not a frame") == ([], [])
    assert L.frame_slots("") == ([], [])


def test_a_digest_written_under_the_older_framing_still_verifies(tmp_path):
    import numpy as np
    from agent.adapters import wiktionary as WK
    from rexgraph.io.rex_state import state_digest

    idx = {"format": 1, "lang_code": "en", "jsonl": "", "n_entries": 1, "n_words": 2,
           "words": ["a", "b"], "pos": ["noun"], "kinds": ["synonyms"],
           "entry_word": np.zeros(1, np.uint16), "entry_pos": np.zeros(1, np.uint16),
           "entry_offset": np.zeros(1, np.int64), "entry_length": np.ones(1, np.uint16),
           "link_src": np.zeros(1, np.uint16), "link_dst": np.ones(1, np.uint16),
           "link_kind": np.zeros(1, np.uint16)}
    p = tmp_path / "i.rexidx"
    WK.write_index(str(p), idx)
    assert WK.read_index(str(p), verify=True)["n_words"] == 2

    # rewrite the metadata as a pre-change writer would have: algo-1 digest, no stamp
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
    with safe_open(str(p), "numpy") as fh:
        meta = dict(fh.metadata() or {})
    t = load_file(str(p))
    meta.pop("digest_algo", None)
    meta["digest"] = state_digest(t, algo=1)
    save_file(t, str(p), metadata=meta)
    assert WK.read_index(str(p), verify=True)["n_words"] == 2, (
        "an unstamped index is OLD, not corrupt")


def test_a_tampered_index_is_still_rejected(tmp_path):
    import numpy as np
    import pytest as _pytest
    from agent.adapters import wiktionary as WK
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file

    idx = {"format": 1, "lang_code": "en", "jsonl": "", "n_entries": 1, "n_words": 2,
           "words": ["a", "b"], "pos": ["noun"], "kinds": ["synonyms"],
           "entry_word": np.zeros(1, np.uint16), "entry_pos": np.zeros(1, np.uint16),
           "entry_offset": np.zeros(1, np.int64), "entry_length": np.ones(1, np.uint16),
           "link_src": np.zeros(1, np.uint16), "link_dst": np.ones(1, np.uint16),
           "link_kind": np.zeros(1, np.uint16)}
    p = tmp_path / "i.rexidx"
    WK.write_index(str(p), idx)
    with safe_open(str(p), "numpy") as fh:
        meta = dict(fh.metadata() or {})
    t = load_file(str(p))
    t["link_dst"] = np.zeros(1, np.uint16)
    save_file(t, str(p), metadata=meta)
    with _pytest.raises(ValueError, match="digest mismatch"):
        WK.read_index(str(p), verify=True)


#### edge types: the gap that made every stored lexical column anonymous ########
def test_edge_types_are_stored_as_codes_beside_a_name_table(tmp_path):
    from agent.adapters.lexical_store import _put
    from agent.rcdb import FileStore
    from rexgraph.construct import from_groups

    groups = [["a", "b", "c"], ["b", "d"], ["e", "f"]]
    labels = [("synonyms", "a"), ("antonyms", "b"), ("synonyms", "e")]
    rex, info = from_groups(groups, pair_mode="none", verify=False)
    store = FileStore(str(tmp_path))
    _put(store, "lex:test", rex, info, source="t", kind="linkage_graph",
         group_labels=labels, log=lambda *a: None)

    meta = store.get_record("lex:test").meta
    names = meta["type_names"]
    assert [names[c] for c in meta["edge_types"]] == ["synonyms", "antonyms", "synonyms"]
    assert len(meta["edge_types"]) == int(rex.nE)         # one code per COLUMN
    assert len(names) == 2                                 # codes, not repeated strings
    # the head is position 0 of its own column, so storing it again would be storing the
    # boundary twice
    assert "group_names" not in meta


def test_a_roget_category_keeps_its_name_because_the_column_cannot_say_it(tmp_path):
    from agent.adapters.lexical_store import _edge_types

    codes, names, gnames = _edge_types([("category", "Existence"), ("category", "Arms")])
    assert codes == [0, 0] and names == ["category"]
    assert gnames == ["Existence", "Arms"]
