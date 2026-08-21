"""Wiktionary binary index."""
from __future__ import annotations

import json

import numpy as np
import pytest
from agent.adapters import wiktionary as W

_ENTRIES = [
    {"word": "dog", "pos": "noun", "lang_code": "en",
     "senses": [{"glosses": ["a canine"], "synonyms": [{"word": "hound"}]}],
     "synonyms": [{"word": "canine"}, {"word": "hound"}],
     "hypernyms": [{"word": "animal"}]},
    {"word": "cat", "pos": "noun", "lang_code": "en",
     "senses": [{"glosses": ["a feline"]}],
     "synonyms": [{"word": "feline"}], "antonyms": [{"word": "dog"}]},
    {"word": "chien", "pos": "noun", "lang_code": "fr",
     "senses": [{"glosses": ["a dog"]}], "synonyms": [{"word": "toutou"}]},
    {"word": "solo", "pos": "noun", "lang_code": "en", "senses": [{"glosses": ["alone"]}]},
]


@pytest.fixture
def heap(tmp_path):
    p = tmp_path / "wikt.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for e in _ENTRIES:
            fh.write(json.dumps(e) + "\n")
    return p


def test_only_the_requested_language_is_indexed(heap):
    idx = W.build_index(heap, lang_code="en")
    assert idx["n_entries"] == 3, "the French entry is skipped"
    assert "chien" not in [idx["words"][int(c)] for c in idx["entry_word"]]


def test_the_byte_pointer_recovers_the_entry_exactly(heap):
    idx = W.build_index(heap)
    for i in range(idx["n_entries"]):
        raw = W.raw_entry(idx, i, jsonl_path=heap)
        assert raw["word"] == idx["words"][int(idx["entry_word"][i])]
        assert raw["pos"] == idx["pos"][int(idx["entry_pos"][i])]


def test_the_index_round_trips_through_safetensors_with_a_digest(heap, tmp_path):
    idx = W.build_index(heap)
    p = tmp_path / "wikt.safetensors"
    W.write_index(p, idx)
    back = W.read_index(p, verify=True)
    assert list(back["words"]) == list(idx["words"])
    assert list(back["kinds"]) == list(idx["kinds"])
    assert np.array_equal(np.asarray(back["entry_offset"]),
                          np.asarray(idx["entry_offset"]))
    assert back["n_entries"] == idx["n_entries"]
    # and the pointers still land after the round trip
    assert W.raw_entry(back, 0, jsonl_path=heap)["word"] == "dog"


def test_a_tampered_index_raises_rather_than_being_read(heap, tmp_path):
    from safetensors.numpy import load_file, save_file
    idx = W.build_index(heap)
    p = tmp_path / "wikt.safetensors"
    W.write_index(p, idx)
    t = load_file(str(p))
    t["entry_offset"] = np.asarray(t["entry_offset"]) + 1        # move every pointer
    from safetensors import safe_open
    with safe_open(str(p), "numpy") as fh:
        meta = dict(fh.metadata() or {})
    save_file(t, str(p), metadata=meta)                          # stale digest
    with pytest.raises(ValueError, match="digest mismatch"):
        W.read_index(p, verify=True)


def test_linkages_become_groups_with_the_head_first(heap):
    idx = W.build_index(heap)
    groups, labels = W.wiktionary_groups(idx, kinds=("synonyms",))
    by_head = {h: g for (k, h), g in zip(labels, groups, strict=True)}
    assert by_head["dog"][0] == "dog", "the head is the distinguished vertex"
    assert set(by_head["dog"]) == {"dog", "canine", "hound"}, \
        "entry-level and sense-level synonyms both count"


def test_kinds_are_kept_apart_because_they_are_different_relations(heap):
    idx = W.build_index(heap)
    syn, syn_lab = W.wiktionary_groups(idx, kinds=("synonyms",))
    ant, ant_lab = W.wiktionary_groups(idx, kinds=("antonyms",))
    assert all(k == "synonyms" for k, _ in syn_lab)
    assert all(k == "antonyms" for k, _ in ant_lab)
    cat_ant = [g for (k, h), g in zip(ant_lab, ant, strict=True) if h == "cat"]
    assert cat_ant and set(cat_ant[0]) == {"cat", "dog"}
    # merging them would assert that a synonym and an antonym are the same relation
    both, _ = W.wiktionary_groups(idx, kinds=("synonyms", "antonyms"))
    assert len(both) == len(syn) + len(ant)


def test_a_word_with_no_linkage_produces_no_group(heap):
    idx = W.build_index(heap)
    groups, labels = W.wiktionary_groups(idx)
    assert "solo" not in [h for _k, h in labels], "a group of one bounds nothing"


def test_the_groups_feed_the_constructor(heap):
    from rexgraph.construct import from_groups
    idx = W.build_index(heap)
    groups, _ = W.wiktionary_groups(idx)
    rex, info = from_groups(groups, verify=False)
    assert int(rex.nE) == info["n_wide"] + info["n_pairs"]
    assert "dog" in info["vertex_of"]


def test_a_missing_heap_says_so_instead_of_returning_nothing(heap, tmp_path):
    idx = W.build_index(heap)
    idx["jsonl"] = str(tmp_path / "gone.jsonl")
    with pytest.raises(FileNotFoundError, match="heap is not where"):
        W.raw_entry(idx, 0)
