"""Answerer registry and hive membership."""
import pytest

from agent import hive as hivemod
from agent.answerers import (exact_answers, register_with_hive, registered)


@pytest.fixture(autouse=True)
def _fresh_hive(lexical_sources, monkeypatch):
    from agent import answerers
    from agent.answerers import lexical, linkage
    monkeypatch.setattr(answerers, "_REGISTRY", {})
    monkeypatch.setattr(lexical, "DEFAULT_WORDNET", str(lexical_sources["wordnet"]))
    monkeypatch.setattr(linkage, "DEFAULT_ROGET", str(lexical_sources["roget"]))
    hivemod.reset_hive()
    yield
    hivemod.reset_hive()


@pytest.fixture
def lexicon(lexical_sources, tmp_path, monkeypatch):
    pytest.importorskip("safetensors", reason="the Wiktionary index requires safetensors")
    from agent.adapters import wiktionary as W
    from agent.answerers import linkage
    path = tmp_path / "wiktionary.safetensors"
    W.write_index(path, W.build_index(lexical_sources["wiktionary"]))
    monkeypatch.setattr(linkage, "DEFAULT_WIKTIONARY", str(path))
    return lexical_sources


def test_every_answerer_becomes_a_hive_member_with_its_own_capability():
    names = register_with_hive()
    h = hivemod.get_hive()
    caps = registered()
    assert names and set(names) == set(caps)
    for name in names:
        assert name in h.providers(caps[name])


def test_a_worker_type_is_scoped_so_the_ontology_can_read_it():
    register_with_hive()
    h = hivemod.get_hive()
    for bee in h._bees.values():
        if bee.name in registered():
            assert bee.worker_type.startswith("answerer:")


def test_dispatch_picks_one_provider_and_the_union_asks_them_all(lexicon):
    register_with_hive()
    h = hivemod.get_hive()
    q = "what is related to harpoon"

    one = h.dispatch_capability("relate", {"query": q})
    assert one["worker"] in h.providers("relate")

    every = exact_answers(q)
    sources = {g["answerer"] for g in every}
    assert len(sources) > 1                       # more than the dispatched one
    assert one["worker"] in sources


def test_the_union_carries_each_answer_with_its_own_provenance(lexicon):
    got = exact_answers("what is related to harpoon")
    assert got
    for g in got:
        assert g["answerer"] and g["source"] and g["kind"]
        assert g["result"]["answered"]


def test_registering_twice_does_not_duplicate_members():
    register_with_hive()
    n = len(hivemod.get_hive()._bees)
    register_with_hive()
    assert len(hivemod.get_hive()._bees) == n
