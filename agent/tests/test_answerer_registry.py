"""Answerer registry and hive membership."""
import pytest

from agent import hive as hivemod
from agent.answerers import (exact_answers, register_with_hive, registered)


@pytest.fixture(autouse=True)
def _fresh_hive():
    hivemod.reset_hive()
    yield
    hivemod.reset_hive()


@pytest.fixture
def lexicon():
    import os

    from agent.answerers.lexical import DEFAULT_WORDNET
    from agent.answerers.linkage import DEFAULT_ROGET, DEFAULT_WIKTIONARY
    present = [p for p in (DEFAULT_WORDNET, DEFAULT_ROGET, DEFAULT_WIKTIONARY)
               if os.path.exists(p)]
    if len(present) < 2:
        pytest.skip("the union needs at least two stored structures to answer from; "
                    f"{len(present)} on disk")
    return present


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
