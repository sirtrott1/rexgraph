"""Load synthetic WordNet records and retain their sense relations."""
import pytest

from agent import senses
from agent.senses import SenseModel
from _rexgraph_agent_tests.lexical_fixtures import write_wordnet


@pytest.fixture
def wn(lexical_sources, monkeypatch):
    monkeypatch.setattr(senses, "_INVENTORY", None)
    monkeypatch.setattr(senses, "_INVENTORY_KEY", None, raising=False)
    monkeypatch.setattr(senses, "_INVENTORY_RAW", None, raising=False)
    return senses.inventory(lexical_sources["wordnet"])


def _model(wn, *, hops=1):
    return SenseModel.for_word("bank", *wn, hops=hops)


def test_loaded_relations_select_the_recorded_sense(wn):
    assert _model(wn).disambiguate(["currency"])["senses"] == ["bank-money"]
    assert _model(wn).disambiguate(["tide"])["senses"] == ["bank-river"]


def test_loaded_shared_context_reports_every_tied_sense(wn):
    result = _model(wn).disambiguate(["currency", "tide"])
    assert result["tied"] and not result["abstain"]
    assert set(result["senses"]) == {"bank-money", "bank-river"}


def test_zero_hops_excludes_neighbouring_synset_members(wn):
    result = _model(wn, hops=0).disambiguate(["currency"])
    assert result["abstain"] and not result["senses"]


def test_two_hops_include_the_shared_neighbour(wn):
    result = _model(wn, hops=2).disambiguate(["currency"])
    assert result["tied"] and set(result["senses"]) == {"bank-money", "bank-river"}


def test_inventory_does_not_reuse_a_different_file(wn, tmp_path):
    path = write_wordnet(tmp_path / "other.xml", [
        ("other", ["alpha", "alias"], "a separate synthetic entry", []),
    ])
    synsets, by_lemma, relations = senses.inventory(path)
    assert set(synsets) == {"other"}
    assert "alpha" in by_lemma and "bank" not in by_lemma
    assert not relations


def test_inventory_refreshes_when_the_source_file_changes(lexical_sources, wn):
    path = lexical_sources["wordnet"]
    original = path.read_bytes()
    try:
        write_wordnet(path, [("changed", ["alpha", "alias"], "changed entry", [])])
        assert set(senses.inventory(path)[0]) == {"changed"}
    finally:
        path.write_bytes(original)


def test_inventory_follows_a_replaced_registered_source(lexical_sources, wn, tmp_path, monkeypatch):
    from agent import answerers
    from agent.answerers.lexical import LexicalAnswerer
    monkeypatch.setattr(answerers, "_REGISTRY", {})
    answerers.register("lexical", LexicalAnswerer(path=str(lexical_sources["wordnet"])))
    assert "bank-money" in senses.inventory()[0]
    path = write_wordnet(tmp_path / "replacement.xml", [
        ("replacement", ["alpha", "alias"], "replacement entry", []),
    ])
    answerers.register("lexical", LexicalAnswerer(path=str(path)))
    assert set(senses.inventory()[0]) == {"replacement"}
