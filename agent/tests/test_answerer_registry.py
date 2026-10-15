"""Answerer registration preserves declared capabilities and source provenance."""
import pytest

from agent import hive as hivemod
from agent.answerers import exact_answers, register, register_with_hive, registered
from agent.answerers.ontology import OntologyAnswerer


@pytest.fixture(autouse=True)
def configured_answerers(tmp_path, monkeypatch):
    from agent import answerers
    monkeypatch.setattr(answerers, "_REGISTRY", {})
    monkeypatch.delenv("REXGRAPH_ONTOLOGY", raising=False)
    hivemod.reset_hive()
    for name in ("alpha", "beta"):
        path = tmp_path / (name + ".obo")
        path.write_text("format-version: 1.2\n\n[Term]\nid: X:0\nname: parent " + name +
                        "\n\n[Term]\nid: X:1\nname: sample\nis_a: X:0\n")
        register(name, OntologyAnswerer.from_file(path, name=name))
    yield
    hivemod.reset_hive()


def test_every_answerer_becomes_a_hive_member_with_its_own_capability():
    names = register_with_hive()
    h = hivemod.get_hive()
    caps = registered()
    assert set(names) == set(caps) == {"alpha", "beta"}
    for name in names:
        assert name in h.providers(caps[name])


def test_a_worker_type_is_scoped_so_the_ontology_can_read_it():
    register_with_hive()
    for bee in hivemod.get_hive()._bees.values():
        if bee.name in registered():
            assert bee.worker_type.startswith("answerer:")


def test_dispatch_picks_one_provider_and_the_union_asks_them_all():
    register_with_hive()
    hive = hivemod.get_hive()
    query = "what is a sample"
    one = hive.dispatch_capability("classify", {"query": query})
    sources = {got["answerer"] for got in exact_answers(query)}
    assert sources == {"alpha", "beta"}
    assert one["worker"] in sources


def test_the_union_carries_each_answer_with_its_own_provenance():
    answers = exact_answers("what is a sample")
    assert len(answers) == 2
    for got in answers:
        assert got["answerer"] and got["source"] and got["kind"]
        assert got["result"]["answered"]


def test_registering_twice_does_not_duplicate_members():
    register_with_hive()
    count = len(hivemod.get_hive()._bees)
    register_with_hive()
    assert len(hivemod.get_hive()._bees) == count


def test_unconfigured_registry_does_not_select_any_data_source(monkeypatch):
    from agent import answerers
    monkeypatch.setattr(answerers, "_REGISTRY", {})
    def refuse(*args, **kwargs):
        pytest.fail("an unconfigured answerer attempted to read a file")
    monkeypatch.setattr(OntologyAnswerer, "from_file", refuse)
    assert registered() == {}
    assert exact_answers("what is a sample") == []
