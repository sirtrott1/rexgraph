"""Typed corpus responses with policy projection before accession arithmetic."""
from fractions import Fraction as Q

import pytest
from rcdb import MemoryStore
from rexgraph import RexGraph
from rcql import BoundSource, Executor, SourcePolicy, parse


@pytest.fixture
def store():
    store = MemoryStore()
    rex = RexGraph.from_hypergraph([0, 2], [0, 1])
    store.put("a", rex, tags=["x", "y"], meta={"vertex_labels": ["secret"]}, analytics=False)
    store.put("b", rex, tags=["x"], analytics=False)
    return store


@pytest.mark.parametrize("reading", ["share", "existence"])
@pytest.mark.parametrize("exact", [True, False])
def test_response_and_members_match_store(store, reading, exact):
    result = Executor(sources={"db": store}, params={"reading": reading, "exact": exact}).execute(parse(
        'FROM $db LET f=CORPUS_FIELD(["x"],reading=$reading,exact=$exact) '
        'RETURN f,f.scores,f.ids,f.versions,f.snapshot_digest,COUNT(f.scores)'))
    expected = store.corpus_snapshot().response(["x"], reading=reading, exact=exact)
    assert result.values == (expected, expected["scores"], ("a", "b"), (1, 1),
                             expected["snapshot_digest"], 2)
    assert result.exactness[1].value == ("rational" if exact else "approximate")
    assert any(m["method"] == "rcdb-native-corpus-response"
               for event in result.execution for m in event.get("methods", []))


def test_explain_does_not_read_corpus(store, monkeypatch):
    monkeypatch.setattr(store, "corpus_snapshot", lambda **kw: pytest.fail("snapshot read"))
    out = Executor(sources={"db": store}).execute(parse(
        'EXPLAIN FROM $db RETURN CORPUS_FIELD(["x"])'))
    assert not out.execution


@pytest.mark.parametrize("expression", [
    'CORPUS_FIELD("x")', 'CORPUS_FIELD([1])', 'CORPUS_FIELD([true])',
    'CORPUS_FIELD(["x"],reading="mass")', 'CORPUS_FIELD([],exact=1)',
    'CORPUS_FIELD([],as_of=true)', 'CORPUS_FIELD([],valid_at="yesterday")',
    'CORPUS_FIELD([]).hidden',
])
@pytest.mark.parametrize("explain", ["", "EXPLAIN "])
def test_bad_contracts_fail_before_execution(store, expression, explain, monkeypatch):
    monkeypatch.setattr(store, "corpus_snapshot", lambda **kw: pytest.fail("snapshot read"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"db": store}).execute(parse(f'{explain}FROM $db RETURN {expression}'))


@pytest.mark.parametrize("missing", ["read", "records", "search", "identity"])
def test_required_authority_is_checked_before_snapshot(store, missing, monkeypatch):
    monkeypatch.setattr(store, "corpus_snapshot", lambda **kw: pytest.fail("snapshot read"))
    permissions = {"read", "records", "search", "identity"} - {missing}
    source = BoundSource(store, SourcePolicy.allow(*permissions))
    with pytest.raises(PermissionError):
        Executor(sources={"db": source}).execute(parse('FROM $db RETURN CORPUS_FIELD(["x"])'))


def test_projected_signature_cannot_score_hidden_metadata(store):
    source = BoundSource(store, SourcePolicy.allow("*", record_fields=["tags"]))
    out = Executor(sources={"db": source}).execute(parse(
        'FROM $db RETURN CORPUS_FIELD(["secret"]).scores,CORPUS_FIELD(["x"]).scores'))
    assert out.values == ((Q(0), Q(0)), (Q(1, 2), Q(1, 2)))


def test_current_field_changes_after_publication(store):
    ex = Executor(sources={"db": store})
    query = parse('FROM $db RETURN CORPUS_FIELD(["x"])')
    before = ex.execute(query).values[0]
    store.delete("a")
    after = ex.execute(query).values[0]
    assert before["ids"] == ("a", "b") and after["ids"] == ("b",)
    assert after["scores"] == (Q(1),)
    assert after["snapshot_digest"] != before["snapshot_digest"]
