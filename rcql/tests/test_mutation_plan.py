"""Mutation input expressions use the same typed native runtime as read queries."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
from rexgraph.graph import RexGraph, TemporalRex

from rcql import (
    BoundSource,
    Executor,
    MutationQuery,
    SourcePolicy,
    call,
    mutation,
    param,
    parse,
    source,
)
from rcql.operators import _REGISTRY


@pytest.fixture
def store():
    module = pytest.importorskip("rcdb")
    return module.MemoryStore().configure_security(require_commits=True)


@pytest.fixture
def rex():
    return RexGraph.from_graph([0, 1], [1, 2])


def test_text_and_builder_share_one_commit_plan_and_lineage(store, rex):
    text = ('FROM RCDB("db") MUTATE "r" SET state = $candidate, actor = "Art", '
            'valid_from = 10, valid_to = 20, expected_version = 0 COMMIT')
    parsed = parse(text)
    built = mutation(call("RCDB", "db"), "r", param("candidate"), actor="Art",
                     valid_from=10, valid_to=20, expected_version=0)
    assert isinstance(parsed, MutationQuery)
    assert parsed == built
    result = Executor(sources={"db": store}, params={"candidate": rex}).execute(parsed)
    assert result.values[0].version == 1
    assert store.verify_commits("r")
    terminal = result.native_plan["nodes"][-1]
    assert terminal["kind"] == "commit" and not terminal["reusable"]
    assert terminal["requires"] == ["identity", "mutate"]
    assert result.provenance[0]["state_digest"] == store.read_record("r").state_digest
    assert result.execution[-1]["methods"][0]["record_version"] == 1
    assert json.loads(json.dumps(result.native_plan, allow_nan=False)) == result.native_plan


def test_explain_never_reads_prepares_signs_or_publishes(store, rex, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("EXPLAIN reached storage or signing")
    for name in ("read_record", "history", "commit_mutation"):
        monkeypatch.setattr(store, name, forbidden)
    import rexgraph.io.mutation as core
    monkeypatch.setattr(core, "prepare_mutation", forbidden)
    executor = Executor(sources={"db": store}, params={"r": rex})
    result = executor.execute(parse('EXPLAIN FROM $db MUTATE "r" SET state = $r COMMIT'))
    assert result.values == (result.native_plan,)
    assert result.execution == ()
    assert result.native_plan["nodes"][-1]["physical"]["adapter"] == "rcdb.commit_mutation"


@pytest.mark.parametrize("prefix", ["", "EXPLAIN "])
@pytest.mark.parametrize("field, literal, message", [
    ("actor", "12", "actor"), ("valid_from", "TRUE", "finite real"),
    ("expected_version", "-1", "nonnegative"), ("expected_version", "1.0", "nonnegative"),
    ("valid_to", "NONE", None),
])
def test_all_fields_are_checked_before_any_input_adapter(store, monkeypatch, prefix, field, literal, message):
    original = _REGISTRY["RCDB_GET"]
    def forbidden(*args):
        pytest.fail("an input adapter ran before mutation fields were checked")
    monkeypatch.setitem(_REGISTRY, "RCDB_GET", replace(original, fn=forbidden))
    request = parse(prefix + f'FROM $db MUTATE "r" SET state = RCDB_GET("other"), {field} = {literal} COMMIT')
    if message is not None:
        with pytest.raises((TypeError, ValueError), match=message):
            Executor(sources={"db": store}).execute(request)
    elif prefix:
        assert Executor(sources={"db": store}).execute(request).execution == ()
    else:
        # Valid runtime inputs intentionally reach the adapter: the spy must fire.
        with pytest.raises(pytest.fail.Exception, match="input adapter ran"):
            Executor(sources={"db": store}).execute(request)


def test_let_candidate_executes_once_in_native_dag(store, rex, monkeypatch):
    store.commit_mutation("old", rex, analytics=False)
    original = store.read_record
    reads = []
    def read(*args, **kwargs):
        reads.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(store, "read_record", read)
    result = Executor(sources={"db": store}).execute(parse(
        'FROM $db LET candidate = RCDB_GET("old") '
        'MUTATE "new" SET state = candidate, expected_version = 0 COMMIT'))
    assert reads == [("old",)]
    assert [item["operator"] for item in result.execution] == ["RCDB_GET", "COMMIT"]
    assert store.verify_commits("new")


@pytest.mark.parametrize("permissions", [("identity",), ("mutate",), ("read",)])
def test_mutation_keeps_both_capabilities_before_storage(store, rex, permissions, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("denied mutation reached storage")
    monkeypatch.setattr(store, "commit_mutation", forbidden)
    bound = BoundSource(store, SourcePolicy.allow(*permissions))
    with pytest.raises(PermissionError):
        Executor(sources={"db": bound}).execute(mutation(source("db"), "r", rex))


@pytest.mark.parametrize("id, value, options", [
    (1, "rex", {}), ("", "rex", {}), ("r", "dict", {}), ("r", "history", {}),
    ("r", "rex", {"valid_from": 4, "valid_to": 4}),
    ("r", "rex", {"valid_from": 5, "valid_to": 4}),
    ("r", "rex", {"valid_from": float("nan")}),
])
def test_invalid_candidate_or_interval_never_publishes(store, rex, id, value, options):
    candidate = {"rex": rex, "dict": {}, "history": TemporalRex([])}[value]
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"db": store}).execute(mutation(source("db"), id, candidate, **options))
    assert store.list() == []


@pytest.mark.parametrize("text", [
    'FROM $db MUTATE "r" SET actor = "a" COMMIT',
    'FROM $db MUTATE "r" SET state = $r, state = $r COMMIT',
    'FROM $db MUTATE "r" SET state = $r, signer = $key COMMIT',
    'FROM $db MUTATE "r" SET state = $r',
    'FROM $db MUTATE "r" SET state = $r COMMIT RETURN STATE_HASH()',
])
def test_mutation_grammar_refuses_missing_state_commit_and_unknown_fields(text):
    with pytest.raises(SyntaxError):
        parse(text)


def test_source_arguments_cannot_execute_an_untyped_expression(monkeypatch):
    def forbidden(*args):
        pytest.fail("FROM argument executed an untyped operator")
    monkeypatch.setitem(_REGISTRY, "SHOW_OPERATORS", replace(_REGISTRY["SHOW_OPERATORS"], fn=forbidden))
    with pytest.raises(TypeError, match="FROM arguments"):
        Executor().execute(parse('FROM RCDB(SHOW_OPERATORS()) RETURN RCDB_STATS()'))


def test_runtime_rechecks_a_lying_adapter_result_before_publication(store, monkeypatch):
    monkeypatch.setitem(_REGISTRY, "RCDB_GET", replace(_REGISTRY["RCDB_GET"], fn=lambda *args: {}))
    with pytest.raises(TypeError, match="native RexGraph"):
        Executor(sources={"db": store}).execute(parse(
            'FROM $db MUTATE "new" SET state = RCDB_GET("old") COMMIT'))
    assert store.list() == []
