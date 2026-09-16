"""RCQL consumes RCDB's published snapshot API, not backend specific read recipes."""
from __future__ import annotations

import pytest
from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest

from rcql import BoundSource, Executor, SourceKindError, SourcePolicy, call, parse, query, rcdb
from rcql.binding import bind


@pytest.fixture
def store():
    module = pytest.importorskip("rcdb")
    value = module.MemoryStore()
    value.put("r", RexGraph.from_graph([0, 1], [1, 2]), analytics=False)
    value.put("r", RexGraph.from_graph([0, 1, 2], [1, 2, 3]), analytics=False)
    return value


def test_explicit_resolver_and_selected_version_provenance(store):
    executor = Executor(sources={"db": store})
    result = executor.execute(parse('FROM RCDB_GET(RCDB("db"), "r@1") RETURN STATE_HASH()'))
    assert result.values == (object_digest(store.get_version("r", 1)),)
    state = result.provenance[0]["source_state"]
    assert (state["record_id"], state["record_version"]) == ("r", 1)
    assert state["state_digest"] == result.values[0]
    assert executor.execute(query(rcdb("db"), call("RCDB_HASH", "r"))).values == (
        object_digest(store.get_version("r", 2)),
    )


@pytest.mark.parametrize("operator", ["RCDB_GET", "RCDB_HASH"])
def test_return_forms_read_once_and_observe_actual_selected_state(store, monkeypatch, operator):
    reads = []
    original = store.read_record
    def read(*args, **kwargs):
        reads.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(store, "read_record", read)
    result = Executor(sources={"db": store}).execute(parse(f'FROM $db RETURN {operator}("r@1")'))
    assert len(reads) == 1
    assert result.execution[0]["methods"] == [{
        "method": "rcdb-record-read", "record_id": "r", "record_version": 1,
        "state_digest": object_digest(store.get_version("r", 1)),
    }]
    if operator == "RCDB_GET":
        assert result.values[0].nE == 2
    else:
        assert result.values[0] == object_digest(store.get_version("r", 1))
    with pytest.raises(KeyError, match="not present"):
        Executor(sources={"db": store}).execute(parse(f'FROM $db RETURN {operator}("absent")'))


@pytest.mark.parametrize("prefix", ["", "EXPLAIN "])
def test_missing_method_refuses_the_entire_phrase_before_any_adapter(prefix):
    class Partial:
        def forbidden(self, *args, **kwargs):
            raise AssertionError("an adapter ran before the whole query was checked")
        get = history = stats = forbidden
    with pytest.raises(SourceKindError, match="read_record.*data contract"):
        Executor(sources={"db": Partial()}).execute(
            parse(prefix + 'FROM $db RETURN RCDB_STATS(), RCDB_GET("r")'))


def test_descriptor_getters_are_not_executed_as_surface_checks():
    class Partial:
        def get(self):
            pass
        def history(self):
            pass
        @property
        def read_record(self):
            raise AssertionError("surface checks must not execute descriptors")
    binding = bind("db", Partial(), SourcePolicy.allow("*"))
    assert not binding.schema.can("read_record")
    with pytest.raises(SourceKindError, match="read_record"):
        Executor(sources={"db": Partial()}).execute(parse('FROM $db RETURN RCDB_GET("r")'))


def test_resolver_never_opens_a_store_and_does_not_widen_policy(store):
    with pytest.raises(KeyError, match="unknown source"):
        Executor().execute(parse('FROM RCDB("rex:///tmp/not-bound") RETURN RCDB_STATS()'))
    with pytest.raises(TypeError, match="not an RCDB store"):
        Executor(sources={"r": store.get("r")}).execute(parse('FROM RCDB("r") RETURN RCDB_STATS()'))
    with pytest.raises(PermissionError, match="identity"):
        Executor(sources={"db": BoundSource(store, SourcePolicy.allow("records"))}).execute(
            parse('FROM RCDB_GET(RCDB("db"), "r") RETURN BETTI(0)'))


def test_explain_declares_required_backend_methods_without_reading(store, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("EXPLAIN executed a return adapter")
    monkeypatch.setattr(store, "read_record", forbidden)
    explained = Executor(sources={"db": store}).execute(
        parse('EXPLAIN FROM RCDB("db") RETURN RCDB_GET("r")')).values[0]
    assert explained["returns"][0]["source_methods"] == ["read_record"]


@pytest.mark.parametrize("permissions", [("read",), ("read", "identity"), ("history", "identity")])
def test_whole_store_hash_requires_every_observed_scope(store, monkeypatch, permissions):
    def forbidden(*args):
        pytest.fail("denied logical history hash reached storage")
    monkeypatch.setattr(store, "state_digest", forbidden)
    bound = BoundSource(store, SourcePolicy.allow(*permissions))
    with pytest.raises(PermissionError):
        Executor(sources={"db": bound}).execute(parse('FROM $db RETURN RCDB_STATE_HASH()'))
