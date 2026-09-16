"""RCDB's new methods retain Agent's existing workspace view, including through RCQL."""
from __future__ import annotations

import pytest
from agent.server.scope import ScopedStore

from rcdb import MemoryStore, copy_record
from rcql import Executor, SourcePolicy, bind, classify, mutation, parse, source
from rexgraph.graph import RexGraph


def rex(n=2):
    return RexGraph.from_graph(list(range(n)), list(range(1, n + 1)))


@pytest.fixture
def stores(monkeypatch):
    inner = MemoryStore().configure_security(require_commits=True)
    inner.commit_mutation("a", rex(), meta={"workspace": "alpha"}, analytics=False)
    inner.commit_mutation("b", rex(3), meta={"workspace": "beta"}, analytics=False)
    scoped = ScopedStore(inner, "beta", "bob")
    monkeypatch.setattr(scoped, "_record", lambda *args, **kw: None)
    return inner, scoped


def test_snapshot_history_hash_and_stats_only_observe_the_workspace(stores, monkeypatch):
    inner, view = stores
    original = inner.get_version
    def guarded(id, version):
        if id == "a":
            pytest.fail("hidden workspace payload was decoded")
        return original(id, version)
    monkeypatch.setattr(inner, "get_version", guarded)
    assert view.read_record("a") is None
    assert view.get_version("a", 1) is None
    assert view.history("a") == view.commit_history("a") == []
    assert not view.verify_commits("a")
    assert view.read_record("b").value.nE == 3
    assert [r["id"] for r in view.state_manifest()["records"]] == ["b"]
    assert view.stats()["count"] == 1
    assert classify(view).value == "RCDBStore"
    binding = bind("db", view, SourcePolicy.allow("*"))
    assert {"read_record", "history", "commit_mutation", "state_digest"} <= binding.schema.surface
    result = Executor(sources={"db": view}).execute(parse('FROM $db RETURN RCDB_STATE_HASH()'))
    assert result.values == (view.state_digest(),)


def test_mutation_targets_are_literal_even_when_a_read_alias_exists(stores):
    inner, view = stores
    assert inner.get_record("a@1").id == "a"
    created = view.commit_mutation("a@1", rex(), expected_version=0, analytics=False)
    assert created.id == "a@1"
    assert created.meta["workspace"] == "beta"
    assert inner.get_record("a").meta["workspace"] == "alpha"


def test_scoped_mutation_stamps_and_checks_ownership(stores):
    inner, view = stores
    ex = Executor(sources={"db": view})
    with pytest.raises(PermissionError, match="another workspace"):
        ex.execute(mutation(source("db"), "a", rex(4)))
    result = ex.execute(mutation(source("db"), "b", rex(4), actor="not-the-caller", expected_version=1))
    assert result.values[0].meta == {"workspace": "beta", "stored_by": "bob"}
    assert inner.commit_history("b")[-1].transition.actor == "bob"
    assert inner.verify_commits("b")
    with pytest.raises(KeyError, match="not present"):
        ex.execute(parse('FROM RCDB_GET($db, "a") RETURN BETTI(0)'))


def test_governed_copy_cannot_escape_the_source_or_destination_view(stores):
    inner, view = stores
    destination = MemoryStore().configure_security(require_commits=True)
    assert copy_record(view, destination, inner.get_record("a")) is None
    copied = copy_record(view, destination, view.get_record("b"))
    assert copied.id == "b" and destination.verify_commits("b")
    alien = MemoryStore()
    record = alien.put("a", rex(5), analytics=False)
    with pytest.raises(PermissionError, match="another workspace"):
        copy_record(alien, view, record)


def test_mixed_workspace_lineage_does_not_return_hidden_predecessor_payloads(stores):
    inner, view = stores
    # An operator outside a request can explicitly reassign metadata. The scoped
    # view still must not return the previous workspace's raw mutation package.
    inner.commit_mutation("a", rex(4), meta={"workspace": "beta"}, analytics=False)
    assert view.read_record("a").record.version == 2
    assert view.read_record("a", version=1) is None
    with pytest.raises(PermissionError, match="lineage crosses"):
        view.commit_history("a")
    assert not view.verify_commits("a")
