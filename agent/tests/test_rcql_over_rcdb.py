"""Querying the database through the query language, with the policy enforced.

This is what the two halves were for. RCQL never imports a store; it evaluates against a
source the agent binds, so the same store can be exposed to one caller as records without
identity and to another as history it may name. The enforcement is the executor's, not the
caller's discipline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

if "rcql" not in sys.modules:
    try:
        import rcql as _probe
        if not hasattr(_probe, "parse"):
            raise ImportError("namespace shadow")
    except ImportError:
        root = Path(__file__).resolve().parents[2] / "rcql"
        if root.is_dir():
            sys.modules.pop("rcql", None)
            sys.path.insert(0, str(root))

pytest.importorskip("rcql")
pytest.importorskip("rcql.parser")


def _rex(n=2):
    from rexgraph.graph import RexGraph
    ptr = [0] + [2 * (i + 1) for i in range(n)]
    idx = []
    for i in range(n):
        idx += [i, (i + 1) % (n + 1)]
    return RexGraph.from_hypergraph(ptr, idx)


@pytest.fixture
def bound():
    """A store with two records, and a runtime to bind it into."""
    from agent.rcdb import MemoryStore
    from agent.rcql_runtime import RCQLRuntime
    store = MemoryStore()
    store.put("alpha", _rex(2), meta={"vertex_labels": ["oncology", "shared"]})
    store.put("beta", _rex(3), meta={"vertex_labels": ["cardiology", "shared"]})
    return store, RCQLRuntime()


def _run(runtime, text, **params):
    from rcql import parse
    return runtime.execute(parse(text), params=params or None)


def test_the_language_can_list_the_database(bound):
    store, runtime = bound
    runtime.register("db", store)
    rows = _run(runtime, "FROM $db RETURN RCDB_LIST()", db=store).values[0]
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {"alpha", "beta"}


def test_a_records_policy_hands_back_no_identity(bound):
    """The property the audit found untested: records without identity."""
    from rcql import SourcePolicy
    store, runtime = bound
    runtime.register("db", store, policy=SourcePolicy.allow(
        "records", record_fields={"nV", "nE"}))
    rows = _run(runtime, "FROM $db RETURN RCDB_LIST()", db=store).values[0]
    assert rows, "the query returned nothing"
    for row in rows:
        assert "id" not in row, row
        assert set(row["signature"]) <= {"nV", "nE"}, row["signature"]


def test_history_without_identity_is_refused(bound):
    """Granting history must not grant the ability to name a record."""
    from rcql import SourcePolicy
    store, runtime = bound
    runtime.register("db", store, policy=SourcePolicy.allow("history"))
    with pytest.raises(PermissionError):
        _run(runtime, 'FROM $db RETURN RCDB_HISTORY("alpha")', db=store)


def test_history_with_identity_is_allowed(bound):
    from rcql import SourcePolicy
    store, runtime = bound
    runtime.register("db", store, policy=SourcePolicy.allow("history", "identity"))
    rows = _run(runtime, 'FROM $db RETURN RCDB_HISTORY("alpha")', db=store).values[0]
    assert len(rows) == 1


def test_the_language_searches_the_vocabulary(bound):
    store, runtime = bound
    runtime.register("db", store)
    rows = _run(runtime, 'FROM $db RETURN RCDB_SEARCH("oncology")', db=store).values[0]
    assert [r["id"] for r in rows] == ["alpha"]


def test_it_searches_a_protected_vocabulary_too(tmp_path):
    """The store holds tokens rather than terms, and the query still resolves."""
    pytest.importorskip("safetensors")
    from agent.rcdb_protected_index import IndexPolicy, StaticIndexKeyProvider
    from agent.rcql_runtime import RCQLRuntime
    from agent.rexstore import RexStore

    keys = StaticIndexKeyProvider({"search": b"s" * 32})
    policy = IndexPolicy({"vertex_labels": "keyed"}, "search")
    store = RexStore(str(tmp_path / "s"), search_policy=policy, search_keys=keys)
    store.put("alpha", _rex(2), meta={"vertex_labels": ["oncology"]})
    store.put("beta", _rex(3), meta={"vertex_labels": ["cardiology"]})
    runtime = RCQLRuntime()
    runtime.register("db", store)
    rows = _run(runtime, 'FROM $db RETURN RCDB_SEARCH("oncology")', db=store).values[0]
    assert [r["id"] for r in rows] == ["alpha"]


def test_the_store_configuration_needs_admin(bound):
    from rcql import SourcePolicy
    store, runtime = bound
    runtime.register("db", store, policy=SourcePolicy.allow("records"))
    with pytest.raises(PermissionError):
        _run(runtime, "FROM $db RETURN RCDB_SECURITY()", db=store)


def test_the_language_can_read_the_commit_chain(tmp_path):
    """The query language over the integrity work: ask a store whether its own history
    is the one its commits attest to."""
    pytest.importorskip("cryptography")
    from agent.rcdb import MemoryStore
    from agent.rcql_runtime import RCQLRuntime
    store = MemoryStore()
    store.commit_mutation("alpha", _rex(2), actor="alice")
    store.commit_mutation("alpha", _rex(3), actor="alice")
    runtime = RCQLRuntime()
    runtime.register("db", store)
    verified = _run(runtime, 'FROM $db RETURN RCDB_VERIFY("alpha")', db=store)
    assert verified.values[0] is True
    commits = _run(runtime, 'FROM $db RETURN RCDB_COMMITS("alpha")', db=store).values[0]
    assert len(commits) == 2
