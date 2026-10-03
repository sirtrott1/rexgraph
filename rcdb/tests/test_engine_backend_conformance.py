"""The same record state engine obligations across publication providers."""
from contextlib import closing
from dataclasses import replace
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from rcdb import BlobCodecSpec, LocalStore, MemoryStore, SQLStore, VersionConflictError
from rexgraph import RexGraph
from rexgraph.object_identity import object_digest


@pytest.fixture(params=["memory", "local", "sql"])
def store(request, tmp_path):
    if request.param == "sql":
        pytest.importorskip("sqlalchemy")
        instance = SQLStore(f"sqlite:///{tmp_path / 'state.sqlite'}")
    else:
        instance = MemoryStore() if request.param == "memory" else LocalStore(tmp_path / "local")
    with closing(instance):
        yield instance


def graph(n=2):
    return RexGraph.from_graph(list(range(n-1)), list(range(1, n)))


def test_put_delete_recreate_time_travel_and_cursor_resume_are_one_contract(store):
    start = store.change_cursor
    store.put("literal/id@1", graph(), analytics=False, _tx_time=10.0)
    first = store.change_cursor
    store.put("literal/id@1", graph(3), analytics=False, _tx_time=20.0)
    assert store.delete("literal/id@1", tx_time=30.0, expected_version=2)
    assert store.get("literal/id@1") is None
    assert store.next_version("literal/id@1") == 3
    assert [r.version for r in store.history("literal/id@1")] == [1, 2]
    assert store.tombstone("literal/id@1").version == 2
    assert store.get("literal/id@1", as_of=25.0).nE == 2
    assert store.get("literal/id@1", as_of=30.0) is None
    assert store.put("literal/id@1", graph(), analytics=False, _tx_time=40.0).version == 3
    assert [f.operation for f in store.changes(start)] == ["put", "put", "delete", "put"]
    assert [f.sequence for f in store.changes(first)] == [2, 3, 4]
    assert store.cursor_for(store.changes()[-1]) == store.change_cursor
    assert object_digest(store.get_version("literal/id@1", 1)) == object_digest(graph())


def test_change_cursors_are_bound_to_store_header_and_published_predecessor(store):
    store.put("r", graph(), analytics=False)
    cursor = store.change_cursor
    for bad in (replace(cursor, store_id="2"*32), replace(cursor, header_digest="2"*64),
                replace(cursor, digest="2"*64), replace(cursor, sequence=cursor.sequence+1)):
        with pytest.raises(ValueError): store.changes(bad)
    with pytest.raises((TypeError, ValueError)): store.changes(123)


def test_refusal_preserves_visible_state_version_and_cursor(store, monkeypatch):
    store.put("r", graph(), analytics=False, _tx_time=10.0)
    previous = store.change_cursor
    def fail(value):
        raise ValueError("injected")
    with monkeypatch.context() as patch:
        patch.setattr(store, "_serialize_payload", fail)
        with pytest.raises(ValueError, match="injected"):
            store.put("r", graph(3), analytics=False, _tx_time=20.0)
    assert store.change_cursor == previous and store.get_record("r").tx_to is None
    assert store.next_version("r") == 2 and store.get("r").nE == 1


def test_clock_rewind_refuses_before_live_interval_is_changed(store):
    store.put("r", graph(), analytics=False, _tx_time=20.0)
    saved = store.change_cursor
    with pytest.raises(ValueError, match="precedes"):
        store.put("r", graph(3), analytics=False, _tx_time=10.0)
    with pytest.raises(ValueError, match="precedes"):
        store.delete("r", tx_time=10.0)
    assert store.change_cursor == saved and store.get_record("r").tx_to is None


def test_exact_metadata_and_corpus_snapshot_remain_owned_across_tombstones(store):
    source = {"exact": Fraction(1, 7), "positions": (), "integer": 2**90}
    store.put("r", graph(), meta=source, tags=["old"], analytics=False)
    snap = store.corpus_snapshot(signature_fields=["tags"])
    row = store.read_record("r")
    row.record.meta["exact"] = 0
    assert store.get_record("r").meta == source
    assert store.delete("r")
    assert store.corpus_snapshot().ids == ()
    assert snap.ids == ("r",) and snap.response(["old"])["scores"] == (Fraction(1),)
    assert store.get_version("r", 1) is not None


def test_optional_commits_do_not_inherit_a_deleted_incarnation_parent(store):
    store.commit_mutation("r", graph(), expected_version=0, tx_time=10.0, analytics=False)
    store.commit_mutation("r", graph(3), expected_version=1, tx_time=20.0, analytics=False)
    assert store.verify_commits("r")
    store.delete("r", tx_time=30.0)
    assert store.commit_history("r") == []
    store.put("r", graph(), analytics=False, _tx_time=40.0)
    result = store.commit_mutation("r", graph(3), expected_version=3, tx_time=50.0, analytics=False)
    assert result.version == 4 and len(store.commit_history("r")) == 1
    assert store.verify_commits("r")
    with pytest.raises(VersionConflictError):
        store.commit_mutation("r", graph(), expected_version=2, tx_time=60.0, analytics=False)


@pytest.mark.parametrize("name", ["none", "zlib"])
def test_explicit_backend_codec_does_not_consult_optional_installs(store, name, tmp_path, monkeypatch):
    from rcdb import core
    if store.backend == "memory":
        constructor = MemoryStore
    elif store.backend == "sql":
        constructor = lambda **kw: SQLStore(f"sqlite:///{tmp_path / (name+'.sqlite')}", **kw)
    else:
        constructor = lambda **kw: LocalStore(tmp_path / name, **kw)
    monkeypatch.setattr(core, "_codec", lambda: pytest.fail("ambient codec consulted"))
    with closing(constructor(compression=BlobCodecSpec(name))) as declared:
        declared.put("r", graph(), analytics=False)
        assert declared.blob_codec == BlobCodecSpec(name)
        assert object_digest(declared.get("r")) == object_digest(graph())


def test_same_handle_concurrent_expected_versions_publish_only_one_successor(store):
    store.commit_mutation("r", graph(), analytics=False, expected_version=0)
    barrier = Barrier(4)
    def commit(_):
        barrier.wait(timeout=10)
        try:
            return store.commit_mutation("r", graph(3), analytics=False, expected_version=1).version
        except VersionConflictError:
            return "conflict"
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(commit, range(4)))
    assert outcomes.count(2) == 1 and outcomes.count("conflict") == 3
    assert store.change_cursor.sequence == 2 and store.verify_commits("r")


def test_equal_tick_collection_order_and_version_choice_match_across_providers(store):
    for rid in ("z", "a", "z"):
        store.put(rid, graph(), analytics=False, _tx_time=10.0)
    assert [(row.id, row.version) for row in store.list()] == [("a", 1), ("z", 2)]
    assert [(row.id, row.version) for row in store.list(include_history=True)] == [("a", 1), ("z", 2), ("z", 1)]
    with pytest.raises(ValueError): store.list(include_history=True, as_of=10.0)


def test_closed_handles_refuse_native_operations_and_close_is_idempotent(store):
    store.put("r", graph(), analytics=False)
    store.corpus_snapshot()  # Populate an optional derived cache before closing.
    store.close()
    store.close()
    for action in (lambda: store.get("r"), lambda: store.get_record("r"), lambda: store.history("r"),
                   lambda: store.put("r", graph()), lambda: store.delete("r"),
                   lambda: store.changes(), lambda: store.next_version("r"), lambda: store.corpus_snapshot()):
        with pytest.raises((ValueError, RuntimeError), match="closed"): action()


def test_claimed_mutation_artifact_is_required_even_when_policy_allows_ordinary_put(store):
    store.commit_mutation("r", graph(), analytics=False)
    if store.backend == "memory":
        store._commit_blobs.pop(("r", 1))
    elif store.backend == "sql":
        from sqlalchemy import delete
        with store.engine.begin() as conn:
            conn.execute(delete(store.commits_table))
    else:
        store._commit_path("r", 1).unlink()
    with pytest.raises(ValueError, match="artifact is missing"): store.get("r")
    with pytest.raises(ValueError, match="artifact is missing"): store.commit_history("r")
    assert not store.verify_commits("r")


def test_published_artifact_cannot_be_rebound_or_removed_through_internal_writer(store):
    store.commit_mutation("r", graph(), analytics=False)
    raw = store._load_commit_bytes("r", 1)
    with pytest.raises(ValueError, match="published"):
        store._store_commit_bytes("r", 1, raw)
    with pytest.raises(ValueError, match="published"):
        store._delete_commit_bytes("r", 1)
    assert store.verify_commits("r")


def test_store_header_is_read_only_on_a_live_engine_handle(store):
    from rcdb import StoreHeader
    original = store.header
    with pytest.raises(AttributeError):
        store.header = StoreHeader(original.identity, BlobCodecSpec("zlib"))
    assert store.header is original
    store.put("r", graph(), analytics=False)
    assert store.get("r").nE == 1


@pytest.mark.parametrize("version", [None, 0, -1, True, 1.5])
def test_native_exact_address_refuses_invalid_versions(store, version):
    store.put("r", graph(), analytics=False)
    with pytest.raises(ValueError, match="positive integer"):
        store.get_version("r", version)


def test_time_selection_precedes_query_predicates_and_history_is_explicit(store):
    store.put("r", graph(), meta={"vertex_labels": ["old"]}, analytics=False,
              valid_from=1.0, valid_to=100.0, _tx_time=10.0)
    store.put("r", graph(3), meta={"vertex_labels": ["new"]}, analytics=False,
              valid_from=1.0, valid_to=100.0, _tx_time=20.0)
    assert store.query(valid_at=5.0, labels_any=["old"]) == []
    assert store.query(valid_at=5.0, max_nE=1) == []
    assert [(r.id, r.version) for r in store.query(valid_at=5.0, min_nE=2)] == [("r", 2)]
    assert [(r.id, r.version) for r in store.query(as_of=10.0, labels_any=["old"])] == [("r", 1)]
    assert store.query(as_of=10.0, min_nE=2) == []
    assert [(r.id, r.version) for r in store.query(include_history=True, labels_any=["old"])] == [("r", 1)]
    with pytest.raises(ValueError, match="history collection"):
        store.query(include_history=True, valid_at=5.0)


def test_display_alias_cannot_bypass_time_selector_but_literal_id_keeps_primacy(store):
    store.put("r", graph(), analytics=False, _tx_time=10.0)
    assert store.get("r@1").nE == 1
    with pytest.raises(ValueError, match="display aliases"):
        store.get("r@1", as_of=10.0)
    store.put("r@1", graph(3), analytics=False, _tx_time=20.0)
    assert store.get("r@1", as_of=10.0) is None
    assert store.get("r@1", as_of=20.0).nE == 2
