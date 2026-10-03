"""Independent SQL handles must arbitrate a whole mutation, not only its insert."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from contextlib import closing
import sys
import subprocess
import time

import pytest

from rcdb import SQLStore, VersionConflictError, PublicationUncertainError
from rexgraph import RexGraph


def graph(n=2):
    return RexGraph.from_graph(list(range(n-1)), list(range(1, n)))


def test_independent_sql_handles_cannot_stage_the_same_expected_version(tmp_path):
    pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'concurrent.sqlite'}"
    with closing(SQLStore(uri)) as owner:
        owner.commit_mutation("r", graph(), expected_version=0, analytics=False)
        handles = [SQLStore(uri), SQLStore(uri)]
        barrier = Barrier(2)
        def commit(handle):
            barrier.wait(timeout=15)
            try:
                return handle.commit_mutation("r", graph(3), expected_version=1, analytics=False).version
            except VersionConflictError:
                return "conflict"
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(commit, handles))
            assert sorted(map(str, results)) == ["2", "conflict"]
            assert [r.version for r in owner.history("r")] == [1, 2]
            assert owner.verify_commits("r")
            assert len(owner.commit_history("r")) == 2
        finally:
            for handle in handles:
                handle.close()


def test_sql_commit_and_record_staging_share_one_transaction(tmp_path, monkeypatch):
    pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'atomic.sqlite'}"
    with closing(SQLStore(uri)) as owner, closing(SQLStore(uri)) as observer:
        original = owner._store_commit_bytes
        seen = []
        def staged(record_id, version, payload):
            original(record_id, version, payload)
            # An independent connection must never see the pending artifact
            # before its record, even during the documented staging interval.
            seen.append(observer._load_commit_bytes(record_id, version))
        monkeypatch.setattr(owner, "_store_commit_bytes", staged)
        record = owner.commit_mutation("r", graph(), expected_version=0, analytics=False)
        assert record.version == 1 and seen == [None]
        assert observer.get("r").nE == 1 and observer.verify_commits("r")


@pytest.fixture
def stores(tmp_path):
    pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'store.sqlite'}"
    with closing(SQLStore(uri)) as owner, closing(SQLStore(uri)) as observer:
        # First identity publication is separate from record transactions.
        assert owner.store_id == observer.store_id
        yield owner, observer


@pytest.mark.parametrize("mutation", [False, True])
def test_sql_failure_after_insert_rolls_back_every_projection(stores, monkeypatch, mutation):
    import rcdb.core as core
    owner, observer = stores
    owner.commit_mutation("r", graph(), meta={"vertex_labels": ["old"]}, analytics=False)
    original = owner._put_impl
    events = []
    monkeypatch.setattr(core, "_ACTIVITY_HOOK", lambda *args: events.append(args))
    def failed(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("failed after insert")
    monkeypatch.setattr(owner, "_put_impl", failed)
    put = owner.commit_mutation if mutation else owner.put
    with pytest.raises(OSError, match="failed after insert"):
        put("r", graph(3), meta={"vertex_labels": ["new"]}, analytics=False)
    assert events == []
    assert [r.version for r in observer.history("r")] == [1]
    assert observer.get_record("r").tx_to is None
    assert observer.get_version("r", 2) is None
    assert observer._load_commit_bytes("r", 2) is None
    assert observer.query(labels_any=["new"]) == []
    assert [r.id for r in observer.query(labels_any=["old"])] == ["r"]
    assert observer.verify_commits("r")
    monkeypatch.setattr(owner, "_put_impl", original)
    assert owner.commit_mutation("r", graph(3), analytics=False).version == 2
    assert observer.verify_commits("r")


def test_sql_delete_rolls_back_checked_history_and_projection_together(stores, monkeypatch):
    import rcdb.core as core
    owner, observer = stores
    owner.commit_mutation("r", graph(), meta={"vertex_labels": ["kept"]}, analytics=False)
    artifact = owner._load_commit_bytes("r", 1)
    events = []
    monkeypatch.setattr(core, "_ACTIVITY_HOOK", lambda *args: events.append(args))
    original = owner._native.journal.publish
    cursor = owner.change_cursor
    def failed(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("tombstone publication failed")
    monkeypatch.setattr(owner._native.journal, "publish", failed)
    with pytest.raises(OSError, match="tombstone publication failed"):
        owner.delete("r")
    assert observer.get("r").nE == 1
    assert [r.id for r in observer.query(labels_any=["kept"])] == ["r"]
    assert observer._load_commit_bytes("r", 1) == artifact
    assert observer.verify_commits("r")
    assert owner.change_cursor == observer.change_cursor == cursor
    assert owner.tombstone("r") is None
    assert events == []


def test_sql_change_hook_observes_only_the_committed_record(stores, monkeypatch):
    import rcdb.core as core
    owner, observer = stores
    seen = []
    def emitted(origin, action, details):
        seen.append((action, observer.get_version(details["id"], details["version"]).nE,
                     len(observer.commit_history(details["id"]))))
    monkeypatch.setattr(core, "_ACTIVITY_HOOK", emitted)
    owner.commit_mutation("r", graph(), analytics=False)
    assert seen == [("rcdb.put", 1, 1)]


def test_sql_published_mutation_artifacts_cannot_be_replaced_or_removed(stores):
    owner, observer = stores
    owner.commit_mutation("r", graph(), analytics=False)
    artifact = owner._load_commit_bytes("r", 1)
    with pytest.raises(ValueError, match="overwrite a published"):
        owner._store_commit_bytes("r", 1, artifact)
    with pytest.raises(ValueError, match="delete a published"):
        owner._delete_commit_bytes("r", 1)
    assert observer._load_commit_bytes("r", 1) == artifact
    assert observer.verify_commits("r")
    assert owner.delete("r")
    assert observer.get("r") is None
    assert observer._load_commit_bytes("r", 1) == artifact
    assert observer.get_version("r", 1).nE == 1
    assert observer.commit_history("r") == []
    assert observer.next_version("r") == 2


def test_sql_read_transaction_cannot_be_promoted_to_a_write(stores):
    owner, _ = stores
    with owner.read_transaction():
        with pytest.raises(RuntimeError, match="cannot write inside"):
            owner.put("r", graph(), analytics=False)
        assert owner.get("r") is None
    assert owner.put("r", graph(), analytics=False).version == 1


def test_sql_read_transaction_pins_records_artifacts_and_indexes_in_wal_mode(stores):
    owner, observer = stores
    with owner.engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA journal_mode=WAL").scalar_one() == "wal"
    owner.commit_mutation("r", graph(), meta={"vertex_labels": ["old"]}, analytics=False)
    with owner.read_transaction():
        assert owner.get("r").nE == 1
        observer.commit_mutation("r", graph(3), meta={"vertex_labels": ["new"]},
                                 analytics=False, expected_version=1)
        assert owner.get_record("r").version == 1
        assert owner.get("r").nE == 1
        assert owner.get_version("r", 2) is None
        assert owner._load_commit_bytes("r", 2) is None
        assert len(owner.commit_history("r")) == 1
        assert owner.query(labels_any=["new"]) == []
        assert [r.id for r in owner.query(labels_any=["old"])] == ["r"]
    assert owner.get_record("r").version == 2
    assert owner.get("r").nE == 2
    assert owner.verify_commits("r")


def test_rcql_selects_all_sources_from_one_sql_snapshot(stores, monkeypatch):
    rcql = pytest.importorskip("rcql")
    owner, observer = stores
    with owner.engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA journal_mode=WAL").scalar_one() == "wal"
    owner.put("a", graph(), analytics=False)
    owner.put("b", graph(), analytics=False)
    original = owner.read_record
    def read(record_id, **kwargs):
        result = original(record_id, **kwargs)
        if record_id == "a":
            observer.put("b", graph(3), analytics=False)
        return result
    monkeypatch.setattr(owner, "read_record", read)
    selected = rcql.SnapshotContext.select(owner, (rcql.SourceSelection("left", "a"),
                                                  rcql.SourceSelection("right", "b")))
    assert selected.sources["right"].ref.record_version == 1
    assert observer.get_record("b").version == 2
    result = rcql.Executor(sources=selected.sources, evidence=selected).execute(
        rcql.parse("FROM $right RETURN COUNT(CELLS(1))"))
    assert result.values == (1,)


def test_sql_memory_database_is_shared_by_threads_of_the_handle():
    pytest.importorskip("sqlalchemy")
    with closing(SQLStore("sqlite:///:memory:")) as owner:
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(lambda _: owner.put("r", graph(), analytics=False), range(8)))
        assert sorted(r.version for r in records) == list(range(1, 9))
        assert owner.get_record("r").version == 8
        assert owner.get("r").nE == 1


@pytest.mark.parametrize("operation", ["get", "get_record", "get_version", "history", "list", "query",
                                       "read_record", "state_manifest", "corpus_snapshot", "commit_history",
                                       "verify_commits", "next_version", "put", "put_prepared", "commit_mutation", "delete"])
def test_closed_sql_handle_refuses_record_and_snapshot_operations(stores, operation):
    owner, _ = stores
    owner.put("r", graph(), analytics=False)
    owner.corpus_snapshot()
    owner.close()
    args = {"get_version": ("r", 1), "put": ("r", graph()), "commit_mutation": ("r", graph()),
            "put_prepared": ("r", b"unused", {})}.get(operation,
        () if operation in {"list", "query", "state_manifest", "corpus_snapshot"} else ("r",))
    with pytest.raises(RuntimeError, match="SQLStore is closed"):
        getattr(owner, operation)(*args)
    owner.close()


@pytest.mark.parametrize("lost", [True, False])
def test_live_sql_handle_refuses_lost_or_replaced_owner(stores, lost):
    from sqlalchemy import delete, update
    from rcdb.store_identity import StoreIdentity
    owner, _ = stores
    owner.put("r", graph(), analytics=False)
    with owner.engine.begin() as conn:
        if lost:
            conn.execute(delete(owner.identity_table))
        else:
            conn.execute(update(owner.identity_table).values(value=StoreIdentity.create("sql").to_bytes()))
    with pytest.raises(ValueError, match="store identity"):
        owner.put("s", graph(), analytics=False)
    with owner.engine.connect() as conn:
        assert conn.execute(owner.table.select()).fetchall()[0].id == "r"
        assert len(conn.execute(owner.table.select()).fetchall()) == 1


def test_sql_lost_commit_acknowledgement_requires_reopen_and_keeps_complete_state(stores, monkeypatch):
    import rcdb.core as core
    from sqlalchemy.engine import Connection
    owner, observer = stores
    original = Connection.commit
    events = []
    monkeypatch.setattr(core, "_ACTIVITY_HOOK", lambda *args: events.append(args))
    def lost_ack(conn):
        original(conn)
        if conn is owner._sql_connection:
            raise OSError("commit acknowledgement lost")
    monkeypatch.setattr(Connection, "commit", lost_ack)
    with pytest.raises(PublicationUncertainError, match="commit outcome is uncertain"):
        owner.commit_mutation("r", graph(), analytics=False)
    assert owner._sql_connection is None
    with pytest.raises(PublicationUncertainError, match="reopen and verify"):
        owner.get("r")
    with pytest.raises(PublicationUncertainError, match="reopen and verify"):
        owner.put("r", graph(), analytics=False)
    assert observer.get("r").nE == 1
    assert observer.verify_commits("r")
    assert len(observer.commit_history("r")) == 1
    assert events == []


def test_sql_caught_nested_write_failure_cannot_commit_the_outer_transaction(stores, monkeypatch):
    owner, observer = stores
    original = owner._put_impl
    def failed(record_id, *args, **kwargs):
        record = original(record_id, *args, **kwargs)
        if record_id == "broken":
            raise OSError("nested failure")
        return record
    monkeypatch.setattr(owner, "_put_impl", failed)
    with pytest.raises(RuntimeError, match="aborted by a nested operation"):
        with owner._sql_transaction(write=True):
            owner.put("r", graph(), analytics=False)
            with pytest.raises(OSError, match="nested failure"):
                owner.put("broken", graph(), analytics=False)
    assert observer.list() == []
    assert owner.list() == []


def test_separate_sql_processes_arbitrate_the_whole_expected_version_mutation(stores, tmp_path):
    owner, _ = stores
    owner.commit_mutation("r", graph(), analytics=False)
    script = tmp_path / "writer.py"
    script.write_text('''import pathlib, sys, time
from rcdb import SQLStore, VersionConflictError
from rexgraph import RexGraph
uri, ready_path, start_path = sys.argv[1:]
store = SQLStore(uri)
try:
    pathlib.Path(ready_path).write_text("ready")
    deadline = time.monotonic()+15
    while not pathlib.Path(start_path).exists():
        if time.monotonic() > deadline:
            raise RuntimeError("writer start timed out")
        time.sleep(0.01)
    try:
        rec = store.commit_mutation("r", RexGraph.from_graph([0, 1], [1, 2]), expected_version=1, analytics=False)
        print(rec.version)
    except VersionConflictError:
        print("conflict")
finally:
    store.close()
''')
    start = tmp_path / "start"
    ready = [tmp_path / f"ready-{n}" for n in range(2)]
    processes = []
    try:
        for path in ready:
            processes.append(subprocess.Popen([sys.executable, "-I", str(script), owner.conn_str,
                                                str(path), str(start)], cwd=tmp_path,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        deadline = time.monotonic()+15
        while not all(p.exists() for p in ready):
            assert time.monotonic() < deadline, "SQL processes did not reach the start barrier"
            assert all(p.poll() is None for p in processes), "SQL process failed before mutation"
            time.sleep(0.01)
        start.write_text("start")
        results = []
        for process in processes:
            out, err = process.communicate(timeout=20)
            assert process.returncode == 0, err
            results.append(out.strip())
        assert sorted(results) == ["2", "conflict"]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
    assert [r.version for r in owner.history("r")] == [1, 2]
    assert len(owner.commit_history("r")) == 2
    assert owner.verify_commits("r")
