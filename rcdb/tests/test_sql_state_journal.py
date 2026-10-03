"""SQL stores the existing checked engine grammar within one owned transaction."""
from contextlib import closing
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import hashlib

import pytest

from rcdb import BlobCodecSpec, ComplexRecord, SQLStore, StoreHeader, VersionConflictError
from rcdb.engine import bind_record
from rcdb.sql_journal import SQLStateJournal
from rcdb.store_identity import StoreIdentity
from rexgraph import RexGraph
from rexgraph.io.safetensors_bridge import state_to_safetensors_bytes
from rexgraph.state import to_state


@pytest.fixture
def store(tmp_path):
    pytest.importorskip("sqlalchemy")
    with closing(SQLStore(f"sqlite:///{tmp_path / 'engine.sqlite'}")) as handle:
        yield handle


def header(store, codec="none"):
    return StoreHeader(StoreIdentity(store.store_id, "sql"), BlobCodecSpec(codec))


def proposal(journal, state, *, record_id="literal/id@1", now=10.0, n=2):
    source = RexGraph.from_graph(list(range(n-1)), list(range(1, n)))
    payload = journal.header.compression.encode(state_to_safetensors_bytes(to_state(source)))
    record = ComplexRecord(record_id, {"nV": n, "nE": n-1}, created=now,
                           version=state.next_version(record_id), tx_from=now, valid_from=now,
                           meta={"exact": Fraction(1, 7), "integer": 2**90})
    record, blob = bind_record(record, payload, store_id=journal.header.identity.id,
                               compression=journal.header.compression)
    return state.prepare_put(record, blob_digest=hashlib.sha256(blob).hexdigest())


def initialize(store, *, prefix="records", codec="none"):
    requested = header(store, codec)
    with store._sql_transaction(write=True) as conn:
        return SQLStateJournal.open(conn, prefix, header=requested)


@pytest.mark.parametrize("codec", ["none", "zlib"])
def test_sql_checked_history_replays_native_tombstones_highwater_and_cursors(store, codec):
    journal = initialize(store, prefix='record "history"', codec=codec)
    with store._sql_transaction(write=True) as conn:
        state = journal.load_state(conn)
        start = state.cursor
        state = journal.publish(conn, proposal(journal, state))
        first = state.cursor
        state = journal.publish(conn, proposal(journal, state, now=20.0, n=3))
        state = journal.publish(conn, state.prepare_delete("literal/id@1", 30.0))
        assert state.current("literal/id@1") is None
        assert state.next_version("literal/id@1") == 3
        state = journal.publish(conn, proposal(journal, state, now=40.0))
    with closing(SQLStore(store.conn_str)) as reopened:
        with reopened.read_transaction():
            opened = SQLStateJournal.open(reopened._sql_connection, 'record "history"')
            replay = opened.load_state(reopened._sql_connection, after=first)
            assert replay.cursor == state.cursor
            assert replay.current("literal/id@1").version == 3
            assert [r.version for r in replay.history("literal/id@1")] == [1, 2, 3]
            assert [r.tx_to for r in replay.history("literal/id@1")] == [20.0, 30.0, None]
            assert [f.operation for f in replay.changes(start)] == ["put", "put", "delete", "put"]
            assert replay.current("literal/id@1").meta == {"exact": Fraction(1, 7), "integer": 2**90}
            assert opened.header == journal.header


def test_sql_journal_creation_belongs_to_outer_transaction(store):
    from sqlalchemy import inspect
    requested = header(store)
    with pytest.raises(OSError, match="outer failure"):
        with store._sql_transaction(write=True) as conn:
            SQLStateJournal.open(conn, "rolled_back", header=requested)
            raise OSError("outer failure")
    assert not any(name.startswith("rolled_back_engine_") for name in inspect(store.engine).get_table_names())


def test_sql_journal_publication_rolls_back_with_its_outer_transaction(store):
    journal = initialize(store)
    with store._sql_transaction(write=True) as conn:
        committed = journal.publish(conn, proposal(journal, journal.load_state(conn)))
    with pytest.raises(OSError, match="outer failure"):
        with store._sql_transaction(write=True) as conn:
            provisional = journal.publish(conn, proposal(journal, journal.load_state(conn), now=20.0))
            assert provisional.cursor.sequence == 2
            raise OSError("outer failure")
    with store.read_transaction():
        assert journal.load_state(store._sql_connection).cursor == committed.cursor


@pytest.mark.parametrize("kind", ["header", "frame", "address", "hole", "head", "head-extra", "header-extra", "lost-header", "integer-blob"])
def test_sql_checked_history_refuses_corruption(store, kind):
    from sqlalchemy import delete, update
    journal = initialize(store)
    with store._sql_transaction(write=True) as conn:
        state = journal.publish(conn, proposal(journal, journal.load_state(conn)))
        journal.publish(conn, proposal(journal, state, now=20.0))
    with store.engine.begin() as conn:
        if kind == "header":
            conn.execute(update(journal.header_table).values(payload=header(store, "zlib").to_bytes()))
        elif kind == "frame":
            conn.execute(update(journal.frames_table).where(journal.frames_table.c.sequence == 1).values(payload=b"bad"))
        elif kind == "address":
            conn.execute(update(journal.frames_table).where(journal.frames_table.c.sequence == 2).values(sequence=3))
        elif kind == "hole":
            conn.execute(delete(journal.frames_table).where(journal.frames_table.c.sequence == 1))
        elif kind == "head":
            conn.execute(update(journal.head_table).values(sequence=1))
        elif kind == "head-extra":
            conn.execute(journal.head_table.insert().values(key="extra", sequence=0, digest="0"*64, header_digest=journal.header.digest))
        elif kind == "header-extra":
            conn.execute(journal.header_table.insert().values(key="extra", payload=journal.header.to_bytes()))
        elif kind == "lost-header":
            conn.execute(delete(journal.header_table))
        else:
            # SQLite BLOB columns can hold integers; bytes(integer) must never
            # turn an invalid persisted value into an attacker sized allocation.
            conn.exec_driver_sql("UPDATE records_engine_frames SET payload = 1000000000 WHERE sequence = 1")
    with store.read_transaction():
        with pytest.raises(ValueError):
            journal.load_state(store._sql_connection)


def test_sql_journal_refuses_partial_schema_and_profile_replacement(store):
    requested = header(store)
    partial = SQLStateJournal("partial", requested)
    with store._sql_transaction(write=True) as conn:
        partial.header_table.create(conn)
        with pytest.raises(ValueError, match="schema is incomplete"):
            SQLStateJournal.open(conn, "partial", header=requested)
    journal = initialize(store)
    with store.read_transaction():
        with pytest.raises(ValueError, match="explicit migration"):
            SQLStateJournal.open(store._sql_connection, "records", header=header(store, "zlib"))
        assert journal.load_state(store._sql_connection).cursor.sequence == 0


def test_sql_journal_requires_an_actual_owned_sqlite_transaction(store):
    requested = header(store)
    with store.engine.connect() as conn:
        with pytest.raises(RuntimeError, match="owned transaction"):
            SQLStateJournal.open(conn, "records", header=requested)
        conn.exec_driver_sql("SELECT 1")  # SQLAlchemy autobegin alone does not pin SQLite.
        with pytest.raises(RuntimeError, match="explicit SQLite transaction"):
            SQLStateJournal.open(conn, "records", header=requested)


def test_sql_journal_refuses_a_stale_proposal_without_changing_the_head(store):
    journal = initialize(store)
    with store._sql_transaction(write=True) as conn:
        initial = journal.load_state(conn)
        frame = proposal(journal, initial)
        published = journal.publish(conn, frame)
        with pytest.raises(VersionConflictError, match="published head"):
            journal.publish(conn, frame)
        assert journal.load_state(conn).cursor == published.cursor


def test_sql_journal_savepoint_removes_frame_when_head_publication_fails(store):
    from sqlalchemy import event
    journal = initialize(store)
    def failed(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE records_engine_head"):
            raise OSError("head unavailable")
    with store._sql_transaction(write=True) as conn:
        frame = proposal(journal, journal.load_state(conn))
        event.listen(store.engine, "before_cursor_execute", failed)
        try:
            with pytest.raises(OSError, match="head unavailable"):
                journal.publish(conn, frame)
        finally:
            event.remove(store.engine, "before_cursor_execute", failed)
        assert journal.load_state(conn).cursor.sequence == 0
    with store.read_transaction():
        assert journal.load_state(store._sql_connection).cursor.sequence == 0


def test_sql_provider_constraint_failure_is_not_reported_as_a_version_conflict(store):
    from sqlalchemy.exc import IntegrityError
    journal = initialize(store)
    with store._sql_transaction(write=True) as conn:
        conn.exec_driver_sql("CREATE TRIGGER refuse_engine_frame BEFORE INSERT ON records_engine_frames "
                             "BEGIN SELECT RAISE(ABORT, 'provider constraint refusal'); END")
        frame = proposal(journal, journal.load_state(conn))
        with pytest.raises(IntegrityError, match="provider constraint refusal"):
            journal.publish(conn, frame)
        assert journal.load_state(conn).cursor.sequence == 0


def test_sql_journal_rejects_legacy_frames_without_fabricating_engine_history(store):
    from dataclasses import replace
    journal = initialize(store)
    with store._sql_transaction(write=True) as conn:
        state = journal.load_state(conn)
        frame = replace(proposal(journal, state), mutation=None)
        with pytest.raises(ValueError, match="engine configuration"):
            journal.publish(conn, frame)
        assert journal.load_state(conn).cursor.sequence == 0


def test_sql_journal_stale_cursors_cannot_hide_truncated_history(store):
    from sqlalchemy import delete, update
    journal = initialize(store)
    with store._sql_transaction(write=True) as conn:
        first = journal.publish(conn, proposal(journal, journal.load_state(conn)))
        second = journal.publish(conn, proposal(journal, first, now=20.0))
    with store.engine.begin() as conn:
        conn.execute(delete(journal.frames_table).where(journal.frames_table.c.sequence == 2))
        conn.execute(update(journal.head_table).values(sequence=1, digest=first.cursor.digest))
    with store.read_transaction():
        with pytest.raises(ValueError, match="published history"):
            journal.load_state(store._sql_connection, after=second.cursor)


def test_sql_journal_independent_handles_arbitrate_prepared_proposals(store):
    journal = initialize(store)
    requested = journal.header
    barrier = Barrier(2)
    handles = [SQLStore(store.conn_str), SQLStore(store.conn_str)]
    def publish(handle):
        with handle.read_transaction():
            opened = SQLStateJournal.open(handle._sql_connection, "records", header=requested)
            frame = proposal(opened, opened.load_state(handle._sql_connection))
        barrier.wait(timeout=15)
        try:
            with handle._sql_transaction(write=True) as conn:
                return opened.publish(conn, frame).cursor.sequence
        except VersionConflictError:
            return "conflict"
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(publish, handles))
        assert sorted(map(str, outcomes)) == ["1", "conflict"]
        with store.read_transaction():
            assert journal.load_state(store._sql_connection).cursor.sequence == 1
    finally:
        for handle in handles:
            handle.close()


def test_sql_journal_incremental_reader_checks_anchor_and_reads_only_new_frames(store, monkeypatch):
    from rcdb.journal import JournalFrame
    journal = initialize(store)
    with store._sql_transaction(write=True) as conn:
        first = journal.publish(conn, proposal(journal, journal.load_state(conn)))
        second = journal.publish(conn, proposal(journal, first, now=20.0))
        third = journal.publish(conn, proposal(journal, second, now=30.0))
    parsed = []
    original = JournalFrame.from_bytes
    def counted(payload):
        frame = original(payload)
        parsed.append(frame.sequence)
        return frame
    monkeypatch.setattr(JournalFrame, "from_bytes", counted)
    with store.read_transaction():
        suffix = journal.read_after(store._sql_connection, second.cursor)
        assert [frame.sequence for frame in suffix] == [3]
        assert parsed == [2, 3]
        second.apply(suffix[0])
        assert second.cursor == third.cursor
        parsed.clear()
        assert journal.read_after(store._sql_connection, third.cursor) == ()
        assert parsed == [3]


def test_sql_journal_checked_cached_publication_does_not_replay_the_history(store, monkeypatch):
    from rcdb.journal import JournalFrame
    journal = initialize(store)
    with store._sql_transaction(write=True) as conn:
        state = journal.load_state(conn)
        for tick in range(1, 7):
            state = journal.publish(conn, proposal(journal, state, now=float(tick)), state=state)
        parsed = []
        original = JournalFrame.from_bytes
        def counted(payload):
            frame = original(payload)
            parsed.append(frame.sequence)
            return frame
        monkeypatch.setattr(JournalFrame, "from_bytes", counted)
        published = journal.publish(conn, proposal(journal, state, now=7.0), state=state)
        assert published.cursor.sequence == 7
        assert parsed == [6]
    with store.read_transaction():
        assert journal.load_state(store._sql_connection).cursor == published.cursor


def test_sql_journal_cached_publication_refuses_an_outdated_state_without_changing_it(store):
    journal = initialize(store)
    with store._sql_transaction(write=True) as conn:
        prior = journal.load_state(conn)
        proposed = proposal(journal, prior)
        published = journal.publish(conn, proposed)
        with pytest.raises(VersionConflictError, match="precedes the published head"):
            journal.publish(conn, proposed, state=prior)
        assert prior.cursor.sequence == 0
        assert journal.load_state(conn).cursor == published.cursor


@pytest.mark.parametrize("key", ["identity", "engine_header"])
def test_open_sql_journal_refuses_loss_of_its_namespace_ownership_claim(store, key):
    from sqlalchemy import delete
    from rcdb.store_identity import sql_identity_table
    journal = initialize(store)
    identity_table = sql_identity_table("records")
    with store.engine.begin() as conn:
        conn.execute(delete(identity_table).where(identity_table.c.key == key))
    with store.read_transaction():
        with pytest.raises(ValueError, match="missing"):
            journal.load_state(store._sql_connection)


@pytest.mark.parametrize("kind", ["owner", "profile", "digest", "future", "lost-anchor", "lost-suffix"])
def test_sql_journal_incremental_reader_refuses_an_invalid_retained_cursor(store, kind):
    from dataclasses import replace
    from sqlalchemy import delete
    journal = initialize(store)
    with store._sql_transaction(write=True) as conn:
        first = journal.publish(conn, proposal(journal, journal.load_state(conn)))
        journal.publish(conn, proposal(journal, first, now=20.0))
    cursor = first.cursor
    if kind in {"lost-anchor", "lost-suffix"}:
        sequence = 1 if kind == "lost-anchor" else 2
        with store.engine.begin() as conn:
            conn.execute(delete(journal.frames_table).where(journal.frames_table.c.sequence == sequence))
    else:
        cursor = replace(cursor, **{"owner": {"store_id": "2"*32}, "profile": {"header_digest": "2"*64},
                                    "digest": {"digest": "2"*64}, "future": {"sequence": 3}}[kind])
    with store.read_transaction():
        with pytest.raises(ValueError):
            journal.read_after(store._sql_connection, cursor)
