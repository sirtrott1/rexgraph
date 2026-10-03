"""A native SQL journal is a durable ownership claim even with no complex rows."""
from contextlib import closing

import pytest

from rcdb import SQLStore, StoreHeader
from rcdb.sql_journal import SQLStateJournal
from rcdb.store_identity import StoreIdentity


@pytest.mark.parametrize("partial", [False, True])
def test_sql_identity_discovery_cannot_replace_the_owner_of_a_native_journal(tmp_path, partial):
    sa = pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'native.sqlite'}"
    with closing(SQLStore(uri)) as store:
        identity = store.store_id
        declared = StoreHeader(StoreIdentity(identity, "sql"))
        with store._sql_transaction(write=True) as conn:
            journal = SQLStateJournal.open(conn, store.table.name, header=declared)
        with store.engine.begin() as conn:
            conn.execute(sa.delete(store.identity_table))
            if partial:
                journal.header_table.drop(conn)
    with pytest.raises(ValueError, match="identity is missing|schema is incomplete|anchor is missing"):
        SQLStore(uri)
    engine = sa.create_engine(uri)
    try:
        with engine.connect() as conn:
            assert conn.execute(sa.select(store.identity_table)).fetchall() == []
    finally:
        engine.dispose()


def test_sql_record_and_journal_owner_claims_cannot_disagree(tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    from rexgraph import RexGraph
    uri = f"sqlite:///{tmp_path / 'native.sqlite'}"
    with closing(SQLStore(uri)) as store:
        store.put("r", RexGraph.from_graph([0], [1]), analytics=False)
        proper = StoreHeader(StoreIdentity(store.store_id, "sql"))
        wrong = StoreHeader(StoreIdentity("2"*32, "sql"))
        with store._sql_transaction(write=True) as conn:
            journal = SQLStateJournal.open(conn, store.table.name, header=proper)
        with store.engine.begin() as conn:
            conn.execute(sa.update(journal.header_table).values(payload=wrong.to_bytes()))
            conn.execute(sa.update(store.identity_table).where(store.identity_table.c.key == "engine_header").values(value=wrong.to_bytes()))
        with pytest.raises(ValueError, match="ownership claims"):
            store.put("new", RexGraph.from_graph([0], [1]), analytics=False)
    with pytest.raises(ValueError, match="ownership claims"):
        SQLStore(uri)


@pytest.mark.parametrize("reopen", [False, True])
def test_sql_identity_binary_column_refuses_integer_values_before_conversion(tmp_path, reopen):
    pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'malformed.sqlite'}"
    with closing(SQLStore(uri)) as store:
        _ = store.store_id
        with store.engine.begin() as conn:
            # Small negative control: SQLite allows an integer in a BLOB column.
            # The same type guard must also refuse bytes(huge_integer) allocation.
            conn.exec_driver_sql("UPDATE rc_complexes_store_identity SET value = 64")
        if not reopen:
            with pytest.raises(ValueError, match="requires bounded binary bytes"):
                store.history("r")
    if reopen:
        with pytest.raises(ValueError, match="requires bounded binary bytes"):
            SQLStore(uri)


def test_sql_native_anchor_prevents_downgrade_after_loss_of_all_journal_tables(tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'lost.sqlite'}"
    with closing(SQLStore(uri)) as store:
        declared = StoreHeader(StoreIdentity(store.store_id, "sql"))
        with store._sql_transaction(write=True) as conn:
            journal = SQLStateJournal.open(conn, store.table.name, header=declared)
        with store.engine.begin() as conn:
            for table in (journal.frames_table, journal.head_table, journal.header_table):
                table.drop(conn)
            conn.execute(sa.delete(store.identity_table).where(store.identity_table.c.key == "identity"))
    with pytest.raises(ValueError, match="claimed SQL state journal schema is missing|identity is missing"):
        SQLStore(uri)
    engine = sa.create_engine(uri)
    try:
        with engine.connect() as conn:
            assert conn.execute(sa.select(store.identity_table.c.key)).scalars().all() == ["engine_header"]
    finally:
        engine.dispose()


def test_sql_native_initialization_refuses_a_different_existing_owner(tmp_path):
    pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'owner.sqlite'}"
    with closing(SQLStore(uri, native=False)) as store:
        _ = store.store_id
        wrong = StoreHeader(StoreIdentity("2"*32, "sql"))
        with store._sql_transaction(write=True) as conn:
            with pytest.raises(ValueError, match="existing store identity"):
                SQLStateJournal.open(conn, store.table.name, header=wrong)
            from rcdb.sql_journal import sql_state_header
            assert sql_state_header(conn, store.table.name) is None
