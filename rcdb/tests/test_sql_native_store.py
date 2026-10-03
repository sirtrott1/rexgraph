"""SQL's indexed projection participates in the common native record engine."""
from contextlib import closing
from fractions import Fraction
import subprocess
import sys

import numpy as np
import pytest

from rcdb import BlobCodecSpec, SQLStore, serialize_complex, structural_signature
from rexgraph import RexGraph, TemporalRex
from rexgraph.object_identity import object_digest


def graph(n=2):
    return RexGraph.from_graph(list(range(n-1)), list(range(1, n)))


@pytest.fixture
def store(tmp_path):
    pytest.importorskip("sqlalchemy")
    with closing(SQLStore(f"sqlite:///{tmp_path / 'native.sqlite'}")) as value:
        yield value


@pytest.mark.parametrize("profile", [BlobCodecSpec(), BlobCodecSpec("zlib", 0),
                                    BlobCodecSpec("zlib", 6), BlobCodecSpec("zstd")])
def test_sql_codec_and_header_survive_reopen_without_ambient_selection(tmp_path, profile, monkeypatch):
    pytest.importorskip("sqlalchemy")
    if profile.name == "zstd":
        pytest.importorskip("zstandard")
    from rcdb import core
    monkeypatch.setattr(core, "_codec", lambda: pytest.fail("ambient codec selected"))
    uri = f"sqlite:///{tmp_path / 'codec.sqlite'}"
    with closing(SQLStore(uri, compression=profile)) as store:
        store.put("r", graph(), analytics=False)
        header, cursor = store.header, store.change_cursor
    with closing(SQLStore(uri)) as reopened:
        assert reopened.header == header and reopened.blob_codec == profile
        assert reopened.change_cursor == cursor
        assert object_digest(reopened.get("r")) == object_digest(graph())
    wrong = BlobCodecSpec("zlib") if profile.name == "none" else BlobCodecSpec()
    with pytest.raises(ValueError, match="compression.*migration"):
        SQLStore(uri, compression=wrong)


def test_sql_native_history_refresh_and_cursor_resume_survive_reopen(store):
    start = store.change_cursor
    with closing(SQLStore(store.conn_str)) as other:
        store.put("literal/id@1", graph(), analytics=False, _tx_time=10.0)
        assert other.get_record("literal/id@1").version == 1
        first = other.change_cursor
        other.put("literal/id@1", graph(3), analytics=False, _tx_time=20.0)
        assert store.delete("literal/id@1", tx_time=30.0, expected_version=2)
        assert other.get("literal/id@1") is None
        assert other.next_version("literal/id@1") == 3
        assert other.put("literal/id@1", graph(), analytics=False, _tx_time=40.0).version == 3
    with closing(SQLStore(store.conn_str)) as reopened:
        assert reopened.store_id == store.store_id
        assert reopened.change_cursor == store.change_cursor
        assert [f.operation for f in reopened.changes(start)] == ["put", "put", "delete", "put"]
        assert [f.sequence for f in reopened.changes(first)] == [2, 3, 4]
        assert [r.tx_to for r in reopened.history("literal/id@1")] == [20.0, 30.0, None]
        assert object_digest(reopened.get_version("literal/id@1", 2)) == object_digest(graph(3))


@pytest.mark.parametrize("populated", [False, True])
def test_existing_headerless_sql_stays_compatible_until_explicit_migration(tmp_path, populated):
    pytest.importorskip("sqlalchemy")
    uri = f"sqlite:///{tmp_path / 'legacy.sqlite'}"
    with closing(SQLStore(uri, native=False)) as legacy:
        identity = legacy.store_id
        if populated:
            legacy.put("r", graph(), analytics=False)
    with closing(SQLStore(uri)) as reopened:
        assert reopened.header is None and reopened.store_id == identity
        assert (reopened.get("r") is not None) == populated
        with pytest.raises(ValueError, match="migration"):
            reopened.changes()
    for options in ({"native": True}, {"compression": BlobCodecSpec()}):
        with pytest.raises(ValueError, match="headerless.*migration"):
            SQLStore(uri, **options)
    with closing(SQLStore(uri, native=False)) as reopened:
        assert reopened.store_id == identity and reopened.header is None
        assert reopened.put("r", graph(), analytics=False).version == (2 if populated else 1)


def test_existing_native_sql_refuses_compatibility_downgrade(store):
    header = store.header
    with pytest.raises(ValueError, match="downgraded.*migration"):
        SQLStore(store.conn_str, native=False)
    assert store.header == header and store.change_cursor.sequence == 0


def test_failed_sql_genesis_publication_rolls_back_schema_identity_and_header(tmp_path, monkeypatch):
    sa = pytest.importorskip("sqlalchemy")
    from rcdb.sql_journal import SQLStateJournal
    uri = f"sqlite:///{tmp_path / 'genesis.sqlite'}"
    original = SQLStateJournal.open
    def failed(cls, connection, prefix, **kwargs):
        original(connection, prefix, **kwargs)
        raise OSError("genesis publication failed")
    with monkeypatch.context() as patch:
        patch.setattr(SQLStateJournal, "open", classmethod(failed))
        with pytest.raises(OSError, match="genesis publication failed"):
            SQLStore(uri)
    engine = sa.create_engine(uri)
    try:
        assert sa.inspect(engine).get_table_names() == []
    finally:
        engine.dispose()
    with closing(SQLStore(uri)) as reopened:
        assert reopened.header is not None and reopened.change_cursor.sequence == 0
        assert reopened.put("r", graph(), analytics=False).version == 1


@pytest.mark.parametrize("missing", ["records", "labels", "commits"])
def test_native_sql_reopen_refuses_lost_projection_schema_before_creating_replacement_tables(store, missing):
    import sqlalchemy as sa
    store.commit_mutation("r", graph(), meta={"vertex_labels": ["kept"]}, analytics=False)
    lost = {"records": store.table, "labels": store.labels_table, "commits": store.commits_table}[missing]
    with store.engine.begin() as conn:
        lost.drop(conn)
    before = set(sa.inspect(store.engine).get_table_names())
    assert lost.name not in before
    with pytest.raises(ValueError, match="projection schema is missing"):
        SQLStore(store.conn_str)
    assert set(sa.inspect(store.engine).get_table_names()) == before


@pytest.mark.parametrize("reopen", [False, True])
@pytest.mark.parametrize("damage", ["metadata", "interval", "promoted", "missing-row", "missing-label", "extra-label"])
def test_sql_refuses_damaged_metadata_indexes_before_query_can_lose_a_record(store, damage, reopen):
    import sqlalchemy as sa
    store.put("r", graph(), meta={"vertex_labels": ["kept"]}, analytics=False, _tx_time=10.0)
    identity = store.store_id
    with store.engine.begin() as conn:
        if damage == "metadata":
            conn.execute(sa.update(store.table).values(meta='{}'))
        elif damage == "interval":
            conn.execute(sa.update(store.table).values(tx_to=20.0))
        elif damage == "promoted":
            conn.execute(sa.update(store.table).values(nE=0))
        elif damage == "missing-row":
            conn.execute(sa.delete(store.table))
        elif damage == "missing-label":
            conn.execute(sa.delete(store.labels_table))
        else:
            conn.execute(sa.insert(store.labels_table).values(id="r", version=1, label="unpublished"))
        before = conn.execute(sa.select(store.table)).fetchall()
        label_rows = conn.execute(sa.select(store.labels_table)).fetchall()
    with pytest.raises(ValueError, match="projection|label index"):
        if reopen:
            SQLStore(store.conn_str)
        else:
            store.query(labels_any=["kept"], min_nE=1)
    with store.engine.connect() as conn:
        assert conn.execute(sa.select(store.table)).fetchall() == before
        assert conn.execute(sa.select(store.labels_table)).fetchall() == label_rows
    assert store.store_id == identity


@pytest.mark.parametrize("damage", ["missing", "swapped", "integer"])
def test_sql_checks_physical_payload_before_open_even_with_verification_disabled(store, damage):
    import sqlalchemy as sa
    store.put("r", graph(), analytics=False)
    with store.engine.begin() as conn:
        if damage == "integer":
            conn.exec_driver_sql("UPDATE rc_complexes SET blob = 64")
        else:
            conn.execute(sa.update(store.table).values(blob=None if damage == "missing" else b"wrong"))
    for read in (lambda: store.get("r", verify=False), lambda: store.get_version("r", 1)):
        with pytest.raises(ValueError, match="missing|content digest|bounded binary"):
            read()
    assert store.get_record("r").version == 1  # Metadata reads do not open graph payloads.


@pytest.mark.parametrize("damage", ["missing", "swapped", "integer"])
def test_sql_checks_claimed_commit_bytes_and_never_certifies_a_damaged_chain(store, damage):
    import sqlalchemy as sa
    store.commit_mutation("r", graph(), analytics=False)
    with store.engine.begin() as conn:
        if damage == "missing":
            conn.execute(sa.delete(store.commits_table))
        elif damage == "integer":
            conn.exec_driver_sql("UPDATE rc_complexes_commits SET artifact = 64")
        else:
            conn.execute(sa.update(store.commits_table).values(artifact=b"wrong"))
    with pytest.raises(ValueError, match="artifact|bounded binary"):
        store.get("r", verify=False)
    with pytest.raises(ValueError, match="artifact|bounded binary"):
        store.commit_history("r")
    assert not store.verify_commits("r")


def test_sql_time_selection_precedes_index_predicates_and_keeps_one_version_per_id(store):
    store.put("r", graph(), meta={"vertex_labels": ["old"]}, analytics=False,
              valid_from=1.0, valid_to=100.0, _tx_time=10.0)
    store.put("r", graph(3), meta={"vertex_labels": ["new"]}, analytics=False,
              valid_from=1.0, valid_to=100.0, _tx_time=20.0)
    assert store.query(valid_at=5.0, labels_any=["old"]) == []
    assert store.query(valid_at=5.0, max_nE=1) == []
    assert [(r.id, r.version) for r in store.list(valid_at=5.0)] == [("r", 2)]
    assert [(r.id, r.version) for r in store.query(valid_at=5.0, min_nE=2)] == [("r", 2)]
    assert [(r.id, r.version) for r in store.query(as_of=10.0, labels_any=["old"])] == [("r", 1)]
    assert store.query(as_of=10.0, min_nE=2) == []
    assert [(r.id, r.version) for r in store.query(include_history=True, labels_any=["old"])] == [("r", 1)]
    with pytest.raises(ValueError, match="history collection"):
        store.query(include_history=True, valid_at=5.0)


def test_sql_native_prepared_temporal_and_exact_metadata_use_core_state_codec(store):
    source = graph()
    metadata = {"exact": Fraction(1, 7), "integer": 2**90, "empty": (), "array": np.array([2**91], object)}
    record = store.put_prepared("prepared", serialize_complex(source), structural_signature(source, analytics=False), meta=metadata)
    metadata["array"][0] = 0
    record.meta["array"][0] = 1
    assert store.get_record("prepared").meta["array"][0] == 2**91
    assert store.get_record("prepared").meta["exact"] == Fraction(1, 7)
    assert object_digest(store.get("prepared")) == object_digest(source)
    temporal = TemporalRex([(np.array([0]), np.array([1])), (np.array([0, 1]), np.array([1, 2]))])
    record = store.put("time", temporal, analytics=False)
    assert record.envelope.object_type == "TemporalRex"
    with closing(SQLStore(store.conn_str)) as reopened:
        assert object_digest(reopened.get("time")) == object_digest(temporal)


@pytest.mark.parametrize("committed", [False, True])
def test_sql_abrupt_process_exit_keeps_complete_checked_publication_or_none(store, tmp_path, committed):
    script = tmp_path / "crash.py"
    script.write_text('''import os, sys
from rcdb import SQLStore
from rexgraph import RexGraph
store = SQLStore(sys.argv[1])
def write():
    store.commit_mutation("r", RexGraph.from_graph([0], [1]), analytics=False, expected_version=0)
if sys.argv[2] == "committed":
    write()
    os._exit(42)
with store._sql_transaction(write=True):
    write()
    os._exit(42)
''')
    start = store.change_cursor
    result = subprocess.run([sys.executable, "-I", str(script), store.conn_str,
                             "committed" if committed else "pending"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 42, result.stderr
    assert store.change_cursor.sequence == int(committed)
    if committed:
        assert store.get("r").nE == 1 and store.verify_commits("r")
        assert len(store.commit_history("r")) == 1
    else:
        assert store.change_cursor == start and store.list() == []
        assert store._load_commit_bytes("r", 1) is None and store.next_version("r") == 1
