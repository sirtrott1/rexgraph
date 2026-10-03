"""SQLite revision certificates save unchanged scans without trusting stale indexes."""
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from runpy import run_path

import pytest

from rcdb import SQLStore

sa = pytest.importorskip("sqlalchemy")
run_isolated = run_path(str(Path(__file__).resolve().parents[2] / "scripts/test_subprocess.py"))["run_isolated"]


@pytest.fixture(params=["file", "memory"])
def store(request, tmp_path):
    uri = f"sqlite:///{tmp_path}/store.sqlite" if request.param == "file" else "sqlite://"
    with closing(SQLStore(uri)) as st:
        for i in range(8):
            st.put_record(f"r{i}", Fraction(i, 7), tags=["kept"], meta={"vertex_labels": ["kept"]})
        st.list(limit=1)
        yield st


def audits(store, monkeypatch):
    original, calls = store._native._audit_projection, []
    def audit(connection):
        calls.append(store._native.state.cursor)
        return original(connection)
    monkeypatch.setattr(store._native, "_audit_projection", audit)
    return calls


def test_unchanged_reads_share_one_certificate(store, monkeypatch):
    calls = audits(store, monkeypatch)
    for _ in range(4):
        assert len(store.list(limit=1)) == len(store.query(limit=1, tags_any=["kept"])) == 1
        assert store.get_record("r0").version == 1
        assert store.read_record("r0").value == Fraction(0, 7)
    assert calls == []


@pytest.mark.parametrize("damage", ["metadata", "signature", "interval", "promoted", "missing-row", "extra-row", "missing-label", "extra-label"])
def test_same_connection_out_of_band_changes_cannot_reuse_a_certificate(store, damage, monkeypatch):
    calls = audits(store, monkeypatch)
    with store.engine.begin() as connection:
        if damage == "metadata": connection.execute(store.table.update().values(meta="{}"))
        elif damage == "signature": connection.execute(store.table.update().values(signature="{}"))
        elif damage == "interval": connection.execute(store.table.update().values(tx_to=0.))
        elif damage == "promoted": connection.execute(store.table.update().values(nE=1))
        elif damage == "missing-row": connection.execute(store.table.delete().where(store.table.c.id == "r0"))
        elif damage == "extra-row":
            original = dict(connection.execute(sa.select(store.table).limit(1)).first()._mapping)
            original["id"] = "unpublished"; connection.execute(store.table.insert().values(**original))
        elif damage == "missing-label": connection.execute(store.labels_table.delete())
        else: connection.execute(store.labels_table.insert().values(id="r0", version=1, label="unpublished"))
    with pytest.raises(ValueError, match="projection|label index"):
        store.query(limit=1, tags_any=["kept"])
    assert len(calls) == 1 and store._native._projection_certificate is None
    with pytest.raises(ValueError, match="projection|label index"):
        store.list(limit=1)
    assert len(calls) == 2


def test_native_writes_and_rollback_invalidate_the_certificate(store, monkeypatch):
    calls = audits(store, monkeypatch)
    original, deltas = store._native._audit_projection_delta, []
    def audit_delta(connection, addresses):
        deltas.append(addresses)
        return original(connection, addresses)
    monkeypatch.setattr(store._native, "_audit_projection_delta", audit_delta)
    store.put_record("new", None)
    assert calls == []
    assert store.get_record("new").version == 1 and calls == []
    assert deltas == [frozenset({("new", 1)})]
    store.get_record("new"); assert calls == [] and len(deltas) == 1
    with pytest.raises(OSError):
        with store.write_scope():
            store.put_record("rollback", None)
            raise OSError("abort")
    assert store._native._projection_certificate is None and store._native._projection_delta is None
    assert store.get_record("rollback") is None and len(calls) == 1
    store.get_record("new"); assert len(calls) == 1


def test_external_rollback_is_conservatively_reaudited(store, monkeypatch):
    calls = audits(store, monkeypatch)
    with store.engine.connect() as connection:
        transaction = connection.begin()
        connection.execute(store.table.update().values(meta="{}"))
        transaction.rollback()
    assert store.get_record("r0").meta["vertex_labels"] == ["kept"]
    assert len(calls) == 1
    store.get_record("r0"); assert len(calls) == 1


def test_schema_changes_invalidate_even_without_a_record_change(store, monkeypatch):
    calls = audits(store, monkeypatch)
    with store.engine.begin() as connection:
        connection.exec_driver_sql('CREATE TABLE unrelated_schema (value INTEGER)')
    store.get_record("r0"); assert len(calls) == 1
    store.get_record("r0"); assert len(calls) == 1


def test_same_cursor_on_another_connection_never_reuses_revision_counters(tmp_path, monkeypatch):
    with closing(SQLStore(f"sqlite:///{tmp_path}/db.sqlite")) as store:
        store.put_record("r", None); store.get_record("r")
        calls = audits(store, monkeypatch)
        with store.engine.connect() as held:
            original = held.connection.driver_connection
            store.get_record("r")
            assert store._native._projection_certificate[0] is not original
            assert len(calls) == 1


@pytest.mark.parametrize("operation", ["native-write", "metadata", "label"])
def test_other_handle_commits_invalidate_data_version(tmp_path, monkeypatch, operation):
    uri = f"sqlite:///{tmp_path}/db.sqlite"
    with closing(SQLStore(uri)) as store, closing(SQLStore(uri)) as other:
        store.put_record("r", None, meta={"vertex_labels": ["kept"]})
        store.get_record("r"); calls = audits(store, monkeypatch)
        if operation == "native-write":
            other.put_record("r", Fraction(1, 7))
            assert store.read_record("r").value == Fraction(1, 7)
        else:
            with other.engine.begin() as connection:
                if operation == "metadata": connection.execute(other.table.update().values(meta="{}"))
                else: connection.execute(other.labels_table.delete())
            with pytest.raises(ValueError, match="projection|label index"): store.list(limit=1)
        assert len(calls) == 1


def test_fresh_process_projection_change_is_detected(tmp_path):
    uri = f"sqlite:///{tmp_path}/db.sqlite"
    with closing(SQLStore(uri)) as store:
        store.put_record("r", None, meta={"vertex_labels": ["kept"]}); store.list(limit=1)
        code = f"""
import sqlite3
with sqlite3.connect({str(tmp_path / 'db.sqlite')!r}) as connection:
 connection.execute('UPDATE rc_complexes SET meta = ?', ('{{}}',))
"""
        result = run_isolated(code, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        with pytest.raises(ValueError, match="projection"): store.query(limit=1, labels_any=["kept"])


@pytest.mark.parametrize("fallback", ["unqualified", "unpinned"])
def test_unqualified_revision_guard_keeps_full_audits(store, monkeypatch, fallback):
    calls = audits(store, monkeypatch)
    if fallback == "unqualified":
        monkeypatch.setattr(store._native, "_projection_revision", lambda connection: None)
        store.get_record("r0"); store.get_record("r0")
        assert len(calls) == 2 and store._native._projection_certificate is None
    else:
        with store.engine.connect() as connection:
            assert not connection.connection.driver_connection.in_transaction
            store._native.check_projection(connection)
            assert store._native._projection_certificate is None
        assert len(calls) == 1


def test_failed_audit_and_revision_change_during_audit_issue_no_certificate(store, monkeypatch):
    calls = audits(store, monkeypatch)
    store._native._projection_certificate = None
    original = store._native._projection_revision
    revisions = []
    def changed(connection):
        raw, values = original(connection)
        revisions.append(1)
        return raw, (*values[:-1], values[-1]+len(revisions))
    monkeypatch.setattr(store._native, "_projection_revision", changed)
    store.get_record("r0"); store.get_record("r0")
    assert len(calls) == 2 and store._native._projection_certificate is None


def test_close_releases_the_certificate_connection(store):
    native = store._native
    assert native._projection_certificate is not None
    store.close()
    assert native.state is None and native._projection_certificate is None


def test_sql_residual_selection_stops_after_complete_matches(store, monkeypatch):
    store.put_record("early", None, tags=["skip"])
    store.put_record("wanted", None, tags=["wanted"])
    store.put_record("newest", None, tags=["skip"])
    store.list(limit=1)
    original, calls = store._row_to_record, []
    def converted(row): calls.append(row.id); return original(row)
    monkeypatch.setattr(store, "_row_to_record", converted)
    result = store.query(limit=1, tags_any=["wanted"])
    assert [r.id for r in result] == ["wanted"] and calls == ["newest", "wanted"]
    if hasattr(store.engine.pool, "checkedout"):
        assert store.engine.pool.checkedout() == 0


@pytest.mark.parametrize("damage", ["missing", "swapped"])
def test_payload_checks_remain_active_on_a_projection_cache_hit(store, damage):
    store.get_record("r0")
    with store.engine.begin() as connection:
        if damage == "missing": blob = None
        else:
            blob = connection.execute(sa.select(store.table.c.blob).where(store.table.c.id == "r1")).scalar_one()
        connection.execute(store.table.update().where(store.table.c.id == "r0").values(blob=blob))
    store.get_record("r0")  # BLOBs are outside the metadata audit, now recertified.
    assert store._native._projection_certificate is not None
    with pytest.raises(ValueError, match="payload"): store.get("r0", verify=False)


def test_wal_reader_keeps_its_pinned_certificate_then_refreshes_after_peer_commit(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/wal.sqlite"
    with closing(SQLStore(uri)) as store, closing(SQLStore(uri)) as other:
        with store.engine.connect() as connection:
            connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        store.put_record("r", "prior"); store.get_record("r")
        calls = audits(store, monkeypatch)
        with store.read_transaction():
            assert store.read_record("r").value == "prior"
            other.put_record("r", "next")
            assert store.read_record("r").value == "prior"
            assert calls == []
        assert store.read_record("r").value == "next" and len(calls) == 1


def test_malicious_sql_trigger_does_not_certify_a_native_publication(tmp_path):
    with closing(SQLStore(f"sqlite:///{tmp_path}/trigger.sqlite")) as store:
        store.put_record("prior", None, meta={"vertex_labels": ["kept"]})
        with store.engine.begin() as connection:
            connection.exec_driver_sql('CREATE TRIGGER corrupt_prior AFTER INSERT ON rc_complexes '
                'BEGIN UPDATE rc_complexes SET meta = \'{}\' WHERE id = \'prior\'; END')
        store.get_record("prior")
        store.put_record("new", None)
        with pytest.raises(ValueError, match="projection"): store.list(limit=1)
