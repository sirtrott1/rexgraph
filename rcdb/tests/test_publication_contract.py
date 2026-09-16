"""A published RCDB version must have a payload and its required attestation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest

from rcdb import FileStore, MemoryStore, ObjectStore, PublicationUncertainError, RexStore, SQLStore
from rcdb.core import serialize_complex, structural_signature


def rex(n=2):
    return RexGraph.from_graph(list(range(n)), list(range(1, n + 1)))


@pytest.fixture(params=["memory", "file", "rex", "sql", "object"])
def store(request, tmp_path):
    if request.param == "memory":
        result = MemoryStore()
    elif request.param == "file":
        result = FileStore(str(tmp_path / "file"))
    elif request.param == "rex":
        result = RexStore(str(tmp_path / "rex"))
    elif request.param == "sql":
        pytest.importorskip("sqlalchemy")
        result = SQLStore(f"sqlite:///{tmp_path / 'sql.db'}")
    else:
        pytest.importorskip("fsspec")
        result = ObjectStore(f"file://{tmp_path / 'object'}")
    yield result
    result.close()


@pytest.mark.parametrize("existing", [False, True])
def test_failed_serialization_does_not_publish_or_close_a_version(store, existing, monkeypatch):
    first = rex()
    if existing:
        store.put("r", first, analytics=False)
    original = store._serialize_payload
    def broken(value):
        raise ValueError("serialization failed")
    monkeypatch.setattr(store, "_serialize_payload", broken)
    with pytest.raises(ValueError, match="serialization failed"):
        store.put("r", rex(3), analytics=False)
    assert len(store.history("r")) == int(existing)
    assert store.get_version("r", 1 + int(existing)) is None
    if existing:
        assert store.get_record("r").tx_to is None
        assert object_digest(store.get("r")) == object_digest(first)
    monkeypatch.setattr(store, "_serialize_payload", original)
    rec = store.put("r", rex(4), analytics=False)
    assert rec.version == 1 + int(existing)


def test_prepared_write_cannot_bypass_required_commits(store):
    value = rex()
    payload = serialize_complex(value)
    signature = structural_signature(value, analytics=False)
    store.configure_security(require_commits=True)
    with pytest.raises(PermissionError, match="mutation commits"):
        store.put_prepared("r", payload, signature)
    with pytest.raises(PermissionError, match="mutation commits"):
        store.put("r", value, _mutation_commit=True)
    assert store.history("r") == []
    assert store.get_version("r", 1) is None
    store.commit_mutation("r", value, analytics=False)
    assert store.verify_commits("r")


def test_optional_prepared_write_keeps_the_existing_payload_contract(store):
    value = rex()
    store.put_prepared("r", serialize_complex(value), structural_signature(value, analytics=False))
    assert object_digest(store.get("r")) == object_digest(value)


def test_failed_commit_payload_keeps_the_prior_history_and_removes_staged_artifact(store, monkeypatch):
    first = rex()
    store.configure_security(require_commits=True)
    store.commit_mutation("r", first, analytics=False)
    original = store._serialize_payload
    def broken(value):
        raise OSError("payload unavailable")
    monkeypatch.setattr(store, "_serialize_payload", broken)
    with pytest.raises(OSError, match="payload unavailable"):
        store.commit_mutation("r", rex(3), analytics=False)
    monkeypatch.setattr(store, "_serialize_payload", original)
    assert len(store.history("r")) == len(store.commit_history("r")) == 1
    assert store._load_commit_bytes("r", 2) is None
    assert store.get_record("r").tx_to is None
    assert store.verify_commits("r")
    assert object_digest(store.get("r")) == object_digest(first)


def test_uncertain_publication_retains_the_staged_attestation(store, monkeypatch):
    def uncertain(*args, **kwargs):
        raise PublicationUncertainError("recovery required")
    monkeypatch.setattr(store, "_put_impl", uncertain)
    with pytest.raises(PublicationUncertainError):
        store.commit_mutation("r", rex(), analytics=False)
    assert store._load_commit_bytes("r", 1) is not None
    with pytest.raises(PublicationUncertainError, match="reopen and verify"):
        store.put_prepared("other", b"unused", {})


@pytest.mark.parametrize("kind", ["file", "object"])
def test_failed_journal_never_exposes_an_orphan_payload_by_version(kind, tmp_path, monkeypatch):
    if kind == "object":
        pytest.importorskip("fsspec")
        def factory():
            return ObjectStore(f"file://{tmp_path / 'object'}")
        journal = "_write_journal"
    else:
        def factory():
            return FileStore(str(tmp_path / "file"))
        journal = "_append_log"
    store = factory()
    store.put("r", rex(), analytics=False)
    original = getattr(store, journal)
    def broken(*args):
        raise OSError("journal unavailable")
    monkeypatch.setattr(store, journal, broken)
    with pytest.raises(OSError, match="journal unavailable"):
        store.put("r", rex(3), analytics=False)
    assert store.get_version("r", 2) is None
    assert len(store.history("r")) == 1
    reopened = factory()
    assert reopened.get_version("r", 2) is None
    assert len(reopened.history("r")) == 1
    reopened.close()
    monkeypatch.setattr(store, journal, original)
    assert store.put("r", rex(4), analytics=False).version == 2
    assert store.get_version("r", 2).nE == 4
    store.close()


def test_same_handle_mutations_do_not_interleave_commit_staging(monkeypatch):
    store = MemoryStore().configure_security(require_commits=True)
    first = Event()
    release = Event()
    attempted = Event()
    second_staged = Event()
    original = store._store_commit_bytes
    def stage(id, version, blob):
        if not first.is_set():
            first.set()
            assert release.wait(5)
        else:
            second_staged.set()
        original(id, version, blob)
    monkeypatch.setattr(store, "_store_commit_bytes", stage)
    def second():
        attempted.set()
        return store.commit_mutation("r", rex(3), analytics=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(store.commit_mutation, "r", rex(), analytics=False)
        try:
            assert first.wait(5)
            b = pool.submit(second)
            assert attempted.wait(5)
            assert not second_staged.wait(0.1)
        finally:
            release.set()
        assert a.result(timeout=5).version == 1
        assert b.result(timeout=5).version == 2
    assert store.verify_commits("r")
    assert len(store.commit_history("r")) == 2


def test_log_encoding_failure_does_not_leave_a_torn_tail(tmp_path):
    from rcdb import ComplexRecord, index
    path = tmp_path / "records.log"
    row = ComplexRecord("r", {"nV": 1})
    index.log_append(path, "put", "r", row)
    before = path.read_bytes()
    with pytest.raises((OverflowError, ValueError)):
        index.log_append(path, "put", "r", row, extra=[2**80])
    assert path.read_bytes() == before
    index.log_append(path, "put", "s", ComplexRecord("s", {"nV": 2}))
    assert [entry[1] for entry in index.log_read(path)] == ["r", "s"]


def test_log_sync_failure_rolls_back_before_a_retry(tmp_path, monkeypatch):
    import os

    from rcdb import ComplexRecord, index
    path = tmp_path / "records.log"
    row = ComplexRecord("r", {"nV": 1})
    index.log_append(path, "put", "r", row)
    before = path.read_bytes()
    sync = os.fsync
    calls = []
    def fail_once(fd):
        calls.append(fd)
        if len(calls) == 1:
            raise OSError("sync failed")
        sync(fd)
    monkeypatch.setattr(os, "fsync", fail_once)
    with pytest.raises(OSError, match="sync failed"):
        index.log_append(path, "put", "s", ComplexRecord("s", {}))
    assert path.read_bytes() == before
    index.log_append(path, "put", "s", ComplexRecord("s", {}))
    assert [entry[1] for entry in index.log_read(path)] == ["r", "s"]


def test_log_reports_uncertain_publication_when_rollback_cannot_sync(tmp_path, monkeypatch):
    import os

    from rcdb import ComplexRecord, index
    def broken(fd):
        raise OSError("device unavailable")
    monkeypatch.setattr(os, "fsync", broken)
    with pytest.raises(PublicationUncertainError, match="rollback failed"):
        index.log_append(tmp_path / "records.log", "put", "r", ComplexRecord("r", {}))
