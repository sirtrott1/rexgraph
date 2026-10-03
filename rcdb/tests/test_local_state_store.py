"""Canonical adapter exercises the engine through durable public operations."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import os
import subprocess
import sys
from threading import Barrier, Event

import numpy as np
import pytest

from rcdb import (BlobCodecSpec, LocalStore, PublicationUncertainError, VersionConflictError,
                  open_store, serialize_complex, structural_signature)
from rexgraph import RexGraph, TemporalRex
from rexgraph.object_identity import object_digest


def graph(n=2):
    value = RexGraph.from_graph(list(range(n-1)), list(range(1, n)))
    value._agent_meta = {"vertex_labels": [f"v{i}" for i in range(n)]}
    return value


@pytest.fixture
def store(tmp_path):
    value = LocalStore(tmp_path / "db")
    yield value
    value.close()


def test_literal_addresses_exact_metadata_detachment_and_prepared_ingest(store):
    source = graph()
    metadata = {"rational": Fraction(1, 7), "large": 2**90, "empty": (), "array": np.array([2**91], object)}
    record = store.put("a/b@1", source, meta=metadata, analytics=False)
    metadata["array"][0] = 0
    record.meta["array"][0] = 1
    assert store.get_record("a/b@1").meta["array"][0] == 2**91
    assert store.get_record("a/b@1").meta["empty"] == ()
    assert store.read_record("a/b@1").state_digest == object_digest(source)
    prepared = store.put_prepared("prepared", serialize_complex(source), structural_signature(source, analytics=False))
    assert prepared.version == 1 and object_digest(store.get("prepared")) == object_digest(source)
    assert store.get_version("a/b@1", 1).nE == 1


def test_tombstone_time_travel_recreate_and_cursor_resume_survive_reopening(store):
    first = store.put("a", graph(), analytics=False, _tx_time=10.0)
    after_first = store.change_cursor
    store.put("a", graph(3), analytics=False, _tx_time=20.0)
    assert store.delete("a", tx_time=30.0, expected_version=2)
    tombstone_cursor = store.change_cursor
    assert store.get("a") is None and store.list() == []
    assert store.tombstone("a").version == 2
    assert store.get("a", as_of=29.0).nE == 2
    assert store.get("a", as_of=30.0) is None
    assert [r.version for r in store.history("a")] == [1, 2]
    assert store.get_version("a", 1).nE == 1
    cursor = store.change_cursor
    assert not store.delete("absent") and store.change_cursor == cursor
    with pytest.raises(VersionConflictError):
        store.delete("a", expected_version=2)
    recreated = store.put("a", graph(4), analytics=False, _tx_time=40.0)
    assert recreated.version == 3 and store.get("a").nE == 3
    assert store.history("a")[0].tx_to == 20.0
    assert first.tx_to is None  # Earlier caller snapshot remains owned.
    assert store.tombstone("a") is None
    resumed = store.changes(after_first)
    assert [f.operation for f in resumed] == ["put", "delete", "put"]
    with LocalStoreContext(store.root) as reopened:
        assert reopened.store_id == store.store_id and reopened.change_cursor == store.change_cursor
        assert reopened.changes(tombstone_cursor)[0].record.version == 3
        assert reopened.cursor_for(reopened.changes()[-1]) == reopened.change_cursor
        assert object_digest(reopened.get_version("a", 1)) == object_digest(graph())


class LocalStoreContext:
    def __init__(self, root, **kwargs):
        self.store = LocalStore(root, **kwargs)

    def __enter__(self): return self.store
    def __exit__(self, *args): self.store.close()


@pytest.mark.parametrize("compression", [BlobCodecSpec(), BlobCodecSpec("zlib", 0), BlobCodecSpec("zlib", 6), BlobCodecSpec("zstd")])
def test_codec_is_a_durable_store_property_and_never_ambient(tmp_path, compression, monkeypatch):
    if compression.name == "zstd": pytest.importorskip("zstandard")
    from rcdb import core
    monkeypatch.setattr(core, "_codec", lambda: pytest.fail("ambient codec selection"))
    with LocalStoreContext(tmp_path / "codec", compression=compression) as store:
        record = store.put("r", graph(), analytics=False)
        assert record.envelope is not None
        expected = object_digest(store.get("r"))
    with LocalStoreContext(tmp_path / "codec") as reopened:
        assert reopened.blob_codec == compression and object_digest(reopened.get("r")) == expected
    wrong = BlobCodecSpec("zlib") if compression.name == "none" else BlobCodecSpec()
    with pytest.raises(ValueError, match="compression"):
        LocalStore(tmp_path / "codec", compression=wrong)


def test_temporal_state_uses_the_same_codec_ownership_and_cursor(store):
    value = TemporalRex([(np.array([0]), np.array([1])), (np.array([0, 1]), np.array([1, 2]))])
    record = store.put("time", value, analytics=False)
    assert record.envelope.object_type == "TemporalRex"
    assert object_digest(store.get("time")) == object_digest(value)


def test_multiple_live_handles_refresh_before_allocating_and_reading(store):
    with LocalStoreContext(store.root) as other:
        store.put("r", graph(), analytics=False)
        assert other.get_record("r").version == 1
        assert other.put("r", graph(3), analytics=False).version == 2
        assert store.get("r").nE == 2 and store.change_cursor.sequence == 2
        assert store.delete("r")
        assert other.get("r") is None
        assert other.put("r", graph(), analytics=False).version == 3
        assert store.get_record("r").version == 3


def test_concurrent_handles_share_one_allocation_and_complete_publication(store):
    handles = [LocalStore(store.root) for _ in range(6)]
    try:
        barrier = Barrier(len(handles))
        def run(handle):
            barrier.wait(timeout=15)
            return [handle.put("r", graph(), analytics=False).version for _ in range(3)]
        with ThreadPoolExecutor(max_workers=len(handles)) as pool:
            versions = [v for result in pool.map(run, handles) for v in result]
        assert sorted(versions) == list(range(1, 19))
        assert store.change_cursor.sequence == 18 and store.get_record("r").version == 18
    finally:
        for handle in handles: handle.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory-flock provider qualification")
def test_independent_processes_arbitrate_whole_local_transactions(store, tmp_path):
    script = """
import sys
from rcdb import LocalStore
from rexgraph import RexGraph
s = LocalStore(sys.argv[1])
try:
    for _ in range(3):
        s.put('shared', RexGraph.from_graph([0], [1]), analytics=False)
finally:
    s.close()
"""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    children = [subprocess.Popen([sys.executable, "-c", script, store.root], cwd=tmp_path,
                                env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(4)]
    try:
        for child in children:
            out, error = child.communicate(timeout=45)
            assert child.returncode == 0, error.decode()
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate()
    assert [r.version for r in store.history("shared")] == list(range(1, 13))
    assert store.change_cursor.sequence == 12


@pytest.mark.parametrize("damage", ["missing_blob", "swapped_blob", "symlink_blob", "missing_journal", "torn_journal", "missing_header", "changed_header"])
def test_authoritative_damage_refuses_without_replacing_store_identity(store, damage, tmp_path):
    from rcdb.header import StoreHeader
    store.put("r", graph(), analytics=False)
    frame = store.changes()[0]
    identity = store.store_id
    blob = store._blobs / frame.mutation.blob_digest
    if damage == "missing_blob": blob.unlink()
    elif damage == "swapped_blob": blob.write_bytes(b"wrong-content")
    elif damage == "symlink_blob":
        copy = tmp_path / "copy"
        copy.write_bytes(blob.read_bytes())
        blob.unlink()
        blob.symlink_to(copy)
    elif damage == "missing_journal": store._journal_path.unlink()
    elif damage == "torn_journal":
        raw = store._journal_path.read_bytes()
        store._journal_path.write_bytes(raw[:-1])
    elif damage == "missing_header": store._header_path.unlink()
    else:
        store._header_path.write_bytes(StoreHeader(store.header.identity, BlobCodecSpec("zlib")).to_bytes())
    with pytest.raises((ValueError, FileNotFoundError)):
        if damage in {"missing_blob", "swapped_blob", "symlink_blob"}:
            store.get("r")
        elif damage == "changed_header":
            store.get("r")
        else:
            LocalStore(store.root)
    assert store.store_id == identity


def test_prepublication_failure_retains_no_record_and_retry_uses_the_same_version(store, monkeypatch):
    initial = store.change_cursor
    def fail(frame): raise OSError("injected journal failure")
    with monkeypatch.context() as patch:
        patch.setattr(store._journal, "publish", fail)
        with pytest.raises(OSError, match="injected"):
            store.put("r", graph(), analytics=False)
    assert store.get("r") is None and store.change_cursor == initial
    assert store.put("r", graph(), analytics=False).version == 1


def test_successful_journal_then_failed_finalization_poison_handle_and_reopen_recovers(store, monkeypatch):
    def fail(frame): raise RuntimeError("injected finalization failure")
    monkeypatch.setattr(store._state, "apply", fail)
    with pytest.raises(PublicationUncertainError, match="journal published"):
        store.put("r", graph(), analytics=False)
    with pytest.raises(PublicationUncertainError): store.get("r")
    with LocalStoreContext(store.root) as reopened:
        assert reopened.get_record("r").version == 1 and reopened.get("r").nE == 1


def test_physical_uncertain_publication_poison_reads_and_writes(store, monkeypatch):
    def fail(frame): raise PublicationUncertainError("injected uncertain journal")
    monkeypatch.setattr(store._journal, "publish", fail)
    with pytest.raises(PublicationUncertainError): store.put("r", graph(), analytics=False)
    with pytest.raises(PublicationUncertainError): store.get("r")
    with pytest.raises(PublicationUncertainError): store.delete("r")


def test_commit_publication_expected_version_and_fresh_incarnation_lineage(store):
    first = store.commit_mutation("r", graph(), expected_version=0, tx_time=10.0, analytics=False)
    assert first.version == 1 and store.verify_commits("r")
    store.commit_mutation("r", graph(3), expected_version=1, tx_time=20.0, analytics=False)
    assert store.verify_commits("r")
    with pytest.raises(VersionConflictError):
        store.commit_mutation("r", graph(4), expected_version=1, tx_time=25.0, analytics=False)
    store.delete("r", tx_time=30.0)
    assert store.commit_history("r") == []
    # An ordinary initial publication of the new incarnation must not inherit
    # the former incarnation's signed parent when a later commit is created.
    store.put("r", graph(), analytics=False, _tx_time=40.0)
    fourth = store.commit_mutation("r", graph(3), expected_version=3, tx_time=50.0, analytics=False)
    assert fourth.version == 4 and store.verify_commits("r")
    assert len(store.commit_history("r")) == 1
    assert len(store.history("r")) == 4


def test_governed_deletion_refuses_and_missing_published_artifact_is_integrity_failure(store):
    store.configure_security(require_commits=True)
    store.commit_mutation("r", graph(), expected_version=0, analytics=False)
    with pytest.raises(PermissionError): store.delete("r")
    path = store._commit_path("r", 1)
    path.unlink()
    with pytest.raises(ValueError, match="artifact is missing"): store.get("r")
    assert not store.verify_commits("r")


def test_corpus_query_manifest_and_uri_bridge_use_the_same_visible_state(store):
    store.put("r", graph(), meta={"vertex_labels": ["alpha", "beta"]}, analytics=False)
    assert store.query(labels_any=["alpha"])[0].id == "r"
    assert store.corpus_snapshot().ids == ("r",)
    manifest = store.state_manifest()
    assert manifest["records"][0]["id"] == "r"
    other = open_store("local://"+store.root)
    try:
        assert isinstance(other, LocalStore) and other.store_id == store.store_id
        assert other.state_digest() == store.state_digest()
    finally:
        other.close()


def test_closed_handle_refuses_and_empty_unknown_query_is_not_silently_accepted(store):
    with pytest.raises(TypeError): store.query(unknown=True)
    store.close()
    with pytest.raises(ValueError, match="closed"): store.put("r", graph(), analytics=False)
    with pytest.raises(ValueError, match="closed"): store.get("r")


def test_agent_compatibility_surface_exposes_the_same_canonical_backend():
    adapter = pytest.importorskip("agent.rcdb")
    assert adapter.LocalStore is LocalStore
    assert adapter.BlobCodecSpec is BlobCodecSpec


def test_payload_encryption_ownership_and_reopen_use_persisted_codec(store):
    pytest.importorskip("cryptography")
    from rexgraph.io.security import StaticKeyProvider, ENVELOPE_MAGIC
    keys = StaticKeyProvider({"records": b"k"*32})
    store.configure_security(key_id="records", keys=keys, signature_mode="minimal")
    store.put("r", graph(), analytics=False, meta={"vertex_labels": ["secret"]})
    frame = store.changes()[0]
    raw = (store._blobs / frame.mutation.blob_digest).read_bytes()
    assert raw.startswith(ENVELOPE_MAGIC) and "vertex_labels" not in store.get_record("r").meta
    with LocalStoreContext(store.root) as reopened:
        with pytest.raises(PermissionError): reopened.get("r")
        reopened.configure_security(key_id="records", keys=StaticKeyProvider({"records": b"w"*32}))
        with pytest.raises(PermissionError): reopened.get("r")
        reopened.configure_security(key_id="records", keys=keys)
        assert object_digest(reopened.get("r")) == object_digest(graph())


def test_independent_mutation_handles_cannot_both_commit_the_same_expected_version(store):
    store.commit_mutation("r", graph(), analytics=False, expected_version=0)
    handles = [LocalStore(store.root), LocalStore(store.root)]
    barrier = Barrier(2)
    def run(handle):
        barrier.wait(timeout=15)
        try:
            return handle.commit_mutation("r", graph(3), analytics=False, expected_version=1).version
        except VersionConflictError:
            return "conflict"
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(run, handles))
        assert set(outcomes) == {2, "conflict"}
        assert store.change_cursor.sequence == 2 and store.verify_commits("r")
        assert store._load_commit_bytes("r", 3) is None
    finally:
        for handle in handles: handle.close()


def test_rcql_complete_selection_holds_one_store_transaction(store, monkeypatch):
    rcql = pytest.importorskip("rcql")
    store.put("a", graph(), analytics=False)
    store.put("b", graph(), analytics=False)
    with LocalStoreContext(store.root) as writer:
        first_read, started, finished = Event(), Event(), Event()
        original = store.read_record
        def read(*args, **kwargs):
            result = original(*args, **kwargs)
            if args[0] == "a":
                first_read.set()
                assert started.wait(10)
                assert not finished.wait(.05), "writer published inside the RCQL selection transaction"
            return result
        def update():
            assert first_read.wait(10)
            started.set()
            writer.put("b", graph(3), analytics=False)
            finished.set()
        monkeypatch.setattr(store, "read_record", read)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(update)
            selected = rcql.SnapshotContext.select(store, (rcql.SourceSelection("left", "a"),
                                                           rcql.SourceSelection("right", "b")))
            future.result(timeout=15)
        assert selected.sources["right"].ref.record_version == 1
        assert writer.get_record("b").version == 2
        result = rcql.Executor(sources=selected.sources, evidence=selected).execute(
            rcql.parse("FROM $right RETURN COUNT(CELLS(1))"))
        assert result.values == (1,)


def test_concurrent_creation_publishes_one_complete_store_identity(tmp_path):
    root = tmp_path / "new"
    with ThreadPoolExecutor(max_workers=6) as pool:
        handles = list(pool.map(lambda _: LocalStore(root), range(6)))
    try:
        assert len({s.store_id for s in handles}) == 1
        assert all(s.change_cursor.sequence == 0 for s in handles)
        assert handles[0].put("r", graph(), analytics=False).version == 1
        assert handles[-1].get("r").nE == 1
    finally:
        for handle in handles: handle.close()


def test_logical_state_digest_excludes_layout_compression_and_opaque_ownership(tmp_path):
    with LocalStoreContext(tmp_path / "none") as first, LocalStoreContext(
            tmp_path / "zlib", compression=BlobCodecSpec("zlib")) as second:
        for store in (first, second):
            store.put("r", graph(), analytics=False, _tx_time=10.0)
            store.put("r", graph(3), analytics=False, _tx_time=20.0)
            store.delete("r", tx_time=30.0)
            store.put("r", graph(), analytics=False, _tx_time=40.0)
        assert first.store_id != second.store_id
        assert first.change_cursor.digest != second.change_cursor.digest
        assert first.state_manifest() == second.state_manifest()
        assert first.state_digest() == second.state_digest()
