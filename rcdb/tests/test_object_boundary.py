"""Legacy object stores are pinned read only sources, with an explicit native exit."""
from contextlib import closing, contextmanager
from fractions import Fraction
import json
import uuid

import pytest

from rcdb import (MemoryStore, ObjectStore, PublicationUncertainError,
                  LegacyMigrationPlan, migrate_legacy_batch, plan_legacy_migration)
from rexgraph import RexGraph


def graph():
    return RexGraph.from_graph([0], [1], w_E=[Fraction(1, 7)])


@pytest.fixture(params=["file", "memory"])
def uri(request, tmp_path):
    pytest.importorskip("fsspec")
    return f"file://{tmp_path / 'objects'}" if request.param == "file" else "memory://object-boundary-"+uuid.uuid4().hex


def written(uri):
    with closing(ObjectStore(uri, read_only=False)) as writer:
        writer.put("literal@1", graph(), meta={"q": Fraction(2, 7)}, tags=["old"],
                   valid_from=1.0, valid_to=9.0, _tx_time=10.0, analytics=False)
        writer.put("literal@1", graph(), meta={"q": Fraction(3, 7)}, tags=["new"],
                   _tx_time=20.0, analytics=False)


def test_existing_source_reads_queries_and_migration_never_write(uri, monkeypatch):
    written(uri)
    import rcdb.objectstore as module
    fs, root = module._fs_for(uri)
    original = fs.open
    def read_only_open(key, mode="rb", **options):
        assert mode in {"rb", "r"}, (key, mode)
        return original(key, mode, **options)
    def refuse(*args, **kwargs):
        raise AssertionError("read-only source attempted a filesystem mutation")
    monkeypatch.setattr(fs, "open", read_only_open)
    for operation in ("makedirs", "rm", "mv", "mkdir"):
        monkeypatch.setattr(fs, operation, refuse)
    with closing(ObjectStore(uri)) as source, closing(MemoryStore()) as target:
        assert source.read_only and source.get("literal@1").edge_metric_exact[0] == Fraction(1, 7)
        assert [r.version for r in source.query(include_history=True, tags_any=["old"])] == [1]
        assert source.query(as_of=15.0)[0].meta["q"] == Fraction(2, 7)
        assert source.stats()["n_versions"] == 2
        assert source.commit_history("literal@1") == []
        plan = plan_legacy_migration(source, target)
        assert plan.source_backend == "object"
        assert LegacyMigrationPlan.from_bytes(plan.to_bytes()) == plan
        first = migrate_legacy_batch(source, target, plan, accept_loss=True, limit=1)
        assert not first.complete
        final = migrate_legacy_batch(source, target, plan, accept_loss=True)
        assert final.complete and len(final.receipts) == 2
        assert all(r.source_digest == r.destination_digest for r in final.receipts)
        assert migrate_legacy_batch(source, target, plan, accept_loss=True, limit=0).receipts == final.receipts


@pytest.mark.parametrize("operation", ["put", "prepared", "typed", "commit", "delete", "compact", "stage", "reclaim"])
def test_default_reader_refuses_all_publication_paths(uri, operation):
    written(uri)
    with closing(ObjectStore(uri)) as source:
        before = source._source_fingerprint()
        with pytest.raises(PermissionError, match="read-only"):
            if operation == "put": source.put("r", object())
            elif operation == "prepared": source.put_prepared("r", b"unused", {})
            elif operation == "typed": source.put_record("r", None)
            elif operation == "commit": source.commit_mutation("r", object())
            elif operation == "delete": source.delete("literal@1")
            elif operation == "compact": source.compact()
            elif operation == "stage": source._store_commit_bytes("r", 1, b"unused")
            else: source._delete_commit_bytes("literal@1", 1)
        assert source._source_fingerprint() == before


def test_missing_source_is_not_created(uri):
    from rcdb.objectstore import _fs_for
    fs, root = _fs_for(uri)
    with pytest.raises(ValueError, match="existing manifest"):
        ObjectStore(uri)
    assert not fs.exists(root)


@pytest.mark.parametrize("damage", ["unknown", "bool", "future", "format", "duplicate"])
def test_invalid_manifest_is_never_adopted_or_rewritten(uri, damage):
    written(uri)
    with closing(ObjectStore(uri)) as source:
        value = {"format": "rexdb-object", "version": 1}
        if damage == "unknown": value["provider"] = "untrusted.module"
        elif damage == "bool": value["version"] = True
        elif damage == "future": value["version"] = 2
        elif damage == "format": value["format"] = "other"
        raw = json.dumps(value).encode()
        if damage == "duplicate": raw = raw[:-1]+b',"version":1}'
        with source.fs.open(source._p("MANIFEST.json"), "wb") as stream: stream.write(raw)
        before = source._source_fingerprint()
    for options in ({}, {"read_only": False}):
        with pytest.raises(ValueError): ObjectStore(uri, **options)
    from rcdb.objectstore import _fs_for
    from rcdb.legacy import object_source_fingerprint
    fs, root = _fs_for(uri)
    assert object_source_fingerprint(fs, root) == before


def test_missing_published_payload_is_an_integrity_error_even_without_graph_verification(uri):
    written(uri)
    with closing(ObjectStore(uri)) as source:
        source.fs.rm(source._blob_key("literal@1", 2))
        with pytest.raises(ValueError, match="payload is missing"):
            source.get("literal@1", verify=False)
        with pytest.raises(ValueError, match="payload is missing"):
            source.read_record("literal@1")
        assert source.get("absent") is None


def test_stale_snapshot_cannot_be_planned_or_resume(uri):
    written(uri)
    with closing(ObjectStore(uri)) as source, closing(MemoryStore()) as target:
        plan = plan_legacy_migration(source, target)
        with closing(ObjectStore(uri, read_only=False)) as writer:
            writer.put("new", graph(), analytics=False)
        with pytest.raises(ValueError, match="reopen"):
            plan_legacy_migration(source, target)
        with pytest.raises(ValueError, match="reopen"):
            migrate_legacy_batch(source, target, plan, accept_loss=True)
        assert target.change_cursor.sequence == 0


@pytest.mark.parametrize("options", [{"unknown": 1}, {"include_history": True, "as_of": 15.0},
                                    {"limit": True}, {"limit": -1}, {"valid_at": float("nan")}])
def test_query_refuses_invalid_contract_even_on_empty_inventory(uri, options):
    with closing(ObjectStore(uri, read_only=False)): pass
    with closing(ObjectStore(uri)) as source:
        with pytest.raises((TypeError, ValueError)): source.query(**options)


def test_closed_handle_refuses_reads_and_writes(uri):
    written(uri)
    source = ObjectStore(uri)
    source.close(); source.close()
    for operation in (lambda: source.get("literal@1"), lambda: source.get_version("literal@1", 1),
                      source.list, source.stats, lambda: source.put("r", object()),
                      lambda: source._load_commit_bytes("literal@1", 1)):
        with pytest.raises(ValueError, match="closed"): operation()


def test_a_close_failure_after_journal_upload_poisoned_handle_requires_reopen(uri, monkeypatch):
    with closing(ObjectStore(uri, read_only=False)) as writer:
        writer.put("r", graph(), analytics=False)
        original = writer.fs.open
        @contextmanager
        def failure(key, mode="rb", **options):
            with original(key, mode, **options) as stream:
                yield stream
            if mode == "wb" and key.endswith("000000000002.json"):
                raise OSError("lost upload acknowledgement")
        monkeypatch.setattr(writer.fs, "open", failure)
        with pytest.raises(PublicationUncertainError, match="outcome is uncertain"):
            writer.put("s", graph(), analytics=False)
        with pytest.raises(PublicationUncertainError): writer.list()
        monkeypatch.setattr(writer.fs, "open", original)
    with closing(ObjectStore(uri)) as source:
        assert {r.id for r in source.list()} == {"r", "s"}


def test_unbound_object_layout_has_a_read_only_content_identity(uri):
    from rcdb.envelope import RecordEnvelope
    with closing(ObjectStore(uri, read_only=False)) as writer:
        row = writer.put("r", graph(), analytics=False)
        row.envelope = None
        blob_key = writer._blob_key("r", 1)
        with writer.fs.open(blob_key, "rb") as stream: _envelope, payload = RecordEnvelope.from_bytes(stream.read())
        with writer.fs.open(blob_key, "wb") as stream: stream.write(payload)
        with writer.fs.open(writer._p("journal", "000000000001.json"), "wb") as stream:
            stream.write(json.dumps({"op": "put", "id": "r", "record": row.to_storage_dict()}).encode())
        writer.fs.rm(writer._p(".rcdb-identity"))
    with closing(ObjectStore(uri)) as source:
        identity, before = source.store_id, source._source_fingerprint()
        assert source.get("r").nE == 1
        assert not source.fs.exists(source._p(".rcdb-identity"))
    with closing(ObjectStore(uri)) as source:
        assert source.store_id == identity and source._source_fingerprint() == before


def test_read_only_remote_identity_does_not_invoke_publication_provider(uri, monkeypatch):
    written(uri)
    from rcdb.objectstore import _fs_for
    import rcdb.objectstore as module
    fs, root = _fs_for(uri)
    class Remote:
        protocol = "test-cloud"
        def __getattr__(self, name): return getattr(fs, name)
    monkeypatch.setattr(module, "_fs_for", lambda uri: (Remote(), root))
    def provider(*args): raise AssertionError("read attempted identity publication")
    with closing(ObjectStore("test-cloud://private", identity_provider=provider)) as source:
        assert source.read_record("literal@1", version=1).record.meta["q"] == Fraction(2, 7)
        with closing(MemoryStore()) as target:
            plan = plan_legacy_migration(source, target)
            assert "private" not in repr(plan.as_record())
            assert migrate_legacy_batch(source, target, plan, accept_loss=True).complete


def test_bound_reader_refuses_identity_loss_without_republishing(uri):
    written(uri)
    with closing(ObjectStore(uri)) as source:
        source.fs.rm(source._p(".rcdb-identity"))
        before = source._source_fingerprint()
        fs, root = source.fs, source.root
    with pytest.raises(ValueError, match="identity is missing"): ObjectStore(uri)
    from rcdb.legacy import object_source_fingerprint
    assert not fs.exists(root+"/.rcdb-identity")
    assert object_source_fingerprint(fs, root) == before


@pytest.mark.parametrize("damage", ["version", "time"])
def test_replay_refuses_regressing_publications(uri, damage):
    written(uri)
    with closing(ObjectStore(uri)) as source:
        key = source._p("journal", "000000000002.json")
        with source.fs.open(key, "rb") as stream: change = json.loads(stream.read())
        if damage == "version":
            change["record"]["version"] = 1
            change["record"]["envelope"]["record_version"] = 1
        else: change["record"]["tx_from"] = 0.0
        with source.fs.open(key, "wb") as stream: stream.write(json.dumps(change).encode())
        before = source._source_fingerprint()
        fs, root = source.fs, source.root
    with pytest.raises(ValueError, match="regresses"): ObjectStore(uri)
    from rcdb.legacy import object_source_fingerprint
    assert object_source_fingerprint(fs, root) == before


def test_backdated_compatibility_write_refuses_before_payload_or_journal(uri):
    written(uri)
    with closing(ObjectStore(uri, read_only=False)) as writer:
        before = writer._source_fingerprint()
        with pytest.raises(ValueError, match="transaction time"):
            writer.put("literal@1", graph(), _tx_time=1.0, analytics=False)
        assert writer._source_fingerprint() == before


def test_short_payload_upload_never_becomes_published_metadata(uri, monkeypatch):
    with closing(ObjectStore(uri, read_only=False)) as writer:
        original = writer.fs.open
        class Short:
            def __init__(self, stream): self.stream = stream
            def write(self, raw): return self.stream.write(raw[:1])
        @contextmanager
        def truncated(key, mode="rb", **options):
            with original(key, mode, **options) as stream:
                yield Short(stream) if mode == "wb" and key.endswith(".safetensors") else stream
        monkeypatch.setattr(writer.fs, "open", truncated)
        with pytest.raises(OSError, match="short object record payload"):
            writer.put("r", graph(), analytics=False)
        assert writer.history("r") == []
        monkeypatch.setattr(writer.fs, "open", original)
        assert writer.put("r", graph(), analytics=False).version == 1


def test_failed_snapshot_upload_poisoned_handle_retains_journal(uri, monkeypatch):
    written(uri)
    with closing(ObjectStore(uri, read_only=False)) as writer:
        original = writer.fs.open
        @contextmanager
        def failure(key, mode="rb", **options):
            with original(key, mode, **options) as stream: yield stream
            if mode == "wb" and key.endswith("index.json"): raise OSError("lost snapshot acknowledgement")
        monkeypatch.setattr(writer.fs, "open", failure)
        with pytest.raises(PublicationUncertainError, match="snapshot write outcome"):
            writer.compact()
        assert len(writer.fs.ls(writer._p("journal"))) == 2
        with pytest.raises(PublicationUncertainError): writer.get("literal@1")
        monkeypatch.setattr(writer.fs, "open", original)
    with closing(ObjectStore(uri)) as source:
        assert [r.version for r in source.history("literal@1")] == [1, 2]


def test_stats_do_not_hide_a_damaged_durable_segment(uri):
    written(uri)
    with closing(ObjectStore(uri)) as source:
        with source.fs.open(source._p("journal", "000000000002.json"), "wb") as stream: stream.write(b"damaged")
        with pytest.raises(ValueError): source.stats()


def test_snapshot_byte_limit_is_enforced_without_rewriting(uri, monkeypatch):
    written(uri)
    with closing(ObjectStore(uri, read_only=False)) as writer: writer.compact()
    import rcdb.objectstore as module
    monkeypatch.setattr(module, "_SNAPSHOT_LIMIT", 8)
    with pytest.raises(ValueError, match="byte limit"): ObjectStore(uri)
