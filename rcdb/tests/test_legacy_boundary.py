"""Legacy reads preserve source bytes; available history plans disclose their scope."""
from contextlib import closing
from fractions import Fraction
from itertools import product
import json
import shutil

import pytest

from rexgraph import RexGraph
from rexgraph.value_codec import pack_value
from rcdb import (BlobCodecSpec, FileStore, LegacyMigrationPlan, LocalStore, MemoryStore, ObjectStore, RexStore,
                  SQLStore, default_store, default_store_uri, migrate_legacy_batch, open_store,
                  plan_legacy_migration, recommend_backend, reset_default_store)
from rcdb.legacy import source_fingerprint


def graph(n=1):
    return RexGraph.from_graph(list(range(n)), list(range(1, n+1)), w_E=[Fraction(1, 7)]*n)


def legacy(kind, root, **options):
    if kind == "object":
        pytest.importorskip("fsspec")
        return ObjectStore(f"file://{root}", **options)
    return (FileStore if kind == "file" else RexStore)(root, **options)


def native(kind, root, **options):
    if kind == "memory": return MemoryStore(**options)
    if kind == "local": return LocalStore(root, **options)
    pytest.importorskip("sqlalchemy")
    return SQLStore(f"sqlite:///{root}.sqlite", **options)


def written(kind, root):
    with closing(legacy(kind, root, read_only=False)) as writer:
        writer.put("a@1", graph(), meta={"q": Fraction(2, 7), "positions": (None, 2**70)},
                   tags=["one"], analytics=False, _tx_time=10.0, valid_from=1.0, valid_to=9.0)
        writer.put("a@1", graph(2), meta={"q": Fraction(3, 7)}, tags=["two"], analytics=False, _tx_time=20.0)
        writer.put("deleted", graph(3), analytics=False, _tx_time=30.0)
        writer.delete("deleted")


@pytest.mark.parametrize("kind", ["file", "rex", "object"])
def test_read_only_open_read_verify_and_close_never_change_source(kind, tmp_path):
    root = tmp_path / kind
    written(kind, root)
    before = source_fingerprint(root)
    with closing(legacy(kind, root)) as source:
        assert source.read_only
        assert source.get("a@1").nE == 2
        assert source.read_record("a@1", version=1).record.meta["q"] == Fraction(2, 7)
        assert source.store_id
        assert source.state_manifest()["records"]
        assert source.commit_history("a@1") == []
        assert source.verify_commits("a@1")
        assert {r.id for r in source.query()} == {"a@1"}
    assert source_fingerprint(root) == before
    assert kind == "object" or not (root / "commits").exists()


@pytest.mark.parametrize("kind,operation", list(product(["file", "rex", "object"], ["put", "prepared", "typed", "commit", "delete", "compact", "derived"])))
def test_legacy_read_only_refuses_every_publication_before_touching_bytes(kind, operation, tmp_path):
    root = tmp_path / kind
    written(kind, root)
    before = source_fingerprint(root)
    with closing(legacy(kind, root)) as source:
        with pytest.raises(PermissionError, match="read-only"):
            if operation == "put": source.put("a", object())
            elif operation == "prepared": source.put_prepared("a", b"invalid", {})
            elif operation == "typed": source.put_record("a", None)
            elif operation == "commit": source.commit_mutation("a", object())
            elif operation == "delete": source.delete("a@1")
            elif operation == "compact": source.compact()
            elif kind == "file": source.recompress()
            elif kind == "rex": source.write_index()
            else: source._store_commit_bytes("a", 1, b"unused")
    assert source_fingerprint(root) == before


@pytest.mark.parametrize("kind", ["file", "rex", "object"])
def test_read_only_missing_source_is_not_initialized(kind, tmp_path):
    root = tmp_path / "missing"
    with pytest.raises(ValueError, match="existing regular directory|existing manifest"):
        legacy(kind, root)
    assert not root.exists()


def test_unowned_json_layout_gets_content_identity_without_writing(tmp_path):
    root = tmp_path / "old"
    with closing(FileStore(root, read_only=False)) as writer:
        writer.put("r", graph(), analytics=False)
        raw = writer.get_record("r").to_dict()
        raw.pop("envelope")
        blob = writer._read_blob("r", 1)
        from rcdb.envelope import RecordEnvelope
        _envelope, payload = RecordEnvelope.from_bytes(blob)
    (root / "index.rexlog").unlink()
    (root / ".rcdb-identity").unlink()
    (root / "index.json").write_text(json.dumps({"r": [raw]}))
    (root / "blobs" / "r@1.safetensors").write_bytes(payload)
    before = source_fingerprint(root)
    with closing(FileStore(root)) as source:
        owner = source.store_id
        assert source.read_record("r").value.nE == 1
        source.state_manifest()
    duplicate = tmp_path / "duplicate"
    shutil.copytree(root, duplicate)
    with closing(FileStore(duplicate)) as source:
        assert source.store_id == owner
    assert source_fingerprint(root) == before
    assert not (root / ".rcdb-identity").exists()


def test_version_gaps_and_large_legacy_versions_have_exact_receipt_mapping(tmp_path):
    from rcdb import ComplexRecord, serialize_complex
    root = tmp_path / "old"
    (root / "blobs").mkdir(parents=True)
    versions = (2**53+1, 2**53+9)
    rows = []
    for n, version in enumerate(versions, start=1):
        row = ComplexRecord("literal@1", signature={"object_type": "RexGraph", "tags": ["old"]},
                            meta={"q": Fraction(n, 7)}, version=version, created=float(n), tx_from=float(n),
                            tx_to=None if n == 2 else 2.0, valid_from=None)
        rows.append(row.to_storage_dict())
        (root / "blobs" / f"literal%401@{version}.safetensors").write_bytes(serialize_complex(graph(n)))
    (root / "index.json").write_text(json.dumps({"literal@1": rows}))
    with closing(FileStore(root)) as src, closing(MemoryStore()) as dst:
        plan = plan_legacy_migration(src, dst)
        result = migrate_legacy_batch(src, dst, plan, accept_loss=True, limit=1)
        assert not result.complete
        result = migrate_legacy_batch(src, dst, plan, accept_loss=True)
        assert [(r.source_version, r.destination_version) for r in result.receipts] == list(zip(versions, (1, 2), strict=True))
        assert result.complete and dst.get("literal@1").nE == 2


def test_reading_old_commit_filename_does_not_rename_it(tmp_path):
    root = tmp_path / "old"
    with closing(FileStore(root, read_only=False)) as writer:
        writer.commit_mutation("a/b", graph(), analytics=False)
        current = writer._commit_path("a/b", 1)
        old = writer._legacy_commit_path("a/b", 1)
    from pathlib import Path
    Path(current).rename(old)
    before = source_fingerprint(root)
    with closing(FileStore(root)) as source:
        assert len(source.commit_history("a/b")) == 1
        assert source.verify_commits("a/b")
    assert source_fingerprint(root) == before
    assert Path(old).exists() and not Path(current).exists()


@pytest.mark.parametrize("kind", ["file", "rex", "object"])
def test_a_stale_reader_cannot_plan_an_incomplete_available_inventory(kind, tmp_path):
    root = tmp_path / "old"
    written(kind, root)
    with closing(legacy(kind, root)) as source:
        with closing(legacy(kind, root, read_only=False)) as writer:
            writer.put("new", graph(), analytics=False)
        with closing(MemoryStore()) as target:
            with pytest.raises(ValueError, match="reopen"):
                plan_legacy_migration(source, target)
            assert target.change_cursor.sequence == 0
    with closing(legacy(kind, root)) as source, closing(MemoryStore()) as target:
        assert len(plan_legacy_migration(source, target).records) == 3


@pytest.mark.parametrize("source_kind,target_kind", list(product(["file", "rex", "object"], ["memory", "local", "sql"])))
def test_available_history_is_pinned_exact_resumable_and_explicit_about_loss(source_kind, target_kind, tmp_path):
    root, target = tmp_path / "old", tmp_path / "new"
    written(source_kind, root)
    before = source_fingerprint(root)
    with closing(legacy(source_kind, root)) as src, closing(native(target_kind, target)) as dst:
        plan = plan_legacy_migration(src, dst)
        assert LegacyMigrationPlan.from_bytes(plan.to_bytes()) == plan
        assert len(plan.records) == 2
        assert "deleted_or_compacted_history_not_reconstructed" in plan.limitations
        with pytest.raises(ValueError, match="accept_loss"):
            migrate_legacy_batch(src, dst, plan)
        assert dst.change_cursor.sequence == 0
        first = migrate_legacy_batch(src, dst, plan, accept_loss=True, limit=1)
        assert first.copied_versions == 1 and not first.complete
        # The saved batch is optional: restart verifies the actual published prefix.
        if target_kind != "memory":
            dst.close()
            dst = native(target_kind, target)
        try:
            final = migrate_legacy_batch(src, dst, LegacyMigrationPlan.from_bytes(plan.to_bytes()), accept_loss=True)
            assert final.complete and final.copied_versions == 2 and len(final.receipts) == 2
            assert final.as_record()["scope"] == "available_history"
            for receipt in final.receipts:
                assert receipt.source_digest == receipt.destination_digest
                assert dst.read_record(receipt.destination_record_id, version=receipt.destination_version).state_digest == receipt.source_digest
            assert dst.get("deleted") is None
            assert pack_value(dst.get_record("a@1").meta) == pack_value(src.get_record("a@1").meta)
            assert dst.read_record("a@1", version=1).record.meta["q"] == Fraction(2, 7)
            assert dst.history("a@1")[0].tx_from != 10.0
            again = migrate_legacy_batch(src, dst, plan, accept_loss=True)
            assert again.receipts == final.receipts and dst.change_cursor.sequence == 2
        finally:
            dst.close()
    assert source_fingerprint(root) == before


@pytest.mark.parametrize("kind,damage", list(product(["file", "rex", "object"], ["source", "policy", "extra", "payload", "missing"])))
def test_pinned_plan_refuses_changed_source_policy_and_invalid_destination_prefix(kind, damage, tmp_path):
    root = tmp_path / "old"
    written(kind, root)
    with closing(legacy(kind, root)) as src, closing(MemoryStore()) as dst:
        plan = plan_legacy_migration(src, dst)
        migrate_legacy_batch(src, dst, plan, accept_loss=True, limit=1)
        if damage == "source": (root / "unexpected").write_bytes(b"changed")
        elif damage == "policy": dst.configure_security(metadata_fields=[])
        elif damage == "extra": dst.delete("a@1")
        elif damage == "payload": dst._blobs[("a@1", 1)] = b"damaged"
        else: dst._blobs.pop(("a@1", 1))
        cursor = dst.change_cursor
        with pytest.raises(ValueError):
            migrate_legacy_batch(src, dst, plan, accept_loss=True)
        assert dst.change_cursor == cursor


@pytest.mark.parametrize("kind", ["file", "rex", "object"])
def test_preflight_refuses_missing_payload_and_lossy_projection_before_first_write(kind, tmp_path):
    root = tmp_path / "old"
    written(kind, root)
    with closing(legacy(kind, root)) as src, closing(MemoryStore()) as dst:
        dst.configure_security(metadata_fields=[])
        with pytest.raises(ValueError, match="policy"):
            plan_legacy_migration(src, dst)
        assert dst.change_cursor.sequence == 0
    if kind == "file": (root / "blobs" / "a%401@1.safetensors").unlink()
    elif kind == "rex": (root / "blobs.pack").write_bytes(b"")
    else: (root / "blobs" / "a%401@1.safetensors").unlink()
    with closing(MemoryStore()) as dst:
        with pytest.raises(ValueError):
            with closing(legacy(kind, root)) as src:
                plan_legacy_migration(src, dst)
        assert dst.change_cursor.sequence == 0


@pytest.mark.parametrize("kind", ["file", "rex", "object"])
def test_governed_available_history_issues_new_destination_commits(kind, tmp_path):
    root = tmp_path / "old"
    written(kind, root)
    with closing(legacy(kind, root)) as src, closing(MemoryStore().configure_security(require_commits=True)) as dst:
        plan = plan_legacy_migration(src, dst, actor="migration")
        result = migrate_legacy_batch(src, dst, plan, accept_loss=True)
        assert result.complete and dst.verify_commits("a@1")
        assert len(dst.commit_history("a@1")) == 2
        assert all(p.transition.actor == "migration" for p in dst.commit_history("a@1"))
        assert migrate_legacy_batch(src, dst, plan, accept_loss=True, limit=0).receipts == result.receipts


def test_canonical_factory_options_reopen_and_legacy_discovery(tmp_path, monkeypatch):
    root = tmp_path / "rcdb"
    with closing(open_store(f"auto://{root}", compression=BlobCodecSpec("zlib"))) as store:
        assert isinstance(store, LocalStore)
        store.put_record("value", Fraction(1, 7))
    assert recommend_backend(str(root))["backend"] == "local"
    with closing(open_store(f"auto://{root}")) as store:
        assert store.get("value") == Fraction(1, 7)
    monkeypatch.delenv("REXGRAPH_RCDB_URI", raising=False)
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    reset_default_store()
    try:
        assert default_store_uri() == f"auto://{root}"
        assert default_store().get("value") == Fraction(1, 7)
    finally:
        reset_default_store()
    (root / "index.log").write_bytes(b"")
    before = source_fingerprint(root)
    with pytest.raises(ValueError, match="conflicting"):
        open_store(f"auto://{root}")
    assert source_fingerprint(root) == before


@pytest.mark.parametrize("kind", ["file", "rex", "object"])
def test_default_existing_legacy_store_opens_read_only_without_orphaning_data(kind, tmp_path, monkeypatch):
    root = tmp_path / "rcdb"
    written(kind, root)
    before = source_fingerprint(root)
    monkeypatch.delenv("REXGRAPH_RCDB_URI", raising=False)
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    reset_default_store()
    try:
        store = default_store()
        assert store.backend == kind and store.read_only
        assert store.get("a@1").nE == 2
        with pytest.raises(PermissionError): store.put("new", graph())
    finally:
        reset_default_store()
    assert source_fingerprint(root) == before


@pytest.mark.parametrize("damage", ["extra", "loss", "duplicate", "version-bool", "digest", "future"])
def test_legacy_plan_is_closed_and_cannot_erase_loss_declarations(damage, tmp_path):
    root = tmp_path / "old"
    written("file", root)
    with closing(FileStore(root)) as src, closing(MemoryStore()) as dst:
        plan = plan_legacy_migration(src, dst)
    record = plan.as_record()
    if damage == "extra": record["provider"] = "uninstalled.module"
    elif damage == "loss": record["limitations"] = ()
    elif damage == "duplicate": record["records"] = (record["records"][0],)*2
    elif damage == "version-bool": record["records"][0]["source_version"] = True
    elif damage == "digest": record["source_fingerprint"] = "G"*64
    else: record["format_version"] = 2
    with pytest.raises(ValueError): LegacyMigrationPlan.from_record(record)
    with pytest.raises(ValueError): LegacyMigrationPlan.from_bytes(plan.to_bytes()+b"extra")
