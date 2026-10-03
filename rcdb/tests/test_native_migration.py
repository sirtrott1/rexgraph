"""Native replay preserves the logical tape across independent store identities."""
from contextlib import closing
from dataclasses import replace
from fractions import Fraction
from itertools import product
import hashlib

import numpy as np
import pytest

from rcdb import (BlobCodecSpec, CopyReceipt, LocalStore, MemoryStore, NativeObjectStore, MigrationPlan, MigrationProgress,
                  MigrationStepReceipt, SQLStore, VersionConflictError, migrate_batch, plan_migration)
from rexgraph import RexGraph, TemporalRex
from rexgraph.object_identity import object_digest
from rexgraph.value_codec import pack_value


def store(kind, path):
    if kind == "memory": return MemoryStore(compression=BlobCodecSpec("zlib"))
    if kind == "local": return LocalStore(path, compression=BlobCodecSpec("zlib"))
    if kind.startswith("object-"):
        prefix = "memory://" if kind == "object-memory" else "file://"
        return NativeObjectStore(prefix+str(path), compression=BlobCodecSpec("zlib"))
    pytest.importorskip("sqlalchemy")
    return SQLStore(f"sqlite:///{path}.sqlite", compression=BlobCodecSpec("zlib"))


def graph(n=2):
    return RexGraph.from_graph(list(range(n-1)), list(range(1, n)), w_E=[Fraction(2, 7)]*(n-1))


def populate(source):
    source.put("literal/id@1", graph(), meta={"large": 2**90, "exact": Fraction(1, 7), "empty": ()},
               tags=["old"], valid_from=1.0, valid_to=80.0, analytics=False, _tx_time=10.0)
    source.put("gone", graph(3), analytics=False, _tx_time=10.0)
    source.put("literal/id@1", graph(3), tags=["new"], analytics=False, _tx_time=20.0)
    source.delete("literal/id@1", tx_time=20.0)
    source.delete("gone", tx_time=30.0)
    source.put("literal/id@1", graph(4), analytics=False, _tx_time=40.0)


@pytest.mark.parametrize("src_kind,dst_kind", list(product(("memory", "local", "sql", "object-file", "object-memory"), repeat=2)))
def test_native_paged_replay_keeps_deleted_history_versions_clocks_and_exact_state(src_kind, dst_kind, tmp_path):
    with closing(store(src_kind, tmp_path / "source")) as source, closing(store(dst_kind, tmp_path / "target")) as target:
        populate(source)
        plan = plan_migration(source, target)
        assert MigrationPlan.from_bytes(plan.to_bytes()) == plan
        progress, receipts = None, []
        while True:
            batch = migrate_batch(source, target, plan, progress=progress, limit=2)
            receipts.extend(batch.receipts)
            progress = MigrationProgress.from_bytes(batch.progress.to_bytes())
            if batch.complete: break
        assert target.store_id != source.store_id
        assert target.state_manifest() == source.state_manifest()
        assert target.state_digest() == source.state_digest()
        assert [r.operation for r in receipts] == ["put", "put", "put", "delete", "delete", "put"]
        assert target.next_version("gone") == 2 and target.get("gone") is None
        assert target.get("gone", as_of=29.0).nE == 2
        assert target.get("literal/id@1", as_of=20.0) is None
        assert target.get_record("literal/id@1").version == 3
        assert all(MigrationStepReceipt.from_bytes(r.to_bytes()) == r for r in receipts)
        assert all(r.copy is None or isinstance(r.copy, CopyReceipt) for r in receipts)
        saved = target.change_cursor
        assert migrate_batch(source, target, plan, progress=progress).receipts == ()
        assert target.change_cursor == saved


def test_pinned_migration_excludes_future_source_growth_and_resumes_from_actual_destination(tmp_path):
    with closing(store("sql", tmp_path / "source")) as source, closing(store("local", tmp_path / "target")) as target:
        populate(source)
        plan = plan_migration(source, target)
        first = migrate_batch(source, target, plan, limit=1)
        source.put("future", graph(), analytics=False, _tx_time=100.0)
        target.close()
        with closing(LocalStore(tmp_path / "target")) as reopened:
            result = migrate_batch(source, reopened, MigrationPlan.from_bytes(plan.to_bytes()), limit=100)
            assert result.complete and len(result.receipts) == 5
            assert reopened.change_cursor.sequence == plan.source_head.sequence == 6
            assert reopened.get("future") is None
            assert reopened.get_version("gone", 1).nE == 2
            assert first.progress.destination_cursor.sequence == 1


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_failure_after_published_copy_is_recovered_without_reusing_its_version(kind, tmp_path, monkeypatch):
    import rcdb.core as core
    with closing(MemoryStore()) as source, closing(store(kind, tmp_path / "target")) as target:
        populate(source)
        plan = plan_migration(source, target)
        original = core.copy_record
        def lost(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("receipt acknowledgement lost")
        with monkeypatch.context() as patch:
            patch.setattr(core, "copy_record", lost)
            with pytest.raises(OSError, match="acknowledgement lost"):
                migrate_batch(source, target, plan)
        assert target.change_cursor.sequence == 1
        result = migrate_batch(source, target, plan)
        assert result.complete and len(result.receipts) == 5
        assert target.state_manifest() == source.state_manifest()


def test_external_destination_change_cannot_be_mistaken_for_saved_progress():
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        populate(source)
        plan = plan_migration(source, target)
        first = migrate_batch(source, target, plan, limit=1)
        target.put("external", graph(), analytics=False)
        before = target.change_cursor
        with pytest.raises(ValueError, match="checked prefix differs"):
            migrate_batch(source, target, plan, progress=first.progress)
        assert target.change_cursor == before


def test_source_payload_substitution_is_refused_before_destination_publication():
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        source.put("r", graph(), analytics=False)
        plan = plan_migration(source, target)
        source._blobs[("r", 1)] = b"wrong"
        with pytest.raises(ValueError, match="content digest"):
            migrate_batch(source, target, plan)
        assert target.change_cursor.sequence == 0


def test_destination_payload_damage_cannot_be_hidden_by_matching_cursor_progress():
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        populate(source)
        plan = plan_migration(source, target)
        first = migrate_batch(source, target, plan, limit=1)
        target._blobs[("literal/id@1", 1)] = b"wrong"
        with pytest.raises(ValueError, match="content digest"):
            migrate_batch(source, target, plan, progress=first.progress)
        assert target.change_cursor.sequence == 1


def test_governed_replay_keeps_source_signature_and_builds_new_destination_commits():
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        source.commit_mutation("r", graph(), analytics=False, tx_time=10.0, actor="source")
        source.commit_mutation("r", graph(3), analytics=False, tx_time=20.0, actor="source")
        target.configure_security(require_commits=True)
        plan = plan_migration(source, target, actor="migration")
        result = migrate_batch(source, target, plan)
        assert result.complete and target.verify_commits("r")
        assert target.get_record("r").signature == source.get_record("r").signature
        assert target.get_record("r").tx_from == 20.0
        assert [p.link.digest for p in target.commit_history("r")] != [p.link.digest for p in source.commit_history("r")]


def test_temporal_and_prepared_signature_are_replayed_through_core_bytes():
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        value = TemporalRex([(np.array([0]), np.array([1])), (np.array([0, 1]), np.array([1, 2]))])
        source.put("temporal", value, analytics=False, _tx_time=10.0)
        from rcdb.engine import encode_native_payload
        signature = {"nV": 2, "nE": 1, "declared_exact": Fraction(3, 7)}
        source.put_prepared("prepared", encode_native_payload(graph(), BlobCodecSpec()), signature, _tx_time=20.0)
        result = migrate_batch(source, target, plan_migration(source, target))
        assert result.complete and object_digest(target.get("temporal")) == object_digest(value)
        assert target.get_record("prepared").signature == signature
        assert target.state_manifest() == source.state_manifest()


@pytest.mark.parametrize("damage", ["delete", "temporal", "projection", "occupied", "same-owner", "legacy"])
def test_incompatible_migration_plans_refuse_before_any_destination_write(damage, tmp_path):
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        source.put("r", graph(), meta={"kept": Fraction(1, 7)}, analytics=False)
        if damage == "delete":
            source.delete("r")
            target.configure_security(require_commits=True)
        elif damage == "temporal":
            source.put("time", TemporalRex([(np.array([0]), np.array([1]))]), analytics=False)
            target.configure_security(require_commits=True)
        elif damage == "projection": target.configure_security(metadata_fields=[])
        elif damage == "occupied": target.put("existing", graph(), analytics=False)
        elif damage == "same-owner": pass
        else:
            pytest.importorskip("sqlalchemy")
            with closing(SQLStore(f"sqlite:///{tmp_path / 'legacy.sqlite'}", native=False)) as legacy:
                with pytest.raises(ValueError, match="native record-state"):
                    plan_migration(legacy, target)
            return
        before = target.change_cursor
        if damage == "same-owner":
            with pytest.raises(ValueError, match="fresh native destination"):
                plan_migration(source, source)
        else:
            with pytest.raises(ValueError): plan_migration(source, target)
        assert target.change_cursor == before


def test_policy_change_and_forged_progress_refuse_without_advancing():
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        populate(source)
        plan = plan_migration(source, target)
        first = migrate_batch(source, target, plan, limit=1)
        forged = replace(first.progress, plan_digest="2"*64)
        with pytest.raises(ValueError, match="progress differs"):
            migrate_batch(source, target, plan, progress=forged)
        target.configure_security(require_commits=True)
        with pytest.raises(ValueError, match="policy differs"):
            migrate_batch(source, target, plan, progress=first.progress)
        assert target.change_cursor.sequence == 1


def test_conditional_copy_checks_cursor_after_source_selection(monkeypatch):
    from rcdb import copy_record
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        record = source.put("r", graph(), analytics=False)
        before = target.change_cursor
        real = source.read_record
        def selected(*args, **kwargs):
            result = real(*args, **kwargs)
            target.put("competing", graph(), analytics=False)
            return result
        monkeypatch.setattr(source, "read_record", selected)
        with pytest.raises(VersionConflictError, match="cursor changed"):
            copy_record(source, target, record, expected_cursor=before, preserve_signature=True)
        assert target.get("r") is None and target.change_cursor.sequence == 1


@pytest.mark.parametrize("kind", ["plan", "progress", "step"])
@pytest.mark.parametrize("damage", ["truncation", "hash", "extra", "bool-version", "trailing"])
def test_migration_artifacts_are_closed_canonical_and_domain_checked(kind, damage):
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        source.put("r", graph(), analytics=False)
        plan = plan_migration(source, target)
        batch = migrate_batch(source, target, plan, limit=1)
        value = {"plan": plan, "progress": batch.progress, "step": batch.receipts[0]}[kind]
        raw = value.to_bytes()
        if damage == "truncation": raw = raw[:-1]
        elif damage == "hash": raw = raw[:5]+b"0"*32+raw[37:]
        else:
            record = value.as_record()
            if damage == "extra": record["provider"] = "untrusted.module"
            elif damage == "bool-version": record["format_version"] = True
            body = pack_value(record)+(b"extra" if damage == "trailing" else b"")
            raw = value._magic+hashlib.sha256(value._domain+body).digest()+body
        with pytest.raises(ValueError): type(value).from_bytes(raw)


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_policy_changes_after_source_selection_refuse_before_publication(kind, tmp_path, monkeypatch):
    with closing(MemoryStore()) as source, closing(store(kind, tmp_path / "target")) as target:
        source.put("r", graph(), meta={"kept": 1}, analytics=False)
        target.configure_security(metadata_fields=["kept", "old"])
        plan = plan_migration(source, target)
        real = source.read_record
        def selected(*args, **kwargs):
            result = real(*args, **kwargs)
            # Same status/count, different allow list. Hash the fields themselves.
            target.configure_security(metadata_fields=["kept", "new"])
            return result
        monkeypatch.setattr(source, "read_record", selected)
        with pytest.raises(ValueError, match="policy differs"):
            migrate_batch(source, target, plan)
        assert target.change_cursor.sequence == 0


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_encrypted_destination_reencodes_source_under_its_own_compression(kind, tmp_path):
    pytest.importorskip("cryptography")
    from rexgraph.io.security import StaticKeyProvider
    keys = StaticKeyProvider({"dest": b"d"*32})
    with closing(MemoryStore()) as source, closing(store(kind, tmp_path / "target")) as target:
        populate(source)
        target.configure_security(key_id="dest", keys=keys)
        batch = migrate_batch(source, target, plan_migration(source, target))
        assert batch.complete and target.state_digest() == source.state_digest()
        assert target.header.compression != source.header.compression
        assert all(r.copy is None or r.copy.source_digest == r.copy.destination_digest for r in batch.receipts)


def test_resume_checks_governed_chain_and_declared_actor(monkeypatch):
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        source.put("r", graph(), analytics=False, _tx_time=10.0)
        source.put("r", graph(3), analytics=False, _tx_time=20.0)
        target.configure_security(require_commits=True)
        plan = plan_migration(source, target, actor="migration")
        migrate_batch(source, target, plan, limit=1)
        wrong_actor = replace(plan, actor="someone else")
        with pytest.raises(ValueError, match="declared actor"):
            migrate_batch(source, target, wrong_actor)
        monkeypatch.setattr(target, "verify_commits", lambda *args: False)
        with pytest.raises(ValueError, match="commit chain is invalid"):
            migrate_batch(source, target, plan)
        assert target.change_cursor.sequence == 1


def test_unpublished_destination_artifact_cannot_be_reused_for_native_migration():
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        source.put("r", graph(), analytics=False)
        plan = plan_migration(source, target)
        target._store_commit_bytes("r", 1, b"unpublished")
        with pytest.raises(ValueError, match="unpublished mutation artifact"):
            migrate_batch(source, target, plan)
        assert target.change_cursor.sequence == 0 and target._load_commit_bytes("r", 1) == b"unpublished"


@pytest.mark.parametrize("limit", [True, -1, 1001, 1.5])
def test_invalid_batch_limits_are_refused_without_publication(limit):
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        source.put("r", graph(), analytics=False)
        with pytest.raises(ValueError, match="batch limit"):
            migrate_batch(source, target, plan_migration(source, target), limit=limit)
        assert target.change_cursor.sequence == 0


def test_empty_migration_and_zero_limit_are_idempotent():
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        plan = plan_migration(source, target)
        empty = migrate_batch(source, target, plan, limit=0)
        assert empty.complete and empty.receipts == ()
        source.put("r", graph(), analytics=False)
        plan = plan_migration(source, target)
        assert not migrate_batch(source, target, plan, limit=0).complete
        assert target.change_cursor.sequence == 0


def test_forged_destination_genesis_is_refused():
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        source.put("r", graph(), analytics=False)
        plan = plan_migration(source, target)
        with pytest.raises(ValueError, match="genesis digest"):
            replace(plan, destination_start=replace(plan.destination_start, digest="f"*64))
        assert target.change_cursor.sequence == 0


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_governed_equal_tick_updates_keep_exact_versions_and_verify_after_resume(kind, tmp_path):
    with closing(MemoryStore()) as source, closing(store(kind, tmp_path / "target")) as target:
        source.put("r", graph(), analytics=False, _tx_time=10.0)
        source.put("r", graph(3), analytics=False, _tx_time=10.0)
        target.configure_security(require_commits=True)
        plan = plan_migration(source, target, actor="migration")
        migrate_batch(source, target, plan, limit=1)
        batch = migrate_batch(source, target, plan)
        assert batch.complete and target.verify_commits("r")
        assert [r.tx_from for r in target.history("r")] == [10.0, 10.0]
        assert target.get("r", as_of=10.0).nE == 2
        assert [object_digest(target.get_version("r", v)) for v in (1, 2)] == [
            object_digest(source.get_version("r", v)) for v in (1, 2)]
