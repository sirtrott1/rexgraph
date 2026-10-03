"""The local store writers publish the shared checked journal contract."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rexgraph import RexGraph
from rcdb import ComplexRecord, FileStore, RexStore, index
from rcdb.journal import JOURNAL_MAGIC, LocalJournal, TornJournalError


def graph():
    return RexGraph.from_graph([0], [1])


def opened(kind, root):
    return (FileStore if kind == "file" else RexStore)(str(root), read_only=False)


def journal_path(store):
    return Path(store._log_path if store.backend == "file" else store._records_path)


@pytest.mark.parametrize("kind", ["file", "rex"])
def test_local_stores_write_checked_owned_changes_and_reopen(kind, tmp_path):
    store = opened(kind, tmp_path)
    store.put("r", graph(), analytics=False)
    store.put("r", graph(), analytics=False)
    store.put("s", graph(), analytics=False)
    store.delete("s")
    path, identity = journal_path(store), store.store_id
    assert path.read_bytes().startswith(JOURNAL_MAGIC)
    frames = list(LocalJournal(path, store_id=identity).frames())
    assert [frame.sequence for frame in frames] == [1, 2, 3, 4]
    assert [frame.operation for frame in frames] == ["put", "put", "put", "delete"]
    assert all(frame.record is None or frame.record.envelope.store_id == identity for frame in frames)
    store.close()
    store = opened(kind, tmp_path)
    assert [record.version for record in store.history("r")] == [1, 2]
    assert store.get("r").nE == 1 and store.get("s") is None
    assert store.store_id == identity
    store.close()


@pytest.mark.parametrize("kind", ["file", "rex"])
def test_torn_store_open_refuses_until_explicit_journal_repair(kind, tmp_path):
    store = opened(kind, tmp_path)
    store.put("r", graph(), analytics=False)
    path, identity = journal_path(store), store.store_id
    boundary = path.stat().st_size
    store.put("s", graph(), analytics=False)
    store.close()
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(TornJournalError) as failure:
        opened(kind, tmp_path)
    assert failure.value.valid_end == boundary
    assert LocalJournal(path, store_id=identity).repair_torn_tail().torn
    store = opened(kind, tmp_path)
    assert store.get("r").nE == 1 and store.get("s") is None
    store.put("t", graph(), analytics=False)
    assert LocalJournal(path, store_id=identity).inspect().last_sequence == 2
    store.close()


@pytest.mark.parametrize("kind", ["file", "rex"])
def test_complete_store_journal_corruption_is_not_skipped(kind, tmp_path):
    store = opened(kind, tmp_path)
    store.put("r", graph(), analytics=False)
    path = journal_path(store)
    store.close()
    raw = bytearray(path.read_bytes())
    raw[-10] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="digest"):
        opened(kind, tmp_path)
    assert path.read_bytes() == raw


def test_checked_snapshot_anchor_resumes_and_survives_compaction(tmp_path):
    store = RexStore(str(tmp_path), read_only=False)
    store.put("r", graph(), analytics=False)
    store.write_index()
    from rcdb.rexstore import RexIndex
    snapshot = RexIndex(store._index_path)
    assert snapshot.open()
    journal = LocalJournal(store._records_path, store_id=store.store_id)
    assert snapshot.log_anchor == journal.anchor()
    store.put("s", graph(), analytics=False)
    store.close()
    store = RexStore(str(tmp_path), read_only=False)
    assert store._tail_count == 1
    assert {record.id for record in store.list()} == {"r", "s"}
    store.delete("r")
    store.compact()
    store.close()
    store = RexStore(str(tmp_path), read_only=False)
    assert store._tail_count == 0 and store.get("s").nE == 1
    assert store.get("r") is None
    store.close()


def test_unanchored_legacy_snapshot_cannot_hide_checked_journal_frames(tmp_path):
    from rcdb.rexstore import RexIndex
    store = RexStore(str(tmp_path), read_only=False)
    row = store.put("r", graph(), analytics=False)
    path = journal_path(store)
    at = dict(store._blob_at)
    # A prior format's byte cursor is not an anchor for the new journal.
    RexIndex.write(store._index_path, {"r": [row]}, at, path.stat().st_size)
    store.put("s", graph(), analytics=False)
    store.close()
    store = RexStore(str(tmp_path), read_only=False)
    assert store._index is None and store._tail_count == 2
    assert {record.id for record in store.list()} == {"r", "s"}
    store.close()


def test_missing_full_record_journal_cannot_be_replaced_by_derived_snapshot(tmp_path):
    store = RexStore(str(tmp_path), read_only=False)
    store.put("r", graph(), analytics=False)
    store.write_index()
    path = journal_path(store)
    store.close()
    path.unlink()
    with pytest.raises(ValueError, match="journal is missing"):
        RexStore(str(tmp_path), read_only=False)


def test_binary_migration_preserves_metadata_history_and_reports_digests(tmp_path):
    import hashlib
    from fractions import Fraction
    path = tmp_path / "log"
    row = ComplexRecord("r", {}, meta={"ratio": Fraction(1, 7), "integer": 2**80+1})
    index.legacy_log_append(path, "put", "r", row, extra=[17, 23])
    index.legacy_log_append(path, "delete", "r")
    raw = path.read_bytes()
    receipt = index.migrate_legacy_log(path, store_id="1"*32)
    assert receipt["converted"] and receipt["frame_count"] == 2
    assert receipt["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["target_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    frames = list(LocalJournal(path, store_id="1"*32).frames())
    assert frames[0].record.meta == row.meta and frames[0].extra == (17, 23)
    assert frames[1].operation == "delete"
    assert not index.migrate_legacy_log(path, store_id="1"*32)["converted"]


@pytest.mark.parametrize("bad", ["torn", "opcode"])
def test_migration_failure_keeps_original_bytes(bad, tmp_path):
    path = tmp_path / "log"
    index.legacy_log_append(path, "put", "r", ComplexRecord("r", {}))
    raw = bytearray(path.read_bytes())
    if bad == "torn":
        raw = raw[:-1]
    else:
        raw[len(index.LOG_MAGIC)] = 99
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        index.migrate_legacy_log(path, store_id="1"*32)
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]


def test_new_and_compatibility_writers_cannot_mix_framing(tmp_path):
    path = tmp_path / "log"
    row = ComplexRecord("r", {})
    index.log_append(path, "put", "r", row, store_id="1"*32)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="format"):
        index.legacy_log_append(path, "put", "r", row)
    assert path.read_bytes() == before


def test_concurrent_migration_and_checked_appends_keep_every_change(tmp_path):
    path = tmp_path / "log"
    index.legacy_log_append(path, "put", "old", ComplexRecord("old", {}))
    def append(number):
        name = str(number)
        index.log_append(path, "put", name, ComplexRecord(name, {}), store_id="1"*32)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(append, range(18)))
    frames = list(LocalJournal(path, store_id="1"*32).frames())
    assert len(frames) == 19 and {frame.record_id for frame in frames} == {"old", *map(str, range(18))}
    assert [frame.sequence for frame in frames] == list(range(1, 20))


@pytest.mark.parametrize("kind", ["file", "rex"])
def test_legacy_binary_backend_migrates_before_its_next_write(kind, tmp_path):
    store = opened(kind, tmp_path)
    row = store.put("r", graph(), analytics=False)
    path = journal_path(store)
    extra = store._blob_at[("r", 1)] if kind == "rex" else None
    store.close()
    path.unlink()
    index.legacy_log_append(path, "put", "r", row, extra=extra)
    store = opened(kind, tmp_path)
    store.put("s", graph(), analytics=False)
    assert path.read_bytes().startswith(JOURNAL_MAGIC)
    store.close()
    store = opened(kind, tmp_path)
    assert store.get("r").nE == store.get("s").nE == 1
    store.close()


@pytest.mark.parametrize("extra", [None, (-1, 1), (0, 2**62), (0, 1, 3), (0, 1, 0x52585331, 1)])
def test_complete_rex_changes_refuse_invalid_backend_coordinates(extra, tmp_path):
    from rcdb.journal import write_checked_journal
    store = RexStore(str(tmp_path), read_only=False)
    record = store.put("r", graph(), analytics=False)
    path, identity = journal_path(store), store.store_id
    store.close()
    with path.open("wb") as stream:
        write_checked_journal(stream, identity, [("put", "r", record, extra)])
    with pytest.raises(ValueError, match="coordinates|token count"):
        RexStore(str(tmp_path), read_only=False)
