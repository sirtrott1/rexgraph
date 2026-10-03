"""Checked journals distinguish complete corruption from an explicitly repairable tail."""
from concurrent.futures import ThreadPoolExecutor
import os

import pytest

from rcdb import ComplexRecord, PublicationUncertainError
from rcdb.journal import LocalJournal, TornJournalError


def row(name="r", version=1):
    return ComplexRecord(name, {"nV": 2}, version=version)


def test_checked_log_roundtrip_and_exact_byte_cursor(tmp_path):
    journal = LocalJournal(tmp_path / "log", store_id="1"*32)
    first = journal.append("put", "r", row(), extra=(17, 23))
    cursor = journal.path.stat().st_size
    second = journal.append("put", "r", row(version=2))
    third = journal.append("delete", "r")
    assert list(journal.frames()) == [first, second, third]
    assert list(journal.frames(cursor)) == [second, third]
    status = journal.inspect()
    assert status.frame_count == status.last_sequence == 3 and not status.torn
    assert status.valid_end == status.size
    with pytest.raises(ValueError, match="cursor|block"):
        list(journal.frames(cursor-1))
    with pytest.raises(ValueError, match="another store"):
        list(LocalJournal(journal.path, store_id="2"*32).frames())


@pytest.mark.parametrize("missing", [1, 8, 13, 31])
def test_torn_tail_is_reported_then_explicitly_repaired(missing, tmp_path):
    journal = LocalJournal(tmp_path / "log", store_id="1"*32)
    first = journal.append("put", "r", row())
    cursor = journal.path.stat().st_size
    journal.append("put", "r", row(version=2))
    raw = journal.path.read_bytes()
    journal.path.write_bytes(raw[:-missing])
    assert list(journal.frames(allow_torn_tail=True)) == [first]
    with pytest.raises(TornJournalError) as failure:
        list(journal.frames())
    assert failure.value.valid_end == cursor
    status = journal.inspect()
    assert status.torn and status.frame_count == 1 and status.valid_end == cursor
    with pytest.raises(ValueError):
        journal.append("put", "s", row("s"))
    before = journal.repair_torn_tail()
    assert before.torn and journal.path.stat().st_size == cursor
    assert not journal.inspect().torn
    frame = journal.append("put", "s", row("s"))
    assert frame.sequence == 2 and frame.previous == first.digest


def test_corruption_is_not_reclassified_as_a_repairable_tail(tmp_path):
    journal = LocalJournal(tmp_path / "log", store_id="1"*32)
    journal.append("put", "r", row())
    raw = bytearray(journal.path.read_bytes())
    raw[-10] ^= 1
    journal.path.write_bytes(raw)
    for operation in (lambda: list(journal.frames(allow_torn_tail=True)), journal.inspect, journal.repair_torn_tail):
        with pytest.raises(ValueError, match="digest"):
            operation()
    assert journal.path.read_bytes() == raw


def test_a_corrupted_length_and_footer_are_refused_before_interpretation(tmp_path):
    journal = LocalJournal(tmp_path / "log", store_id="1"*32)
    journal.append("put", "r", row())
    raw = journal.path.read_bytes()
    from rcdb.journal import _read_header
    with journal.path.open("rb") as stream:
        _identity, end = _read_header(stream)
    for offset in (end, len(raw)-1):
        bad = bytearray(raw)
        bad[offset] ^= 1
        journal.path.write_bytes(bad)
        with pytest.raises(ValueError, match="length|footer"):
            journal.inspect()
        with pytest.raises(ValueError):
            journal.repair_torn_tail()
        assert journal.path.read_bytes() == bad


def test_reordered_complete_frames_fail_the_sequence_chain(tmp_path):
    from rcdb.journal import _block, _header
    journal = LocalJournal(tmp_path / "log", store_id="1"*32)
    first = journal.append("put", "r", row())
    second = journal.append("delete", "r")
    journal.path.write_bytes(_header("1"*32)+_block(second)+_block(first))
    with pytest.raises(ValueError, match="chain"):
        list(journal.frames())
    with pytest.raises(ValueError, match="chain"):
        journal.repair_torn_tail()


def test_new_journal_requires_identity_and_refuses_symbolic_link(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        LocalJournal(tmp_path / "log").append("put", "r", row())
    assert not (tmp_path / "log").exists()
    journal = LocalJournal(tmp_path / "target", store_id="1"*32)
    journal.append("put", "r", row())
    link = tmp_path / "link"
    link.symlink_to(journal.path)
    with pytest.raises(ValueError, match="symbolic"):
        LocalJournal(link).append("delete", "r")


@pytest.mark.parametrize("exists", [False, True])
def test_nonzero_resume_from_unpublished_content_refuses(exists, tmp_path):
    path = tmp_path / "log"
    if exists:
        path.touch()
    with pytest.raises(ValueError, match="cursor"):
        list(LocalJournal(path).frames(1))


def test_exact_numpy_coordinates_are_normalized_without_float_coercion(tmp_path):
    import numpy as np
    journal = LocalJournal(tmp_path / "log", store_id="1"*32)
    frame = journal.append("put", "r", row(), extra=np.array([17, 23], np.int64))
    assert frame.extra == (17, 23)
    for value in (True, np.bool_(True), 1.5, np.float64(1), 2**63):
        with pytest.raises(ValueError, match="coordinates"):
            journal.append("put", "r", row(), extra=[value])
    assert journal.inspect().frame_count == 1


def test_failed_sync_rolls_back_before_retry(tmp_path, monkeypatch):
    journal = LocalJournal(tmp_path / "log", store_id="1"*32)
    journal.append("put", "r", row())
    before = journal.path.read_bytes()
    sync = os.fsync
    calls = []
    def fail_once(descriptor):
        calls.append(descriptor)
        if len(calls) == 1:
            raise OSError("sync failed")
        sync(descriptor)
    monkeypatch.setattr(os, "fsync", fail_once)
    with pytest.raises(OSError, match="sync failed"):
        journal.append("put", "s", row("s"))
    assert journal.path.read_bytes() == before
    assert journal.append("put", "s", row("s")).sequence == 2


def test_failed_rollback_sync_reports_uncertain_publication(tmp_path, monkeypatch):
    journal = LocalJournal(tmp_path / "log", store_id="1"*32)
    def unavailable(descriptor):
        raise OSError("device unavailable")
    monkeypatch.setattr(os, "fsync", unavailable)
    with pytest.raises(PublicationUncertainError, match="rollback failed"):
        journal.append("put", "r", row())


def test_append_coordinates_are_arbitrated_across_thread_handles(tmp_path):
    path = tmp_path / "log"
    def append(index):
        return LocalJournal(path, store_id="1"*32).append("put", str(index), row(str(index)))
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(append, range(24)))
    assert {frame.sequence for frame in results} == set(range(1, 25))
    assert LocalJournal(path, store_id="1"*32).inspect().frame_count == 24


def _append_process(path, group):
    journal = LocalJournal(path, store_id="1"*32)
    for index in range(4):
        name = f"{group}:{index}"
        journal.append("put", name, row(name))


def test_append_coordinates_are_arbitrated_across_posix_processes(tmp_path):
    import multiprocessing
    if os.name != "posix" or "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("the qualified process lock profile uses POSIX fork")
    path = tmp_path / "log"
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_append_process, args=(path, group)) for group in range(4)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        frames = list(LocalJournal(path, store_id="1"*32).frames())
        assert [frame.sequence for frame in frames] == list(range(1, 17))
        assert len({frame.record_id for frame in frames}) == 16
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
