"""Operation, metadata, ownership and change chain integrity are one frame contract."""
from dataclasses import replace
from fractions import Fraction
import hashlib

import numpy as np
import pytest

from rcdb.core import ComplexRecord
from rcdb.journal import FRAME_MAGIC, GENESIS_DIGEST, JournalFrame
from rexgraph._binary import uint
from rexgraph.value_codec import pack_value


def sample():
    record = ComplexRecord("literal/id@1", {"nV": 2}, version=2**53+1,
                           meta={"empty": (), "rational": Fraction(1, 7), "array": np.array([2**90], object)})
    return JournalFrame.put(store_id="1"*32, sequence=2**53+3, previous=GENESIS_DIGEST,
                            record=record, extra=(-7, 2**62))


def test_exact_native_record_roundtrip_and_owned_metadata():
    original = sample()
    restored = JournalFrame.from_bytes(original.to_bytes())
    assert restored == original and restored.digest == original.digest
    assert restored.record.version == 2**53+1
    assert restored.record.meta["rational"] == Fraction(1, 7)
    assert restored.record.meta["empty"] == ()
    assert restored.record.meta["array"].dtype == object
    returned = restored.record
    returned.meta["array"][0] = 0
    assert restored.record.meta["array"][0] == 2**90


@pytest.mark.parametrize("changes", [{"operation": "other"}, {"operation": 1}, {"operation": []}, {"sequence": True},
    {"sequence": 0}, {"sequence": 2**63}, {"record_id": "other"}, {"extra": (True,)},
    {"extra": (2**63,)}, {"store_id": "x"*32}, {"previous": "X"*64}])
def test_invalid_declared_coordinates_and_unknown_operations_are_refused(changes):
    with pytest.raises(ValueError):
        replace(sample(), **changes)


def test_chain_verification_refuses_reorder_foreign_store_and_predecessor_change():
    first = sample()
    deleted = JournalFrame(first.store_id, first.sequence+1, first.digest, "delete", first.record_id)
    deleted.check_successor(store_id=first.store_id, sequence=first.sequence+1, previous=first.digest)
    assert deleted.record is None
    for coordinates in (("2"*32, first.sequence+1, first.digest),
                        (first.store_id, first.sequence, first.digest),
                        (first.store_id, first.sequence+1, GENESIS_DIGEST)):
        with pytest.raises(ValueError, match="chain"):
            deleted.check_successor(store_id=coordinates[0], sequence=coordinates[1], previous=coordinates[2])
    with pytest.raises(ValueError, match="delete"):
        replace(first, operation="delete")


def test_corrupt_truncated_and_trailing_frames_are_refused():
    raw = sample().to_bytes()
    for change in (raw[:-1], raw+b"x", raw[:-1]+bytes([raw[-1]^1]), b"RGJF2"+raw[5:]):
        with pytest.raises(ValueError):
            JournalFrame.from_bytes(change)


@pytest.mark.parametrize("change", [{"unknown": 1}, {"frame_version": True}, {"frame_version": 2}, {"operation": "other"}])
def test_reframing_invalid_declarations_does_not_make_them_valid(change):
    body = pack_value({**sample().as_record(), **change})
    raw = FRAME_MAGIC+uint(len(body))+hashlib.sha256(b"rexgraph-record-journal\x00"+body).digest()+body
    with pytest.raises(ValueError):
        JournalFrame.from_bytes(raw)


def test_a_bound_record_must_belong_to_the_journal_store():
    from rcdb import MemoryStore
    from rexgraph import RexGraph
    store = MemoryStore()
    try:
        record = store.put("r", RexGraph.from_graph([0], [1]), analytics=False)
        frame = JournalFrame.put(store_id=store.store_id, sequence=1, previous=GENESIS_DIGEST, record=record)
        assert frame.record.envelope == record.envelope
        with pytest.raises(ValueError, match="binding"):
            replace(frame, store_id="2"*32)
    finally:
        store.close()


def test_changed_bound_metadata_cannot_be_resealed_as_a_valid_change():
    from rcdb import MemoryStore
    from rexgraph import RexGraph
    store = MemoryStore()
    try:
        record = store.put("r", RexGraph.from_graph([0], [1]), analytics=False)
        record.meta["changed"] = True
        with pytest.raises(ValueError, match="metadata.*binding"):
            JournalFrame.put(store_id=store.store_id, sequence=1, previous=GENESIS_DIGEST, record=record)
    finally:
        store.close()
