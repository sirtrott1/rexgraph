"""Replay is the shared authority for version allocation, time and tombstones."""
from dataclasses import replace
from fractions import Fraction
import hashlib

import pytest

from rcdb import BlobCodecSpec, ChangeCursor, ComplexRecord, StoreHeader, StoreIdentity, StoreState
from rcdb.core import serialize_complex
from rcdb.engine import RecordChange, bind_record
from rcdb.journal import JournalFrame, LocalJournal
from rexgraph import RexGraph


@pytest.fixture
def state():
    return StoreState(StoreHeader(StoreIdentity("1"*32, "local")))


@pytest.fixture
def payload():
    return serialize_complex(RexGraph.from_graph([0], [1]))


def proposal(state, payload, rid="literal/id@1", time=10.0, **kwargs):
    record = ComplexRecord(rid, {"nV": 2}, meta={"exact": Fraction(1, 7), "empty": ()},
        version=state.next_version(rid), created=time, tx_from=time, valid_from=time)
    record, raw = bind_record(record, payload, store_id=state.header.identity.id)
    return state.prepare_put(record, blob_digest=hashlib.sha256(raw).hexdigest(), **kwargs)


def test_delete_and_recreate_preserve_history_addresses_and_monotone_change_cursor(state, payload):
    initial = state.cursor
    first = proposal(state, payload)
    state.apply(first)
    saved = state.cursor
    second = proposal(state, payload, time=20.0)
    state.apply(second)
    deleted = state.prepare_delete(first.record_id, 30.0)
    state.apply(deleted)
    assert state.current(first.record_id) is None
    assert state.next_version(first.record_id) == 3
    assert state.tombstone(first.record_id).version == 2
    assert [r.tx_to for r in state.history(first.record_id)] == [20.0, 30.0]
    assert state.records(as_of=25.0)[0].version == 2
    assert state.records(as_of=30.0) == []
    recreated = proposal(state, payload, time=40.0)
    state.apply(recreated)
    assert recreated.record.version == 3 and recreated.mutation.previous_version == 0
    assert state.tombstone(first.record_id) is None
    assert [f.operation for f in state.changes(initial)] == ["put", "put", "delete", "put"]
    assert [f.sequence for f in state.changes(saved)] == [2, 3, 4]
    restored = StoreState.replay(state.header, (JournalFrame.from_bytes(f.to_bytes()) for f in state.changes()))
    assert restored.cursor == state.cursor
    assert restored.current(first.record_id).version == 3
    assert restored.history(first.record_id)[0].meta["exact"] == Fraction(1, 7)


def test_refused_transition_leaves_previous_visible_state_and_cursor_intact(state, payload):
    first = proposal(state, payload)
    state.apply(first)
    saved = state.cursor
    candidate = proposal(state, payload, time=20.0)
    invalid = replace(candidate, mutation=replace(candidate.mutation, previous_version=0))
    with pytest.raises(ValueError, match="previous visible"):
        state.apply(invalid)
    assert state.cursor == saved and state.current(first.record_id).version == 1
    assert state.history(first.record_id)[0].tx_to is None


@pytest.mark.parametrize("operation", ["put", "delete", "recreate"])
def test_clock_rewind_refuses_across_every_lifecycle_transition(state, payload, operation):
    first = proposal(state, payload, time=20.0)
    state.apply(first)
    if operation == "recreate":
        state.apply(state.prepare_delete(first.record_id, 30.0))
    with pytest.raises(ValueError, match="precedes"):
        if operation == "delete":
            state.prepare_delete(first.record_id, 10.0)
        else:
            proposal(state, payload, time=10.0)


def test_equal_clock_ticks_still_allocate_different_versions_and_cursors(state, payload):
    for _ in range(3):
        state.apply(proposal(state, payload))
    assert state.cursor.sequence == 3
    assert [r.version for r in state.history("literal/id@1")] == [1, 2, 3]
    assert [r.tx_to for r in state.history("literal/id@1")] == [10.0, 10.0, None]
    assert state.records(as_of=10.0)[0].version == 3


def test_owning_records_and_change_frames_does_not_mutate_published_state(state, payload):
    state.apply(proposal(state, payload))
    record = state.current("literal/id@1")
    record.meta["exact"] = 0
    record.tx_to = 100.0
    state.changes()[0].record.signature["nV"] = 999
    assert state.current(record.id).meta["exact"] == Fraction(1, 7)
    assert state.current(record.id).signature["nV"] == 2
    assert state.current(record.id).tx_to is None


@pytest.mark.parametrize("change", ["owner", "header", "future", "digest", "byte_offset", "bool"])
def test_change_cursor_refuses_foreign_or_nonlogical_coordinates(state, payload, change):
    state.apply(proposal(state, payload))
    cursor = state.cursor
    if change == "owner":
        cursor = replace(cursor, store_id="2"*32)
    elif change == "header":
        cursor = replace(cursor, header_digest="2"*64)
    elif change == "future":
        cursor = replace(cursor, sequence=2)
    elif change == "digest":
        cursor = replace(cursor, digest="2"*64)
    elif change == "byte_offset":
        cursor = 100
    else:
        with pytest.raises(ValueError):
            replace(cursor, sequence=True)
        return
    with pytest.raises((ValueError, TypeError)):
        state.changes(cursor)


def test_cursor_closed_native_record_roundtrip(state, payload):
    state.apply(proposal(state, payload))
    assert ChangeCursor.from_record(state.cursor.as_record()) == state.cursor
    for field, value in (("extra", 1), ("cursor_version", True), ("cursor_version", 2)):
        with pytest.raises(ValueError):
            ChangeCursor.from_record({**state.cursor.as_record(), field: value})


def test_checked_provider_publishes_exact_prepared_frame_and_refuses_stale_head(state, payload, tmp_path):
    first = proposal(state, payload)
    journal = LocalJournal(tmp_path / "journal", store_id=state.header.identity.id)
    assert journal.publish(first) == first
    before = journal.path.read_bytes()
    with pytest.raises(ValueError, match="prepared transaction"):
        journal.publish(first)
    assert journal.path.read_bytes() == before
    state.apply(first)
    assert StoreState.replay(state.header, journal.frames()).cursor == state.cursor


def test_legacy_frames_require_explicit_state_migration_even_if_structurally_valid(state, payload):
    first = replace(proposal(state, payload), mutation=None)
    with pytest.raises(ValueError, match="engine configuration"):
        state.apply(first)


@pytest.mark.parametrize("mutation", ["version", "boolean", "float_version", "time", "int_time", "unknown", "format"])
def test_closed_change_declarations_refuse_semantic_coercion(state, payload, mutation):
    value = proposal(state, payload).mutation.as_record()
    if mutation == "version": value["version"] = 0
    elif mutation == "boolean": value["previous_version"] = True
    elif mutation == "float_version": value["version"] = 1.0
    elif mutation == "time": value["tx_time"] = float("nan")
    elif mutation == "int_time": value["tx_time"] = 10
    elif mutation == "unknown": value["extra"] = 1
    else: value["change_version"] = 2
    with pytest.raises(ValueError):
        RecordChange.from_record(value)


def test_header_codec_configuration_and_chain_predecessor_cannot_change_during_replay(state, payload):
    first = proposal(state, payload)
    changed = StoreHeader(state.header.identity, BlobCodecSpec("zlib"))
    with pytest.raises(ValueError, match="engine configuration"):
        StoreState.replay(changed, (first,))
    with pytest.raises(ValueError, match="chain"):
        state.apply(replace(first, sequence=2))
