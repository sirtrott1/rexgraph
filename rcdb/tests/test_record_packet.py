"""Portable selected records keep exact semantics across the one copy boundary."""
from contextlib import closing
from fractions import Fraction
from itertools import product
import hashlib

import numpy as np
import pytest

from rexgraph import Absent, RexGraph, TemporalRex
from rexgraph.value_codec import pack_value, unpack_value
from rcdb import (BlobCodecSpec, CodecRef, LocalStore, MemoryStore, RecordCodec,
                  RecordPacket, SQLStore, copy_record, record_packet, register_record_codec,
                  unregister_record_codec)
from rcdb.header import DEFAULT_RECORD_CODECS


def opened(kind, path, **options):
    if kind == "memory": return MemoryStore(**options)
    if kind == "local": return LocalStore(path, **options)
    pytest.importorskip("sqlalchemy")
    return SQLStore(f"sqlite:///{path}.sqlite", **options)


def exact_value():
    return {"q": Fraction(1, 7), "big": 2**90, "absent": Absent,
            "tuple": (1, None), "array": np.array([[1, 2]], dtype=np.int16)}


@pytest.mark.parametrize("source_kind,target_kind", list(product(("memory", "local", "sql"), repeat=2)))
def test_selected_exact_value_copies_to_a_different_literal_address(source_kind, target_kind, tmp_path):
    with closing(opened(source_kind, tmp_path/"source", compression=BlobCodecSpec("zlib"))) as source:
        first = source.put_record("literal/id@2", exact_value(), meta={"q": Fraction(2, 7)},
                                  tags=["exact"], valid_from=1.0, valid_to=9.0, tx_time=10.0)
        source.put_record(first.id, None, tx_time=20.0)
        packet = record_packet(source, first.id, version=1)
        packet = RecordPacket.from_bytes(packet.to_bytes())
        assert packet.record.tx_to == 20.0 and packet.source_store_id == source.store_id
        assert packet.snapshot().state_digest == first.envelope.object_digest
        assert pack_value(packet.snapshot().value) == pack_value(exact_value())
        with closing(opened(target_kind, tmp_path/"target", compression=BlobCodecSpec("zlib"))) as target:
            receipt = copy_record(packet.source(), target, packet.record,
                                  destination_id="other/literal@3", return_receipt=True, preserve_signature=True)
            assert (receipt.source_record_id, receipt.source_version) == (first.id, 1)
            assert (receipt.destination_record_id, receipt.destination_version) == ("other/literal@3", 1)
            assert receipt.source_digest == receipt.destination_digest == packet.state_digest
            assert target.read_record(first.id) is None
            selected = target.read_record(receipt.destination_record_id)
            assert pack_value(selected.value) == pack_value(exact_value())
            assert selected.record.meta == {"q": Fraction(2, 7)}
            assert selected.record.signature == packet.record.signature
            assert (selected.record.valid_from, selected.record.valid_to) == (1.0, 9.0)


@pytest.mark.parametrize("value", [None, Absent, False, 0, (), [], {}])
def test_missing_records_and_stored_empty_values_are_distinct(value):
    with closing(MemoryStore()) as source:
        assert record_packet(source, "missing") is None
        source.put_record("r", value)
        packet = record_packet(source, "r")
        assert pack_value(RecordPacket.from_bytes(packet.to_bytes()).snapshot().value) == pack_value(value)


@pytest.mark.parametrize("temporal", [False, True])
def test_graph_and_temporal_packets_use_core_identity_and_fresh_governance(temporal):
    value = (TemporalRex([(np.array([0]), np.array([1])), (np.array([0, 1]), np.array([1, 2]))])
             if temporal else RexGraph.from_graph([0], [1]))
    with closing(MemoryStore()) as source, closing(MemoryStore()) as target:
        source.put("r", value, analytics=False)
        packet = record_packet(source, "r")
        assert packet.snapshot().state_digest == source.read_record("r").state_digest
        if temporal:
            assert packet.cell_counts()["T"] == 2
        else:
            target.configure_security(require_commits=True)
        receipt = copy_record(packet.source(), target, packet.record, destination_id="new", return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest
        if not temporal: assert target.verify_commits("new")


def test_admission_uses_payload_counts_before_graph_materialization(monkeypatch):
    with closing(MemoryStore()) as store:
        from rcdb.engine import encode_native_payload
        store.put_prepared("r", encode_native_payload(RexGraph.from_graph([0, 1], [1, 2]), BlobCodecSpec()),
                           {"nV": 0, "nE": 0, "nF": 0})
        packet = record_packet(store, "r")
    import rcdb.engine as engine
    def rebuilt(*args, **kwargs): raise AssertionError("rebuilt before admission")
    monkeypatch.setattr(engine, "rebuild_record", rebuilt)
    def refused(counts):
        assert counts == {"nV": 3, "nE": 2, "nF": 0}
        raise RuntimeError("refused size")
    with pytest.raises(RuntimeError, match="refused size"):
        packet.snapshot(check_counts=refused)


def test_selected_identity_ignores_physical_container_order_but_names_metadata():
    with closing(MemoryStore()) as source:
        source.put("r", RexGraph.from_graph([0, 1], [1, 2]), analytics=False, meta={"a": Fraction(1, 7)})
        identities = {record_packet(source, "r").selection_digest for _ in range(10)}
        assert len(identities) == 1
        source.put("r", source.get("r"), analytics=False, meta={"a": Fraction(2, 7)})
        assert record_packet(source, "r").selection_digest not in identities


def test_snapshot_mutation_and_owner_substitution_are_refused():
    with closing(MemoryStore()) as source:
        source.put_record("r", exact_value())
        snapshot = source.read_record("r")
        with pytest.raises(ValueError, match="binding"):
            RecordPacket.from_snapshot(snapshot, source_store_id="f"*32)
        snapshot.record.meta["new"] = 1
        with pytest.raises(ValueError, match="published binding"):
            RecordPacket.from_snapshot(snapshot, source_store_id=source.store_id)
        snapshot = source.read_record("r")
        snapshot.value["q"] = Fraction(2, 7)
        with pytest.raises(ValueError, match="changed"):
            RecordPacket.from_snapshot(snapshot, source_store_id=source.store_id)


def test_encrypted_storage_exports_only_plain_portable_state():
    pytest.importorskip("cryptography")
    from rexgraph.io.security import StaticKeyProvider
    with closing(MemoryStore(compression=BlobCodecSpec("zlib"))) as source:
        source.configure_security(key_id="secret-key", keys=StaticKeyProvider({"secret-key": b"k"*32}))
        source.put_record("r", exact_value())
        packet = record_packet(source, "r")
        assert b"secret-key" not in packet.to_bytes()
        assert pack_value(packet.snapshot().value) == pack_value(exact_value())


@pytest.mark.parametrize("mutation", ["flip", "append", "truncate", "version", "extra", "time", "meta"])
def test_packet_tampering_and_closed_metadata_are_refused(mutation):
    from rexgraph._binary import uint
    with closing(MemoryStore()) as source:
        source.put_record("r", exact_value())
        packet = record_packet(source, "r")
    raw = packet.to_bytes()
    if mutation == "flip": raw = raw[:-1]+bytes([raw[-1]^1])
    elif mutation == "append": raw += b"x"
    elif mutation == "truncate": raw = raw[:-1]
    else:
        data = packet.record.to_dict()
        if mutation == "version": data["version"] = True
        if mutation == "extra": data["unrecognized"] = 1
        if mutation == "time": data["created"] = 1
        if mutation == "meta": data["meta"]["forged"] = 1
        metadata = pack_value(data)
        body = uint(len(metadata))+metadata+packet.blob
        raw = b"RGRX1"+hashlib.sha256(b"rexgraph-record-packet\x00\x01"+body).digest()+body
    with pytest.raises((ValueError, TypeError)):
        RecordPacket.from_bytes(raw)


def test_packet_bound_and_unknown_provider_refuse_without_publication(monkeypatch):
    import rcdb.packet as module
    reference = CodecRef("uninstalled.literal", 1)
    codec = RecordCodec(reference, "TestValue", pack_value, unpack_value)
    register_record_codec(codec)
    try:
        with closing(MemoryStore(record_codecs=(*DEFAULT_RECORD_CODECS, reference))) as source:
            source.put_record("r", exact_value(), codec=reference)
            packet = record_packet(source, "r")
        unregister_record_codec(reference)
        with closing(MemoryStore()) as target:
            with pytest.raises(ValueError, match="installed capability"):
                copy_record(packet.source(), target, packet.record)
            assert target.change_cursor.sequence == 0
        monkeypatch.setattr(module, "PACKET_LIMIT", 100)
        with pytest.raises(ValueError, match="byte limit"):
            RecordPacket.from_bytes(packet.to_bytes())
    finally:
        unregister_record_codec(reference)
