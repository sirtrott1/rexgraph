"""Closed payload frames bind addresses and reject reframed malformed declarations."""
from dataclasses import replace
import hashlib

import pytest

from rcdb.envelope import RecordEnvelope


def envelope(payload=b"native payload"):
    return RecordEnvelope("1"*32, "literal/id@1", 2, "RexGraph", "2"*64,
                          "rexgraph.safetensors", 1, hashlib.sha256(payload).hexdigest(), len(payload), "3"*64)


def test_native_envelope_roundtrip_retains_literal_address_and_codec():
    expected = envelope()
    restored, payload = RecordEnvelope.from_bytes(expected.to_bytes(b"native payload"))
    assert restored == expected and payload == b"native payload"
    assert restored.digest == expected.digest
    restored.check_address(store_id="1"*32, record_id="literal/id@1", record_version=2)


@pytest.mark.parametrize("change", [{"store_id": "3"*32}, {"record_id": "other"}, {"record_version": 1}])
def test_content_identity_does_not_override_a_different_storage_address(change):
    restored = replace(envelope(), **change)
    with pytest.raises(ValueError, match="binding"):
        restored.check_address(store_id="1"*32, record_id="literal/id@1", record_version=2)


@pytest.mark.parametrize("change", [{"record_version": True}, {"record_version": 0},
    {"codec_version": True}, {"payload_size": -1}, {"store_id": "X"*32}, {"object_digest": "x"*64}])
def test_envelope_coordinates_are_declared_without_coercion(change):
    with pytest.raises(ValueError):
        replace(envelope(), **change)


def test_metadata_payload_and_trailing_byte_corruption_are_refused():
    record = envelope()
    encoded = record.to_bytes(b"native payload")
    for bad in (encoded[:-1], encoded+b"x", encoded[:40]+bytes([encoded[40]^1])+encoded[41:],
                encoded[:-1]+bytes([encoded[-1]^1])):
        with pytest.raises(ValueError):
            RecordEnvelope.from_bytes(bad)
    with pytest.raises(ValueError, match="payload"):
        record.to_bytes(b"another payload")


def test_fully_reframed_unknown_header_field_is_refused():
    from rexgraph.value_codec import pack_value
    from rexgraph._binary import uint
    payload = b"native payload"
    header = pack_value({**envelope().as_record(), "unknown": True})
    body = uint(len(header))+header+payload
    with pytest.raises(ValueError, match="envelope"):
        RecordEnvelope.from_bytes(b"RGRE1"+hashlib.sha256(body).digest()+body)


@pytest.mark.parametrize("native_parser", [False, True])
def test_envelope_metadata_survives_index_and_both_journal_readers(native_parser, tmp_path, monkeypatch):
    from rcdb.core import ComplexRecord
    from rcdb import index
    if native_parser and index._read_frames is None:
        pytest.skip("compiled record log parser is not installed")
    if not native_parser:
        monkeypatch.setattr(index, "_read_frames", None)
    frame = envelope()
    record = ComplexRecord(frame.record_id, {"object_type": "RexGraph", "nV": 2, "nE": 1, "nF": 0},
                           created=7., version=frame.record_version, tx_from=7., envelope=frame)
    assert ComplexRecord.from_dict(record.to_storage_dict()).envelope == frame
    path = tmp_path / "index"
    index.write(path, index.build([(record.id, record)]))
    restored = index.record_at(index.read(path), 0)
    assert restored.to_dict() == record.to_dict()
    path = tmp_path / "journal"
    index.log_append(path, "put", record.id, record)
    output = list(index.log_read(path))
    assert len(output) == 1 and output[0][2].to_dict() == record.to_dict()


def test_record_metadata_cannot_claim_an_envelope_from_another_address():
    from rcdb.core import ComplexRecord
    with pytest.raises(ValueError, match="address"):
        ComplexRecord("different", {}, version=2, envelope=envelope())


@pytest.mark.parametrize("field,value", [("n", "0"), ("n_terms", "5"), ("digest", "")])
def test_native_index_metadata_is_checked_against_sealed_dimensions(field, value, tmp_path):
    from rcdb import ComplexRecord, index
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
    path = tmp_path / "index"
    index.write(path, index.build([("r", ComplexRecord("r", {"nV": 2}))]))
    with safe_open(str(path), framework="numpy") as opened:
        metadata = opened.metadata()
    metadata[field] = value
    save_file(load_file(str(path)), str(path), metadata=metadata)
    with pytest.raises(ValueError, match="header|digest"):
        index.read(path)
