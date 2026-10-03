"""One closed metadata contract across native indexes and JSON storage adapters.

JSON adapters carry framed native bytes; they do not reinterpret exact values.
Ordinary legacy JSON metadata remains readable. No object import or pickle path.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass

from rexgraph.value_codec import pack_value, unpack_value

_KEY = "rexgraph.record-value"
RECORD_MAGIC = b"RGRE1"
_RECORD_HEADER_LIMIT = 1024*1024
_RECORD_PAYLOAD_LIMIT = 256*1024*1024


def _hex_identity(value, size, name):
    if type(value) is not str or len(value) != size or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid record {name}")


@dataclass(frozen=True)
class RecordEnvelope:
    """Closed payload ownership declaration; content integrity is not authentication.

    The record engine supplies the expected store, record and version. A valid
    object at another storage address is still a refused substitution. Codec
    names are declarations resolved by installed providers, never import paths.
    """
    store_id: str
    record_id: str
    record_version: int
    object_type: str
    object_digest: str
    codec: str
    codec_version: int
    payload_digest: str
    payload_size: int
    metadata_digest: str

    def __post_init__(self):
        _hex_identity(self.store_id, 32, "store identity")
        _hex_identity(self.object_digest, 64, "object digest")
        _hex_identity(self.payload_digest, 64, "payload digest")
        _hex_identity(self.metadata_digest, 64, "metadata digest")
        if type(self.record_id) is not str or not self.record_id:
            raise ValueError("record identity must be nonempty text")
        if type(self.record_version) is not int or not 0 < self.record_version < 2**63:
            raise ValueError("record version must be a positive native integer")
        if (type(self.codec_version) is not int or self.codec_version <= 0
                or type(self.payload_size) is not int or not 0 <= self.payload_size <= _RECORD_PAYLOAD_LIMIT):
            raise ValueError("invalid record codec version or payload size")
        if any(type(value) is not str or not value for value in (self.object_type, self.codec)):
            raise ValueError("record object type and codec must be declared")

    def as_record(self):
        from dataclasses import asdict
        return {"envelope_version": 1, **asdict(self)}

    @property
    def digest(self):
        return hashlib.sha256(b"rexgraph-record-envelope\x00"+pack_value(self.as_record())).hexdigest()

    @classmethod
    def from_record(cls, record):
        expected = {"envelope_version", "store_id", "record_id", "record_version", "object_type",
                    "object_digest", "codec", "codec_version", "payload_digest", "payload_size", "metadata_digest"}
        if (type(record) is not dict or set(record) != expected
                or type(record["envelope_version"]) is not int or record["envelope_version"] != 1):
            raise ValueError("invalid native record envelope")
        return cls(**{key: record[key] for key in expected - {"envelope_version"}})

    def check_address(self, *, store_id, record_id, record_version):
        if (self.store_id, self.record_id, self.record_version) != (store_id, record_id, record_version):
            raise ValueError("record payload binding differs from its store, record or version")

    def check_payload(self, payload):
        if type(payload) is not bytes or len(payload) != self.payload_size:
            raise ValueError("record payload length differs from its envelope")
        if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), self.payload_digest):
            raise ValueError("record payload digest mismatch")

    def to_bytes(self, payload):
        from rexgraph._binary import uint
        self.check_payload(payload)
        header = pack_value(self.as_record())
        if len(header) > _RECORD_HEADER_LIMIT:
            raise ValueError("record envelope header exceeds its byte limit")
        framed_header = uint(len(header))+header
        checksum = hashlib.sha256(framed_header)
        checksum.update(payload)
        return RECORD_MAGIC+checksum.digest()+framed_header+payload

    @classmethod
    def from_bytes(cls, blob):
        from rexgraph._binary import Cursor
        if type(blob) is not bytes or len(blob) > _RECORD_PAYLOAD_LIMIT+_RECORD_HEADER_LIMIT+64:
            raise ValueError("record envelope exceeds its byte limit or is not bytes")
        cursor = Cursor(blob)
        if cursor.take(len(RECORD_MAGIC)) != RECORD_MAGIC:
            raise ValueError("not a native record envelope")
        expected = cursor.take(32)
        start = cursor.pos
        size = cursor.uint()
        if size <= 0 or size > _RECORD_HEADER_LIMIT:
            raise ValueError("invalid record envelope header length")
        if not hmac.compare_digest(hashlib.sha256(memoryview(blob)[start:]).digest(), expected):
            raise ValueError("record envelope content digest mismatch")
        envelope = cls.from_record(unpack_value(cursor.take(size)))
        if cursor.remaining != envelope.payload_size:
            raise ValueError("record envelope payload length mismatch")
        payload = cursor.take(envelope.payload_size)
        envelope.check_payload(payload)
        cursor.finish()
        return envelope, payload


def snapshot_mapping(value):
    if not isinstance(value, dict) or any(type(k) is not str for k in value):
        raise TypeError("record metadata and signature require string-keyed mappings")
    return unpack_value(pack_value(value))


def encode_mapping(value):
    payload = pack_value(snapshot_mapping(value))
    return {_KEY: {"version": 1, "bytes": payload.hex(), "digest": hashlib.sha256(payload).hexdigest()}}


def decode_mapping(value):
    if not isinstance(value, dict):
        raise ValueError("invalid record value mapping")
    if set(value) != {_KEY}:
        return snapshot_mapping(value)
    record = value[_KEY]
    if (not isinstance(record, dict) or set(record) != {"version", "bytes", "digest"}
            or type(record["version"]) is not int or record["version"] != 1
            or type(record["bytes"]) is not str or type(record["digest"]) is not str):
        raise ValueError("invalid native record value envelope")
    try:
        payload = bytes.fromhex(record["bytes"])
    except ValueError as exc:
        raise ValueError("invalid native record value bytes") from exc
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), record["digest"]):
        raise ValueError("record value content digest mismatch")
    return snapshot_mapping(unpack_value(payload))


def dumps_mapping(value):
    return "RGVM1:" + json.dumps(encode_mapping(value), sort_keys=True)


def loads_mapping(value):
    return (decode_mapping(json.loads(value[6:])) if value.startswith("RGVM1:")
            else snapshot_mapping(json.loads(value)))


def wire_mapping(value):
    """Keep ordinary JSON fields readable; frame values JSON cannot preserve.

    JSON numbers beyond the interoperable integer range, array dtypes, tuple
    identities and exact values use the same envelope as the storage adapters.
    The caller marks the record version so a user's reserved key stays data.
    """
    def safe(item):
        if item is None or type(item) in (bool, str):
            return True
        if type(item) is int:
            return abs(item) <= 2**53-1
        if type(item) is float:
            return math.isfinite(item)
        if type(item) is list:
            return all(safe(child) for child in item)
        if type(item) is dict:
            return all(type(key) is str and safe(child) for key, child in item.items())
        return False
    return value if safe(value) and set(value) != {_KEY} else encode_mapping(value)
