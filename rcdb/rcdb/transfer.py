"""Closed identity mappings for the existing cross store copy boundary.

Receipts name the actual published versions and Core object identities. They are
integrity records, not signatures or a claim that an entire migration was atomic.
No storage URI, credentials, provider name or executable object is serialized.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac

from rexgraph.value_codec import pack_value, unpack_value
from .envelope import _hex_identity

RECEIPT_MAGIC = b"RGRP1"
RECEIPT_LIMIT = 4*1024*1024
_DOMAIN = b"rexgraph-copy-receipt\x00"
_FIELDS = {"receipt_version", "source_store_id", "source_record_id", "source_version", "source_digest",
           "destination_store_id", "destination_record_id", "destination_version", "destination_digest"}


@dataclass(frozen=True)
class CopyReceipt:
    source_store_id: str
    source_record_id: str
    source_version: int
    source_digest: str
    destination_store_id: str
    destination_record_id: str
    destination_version: int
    destination_digest: str

    def __post_init__(self):
        for name in ("source_store_id", "destination_store_id"):
            _hex_identity(getattr(self, name), 32, "store identity")
        for name in ("source_digest", "destination_digest"):
            _hex_identity(getattr(self, name), 64, "object digest")
        for name in ("source_record_id", "destination_record_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError("receipt record identity requires nonempty literal text")
        for name in ("source_version", "destination_version"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 2**63:
                raise ValueError("receipt version requires a bounded positive native integer")

    def as_record(self):
        return {"receipt_version": 1, **asdict(self)}

    @classmethod
    def from_record(cls, record):
        if (type(record) is not dict or set(record) != _FIELDS
                or type(record["receipt_version"]) is not int or record["receipt_version"] != 1):
            raise ValueError("unknown copy receipt declaration")
        return cls(**{name: record[name] for name in _FIELDS-{"receipt_version"}})

    def _body(self):
        body = pack_value(self.as_record())
        if len(body) > RECEIPT_LIMIT-37:
            raise ValueError("copy receipt exceeds its byte limit")
        return body

    @property
    def digest(self):
        return hashlib.sha256(_DOMAIN+self._body()).hexdigest()

    def to_bytes(self):
        body = self._body()
        return RECEIPT_MAGIC+hashlib.sha256(_DOMAIN+body).digest()+body

    @classmethod
    def from_bytes(cls, payload):
        if (type(payload) is not bytes or not 37 < len(payload) <= RECEIPT_LIMIT
                or not payload.startswith(RECEIPT_MAGIC)):
            raise ValueError("invalid copy receipt frame")
        body = payload[37:]
        if not hmac.compare_digest(hashlib.sha256(_DOMAIN+body).digest(), payload[5:37]):
            raise ValueError("copy receipt digest mismatch")
        value = cls.from_record(unpack_value(body))
        if value._body() != body:
            raise ValueError("copy receipt body is not canonical")
        return value
