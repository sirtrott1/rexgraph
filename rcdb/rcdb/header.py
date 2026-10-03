"""Closed store ownership and explicit codec configuration for the record engine.

Codec references are declarations. Reading this header never imports a provider
named by persisted data; the engine resolves it against installed capabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac

from rexgraph._binary import Cursor, uint
from rexgraph.value_codec import pack_value, unpack_value

from .engine import COMPLEX_CODEC, COMPLEX_CODEC_VERSION
from .envelope import _RECORD_PAYLOAD_LIMIT
from .store_identity import StoreIdentity

HEADER_MAGIC = b"RGSH1"
HEADER_LIMIT = 16*1024
_DOMAIN = b"rexgraph-store-header\x00"
_FIELDS = {"header_version", "engine_version", "identity", "compression", "record_codecs", "journal_version", "envelope_version"}


@dataclass(frozen=True)
class CodecRef:
    name: str
    version: int

    def __post_init__(self):
        if type(self.name) is not str or not self.name or len(self.name.encode("utf-8")) > 256 or "\x00" in self.name:
            raise ValueError("codec reference requires a bounded literal name")
        if type(self.version) is not int or not 0 < self.version < 2**31:
            raise ValueError("codec reference version requires a positive native integer")

    def as_record(self):
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_record(cls, record):
        if type(record) is not dict or set(record) != {"name", "version"}:
            raise ValueError("unknown codec reference declaration")
        return cls(**record)


# This inventory is explicit and independent of installed optional packages.
# Old headers retain their serialized tuple; a new codec requires a new header/store.
DEFAULT_RECORD_CODECS = (CodecRef(COMPLEX_CODEC, COMPLEX_CODEC_VERSION),
    CodecRef("rexgraph.value", 1), CodecRef("rexgraph.provenance", 1),
    CodecRef("rexgraph.dataset-declaration", 1), CodecRef("rcdb.copy-receipt", 1),
    CodecRef("rcdb.migration-plan", 1), CodecRef("rcdb.migration-progress", 1),
    CodecRef("rcdb.migration-step", 1), CodecRef("rcql.query-result", 1))


def record_codec_inventory(value):
    value = DEFAULT_RECORD_CODECS if value is None else value
    if (type(value) is not tuple or not value or any(not isinstance(ref, CodecRef) for ref in value)
            or len(set(value)) != len(value)):
        raise ValueError("store requires unique ordered CodecRef declarations")
    return value


@dataclass(frozen=True)
class BlobCodecSpec:
    """Deterministic compression choice; defaults never inspect optional installs."""
    name: str = "none"
    level: int | None = None
    version: int = 1

    def __post_init__(self):
        if type(self.name) is not str or self.name not in {"none", "zlib", "zstd"}:
            raise ValueError("unknown store blob codec")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported store blob codec version")
        if self.level is not None and (type(self.level) is not int or not -(2**31) <= self.level < 2**31):
            raise ValueError("store compression level requires a native int32 value")
        if self.name == "none":
            if self.level is not None:
                raise ValueError("uncompressed store blobs cannot declare a level")
        else:
            if self.level is None:
                object.__setattr__(self, "level", 6 if self.name == "zlib" else 3)
            if self.name == "zlib" and not 0 <= self.level <= 9:
                raise ValueError("invalid store zlib compression level")

    def as_record(self):
        return {"name": self.name, "version": self.version, "level": self.level}

    @classmethod
    def from_record(cls, record):
        if type(record) is not dict or set(record) != {"name", "version", "level"}:
            raise ValueError("unknown store blob codec declaration")
        value = cls(**record)
        if value.as_record() != record:
            raise ValueError("store blob codec defaults must be explicit in sealed records")
        return value

    def encode(self, raw):
        if type(raw) is not bytes or len(raw) > _RECORD_PAYLOAD_LIMIT:
            raise ValueError("store blob requires bounded native payload bytes")
        from .core import compress_blob
        encoded = compress_blob(raw, self.level, codec=self.name)
        if len(encoded) > _RECORD_PAYLOAD_LIMIT:
            raise ValueError("encoded store blob exceeds its byte limit")
        return encoded

    def decode(self, raw, *, max_output_bytes=_RECORD_PAYLOAD_LIMIT):
        """Refuse a different compression choice before dispatching a provider."""
        if (type(max_output_bytes) is not int or not 0 < max_output_bytes <= _RECORD_PAYLOAD_LIMIT
                or type(raw) is not bytes or len(raw) > _RECORD_PAYLOAD_LIMIT):
            raise ValueError("store blob requires bounded native payload bytes")
        if self.name == "none":
            if len(raw) > max_output_bytes:
                raise ValueError("store blob exceeds its decoded byte limit")
            return raw
        from .core import _BLOB_MAGIC, _CODEC_ZLIB, _CODEC_ZSTD, decompress_blob
        expected = {"zlib": _CODEC_ZLIB, "zstd": _CODEC_ZSTD}.get(self.name)
        if not raw.startswith(_BLOB_MAGIC+expected):
            raise ValueError("blob compression differs from its store header")
        return decompress_blob(raw, max_output_bytes=max_output_bytes)


@dataclass(frozen=True)
class StoreHeader:
    """Immutable engine configuration; mutable counters belong to durable state."""
    identity: StoreIdentity
    compression: BlobCodecSpec = BlobCodecSpec()
    record_codecs: tuple[CodecRef, ...] = DEFAULT_RECORD_CODECS

    def __post_init__(self):
        if not isinstance(self.identity, StoreIdentity) or not isinstance(self.compression, BlobCodecSpec):
            raise TypeError("store header requires declared identity and codec values")
        if (type(self.record_codecs) is not tuple or not self.record_codecs
                or any(not isinstance(value, CodecRef) for value in self.record_codecs)
                or len(set(self.record_codecs)) != len(self.record_codecs)):
            raise ValueError("store header requires unique ordered codec references")

    def check_record_codec(self, name, version):
        if CodecRef(name, version) not in self.record_codecs:
            raise ValueError("record codec is not declared by this store header")

    def as_record(self):
        return {"header_version": 1, "engine_version": 1,
                "identity": {"id": self.identity.id, "backend": self.identity.backend},
                "compression": self.compression.as_record(),
                "record_codecs": tuple(value.as_record() for value in self.record_codecs),
                "journal_version": 1, "envelope_version": 1}

    def _body(self):
        body = pack_value(self.as_record())
        if len(body) > HEADER_LIMIT:
            raise ValueError("store header exceeds its byte limit")
        return body

    @property
    def digest(self):
        return hashlib.sha256(_DOMAIN+self._body()).hexdigest()

    def to_bytes(self):
        body = self._body()
        return HEADER_MAGIC+uint(len(body))+hashlib.sha256(_DOMAIN+body).digest()+body

    @classmethod
    def from_record(cls, record):
        if type(record) is not dict or set(record) != _FIELDS:
            raise ValueError("unknown store header declaration")
        for name in ("header_version", "engine_version", "journal_version", "envelope_version"):
            if type(record[name]) is not int or record[name] != 1:
                raise ValueError(f"unsupported store {name}")
        identity = record["identity"]
        if type(identity) is not dict or set(identity) != {"id", "backend"}:
            raise ValueError("unknown store ownership declaration")
        if type(record["record_codecs"]) is not tuple:
            raise ValueError("store codec references require a native ordered tuple")
        return cls(StoreIdentity(**identity), BlobCodecSpec.from_record(record["compression"]),
                   tuple(CodecRef.from_record(value) for value in record["record_codecs"]))

    @classmethod
    def from_bytes(cls, raw):
        if type(raw) is not bytes or len(raw) > HEADER_LIMIT+64:
            raise ValueError("store header exceeds its byte limit or is not bytes")
        cursor = Cursor(raw)
        if cursor.take(len(HEADER_MAGIC)) != HEADER_MAGIC:
            raise ValueError("unsupported store header format")
        size = cursor.uint()
        if not 0 < size <= HEADER_LIMIT:
            raise ValueError("invalid store header length")
        expected = cursor.take(32)
        body = cursor.take(size)
        cursor.finish()
        if not hmac.compare_digest(expected, hashlib.sha256(_DOMAIN+body).digest()):
            raise ValueError("store header digest mismatch")
        return cls.from_record(unpack_value(body))
