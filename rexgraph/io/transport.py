"""Framed transport packages for RexGraph native artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .manifest import canonical_json

MAGIC = b"REXPKG\x00"
FORMAT_VERSION = 1
MAX_HEADER = 1024 * 1024
_SHA256_HEX_LENGTH = 64

__all__ = [
    "FORMAT_VERSION",
    "MAGIC",
    "MAX_HEADER",
    "TransportInfo",
    "inspect",
    "pack",
    "unpack",
]


@dataclass(frozen=True)
class TransportInfo:
    """Bounded public metadata for one native transport package."""

    object_type: str
    payload_sha256: str
    payload_size: int
    version: int = FORMAT_VERSION


def _decode_header(blob: bytes, max_header: int) -> tuple[dict[str, Any], int]:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("transport packages must be bytes")
    blob = bytes(blob)
    prefix = len(MAGIC)
    if not blob.startswith(MAGIC) or len(blob) < prefix + 4:
        raise ValueError("not a RexGraph transport package")
    header_length = int.from_bytes(blob[prefix:prefix + 4], "big")
    header_end = prefix + 4 + header_length
    if header_length <= 0 or header_length > max_header or header_end > len(blob):
        raise ValueError("invalid transport header length")
    try:
        header = json.loads(blob[prefix + 4:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid transport header") from exc
    if not isinstance(header, dict):
        raise ValueError("invalid transport header")
    return header, header_end


def _transport_info(header: dict[str, Any]) -> TransportInfo:
    version = header.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("invalid transport version")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported transport version {version}")
    object_type = header.get("object_type")
    if not isinstance(object_type, str) or not object_type:
        raise ValueError("invalid transport object type")
    payload_sha256 = header.get("payload_sha256")
    if (
        not isinstance(payload_sha256, str)
        or len(payload_sha256) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in payload_sha256)
    ):
        raise ValueError("invalid transport payload digest")
    payload_size = header.get("payload_size")
    if (
        not isinstance(payload_size, int)
        or isinstance(payload_size, bool)
        or payload_size < 0
    ):
        raise ValueError("invalid transport payload size")
    metadata = header.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("invalid transport metadata")
    return TransportInfo(object_type, payload_sha256, payload_size, version)


def pack(
    payload: bytes,
    *,
    object_type: str,
    metadata: dict | None = None,
) -> bytes:
    """Frame an opaque payload with its digest and public metadata."""
    if not isinstance(object_type, str) or not object_type:
        raise ValueError("object_type must be a nonempty string")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a mapping")
    raw = bytes(payload)
    header = {
        "metadata": dict(metadata or {}),
        "object_type": object_type,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_size": len(raw),
        "version": FORMAT_VERSION,
    }
    header_bytes = canonical_json(header)
    if len(header_bytes) > MAX_HEADER:
        raise ValueError("transport header is too large")
    return MAGIC + len(header_bytes).to_bytes(4, "big") + header_bytes + raw


def inspect(blob: bytes, *, max_header: int = MAX_HEADER) -> TransportInfo:
    """Read bounded package metadata without decoding the payload."""
    header, _header_end = _decode_header(blob, max_header)
    return _transport_info(header)


def unpack(blob: bytes, *, verify: bool = True) -> tuple[bytes, dict]:
    """Return payload and header, rejecting truncation or digest mismatch."""
    raw = bytes(blob)
    header, header_end = _decode_header(raw, MAX_HEADER)
    info = _transport_info(header)
    payload = raw[header_end:]
    if len(payload) != info.payload_size:
        raise ValueError("transport payload length mismatch")
    if verify and hashlib.sha256(payload).hexdigest() != info.payload_sha256:
        raise ValueError("transport payload digest mismatch")
    return payload, header
