"""Authenticated encryption and signing interfaces for RexGraph artifacts."""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Protocol

from .manifest import canonical_json

ENVELOPE_MAGIC = b"REXENC\x00"
ENVELOPE_VERSION = 1
_NONCE_BYTES = 12
_TAG_BYTES = 16

__all__ = [
    "ENVELOPE_MAGIC",
    "ENVELOPE_VERSION",
    "Ed25519Signer",
    "Ed25519Verifier",
    "EnvelopeInfo",
    "KeyProvider",
    "Signer",
    "StaticKeyProvider",
    "Verifier",
    "decrypt_bytes",
    "encrypt_bytes",
    "envelope_info",
]


class KeyProvider(Protocol):
    """Resolve encryption material by opaque key identity."""

    def key(self, key_id: str) -> bytes: ...


@dataclass(frozen=True)
class StaticKeyProvider:
    """Small in-process key provider intended for tests."""

    keys: dict[str, bytes]

    def key(self, key_id: str) -> bytes:
        try:
            value = self.keys[str(key_id)]
        except KeyError as exc:
            raise KeyError(f"unknown key id {key_id!r}") from exc
        return bytes(value)


@dataclass(frozen=True)
class EnvelopeInfo:
    """Public metadata carried by an authenticated encrypted envelope."""

    object_type: str
    key_id: str
    algorithm: str = "AES-256-GCM"
    version: int = ENVELOPE_VERSION


def _header(info: EnvelopeInfo, nonce: bytes) -> bytes:
    return canonical_json(
        {
            "algorithm": info.algorithm,
            "key_id": info.key_id,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "object_type": info.object_type,
            "version": int(info.version),
        }
    )


def _key(keys: KeyProvider, key_id: str) -> bytes:
    value = keys.key(key_id)
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("key providers must return bytes")
    value = bytes(value)
    if len(value) != 32:
        raise ValueError("AES-256-GCM requires a 32-byte key")
    return value


def _decode_header(blob: bytes, max_header: int) -> tuple[dict, bytes, int]:
    prefix = len(ENVELOPE_MAGIC)
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("encrypted envelopes must be bytes")
    blob = bytes(blob)
    if not blob.startswith(ENVELOPE_MAGIC) or len(blob) < prefix + 4:
        raise ValueError("not a RexGraph encrypted envelope")
    header_length = int.from_bytes(blob[prefix:prefix + 4], "big")
    header_end = prefix + 4 + header_length
    if header_length <= 0 or header_length > max_header or header_end > len(blob):
        raise ValueError("invalid encrypted envelope header length")
    try:
        header = blob[prefix + 4:header_end]
        data = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid encrypted envelope header") from exc
    if not isinstance(data, dict):
        raise ValueError("invalid encrypted envelope header")
    return data, header, header_end


def _parse_info(data: dict) -> tuple[EnvelopeInfo, bytes]:
    try:
        version = int(data.get("version", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid encrypted envelope version") from exc
    if version != ENVELOPE_VERSION:
        raise ValueError(f"unsupported encrypted envelope version {version}")
    if data.get("algorithm") != "AES-256-GCM":
        raise ValueError("unsupported encrypted envelope algorithm")
    object_type = data.get("object_type")
    key_id = data.get("key_id")
    if not isinstance(object_type, str) or not object_type:
        raise ValueError("invalid encrypted envelope object type")
    if not isinstance(key_id, str) or not key_id:
        raise ValueError("invalid encrypted envelope key id")
    encoded_nonce = data.get("nonce")
    if not isinstance(encoded_nonce, str):
        raise ValueError("invalid AES-GCM nonce")
    try:
        nonce = base64.b64decode(encoded_nonce, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid AES-GCM nonce") from exc
    if len(nonce) != _NONCE_BYTES:
        raise ValueError("invalid AES-GCM nonce")
    return EnvelopeInfo(object_type, key_id, "AES-256-GCM", version), nonce


def encrypt_bytes(
    payload: bytes,
    *,
    key_id: str,
    keys: KeyProvider,
    object_type: str = "bytes",
) -> bytes:
    """Encrypt one payload with AES-256-GCM and authenticated public metadata."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not isinstance(key_id, str) or not key_id:
        raise ValueError("key_id must be a nonempty string")
    if not isinstance(object_type, str) or not object_type:
        raise ValueError("object_type must be a nonempty string")
    nonce = os.urandom(_NONCE_BYTES)
    info = EnvelopeInfo(object_type, key_id)
    header = _header(info, nonce)
    ciphertext = AESGCM(_key(keys, key_id)).encrypt(nonce, bytes(payload), header)
    return ENVELOPE_MAGIC + len(header).to_bytes(4, "big") + header + ciphertext


def envelope_info(blob: bytes, *, max_header: int = 64 * 1024) -> EnvelopeInfo:
    """Read bounded public envelope metadata without decrypting the payload."""
    data, _header_bytes, _header_end = _decode_header(blob, max_header)
    info, _nonce = _parse_info(data)
    return info


def decrypt_bytes(
    blob: bytes,
    *,
    keys: KeyProvider,
    max_header: int = 64 * 1024,
) -> bytes:
    """Authenticate and decrypt one RexGraph encrypted envelope."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    data, header, header_end = _decode_header(blob, max_header)
    info, nonce = _parse_info(data)
    ciphertext = bytes(blob)[header_end:]
    if len(ciphertext) < _TAG_BYTES:
        raise ValueError("invalid encrypted envelope ciphertext")
    return AESGCM(_key(keys, info.key_id)).decrypt(nonce, ciphertext, header)


class Signer(Protocol):
    """Sign canonical bytes under an opaque signer identity."""

    @property
    def signer_id(self) -> str: ...

    def sign(self, payload: bytes) -> bytes: ...


class Verifier(Protocol):
    """Verify canonical bytes for one signer identity."""

    @property
    def signer_id(self) -> str: ...

    def verify(self, payload: bytes, signature: bytes) -> bool: ...


@dataclass(frozen=True)
class Ed25519Signer:
    """Ed25519 signer backed by a cryptography private key."""

    signer_id: str
    private_key: object

    @classmethod
    def generate(cls, signer_id: str) -> Ed25519Signer:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        return cls(str(signer_id), Ed25519PrivateKey.generate())

    def sign(self, payload: bytes) -> bytes:
        return bytes(self.private_key.sign(bytes(payload)))

    def verifier(self) -> Ed25519Verifier:
        return Ed25519Verifier(self.signer_id, self.private_key.public_key())


@dataclass(frozen=True)
class Ed25519Verifier:
    """Ed25519 verifier backed by a cryptography public key."""

    signer_id: str
    public_key: object

    def verify(self, payload: bytes, signature: bytes) -> bool:
        from cryptography.exceptions import InvalidSignature

        try:
            self.public_key.verify(bytes(signature), bytes(payload))
            return True
        except InvalidSignature:
            return False
