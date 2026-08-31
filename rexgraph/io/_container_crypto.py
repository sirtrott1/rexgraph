"""Authenticated, indexed envelopes shared by Rex container backends.

The core never accepts key bytes.  A caller supplies an opaque property object
that owns its KMS/envelope state and implements authenticated ``seal``/``open``
actions.  This module supplies the container framing, policy validation,
canonical associated data, and chunk inventory; it does not implement a cipher
or import the Agent package.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import secrets
import struct
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ._compat import dumps as _dumps

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "ENCRYPTION_METADATA_KEY",
    "ContainerDecryptionProperties",
    "ContainerEncryptionConfig",
    "ContainerEncryptionError",
    "ContainerEncryptionProperties",
    "encrypted_metadata",
    "open_encrypted_manifest",
    "protect_tensors",
    "read_protected_tensor",
    "validate_storage_inventory",
]


_FORMAT = "rexgraph-indexed-aead"
_VERSION = 1
DEFAULT_CHUNK_SIZE = 1 << 20
ENCRYPTION_METADATA_KEY = "rex_encryption"
_STORAGE_PREFIX = "__rex_encrypted_storage__/"
_MANIFEST_AAD = b"rexgraph-container-manifest-v1\0"
_CHUNK_AAD = b"rexgraph-container-chunk-v1\0"
_STATISTICS_AAD = b"rexgraph-container-statistics-v1\0"
_STATISTICS_MAGIC = b"RXS1"
_STATISTICS_VERSION = 1
_STATISTICS_HEADER = struct.Struct(">4sQ")
_STATISTICS_ENTRY = struct.Struct(">BQQ")


class ContainerEncryptionError(ValueError):
    """An encrypted container is malformed or fails authentication."""


@dataclass(frozen=True)
class ContainerEncryptionConfig:
    """Caller policy for an indexed encrypted Rex container.

    Values are key identifiers and exact logical tensor names, never key bytes.
    Any tensor not named by ``tensor_keys`` or ``plaintext_tensors`` is protected
    by ``footer_key``.  This fail-closed default prevents a newly added state
    tensor from becoming plaintext by omission.
    """

    footer_key: str
    tensor_keys: Mapping[str, Collection[str]]
    plaintext_tensors: Collection[str] = ()
    plaintext_manifest: bool = False
    chunk_size: int = DEFAULT_CHUNK_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.footer_key, str) or not self.footer_key:
            raise ValueError("footer_key must be a nonempty key identifier")
        if not isinstance(self.tensor_keys, Mapping):
            raise TypeError("tensor_keys must map key identifiers to exact tensor names")
        normalized: dict[str, tuple[str, ...]] = {}
        for key_id, names in self.tensor_keys.items():
            if not isinstance(key_id, str) or not key_id:
                raise ValueError("tensor_keys contains an invalid key identifier")
            if isinstance(names, str):
                raise TypeError("each tensor_keys value must be a collection of names")
            normalized[key_id] = tuple(names)
        if isinstance(self.plaintext_tensors, str):
            raise TypeError("plaintext_tensors must be a collection of exact names")
        plaintext = tuple(self.plaintext_tensors)
        try:
            chunk_size = int(self.chunk_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("chunk_size must be an integer") from exc
        if chunk_size != self.chunk_size or not 4096 <= chunk_size <= (1 << 30):
            raise ValueError("chunk_size must be between 4096 bytes and 1 GiB")
        object.__setattr__(self, "tensor_keys", normalized)
        object.__setattr__(self, "plaintext_tensors", plaintext)
        object.__setattr__(self, "plaintext_manifest", bool(self.plaintext_manifest))
        object.__setattr__(self, "chunk_size", chunk_size)


@runtime_checkable
class ContainerEncryptionProperties(Protocol):
    """Opaque caller/KMS-owned sealing context consumed by core writers."""

    configuration: ContainerEncryptionConfig
    authenticated_encryption: bool | Callable[[], bool]

    def seal(self, key_id: str, plaintext: bytes, aad: bytes) -> bytes: ...


@runtime_checkable
class ContainerDecryptionProperties(Protocol):
    """Opaque caller/KMS-owned opening context consumed by core readers.

    Implementations may additionally expose ``open_with(key_id, envelope, aad)``.
    Core calls that fast path only after the key identifier has come from an
    authenticated inner manifest; older properties that expose only ``open``
    retain the same behavior.
    """

    authenticated_encryption: bool | Callable[[], bool]

    def open(self, envelope: bytes, aad: bytes) -> bytes: ...


def _canonical(value: Any) -> bytes:
    return _dumps(
        value,
        nan="raise",
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: Any, field: str) -> bytes:
    if not isinstance(value, str):
        raise ContainerEncryptionError(f"{field} must be base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ContainerEncryptionError(f"{field} is not valid base64") from exc


def _authenticated(properties: Any) -> bool:
    value = getattr(properties, "authenticated_encryption", False)
    try:
        value = value() if callable(value) else value
    except Exception as exc:
        raise TypeError(
            "container properties could not state authenticated-encryption semantics"
        ) from exc
    return value is True


def _require_encryption_properties(properties: Any) -> None:
    if properties is None or not _authenticated(properties):
        raise TypeError(
            "encryption_properties must state authenticated_encryption=True"
        )
    if not callable(getattr(properties, "seal", None)):
        raise TypeError("encryption_properties must provide seal(key_id, plaintext, aad)")


def _require_decryption_properties(properties: Any) -> None:
    if properties is None or not _authenticated(properties):
        raise PermissionError(
            "authenticated decryption properties are required for this container"
        )
    if not callable(getattr(properties, "open", None)):
        raise TypeError("decryption_properties must provide open(envelope, aad)")


def _seal(properties: Any, key_id: str, plaintext: bytes, aad: bytes) -> bytes:
    try:
        envelope = properties.seal(key_id, plaintext, aad)
    except Exception as exc:
        raise ContainerEncryptionError(
            f"could not seal data with key identifier {key_id!r}"
        ) from exc
    if not isinstance(envelope, (bytes, bytearray, memoryview)):
        raise TypeError("seal() must return a self-framing bytes envelope")
    envelope = bytes(envelope)
    if not envelope:
        raise ContainerEncryptionError("seal() returned an empty envelope")
    return envelope


def _open(properties: Any, envelope: bytes, aad: bytes) -> bytes:
    _require_decryption_properties(properties)
    try:
        plaintext = properties.open(envelope, aad)
    except Exception as exc:
        raise PermissionError("container authentication or key resolution failed") from exc
    if not isinstance(plaintext, (bytes, bytearray, memoryview)):
        raise TypeError("open() must return plaintext bytes")
    return bytes(plaintext)


def _open_member(
    properties: Any,
    key_id: str,
    envelope: bytes,
    aad: bytes,
) -> bytes:
    """Open data whose key id came from an authenticated inner manifest."""
    _require_decryption_properties(properties)
    keyed = getattr(properties, "open_with", None)
    try:
        plaintext = (
            keyed(key_id, envelope, aad)
            if callable(keyed)
            else properties.open(envelope, aad)
        )
    except Exception as exc:
        raise PermissionError("container authentication or key resolution failed") from exc
    if not isinstance(plaintext, (bytes, bytearray, memoryview)):
        raise TypeError("open_with()/open() must return plaintext bytes")
    return bytes(plaintext)


def _encryption_policy(
    properties: Any,
    logical_names: set[str],
) -> tuple[dict[str, str | None], int, bool, str]:
    _require_encryption_properties(properties)
    configuration = getattr(properties, "configuration", properties)
    footer_key = getattr(configuration, "footer_key", None)
    if not isinstance(footer_key, str) or not footer_key:
        raise ValueError("encryption_properties.footer_key must be a nonempty key identifier")

    try:
        chunk_size = int(getattr(configuration, "chunk_size", DEFAULT_CHUNK_SIZE))
    except (TypeError, ValueError) as exc:
        raise ValueError("encryption_properties.chunk_size must be an integer") from exc
    if chunk_size < 4096 or chunk_size > (1 << 30):
        raise ValueError("chunk_size must be between 4096 bytes and 1 GiB")

    plaintext = getattr(configuration, "plaintext_tensors", ())
    if isinstance(plaintext, str):
        raise TypeError("plaintext_tensors must be a collection of exact logical names")
    plaintext = set(plaintext)
    if any(not isinstance(name, str) or not name for name in plaintext):
        raise ValueError("plaintext_tensors contains an invalid logical name")

    tensor_keys = getattr(configuration, "tensor_keys", {})
    if tensor_keys is None:
        tensor_keys = {}
    if not isinstance(tensor_keys, Mapping):
        raise TypeError("tensor_keys must map key identifiers to logical-name collections")

    assigned: dict[str, str | None] = {name: footer_key for name in logical_names}
    seen: set[str] = set()
    for key_id, names in tensor_keys.items():
        if not isinstance(key_id, str) or not key_id:
            raise ValueError("tensor_keys contains an invalid key identifier")
        if isinstance(names, str):
            raise TypeError("each tensor_keys value must be a collection of exact names")
        names = list(names)
        if not names:
            raise ValueError(f"tensor key {key_id!r} maps no logical names")
        for name in names:
            if not isinstance(name, str) or not name:
                raise ValueError(f"tensor key {key_id!r} contains an invalid name")
            if name in seen:
                raise ValueError(f"logical tensor {name!r} is assigned to multiple keys")
            seen.add(name)
            assigned[name] = key_id

    named = seen | plaintext
    unknown = named - logical_names
    if unknown:
        raise ValueError(f"encryption policy names absent tensors: {sorted(unknown)!r}")
    overlap = seen & plaintext
    if overlap:
        raise ValueError(
            f"tensors cannot be both encrypted and plaintext: {sorted(overlap)!r}"
        )
    for name in plaintext:
        assigned[name] = None

    plaintext_manifest = bool(getattr(configuration, "plaintext_manifest", False))
    return assigned, chunk_size, plaintext_manifest, footer_key


def _manifest_aad(kind: str, bundle_id: str) -> bytes:
    return _MANIFEST_AAD + kind.encode("utf-8") + b"\0" + bundle_id.encode("ascii")


def _chunk_aad(
    *,
    kind: str,
    bundle_id: str,
    logical_name: str,
    dtype: str,
    shape: list[int],
    chunk_index: int,
    chunk_count: int,
    plain_start: int,
    plain_stop: int,
) -> bytes:
    fields = {
        "bundle_id": bundle_id,
        "chunk_count": chunk_count,
        "chunk_index": chunk_index,
        "dtype": dtype,
        "kind": kind,
        "logical_name": logical_name,
        "plain_start": plain_start,
        "plain_stop": plain_stop,
        "shape": shape,
    }
    return _CHUNK_AAD + _canonical(fields)


def _statistics_aad(manifest: dict[str, Any], member: dict[str, Any]) -> bytes:
    fields = {
        "bundle_id": manifest["bundle_id"],
        "chunk_count": member["chunk_count"],
        "dtype": member["dtype"],
        "key_id": member["key_id"],
        "kind": manifest["kind"],
        "logical_name": member["logical_name"],
        "shape": member["shape"],
        "statistics_version": _STATISTICS_VERSION,
        "storage_sha256": member["storage_sha256"],
    }
    return _STATISTICS_AAD + _canonical(fields)


def _chunk_ranges(nbytes: int, chunk_size: int) -> list[tuple[int, int]]:
    if nbytes == 0:
        # An encrypted empty tensor still carries a key-authenticated envelope, so
        # possession of no data does not become possession of the protected grade.
        return [(0, 0)]
    return [
        (start, min(start + chunk_size, nbytes))
        for start in range(0, nbytes, chunk_size)
    ]


def _null_mask(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"f", "c"}:
        return np.isnan(values)
    if values.dtype.kind in {"m", "M"}:
        return np.isnat(values)
    return np.zeros(values.shape, dtype=bool)


def _encode_chunk_statistics(
    plaintext: bytes,
    dtype: np.dtype,
    ranges: list[tuple[int, int]],
) -> bytes:
    """Encode fixed-size per-chunk min/max/null facts without length leakage."""
    ordered = dtype.kind in {"b", "i", "u", "f", "m", "M"}
    zero = b"\0" * dtype.itemsize
    payload = bytearray(_STATISTICS_HEADER.pack(_STATISTICS_MAGIC, len(ranges)))
    for plain_start, plain_stop in ranges:
        piece = plaintext[plain_start:plain_stop]
        if len(piece) % dtype.itemsize:
            raise ValueError("container chunks must end on tensor element boundaries")
        values = np.frombuffer(piece, dtype=dtype)
        nulls = _null_mask(values)
        valid = values[~nulls]
        has_minmax = bool(ordered and valid.size)
        payload.extend(
            _STATISTICS_ENTRY.pack(
                int(has_minmax),
                int(values.size),
                int(np.count_nonzero(nulls)),
            )
        )
        if has_minmax:
            minimum = np.asarray(valid.min(), dtype=dtype).reshape(()).tobytes()
            maximum = np.asarray(valid.max(), dtype=dtype).reshape(()).tobytes()
            payload.extend(minimum)
            payload.extend(maximum)
        else:
            payload.extend(zero)
            payload.extend(zero)
    return bytes(payload)


def _decode_chunk_statistics(
    payload: bytes,
    member: dict[str, Any],
) -> list[dict[str, Any]]:
    dtype = np.dtype(member["dtype"])
    chunk_count = member["chunk_count"]
    entry_size = _STATISTICS_ENTRY.size + 2 * dtype.itemsize
    expected_size = _STATISTICS_HEADER.size + chunk_count * entry_size
    if len(payload) != expected_size:
        raise ContainerEncryptionError("member chunk statistics length is malformed")
    magic, encoded_count = _STATISTICS_HEADER.unpack_from(payload)
    if magic != _STATISTICS_MAGIC or encoded_count != chunk_count:
        raise ContainerEncryptionError("member chunk statistics header is malformed")

    ordered = dtype.kind in {"b", "i", "u", "f", "m", "M"}
    ranges = [tuple(pair) for pair in member["chunk_plain_ranges"]]
    statistics: list[dict[str, Any]] = []
    offset = _STATISTICS_HEADER.size
    zero = b"\0" * dtype.itemsize
    for chunk_index, (plain_start, plain_stop) in enumerate(ranges):
        flags, count, null_count = _STATISTICS_ENTRY.unpack_from(payload, offset)
        offset += _STATISTICS_ENTRY.size
        minimum_bytes = payload[offset:offset + dtype.itemsize]
        offset += dtype.itemsize
        maximum_bytes = payload[offset:offset + dtype.itemsize]
        offset += dtype.itemsize
        expected_count = (plain_stop - plain_start) // dtype.itemsize
        if count != expected_count or null_count > count or flags not in {0, 1}:
            raise ContainerEncryptionError("member chunk statistics counts are malformed")
        if flags and (not ordered or count == null_count):
            raise ContainerEncryptionError("member chunk statistics ordering is malformed")
        if not flags and (minimum_bytes != zero or maximum_bytes != zero):
            raise ContainerEncryptionError("empty member statistics carry value bytes")
        minimum = maximum = None
        if flags:
            minimum = np.frombuffer(minimum_bytes, dtype=dtype, count=1)[0]
            maximum = np.frombuffer(maximum_bytes, dtype=dtype, count=1)[0]
            if (
                bool(_null_mask(np.asarray([minimum], dtype=dtype))[0])
                or bool(_null_mask(np.asarray([maximum], dtype=dtype))[0])
                or bool(minimum > maximum)
            ):
                raise ContainerEncryptionError("member chunk statistics bounds are malformed")
        statistics.append(
            {
                "chunk_index": chunk_index,
                "count": int(count),
                "min": minimum,
                "max": maximum,
                "null_count": int(null_count),
            }
        )
    return statistics


def _protect_tensor_members(
    tensors: dict[str, Any],
    metadata: dict[str, Any],
    encryption_properties: Any,
    *,
    kind: str,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    """Protect named contiguous arrays inside an indexed safetensors envelope.

    The returned arrays are explicit uint8 storage entries.  The returned metadata
    is string-only and can be handed directly to ``safetensors.save_file``.
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError("container kind must be a nonempty string")
    if not isinstance(metadata, dict):
        raise TypeError("container metadata must be a dict")
    logical_names = set(tensors)
    if any(not isinstance(name, str) or not name for name in logical_names):
        raise ValueError("tensor names must be nonempty strings")

    policy, chunk_size, plaintext_manifest, footer_key = _encryption_policy(
        encryption_properties, logical_names
    )
    bundle_id = secrets.token_hex(16)  # random per export; never content-derived
    storage: dict[str, np.ndarray] = {}
    members: list[dict[str, Any]] = []

    for number, logical_name in enumerate(sorted(logical_names)):
        array = np.ascontiguousarray(np.asarray(tensors[logical_name]))
        if array.dtype.hasobject or array.dtype.kind in {"S", "U", "V"}:
            raise TypeError(
                f"logical tensor {logical_name!r} has unsupported dtype {array.dtype!r}"
            )
        dtype = array.dtype.str
        shape = [int(axis) for axis in array.shape]
        plaintext = array.tobytes(order="C")
        # Query statistics describe complete typed values. Keep every data chunk
        # on an element boundary while respecting the configured byte ceiling.
        member_chunk_size = chunk_size - (chunk_size % array.dtype.itemsize)
        if member_chunk_size < 4096:
            raise ValueError(
                f"chunk_size is too small for tensor dtype {array.dtype!r}"
            )
        ranges = _chunk_ranges(len(plaintext), member_chunk_size)
        chunk_count = len(ranges)
        key_id = policy[logical_name]
        protected = key_id is not None
        storage_name = f"{_STORAGE_PREFIX}{number:08d}"
        offsets = [0]
        pieces: list[bytes] = []
        hashes: list[str] = []

        for chunk_index, (plain_start, plain_stop) in enumerate(ranges):
            piece = plaintext[plain_start:plain_stop]
            if protected:
                aad = _chunk_aad(
                    kind=kind,
                    bundle_id=bundle_id,
                    logical_name=logical_name,
                    dtype=dtype,
                    shape=shape,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    plain_start=plain_start,
                    plain_stop=plain_stop,
                )
                piece = _seal(encryption_properties, key_id, piece, aad)
            else:
                hashes.append(hashlib.sha256(piece).hexdigest())
            pieces.append(piece)
            offsets.append(offsets[-1] + len(piece))

        packed = b"".join(pieces)
        storage[storage_name] = np.frombuffer(packed, dtype=np.uint8).copy()
        member = {
            "chunk_count": chunk_count,
            "chunk_offsets": offsets,
            "chunk_plain_ranges": [[start, stop] for start, stop in ranges],
            "chunk_size": member_chunk_size,
            "dtype": dtype,
            "key_id": key_id,
            "logical_name": logical_name,
            "plain_nbytes": len(plaintext),
            "protected": protected,
            "shape": shape,
            "storage_name": storage_name,
            "storage_nbytes": len(packed),
            "storage_sha256": hashlib.sha256(packed).hexdigest(),
        }
        if not protected:
            member["chunk_hashes"] = hashes

        statistics = _encode_chunk_statistics(plaintext, array.dtype, ranges)
        member["statistics_version"] = _STATISTICS_VERSION
        statistics_manifest = {
            "bundle_id": bundle_id,
            "kind": kind,
        }
        if protected:
            member["statistics_envelope"] = _b64encode(
                _seal(
                    encryption_properties,
                    key_id,
                    statistics,
                    _statistics_aad(statistics_manifest, member),
                )
            )
        else:
            member["statistics"] = _b64encode(statistics)
        members.append(member)

    manifest = {
        # Reserved inside the authenticated manifest so a future external
        # monotonic anchor can be referenced without changing this format.
        "anchor_reference": None,
        "bundle_id": bundle_id,
        "footer_key": footer_key,
        "format": _FORMAT,
        "kind": kind,
        "members": members,
        "metadata": metadata,
        "version": _VERSION,
    }
    manifest_bytes = _canonical(manifest)
    aad = _manifest_aad(kind, bundle_id)
    descriptor: dict[str, Any] = {
        "bundle_id": bundle_id,
        "footer_key": footer_key,
        "format": _FORMAT,
        "kind": kind,
        "manifest_mode": "signed_plaintext" if plaintext_manifest else "encrypted",
        "version": _VERSION,
    }
    if plaintext_manifest:
        descriptor["manifest"] = manifest
        descriptor["manifest_auth"] = _b64encode(
            _seal(encryption_properties, footer_key, b"", aad + b"\0" + manifest_bytes)
        )
    else:
        descriptor["manifest_envelope"] = _b64encode(
            _seal(encryption_properties, footer_key, manifest_bytes, aad)
        )

    outer_metadata = {
        ENCRYPTION_METADATA_KEY: _canonical(descriptor).decode("utf-8"),
        "rex_encrypted": "1",
    }
    return storage, outer_metadata, manifest


def protect_tensors(
    tensors: dict[str, Any],
    metadata: dict[str, Any],
    encryption_properties: Any,
    *,
    kind: str,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Protect arrays for a safetensors outer container.

    Directory backends use the private three-value helper so they can choose
    native ``.npy`` storage for explicitly plaintext members.  The public
    common seam retains its two-value return contract.
    """
    storage, outer_metadata, _ = _protect_tensor_members(
        tensors,
        metadata,
        encryption_properties,
        kind=kind,
    )
    return storage, outer_metadata


def encrypted_metadata(metadata: dict[str, str] | None) -> bool:
    """Whether a safetensors metadata map carries the Rex encrypted envelope."""
    return bool(metadata and ENCRYPTION_METADATA_KEY in metadata)


def _descriptor(metadata: dict[str, str]) -> dict[str, Any]:
    raw = metadata.get(ENCRYPTION_METADATA_KEY)
    if not isinstance(raw, str):
        raise ContainerEncryptionError("encrypted container descriptor is missing")
    try:
        descriptor = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ContainerEncryptionError("encrypted container descriptor is not JSON") from exc
    if not isinstance(descriptor, dict):
        raise ContainerEncryptionError("encrypted container descriptor must be an object")
    if descriptor.get("format") != _FORMAT or descriptor.get("version") != _VERSION:
        raise ContainerEncryptionError("unsupported encrypted container format or version")
    bundle_id = descriptor.get("bundle_id")
    if not isinstance(bundle_id, str) or len(bundle_id) != 32:
        raise ContainerEncryptionError("encrypted container has an invalid bundle id")
    try:
        bytes.fromhex(bundle_id)
    except ValueError as exc:
        raise ContainerEncryptionError("encrypted container has an invalid bundle id") from exc
    kind = descriptor.get("kind")
    footer_key = descriptor.get("footer_key")
    if not isinstance(kind, str) or not kind:
        raise ContainerEncryptionError("encrypted container kind is invalid")
    if not isinstance(footer_key, str) or not footer_key:
        raise ContainerEncryptionError("encrypted container footer key id is invalid")
    return descriptor


def open_encrypted_manifest(
    metadata: dict[str, str],
    decryption_properties: Any,
    *,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    """Authenticate/decrypt and structurally validate an inner manifest."""
    descriptor = _descriptor(metadata)
    kind = descriptor["kind"]
    bundle_id = descriptor["bundle_id"]
    if expected_kind is not None and kind != expected_kind:
        raise ContainerEncryptionError(
            f"encrypted container kind {kind!r}, expected {expected_kind!r}"
        )
    aad = _manifest_aad(kind, bundle_id)
    mode = descriptor.get("manifest_mode")
    if mode == "encrypted":
        envelope = _b64decode(descriptor.get("manifest_envelope"), "manifest_envelope")
        manifest_bytes = _open(decryption_properties, envelope, aad)
        try:
            manifest = json.loads(manifest_bytes)
        except (TypeError, ValueError) as exc:
            raise ContainerEncryptionError("decrypted manifest is not JSON") from exc
    elif mode == "signed_plaintext":
        manifest = descriptor.get("manifest")
        if not isinstance(manifest, dict):
            raise ContainerEncryptionError("signed plaintext manifest is missing")
        manifest_bytes = _canonical(manifest)
        authenticator = _b64decode(
            descriptor.get("manifest_auth"), "manifest_auth"
        )
        opened = _open(
            decryption_properties,
            authenticator,
            aad + b"\0" + manifest_bytes,
        )
        if opened != b"":
            raise ContainerEncryptionError("manifest authenticator opened to payload data")
    else:
        raise ContainerEncryptionError("unknown encrypted manifest mode")

    if not isinstance(manifest, dict):
        raise ContainerEncryptionError("inner manifest must be an object")
    for field, value in (
        ("format", _FORMAT),
        ("version", _VERSION),
        ("bundle_id", bundle_id),
        ("kind", kind),
        ("footer_key", descriptor["footer_key"]),
    ):
        if manifest.get(field) != value:
            raise ContainerEncryptionError(f"inner manifest {field} does not match envelope")
    _validate_manifest_members(manifest)
    if not isinstance(manifest.get("metadata"), dict):
        raise ContainerEncryptionError("inner manifest metadata must be an object")
    return manifest


def _as_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContainerEncryptionError(f"{field} must be a nonnegative integer")
    return value


def _validate_manifest_members(manifest: dict[str, Any]) -> None:
    members = manifest.get("members")
    if not isinstance(members, list):
        raise ContainerEncryptionError("inner manifest members must be a list")
    logical_seen: set[str] = set()
    storage_seen: set[str] = set()
    for member_number, member in enumerate(members):
        if not isinstance(member, dict):
            raise ContainerEncryptionError("manifest member must be an object")
        logical_name = member.get("logical_name")
        storage_name = member.get("storage_name")
        if not isinstance(logical_name, str) or not logical_name:
            raise ContainerEncryptionError("manifest member logical name is invalid")
        if not isinstance(storage_name, str) or not storage_name.startswith(_STORAGE_PREFIX):
            raise ContainerEncryptionError("manifest member storage name is invalid")
        if storage_name != f"{_STORAGE_PREFIX}{member_number:08d}":
            raise ContainerEncryptionError("manifest storage names are not canonical")
        if logical_name in logical_seen or storage_name in storage_seen:
            raise ContainerEncryptionError("manifest contains duplicate member names")
        logical_seen.add(logical_name)
        storage_seen.add(storage_name)

        try:
            dtype = np.dtype(member.get("dtype"))
        except (TypeError, ValueError) as exc:
            raise ContainerEncryptionError(
                f"manifest member {logical_name!r} has invalid dtype"
            ) from exc
        if dtype.hasobject or dtype.kind in {"S", "U", "V"}:
            raise ContainerEncryptionError(
                f"manifest member {logical_name!r} has unsupported dtype"
            )
        shape = member.get("shape")
        if not isinstance(shape, list):
            raise ContainerEncryptionError("manifest member shape must be a list")
        shape = [_as_nonnegative_int(axis, "shape axis") for axis in shape]
        expected_nbytes = int(dtype.itemsize) * math.prod(shape)
        plain_nbytes = _as_nonnegative_int(member.get("plain_nbytes"), "plain_nbytes")
        if expected_nbytes != plain_nbytes:
            raise ContainerEncryptionError(
                f"manifest member {logical_name!r} dtype/shape byte size disagrees"
            )

        chunk_size = _as_nonnegative_int(member.get("chunk_size"), "chunk_size")
        if chunk_size < 4096 or chunk_size > (1 << 30):
            raise ContainerEncryptionError("manifest chunk_size is outside allowed bounds")
        chunk_count = _as_nonnegative_int(member.get("chunk_count"), "chunk_count")
        expected_count = max(1, (plain_nbytes + chunk_size - 1) // chunk_size)
        if chunk_count != expected_count:
            raise ContainerEncryptionError(
                f"manifest member {logical_name!r} chunk count is incomplete"
            )
        offsets = member.get("chunk_offsets")
        ranges = member.get("chunk_plain_ranges")
        if not isinstance(offsets, list) or len(offsets) != chunk_count + 1:
            raise ContainerEncryptionError("chunk offsets do not match chunk count")
        if not isinstance(ranges, list) or len(ranges) != chunk_count:
            raise ContainerEncryptionError("plain ranges do not match chunk count")
        offsets = [_as_nonnegative_int(value, "chunk offset") for value in offsets]
        if offsets[0] != 0 or any(
            b < a for a, b in zip(offsets, offsets[1:], strict=False)
        ):
            raise ContainerEncryptionError("chunk offsets are not ordered from zero")
        storage_nbytes = _as_nonnegative_int(
            member.get("storage_nbytes"), "storage_nbytes"
        )
        if offsets[-1] != storage_nbytes:
            raise ContainerEncryptionError("chunk offsets do not cover storage payload")
        storage_sha256 = member.get("storage_sha256")
        if storage_sha256 is not None:
            if not isinstance(storage_sha256, str) or len(storage_sha256) != 64:
                raise ContainerEncryptionError("storage_sha256 is malformed")
            try:
                int(storage_sha256, 16)
            except ValueError as exc:
                raise ContainerEncryptionError("storage_sha256 is malformed") from exc
        expected_ranges = _chunk_ranges(plain_nbytes, chunk_size)
        normalized_ranges = []
        for pair in ranges:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ContainerEncryptionError("chunk plaintext range is malformed")
            normalized_ranges.append(
                (_as_nonnegative_int(pair[0], "plain range"),
                 _as_nonnegative_int(pair[1], "plain range"))
            )
        if normalized_ranges != expected_ranges:
            raise ContainerEncryptionError("chunk plaintext ranges are incomplete")

        protected = member.get("protected")
        key_id = member.get("key_id")
        if not isinstance(protected, bool):
            raise ContainerEncryptionError("manifest protected flag must be boolean")
        if protected:
            if not isinstance(key_id, str) or not key_id:
                raise ContainerEncryptionError("protected member has no key identifier")
            if "chunk_hashes" in member:
                raise ContainerEncryptionError("protected member must not carry plaintext hashes")
            if any(b == a for a, b in zip(offsets, offsets[1:], strict=False)):
                raise ContainerEncryptionError("protected chunk has an empty envelope")
        else:
            if key_id is not None:
                raise ContainerEncryptionError("plaintext member carries a key identifier")
            hashes = member.get("chunk_hashes")
            if not isinstance(hashes, list) or len(hashes) != chunk_count:
                raise ContainerEncryptionError("plaintext hashes do not match chunk count")
            if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
                raise ContainerEncryptionError("plaintext chunk hash is malformed")

        statistics_version = member.get("statistics_version")
        has_statistics_fields = any(
            field in member for field in ("statistics", "statistics_envelope")
        )
        if statistics_version is None:
            if has_statistics_fields:
                raise ContainerEncryptionError("member statistics version is missing")
            continue  # encrypted container v1 written before query statistics
        if (
            isinstance(statistics_version, bool)
            or statistics_version != _STATISTICS_VERSION
        ):
            raise ContainerEncryptionError("member statistics version is unsupported")
        if storage_sha256 is None:
            raise ContainerEncryptionError("member statistics require a storage digest")
        if chunk_size % dtype.itemsize:
            raise ContainerEncryptionError(
                "statistics-bearing chunks must end on tensor element boundaries"
            )
        if protected:
            if "statistics" in member:
                raise ContainerEncryptionError(
                    "protected member carries plaintext statistics"
                )
            envelope = _b64decode(
                member.get("statistics_envelope"), "statistics_envelope"
            )
            if not envelope:
                raise ContainerEncryptionError("member statistics envelope is empty")
        else:
            if "statistics_envelope" in member:
                raise ContainerEncryptionError(
                    "plaintext member carries encrypted statistics"
                )
            statistics = _b64decode(member.get("statistics"), "statistics")
            _decode_chunk_statistics(statistics, member)


def validate_storage_inventory(opened: Any, manifest: dict[str, Any]) -> None:
    """Bind the authenticated member inventory to the actual outer header."""
    members = manifest["members"]
    expected = {member["storage_name"] for member in members}
    actual = set(opened.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContainerEncryptionError(
            f"outer storage inventory differs from manifest; missing={missing!r}, extra={extra!r}"
        )
    for member in members:
        view = opened.get_slice(member["storage_name"])
        if view.get_dtype() != "U8" or view.get_shape() != [member["storage_nbytes"]]:
            raise ContainerEncryptionError(
                f"outer storage spec for {member['logical_name']!r} differs from manifest"
            )


def _member_by_name(manifest: dict[str, Any], logical_name: str) -> dict[str, Any]:
    for member in manifest["members"]:
        if member["logical_name"] == logical_name:
            return member
    raise KeyError(logical_name)


def _first_axis_selection(
    member: dict[str, Any], index: int | slice | None
) -> tuple[int, int, tuple[int, ...]]:
    dtype = np.dtype(member["dtype"])
    shape = tuple(member["shape"])
    nbytes = member["plain_nbytes"]
    if index is None:
        return 0, nbytes, shape
    if not shape:
        raise IndexError("a scalar tensor has no first axis")
    row_nbytes = dtype.itemsize * math.prod(shape[1:])
    if isinstance(index, (int, np.integer)):
        position = int(index)
        if position < 0:
            position += shape[0]
        if position < 0 or position >= shape[0]:
            raise IndexError("tensor first-axis index is out of range")
        return position * row_nbytes, (position + 1) * row_nbytes, shape[1:]
    if not isinstance(index, slice):
        raise TypeError("index must be an integer, first-axis slice, or None")
    start, stop, step = index.indices(shape[0])
    if step != 1:
        raise ValueError("encrypted tensor slicing currently requires a unit step")
    return start * row_nbytes, stop * row_nbytes, (stop - start, *shape[1:])


def _read_member_chunk(
    opened: Any,
    manifest: dict[str, Any],
    member: dict[str, Any],
    decryption_properties: Any,
    chunk_index: int,
    *,
    chunk_cache: dict[tuple[str, str, int], bytes] | None = None,
) -> bytes:
    cache_key = (manifest["bundle_id"], member["storage_name"], chunk_index)
    if chunk_cache is not None and cache_key in chunk_cache:
        return chunk_cache[cache_key]

    ranges = [tuple(pair) for pair in member["chunk_plain_ranges"]]
    offsets = member["chunk_offsets"]
    storage_start = offsets[chunk_index]
    storage_stop = offsets[chunk_index + 1]
    view = opened.get_slice(member["storage_name"])
    envelope = np.asarray(view[storage_start:storage_stop]).tobytes()
    plain_start, plain_stop = ranges[chunk_index]
    if member["protected"]:
        aad = _chunk_aad(
            kind=manifest["kind"],
            bundle_id=manifest["bundle_id"],
            logical_name=member["logical_name"],
            dtype=member["dtype"],
            shape=member["shape"],
            chunk_index=chunk_index,
            chunk_count=member["chunk_count"],
            plain_start=plain_start,
            plain_stop=plain_stop,
        )
        piece = _open_member(
            decryption_properties,
            member["key_id"],
            envelope,
            aad,
        )
    else:
        digest = hashlib.sha256(envelope).hexdigest()
        if not hmac.compare_digest(digest, member["chunk_hashes"][chunk_index]):
            raise ContainerEncryptionError(
                f"plaintext chunk authentication failed for {member['logical_name']!r}"
            )
        piece = envelope
    if len(piece) != plain_stop - plain_start:
        raise ContainerEncryptionError(
            f"opened chunk length is wrong for {member['logical_name']!r}"
        )
    if chunk_cache is not None:
        chunk_cache[cache_key] = piece
    return piece


def _member_statistics(
    manifest: dict[str, Any],
    member: dict[str, Any],
    decryption_properties: Any,
    *,
    statistics_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]] | None:
    if member.get("statistics_version") is None:
        return None
    logical_name = member["logical_name"]
    if statistics_cache is not None and logical_name in statistics_cache:
        return statistics_cache[logical_name]
    if member["protected"]:
        envelope = _b64decode(
            member.get("statistics_envelope"), "statistics_envelope"
        )
        payload = _open_member(
            decryption_properties,
            member["key_id"],
            envelope,
            _statistics_aad(manifest, member),
        )
    else:
        payload = _b64decode(member.get("statistics"), "statistics")
    statistics = _decode_chunk_statistics(payload, member)
    if statistics_cache is not None:
        statistics_cache[logical_name] = statistics
    return statistics


def read_protected_tensor(
    opened: Any,
    manifest: dict[str, Any],
    logical_name: str,
    decryption_properties: Any,
    *,
    index: int | slice | None = None,
    _chunk_cache: dict[tuple[str, str, int], bytes] | None = None,
) -> np.ndarray:
    """Read one logical tensor, decrypting only touched first-axis chunks."""
    member = _member_by_name(manifest, logical_name)
    byte_start, byte_stop, output_shape = _first_axis_selection(member, index)
    ranges = [tuple(pair) for pair in member["chunk_plain_ranges"]]
    protected = member["protected"]
    if protected:
        _require_decryption_properties(decryption_properties)

    chosen = [
        chunk_index
        for chunk_index, (start, stop) in enumerate(ranges)
        if (start < byte_stop and stop > byte_start)
        or (member["plain_nbytes"] == 0 and chunk_index == 0)
    ]
    if protected and not chosen:
        # An empty first-axis selection returns no payload, but it must still
        # prove possession of the tensor's key instead of becoming an auth bypass.
        chunk_size = member["chunk_size"]
        chosen = [min(byte_start // chunk_size, member["chunk_count"] - 1)]

    pieces = [
        _read_member_chunk(
            opened,
            manifest,
            member,
            decryption_properties,
            chunk_index,
            chunk_cache=_chunk_cache,
        )
        for chunk_index in chosen
    ]
    joined = b"".join(pieces)
    origin = ranges[chosen[0]][0] if chosen else byte_start
    selected = joined[byte_start - origin:byte_stop - origin]

    dtype = np.dtype(member["dtype"])
    expected = dtype.itemsize * math.prod(output_shape)
    if len(selected) != expected:
        raise ContainerEncryptionError(
            f"selected payload length is wrong for {logical_name!r}"
        )
    return np.frombuffer(selected, dtype=dtype).copy().reshape(output_shape)


_PREDICATE_OPERATORS = frozenset(
    {"==", "!=", "<", "<=", ">", ">=", "isnull", "notnull"}
)


def _predicate_value(dtype: np.dtype, operator: str, value: Any) -> Any:
    if operator not in _PREDICATE_OPERATORS:
        raise ValueError(
            f"unsupported predicate operator {operator!r}; "
            f"expected one of {sorted(_PREDICATE_OPERATORS)!r}"
        )
    if operator in {"isnull", "notnull"}:
        if value is not None:
            raise ValueError(f"{operator} predicate value must be None")
        return None
    try:
        converted = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"predicate value cannot be represented as {dtype}") from exc
    if converted.shape != ():
        raise TypeError("predicate value must be scalar")
    if operator in {"<", "<=", ">", ">="} and dtype.kind not in {
        "b", "i", "u", "f", "m", "M"
    }:
        raise TypeError(f"operator {operator!r} is not ordered for dtype {dtype}")
    return converted[()]


def _scalar_is_null(value: Any, dtype: np.dtype) -> bool:
    if dtype.kind not in {"f", "c", "m", "M"}:
        return False
    return bool(_null_mask(np.asarray([value], dtype=dtype))[0])


def _predicate_mask(
    values: np.ndarray,
    operator: str,
    value: Any,
) -> np.ndarray:
    nulls = _null_mask(values)
    if operator == "isnull":
        return nulls
    if operator == "notnull":
        return ~nulls
    if _scalar_is_null(value, values.dtype):
        return np.zeros(values.shape, dtype=bool)
    if operator == "==":
        result = values == value
    elif operator == "!=":
        result = values != value
    elif operator == "<":
        result = values < value
    elif operator == "<=":
        result = values <= value
    elif operator == ">":
        result = values > value
    else:
        result = values >= value
    return np.asarray(result, dtype=bool) & ~nulls


def _statistics_may_match(
    statistics: dict[str, Any],
    dtype: np.dtype,
    operator: str,
    value: Any,
) -> bool:
    valid_count = statistics["count"] - statistics["null_count"]
    if operator == "isnull":
        return statistics["null_count"] > 0
    if operator == "notnull":
        return valid_count > 0
    if valid_count == 0 or _scalar_is_null(value, dtype):
        return False
    minimum = statistics["min"]
    maximum = statistics["max"]
    if minimum is None or maximum is None:
        return True
    if operator == "==":
        return bool(minimum <= value <= maximum)
    if operator == "!=":
        return not bool(minimum == maximum == value)
    if operator == "<":
        return bool(minimum < value)
    if operator == "<=":
        return bool(minimum <= value)
    if operator == ">":
        return bool(maximum > value)
    return bool(maximum >= value)


def _query_protected_indices(
    opened: Any,
    manifest: dict[str, Any],
    logical_name: str,
    operator: str,
    value: Any,
    decryption_properties: Any,
    *,
    chunk_cache: dict[tuple[str, str, int], bytes] | None = None,
    statistics_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> np.ndarray:
    """Return first-axis positions matching one scalar predicate."""
    if operator not in _PREDICATE_OPERATORS:
        raise ValueError(
            f"unsupported predicate operator {operator!r}; "
            f"expected one of {sorted(_PREDICATE_OPERATORS)!r}"
        )
    member = _member_by_name(manifest, logical_name)
    if len(member["shape"]) != 1:
        raise ValueError("query predicates require a one-dimensional tensor")
    dtype = np.dtype(member["dtype"])
    converted = _predicate_value(dtype, operator, value)
    statistics = _member_statistics(
        manifest,
        member,
        decryption_properties,
        statistics_cache=statistics_cache,
    )
    if statistics is None:
        values = read_protected_tensor(
            opened,
            manifest,
            logical_name,
            decryption_properties,
            _chunk_cache=chunk_cache,
        )
        return np.flatnonzero(
            _predicate_mask(values, operator, converted)
        ).astype(np.int64, copy=False)

    matches: list[np.ndarray] = []
    ranges = [tuple(pair) for pair in member["chunk_plain_ranges"]]
    for chunk_statistics in statistics:
        if not _statistics_may_match(
            chunk_statistics, dtype, operator, converted
        ):
            continue
        chunk_index = chunk_statistics["chunk_index"]
        piece = _read_member_chunk(
            opened,
            manifest,
            member,
            decryption_properties,
            chunk_index,
            chunk_cache=chunk_cache,
        )
        values = np.frombuffer(piece, dtype=dtype)
        local = np.flatnonzero(_predicate_mask(values, operator, converted))
        if local.size:
            row_start = ranges[chunk_index][0] // dtype.itemsize
            matches.append(local.astype(np.int64, copy=False) + row_start)
    if not matches:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(matches)


def _read_protected_indices(
    opened: Any,
    manifest: dict[str, Any],
    logical_name: str,
    indices: np.ndarray,
    decryption_properties: Any,
    *,
    chunk_cache: dict[tuple[str, str, int], bytes] | None = None,
) -> np.ndarray:
    """Gather first-axis rows while opening every touched chunk at most once."""
    member = _member_by_name(manifest, logical_name)
    shape = tuple(member["shape"])
    if not shape:
        raise ValueError("query results require a tensor with a first axis")
    positions = np.asarray(indices)
    if positions.ndim != 1 or positions.dtype.kind not in {"i", "u"}:
        raise TypeError("query indices must be a one-dimensional integer array")
    if np.any(positions < 0) or np.any(positions >= shape[0]):
        raise IndexError("query result index is out of range")
    positions = positions.astype(np.int64, copy=False)
    dtype = np.dtype(member["dtype"])
    output_shape = (int(positions.size), *shape[1:])
    row_nbytes = dtype.itemsize * math.prod(shape[1:])

    if positions.size == 0 or row_nbytes == 0:
        if member["protected"]:
            _read_member_chunk(
                opened,
                manifest,
                member,
                decryption_properties,
                0,
                chunk_cache=chunk_cache,
            )
        return np.empty(output_shape, dtype=dtype)

    runs: list[tuple[int, int]] = []
    run_start = int(positions[0])
    previous = run_start
    for raw_position in positions[1:]:
        position = int(raw_position)
        if position != previous + 1:
            runs.append((run_start, previous + 1))
            run_start = position
        previous = position
    runs.append((run_start, previous + 1))

    chunk_size = member["chunk_size"]
    chunk_count = member["chunk_count"]
    chosen: set[int] = set()
    for run_start, run_stop in runs:
        byte_start = run_start * row_nbytes
        byte_stop = run_stop * row_nbytes
        first = min(byte_start // chunk_size, chunk_count - 1)
        last = min((byte_stop - 1) // chunk_size, chunk_count - 1)
        chosen.update(range(first, last + 1))
    pieces = {
        chunk_index: _read_member_chunk(
            opened,
            manifest,
            member,
            decryption_properties,
            chunk_index,
            chunk_cache=chunk_cache,
        )
        for chunk_index in sorted(chosen)
    }
    ranges = [tuple(pair) for pair in member["chunk_plain_ranges"]]
    selected = bytearray()
    for run_start, run_stop in runs:
        byte_start = run_start * row_nbytes
        byte_stop = run_stop * row_nbytes
        first = min(byte_start // chunk_size, chunk_count - 1)
        last = min((byte_stop - 1) // chunk_size, chunk_count - 1)
        before = len(selected)
        for chunk_index in range(first, last + 1):
            plain_start, plain_stop = ranges[chunk_index]
            left = max(byte_start, plain_start) - plain_start
            right = min(byte_stop, plain_stop) - plain_start
            selected.extend(pieces[chunk_index][left:right])
        expected_run_nbytes = (run_stop - run_start) * row_nbytes
        if len(selected) - before != expected_run_nbytes:
            raise ContainerEncryptionError(
                f"gathered run length is wrong for {logical_name!r}"
            )
    return np.frombuffer(selected, dtype=dtype).copy().reshape(output_shape)
