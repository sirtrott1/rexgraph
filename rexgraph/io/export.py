"""Lineage-sealed, in-memory Parquet export for derived RexGraph partitions.

Encryption wraps the complete Parquet byte artifact in one RexGraph AES-GCM envelope.
It is not Parquet modular column encryption and provides no per-column key isolation.
An encrypted export must be opened whole before projection or predicate pushdown; that
tradeoff is deliberate for a handoff artifact and is not suitable for a working store.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .manifest import canonical_json, manifest_digest

EXPORT_VERSION = 1

__all__ = [
    "EXPORT_VERSION",
    "ExportManifest",
    "export_parquet",
    "parquet_bytes",
    "verify_export",
]


@dataclass(frozen=True)
class ExportManifest:
    """Identify one derived export without embedding source values."""

    partition_digest: str
    schema_digest: str
    payload_sha256: str
    encrypted: bool
    key_id: str | None = None

    @property
    def digest(self) -> str:
        """Return the stable identity of the complete export artifact."""
        return manifest_digest(
            {
                "encrypted": bool(self.encrypted),
                "key_id": self.key_id,
                "object_type": "ExportManifest",
                "partition_digest": self.partition_digest,
                "payload_sha256": self.payload_sha256,
                "schema_digest": self.schema_digest,
                "version": EXPORT_VERSION,
            }
        )


def _arrays(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(data, Mapping):
        raise TypeError("Parquet export data must be a mapping")
    arrays: dict[str, np.ndarray] = {}
    for raw_name, value in data.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("Parquet export column names must be nonempty strings")
        if raw_name in arrays:
            raise ValueError(f"duplicate Parquet export column {raw_name!r}")
        array = np.asarray(value)
        if array.ndim not in (1, 2):
            raise ValueError(f"column {raw_name!r} must be one or two dimensional")
        if array.ndim == 2 and array.shape[1] == 0:
            raise ValueError(f"two-dimensional column {raw_name!r} may not be empty")
        arrays[raw_name] = array
    lengths = {int(array.shape[0]) for array in arrays.values()}
    if len(lengths) > 1:
        raise ValueError("Parquet export columns must have equal row counts")
    return arrays


def _schema(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"dtype": str(arrays[name].dtype), "shape": list(arrays[name].shape)}
        for name in sorted(arrays)
    }


def _physical_columns(arrays: Mapping[str, np.ndarray]):
    import pyarrow as pa

    columns = {}
    split = {}
    logical_names = set(arrays)
    for name in sorted(arrays):
        array = arrays[name]
        if array.ndim == 1:
            physical = name
            if physical in columns:
                raise ValueError(f"duplicate physical Parquet column {physical!r}")
            columns[physical] = pa.array(array)
            continue
        split[name] = {"shape": list(array.shape), "split": True}
        for index in range(array.shape[1]):
            physical = f"{name}_{index}"
            if physical in columns or (physical in logical_names and physical != name):
                raise ValueError(
                    f"two-dimensional column {name!r} collides with {physical!r}"
                )
            columns[physical] = pa.array(array[:, index])
    return columns, split


def parquet_bytes(
    data: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    """Encode array columns to Parquet entirely in memory.

    Logical names are sorted before physical columns are built, so equivalent mappings
    produce identical bytes regardless of caller insertion order. This deliberately
    normalizes the order-preserving reference writer.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrays = _arrays(data)
    columns, split = _physical_columns(arrays)
    table = pa.table(columns)
    schema_metadata = {}
    if split:
        schema_metadata[b"rex_col_meta"] = canonical_json(split)
    if metadata:
        if not isinstance(metadata, Mapping):
            raise TypeError("Parquet export metadata must be a mapping")
        schema_metadata[b"rex_metadata"] = canonical_json(dict(metadata))
    if schema_metadata:
        table = table.replace_schema_metadata(schema_metadata)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def export_parquet(
    data: Mapping[str, Any],
    *,
    partition_digest: str,
    key_id: str | None = None,
    keys=None,
) -> tuple[bytes, ExportManifest]:
    """Return plain Parquet or one whole-artifact encrypted envelope and its manifest.

    ``key_id`` selects the single RexGraph envelope key. It does not select Parquet
    modular column keys, and consumers must not infer per-column isolation. The whole
    artifact must be decrypted before PyArrow can project or filter it.
    """
    if not isinstance(partition_digest, str) or not partition_digest:
        raise ValueError("partition_digest must be a nonempty string")
    arrays = _arrays(data)
    schema_digest = manifest_digest(
        {
            "object_type": "ParquetSchema",
            "schema": _schema(arrays),
            "version": EXPORT_VERSION,
        }
    )
    raw = parquet_bytes(
        arrays,
        metadata={
            "partition_digest": partition_digest,
            "schema_digest": schema_digest,
        },
    )
    if key_id is not None:
        if not isinstance(key_id, str) or not key_id:
            raise ValueError("encrypted export key_id must be a nonempty string")
        if keys is None:
            raise ValueError("encrypted export requires a KeyProvider")
        from .security import encrypt_bytes

        payload = encrypt_bytes(raw, key_id=key_id, keys=keys, object_type="Parquet")
        encrypted = True
    else:
        payload = raw
        encrypted = False
    manifest = ExportManifest(
        partition_digest,
        schema_digest,
        hashlib.sha256(payload).hexdigest(),
        encrypted,
        key_id,
    )
    return payload, manifest


def _parquet_metadata(payload: bytes) -> dict[str, Any] | None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        metadata = pq.read_metadata(pa.BufferReader(payload)).metadata or {}
        encoded = metadata.get(b"rex_metadata")
        value = json.loads(encoded.decode("utf-8"))
    except Exception:  # noqa: BLE001 - malformed Parquet fails verification
        return None
    return value if isinstance(value, dict) else None


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def verify_export(payload: bytes, manifest: ExportManifest, *, keys=None) -> bool:
    """Verify bytes, envelope metadata, and embedded lineage against ``manifest``.

    Encrypted exports require their key provider because their embedded partition and
    schema identities cannot be compared without authenticating the ciphertext.
    """
    if not isinstance(manifest, ExportManifest):
        return False
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return False
    raw = bytes(payload)
    if not _is_sha256(manifest.payload_sha256) or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), manifest.payload_sha256
    ):
        return False
    if manifest.encrypted:
        if not isinstance(manifest.key_id, str) or not manifest.key_id or keys is None:
            return False
        try:
            from .security import decrypt_bytes, envelope_info

            info = envelope_info(raw)
            parquet = decrypt_bytes(raw, keys=keys)
        except Exception:  # noqa: BLE001 - malformed envelopes fail verification
            return False
        if info.object_type != "Parquet" or info.key_id != manifest.key_id:
            return False
    else:
        if manifest.key_id is not None:
            return False
        parquet = raw
    embedded = _parquet_metadata(parquet)
    return bool(
        embedded is not None
        and embedded.get("partition_digest") == manifest.partition_digest
        and embedded.get("schema_digest") == manifest.schema_digest
    )
