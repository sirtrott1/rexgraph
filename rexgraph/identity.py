"""Container independent identity primitives for RexGraph values and artifacts.

These functions frame metadata and tensor bytes.  They know nothing about HDF5, Zarr,
RexState, RCDB, or RexGraph construction, which makes them safe dependencies for math,
model, and persistence layers alike.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np

MANIFEST_FORMAT_VERSION = 1
TENSOR_DIGEST_ALGO = 2


def canonical_json(value: Any) -> bytes:
    """Return one deterministic UTF-8 encoding for JSON safe metadata."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def manifest_digest(value: Any, *, algorithm: str = "sha256") -> str:
    """Digest JSON safe metadata with explicit framing and format identity."""
    payload = canonical_json(value)
    digest = hashlib.new(algorithm)
    digest.update(b"rexgraph-manifest\x00")
    digest.update(MANIFEST_FORMAT_VERSION.to_bytes(2, "big"))
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def digest_parts(
    kind: str,
    parts: Sequence[tuple[str, str]],
    *,
    algorithm: str = "sha256",
) -> str:
    """Digest named hexadecimal digests without ambiguous concatenation."""
    digest = hashlib.new(algorithm)
    digest.update(b"rexgraph-digest-parts\x00")
    kind_bytes = str(kind).encode("utf-8")
    digest.update(len(kind_bytes).to_bytes(4, "big"))
    digest.update(kind_bytes)
    for name, value in parts:
        name_bytes = str(name).encode("utf-8")
        value_bytes = str(value).encode("ascii")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(value_bytes).to_bytes(4, "big"))
        digest.update(value_bytes)
    return digest.hexdigest()


def tensor_digest(tensors: dict, names=None, *, algo: int = TENSOR_DIGEST_ALGO) -> str:
    """Order independent SHA-256 over named non object tensor payloads.

    Algorithm 1 is the historical unframed stream retained only for verification of
    existing states.  Algorithm 2 length prefixes every name/dtype/shape/payload field.
    """
    algo = int(algo)
    if algo not in (1, TENSOR_DIGEST_ALGO):
        raise ValueError(f"unsupported tensor digest algorithm {algo}")
    digest = hashlib.sha256()
    legacy = algo == 1
    for name in (sorted(tensors) if names is None else list(names)):
        arr = np.ascontiguousarray(tensors[name])
        if arr.dtype.hasobject:
            raise TypeError("object tensors must use an exact coefficient codec before hashing")
        for part in (
            str(name).encode("utf-8"),
            str(arr.dtype).encode("utf-8"),
            str(arr.shape).encode("utf-8"),
            arr.tobytes(),
        ):
            if not legacy:
                digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    return digest.hexdigest()


__all__ = [
    "MANIFEST_FORMAT_VERSION",
    "TENSOR_DIGEST_ALGO",
    "canonical_json",
    "digest_parts",
    "manifest_digest",
    "tensor_digest",
]
