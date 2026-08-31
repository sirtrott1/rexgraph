"""Canonical digests for small RexGraph manifests."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

FORMAT_VERSION = 1

__all__ = ["FORMAT_VERSION", "canonical_json", "digest_parts", "manifest_digest"]


def canonical_json(value: Any) -> bytes:
    """Return one deterministic UTF-8 encoding for JSON-safe metadata."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def manifest_digest(value: Any, *, algorithm: str = "sha256") -> str:
    """Digest JSON-safe metadata with explicit framing and format identity."""
    payload = canonical_json(value)
    digest = hashlib.new(algorithm)
    digest.update(b"rexgraph-manifest\x00")
    digest.update(FORMAT_VERSION.to_bytes(2, "big"))
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
