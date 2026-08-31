"""Checkpoint plus ordered, verified mutation replication packages.

The transport's unkeyed SHA-256 chain detects corruption and inconsistent ordering; it
does not authenticate a producer. An active adversary can fabricate a self-consistent
unsigned chain when :func:`apply_replication` uses its default no-signature policy.
Require producer signatures through ``MutationPolicy`` and supply the corresponding
verifiers when authenticity is a security boundary. A signed chain cannot be downgraded
silently because each mutation binds its policy digest.
"""
from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .manifest import manifest_digest
from .transport import pack as pack_transport
from .transport import unpack as unpack_transport

REPLICATION_VERSION = 1
MAX_MUTATIONS = 10_000_000

__all__ = [
    "MAX_MUTATIONS",
    "REPLICATION_VERSION",
    "AppliedReplication",
    "ReplicationManifest",
    "apply_replication",
    "pack_replication",
    "unpack_replication",
]


@dataclass(frozen=True)
class ReplicationManifest:
    """Canonical identity for one checkpoint plus an ordered mutation stream."""

    checkpoint_sha256: str
    mutation_sha256: tuple[str, ...]
    checkpoint_state: str = ""
    checkpoint_commit: str = ""
    terminal_state: str = ""

    @property
    def digest(self) -> str:
        """Return the stable replication inventory identity."""
        return manifest_digest(
            {
                "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_state": self.checkpoint_state,
                "checkpoint_commit": self.checkpoint_commit,
                "mutation_sha256": list(self.mutation_sha256),
                "object_type": "ReplicationManifest",
                "terminal_state": self.terminal_state,
                "version": REPLICATION_VERSION,
            }
        )


@dataclass(frozen=True)
class AppliedReplication:
    """A checkpoint, its verified ordered packages, and the exact resulting RexGraph."""

    checkpoint: object | None
    result: object | None
    packages: tuple[object, ...]
    manifest: ReplicationManifest


def _bytes(value: Any, field: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{field} must be bytes")
    return bytes(value)


def _sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return value == value.lower()


def _frame(chunks: Iterable[bytes]) -> bytes:
    out = bytearray()
    for chunk in chunks:
        raw = _bytes(chunk, "replication chunk")
        out += len(raw).to_bytes(8, "big")
        out += raw
    return bytes(out)


def _unframe(payload: bytes, count: int) -> list[bytes]:
    if count < 0 or count > MAX_MUTATIONS + 1:
        raise ValueError("invalid replication chunk count")
    if count > len(payload) // 8:
        raise ValueError("replication chunk count exceeds framed payload")
    out = []
    at = 0
    for _ in range(count):
        if at + 8 > len(payload):  # pragma: no cover - preflight count check
            raise ValueError("truncated replication payload")
        length = int.from_bytes(payload[at:at + 8], "big")
        at += 8
        if length > len(payload) - at:
            raise ValueError("invalid replication chunk length")
        out.append(payload[at:at + length])
        at += length
    if at != len(payload):
        raise ValueError("replication payload has trailing bytes")
    return out


def _mutation_lineage(
    mutations: Iterable[bytes],
    checkpoint_state: str = "",
    checkpoint_commit: str = "",
):
    """Check package decoding and claimed ordering without materializing checkpoints."""
    from .mutation import mutation_from_bytes

    packages = tuple(mutation_from_bytes(raw) for raw in mutations)
    parent = checkpoint_commit or None
    expected_state = checkpoint_state
    for package in packages:
        if package.link.parent_digest != parent:
            raise ValueError("replication mutation lineage is not contiguous")
        if expected_state and package.transition.previous_state != expected_state:
            raise ValueError("replication mutation previous state does not match lineage")
        expected_state = package.transition.resulting_state
        parent = package.link.digest
    return packages, expected_state


def pack_replication(
    checkpoint: bytes,
    mutations: Iterable[bytes],
    *,
    checkpoint_state: str = "",
    checkpoint_commit: str = "",
) -> tuple[bytes, ReplicationManifest]:
    """Frame one checkpoint and an ordered v2 mutation stream with independent digests."""
    checkpoint_bytes = _bytes(checkpoint, "replication checkpoint")
    mutation_bytes = tuple(_bytes(value, "replication mutation") for value in mutations)
    if len(mutation_bytes) > MAX_MUTATIONS:
        raise ValueError("replication mutation count exceeds its bound")
    if not isinstance(checkpoint_state, str) or not isinstance(checkpoint_commit, str):
        raise TypeError("checkpoint state and commit identities must be strings")
    _packages, terminal = _mutation_lineage(
        mutation_bytes, checkpoint_state, checkpoint_commit
    )
    manifest = ReplicationManifest(
        hashlib.sha256(checkpoint_bytes).hexdigest(),
        tuple(hashlib.sha256(value).hexdigest() for value in mutation_bytes),
        checkpoint_state,
        checkpoint_commit,
        terminal or checkpoint_state,
    )
    payload = _frame((checkpoint_bytes, *mutation_bytes))
    blob = pack_transport(
        payload,
        object_type="ReplicationPackage",
        metadata={
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "checkpoint_state": manifest.checkpoint_state,
            "checkpoint_commit": manifest.checkpoint_commit,
            "manifest_digest": manifest.digest,
            "mutation_count": len(mutation_bytes),
            "mutation_sha256": list(manifest.mutation_sha256),
            "terminal_state": manifest.terminal_state,
            "version": REPLICATION_VERSION,
        },
    )
    return blob, manifest


def _metadata_manifest(metadata: Mapping[str, Any]) -> tuple[int, ReplicationManifest]:
    version = metadata.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != REPLICATION_VERSION
    ):
        raise ValueError("unsupported replication package version")
    count = metadata.get("mutation_count")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        or count > MAX_MUTATIONS
    ):
        raise ValueError("invalid replication mutation count")
    raw_hashes = metadata.get("mutation_sha256")
    if not isinstance(raw_hashes, list) or len(raw_hashes) != count:
        raise ValueError("replication mutation digest count mismatch")
    hashes = tuple(raw_hashes)
    checkpoint_hash = metadata.get("checkpoint_sha256")
    if not _sha256(checkpoint_hash) or any(not _sha256(value) for value in hashes):
        raise ValueError("replication contains an invalid artifact digest")
    identities = []
    for field in ("checkpoint_state", "checkpoint_commit", "terminal_state"):
        value = metadata.get(field, "")
        if not isinstance(value, str):
            raise ValueError(f"replication {field} must be a string")
        identities.append(value)
    manifest = ReplicationManifest(checkpoint_hash, hashes, *identities)
    declared = metadata.get("manifest_digest")
    if not _sha256(declared) or not hmac.compare_digest(manifest.digest, declared):
        raise ValueError("replication manifest digest mismatch")
    return count, manifest


def unpack_replication(blob: bytes) -> tuple[bytes, tuple[bytes, ...], ReplicationManifest]:
    """Verify framing, byte hashes, and claimed order, then return the inventory.

    This does not claim that mutations follow the checkpoint state because it does not
    materialize that state. Use :func:`apply_replication` for endpoint verification.
    """
    payload, outer = unpack_transport(_bytes(blob, "replication package"), verify=True)
    if outer.get("object_type") != "ReplicationPackage":
        raise TypeError("transport payload is not a ReplicationPackage")
    metadata = outer.get("metadata")
    if not isinstance(metadata, dict):  # transport validates this; retain local contract
        raise ValueError("replication package metadata is invalid")
    count, manifest = _metadata_manifest(metadata)
    chunks = _unframe(payload, count + 1)
    if not hmac.compare_digest(
        hashlib.sha256(chunks[0]).hexdigest(), manifest.checkpoint_sha256
    ):
        raise ValueError("replication checkpoint digest mismatch")
    for raw, expected in zip(chunks[1:], manifest.mutation_sha256, strict=True):
        if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
            raise ValueError("replication mutation digest mismatch")
    _packages, terminal = _mutation_lineage(
        chunks[1:], manifest.checkpoint_state, manifest.checkpoint_commit
    )
    if (terminal or manifest.checkpoint_state) != manifest.terminal_state:
        raise ValueError("replication terminal state mismatch")
    return chunks[0], tuple(chunks[1:]), manifest


def apply_replication(
    blob: bytes,
    *,
    checkpoint_loader: Callable[[bytes], object | None],
    policy=None,
    verifiers: Mapping[str, Any] | None = None,
) -> AppliedReplication:
    """Verify, load, and apply a complete ordered replication package.

    The loader is injected so core remains independent of RCDB and storage backends. It
    must return the checkpoint ``RexGraph`` or ``None`` for an explicit empty genesis
    checkpoint. Every mutation is verified against the real prior state before its
    carried canonical result becomes the next state. Authentication additionally requires
    a signature-requiring ``policy`` and its ``verifiers``; the default policy verifies
    integrity and consistency but does not require a producer signature.
    """
    from rexgraph.graph import RexGraph

    from .catalog import object_digest
    from .mutation import MutationPolicy, mutation_from_bytes, verify_mutation
    from .rex_state import from_state

    if not callable(checkpoint_loader):
        raise TypeError("checkpoint_loader must be callable")
    checkpoint_bytes, mutation_bytes, manifest = unpack_replication(blob)
    checkpoint = checkpoint_loader(checkpoint_bytes)
    if checkpoint is None:
        if checkpoint_bytes:
            raise ValueError("a genesis replication checkpoint must be empty bytes")
        if manifest.checkpoint_state or manifest.checkpoint_commit:
            raise ValueError("replication checkpoint identities require a RexGraph")
    elif not isinstance(checkpoint, RexGraph):
        raise TypeError("checkpoint_loader must return a RexGraph or None")
    elif not manifest.checkpoint_state:
        raise ValueError("a loaded checkpoint requires checkpoint_state identity")
    elif object_digest(checkpoint) != manifest.checkpoint_state:
        raise ValueError("loaded replication checkpoint state does not match manifest")

    mutation_policy = MutationPolicy() if policy is None else policy
    if not isinstance(mutation_policy, MutationPolicy):
        raise TypeError("policy must be a MutationPolicy")
    verifier_map = dict(verifiers or {})
    packages = tuple(mutation_from_bytes(raw) for raw in mutation_bytes)
    current = checkpoint
    parent = manifest.checkpoint_commit or None
    for package in packages:
        if not verify_mutation(
            package,
            previous=current,
            policy=mutation_policy,
            verifiers=verifier_map,
            parent_digest=parent,
        ):
            raise ValueError("replication mutation failed endpoint or policy verification")
        current = from_state(package.resulting_state)
        parent = package.link.digest
    terminal = "" if current is None else object_digest(current)
    if terminal != manifest.terminal_state:
        raise ValueError("applied replication state does not match terminal identity")
    return AppliedReplication(checkpoint, current, packages, manifest)
