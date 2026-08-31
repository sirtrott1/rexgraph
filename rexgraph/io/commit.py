"""Canonical lineage links over TemporalRex transitions."""
from __future__ import annotations

from dataclasses import dataclass

from .manifest import canonical_json, manifest_digest
from .security import Signer, Verifier

COMMIT_VERSION = 1

__all__ = ["COMMIT_VERSION", "CommitLink"]


@dataclass(frozen=True)
class CommitLink:
    """Place one transition in an append-only lineage."""

    transition_digest: str
    parent_digest: str | None = None
    signer_id: str | None = None
    signature: bytes | None = None

    def manifest(self) -> dict:
        """Return the unsigned canonical lineage fields."""
        return {
            "parent_digest": self.parent_digest,
            "transition_digest": self.transition_digest,
            "version": COMMIT_VERSION,
        }

    @property
    def digest(self) -> str:
        """Return the stable identity of this unsigned lineage link."""
        return manifest_digest({"object_type": "CommitLink", **self.manifest()})

    def signing_bytes(self) -> bytes:
        """Return domain-separated bytes covered by a lineage signature."""
        return canonical_json({"digest": self.digest, "object_type": "CommitLink"})

    def signed(self, signer: Signer) -> CommitLink:
        """Return a copy signed by ``signer`` without changing link identity."""
        return CommitLink(
            self.transition_digest,
            self.parent_digest,
            signer.signer_id,
            signer.sign(self.signing_bytes()),
        )

    def verify(self, verifier: Verifier) -> bool:
        """Return whether the signature is valid; callers must reject a false result."""
        return bool(
            self.signature is not None
            and self.signer_id == verifier.signer_id
            and verifier.verify(self.signing_bytes(), self.signature)
        )
