"""Canonical identity for one TemporalRex state transition."""
from __future__ import annotations

from dataclasses import dataclass

from .manifest import canonical_json, manifest_digest
from .security import Signer, Verifier

TRANSITION_VERSION = 1

__all__ = ["TRANSITION_VERSION", "TransitionCommit"]


@dataclass(frozen=True)
class TransitionCommit:
    """The canonical identity and lineage of one relational state change."""

    previous_state: str
    delta_state: str
    resulting_state: str
    tx_time: float
    actor: str = ""
    policy: str = ""
    signer_id: str | None = None
    signature: bytes | None = None

    def manifest(self) -> dict:
        """Return the unsigned canonical transition fields."""
        return {
            "actor": self.actor,
            "delta_state": self.delta_state,
            "policy": self.policy,
            "previous_state": self.previous_state,
            "resulting_state": self.resulting_state,
            "tx_time": float(self.tx_time),
            "version": TRANSITION_VERSION,
        }

    @property
    def digest(self) -> str:
        """Return the stable identity of the unsigned transition."""
        return manifest_digest({"object_type": "TransitionCommit", **self.manifest()})

    def signing_bytes(self) -> bytes:
        """Return domain-separated bytes covered by a transition signature."""
        return canonical_json({"digest": self.digest, "object_type": "TransitionCommit"})

    def signed(self, signer: Signer) -> TransitionCommit:
        """Return a copy signed by ``signer`` without changing transition identity."""
        return TransitionCommit(
            self.previous_state,
            self.delta_state,
            self.resulting_state,
            self.tx_time,
            self.actor,
            self.policy,
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
