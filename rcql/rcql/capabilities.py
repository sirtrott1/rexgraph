"""Capability scoped RCQL sources."""
from __future__ import annotations

from dataclasses import dataclass

from .types import SourceRef, TemporalRef


@dataclass(frozen=True)
class SourcePolicy:
    """Explicit operations and structural fields allowed through one source binding."""

    permissions: frozenset[str]
    record_fields: frozenset[str] | None = None

    @classmethod
    def allow(cls, *permissions: str, record_fields=None):
        fields = None if record_fields is None else frozenset(str(x) for x in record_fields)
        return cls(frozenset(str(x).lower() for x in permissions), fields)

    def permits(self, permission: str) -> bool:
        return "*" in self.permissions or str(permission).lower() in self.permissions

    @classmethod
    def intersection(cls, *policies: SourcePolicy) -> SourcePolicy:
        """Return the policy safe for a value jointly derived from every input.

        A multi-stalk phrase may read each stalk under its own policy, but a result that
        contains information from all stalks can be exposed only under permissions every
        contributor grants.  This is intentionally not a source-policy merge: union
        would escalate one stalk through another, while applying this intersection to
        source reads would incorrectly deny legal independent reads.
        """
        if not policies:
            return cls.allow()
        if all("*" in policy.permissions for policy in policies):
            permissions = frozenset({"*"})
        else:
            candidates = set().union(*(policy.permissions for policy in policies))
            candidates.discard("*")
            permissions = frozenset(
                permission for permission in candidates
                if all(policy.permits(permission) for policy in policies)
            )
        bounded_fields = [policy.record_fields for policy in policies
                          if policy.record_fields is not None]
        fields = None if not bounded_fields else frozenset.intersection(*bounded_fields)
        return cls(permissions, fields)

    def project_record(self, value):
        """Project a bounded RCDB record view without granting hidden identity."""
        if isinstance(value, list):
            return [self.project_record(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.project_record(item) for item in value)
        if not isinstance(value, dict):
            return value
        out = dict(value)
        if not self.permits("identity"):
            out.pop("id", None)
        fields = self.record_fields
        if fields is not None and isinstance(out.get("signature"), dict):
            out["signature"] = {key: out["signature"][key] for key in sorted(fields)
                                if key in out["signature"]}
        return out

    @property
    def digest(self) -> str:
        from rexgraph.io.manifest import manifest_digest
        return manifest_digest({
            "object_type": "RCQLSourcePolicy",
            "permissions": sorted(self.permissions),
            "record_fields": None if self.record_fields is None else sorted(self.record_fields),
            "version": 1,
        })


@dataclass(frozen=True)
class BoundSource:
    """A live source and the capabilities retained while deriving child sources."""

    value: object
    policy: SourcePolicy
    ref: SourceRef | None = None
    temporal: TemporalRef | None = None

    def require(self, permission: str):
        if not self.policy.permits(permission):
            raise PermissionError(f"RCQL source does not permit {permission!r}")
        return self.value
