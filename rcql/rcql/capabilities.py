"""Capability scoped RCQL sources."""
from __future__ import annotations

from dataclasses import dataclass


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

    def require(self, permission: str):
        if not self.policy.permits(permission):
            raise PermissionError(f"RCQL source does not permit {permission!r}")
        return self.value
