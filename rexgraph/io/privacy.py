"""Scoped pseudonymous projections for derived RexGraph artifacts."""
from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .manifest import manifest_digest

PRIVACY_VERSION = 1

__all__ = [
    "PRIVACY_VERSION",
    "IdentityKeyProvider",
    "PrivacyProjection",
    "StaticIdentityKeyProvider",
    "project_rows",
    "scoped_pseudonym",
]


class IdentityKeyProvider(Protocol):
    """Resolve pseudonym material by opaque key identity."""

    def key(self, key_id: str) -> bytes: ...


@dataclass(frozen=True)
class StaticIdentityKeyProvider:
    """Small in-process identity key provider for tests and local deployments."""

    keys: Mapping[str, bytes]

    def key(self, key_id: str) -> bytes:
        try:
            value = self.keys[str(key_id)]
        except KeyError as exc:
            raise KeyError(f"unknown identity key id {key_id!r}") from exc
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError("identity key providers must return bytes")
        value = bytes(value)
        if not value:
            raise ValueError("identity keys may not be empty")
        return value


@dataclass(frozen=True)
class PrivacyProjection:
    """Canonical field projection and scoped pseudonym policy."""

    fields: tuple[str, ...]
    pseudonym_fields: tuple[str, ...] = ()
    scope: str = ""
    key_id: str | None = None

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        pseudonyms = tuple(self.pseudonym_fields)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "pseudonym_fields", pseudonyms)
        if any(not isinstance(field, str) or not field for field in fields):
            raise ValueError("projection fields must be nonempty strings")
        if any(not isinstance(field, str) or not field for field in pseudonyms):
            raise ValueError("pseudonym fields must be nonempty strings")
        if len(set(fields)) != len(fields):
            raise ValueError("projection fields must be unique")
        if len(set(pseudonyms)) != len(pseudonyms):
            raise ValueError("pseudonym fields must be unique")
        if pseudonyms and (not isinstance(self.scope, str) or not self.scope):
            raise ValueError("pseudonymous projection requires a nonempty scope")
        if pseudonyms and (not isinstance(self.key_id, str) or not self.key_id):
            raise ValueError("pseudonymous projection requires a nonempty key_id")
        missing = set(pseudonyms) - set(fields)
        if missing:
            raise ValueError(f"pseudonym fields are not projected: {sorted(missing)!r}")

    @property
    def digest(self) -> str:
        """Return the stable identity of the disclosure policy."""
        return manifest_digest(
            {
                "fields": list(self.fields),
                "key_id": self.key_id,
                "object_type": "PrivacyProjection",
                "pseudonym_fields": list(self.pseudonym_fields),
                "scope": self.scope,
                "version": PRIVACY_VERSION,
            }
        )


def scoped_pseudonym(
    value: Any,
    *,
    scope: str,
    key_id: str,
    keys: IdentityKeyProvider,
) -> str:
    """Return a deterministic pseudonym that cannot be joined across scopes by value."""
    if not isinstance(scope, str) or not scope:
        raise ValueError("pseudonym scope must be a nonempty string")
    if not isinstance(key_id, str) or not key_id:
        raise ValueError("pseudonym key_id must be a nonempty string")
    if keys is None:
        raise ValueError("pseudonym generation requires an IdentityKeyProvider")
    scope_bytes = scope.encode("utf-8")
    value_bytes = str(value).encode("utf-8")
    framed = (
        b"rexgraph-pseudonym\x00"
        + len(scope_bytes).to_bytes(4, "big")
        + scope_bytes
        + len(value_bytes).to_bytes(8, "big")
        + value_bytes
    )
    token = hmac.new(keys.key(key_id), framed, hashlib.sha256).digest()
    return base64.b32encode(token).decode("ascii").rstrip("=").lower()


def project_rows(
    rows: Iterable[Mapping[str, Any]],
    projection: PrivacyProjection,
    *,
    keys: IdentityKeyProvider | None = None,
) -> list[dict[str, Any]]:
    """Return plain mappings containing exactly the fields authorized by ``projection``."""
    if not isinstance(projection, PrivacyProjection):
        raise TypeError("projection must be a PrivacyProjection")
    pseudonyms = set(projection.pseudonym_fields)
    out = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("projected rows must be mappings")
        item = {}
        for field in projection.fields:
            value = row[field]
            if field in pseudonyms:
                if keys is None:
                    raise ValueError(
                        "pseudonymous projection requires an IdentityKeyProvider"
                    )
                value = scoped_pseudonym(
                    value,
                    scope=projection.scope,
                    key_id=str(projection.key_id),
                    keys=keys,
                )
            item[field] = value
        out.append(item)
    return out
