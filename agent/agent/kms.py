"""
agent.kms: the key context a rexgraph container is sealed and opened with.

``rexgraph.io`` takes an opaque properties object and never sees key material. It calls
``seal(key_id, plaintext, aad)`` and ``open(envelope, aad)`` and records only the key
IDENTIFIER, so core carries no crypto policy and no dependency on this module. The
implementation lives here because resolving an identifier to a key is an authorization
question, and authorization is an agent concern.

A key identifier is namespaced by workspace exactly as a saved connection is, because it
resolves through the same scoped store: the same identifier in two workspaces names two
different keys, and neither tenant can name the other one. An identifier that arrived in
a REQUEST is refused unless the operator listed it, the same rule
``secrets.resolve_request_ref`` applies, because a request must not have config's reach.

One data key is derived per identifier per process and held in memory only. It is never
serialized, never written into a manifest and never logged. HKDF salts the derivation
with the identifier, so two identifiers backed by the same stored secret still produce
different keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

NONCE_BYTES = 12
KEY_BYTES = 32
MAGIC = b"RXK1"


def _aesgcm():
    """The AEAD primitive, imported at call time so importing this module is cheap."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:                       # pragma: no cover
        raise RuntimeError(
            "sealing a container needs the 'cryptography' package") from exc
    return AESGCM


def _derive(secret: str, key_id: str) -> bytes:
    """A 32 byte key from a stored secret, salted by the identifier that named it."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES,
                salt=key_id.encode("utf-8"),
                info=b"rexgraph container key").derive(secret.encode("utf-8"))


@dataclass
class WorkspaceKeyring:
    """Seals and opens container envelopes with keys this workspace can name.

    Satisfies both `ContainerEncryptionProperties` and `ContainerDecryptionProperties`,
    because a process that writes a container almost always reads one back and splitting
    them would mean two objects resolving the same identifiers by two paths.

    `caller_named` says the identifiers came from a request rather than from operator
    configuration. It changes how they resolve, not how they are used.
    """

    configuration: object = None
    caller_named: bool = False
    authenticated_encryption: bool = True

    def __post_init__(self):
        self._keys: dict[str, bytes] = {}

    def _material(self, key_id: str) -> bytes:
        if not isinstance(key_id, str) or not key_id:
            raise PermissionError("a key identifier must be a nonempty string")
        if key_id in self._keys:
            return self._keys[key_id]
        secret = self._resolve(key_id)
        if not secret:
            raise PermissionError(
                f"no key material for {key_id!r} in this workspace")
        self._keys[key_id] = _derive(secret, key_id)
        return self._keys[key_id]

    def _resolve(self, key_id: str) -> str:
        from agent.secrets import resolve_ref, resolve_request_ref
        if self.caller_named:
            # Raises PermissionError when the operator has not listed it, which is the
            # point: a caller naming a key identifier must not reach further than the
            # operator allowed, and this is the path that buys key material rather than
            # a listing.
            return resolve_request_ref(key_id)
        # The workspace's own keyring first. secret_store() is the per-request scoped
        # view, so this is where namespacing happens; open_secret_store() would reach
        # the flat namespace and hand one tenant another tenant's keys.
        try:
            from agent.server.scope import secret_store
            found = secret_store().get(key_id)
            if found:
                return found
        except Exception:                            # noqa: BLE001 - not stored here
            pass
        return resolve_ref(key_id)

    @staticmethod
    def _split(envelope: bytes) -> tuple[bytes, bytes]:
        least = len(MAGIC) + NONCE_BYTES
        if not isinstance(envelope, (bytes, bytearray)) or len(envelope) <= least:
            raise ValueError("not a container envelope")
        if bytes(envelope[:len(MAGIC)]) != MAGIC:
            raise ValueError("not a container envelope")
        body = bytes(envelope[len(MAGIC):])
        return body[:NONCE_BYTES], body[NONCE_BYTES:]

    def seal(self, key_id: str, plaintext: bytes, aad: bytes) -> bytes:
        nonce = os.urandom(NONCE_BYTES)
        sealed = _aesgcm()(self._material(key_id)).encrypt(nonce, plaintext, aad)
        return MAGIC + nonce + sealed

    def open(self, envelope: bytes, aad: bytes) -> bytes:
        """Open an envelope this keyring sealed.

        The identifier is not carried in the envelope. The manifest already records
        which key each member was sealed under and is itself authenticated, so putting
        it in the envelope too would be a second copy that could disagree with the
        first.
        """
        nonce, sealed = self._split(envelope)
        last = None
        for key in self._candidates():
            try:
                return _aesgcm()(key).decrypt(nonce, sealed, aad)
            except Exception as exc:                 # noqa: BLE001 - try the next key
                last = exc
        raise PermissionError("no key in this workspace opens this envelope") from last

    def open_with(self, key_id: str, envelope: bytes, aad: bytes) -> bytes:
        """Open an envelope whose key the caller already knows, in one attempt.

        Use this wherever the identifier came out of an AUTHENTICATED manifest. It is
        not attacker-chosen there, so resolving it is safe, and naming the key directly
        avoids both the trial loop and the need to prime a reader with keys it will not
        use. `open` remains for the case where the reader has only the envelope.
        """
        nonce, sealed = self._split(envelope)
        try:
            return _aesgcm()(self._material(key_id)).decrypt(nonce, sealed, aad)
        except PermissionError:
            raise
        except Exception as exc:                     # noqa: BLE001 - wrong key or tamper
            raise PermissionError(
                f"{key_id!r} does not open this envelope") from exc

    def _candidates(self):
        """Every key already derived, newest first. Nothing is resolved here.

        `open` is handed an envelope and the AAD, never an identifier, so it tries what
        this keyring holds. Resolving an identifier that did NOT come from an
        authenticated manifest would let a reader probe which ones exist by watching
        which attempts take longer; `open_with` is the path for one that did.
        """
        return list(reversed(list(self._keys.values())))

    def load(self, *key_ids: str) -> WorkspaceKeyring:
        """Resolve these identifiers now, so `open` has them to try. Returns self.

        Prime exactly what a read will touch. Priming every identifier a manifest
        mentions would resolve keys for members the read never opens, which is the cost
        a projection exists to avoid.
        """
        for key_id in key_ids:
            self._material(key_id)
        return self

    def key(self, key_id: str) -> bytes:
        """The key for this identifier, satisfying `IndexKeyProvider`.

        `rcdb_protected_index` takes a provider rather than key bytes, and this is what
        makes a search token workspace scoped: the identifier resolves through the scoped
        secret store, so the same name in two workspaces derives two different keys and
        one tenant's tokens are meaningless to another. Constructing a
        StaticIndexKeyProvider inside a request would hand every tenant the same key.
        """
        return self._material(key_id)

    def holds(self, key_id: str) -> bool:
        """Whether this identifier has already been resolved in this process."""
        return key_id in self._keys
