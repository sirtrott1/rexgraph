"""
agent.secrets: pluggable secret storage for database connections.

Connection URIs carry credentials. This module puts them behind one interface
so the file backed default can be swapped for a real secrets manager
(Vault/KMS) without touching call sites. The env reference backend models the
production pattern: config stores a *reference* (an env var / secret path), and
the real secret is fetched at resolve time and never persisted by us.

Select via ``REXGRAPH_SECRETS_URI``:
  * ``file://…``  (default): FileSecretStore, a local JSON store.
  * ``env://``    - EnvSecretStore, URIs resolved from environment references.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import os
import re
import tempfile
from urllib.parse import urlparse, urlunparse


def mask_uri(uri: str) -> str:
    """Hide the password in a connection URI for display."""
    try:
        p = urlparse(uri)
        if p.password:
            netloc = p.netloc.replace(":" + p.password + "@", ":****@")
            return urlunparse(p._replace(netloc=netloc))
    except Exception:
        pass
    return re.sub(r"(://[^:/@]+:)[^@/]+(@)", r"\1****\2", uri)


class SecretStore:
    """Interface: store connection secrets, resolve them, list them masked."""

    def get(self, name: str) -> str:          # returns the uri WITH credentials
        raise NotImplementedError

    def put(self, name: str, uri: str, kind: str = "sql") -> None:
        raise NotImplementedError

    def list(self) -> builtins.list[dict]:             # masked; never returns raw creds
        raise NotImplementedError

    def delete(self, name: str) -> bool:
        raise NotImplementedError


def _load_store(path: str, field: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ValueError("unreadable secret store; refusing to modify") from exc
    if not isinstance(data, dict) or any(
        not isinstance(name, str) or not isinstance(record, dict)
        or not isinstance(record.get(field), str)
        or ("kind" in record and not isinstance(record["kind"], str))
        for name, record in data.items()
    ):
        raise ValueError("invalid secret store; refusing to modify")
    return data


def _save_store(path: str, data: dict) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".rex-secrets-", dir=parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class FileSecretStore(SecretStore):
    """Local JSON store (development default). Masks on list."""

    def __init__(self, path: str):
        self.path = os.path.expanduser(path)

    def _load(self) -> dict:
        return _load_store(self.path, "uri")

    def _save(self, data: dict) -> None:
        # This file holds connection URIs WITH embedded credentials in plaintext.
        # Create it owner only (0o600) and lock down the parent dir (0o700) so
        # other local users can't read stored secrets. For production, prefer the
        # env:// backend (REXGRAPH_SECRETS_URI=env://) over a file.
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(parent, 0o700)
        _save_store(self.path, data)
        with contextlib.suppress(OSError):
            os.chmod(self.path, 0o600)

    def get(self, name: str) -> str:
        rec = self._load().get(name)
        if not rec:
            raise KeyError(name)
        return rec["uri"]

    def put(self, name: str, uri: str, kind: str = "sql") -> None:
        data = self._load()
        data[name] = {"uri": uri, "kind": kind}
        self._save(data)

    def list(self) -> builtins.list[dict]:
        return [{"name": n, "kind": r.get("kind", "sql"), "uri": mask_uri(r["uri"])}
                for n, r in self._load().items()]

    def delete(self, name: str) -> bool:
        data = self._load()
        existed = data.pop(name, None) is not None
        self._save(data)
        return existed


class EnvSecretStore(SecretStore):
    """Reference based store modeling a real secrets manager: config holds a
    *reference* (an env var name); the secret is fetched from the environment
    at resolve time and never persisted here. The same shape a Vault/KMS
    backend takes: swap ``os.environ`` for the vault client.
    """

    def __init__(self, index_path: str = "~/.config/rexgraph/secret_refs.json"):
        self.path = os.path.expanduser(index_path)

    def _load(self) -> dict:
        return _load_store(self.path, "ref")

    def _save(self, data: dict) -> None:
        _save_store(self.path, data)

    def get(self, name: str) -> str:
        rec = self._load().get(name)
        if not rec:
            raise KeyError(name)
        val = os.environ.get(rec["ref"])
        if val is None:
            raise KeyError(f"secret reference '{rec['ref']}' not set in environment")
        return val

    def put(self, name: str, uri: str, kind: str = "sql") -> None:
        # `uri` is interpreted as a reference name (env var / vault path)
        data = self._load()
        data[name] = {"ref": uri, "kind": kind}
        self._save(data)

    def list(self) -> builtins.list[dict]:
        return [{"name": n, "kind": r.get("kind", "sql"), "uri": f"ref:{r['ref']}"}
                for n, r in self._load().items()]

    def delete(self, name: str) -> bool:
        data = self._load()
        existed = data.pop(name, None) is not None
        self._save(data)
        return existed


#: Which references a REQUEST may name, as a comma separated allow list. Empty or
#: unset denies every one, which is the safe default: a reference arriving in a request
#: is chosen by the caller, and `resolve_ref` reads any environment variable, so without
#: this a caller names AWS_SECRET_ACCESS_KEY and the credential leaves as a bearer
#: header on the first request routed to whatever endpoint they attached.
REQUEST_REFS_ENV = "REXGRAPH_REQUEST_KEY_REFS"


def request_refs_allowed() -> set:
    """The references an operator has agreed a request may name."""
    raw = os.environ.get(REQUEST_REFS_ENV, "")
    return {r.strip() for r in raw.split(",") if r.strip()}


def resolve_request_ref(ref: str) -> str:
    """Resolve a reference that arrived in a REQUEST rather than in operator config.

    Config is written by whoever runs the server and may name anything it likes. A
    request is written by whoever can reach the server, and the two must not have the
    same reach. Deny by default and let the operator name the exceptions: the set is
    small in practice, one entry per endpoint anyone is allowed to attach.

    Raises PermissionError rather than returning "" so a refusal is reported to the
    caller instead of degrading silently into an unauthenticated request, which would
    look like the reference was simply wrong.
    """
    if not ref:
        return ""
    if ref not in request_refs_allowed():
        raise PermissionError(
            f"{ref!r} is not an allowed request reference; an operator lists the "
            f"permitted ones in {REQUEST_REFS_ENV}")
    return resolve_ref(ref)


def resolve_ref(ref: str) -> str:
    """Resolve a secret *reference* to its value, or "" when it cannot be resolved.

    A reference is an environment variable name or a name in the configured secret store -
    never the secret itself. Config (a hive profile, a bee) holds only the reference, so a
    credential is fetched at call time and is never written to disk or serialized by us.

    Environment first (the cheap, container native case), then the secret store. Missing is
    not an error: callers degrade to an unauthenticated request rather than crashing, and an
    unresolved reference must never be sent as if it were a key.
    """
    if not ref:
        return ""
    val = os.environ.get(ref)
    if val:
        return val
    try:
        return open_secret_store().get(ref) or ""
    except Exception:
        return ""


def open_secret_store(uri: str = None) -> SecretStore:
    """Open the configured secret store (``REXGRAPH_SECRETS_URI``)."""
    uri = uri or os.environ.get("REXGRAPH_SECRETS_URI") \
        or "file://~/.config/rexgraph/connections.json"
    if uri.startswith("env://"):
        return EnvSecretStore()
    if uri.startswith("file://"):
        return FileSecretStore(uri[len("file://"):])
    # Bare paths select a file store. Unsupported schemes raise before file creation.
    if "://" in uri:
        scheme = uri.split("://", 1)[0]
        raise ValueError(
            f"unsupported secret-store scheme {scheme!r} in {uri!r}: "
            f"supported schemes are env:// and file://, or pass a bare filesystem path. "
            f"Register a backend before using {scheme}://.")
    return FileSecretStore(uri)
