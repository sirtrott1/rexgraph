"""
agent.secrets: pluggable secret storage for database connections.

Connection URIs carry credentials. This module puts them behind one interface
so the file-backed default can be swapped for a real secrets manager
(Vault/KMS) without touching call sites. The env-reference backend models the
production pattern: config stores a *reference* (an env var / secret path), and
the real secret is fetched at resolve time - never persisted by us.

Select via ``REXGRAPH_SECRETS_URI``:
  * ``file://…``  (default) - FileSecretStore, a local JSON store.
  * ``env://``    - EnvSecretStore, URIs resolved from environment references.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import os
import re
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


class FileSecretStore(SecretStore):
    """Local JSON store (development default). Masks on list."""

    def __init__(self, path: str):
        self.path = os.path.expanduser(path)

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self, data: dict) -> None:
        # This file holds connection URIs WITH embedded credentials in plaintext.
        # Create it owner-only (0o600) and lock down the parent dir (0o700) so
        # other local users can't read stored secrets. For production, prefer the
        # env:// backend (REXGRAPH_SECRETS_URI=env://) over a file.
        parent = os.path.dirname(self.path)
        os.makedirs(parent, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(parent, 0o700)
        tmp = self.path + ".tmp"
        # Open with 0o600 from the start so the secrets never briefly exist world-readable.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)
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
    """Reference-based store modeling a real secrets manager: config holds a
    *reference* (an env var name); the secret is fetched from the environment
    at resolve time and never persisted here. The same shape a Vault/KMS
    backend takes - swap ``os.environ`` for the vault client.
    """

    def __init__(self, index_path: str = "~/.config/rexgraph/secret_refs.json"):
        self.path = os.path.expanduser(index_path)

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f)

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


def open_secret_store(uri: str = None) -> SecretStore:
    """Open the configured secret store (``REXGRAPH_SECRETS_URI``)."""
    uri = uri or os.environ.get("REXGRAPH_SECRETS_URI") \
        or "file://~/.config/rexgraph/connections.json"
    if uri.startswith("env://"):
        return EnvSecretStore()
    if uri.startswith("file://"):
        return FileSecretStore(uri[len("file://"):])
    # A bare path is a file store. Anything carrying an unsupported scheme is a
    # configuration error: falling through to FileSecretStore(uri) used to create a
    # file literally named "vault://team/prod" and report success, so every secret
    # went somewhere the operator did not intend.
    if "://" in uri:
        scheme = uri.split("://", 1)[0]
        raise ValueError(
            f"unsupported secret-store scheme {scheme!r} in {uri!r}: "
            f"supported schemes are env:// and file://, or pass a bare filesystem path. "
            f"Register a backend before using {scheme}://.")
    return FileSecretStore(uri)
