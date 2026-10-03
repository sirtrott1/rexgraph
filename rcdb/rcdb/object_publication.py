"""Explicit object publication capabilities; fsspec writes alone are insufficient.

A provider supplies complete prefix inspection, strong bounded reads, atomic
compare and swap and durability of acknowledged writes. Unknown outcomes raise;
the record adapter decides whether a publication point may have been crossed.
Capabilities are trusted process objects, never loaded from persisted metadata.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import posixpath
from threading import RLock
from weakref import WeakValueDictionary

from .core import VersionConflictError

_MEMORY_GUARD = RLock()
_MEMORY_GATES = WeakValueDictionary()


@dataclass(frozen=True)
class PublishedObject:
    """Owned bytes and an opaque current version token, kept out of store frames."""
    data: bytes
    token: bytes

    def __post_init__(self):
        if type(self.data) is not bytes or type(self.token) is not bytes or not 0 < len(self.token) <= 4096:
            raise ValueError("published objects require bytes and a bounded opaque version token")


class ObjectPublication(ABC):
    """Bound I/O capability; a negative CAS raises VersionConflictError.

    A returned CAS result acknowledges durable publication. Exceptions may mean
    the CAS applied without acknowledgement. Providers must not retry with a new
    precondition or return success for an unperformed operation. Tokens identify
    the bytes returned by the same strong read; they are not record cursors.
    """
    def __init__(self, fs, root):
        if type(root) is not str or not root:
            raise ValueError("object publication requires a declared prefix")
        self.fs, self.root = fs, root.rstrip("/") or "/"

    def path(self, key):
        if (type(key) is not str or not key or key in {".", ".."} or key.startswith("/")
                or posixpath.normpath(key) != key or key.startswith("../") or "\\" in key or "\x00" in key):
            raise ValueError("object publication requires a literal relative key")
        return posixpath.join(self.root, key)

    @abstractmethod
    def read(self, key, *, limit):
        """Return PublishedObject or None for absence; refuse oversized bytes."""

    @abstractmethod
    def compare_and_swap(self, key, expected, raw):
        """Publish iff the current token equals expected (None means absent)."""

    @abstractmethod
    def has_objects(self):
        """Completely inspect the prefix before creating a new store identity."""

    @contextmanager
    def transaction(self):
        """Optional writer arbitration; remote providers still arbitrate by CAS."""
        yield


class MemoryObjectPublication(ObjectPublication):
    """fsspec memory namespace: atomic within this process, without disk durability."""
    def __init__(self, fs, root):
        from fsspec.implementations.memory import MemoryFileSystem
        if not isinstance(fs, MemoryFileSystem):
            raise TypeError("memory publication requires the memory filesystem")
        super().__init__(fs, root)
        with _MEMORY_GUARD:
            self._gate = _MEMORY_GATES.get(self.root)
            if self._gate is None:
                self._gate = RLock()
                _MEMORY_GATES[self.root] = self._gate

    def read(self, key, *, limit):
        if type(limit) is not int or limit < 0:
            raise ValueError("object read limit requires a nonnegative native integer")
        with self._gate:
            try:
                with self.fs.open(self.path(key), "rb") as stream:
                    raw = stream.read(limit+1)
            except FileNotFoundError:
                return None
            if len(raw) > limit:
                raise ValueError("object storage exceeds its byte limit")
            return PublishedObject(raw, hashlib.sha256(raw).digest())

    def compare_and_swap(self, key, expected, raw):
        if type(raw) is not bytes or expected is not None and type(expected) is not bytes:
            raise TypeError("conditional publication requires native bytes")
        with self._gate:
            path = self.path(key)
            token = None
            if self.fs.exists(path):
                digest = hashlib.sha256()
                with self.fs.open(path, "rb") as stream:
                    while chunk := stream.read(1024*1024):
                        digest.update(chunk)
                token = digest.digest()
            if token != expected:
                raise VersionConflictError("object publication precondition differs from current bytes")
            self.fs.makedirs(posixpath.dirname(path), exist_ok=True)
            with self.fs.open(path, "wb") as stream:
                if stream.write(raw) != len(raw):
                    raise OSError("short conditional memory object write")
            return PublishedObject(raw, hashlib.sha256(raw).digest())

    def has_objects(self):
        with self._gate:
            return self.fs.exists(self.root) and bool(self.fs.ls(self.root, detail=False))

    @contextmanager
    def transaction(self):
        with self._gate:
            yield


class LocalObjectPublication(ObjectPublication):
    """Regular files: staged durable writes under the shared directory lock.

    POSIX directory flock supplies process exclusion. The shared publication
    layer's fallback on platforms without flock supplies thread exclusion only.
    """
    def __init__(self, fs, root):
        from fsspec.implementations.local import LocalFileSystem
        if not isinstance(fs, LocalFileSystem):
            raise TypeError("local publication requires the local filesystem")
        super().__init__(fs, os.path.abspath(root))
        path = Path(self.root)
        if path.is_symlink() or path.exists() and not path.is_dir():
            raise ValueError("native object root requires a regular directory")

    def _path(self, key):
        path = Path(self.path(key))
        # Refuse links along the owned prefix, including directories of blobs.
        for part in (path, *path.parents):
            if part.is_symlink():
                raise ValueError("native object storage cannot use symbolic links")
            if part == Path(self.root):
                break
        if path.exists() and not path.is_file():
            raise ValueError("native object storage requires regular files")
        return path

    def read(self, key, *, limit):
        from .localstore import _read_bytes
        if type(limit) is not int or limit < 0:
            raise ValueError("object read limit requires a nonnegative native integer")
        try:
            raw = _read_bytes(self._path(key), limit)
        except FileNotFoundError:
            return None
        return PublishedObject(raw, hashlib.sha256(raw).digest())

    def compare_and_swap(self, key, expected, raw):
        from rexgraph.io.publication import staged_publication
        if type(raw) is not bytes or expected is not None and type(expected) is not bytes:
            raise TypeError("conditional publication requires native bytes")
        with self.transaction():
            path = self._path(key)
            token = None
            if path.exists():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024*1024):
                        digest.update(chunk)
                token = digest.digest()
            if token != expected:
                raise VersionConflictError("object publication precondition differs from current bytes")
            path.parent.mkdir(parents=True, exist_ok=True)
            with staged_publication(path) as staged:
                staged.write_bytes(raw)
            return PublishedObject(raw, hashlib.sha256(raw).digest())

    def has_objects(self):
        root = Path(self.root)
        if root.is_symlink():
            raise ValueError("native object root cannot be a symbolic link")
        return root.exists() and any(root.iterdir())

    @contextmanager
    def transaction(self):
        from rexgraph.io.publication import publication_lock
        root = Path(self.root)
        if root.is_symlink():
            raise ValueError("native object root cannot be a symbolic link")
        root.mkdir(parents=True, exist_ok=True)
        with publication_lock(root):
            yield


def publication_for(fs, root, provider=None):
    """Resolve installed capabilities; never infer remote CAS from fsspec methods."""
    if provider is not None:
        result = provider(fs, root)
        if not isinstance(result, ObjectPublication) or result.fs is not fs or result.root != (root.rstrip("/") or "/"):
            raise ValueError("publication provider must return a capability bound to this filesystem and prefix")
        return result
    from fsspec.implementations.local import LocalFileSystem
    from fsspec.implementations.memory import MemoryFileSystem
    if isinstance(fs, MemoryFileSystem):
        return MemoryObjectPublication(fs, root)
    if isinstance(fs, LocalFileSystem):
        return LocalObjectPublication(fs, root)
    raise NotImplementedError("native object storage requires an explicit strong-read and conditional-publication provider")
