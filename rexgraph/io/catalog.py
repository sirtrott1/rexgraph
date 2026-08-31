"""Bounded local catalog for RexGraph-owned files and stores."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import manifest_digest

OBJECT_IDENTITY_VERSION = 2

_RCDB_FILES = {
    "MANIFEST.json",
    "records.log",
    "blobs.pack",
    "index.safetensors",
    "search.safetensors",
    "index.rexidx",
    "index.rexlog",
    "index.json",
    "index.log",
}
_RCDB_DIRS = {"blobs", "commits"}
_CHUNK = 1024 * 1024

__all__ = [
    "CatalogEntry",
    "FileCatalog",
    "OBJECT_IDENTITY_VERSION",
    "object_digest",
    "state_object_digest",
]


@dataclass(frozen=True)
class CatalogEntry:
    """Public metadata for one cataloged RexGraph file or store."""

    name: str
    kind: str
    size: int
    sha256: str | None = None
    object_type: str | None = None
    state_digest: str | None = None
    tensors: int | None = None
    records: int | None = None


class FileCatalog:
    """Index RexGraph files below explicit local roots.

    Public names are root-relative labels. Absolute paths remain private to the catalog.
    Higher packages can inject kind loaders without making core import those packages.
    """

    def __init__(
        self,
        roots: Iterable[str | os.PathLike],
        *,
        max_entries: int = 100_000,
        loaders: Mapping[str, Callable[[Path], Any]] | None = None,
    ):
        resolved = []
        for root in roots:
            path = Path(root).expanduser().resolve(strict=True)
            if not path.is_dir():
                raise ValueError(f"catalog root is not a directory: {root!r}")
            resolved.append(path)
        if not resolved:
            raise ValueError("catalog requires at least one root")
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        normalized_loaders = dict(loaders or {})
        if any(not isinstance(kind, str) or not kind for kind in normalized_loaders):
            raise ValueError("catalog loader kinds must be nonempty strings")
        if any(not callable(loader) for loader in normalized_loaders.values()):
            raise TypeError("catalog loaders must be callable")
        self._roots = tuple(resolved)
        self.max_entries = max_entries
        self._loaders = normalized_loaders
        self._entries: dict[str, CatalogEntry] = {}
        self.refresh()

    @property
    def roots(self) -> tuple[str, ...]:
        """Return opaque root labels rather than filesystem paths."""
        return tuple(f"root{index}" for index in range(len(self._roots)))

    def refresh(self, *, hash_files: bool = False) -> int:
        """Rescan roots and atomically replace the in-memory index."""
        entries: dict[str, CatalogEntry] = {}
        for index, root in enumerate(self._roots):
            prefix = f"root{index}"
            for path in _walk(root):
                kind = _kind(path)
                if kind is None:
                    continue
                name = f"{prefix}/{path.relative_to(root).as_posix()}"
                entries[name] = self._entry(name, path, kind, hash_file=hash_files)
                if len(entries) > self.max_entries:
                    raise ValueError("catalog entry limit exceeded")
        self._entries = dict(sorted(entries.items()))
        return len(self._entries)

    def list(self, *, limit: int = 100, offset: int = 0) -> list[CatalogEntry]:
        """Return a bounded slice of catalog entries."""
        bounded = _bounded_limit(limit)
        offset = max(0, int(offset))
        return list(self._entries.values())[offset:offset + bounded]

    def search(self, text: str, *, limit: int = 100) -> list[CatalogEntry]:
        """Search relative names and bounded metadata using literal terms."""
        terms = [term.casefold() for term in str(text).split() if term]
        if not terms:
            return self.list(limit=limit)
        bounded = _bounded_limit(limit)
        found = []
        for entry in self._entries.values():
            values = (
                entry.name,
                entry.kind,
                entry.object_type or "",
                entry.sha256 or "",
                entry.state_digest or "",
            )
            haystack = " ".join(value for value in values if value).casefold()
            if all(term in haystack for term in terms):
                found.append(entry)
                if len(found) >= bounded:
                    break
        return found

    def info(self, name: str) -> CatalogEntry:
        """Return metadata for one exact relative catalog name."""
        try:
            return self._entries[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown catalog entry {name!r}") from exc

    def hash(self, name: str) -> str:
        """Hash an entry after resolving and rechecking its catalog path."""
        entry = self.info(name)
        path = self._resolve(name)
        kind = _kind(path)
        if kind != entry.kind:
            raise ValueError("catalog entry kind changed; refresh before hashing")
        digest = _hash_path(path, kind)
        self._entries[str(name)] = CatalogEntry(
            name=entry.name,
            kind=entry.kind,
            size=_path_size(path, kind),
            sha256=digest,
            object_type=entry.object_type,
            state_digest=entry.state_digest,
            tensors=entry.tensors,
            records=entry.records,
        )
        return digest

    def hash_all(self) -> int:
        """Hash every current entry and cache the results."""
        for name in tuple(self._entries):
            self.hash(name)
        return len(self._entries)

    def load(self, name: str) -> Any:
        """Load an entry through a built-in or explicitly injected kind loader."""
        entry = self.info(name)
        path = self._resolve(name)
        kind = _kind(path)
        if kind != entry.kind:
            raise ValueError("catalog entry kind changed; refresh before loading")
        loader = self._loaders.get(kind)
        if loader is not None:
            return loader(path)
        if kind in {"rex", "safetensors"}:
            from rexgraph.io import load

            return load(str(path))
        raise ValueError(
            f"catalog entry kind {kind!r} needs an injected loader; "
            "core does not import higher packages"
        )

    def tensors(self, name: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        """Return bounded tensor metadata from one safetensors file."""
        entry = self.info(name)
        path = self._resolve(name)
        if entry.kind != "safetensors" or _kind(path) != "safetensors":
            raise ValueError("tensor metadata expects a safetensors file")
        from safetensors import safe_open

        rows = []
        with safe_open(str(path), framework="numpy") as handle:
            for key in list(handle.keys())[:_bounded_limit(limit)]:
                view = handle.get_slice(key)
                rows.append(
                    {
                        "name": key,
                        "shape": list(view.get_shape()),
                        "dtype": str(view.get_dtype()),
                    }
                )
        return rows

    def search_tensors(
        self,
        name: str,
        text: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search tensor names inside one safetensors file using literal terms."""
        terms = [term.casefold() for term in str(text).split() if term]
        rows = self.tensors(name, limit=1000)
        bounded = _bounded_limit(limit)
        if not terms:
            return rows[:bounded]
        found = []
        for row in rows:
            value = row["name"].casefold()
            if all(term in value for term in terms):
                found.append(row)
                if len(found) >= bounded:
                    break
        return found

    def _resolve(self, name: str) -> Path:
        entry = self.info(name)
        root_name, separator, relative = entry.name.partition("/")
        if not separator or not root_name.startswith("root"):
            raise ValueError("invalid catalog name")
        try:
            root = self._roots[int(root_name[4:])]
        except (ValueError, IndexError) as exc:
            raise ValueError("invalid catalog root") from exc
        current = root
        for part in Path(relative).parts:
            if part in {"", ".", ".."}:
                raise ValueError("invalid catalog path")
            current = current / part
            if current.is_symlink():
                raise ValueError("catalog paths may not traverse symlinks")
        candidate = (root / relative).resolve(strict=True)
        if candidate != root and root not in candidate.parents:
            raise ValueError("catalog path escapes its root")
        return candidate

    def _entry(
        self,
        name: str,
        path: Path,
        kind: str,
        *,
        hash_file: bool = False,
    ) -> CatalogEntry:
        return CatalogEntry(
            name=name,
            kind=kind,
            size=_path_size(path, kind),
            sha256=_hash_path(path, kind) if hash_file else None,
            **_metadata(path, kind),
        )


def object_digest(value: Any) -> str:
    """Return a semantic identity for a ``RexGraph`` or verified ``TemporalRex``."""
    from rexgraph.graph import TemporalRex

    if isinstance(value, TemporalRex):
        from rexgraph.io.temporal_state import to_temporal_state, verify_temporal_state

        state = to_temporal_state(value)
        if not verify_temporal_state(state):  # pragma: no cover - writer invariant
            raise ValueError("could not produce a verified TemporalState identity")
        return str(state.header["digest"])

    from rexgraph.io.rex_state import to_state

    return state_object_digest(to_state(value))


def state_object_digest(state: Any) -> str:
    """Return semantic identity directly from a verified canonical ``RexState``."""
    from rexgraph.io.rex_state import RexState, verify_state

    if not isinstance(state, RexState) or not verify_state(state):
        raise ValueError("a verified canonical RexState is required")
    return manifest_digest(
        {
            "header": state.header,
            "object_identity_version": OBJECT_IDENTITY_VERSION,
            "object_type": "RexGraphState",
        }
    )


def _read_json(path: Path, *, max_bytes: int = 4 * 1024 * 1024) -> Any:
    """Read one bounded JSON metadata file."""
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"metadata file exceeds {max_bytes} bytes")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bounded_limit(value: int) -> int:
    return min(1000, max(1, int(value)))


def _walk(root: Path):
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories if not (current_path / name).is_symlink()
        )
        if _kind(current_path) in {"rex", "rcdb"}:
            yield current_path
            directories[:] = []
            continue
        for name in sorted(files):
            path = current_path / name
            if not path.is_symlink() and _kind(path) is not None:
                yield path


def _kind(path: Path) -> str | None:
    if path.is_dir():
        manifest = path / "MANIFEST.json"
        if manifest.is_file():
            try:
                data = _read_json(manifest)
            except Exception:  # noqa: BLE001 - malformed candidates are not entries
                data = {}
            if isinstance(data, dict) and data.get("magic") == "rex-bundle":
                return "rex"
            if isinstance(data, dict) and data.get("format") in {"rcdb-file", "rexstore"}:
                return "rcdb"
        if all((path / name).exists() for name in ("records.log", "blobs.pack")):
            return "rcdb"
        if (path / "blobs").is_dir() and any(
            (path / name).exists()
            for name in ("index.rexidx", "index.rexlog", "index.json", "index.log")
        ):
            return "rcdb"
        return None
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return "safetensors"
    if suffix in {".rexenc", ".rexpkg"}:
        try:
            with path.open("rb") as handle:
                head = handle.read(16)
        except OSError:
            return None
        if head.startswith(b"REXENC\x00"):
            return "encrypted"
        if head.startswith(b"REXPKG\x00"):
            return "transport"
    return None


def _iter_files(path: Path, kind: str | None):
    if path.is_file():
        yield path, path.name
        return
    for current, directories, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories if not (current_path / name).is_symlink()
        )
        if kind == "rcdb":
            relative_directory = current_path.relative_to(path)
            if relative_directory.parts and relative_directory.parts[0] not in _RCDB_DIRS:
                directories[:] = []
            elif not relative_directory.parts:
                directories[:] = [name for name in directories if name in _RCDB_DIRS]
        for name in sorted(files):
            child = current_path / name
            if child.is_symlink():
                continue
            relative = child.relative_to(path).as_posix()
            if kind != "rcdb":
                yield child, relative
                continue
            parts = Path(relative).parts
            if relative in _RCDB_FILES or (parts and parts[0] in _RCDB_DIRS):
                yield child, relative


def _hash_path(path: Path, kind: str | None) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for child, relative in _iter_files(path, kind):
        relative_bytes = relative.encode("utf-8")
        size = child.stat().st_size
        digest.update(len(relative_bytes).to_bytes(8, "little"))
        digest.update(relative_bytes)
        digest.update(int(size).to_bytes(8, "little"))
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _path_size(path: Path, kind: str) -> int:
    return sum(child.stat().st_size for child, _relative in _iter_files(path, kind))


def _metadata(path: Path, kind: str) -> dict[str, Any]:
    if kind == "rex":
        try:
            data = _read_json(path / "MANIFEST.json")
            if not isinstance(data, dict):
                return {}
            return {
                "object_type": str(data.get("object_type")) if data.get("object_type") else None,
                "state_digest": data.get("digest"),
                "tensors": len(data.get("tensor_names", ())),
            }
        except Exception:  # noqa: BLE001 - metadata is optional catalog detail
            return {}
    if kind == "safetensors":
        try:
            from safetensors import safe_open

            with safe_open(str(path), framework="numpy") as handle:
                raw = handle.metadata() or {}
                tensors = len(list(handle.keys()))
            data = json.loads(raw["rex_meta"]) if "rex_meta" in raw else {}
            return {
                "object_type": (
                    str(data.get("object_type"))
                    if data.get("object_type")
                    else "Safetensors"
                ),
                "state_digest": data.get("digest"),
                "tensors": tensors,
            }
        except Exception:  # noqa: BLE001 - malformed metadata does not expose file contents
            return {"object_type": "Safetensors"}
    if kind == "encrypted":
        try:
            from .security import ENVELOPE_MAGIC, envelope_info

            with path.open("rb") as handle:
                prefix = handle.read(len(ENVELOPE_MAGIC) + 4)
                if not prefix.startswith(ENVELOPE_MAGIC):
                    return {}
                length = int.from_bytes(prefix[-4:], "big")
                if length <= 0 or length > 64 * 1024:
                    return {}
                header = handle.read(length)
            info = envelope_info(prefix + header)
            return {"object_type": info.object_type}
        except Exception:  # noqa: BLE001 - public envelope metadata is best effort
            return {"object_type": "EncryptedEnvelope"}
    if kind == "transport":
        try:
            from .transport import MAGIC, inspect

            with path.open("rb") as handle:
                prefix = handle.read(len(MAGIC) + 4)
                if not prefix.startswith(MAGIC):
                    return {}
                length = int.from_bytes(prefix[-4:], "big")
                if length <= 0 or length > 1024 * 1024:
                    return {}
                header = handle.read(length)
            info = inspect(prefix + header)
            return {"object_type": info.object_type}
        except Exception:  # noqa: BLE001 - public transport metadata is best effort
            return {"object_type": "RexTransport"}
    if kind == "rcdb":
        return {"object_type": "RCDB"}
    return {}
