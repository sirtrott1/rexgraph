"""Cache serialized complexes, analysis and JSON results by content and parameters.

Complexes use the RCDB serializer. Their JSON sidecars contain analysis, source
metadata and a digest of the serialized complex. Reads reject incomplete or
inconsistent pairs, and filesystem or serialization errors produce a cache miss.
Entry retention is bounded by periodic pruning in the configured cache directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from agent.data_paths import cache_root

logger = logging.getLogger(__name__)

_ENV_DISABLE = "REXGRAPH_NO_CACHE"

#: Included in keys and payloads to invalidate incompatible serialized entries.
CACHE_VERSION = "4"

#: Completed result entries retained after a prune sweep.
try:
    MAX_ENTRIES = max(0, int(os.environ.get("REXGRAPH_CACHE_MAX_ENTRIES", "2000")))
except ValueError:
    MAX_ENTRIES = 2000
    logger.warning("Invalid REXGRAPH_CACHE_MAX_ENTRIES; using 2000")

#: writes between prune sweeps: scanning the directory on every write is O(entries).
_PRUNE_EVERY = 128
# Short command line runs must enforce retention without reaching a full interval.
_writes_since_prune = _PRUNE_EVERY - 1


def cache_dir(*, create: bool = True) -> Path:
    """Return the cache directory, creating it only when requested."""
    p = cache_root()
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def enabled() -> bool:
    return MAX_ENTRIES > 0 and os.environ.get(_ENV_DISABLE, "").lower() not in ("1", "true", "yes")


def content_key(content: Any, depth: str = "standard", extra: str = "") -> str:
    """Return a stable key for content, analysis depth, extra parameters and format.

    ``content`` may be bytes, str, or a file path (its bytes are read).
    """
    h = hashlib.sha256()
    try:
        if isinstance(content, (bytes, bytearray)):
            h.update(content)
        elif isinstance(content, (str, os.PathLike)):
            text = os.fspath(content)
            try:
                is_file = Path(text).is_file()
            except (OSError, ValueError):
                is_file = False
            if is_file:
                with open(content, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            else:
                h.update(str(text).encode("utf-8", "replace"))
        else:
            h.update(str(content).encode("utf-8", "replace"))
    except Exception:
        h = hashlib.sha256()
        h.update(repr(content).encode("utf-8", "replace"))
    parameters = json.dumps([depth, extra, CACHE_VERSION], ensure_ascii=True).encode()
    return hashlib.sha256(h.digest() + parameters).hexdigest()


def _version_salt() -> str:
    return CACHE_VERSION


def _entry_path(key: str, suffix: str) -> Path:
    if (not isinstance(key, str) or not key or key in {".", ".."}
            or any(char in key for char in ("/", "\\", ":", "\x00"))):
        raise ValueError("cache key must be a single filename component")
    return cache_dir(create=False) / f"{key}{suffix}"


def _blob_path(key: str) -> Path:
    return _entry_path(key, ".rexblob")


def _side_path(key: str) -> Path:
    return _entry_path(key, ".analysis.json")


def _write_atomic(path: Path, data: bytes) -> None:
    """Publish a complete file with a temporary name unique to each write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _entries(root: Path):
    """Yield completed entry files without including configuration or process files."""
    for path in root.glob("*.json"):
        try:
            if path.name.endswith(".payload.json"):
                victims = (path,)
            else:
                if path.name.endswith(".analysis.json"):
                    blob = root / (path.name.removesuffix(".analysis.json") + ".rexblob")
                else:
                    blob = path.with_suffix(".rexblob")
                if not blob.is_file():
                    continue
                victims = (path, blob)
            yield path.stat().st_mtime_ns, victims
        except OSError:
            continue


def entry_count() -> int:
    try:
        return sum(1 for _ in _entries(cache_dir(create=False)))
    except Exception:
        return 0


def prune(max_entries: int | None = None) -> int:
    """Remove the oldest completed entries beyond the limit and return files removed."""
    cap = max(0, MAX_ENTRIES if max_entries is None else int(max_entries))
    removed = 0
    try:
        entries = sorted(_entries(cache_dir(create=False)), key=lambda entry: entry[0])
        for _, victims in entries[:max(0, len(entries) - cap)]:
            for victim in victims:
                try:
                    victim.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
    except Exception as e:
        logger.debug("cache prune failed: %s", e)
    return removed


def _maybe_prune() -> None:
    global _writes_since_prune
    _writes_since_prune += 1
    if _writes_since_prune >= _PRUNE_EVERY:
        _writes_since_prune = 0
        prune()


def _generic_path(key: str) -> Path:
    """Return the JSON result path, distinct from an analysis sidecar."""
    return _entry_path(key, ".payload.json")


def get(key: str) -> dict | None:
    """Return the cached JSON payload for `key`, or None."""
    if not enabled():
        return None
    try:
        payload = json.loads(_generic_path(key).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
            return None
        data = payload.get("data")
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.debug("cache read failed for %s: %s", key, e)
        return None


def set(key: str, payload: dict) -> bool:
    """Store a JSON serializable `payload` under `key`."""
    if not enabled() or MAX_ENTRIES == 0:
        return False
    try:
        from rexgraph.io._compat import dumps
        blob = dumps({"version": CACHE_VERSION, "data": payload}).encode("utf-8")
    except Exception as e:
        logger.debug("cache serialisation skipped for %s: %s", key, e)
        return False
    try:
        _write_atomic(_generic_path(key), blob)
    except Exception as e:
        logger.debug("cache write failed for %s: %s", key, e)
        return False
    _maybe_prune()
    return True


def get_rex_and_analysis(key: str):
    """Return ``(rex, analysis, meta)`` from cache, or ``(None, None, None)``.

    The sidecar digest must match the blob before it is deserialized.
    """
    if not enabled():
        return None, None, None
    try:
        side, blob = _side_path(key), _blob_path(key)
        payload = json.loads(side.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
            return None, None, None
        meta = payload.get("meta") or {}
        analysis = payload.get("analysis")
        if not isinstance(meta, dict) or not isinstance(analysis, dict):
            return None, None, None
        data = blob.read_bytes()
        if payload.get("blob_digest") != hashlib.sha256(data).hexdigest():
            return None, None, None
        from agent.rcdb import deserialize_complex
        rex = deserialize_complex(data)
    except Exception as e:
        logger.debug("cache read failed for %s: %s", key, e)
        return None, None, None
    if rex is not None and meta:
        # Restore labels and source metadata on the complex as well as in the result.
        rex._agent_meta = dict(meta)
    return rex, analysis, meta


def store_rex_and_analysis(key: str, rex, analysis: dict, meta: dict) -> bool:
    """Cache a serialized complex with its analysis, metadata and blob digest."""
    if not enabled() or MAX_ENTRIES == 0:
        return False
    try:
        from agent.rcdb import serialize_complex
        from rexgraph.io._compat import dumps
        blob = serialize_complex(rex)
        side = dumps({"version": CACHE_VERSION, "blob_digest": hashlib.sha256(blob).hexdigest(),
                      "analysis": analysis,
                      "meta": meta or {}}).encode("utf-8")
    except Exception as e:
        logger.debug("cache serialisation skipped for %s: %s", key, e)
        return False
    try:
        # Publish the sidecar last; readers also verify the pair's digest.
        _write_atomic(_blob_path(key), blob)
        _write_atomic(_side_path(key), side)
    except Exception as e:
        logger.debug("cache write failed for %s: %s", key, e)
        return False
    _maybe_prune()
    return True
