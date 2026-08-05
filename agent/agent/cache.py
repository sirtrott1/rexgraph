"""
agent.cache: content-addressed cache for pipeline analysis.

Each pipeline run otherwise rebuilds the complex, recomputes eigenvalues and re-runs
Hodge from scratch, even for an identical file. This module caches the built RexGraph
(via ``RexGraph.to_dict()``) together with its analysis dict, keyed by a SHA-256 of the
input content plus the analysis depth, under ``~/.cache/rexgraph/``.

The cache is best-effort: any failure to read or write falls back to
recomputation, so it can never break a run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_DISABLE = "REXGRAPH_NO_CACHE"

#: Bumped whenever what gets written changes shape. It is mixed into the key, so a
#: change MISSES stale entries rather than deserializing them into new code.
CACHE_VERSION = "2"

#: entries kept before the oldest are dropped. A cache nothing ever evicts is a disk
#: leak, which at ingest scale is an operational problem rather than an untidiness.
MAX_ENTRIES = int(os.environ.get("REXGRAPH_CACHE_MAX_ENTRIES", "2000"))

#: writes between prune sweeps: scanning the directory on every write is O(entries).
_PRUNE_EVERY = 128
_writes_since_prune = 0


def cache_dir() -> Path:
    root = os.environ.get("REXGRAPH_CACHE_DIR")
    p = Path(root) if root else Path.home() / ".cache" / "rexgraph"
    p.mkdir(parents=True, exist_ok=True)
    return p


def enabled() -> bool:
    return os.environ.get(_ENV_DISABLE, "").lower() not in ("1", "true", "yes")


def content_key(content: Any, depth: str = "standard", extra: str = "") -> str:
    """Return a stable cache key for content + depth.

    ``content`` may be bytes, str, or a file path (its bytes are read).
    """
    h = hashlib.sha256()
    try:
        if isinstance(content, (bytes, bytearray)):
            h.update(bytes(content))
        elif isinstance(content, (str, os.PathLike)) and Path(str(content)).exists():
            with open(content, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        else:
            h.update(str(content).encode("utf-8", "replace"))
    except Exception:
        h.update(repr(content).encode("utf-8", "replace"))
    h.update(f"|depth={depth}|{extra}|v={CACHE_VERSION}".encode())
    return h.hexdigest()


def _version_salt() -> str:
    return CACHE_VERSION


def _blob_path(key: str) -> Path:
    return cache_dir() / f"{key}.safetensors"


def _side_path(key: str) -> Path:
    return cache_dir() / f"{key}.json"


def _write_atomic(path: Path, data: bytes) -> None:
    """Publish `path` atomically, through a temp name unique to this writer.

    with_suffix('.tmp') gives every writer of a key the SAME temp path, so two
    processes caching the same input could interleave into it and publish a spliced
    file. os.replace is atomic; the write feeding it has to be private.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def entry_count() -> int:
    try:
        return sum(1 for _ in cache_dir().glob("*.json"))
    except Exception:
        return 0


def prune(max_entries: int | None = None) -> int:
    """Drop the oldest entries beyond `max_entries`. Returns how many files went."""
    cap = MAX_ENTRIES if max_entries is None else int(max_entries)
    removed = 0
    try:
        sides = sorted(cache_dir().glob("*.json"), key=lambda p: p.stat().st_mtime)
        for side in sides[:max(0, len(sides) - cap)]:
            stem = side.name.split(".")[0]
            for victim in (side, cache_dir() / f"{stem}.safetensors"):
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
    """Generic payloads get their own suffix: a bare `<key>.json` would collide with
    the analysis sidecar, which has a different shape."""
    return cache_dir() / f"{key}.payload.json"


def get(key: str) -> dict | None:
    """Return the cached JSON payload for `key`, or None."""
    if not enabled():
        return None
    p = _generic_path(key)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("cache read failed for %s: %s", key, e)
        return None
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        return None
    return payload.get("data")


def set(key: str, payload: dict) -> bool:
    """Store a JSON-serializable `payload` under `key`."""
    if not enabled():
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

    The sidecar is written last, so a torn pair reads as a miss rather than as a
    complex with no analysis beside it.
    """
    if not enabled():
        return None, None, None
    side, blob = _side_path(key), _blob_path(key)
    if not side.exists() or not blob.exists():
        return None, None, None
    try:
        payload = json.loads(side.read_text(encoding="utf-8"))
        if payload.get("version") != CACHE_VERSION:
            return None, None, None
        meta = payload.get("meta") or {}
        analysis = payload.get("analysis")
        from agent.rcdb import deserialize_complex
        rex = deserialize_complex(blob.read_bytes())
    except Exception as e:
        logger.debug("cache read failed for %s: %s", key, e)
        return None, None, None
    if rex is not None and meta:
        # to_dict/from_dict never carried _agent_meta, so a cached complex came back
        # stripped of its labels and source text while the caller got them as a
        # separate return value. Harmless while everything read that third value;
        # not harmless once the complex itself is what gets persisted or handed on.
        rex._agent_meta = dict(meta)
    return rex, analysis, meta


def store_rex_and_analysis(key: str, rex, analysis: dict, meta: dict) -> bool:
    """Cache a built complex plus its analysis and meta.

    The complex goes through the same safetensors serializer the RCDB uses; the
    analysis and meta are JSON. Neither is pickle. Every other serializer in the
    tree avoids it deliberately, and this is the one whose filenames are a
    predictable hash of content anybody can supply.
    """
    if not enabled():
        return False
    try:
        from agent.rcdb import serialize_complex
        from rexgraph.io._compat import dumps
        blob = serialize_complex(rex)
        side = dumps({"version": CACHE_VERSION, "analysis": analysis,
                      "meta": meta or {}}).encode("utf-8")
    except Exception as e:
        logger.debug("cache serialisation skipped for %s: %s", key, e)
        return False
    try:
        # blob first, sidecar last: the sidecar's presence is what makes an entry
        # readable, so a crash between the two leaves a miss, not a half-entry.
        _write_atomic(_blob_path(key), blob)
        _write_atomic(_side_path(key), side)
    except Exception as e:
        logger.debug("cache write failed for %s: %s", key, e)
        return False
    _maybe_prune()
    return True
