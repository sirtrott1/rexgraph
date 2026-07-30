"""
agent.cache: content-addressed cache for pipeline analysis.

Each pipeline run otherwise rebuilds the complex, recomputes
eigenvalues and re-runs Hodge from scratch, even for an identical file
(audit 3.4).  This module caches the built RexGraph (via
``RexGraph.to_dict()``) together with its analysis dict, keyed by a
SHA-256 of the input content plus the analysis depth, under
``~/.cache/rexgraph/``.

The cache is best-effort: any failure to read or write falls back to
recomputation, so it can never break a run.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENV_DISABLE = "REXGRAPH_NO_CACHE"


def cache_dir() -> Path:
    root = os.environ.get("REXGRAPH_CACHE_DIR")
    if root:
        p = Path(root)
    else:
        p = Path.home() / ".cache" / "rexgraph"
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
    h.update(f"|depth={depth}|{extra}".encode("utf-8"))
    return h.hexdigest()


def _path_for(key: str) -> Path:
    return cache_dir() / f"{key}.pkl"


def get(key: str) -> Optional[dict]:
    """Return the cached payload dict for ``key`` or None."""
    if not enabled():
        return None
    p = _path_for(key)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        logger.debug("cache read failed for %s: %s", key, e)
        return None


def set(key: str, payload: dict) -> bool:
    """Store ``payload`` (a picklable dict) under ``key``."""
    if not enabled():
        return False
    p = _path_for(key)
    try:
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)
        return True
    except Exception as e:
        logger.debug("cache write failed for %s: %s", key, e)
        return False


def get_rex_and_analysis(key: str):
    """Return ``(rex, analysis, meta)`` from cache, or ``(None, None, None)``.

    Rebuilds the RexGraph from its cached ``to_dict()`` payload.
    """
    payload = get(key)
    if not payload:
        return None, None, None
    rex = None
    rex_dict = payload.get("rex_dict")
    if rex_dict is not None:
        try:
            from rexgraph.graph import RexGraph
            rex = RexGraph.from_dict(rex_dict)
        except Exception as e:
            logger.debug("cache rex rebuild failed: %s", e)
            rex = None
    return rex, payload.get("analysis"), payload.get("meta")


def store_rex_and_analysis(key: str, rex, analysis: dict, meta: dict) -> bool:
    """Cache a built RexGraph (as dict) plus its analysis and meta."""
    rex_dict = None
    try:
        rex_dict = rex.to_dict()
    except Exception as e:
        logger.debug("cache rex serialisation skipped: %s", e)
    return set(key, {
        "rex_dict": rex_dict,
        "analysis": analysis,
        "meta": meta,
    })
