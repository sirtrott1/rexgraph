"""Publish molecular bundles through the existing native record interface."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from fractions import Fraction
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

__all__ = ["ingest_molecules", "verify_molecular_sources"]


def verify_molecular_sources(source, root):
    manifest = getattr(source, "_agent_meta", {}).get("source_manifest", {})
    if manifest.get("schema") != "molecular_bundle/1":
        raise ValueError("source is not a molecular bundle")
    root = Path(root).resolve()
    verified = []
    for item in manifest["files"]:
        name = item["name"]
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("molecular source name must be local")
        path = (root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("molecular source escapes the bundle")
        raw = path.read_bytes()
        if len(raw) != item["bytes"] or sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("molecular source bytes changed")
        verified.append(str(path))
    return tuple(verified)


def ingest_molecules(store, path, *, document_id, record_id=None, reader=None, expected_version=None,
                     skip_unchanged=True, valid_from=None, valid_to=None, tx_time=None, actor="", tags=(), **options):
    """Read a registered molecular bundle and publish its complete native state."""
    from agent.adapters.formats import read, reader_fn
    from agent.auto import build_rex_from_edges
    from rcdb.core import VersionConflictError
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id is required")
    rid = document_id if record_id is None else record_id
    if not isinstance(rid, str) or not rid.strip():
        raise ValueError("record_id is required")
    if not isinstance(skip_unchanged, bool):
        raise TypeError("skip_unchanged must be boolean")
    if expected_version is not None and (isinstance(expected_version, bool) or not isinstance(expected_version, Integral) or expected_version < 0):
        raise ValueError("expected_version must be a nonnegative integer")
    edges = (read if reader is None else reader_fn(reader))(path, document_id=document_id, **options)
    manifest = deepcopy(getattr(edges, "source_manifest", {}))
    if manifest.get("schema") != "molecular_bundle/1":
        raise ValueError("reader must return a native molecular bundle")
    current = store.read_record(rid)
    version = 0 if current is None else current.record.version
    if expected_version is not None and version != expected_version:
        raise VersionConflictError("molecular record version changed")
    def clock(v):
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (Real, Fraction)):
            raise TypeError("record clocks require finite numbers")
        v = float(v)
        if not isfinite(v):
            raise ValueError("record clocks must be finite")
        return v
    valid_from, valid_to, tx_time = map(clock, (valid_from, valid_to, tx_time))
    if valid_from is not None and valid_to is not None and valid_from >= valid_to:
        raise ValueError("record validity endpoints must increase")
    if current is not None and tx_time is not None and tx_time < current.record.tx_from:
        raise ValueError("transaction time precedes the current record")
    selection = {"valid_from": valid_from, "valid_to": valid_to}
    meta = {"object_type": "molecular_bundle", "document_id": document_id,
            "bundle_digest": manifest["bundle_digest"], "source_manifest": manifest, "selection": selection}
    tags = list(dict.fromkeys((*tags, "molecular_bundle")))
    if current is not None and skip_unchanged:
        previous = current.record.meta or {}
        if (previous.get("bundle_digest") == meta["bundle_digest"] and previous.get("selection") == selection
                and (current.record.signature or {}).get("tags", []) == tags):
            return rid, None
    source = build_rex_from_edges(edges, face_selection="none", input_type="molecular_bundle")
    record = store.commit_mutation(rid, source, meta=meta, tags=tags, expected_version=version,
                                  valid_from=valid_from, valid_to=valid_to, tx_time=tx_time, actor=actor, analytics=False)
    return rid, {**meta, "version": record.version}
