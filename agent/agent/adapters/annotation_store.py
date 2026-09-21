"""Publish registered annotation bundles through ordinary native store commits."""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real
from fractions import Fraction
from math import isfinite
from pathlib import Path

__all__ = ["ingest_annotations", "verify_annotation_sources"]


def verify_annotation_sources(rex, root):
    """Verify external source bytes against the selected native bundle manifest."""
    import hashlib
    manifest = getattr(rex, "_agent_meta", {}).get("source_manifest", {})
    if manifest.get("schema") != "annotation_bundle/1":
        raise ValueError("source is not a declared annotation bundle")
    root = Path(root).resolve()
    checked = []
    for source in manifest["files"]:
        name = source["name"]
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("source manifest requires a local file name")
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("source file escapes the bundle directory")
        raw = path.read_bytes()
        if len(raw) != source["bytes"] or hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("source bytes differ from the selected version: " + name)
        checked.append((source["role"], str(path)))
    return tuple(checked)


def ingest_annotations(store, source, *, document_id, record_id=None, reader=None,
                       expected_version=None, skip_unchanged=True, valid_from=None,
                       valid_to=None, tx_time=None, actor="", tags=(), **reader_options):
    """Parse a registered bundle and commit only its native state and source manifest."""
    from agent.adapters.formats import read, reader_fn
    from agent.auto import build_rex_from_edges
    from rcdb.core import VersionConflictError

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a nonempty string")
    rid = document_id if record_id is None else record_id
    if not isinstance(rid, str) or not rid.strip():
        raise ValueError("record_id must be a nonempty string")
    if not isinstance(skip_unchanged, bool):
        raise TypeError("skip_unchanged must be boolean")
    if expected_version is not None and (isinstance(expected_version, bool)
            or not isinstance(expected_version, Integral) or expected_version < 0):
        raise ValueError("expected_version must be a nonnegative integer")
    parser = read if reader is None else reader_fn(reader)
    edges = parser(source, document_id=document_id, **reader_options)
    manifest = deepcopy(getattr(edges, "source_manifest", {}))
    if manifest.get("schema") != "annotation_bundle/1":
        raise ValueError("the selected reader did not declare an annotation bundle")
    current = store.read_record(rid)
    version = 0 if current is None else current.record.version
    if expected_version is not None and expected_version != version:
        raise VersionConflictError(f"record {rid!r} expected version {expected_version}, current version {version}")
    def clock(value):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (Real, Fraction)):
            raise TypeError("record clocks require finite numbers")
        result = float(value)
        if not isfinite(result):
            raise ValueError("record clocks must be finite")
        return result
    valid_from, valid_to, tx_time = map(clock, (valid_from, valid_to, tx_time))
    if valid_from is not None and valid_to is not None and valid_to <= valid_from:
        raise ValueError("record validity must have increasing endpoints")
    if current is not None and tx_time is not None and tx_time < current.record.tx_from:
        raise ValueError("transaction time precedes the current record")
    selection = {"valid_from": valid_from, "valid_to": valid_to}
    meta = {"object_type": "annotation_bundle", "document_id": document_id,
            "bundle_digest": manifest["bundle_digest"], "source_manifest": manifest,
            "selection": selection}
    tag_list = list(dict.fromkeys([*tags, "annotation_bundle"]))
    if current is not None and skip_unchanged:
        previous = current.record.meta or {}
        previous_tags = (current.record.signature or {}).get("tags", [])
        if (previous.get("bundle_digest") == meta["bundle_digest"]
                and previous.get("selection") == selection and list(previous_tags) == tag_list):
            return rid, None
    rex = build_rex_from_edges(edges, face_selection="none", input_type="annotation_bundle")
    record = store.commit_mutation(rid, rex, meta=meta, tags=tag_list, valid_from=valid_from,
                                   valid_to=valid_to, tx_time=tx_time, actor=actor,
                                   expected_version=version, analytics=False)
    return rid, {**meta, "version": record.version}
