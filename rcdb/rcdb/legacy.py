"""Strict readers for the two retired JSON journal layouts.

Only declared RCDB metadata is reconstructed. A malformed complete record is
corruption; incomplete physical tails require explicit recovery.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import struct

import numpy as np

from .journal import FRAME_LIMIT, TornJournalError, _RECORD_FIELDS

SEARCH_EXTRA_MAGIC = 0x52585331
SEARCH_TOKEN_WORDS = 4
_LENGTH = struct.Struct("<I")


def check_source_root(root, backend):
    """A legacy read must never initialize or adopt another layout's directory."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("legacy source requires an existing regular directory")
    if any((root / name).exists() or (root / name).is_symlink()
           for name in ("store.header", "records.journal")):
        raise ValueError("canonical local data requires LocalStore")
    rex = any((root / name).exists() for name in ("records.log", "blobs.pack", "MANIFEST.json"))
    file = any((root / name).exists() for name in ("index.rexidx", "index.rexlog", "index.json", "index.log"))
    if rex and file or backend == "file" and rex or backend == "rex" and file:
        raise ValueError("legacy source has conflicting store layouts")
    if backend == "file" and not (file or (root / "blobs").is_dir()):
        raise ValueError("legacy file source has no store data")
    if backend == "rex" and not rex:
        raise ValueError("legacy Rex source has no store data")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("legacy source requires regular files and directories")


def source_fingerprint(root):
    """Content bound legacy snapshot, including metadata, blobs and commit artifacts.

    Hash in bounded chunks; paths and byte counts are part of the declaration.
    This is a checked fingerprint, not a filesystem transaction or writer lock.
    Callers must keep historical sources quiescent throughout migration.
    """
    from rexgraph.value_codec import pack_value
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("legacy source requires an existing regular directory")
    digest = hashlib.sha256(b"rexgraph-legacy-source-snapshot\x00\x01")
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("legacy source requires regular files and directories")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            digest.update(pack_value((relative, "directory")))
            continue
        content, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            while raw := stream.read(1024*1024):
                content.update(raw)
                size += len(raw)
        digest.update(pack_value((relative, "file", size, content.hexdigest())))
    return digest.hexdigest()


def object_source_fingerprint(fs, root):
    """Pin legacy fsspec objects without writing, trusting listings as transactions,
    or persisting provider credentials. Local paths use the regular file checks
    of the existing directory fingerprint; remote prefixes hash sorted names,
    byte counts and content in bounded chunks. Sources must remain quiescent.
    """
    import posixpath
    from rexgraph.value_codec import pack_value
    protocols = (fs.protocol,) if isinstance(fs.protocol, str) else tuple(fs.protocol)
    if set(protocols) & {"file", "local"}:
        return source_fingerprint(root)
    prefix = root.rstrip("/")+"/"
    entries = fs.find(root, detail=True)
    if type(entries) is not dict:
        raise ValueError("legacy object inventory requires declared file details")
    digest = hashlib.sha256(b"rexgraph-legacy-object-snapshot\x00\x01")
    for key, info in sorted(entries.items()):
        if (type(key) is not str or not key.startswith(prefix) or type(info) is not dict
                or info.get("type") != "file"):
            raise ValueError("legacy object inventory has an invalid address or object type")
        relative = key[len(prefix):]
        if not relative or posixpath.normpath(relative) != relative or relative.startswith("../"):
            raise ValueError("legacy object inventory has an invalid relative address")
        content, size = hashlib.sha256(), 0
        with fs.open(key, "rb") as stream:
            while raw := stream.read(1024*1024):
                content.update(raw)
                size += len(raw)
        digest.update(pack_value((relative, "file", size, content.hexdigest())))
    return digest.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate legacy JSON field {key!r}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("legacy JSON contains a nonfinite number")


def loads_metadata(raw):
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)


def record_metadata(raw, record_id):
    from .core import ComplexRecord
    if (type(raw) is not dict or "id" not in raw or raw["id"] != record_id
            or set(raw)-_RECORD_FIELDS-{"envelope", "record_values_version"}):
        raise ValueError("legacy record differs from its declared metadata address or schema")
    record = ComplexRecord.from_dict(raw)
    if type(record.version) is not int or not 0 < record.version < 2**63:
        raise ValueError("legacy record version requires a positive native integer")
    for value in (record.created, record.tx_from, record.tx_to, record.valid_from, record.valid_to):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("invalid legacy record time")
    return record


def record_change(entry):
    if type(entry) is not dict or type(entry.get("id")) is not str or not entry["id"]:
        raise ValueError("legacy change requires a literal record identity")
    operation = entry.get("op")
    if type(operation) is not str or operation not in {"put", "delete"}:
        raise ValueError("unknown legacy journal operation")
    expected = {"op", "id"} if operation == "delete" else {"op", "id", "record"}
    if set(entry) != expected:
        raise ValueError("unknown or incomplete legacy journal change declaration")
    record = None if operation == "delete" else record_metadata(entry["record"], entry["id"])
    return operation, entry["id"], record, None


def line_changes(path, *, allow_torn_tail=False):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("legacy journal cannot be a symbolic link")
    end = 0
    with path.open("rb") as stream:
        while True:
            raw = stream.readline(FRAME_LIMIT+1)
            if not raw:
                return
            if len(raw) > FRAME_LIMIT:
                raise ValueError("legacy journal line exceeds its byte limit")
            if raw.strip():
                try:
                    entry = loads_metadata(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as failure:
                    if not raw.endswith(b"\n"):
                        if allow_torn_tail:
                            return
                        raise TornJournalError(end) from failure
                    raise ValueError("malformed complete legacy JSON journal line") from failure
                yield record_change(entry)
            end = stream.tell()


def rex_change(entry):
    if type(entry) is not dict or type(entry.get("id")) is not str or not entry["id"]:
        raise ValueError("legacy Rex change requires a literal record identity")
    operation = entry.get("op")
    if type(operation) is not str or operation not in {"put", "delete"}:
        raise ValueError("unknown legacy journal operation")
    if operation == "delete":
        if set(entry) != {"op", "id"}:
            raise ValueError("legacy delete cannot declare record metadata or coordinates")
        return operation, entry["id"], None, None
    extras = {"op", "blob_off", "blob_len", "search_tokens"}
    if set(entry)-_RECORD_FIELDS-extras-{"envelope", "record_values_version"}:
        raise ValueError("unknown legacy Rex change declaration")
    offset, length = entry.get("blob_off"), entry.get("blob_len")
    if (type(offset) is not int or type(length) is not int or offset < 0 or length <= 0
            or offset >= 2**63 or length >= 2**63):
        raise ValueError("legacy Rex coordinates require nonnegative native int64 values")
    raw = {key: value for key, value in entry.items() if key not in extras}
    raw.setdefault("created", 0.0)
    raw.setdefault("tx_from", 0.0)
    record = record_metadata(raw, entry["id"])
    tokens = entry.get("search_tokens", ())
    if type(tokens) not in (list, tuple):
        raise ValueError("legacy search tokens require a sequence")
    extra = [offset, length]
    if tokens:
        from .envelope import _hex_identity
        extra.extend((SEARCH_EXTRA_MAGIC, len(tokens)))
        for token in tokens:
            _hex_identity(token, 64, "legacy search token")
            extra.extend(np.frombuffer(bytes.fromhex(token), "<i8").tolist())
    return operation, entry["id"], record, tuple(extra)


def rex_json_entries(path, start=0, *, allow_torn_tail=False):
    """Validate the whole legacy prefix and exact resume boundary before replay."""
    if type(start) is not int or start < 0:
        raise ValueError("journal cursor requires a nonnegative native byte offset")
    path = Path(path)
    if path.is_symlink():
        raise ValueError("legacy journal cannot be a symbolic link")
    entries, boundaries, valid_end, torn = [], {0}, 0, False
    with path.open("rb") as stream:
        while True:
            first = stream.read(4)
            if not first:
                break
            if len(first) != 4:
                torn = True
                break
            size = _LENGTH.unpack(first)[0]
            if not 0 < size <= FRAME_LIMIT:
                raise ValueError("invalid legacy Rex journal frame length")
            raw = stream.read(size)
            if len(raw) != size:
                torn = True
                break
            try:
                entry = rex_change(loads_metadata(raw))
            except (UnicodeDecodeError, json.JSONDecodeError) as failure:
                raise ValueError("malformed complete legacy Rex JSON frame") from failure
            if valid_end >= start:
                entries.append(entry)
            valid_end = stream.tell()
            boundaries.add(valid_end)
    if start not in boundaries:
        raise ValueError("journal cursor is not a complete legacy JSON frame boundary")
    if torn and not allow_torn_tail:
        raise TornJournalError(valid_end)
    yield from entries
