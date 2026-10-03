"""
Read only adapter for historical fsspec RCDB object stores.

The legacy layout stores JSON segments and address named payloads:

    <root>/MANIFEST.json          format version
    <root>/journal/<seq>.json     one entry per change, immutable once written
    <root>/index.json             optional snapshot, written by compact()
    <root>/blobs/<id>@<v>.st      the payload

Opening validates existing data without publishing a manifest, identity or cache.
Available history migration into the shared native engine uses
plan_legacy_migration/migrate_legacy_batch and the sole copy_record seam.

Explicit read_only=False retains temporary compatibility writing. It requires
one quiescent writer and does not provide distributed conditional publication,
retained deletion history or native version allocation. Memory filesystem tests
qualify this adapter's layout, not S3/GCS/Azure consistency or durability.
"""

from __future__ import annotations

import contextlib
import hashlib
import posixpath
from typing import Any

from rexgraph.io._compat import dumps

from .core import (
    ComplexRecord,
    RCStore,
    _matches,
    _owned_records,
    _record_labels,
    _serialized,
    _read_selector,
    _QUERY_KEYS,
)
from .envelope import _RECORD_HEADER_LIMIT, _RECORD_PAYLOAD_LIMIT
from .journal import FRAME_LIMIT

_BLOB_LIMIT = _RECORD_PAYLOAD_LIMIT+_RECORD_HEADER_LIMIT+1024*1024
_SNAPSHOT_LIMIT = 64*1024*1024

MANIFEST = "MANIFEST.json"
JOURNAL = "journal"
SNAPSHOT = "index.json"
BLOBS = "blobs"
FORMAT_VERSION = 1

#: schemes fsspec routes for us once the matching driver is installed.
SCHEMES = ("s3", "gs", "gcs", "az", "abfs", "adl", "memory", "file")


def _fs_for(uri: str):
    """The fsspec filesystem and root path for a URI, with a legible error when the
    provider's driver is missing: 'install s3fs' beats an ImportError from three
    frames down."""
    try:
        import fsspec
    except ImportError as e:
        raise ImportError(
            "object storage needs fsspec: pip install fsspec") from e
    from urllib.parse import urlparse
    scheme = urlparse(uri).scheme or "file"
    try:
        fs, _, paths = fsspec.get_fs_token_paths(uri)
    except ImportError as e:
        hint = {"s3": "s3fs", "gs": "gcsfs", "gcs": "gcsfs",
                "az": "adlfs", "abfs": "adlfs", "adl": "adlfs"}.get(scheme, scheme)
        raise ImportError(
            f"{scheme}:// needs the {hint} driver: pip install {hint}") from e
    return fs, (paths[0] if paths else uri)


class ObjectStore(RCStore):
    """Legacy object layout reader; explicit read_only=False enables compatibility writes."""

    backend = "object"

    def __init__(self, uri: str, *, identity_provider=None, read_only=True):
        if type(read_only) is not bool:
            raise TypeError("read_only must be a bool")
        self.read_only, self._closed = read_only, False
        self.uri = uri
        self.fs, self.root = _fs_for(uri)
        self._recs: dict[str, list[ComplexRecord]] = {}
        self._labels: dict[str, set] = {}
        self._seq = 0
        self._identity_provider = identity_provider
        self._check_layout()
        if read_only:
            self._legacy_source_fingerprint = self._source_fingerprint()
        self._ensure_manifest()
        # Object providers often synthesize prefixes, but the supported local
        # fsspec filesystem requires these parent directories to exist.
        if not read_only:
            for name in (BLOBS, JOURNAL, "commits"):
                self.fs.makedirs(self._p(name), exist_ok=True)
        self._load()
        from .store_identity import bound_identity
        if read_only or bound_identity(record for rows in self._recs.values() for record in rows) is not None:
            self._store_identity = self._load_store_identity()
        if read_only and self._source_fingerprint() != self._legacy_source_fingerprint:
            raise ValueError("legacy source changed while opening its read-only snapshot")

    def _check_open(self):
        if self._closed:
            raise ValueError("object store handle is closed")

    def _check_writable(self):
        self._check_open()
        super()._check_writable()

    def _source_fingerprint(self):
        from .legacy import object_source_fingerprint
        return object_source_fingerprint(self.fs, self.root)

    def _check_layout(self):
        from pathlib import Path
        protocols = (self.fs.protocol,) if isinstance(self.fs.protocol, str) else tuple(self.fs.protocol)
        if set(protocols) & {"file", "local"}:
            root = Path(self.root)
            if root.is_symlink() or (root.exists() and not root.is_dir()):
                raise ValueError("legacy object source requires a regular directory")
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_symlink() or not (path.is_file() or path.is_dir()):
                        raise ValueError("legacy object source requires regular files and directories")
        if any(self.fs.exists(self._p(name)) for name in
               ("store.header", "records.journal", "records.log", "blobs.pack",
                "index.log", "index.rexidx", "index.rexlog")):
            raise ValueError("ObjectStore cannot adopt another store layout")
        if self.read_only and not self.fs.exists(self._p(MANIFEST)):
            raise ValueError("legacy object source requires an existing manifest")

    def _read_bytes(self, key, limit):
        with self.fs.open(key, "rb") as stream:
            raw = stream.read(limit+1)
        if len(raw) > limit:
            raise ValueError("legacy object storage exceeds its byte limit")
        return raw

    def _load_store_identity(self):
        from .store_identity import StoreIdentity, bound_identity, local_identity, memory_filesystem_identity
        protocol = self.fs.protocol
        protocols = (protocol,) if isinstance(protocol, str) else tuple(protocol)
        path = self._p(".rcdb-identity")
        self._check_open()
        known = bound_identity(record for versions in self._recs.values() for record in versions)
        if known is not None and not self.fs.exists(path):
            raise ValueError("store identity is missing from bound durable state; explicit recovery required")
        if self.read_only:
            if self.fs.exists(path):
                identity = StoreIdentity.from_bytes(self._read_bytes(path, 4096), backend="object")
                if known is not None and identity.id != known:
                    raise ValueError("store identity differs from its durable ownership claim")
                return identity
            digest = self._legacy_source_fingerprint
            return StoreIdentity(hashlib.sha256(b"rexgraph-legacy-source-identity\x00object"+
                                               bytes.fromhex(digest)).hexdigest()[:32], "object")
        if set(protocols) & {"file", "local"}:
            return local_identity(path, backend="object", existing_id=known)
        if "memory" in protocols:
            return memory_filesystem_identity(self.fs, path, existing_id=known)
        if self._identity_provider is not None:
            identity = self._identity_provider(self.fs, path)
            if not isinstance(identity, StoreIdentity) or identity.backend != "object":
                raise ValueError("object identity provider returned an undeclared store identity")
            if known is not None and identity.id != known:
                raise ValueError("store identity differs from its durable ownership claim")
            return identity
        if self.fs.exists(path):
            with self.fs.open(path, "rb") as stream:
                identity = StoreIdentity.from_bytes(stream.read(4097), backend="object")
            if known is not None and identity.id != known:
                raise ValueError("store identity differs from its durable ownership claim")
            return identity
        raise NotImplementedError("this object provider requires conditional store-identity publication")

    #### layout
    def _p(self, *parts) -> str:
        return posixpath.join(self.root, *parts)

    def _ensure_manifest(self) -> None:
        from .legacy import loads_metadata
        path = self._p(MANIFEST)
        if self.fs.exists(path):
            value = loads_metadata(self._read_bytes(path, 4096))
            if (type(value) is not dict or set(value) != {"format", "version"}
                    or value["format"] != "rexdb-object" or type(value["version"]) is not int
                    or value["version"] != FORMAT_VERSION):
                raise ValueError("unknown legacy object manifest declaration")
        else:
            self._check_writable()
            if self.fs.exists(self.root) and self.fs.find(self.root):
                raise ValueError("object manifest is missing from existing data; explicit recovery required")
            self.fs.makedirs(self.root, exist_ok=True)
            with self.fs.open(path, "wb") as fh:
                fh.write(dumps({"format": "rexdb-object",
                                "version": FORMAT_VERSION}).encode("utf-8"))

    @staticmethod
    def _safe(id: str) -> str:
        from rexgraph.state import RESERVED_PATH, encode_name
        return encode_name(id, RESERVED_PATH)

    def _blob_key(self, id: str, version: int) -> str:
        return self._p(BLOBS, f"{self._safe(id)}@{version}.safetensors")

    def _commit_key(self, id: str, version: int) -> str:
        return self._p("commits", f"{self._safe(id)}@{int(version)}.rexpkg")

    def _store_commit_bytes(self, id, version, blob):
        self._check_open()
        self._check_writable()
        with self.fs.open(self._commit_key(id, version), "wb") as fh:
            if fh.write(bytes(blob)) != len(blob):
                raise OSError("short object mutation artifact write")

    def _load_commit_bytes(self, id, version):
        self._check_open()
        key = self._commit_key(id, version)
        if not self.fs.exists(key):
            return None
        return self._read_bytes(key, _BLOB_LIMIT)

    def _delete_commit_bytes(self, id, version):
        self._check_open()
        self._check_writable()
        import contextlib
        with contextlib.suppress(Exception):
            self.fs.rm(self._commit_key(id, version))

    #### load
    def _load(self) -> None:
        from .legacy import loads_metadata, record_metadata
        snapshot_seq = -1
        path = self._p(SNAPSHOT)
        if self.fs.exists(path):
            snap = loads_metadata(self._read_bytes(path, _SNAPSHOT_LIMIT))
            if (type(snap) is not dict or set(snap) != {"through_seq", "records"}
                    or type(snap["through_seq"]) is not int or not 0 <= snap["through_seq"] < 2**63
                    or type(snap["records"]) is not dict):
                raise ValueError("unknown legacy object snapshot declaration")
            snapshot_seq = snap["through_seq"]
            for rid, versions in snap["records"].items():
                if type(rid) is not str or not rid or type(versions) is not list:
                    raise ValueError("invalid legacy object snapshot record address")
                self._recs[rid] = [record_metadata(v, rid) for v in versions]
                numbers = [record.version for record in self._recs[rid]]
                if numbers != sorted(set(numbers)):
                    raise ValueError("legacy object snapshot versions are duplicated or out of order")

        expected = max(snapshot_seq, 0)+1
        for seq, entry in self._journal_entries():
            if seq <= snapshot_seq:
                continue                      # already folded into the snapshot
            if seq != expected:
                raise ValueError("object journal sequence has a missing change")
            self._apply(entry)
            self._seq = max(self._seq, seq)
            expected += 1
        self._seq = max(self._seq, snapshot_seq)
        self._reindex_labels()

    def _journal_entries(self):
        from .legacy import loads_metadata, record_change
        from .journal import FRAME_LIMIT
        prefix = self._p(JOURNAL)
        if not self.fs.exists(prefix):
            return []
        out, seen = [], set()
        for key in self.fs.ls(prefix, detail=False):
            name = posixpath.basename(str(key))
            stem = name[:-5] if name.endswith(".json") else name
            if not stem.isdigit():
                continue
            sequence = int(stem)
            if sequence in seen or not 0 < sequence < 2**63:
                raise ValueError("duplicate or invalid object journal sequence")
            seen.add(sequence)
            with self.fs.open(key, "rb") as fh:
                raw = fh.read(FRAME_LIMIT+1)
            if len(raw) > FRAME_LIMIT:
                raise ValueError("legacy object journal segment exceeds its byte limit")
            entry = loads_metadata(raw)
            record_change(entry)
            out.append((sequence, entry))
        out.sort(key=lambda t: t[0])
        return out

    def _apply(self, entry: dict[str, Any]) -> None:
        from .legacy import record_change
        operation, rid, rec, _extra = record_change(entry)
        if operation == "delete":
            self._recs.pop(rid, None)
            return
        versions = list(self._recs.get(rid, []))
        if versions and (rec.version <= versions[-1].version or rec.tx_from < versions[-1].tx_from):
            raise ValueError("legacy object publication regresses its version or transaction time")
        for prior in versions:
            if prior.tx_to is None:
                prior.tx_to = rec.tx_from
        versions.append(rec)
        versions.sort(key=lambda r: r.version)
        self._recs[rid] = versions

    def _reindex_labels(self) -> None:
        self._labels = {}
        for rid, versions in self._recs.items():
            for rec in versions:
                for label in _record_labels(rec.signature, rec.meta):
                    self._labels.setdefault(label, set()).add(rid)

    def _write_journal(self, entry: dict[str, Any]) -> None:
        self._check_open()
        self._check_writable()
        from .legacy import record_change
        from .core import PublicationUncertainError
        record_change(entry)
        raw = dumps(entry).encode("utf-8")
        if len(raw) > FRAME_LIMIT:
            raise ValueError("legacy object journal segment exceeds its byte limit")
        sequence = self._seq+1
        key = self._p(JOURNAL, f"{sequence:012d}.json")
        if self.fs.exists(key):
            raise PublicationUncertainError("object journal next sequence already exists; reload or recovery required")
        try:
            with self.fs.open(key, "wb") as fh:
                if fh.write(raw) != len(raw):
                    raise OSError("short object journal write")
        except Exception as exc:
            raise PublicationUncertainError("object journal write outcome is uncertain; reopen and verify") from exc
        self._seq = sequence

    #### writes
    def next_version(self, id):
        _read_selector(id, None, None, None)
        with self.read_transaction():
            return (self._recs[id][-1].version + 1) if self._recs.get(id) else 1

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to,
                  search_terms=None, tx_time=None):
        self._check_open()
        self._check_writable()
        from .core import _now
        now = _now() if tx_time is None else float(tx_time)
        rows = self._recs.get(id, [])
        if rows and now < rows[-1].tx_from:
            raise ValueError("legacy object publication regresses its transaction time")
        v = self.next_version(id)
        rec = ComplexRecord(id=id, signature=sig, created=now, meta=meta or {},
                            version=v, tx_from=now, tx_to=None,
                            valid_from=valid_from if valid_from is not None else now,
                            valid_to=valid_to)
        rec, blob = self._record_payload(rec, rex)
        # blob first: a crash between the two leaves an unreferenced object, which
        # is inert, rather than a journal entry pointing at nothing.
        with self.fs.open(self._blob_key(id, v), "wb") as fh:
            if fh.write(blob) != len(blob):
                raise OSError("short object record payload write")
        entry = {"op": "put", "id": id, "record": rec.to_storage_dict()}
        self._write_journal(entry)
        self._apply(entry)
        for label in _record_labels(sig, meta):
            self._labels.setdefault(label, set()).add(id)
        return self._recs[id][-1]

    @_serialized
    def delete(self, id):
        reclaim = self._claim_delete(id)
        if id not in self._recs:
            return False
        versions = list(self._recs.get(id, []))
        entry = {"op": "delete", "id": id}
        self._write_journal(entry)
        self._apply(entry)
        for ids in self._labels.values():
            ids.discard(id)
        for rec in versions:
            with contextlib.suppress(Exception):
                self.fs.rm(self._blob_key(id, rec.version))
        self._reclaim_commits(id, reclaim)
        self._emit("rcdb.delete", id, 0, {})
        return True

    #### reads
    @_owned_records
    def history(self, id):
        _read_selector(id, None, None, None)
        return list(self._recs.get(id, []))

    @_owned_records
    def get_record(self, id, *, as_of=None, valid_at=None):
        _read_selector(id, None, as_of, valid_at)
        return self._select_version(self._recs.get(id, []), as_of, valid_at)

    def get_version(self, id, version):
        with self.read_transaction():
            return self._get_version(id, version)

    def _get_version(self, id, version, *, verify=True):
        _read_selector(id, version, None, None)
        rec = next((r for r in self.history(id) if r.version == version), None)
        if rec is None:
            return None
        key = self._blob_key(id, int(version))
        if not self.fs.exists(key):
            raise ValueError("published object record payload is missing")
        return self._read_record_payload(rec, self._read_bytes(key, _BLOB_LIMIT), verify=verify)

    def get(self, id, *, as_of=None, valid_at=None, verify=True):
        _read_selector(id, None, as_of, valid_at)
        with self.read_transaction():
            if id not in self._recs:
                split = self._split_versioned_id(id)
                if split is not None:
                    if as_of is not None or valid_at is not None:
                        raise ValueError("version display aliases cannot combine time selectors")
                    return self._get_version(split[0], split[1], verify=verify)
            rec = self.get_record(id, as_of=as_of, valid_at=valid_at)
            return None if rec is None else self._get_version(id, rec.version, verify=verify)

    @_owned_records
    def list(self, limit=100, offset=0, *, as_of=None, valid_at=None,
             include_history=False):
        if type(limit) is not int or type(offset) is not int or limit < 0 or offset < 0:
            raise ValueError("object collection bounds require nonnegative native integers")
        recs = self._selected_records(self._recs, as_of, valid_at, include_history)
        return recs[offset:offset + limit]

    def _selected_records(self, ids, as_of, valid_at, include_history):
        _read_selector("selector", None, as_of, valid_at)
        if type(include_history) is not bool:
            raise TypeError("include_history must be a bool")
        if include_history and (as_of is not None or valid_at is not None):
            raise ValueError("history collection cannot combine time selectors")
        if include_history:
            recs = [r for id in ids for r in self._recs.get(id, ())]
        else:
            recs = [self._select_version(self._recs.get(id, ()), as_of, valid_at) for id in ids]
        recs = [r for r in recs if r is not None]
        recs.sort(key=lambda r: (-r.tx_from, r.id, -r.version))
        return recs

    @_owned_records
    def query(self, limit=100, *, as_of=None, valid_at=None, include_history=False, **predicate):
        if type(limit) is not int or limit < 0:
            raise ValueError("object query limit requires a nonnegative native integer")
        if set(predicate)-_QUERY_KEYS:
            raise TypeError("unsupported query keys: "+", ".join(sorted(set(predicate)-_QUERY_KEYS))
                            +". Supported: "+", ".join(sorted(_QUERY_KEYS)))
        wanted = predicate.get("labels_any")
        if wanted:
            ids: set = set()
            for label in wanted:
                ids |= self._labels.get(str(label).lower(), set())
        else:
            ids = self._recs
        cands = self._selected_records(ids, as_of, valid_at, include_history)
        out = [r for r in cands if r is not None
               and _matches(r.signature, predicate, r.meta,
                            is_complex=r.is_complex, record_type=r.object_type)]
        return out[:limit]

    def stats(self) -> dict[str, Any]:
        with self.read_transaction():
            n_segments = len(self._journal_entries())
            value = super().stats()
            value.update({"backend": self.backend, "uri": self.uri,
                    "n_labels": len(self._labels),
                    "journal_segments": n_segments})
            return value

    @_serialized
    def compact(self) -> dict[str, Any]:
        """Fold the journal into a snapshot and delete the segments it replaces.

        A listing whose cost grows with every write is how an object store index
        degrades; this is what keeps opening cheap.
        """
        self._check_writable()
        before = self.stats()
        snap = {"through_seq": self._seq,
                "records": {rid: [r.to_storage_dict() for r in versions]
                            for rid, versions in self._recs.items()}}
        raw = dumps(snap).encode("utf-8")
        if len(raw) > _SNAPSHOT_LIMIT:
            raise ValueError("legacy object snapshot exceeds its byte limit")
        try:
            with self.fs.open(self._p(SNAPSHOT), "wb") as fh:
                if fh.write(raw) != len(raw):
                    raise OSError("short object snapshot write")
        except Exception as exc:
            from .core import PublicationUncertainError
            raise PublicationUncertainError("object snapshot write outcome is uncertain; reopen and verify") from exc
        for seq, _ in self._journal_entries():
            if seq <= self._seq:
                with contextlib.suppress(Exception):
                    self.fs.rm(self._p(JOURNAL, f"{seq:012d}.json"))
        return {"before": before, "after": self.stats()}

    def close(self):
        with self._transaction_lock:
            self._closed = True


from .object_native import NativeObjectStore


def open_object_store(uri, *, native=None, **options):
    """Open a native object provider or a read only legacy migration source.

    Existing manifest only layouts remain legacy. Native ownership/head markers
    choose native replay, including partial data which must never be reinitialized.
    A new prefix uses NativeObjectStore and requires a qualified publication capability.
    Explicit native=False retains the legacy transition API.
    """
    if native is not None and type(native) is not bool:
        raise TypeError("native must be a bool or None")
    if native is None:
        fs, root = _fs_for(uri)
        native = (any(fs.exists(posixpath.join(root, name)) for name in ("store.head", "store.header"))
                  or not fs.exists(posixpath.join(root, MANIFEST)))
    return (NativeObjectStore if native else ObjectStore)(uri, **options)
