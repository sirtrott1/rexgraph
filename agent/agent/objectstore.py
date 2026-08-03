"""
agent.objectstore: the RCDB on cloud object storage. S3, GCS, Azure Blob.

Built on fsspec, so one backend serves every provider: the wire protocol is the
driver's problem (s3fs, gcsfs, adlfs), and what lives here is the layout. That also
means it is testable, because fsspec ships an in-memory filesystem that exercises
the same code path as S3 rather than a stand-in for it.

Object storage cannot append, so RexStore's single growing log is the wrong shape.
What object stores ARE good at is many immutable objects and a cheap prefix listing,
so the journal is segmented -- one small object per change -- and the blobs are one
object each:

    <root>/MANIFEST.json          format version
    <root>/journal/<seq>.json     one entry per change, immutable once written
    <root>/index.json             optional snapshot, written by compact()
    <root>/blobs/<id>@<v>.st      the payload

Opening reads the snapshot if there is one and then the journal segments after it,
which is the same layering FileStore uses locally. Compaction folds the journal into
a new snapshot and deletes what it replaced.

Immutability is the point rather than a constraint: every object is written once and
never modified, so there is no read-modify-write to lose a concurrent writer's entry,
and a half-finished upload leaves an object that simply is not referenced.

    store = rcdb.open_store("s3://bucket/prefix")
    store = rcdb.open_store("gs://bucket/prefix")
    store = rcdb.open_store("az://container/prefix")
"""

from __future__ import annotations

import contextlib
import json
import posixpath
from typing import Any

from rexgraph.io._compat import dumps

from .rcdb import (
    ComplexRecord,
    RCStore,
    _matches,
    _record_labels,
    deserialize_complex,
    serialize_complex,
)

MANIFEST = "MANIFEST.json"
JOURNAL = "journal"
SNAPSHOT = "index.json"
BLOBS = "blobs"
FORMAT_VERSION = 1

#: schemes fsspec routes for us once the matching driver is installed.
SCHEMES = ("s3", "gs", "gcs", "az", "abfs", "adl", "memory", "file")


def _fs_for(uri: str):
    """The fsspec filesystem and root path for a URI, with a legible error when the
    provider's driver is missing -- 'install s3fs' beats an ImportError from three
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
    """RCStore over any fsspec-addressable object storage."""

    backend = "object"

    def __init__(self, uri: str):
        self.uri = uri
        self.fs, self.root = _fs_for(uri)
        self._recs: dict[str, list[ComplexRecord]] = {}
        self._labels: dict[str, set] = {}
        self._seq = 0
        self._ensure_manifest()
        self._load()

    # --- layout ---------------------------------------------------------------

    def _p(self, *parts) -> str:
        return posixpath.join(self.root, *parts)

    def _ensure_manifest(self) -> None:
        path = self._p(MANIFEST)
        if not self.fs.exists(path):
            self.fs.makedirs(self.root, exist_ok=True)
            with self.fs.open(path, "wb") as fh:
                fh.write(dumps({"format": "rexdb-object",
                                "version": FORMAT_VERSION}).encode("utf-8"))

    @staticmethod
    def _safe(id: str) -> str:
        from rexgraph.io.rex_state import RESERVED_PATH, encode_name
        return encode_name(id, RESERVED_PATH)

    def _blob_key(self, id: str, version: int) -> str:
        return self._p(BLOBS, f"{self._safe(id)}@{version}.safetensors")

    # --- load -----------------------------------------------------------------

    def _load(self) -> None:
        snapshot_seq = -1
        path = self._p(SNAPSHOT)
        if self.fs.exists(path):
            try:
                with self.fs.open(path, "rb") as fh:
                    snap = json.loads(fh.read().decode("utf-8"))
                snapshot_seq = int(snap.get("through_seq", -1))
                for rid, versions in (snap.get("records") or {}).items():
                    self._recs[rid] = [ComplexRecord.from_dict(v) for v in versions]
            except Exception:
                self._recs = {}
                snapshot_seq = -1

        for seq, entry in self._journal_entries():
            if seq <= snapshot_seq:
                continue                      # already folded into the snapshot
            self._apply(entry)
            self._seq = max(self._seq, seq)
        self._seq = max(self._seq, snapshot_seq)
        self._reindex_labels()

    def _journal_entries(self):
        prefix = self._p(JOURNAL)
        if not self.fs.exists(prefix):
            return []
        out = []
        for key in self.fs.ls(prefix, detail=False):
            name = posixpath.basename(str(key))
            stem = name[:-5] if name.endswith(".json") else name
            if not stem.isdigit():
                continue
            try:
                with self.fs.open(key, "rb") as fh:
                    out.append((int(stem), json.loads(fh.read().decode("utf-8"))))
            except Exception:
                continue                      # an unreadable segment is skipped, not fatal
        out.sort(key=lambda t: t[0])
        return out

    def _apply(self, entry: dict[str, Any]) -> None:
        rid = entry.get("id")
        if entry.get("op") == "delete":
            self._recs.pop(rid, None)
            return
        rec = ComplexRecord.from_dict(entry["record"])
        versions = [r for r in self._recs.get(rid, []) if r.version != rec.version]
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
        self._seq += 1
        key = self._p(JOURNAL, f"{self._seq:012d}.json")
        with self.fs.open(key, "wb") as fh:
            fh.write(dumps(entry).encode("utf-8"))

    # --- writes ---------------------------------------------------------------

    def next_version(self, id):
        return (self._recs[id][-1].version + 1) if self._recs.get(id) else 1

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to):
        from .rcdb import _now
        now = _now()
        v = self.next_version(id)
        # blob first: a crash between the two leaves an unreferenced object, which
        # is inert, rather than a journal entry pointing at nothing.
        with self.fs.open(self._blob_key(id, v), "wb") as fh:
            fh.write(serialize_complex(rex))
        rec = ComplexRecord(id=id, signature=sig, created=now, meta=meta or {},
                            version=v, tx_from=now, tx_to=None,
                            valid_from=valid_from if valid_from is not None else now,
                            valid_to=valid_to)
        entry = {"op": "put", "id": id, "record": rec.to_dict()}
        self._write_journal(entry)
        self._apply(entry)
        for label in _record_labels(sig, meta):
            self._labels.setdefault(label, set()).add(id)
        return self._recs[id][-1]

    def delete(self, id):
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
        self._emit("rcdb.delete", id, 0, {})
        return True

    # --- reads ----------------------------------------------------------------

    def history(self, id):
        return list(self._recs.get(id, []))

    def get_record(self, id, *, as_of=None, valid_at=None):
        return self._select_version(self._recs.get(id, []), as_of, valid_at)

    def get_version(self, id, version):
        key = self._blob_key(id, int(version))
        if not self.fs.exists(key):
            return None
        with self.fs.open(key, "rb") as fh:
            return deserialize_complex(fh.read())

    def get(self, id, *, as_of=None, valid_at=None):
        split = self._split_versioned_id(id)
        if split is not None:
            return self.get_version(split[0], split[1])
        rec = self.get_record(id, as_of=as_of, valid_at=valid_at)
        return None if rec is None else self.get_version(id, rec.version)

    def list(self, limit=100, offset=0, *, as_of=None, valid_at=None,
             include_history=False):
        if include_history:
            recs = [r for versions in self._recs.values() for r in versions]
        else:
            recs = [self._select_version(v, as_of, valid_at)
                    for v in self._recs.values()]
        recs = [r for r in recs if r is not None]
        recs.sort(key=lambda r: -r.tx_from)
        return recs[offset:offset + limit]

    def query(self, limit=100, *, as_of=None, valid_at=None, **predicate):
        wanted = predicate.get("labels_any")
        if wanted:
            ids: set = set()
            for label in wanted:
                ids |= self._labels.get(str(label).lower(), set())
            cands = [self._select_version(self._recs.get(i, []), as_of, valid_at)
                     for i in sorted(ids)]
        else:
            cands = self.list(limit=10 ** 9, as_of=as_of, valid_at=valid_at)
        out = [r for r in cands if r is not None
               and _matches(r.signature, predicate, r.meta)]
        out.sort(key=lambda r: -r.tx_from)
        return out[:limit]

    def stats(self) -> dict[str, Any]:
        try:
            n_segments = len(self._journal_entries())
        except Exception:
            n_segments = -1
        return {"backend": self.backend, "uri": self.uri,
                "n_records": len(self._recs),
                "n_versions": sum(len(v) for v in self._recs.values()),
                "n_labels": len(self._labels),
                "journal_segments": n_segments}

    def compact(self) -> dict[str, Any]:
        """Fold the journal into a snapshot and delete the segments it replaces.

        A listing whose cost grows with every write is how an object-store index
        degrades; this is what keeps opening cheap.
        """
        before = self.stats()
        snap = {"through_seq": self._seq,
                "records": {rid: [r.to_dict() for r in versions]
                            for rid, versions in self._recs.items()}}
        with self.fs.open(self._p(SNAPSHOT), "wb") as fh:
            fh.write(dumps(snap).encode("utf-8"))
        for seq, _ in self._journal_entries():
            if seq <= self._seq:
                with contextlib.suppress(Exception):
                    self.fs.rm(self._p(JOURNAL, f"{seq:012d}.json"))
        return {"before": before, "after": self.stats()}

    def close(self):
        return None
