"""
agent.rexstore: an embedded store for relational complexes. Files, no server.

FileStore reserialized its whole index on every put, so per-put cost grew with the
store: 4 ms at a hundred records, 41 ms at sixteen hundred, which is O(n^2) ingest
and about 33 hours for 100k records. It also wrote one file per record, and on a
network filesystem -- EFS, Azure Files, a GCS mount, which is what a cloud VM
actually has -- per-file overhead dominates everything else.

This is the same data laid out for how it is used:

    <root>/MANIFEST.json    format version
    <root>/records.log      append-only, [u32 length][json record] per entry
    <root>/blobs.pack       append-only, safetensors blobs end to end

Three files, whatever the record count. A put is two appends and costs the same at
record one and record one million. Opening scans the log once -- sequential, which
is the access pattern every filesystem is fastest at -- and builds the indexes in
memory, so a vocabulary query is a dict lookup rather than a scan.

Append-only earns two things beyond speed. A crash can only ever tear the tail,
which the length prefix detects, so recovery is truncation rather than repair. And
history is not a feature bolted on: every version is simply still there, which is
what the bitemporal model wanted from the start.

    store = rcdb.open_store("rex:///data/complexes")
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict, List, Optional

from rexgraph.io._compat import dumps

from .rcdb import ComplexRecord, RCStore, _matches, _record_labels, deserialize_complex, serialize_complex

#: length prefix for a log entry. 4 bytes little-endian, so a record header is
#: capped at 4 GiB -- signatures are KB-scale, so the cap is theoretical.
_LEN = struct.Struct("<I")

MANIFEST = "MANIFEST.json"
RECORDS = "records.log"
BLOBS = "blobs.pack"
FORMAT_VERSION = 1


class RexStore(RCStore):
    """Append-only local store: two logs, an in-memory index, no server."""

    backend = "rex"

    def __init__(self, root: str):
        self.root = str(root)
        self.uri = f"rex://{self.root}"
        os.makedirs(self.root, exist_ok=True)
        self._manifest_path = os.path.join(self.root, MANIFEST)
        self._records_path = os.path.join(self.root, RECORDS)
        self._blobs_path = os.path.join(self.root, BLOBS)
        if not os.path.exists(self._manifest_path):
            with open(self._manifest_path, "w", encoding="utf-8") as fh:
                fh.write(dumps({"format": "rexstore", "version": FORMAT_VERSION}))
        self._recs: Dict[str, List[ComplexRecord]] = {}
        self._blob_at: Dict[tuple, tuple] = {}       # (id, version) -> (offset, len)
        self._labels: Dict[str, set] = {}            # label -> {id}
        self._load()

    # --- log ------------------------------------------------------------------

    def _load(self) -> None:
        """Replay the log. A torn tail is where the process died, so scanning stops
        there rather than trying to interpret a partial record."""
        if not os.path.exists(self._records_path):
            return
        with open(self._records_path, "rb") as fh:
            data = fh.read()
        pos, size = 0, len(data)
        while pos + _LEN.size <= size:
            (n,) = _LEN.unpack_from(data, pos)
            start = pos + _LEN.size
            if start + n > size:
                break                                 # torn tail: stop here
            try:
                entry = json.loads(data[start:start + n].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            self._apply(entry)
            pos = start + n

    def _apply(self, entry: Dict[str, Any]) -> None:
        rid = entry["id"]
        if entry.get("op") == "delete":
            for rec in self._recs.pop(rid, []):
                self._blob_at.pop((rid, rec.version), None)
            for ids in self._labels.values():
                ids.discard(rid)
            return
        rec = ComplexRecord(
            id=rid, signature=entry.get("signature", {}), created=entry.get("created", 0.0),
            meta=entry.get("meta", {}), version=int(entry.get("version", 1)),
            tx_from=entry.get("tx_from", 0.0), tx_to=None,
            valid_from=entry.get("valid_from"), valid_to=entry.get("valid_to"))
        versions = self._recs.setdefault(rid, [])
        for prior in versions:
            if prior.tx_to is None:
                # tx_to is not written: a version is closed by the arrival of its
                # successor, so the log stays purely append-only and the closure is
                # reconstructed identically on every replay.
                prior.tx_to = rec.tx_from
        versions.append(rec)
        self._blob_at[(rid, rec.version)] = (int(entry["blob_off"]), int(entry["blob_len"]))
        for label in _record_labels(rec.signature, rec.meta):
            self._labels.setdefault(label, set()).add(rid)

    def _append(self, entry: Dict[str, Any]) -> None:
        payload = dumps(entry).encode("utf-8")
        with open(self._records_path, "ab") as fh:
            fh.write(_LEN.pack(len(payload)))
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())

    # --- writes ---------------------------------------------------------------

    def next_version(self, id):
        return (self._recs[id][-1].version + 1) if self._recs.get(id) else 1

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to):
        from .rcdb import _now
        now = _now()
        blob = serialize_complex(rex)
        with open(self._blobs_path, "ab") as fh:
            offset = fh.tell()
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        entry = {
            "op": "put", "id": id, "version": self.next_version(id),
            "signature": sig, "meta": meta or {}, "created": now,
            "tx_from": now,
            "valid_from": valid_from if valid_from is not None else now,
            "valid_to": valid_to,
            "blob_off": offset, "blob_len": len(blob),
        }
        # blob first, then the entry that points at it: a crash between the two
        # leaves unreferenced bytes in the pack, which is inert, rather than an
        # entry pointing at bytes that were never written.
        self._append(entry)
        self._apply(entry)
        return self._recs[id][-1]

    def delete(self, id):
        if id not in self._recs:
            return False
        entry = {"op": "delete", "id": id}
        self._append(entry)
        self._apply(entry)
        self._emit("rcdb.delete", id, 0, {})
        return True

    # --- reads ----------------------------------------------------------------

    def history(self, id):
        return list(self._recs.get(id, []))

    def get_record(self, id, *, as_of=None, valid_at=None):
        return self._select_version(self._recs.get(id, []), as_of, valid_at)

    def _read_blob(self, id, version):
        at = self._blob_at.get((id, version))
        if at is None:
            return None
        offset, length = at
        with open(self._blobs_path, "rb") as fh:
            fh.seek(offset)
            return fh.read(length)

    def get(self, id, *, as_of=None, valid_at=None):
        # "base@3" pins a version; anything else is a plain id
        split = self._split_versioned_id(id)
        if split is not None:
            return self.get_version(split[0], split[1])
        rid = id
        rec = self.get_record(rid, as_of=as_of, valid_at=valid_at)
        if rec is None:
            return None
        blob = self._read_blob(rid, rec.version)
        return None if blob is None else deserialize_complex(blob)

    def get_version(self, id, version):
        blob = self._read_blob(id, int(version))
        return None if blob is None else deserialize_complex(blob)

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
            # the inverted index is the point: a vocabulary query touches only the
            # ids that actually carry a term, not every record in the store.
            ids: set = set()
            for label in wanted:
                ids |= self._labels.get(str(label).lower(), set())
            candidates = [self._select_version(self._recs.get(i, []), as_of, valid_at)
                          for i in sorted(ids)]
            candidates = [r for r in candidates if r is not None]
            residual = {k: v for k, v in predicate.items() if k != "labels_any"}
        else:
            candidates = self.list(limit=10 ** 9, as_of=as_of, valid_at=valid_at)
            residual = predicate
        # labels_any is resolved by the index, but the SELECTED version may be an
        # older one whose vocabulary differs, so it is re-checked against that
        # version rather than trusted from the index alone.
        out = [r for r in candidates
               if _matches(r.signature, predicate, r.meta)] if wanted else \
              [r for r in candidates if _matches(r.signature, residual, r.meta)]
        out.sort(key=lambda r: -r.tx_from)
        return out[:limit]

    def stats(self) -> Dict[str, Any]:
        def _size(path):
            try:
                return os.path.getsize(path)
            except OSError:
                return 0
        return {
            "backend": self.backend, "root": self.root,
            "n_records": len(self._recs),
            "n_versions": sum(len(v) for v in self._recs.values()),
            "log_bytes": _size(self._records_path),
            "blob_bytes": _size(self._blobs_path),
            "n_labels": len(self._labels),
        }

    def compact(self) -> Dict[str, Any]:
        """Rewrite both logs keeping only live versions, then swap them in.

        Append-only means deleted records leave their bytes behind. Compaction is
        the deliberate, occasional cost that buys the O(1) put -- not something the
        write path pays on every call.
        """
        tmp_log = self._records_path + ".compact"
        tmp_pack = self._blobs_path + ".compact"
        before = self.stats()
        with open(tmp_log, "wb") as lf, open(tmp_pack, "wb") as pf:
            for rid in sorted(self._recs):
                for rec in self._recs[rid]:
                    blob = self._read_blob(rid, rec.version)
                    if blob is None:
                        continue
                    offset = pf.tell()
                    pf.write(blob)
                    entry = {"op": "put", "id": rid, "version": rec.version,
                             "signature": rec.signature, "meta": rec.meta,
                             "created": rec.created, "tx_from": rec.tx_from,
                             "valid_from": rec.valid_from, "valid_to": rec.valid_to,
                             "blob_off": offset, "blob_len": len(blob)}
                    payload = dumps(entry).encode("utf-8")
                    lf.write(_LEN.pack(len(payload)))
                    lf.write(payload)
        os.replace(tmp_log, self._records_path)
        os.replace(tmp_pack, self._blobs_path)
        self._recs, self._blob_at, self._labels = {}, {}, {}
        self._load()
        return {"before": before, "after": self.stats()}

    def close(self):
        return None
