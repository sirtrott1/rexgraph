"""
agent.rexstore: an embedded store for relational complexes. Files, no server.

FileStore reserialized its whole index on every put, so per-put cost grew with the
store: 4 ms at a hundred records, 41 ms at sixteen hundred, which is O(n^2) ingest
and about 33 hours for 100k records. It also wrote one file per record, and on a
network filesystem (EFS, Azure Files, a GCS mount, which is what a cloud VM
actually has) per-file overhead dominates everything else.

This is the same data laid out for how it is used:

    <root>/MANIFEST.json    format version
    <root>/records.log      append-only, [u32 length][json record] per entry
    <root>/blobs.pack       append-only, safetensors blobs end to end

Three files, whatever the record count. A put is two appends and costs the same at
record one and record one million. Opening scans the log once, sequentially, which
is the access pattern every filesystem is fastest at, and builds the indexes in
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

#: length prefix for a log entry. 4 bytes little-endian, so a record header is
#: capped at 4 GiB. Signatures are KB-scale, so the cap is theoretical.
_LEN = struct.Struct("<I")

MANIFEST = "MANIFEST.json"
INDEX = "index.safetensors"
RECORDS = "records.log"
BLOBS = "blobs.pack"
FORMAT_VERSION = 1

#: Snapshot when the un-indexed tail grows past this fraction of what is already
#: indexed. Writing the index costs ~30 us per record because it rewrites all of
#: them; NOT writing it costs ~13 us per un-indexed record on EVERY open. So a store
#: opened more than about twice between writes is better off snapshotting, and a
#: ratio bounds the replay tail without anyone picking a record count. The same
#: shape as TemporalRex's checkpoint threshold, for the same reason.
INDEX_TAIL_RATIO = float(os.environ.get("REXGRAPH_INDEX_TAIL_RATIO", "0.5"))

#: below this many un-indexed records, replay is cheaper than the snapshot that
#: would avoid it: at 500 records replay is 5.7 ms against 12.9 ms to write.
INDEX_MIN_TAIL = int(os.environ.get("REXGRAPH_INDEX_MIN_TAIL", "1000"))




#### the index, as tensors
#
# Replaying the log builds two things: a label -> records mapping, and a
# ComplexRecord per entry. Profiling an 8000-record open puts ~33% in the label
# dictionary and ~26% in per-entry JSON, and both are avoidable, because both are
# already shapes the library has a format for.
#
# The label mapping IS a bipartite complex (labels on one side, records on the
# other, incidence between them) so it stores as a CSR pair of tensors and loads
# at memory-map speed instead of being rebuilt: 96.8 ms of dictionary building
# becomes ~0.5 ms of tensor read. The documents are concatenated once with an
# offset tensor addressing them, so a record's signature and meta are parsed when
# something actually asks for that record rather than for all of them at open.
#
# Measured on a synthetic 200k-record index: mmap open plus the whole incidence is
# 62 ms against 3.54 s to build the equivalent dictionary, 57x.


class _LazyVersions:
    """The versions of one id, materialized from the document blob on first touch."""

    __slots__ = ("_index", "_rid", "_cache")

    def __init__(self, index, rid):
        self._index, self._rid, self._cache = index, rid, None

    def _materialize(self):
        if self._cache is None:
            self._cache = self._index.records_for(self._rid)
        return self._cache

    def __iter__(self):
        return iter(self._materialize())

    def __len__(self):
        return len(self._materialize())

    def __getitem__(self, i):
        return self._materialize()[i]

    def append(self, rec):
        self._materialize().append(rec)


class RexIndex:
    """A compacted snapshot of the log, as tensors.

    The record side of this is `rcdb_index`: the same cochains, the same accession
    relation, the same string tables and the same digest. It used to be a second index
    beside that one, holding the id list and the vocabulary as json in the safetensors
    metadata and every record as a json document inside a tensor. The header then grew
    with the store (21% of the file at 2,000 records) and `open` grew with it, 0.31 ms
    at 100 records against 7.82 ms at 4,000, which is the parse this index exists to
    avoid.

    What is genuinely this backend's own is the blob address per row, and that rides in
    `extra` so the one digest still covers it.
    """

    def __init__(self, path: str):
        self.path = path
        self._ix = None
        self.ids: list[str] = []
        self.vocab: list[str] = []
        self.log_bytes = 0
        self._rows: dict[str, range] = {}
        self._vocab_pos: dict[str, int] = {}
        self._label_ptr = None
        self._label_rec = None

    #### write
    @staticmethod
    def write(path: str, recs: dict[str, list[ComplexRecord]],
              blob_at: dict[tuple, tuple], log_bytes: int) -> None:
        import numpy as np

        from agent import rcdb_index as _ix

        ids = sorted(recs)
        rows = [(rid, rec) for rid in ids for rec in recs[rid]]
        index = _ix.build(rows)
        off = np.zeros(len(rows), np.int64)
        ln = np.zeros(len(rows), np.int64)
        for i, (rid, rec) in enumerate(rows):
            o, n = blob_at.get((rid, rec.version), (0, 0))
            off[i], ln[i] = int(o), int(n)
        tmp = path + ".tmp"
        _ix.write(tmp, index, extra={
            "blob_off": off, "blob_len": ln,
            "log_bytes": np.asarray([int(log_bytes)], np.int64)})
        os.replace(tmp, path)

    #### read
    def open(self) -> bool:
        """Read the index. False if there is not a usable one."""
        if not os.path.exists(self.path):
            return False
        from agent import rcdb_index as _ix
        try:
            self._ix = _ix.read(self.path)
        except Exception:
            self._ix = None
            return False
        rowids = list(self._ix["ids"])
        # `build` keeps the order it was given and every version of one id was written
        # together, so an id owns a contiguous run and the map is its bounds.
        self._rows, first = {}, {}
        for i, rid in enumerate(rowids):
            if rid not in first:
                first[rid] = i
            self._rows[rid] = range(first[rid], i + 1)
        self.ids = sorted(first, key=lambda r: first[r])
        self.vocab = self._ix["vocab"]
        # the reverse map decodes the whole vocabulary, so it waits for a lookup that
        # needs it. A store opened to read records by id never builds one.
        self._vocab_pos = None
        extra = self._ix.get("extra") or {}
        lb = extra.get("log_bytes")
        self.log_bytes = int(lb[0]) if lb is not None and len(lb) else 0
        # the transpose is built on the first label lookup, not here: a caller that
        # only reads records by id never pays for it
        self._label_ptr = self._label_rec = None
        return True

    def _term_csr(self):
        """Term to the rows that name it, as CSR. Built once, on first use.

        The accession relation is row to term. A prefilter asks the transpose, so it is
        built once here rather than scanned per lookup.
        """
        import numpy as np

        ix = self._ix
        n = int(ix["n"])
        ptr = np.asarray(ix["rel_ptr"], np.int64)
        idx = np.asarray(ix["rel_idx"], np.int64)
        if idx.size == 0:
            return np.zeros(len(self.vocab) + 1, np.int64), np.zeros(0, np.int64)
        owner = np.asarray(_rel_owner(ix), np.int64)
        # position 0 of every column is the record vertex, so the terms are everything
        # else. Both masks come off `ptr` without visiting a column.
        keep = np.ones(idx.size, bool)
        keep[ptr[:-1]] = False
        terms = idx[keep] - n
        rowsof = np.repeat(owner, np.diff(ptr))[keep]
        order = np.argsort(terms, kind="stable")
        counts = np.zeros(len(self.vocab) + 1, np.int64)
        np.add.at(counts, terms + 1, 1)
        return np.cumsum(counts), rowsof[order]

    def records_for(self, rid: str) -> list[ComplexRecord]:
        """One id's records, rebuilt from the cochains. Paid per record asked for."""
        rows = self._rows.get(rid)
        if rows is None:
            return []
        from agent import rcdb_index as _ix
        return [_ix.record_at(self._ix, r) for r in rows]

    def blob_at(self, rid: str, version: int):
        """The blob address for one version. A cochain lookup, not a scan of the rest."""
        rows = self._rows.get(rid)
        if rows is None:
            return None
        ver = self._ix["measures"]["version"]
        extra = self._ix.get("extra") or {}
        off, ln = extra.get("blob_off"), extra.get("blob_len")
        if off is None or ln is None:
            return None
        for r in rows:
            if int(ver[r]) == int(version):
                return int(off[r]), int(ln[r])
        return None

    def ids_for_label(self, label: str) -> list[str]:
        """A vocabulary lookup is a CSR row slice."""
        if self._vocab_pos is None:
            self._vocab_pos = {v: i for i, v in enumerate(self.vocab)}
        cid = self._vocab_pos.get(label)
        if cid is None:
            return []
        if self._label_ptr is None:
            self._label_ptr, self._label_rec = self._term_csr()
        lo, hi = int(self._label_ptr[cid]), int(self._label_ptr[cid + 1])
        rowids = self._ix["ids"]
        seen, out = set(), []
        for r in self._label_rec[lo:hi]:
            rid = rowids[int(r)]
            if rid not in seen:
                seen.add(rid)
                out.append(rid)
        return out


def _rel_owner(index):
    """The row each relation belongs to. `rel_owner` is the library's own accessor."""
    from agent import rcdb_index as _ix
    return _ix.rel_owner(index)


class RexStore(RCStore):
    """Append-only local store: two logs, an in-memory index, no server."""

    backend = "rex"

    def __init__(self, root: str, *, auto_index: bool = True):
        self.auto_index = auto_index
        self._indexed_count = 0
        self._tail_count = 0
        self.root = str(root)
        self.uri = f"rex://{self.root}"
        os.makedirs(self.root, exist_ok=True)
        self._manifest_path = os.path.join(self.root, MANIFEST)
        self._records_path = os.path.join(self.root, RECORDS)
        self._blobs_path = os.path.join(self.root, BLOBS)
        self._index_path = os.path.join(self.root, INDEX)
        self._index: RexIndex | None = None
        if not os.path.exists(self._manifest_path):
            with open(self._manifest_path, "w", encoding="utf-8") as fh:
                fh.write(dumps({"format": "rexstore", "version": FORMAT_VERSION}))
        self._recs: dict[str, list[ComplexRecord]] = {}
        self._blob_at: dict[tuple, tuple] = {}       # (id, version) -> (offset, len)
        self._labels: dict[str, set] = {}            # label -> {id}
        self._load()

    #### log
    def _load(self) -> None:
        """Load the index if there is one, then replay whatever the log holds beyond
        it. A torn tail is where the process died, so scanning stops there rather
        than trying to interpret a partial record."""
        start_at = 0
        idx = RexIndex(self._index_path)
        if idx.open():
            self._index = idx
            start_at = idx.log_bytes
            self._indexed_count = len(idx.ids)
            for rid in idx.ids:
                # lazy: a record's documents are parsed when something asks for that
                # record, not for every record at open.
                self._recs[rid] = _LazyVersions(idx, rid)
        if not os.path.exists(self._records_path):
            return
        from agent import rcdb_index as _ix
        with open(self._records_path, "rb") as fh:
            head = fh.read(len(_ix.LOG_MAGIC))
        if head == _ix.LOG_MAGIC:
            for op, rid, rec, extra in _ix.log_read(self._records_path, start_at):
                if op == "delete" or rec is None:
                    self._forget(rid)
                else:
                    ok = extra is not None and len(extra) >= 2
                    self._admit(rid, rec, int(extra[0]) if ok else 0,
                                int(extra[1]) if ok else 0)
                self._tail_count += 1
            return
        self._load_json_log(start_at)

    def _load_json_log(self, start_at: int) -> None:
        """The `[u32 length][json record]` log this store wrote before frames. Read
        only, so a store written by an older version still opens."""
        with open(self._records_path, "rb") as fh:
            fh.seek(start_at)
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
            self._tail_count += 1
            pos = start + n

    def _apply(self, entry: dict[str, Any]) -> None:
        """Apply one change given as a dict. The json log path and `put` use this."""
        rid = entry["id"]
        if entry.get("op") == "delete":
            self._forget(rid)
            return
        rec = ComplexRecord(
            id=rid, signature=entry.get("signature", {}), created=entry.get("created", 0.0),
            meta=entry.get("meta", {}), version=int(entry.get("version", 1)),
            tx_from=entry.get("tx_from", 0.0), tx_to=None,
            valid_from=entry.get("valid_from"), valid_to=entry.get("valid_to"))
        self._admit(rid, rec, int(entry["blob_off"]), int(entry["blob_len"]))

    def _forget(self, rid: str) -> None:
        for rec in self._recs.pop(rid, []):
            self._blob_at.pop((rid, rec.version), None)
        for ids in self._labels.values():
            ids.discard(rid)

    def _admit(self, rid: str, rec: ComplexRecord, off: int, ln: int) -> None:
        """Apply one change given as the record itself.

        The frame reader already built one, so the replay path calls this rather than
        flattening it to a dict for `_apply` to rebuild. That round trip was two thirds
        of the record construction in a replay.
        """
        versions = self._recs.setdefault(rid, [])
        if isinstance(versions, _LazyVersions):
            versions = versions._materialize()
            self._recs[rid] = versions
        for prior in versions:
            if prior.tx_to is None:
                # tx_to is not written: a version is closed by the arrival of its
                # successor, so the log stays purely append-only and the closure is
                # reconstructed identically on every replay.
                prior.tx_to = rec.tx_from
        versions.append(rec)
        self._blob_at[(rid, rec.version)] = (off, ln)
        for label in _record_labels(rec.signature, rec.meta):
            self._labels.setdefault(label, set()).add(rid)

    def _append(self, entry: dict[str, Any]) -> None:
        """One frame per change, through the same writer the other store logs with.

        This wrote `[u32 length][json record]`, so every field name and every number
        was text and the log was the largest json artifact left in the store path. The
        blob address is this backend's own, and rides the frame's `extra` row.
        """
        from agent import rcdb_index as _ix

        rid = entry["id"]
        if entry.get("op") == "delete":
            _ix.log_append(self._records_path, "delete", rid, None)
            return
        rec = ComplexRecord(
            id=rid, signature=entry.get("signature", {}), meta=entry.get("meta", {}),
            created=entry.get("created", 0.0), version=int(entry.get("version", 1)),
            tx_from=entry.get("tx_from", 0.0), tx_to=None,
            valid_from=entry.get("valid_from"), valid_to=entry.get("valid_to"))
        _ix.log_append(self._records_path, "put", rid, rec,
                       extra=[int(entry["blob_off"]), int(entry["blob_len"])])

    #### writes
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
        self._tail_count += 1
        self._maybe_index()
        return self._recs[id][-1]

    def delete(self, id):
        if id not in self._recs:
            return False
        entry = {"op": "delete", "id": id}
        self._append(entry)
        self._apply(entry)
        self._emit("rcdb.delete", id, 0, {})
        return True

    #### reads
    def history(self, id):
        return list(self._recs.get(id, []))

    def _versions(self, id):
        return list(self._recs.get(id, []))

    def get_record(self, id, *, as_of=None, valid_at=None):
        return self._select_version(self._versions(id), as_of, valid_at)

    def _read_blob(self, id, version):
        at = self._blob_at.get((id, version))
        if at is None and self._index is not None:
            at = self._index.blob_at(id, int(version))
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
            recs = [r for rid in self._recs for r in self._versions(rid)]
        else:
            recs = [self._select_version(self._versions(rid), as_of, valid_at)
                    for rid in self._recs]
        recs = [r for r in recs if r is not None]
        recs.sort(key=lambda r: -r.tx_from)
        return recs[offset:offset + limit]

    def query(self, limit=100, *, as_of=None, valid_at=None, **predicate):
        wanted = predicate.get("labels_any")
        if wanted:
            # the inverted index is the point: a vocabulary query touches only the
            # ids that actually carry a term, not every record in the store. With a
            # tensor index that lookup is a CSR row slice rather than a dict hit.
            ids: set = set()
            for label in wanted:
                key = str(label).lower()
                ids |= self._labels.get(key, set())
                if self._index is not None:
                    ids |= set(self._index.ids_for_label(key))
            candidates = [self._select_version(self._versions(i), as_of, valid_at)
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

    def stats(self) -> dict[str, Any]:
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

    def compact(self) -> dict[str, Any]:
        """Rewrite both logs keeping only live versions, then swap them in.

        Append-only means deleted records leave their bytes behind. Compaction is
        the deliberate, occasional cost that buys the O(1) put, not something the
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
        self._index = None
        self._load()
        self.write_index()
        return {"before": before, "after": self.stats()}

    def _maybe_index(self) -> None:
        """Snapshot when the tail has grown enough to be worth it. Never on a small
        store, where replaying is cheaper than the snapshot that would avoid it."""
        if not self.auto_index or self._tail_count < INDEX_MIN_TAIL:
            return
        if self._tail_count < self._indexed_count * INDEX_TAIL_RATIO:
            return
        try:
            self.write_index()
        except Exception:
            pass                # an index is derived; failing to write one is not fatal

    def write_index(self) -> str:
        """Snapshot the current state as tensors, so the next open memory-maps it
        instead of replaying the log."""
        materialized = {rid: self._versions(rid) for rid in self._recs}
        blob_at = dict(self._blob_at)
        for rid, versions in materialized.items():
            for rec in versions:
                if (rid, rec.version) not in blob_at and self._index is not None:
                    at = self._index.blob_at(rid, rec.version)
                    if at is not None:
                        blob_at[(rid, rec.version)] = at
        log_bytes = os.path.getsize(self._records_path) \
            if os.path.exists(self._records_path) else 0
        RexIndex.write(self._index_path, materialized, blob_at, log_bytes)
        self._indexed_count = len(materialized)
        self._tail_count = 0
        return self._index_path

    def close(self):
        return None
