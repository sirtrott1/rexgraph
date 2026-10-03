"""Read only adapter for the historical packed RCDB layout.

The layout uses MANIFEST.json, records.log and blobs.pack, with optional tensor
and protected search indexes. Readers support strict legacy JSON/binary logs and
checked RGJL1 framing without rewriting source bytes. Corrupt or incomplete
published frames require explicit recovery.

RexStore does not implement the native engine's retained tombstones or version
high water rules. Migrate its available inventory into LocalStore or another
native provider using plan_legacy_migration/migrate_legacy_batch. Historical
writing remains an explicit read_only=False compatibility option during the
transition; production defaults use the canonical local engine.
"""

from __future__ import annotations

import os
from typing import Any

from rexgraph.io._compat import dumps

from .core import (
    ComplexRecord,
    RCStore,
    _matches,
    _owned_records,
    _record_labels,
    _serialized,
)

MANIFEST = "MANIFEST.json"
INDEX = "index.safetensors"
RECORDS = "records.log"
BLOBS = "blobs.pack"
SEARCH = "search.safetensors"
FORMAT_VERSION = 1

#: Snapshot when the un indexed tail grows past this fraction of what is already
#: indexed. Writing the index costs ~30 us per record because it rewrites all of
#: them; NOT writing it costs ~13 us per un indexed record on EVERY open. So a store
#: opened more than about twice between writes is better off snapshotting, and a
#: ratio bounds the replay tail without anyone picking a record count. The same
#: shape as TemporalRex's checkpoint threshold, for the same reason.
INDEX_TAIL_RATIO = float(os.environ.get("REXGRAPH_INDEX_TAIL_RATIO", "0.5"))

#: below this many un indexed records, replay is cheaper than the snapshot that
#: would avoid it: at 500 records replay is 5.7 ms against 12.9 ms to write.
INDEX_MIN_TAIL = int(os.environ.get("REXGRAPH_INDEX_MIN_TAIL", "1000"))




# Persist the label incidence as CSR tensors. Concatenated record bytes and
# offsets allow signatures and metadata to decode when requested.


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

    The record side is `rcdb_index`: the same cochains, the same accession relation,
    the same string tables and the same digest, so a record reads the same whichever
    backend holds it.

    What is this backend's own is the blob address per row, and that rides in `extra`
    so the one digest covers it.
    """

    def __init__(self, path: str):
        self.path = path
        self._ix = None
        self.ids: list[str] = []
        self.vocab: list[str] = []
        self.log_bytes = 0
        self.log_anchor = None
        self._rows: dict[str, range] = {}
        self._vocab_pos: dict[str, int] = {}
        self._label_ptr = None
        self._label_rec = None

    #### write
    @staticmethod
    def write(path: str, recs: dict[str, list[ComplexRecord]],
              blob_at: dict[tuple, tuple], log_bytes: int, *, log_anchor=None) -> None:
        import numpy as np

        from . import index as _ix

        ids = sorted(recs)
        rows = [(rid, rec) for rid in ids for rec in recs[rid]]
        index = _ix.build(rows)
        off = np.zeros(len(rows), np.int64)
        ln = np.zeros(len(rows), np.int64)
        for i, (rid, rec) in enumerate(rows):
            o, n = blob_at.get((rid, rec.version), (0, 0))
            off[i], ln[i] = int(o), int(n)
        tmp = path + ".tmp"
        extra = {
            "blob_off": off, "blob_len": ln,
            "log_bytes": np.asarray([int(log_bytes)], np.int64)}
        if log_anchor is not None:
            from rexgraph.value_codec import pack_value
            extra["log_anchor"] = np.frombuffer(pack_value(log_anchor.as_record()), np.uint8).copy()
        _ix.write(tmp, index, extra=extra)
        os.replace(tmp, path)

    #### read
    def open(self) -> bool:
        """Read the index. False if there is not a usable one."""
        if not os.path.exists(self.path):
            return False
        from . import index as _ix
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
        self.log_anchor = None
        if "log_anchor" in extra:
            from rexgraph.value_codec import unpack_value
            from .journal import JournalAnchor
            self.log_anchor = JournalAnchor.from_record(unpack_value(extra["log_anchor"].tobytes()))
            if self.log_anchor.byte_offset != self.log_bytes:
                raise ValueError("snapshot anchor differs from its legacy cursor projection")
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
        from . import index as _ix
        return [_ix.record_at(self._ix, r) for r in rows]

    def version_refs(self):
        """Return `(id, version)` rows without materializing record documents.

        The protected search index keys everything by record VERSION, and rebuilding its
        resolver only needs the pairs, so reconstructing each record to read one integer
        off it would be the expensive way to ask.
        """
        if self._ix is None:
            return []
        versions = self._ix["measures"]["version"]
        out = []
        for rid in self.ids:
            for row in self._rows.get(rid, ()):
                out.append((str(rid), int(versions[row])))
        return out

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
    from . import index as _ix
    return _ix.rel_owner(index)


#: Marks the protected search tokens riding a log frame's `extra` row. `extra` is a
#: variable length int64 row, so an older reader that takes only the first two words
#: still reads the blob address and simply does not see the tokens.
from .legacy import SEARCH_EXTRA_MAGIC as _SEARCH_EXTRA_MAGIC, SEARCH_TOKEN_WORDS as _SEARCH_TOKEN_WORDS


class RexStore(RCStore):
    """Legacy packed layout reader; explicit ``read_only=False`` enables compatibility writes."""

    def _load_store_identity(self):
        from .store_identity import bound_identity, local_identity
        from .journal import journal_identity
        path = os.path.join(self.root, ".rcdb-identity")
        known = journal_identity(self._records_path)
        if known is None and not os.path.exists(path):
            known = bound_identity(record for versions in self._recs.values() for record in versions)
        return local_identity(path, backend="rex", existing_id=known, read_only=self.read_only)

    backend = "rex"
    _cache_corpus = True

    def __init__(self, root: str, *, auto_index: bool = True,
                 search_policy=None, search_keys=None, read_only=True):
        if type(read_only) is not bool:
            raise TypeError("read_only must be a bool")
        self.read_only = read_only
        self.auto_index = auto_index
        self.search_policy = search_policy
        self.search_keys = search_keys
        self._search = None
        self._search_tail: dict[bytes, set[tuple[str, int]]] = {}
        self._search_records: dict[bytes, tuple[str, int]] = {}
        self._indexed_count = 0
        self._tail_count = 0
        self.root = str(root)
        self.uri = f"rex://{self.root}"
        from .core import _existing_backend
        if _existing_backend(self.root) not in {None, "rex"}:
            raise ValueError("RexStore cannot adopt another store layout")
        if read_only:
            from .legacy import check_source_root, source_fingerprint
            check_source_root(self.root, "rex")
            self._legacy_source_fingerprint = source_fingerprint(self.root)
        else:
            os.makedirs(self.root, exist_ok=True)
        self._manifest_path = os.path.join(self.root, MANIFEST)
        self._records_path = os.path.join(self.root, RECORDS)
        self._blobs_path = os.path.join(self.root, BLOBS)
        self._index_path = os.path.join(self.root, INDEX)
        self._search_path = os.path.join(self.root, SEARCH)
        self._commits_path = os.path.join(self.root, "commits")
        self._index: RexIndex | None = None
        if not read_only and not os.path.exists(self._manifest_path):
            with open(self._manifest_path, "w", encoding="utf-8") as fh:
                fh.write(dumps({"format": "rexstore", "version": FORMAT_VERSION}))
        self._recs: dict[str, list[ComplexRecord]] = {}
        self._blob_at: dict[tuple, tuple] = {}       # (id, version) -> (offset, len)
        self._labels: dict[str, set] = {}            # label -> {id}, public mode only
        self._load()
        if not os.path.exists(os.path.join(self.root, ".rcdb-identity")):
            from .store_identity import bound_identity
            if bound_identity(record for versions in self._recs.values() for record in versions) is not None:
                self._load_store_identity()
        self._load_search_index()
        self._rebuild_search_records()
        if read_only and source_fingerprint(self.root) != self._legacy_source_fingerprint:
            raise ValueError("legacy source changed while opening its read-only snapshot")

    #### protected search
    def _protected_labels(self) -> bool:
        """Whether labels are held as tokens rather than in the clear."""
        if self.search_policy is None:
            return False
        return any(self.search_policy.mode(kind) in {"keyed", "structural"}
                   for kind in ("vertex_labels", "labels_sample"))

    def _label_kind(self) -> str:
        return "vertex_labels" if self.search_policy.mode("vertex_labels") != "none" \
            else "labels_sample"

    def _label_token(self, label: str) -> bytes:
        from .protected_index import term_token
        kind = self._label_kind()
        return term_token(kind, str(label).lower(), mode=self.search_policy.mode(kind),
                          key_id=self.search_policy.key_id, keys=self.search_keys)

    def _record_token(self, rid: str, version: int) -> bytes:
        from .protected_index import version_record_token
        return version_record_token(rid, version, key_id=self.search_policy.key_id,
                                    keys=self.search_keys)

    @staticmethod
    def _tokens_extra(tokens):
        """Pack tokens into the int64 `extra` row behind a magic word."""
        import numpy as np
        values = [int(_SEARCH_EXTRA_MAGIC), int(len(tokens))]
        for token in tokens:
            raw = bytes(token)
            if len(raw) != 32:
                raise ValueError("protected search tokens must be 32 bytes")
            values.extend(np.frombuffer(raw, dtype="<i8").tolist())
        return values

    @staticmethod
    def _tokens_from_extra(extra):
        """Unpack them, or None when this frame carries none."""
        import numpy as np
        if extra is None or len(extra) == 2:
            return None
        if len(extra) < 4 or int(extra[2]) != _SEARCH_EXTRA_MAGIC:
            raise ValueError("unknown Rex journal backend coordinates")
        count = int(extra[3])
        start = 4
        end = start + count * _SEARCH_TOKEN_WORDS
        if count <= 0 or end != len(extra):
            raise ValueError("invalid Rex journal search token count")
        out = []
        for at in range(start, end, _SEARCH_TOKEN_WORDS):
            words = np.asarray(extra[at:at + _SEARCH_TOKEN_WORDS], dtype="<i8")
            out.append(words.tobytes())
        return tuple(out)

    def _load_search_index(self) -> None:
        """Adopt the persisted relation only when it was built under this policy.

        A relation built under a different policy tokenises the same term differently, so
        adopting it would turn every lookup into a silent miss.
        """
        if self.search_policy is None or not os.path.exists(self._search_path):
            return
        from .protected_index import load_search_relation
        try:
            rel = load_search_relation(self._search_path)
            if rel.policy_digest == self.search_policy.digest:
                self._search = rel
        except Exception:                            # noqa: BLE001  # derived, not fatal
            self._search = None

    def _rebuild_search_records(self) -> None:
        """The token to (id, version) resolver, which lives only in memory."""
        if not self._protected_labels():
            self._search_records = {}
            return
        records = {}
        if self._search is not None and self._search.record_ids is not None:
            for token, ref in zip(self._search.record_tokens, self._search.record_ids,
                                  strict=False):
                rid, sep, version = str(ref).rpartition("\x00")
                if sep:
                    records[bytes(token)] = (rid, int(version))
        elif self._index is not None:
            for rid, version in self._index.version_refs():
                records[self._record_token(rid, version)] = (rid, version)
        for rid, versions in self._recs.items():
            if isinstance(versions, _LazyVersions):
                continue
            for rec in versions:
                records[self._record_token(rid, rec.version)] = (rid, rec.version)
        self._search_records = records

    def _search_versions(self, label: str) -> set[tuple[str, int]]:
        """Which record versions carry this exact term, across index and un indexed tail."""
        if self.search_policy is None:
            return set()
        token = self._label_token(label)
        refs = set(self._search_tail.get(token, ()))
        if self._search is not None:
            for record in self._search.tokens_for(
                    self._label_kind(), str(label).lower(), policy=self.search_policy,
                    keys=self.search_keys):
                ref = self._search_records.get(record)
                if ref is not None:
                    refs.add(ref)
        return refs

    #### log
    def _load(self, *, use_index=True) -> None:
        """Use a sealed snapshot only at its validated journal anchor, then replay."""
        from . import index as _ix
        from .journal import JOURNAL_MAGIC, LocalJournal, journal_identity
        head, owner = b"", None
        if os.path.exists(self._records_path):
            with open(self._records_path, "rb") as fh:
                head = fh.read(len(_ix.LOG_MAGIC))
            if journal_identity(self._records_path) is not None:
                owner = self.store_id
        start_at = 0
        idx = RexIndex(self._index_path)
        usable = use_index and idx.open()
        if usable and head.startswith(JOURNAL_MAGIC):
            usable = idx.log_anchor is not None
            if usable:
                try:
                    LocalJournal(self._records_path, store_id=owner).check_anchor(idx.log_anchor)
                except ValueError:
                    # The full record journal is authoritative; a stale derived
                    # snapshot can be discarded and rebuilt after format migration.
                    usable = False
        elif usable and idx.log_anchor is not None:
            usable = False
        if usable:
            self._index = idx
            start_at = idx.log_bytes
            self._indexed_count = len(idx.ids)
            for rid in idx.ids:
                # lazy: a record's documents are parsed when something asks for that
                # record, not for every record at open.
                self._recs[rid] = _LazyVersions(idx, rid)
        if not os.path.exists(self._records_path):
            if idx.ids:
                raise ValueError("record journal is missing beneath a nonempty derived snapshot")
            return
        if head == _ix.LOG_MAGIC or head.startswith(JOURNAL_MAGIC):
            blob_size = os.path.getsize(self._blobs_path) if os.path.exists(self._blobs_path) else 0
            for op, rid, rec, extra in _ix.log_read(self._records_path, start_at, store_id=owner):
                if op == "delete" or rec is None:
                    self._forget(rid)
                else:
                    if (extra is None or len(extra) < 2 or extra[0] < 0 or extra[1] <= 0
                            or int(extra[0])+int(extra[1]) > blob_size):
                        raise ValueError("Rex journal blob coordinates exceed the published pack")
                    self._admit(rid, rec, int(extra[0]), int(extra[1]), self._tokens_from_extra(extra))
                self._tail_count += 1
            return
        self._load_json_log(start_at)

    def _load_json_log(self, start_at: int) -> None:
        """The `[u32 length][json record]` log, read only, so a store written by
        an older version still opens."""
        from .legacy import rex_json_entries
        blob_size = os.path.getsize(self._blobs_path) if os.path.exists(self._blobs_path) else 0
        for operation, rid, record, extra in rex_json_entries(self._records_path, start_at):
            if operation == "delete":
                self._forget(rid)
            else:
                if extra[0]+extra[1] > blob_size:
                    raise ValueError("legacy Rex journal blob coordinates exceed the published pack")
                self._admit(rid, record, extra[0], extra[1], self._tokens_from_extra(extra))
            self._tail_count += 1

    def _apply(self, entry: dict[str, Any]) -> None:
        """Apply one change given as a dict. The json log path and `put` use this."""
        from .legacy import rex_change
        operation, rid, rec, extra = rex_change(entry)
        if operation == "delete":
            self._forget(rid)
            return
        self._admit(rid, rec, extra[0], extra[1], self._tokens_from_extra(extra))

    def _forget(self, rid: str) -> None:
        for rec in self._recs.pop(rid, []):
            self._blob_at.pop((rid, rec.version), None)
        for ids in self._labels.values():
            ids.discard(rid)
        for refs in self._search_tail.values():
            refs.discard(rid)
        if self._protected_labels():
            self._search_records = {token: ref for token, ref
                                    in self._search_records.items() if ref[0] != rid}

    def _admit(self, rid: str, rec: ComplexRecord, off: int, ln: int,
               search_tokens=None) -> None:
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
                # successor, so the log stays purely append only and the closure is
                # reconstructed identically on every replay.
                prior.tx_to = rec.tx_from
        versions.append(rec)
        self._blob_at[(rid, rec.version)] = (off, ln)
        if self._protected_labels():
            # Tokens ride the frame, so a replay does not need the key to rebuild the
            # tail: it re adds tokens it cannot itself compute.
            ref = (rid, int(rec.version))
            self._search_records[self._record_token(*ref)] = ref
            tokens = search_tokens
            if tokens is None:
                tokens = tuple(self._label_token(label)
                               for label in _record_labels(rec.signature, rec.meta))
            for token in tokens:
                self._search_tail.setdefault(bytes(token), set()).add(ref)
        else:
            for label in _record_labels(rec.signature, rec.meta):
                self._labels.setdefault(label, set()).add(rid)

    def _append(self, entry: dict[str, Any]) -> None:
        """One frame per change, through the same writer the other store logs with.

        The frame carries the record; the blob address is this backend's own and rides
        its `extra` row.
        """
        from . import index as _ix
        from .journal import JOURNAL_MAGIC
        from .legacy import rex_change
        operation, rid, record, extra = rex_change(entry)
        # Publish a whole file migration before the first checked append. Never
        # add a different grammar after an old JSON prefix.
        if os.path.exists(self._records_path):
            with open(self._records_path, "rb") as fh:
                head = fh.read(len(_ix.LOG_MAGIC))
            if head and head != _ix.LOG_MAGIC and not head.startswith(JOURNAL_MAGIC):
                _ix.migrate_legacy_log(self._records_path, store_id=self.store_id, legacy_format="rex-json")
        _ix.log_append(self._records_path, operation, rid, record, extra=extra, store_id=self.store_id)

    #### mutation artifacts
    def _commit_path(self, id, version) -> str:
        from rexgraph.state import RESERVED_PATH, encode_name
        name = encode_name(str(id), RESERVED_PATH)
        return os.path.join(self._commits_path, f"{name}@{int(version)}.rexpkg")

    def _store_commit_bytes(self, id, version, blob):
        self._check_writable()
        path = self._commit_path(id, version)
        os.makedirs(self._commits_path, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(bytes(blob))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def _load_commit_bytes(self, id, version):
        path = self._commit_path(id, version)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return fh.read()

    def _delete_commit_bytes(self, id, version):
        import contextlib
        with contextlib.suppress(OSError):
            os.unlink(self._commit_path(id, version))

    #### writes
    def next_version(self, id):
        return (self._recs[id][-1].version + 1) if self._recs.get(id) else 1

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to,
                  search_terms=None, tx_time=None):
        from .core import _now
        now = _now() if tx_time is None else float(tx_time)
        version = self.next_version(id)
        rec = ComplexRecord(id, sig, now, meta or {}, version, now, None,
                            valid_from if valid_from is not None else now, valid_to)
        rec, blob = self._record_payload(rec, rex)
        with open(self._blobs_path, "ab") as fh:
            offset = fh.tell()
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        protected_tokens = ()
        if self._protected_labels() and search_terms:
            protected_tokens = tuple(self._label_token(label) for label in search_terms)
        entry = {
            "op": "put", "id": id, "version": version,
            "signature": sig, "meta": meta or {}, "created": now,
            "tx_from": now,
            "valid_from": valid_from if valid_from is not None else now,
            "valid_to": valid_to,
            "blob_off": offset, "blob_len": len(blob),
            "search_tokens": [token.hex() for token in protected_tokens],
            "envelope": rec.envelope.as_record(),
        }
        # blob first, then the entry that points at it: a crash between the two
        # leaves unreferenced bytes in the pack, which is inert, rather than an
        # entry pointing at bytes that were never written.
        self._append(entry)
        self._apply(entry)
        self._tail_count += 1
        self._maybe_index()
        return self._recs[id][-1]

    @_serialized
    def delete(self, id):
        reclaim = self._claim_delete(id)
        if id not in self._recs:
            return False
        entry = {"op": "delete", "id": id}
        self._append(entry)
        self._apply(entry)
        self._reclaim_commits(id, reclaim)
        self._emit("rcdb.delete", id, 0, {})
        return True

    #### reads
    @_owned_records
    def history(self, id):
        return list(self._recs.get(id, []))

    def _versions(self, id):
        return list(self._recs.get(id, []))

    @_owned_records
    def get_record(self, id, *, as_of=None, valid_at=None):
        return self._select_version(self._versions(id), as_of, valid_at)

    def _read_blob(self, id, version):
        if not any(int(r.version) == int(version) for r in self._versions(id)):
            return None
        at = self._blob_at.get((id, version))
        if at is None and self._index is not None:
            at = self._index.blob_at(id, int(version))
        if at is None:
            return None
        offset, length = at
        with open(self._blobs_path, "rb") as fh:
            fh.seek(offset)
            return fh.read(length)

    def get(self, id, *, as_of=None, valid_at=None, verify=True):
        if id not in self._recs:
            split = self._split_versioned_id(id)
            if split is not None:
                return self._get_version(split[0], split[1], verify=verify)
        rid = id
        rec = self.get_record(rid, as_of=as_of, valid_at=valid_at)
        if rec is None:
            return None
        blob = self._read_blob(rid, rec.version)
        return None if blob is None else self._read_record_payload(rec, blob, verify=verify)

    def get_version(self, id, version):
        return self._get_version(id, version)

    def _get_version(self, id, version, *, verify=True):
        rec = next((r for r in self._versions(id) if r.version == version), None)
        if rec is None:
            return None
        blob = self._read_blob(id, int(version))
        return None if blob is None else self._read_record_payload(rec, blob, verify=verify)

    @_owned_records
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

    @_owned_records
    def query(self, limit=100, *, as_of=None, valid_at=None, **predicate):
        wanted = predicate.get("labels_any")
        if wanted:
            # the inverted index is the point: a vocabulary query touches only the
            # ids that actually carry a term, not every record in the store. With a
            # tensor index that lookup is a CSR row slice rather than a dict hit.
            ids: set = set()
            protected_versions: set[tuple[str, int]] = set()
            for label in wanted:
                key = str(label).lower()
                if self._protected_labels():
                    protected_versions |= self._search_versions(key)
                else:
                    ids |= self._labels.get(key, set())
                    if self._index is not None:
                        ids |= set(self._index.ids_for_label(key))
            if self._protected_labels():
                ids = {rid for rid, _version in protected_versions}
            candidates = [self._select_version(self._versions(i), as_of, valid_at)
                          for i in sorted(ids)]
            candidates = [r for r in candidates if r is not None]
            residual = {k: v for k, v in predicate.items() if k != "labels_any"}
        else:
            protected_versions = set()
            candidates = self.list(limit=10 ** 9, as_of=as_of, valid_at=valid_at)
            residual = predicate
        if wanted and self._protected_labels():
            # A protected record keeps no plaintext vocabulary to re check, so the
            # matching VERSION is what the index answered, and only the rest of the
            # predicate is applied to the record.
            out = [r for r in candidates
                   if (r.id, int(r.version)) in protected_versions
                   and _matches(r.signature, residual, r.meta)]
        elif wanted:
            # A public index leaves labels on the record, so the SELECTED version may be
            # an older one whose vocabulary differs and is re checked against its own
            # rather than trusted from the index alone.
            out = [r for r in candidates if _matches(r.signature, predicate, r.meta)]
        else:
            out = [r for r in candidates if _matches(r.signature, residual, r.meta)]
        out.sort(key=lambda r: -r.tx_from)
        return out[:limit]

    def stats(self) -> dict[str, Any]:
        def _size(path):
            try:
                return os.path.getsize(path)
            except OSError:
                return 0
        value = super().stats()
        value.update({
            "backend": self.backend, "root": self.root,
            "log_bytes": _size(self._records_path),
            "blob_bytes": _size(self._blobs_path),
            "n_labels": len(self._labels),
        })
        return value

    @_serialized
    def compact(self) -> dict[str, Any]:
        """Rewrite both logs keeping only live versions, then swap them in.

        Append only means deleted records leave their bytes behind. Compaction is
        the deliberate, occasional cost that buys the O(1) put, not something the
        write path pays on every call.
        """
        self._check_writable()
        tmp_log = self._records_path + ".compact"
        tmp_pack = self._blobs_path + ".compact"
        # Read before the logs are replaced, because compaction clears the tail it would
        # otherwise be rebuilt from.
        protected_tokens = (self._protected_tokens_by_version()
                            if self._protected_labels() else None)
        before = self.stats()
        from . import index as _ix
        from .journal import _header
        with open(tmp_log, "wb") as lf:
            lf.write(_header(self.store_id))
        with open(tmp_pack, "wb") as pf:
            for rid in sorted(self._recs):
                for rec in self._recs[rid]:
                    blob = self._read_blob(rid, rec.version)
                    if blob is None:
                        raise ValueError("cannot compact a record with missing blob bytes")
                    offset = pf.tell()
                    pf.write(blob)
                    extra = [offset, len(blob)]
                    tokens = (() if protected_tokens is None else
                              protected_tokens.get((rid, int(rec.version)), ()))
                    if tokens:
                        extra.extend(self._tokens_extra(tokens))
                    _ix.log_append(tmp_log, "put", rid, rec, extra=extra, store_id=self.store_id)
        os.replace(tmp_log, self._records_path)
        os.replace(tmp_pack, self._blobs_path)
        self._recs, self._blob_at, self._labels = {}, {}, {}
        self._search_tail = {}
        self._search_records = {}
        self._index = None
        self._search = None
        self._load(use_index=False)
        self.write_index(protected_tokens=protected_tokens)
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

    def _protected_tokens_by_version(self):
        """Current protected term tokens keyed by record version.

        Compaction and reindexing rebuild the relation from what is already held, so the
        tokens are carried forward rather than recomputed, which is what lets a store
        reindex without holding the search key.
        """
        out = {}
        if self._search is not None:
            rel = self._search
            for relation in range(len(rel.rel_ptr) - 1):
                lo, hi = int(rel.rel_ptr[relation]), int(rel.rel_ptr[relation + 1])
                span = rel.rel_idx[lo:hi]
                if span.size < 2:
                    continue
                row = int(span[0])
                if row < 0 or row >= rel.n_records:
                    continue
                ref = self._search_records.get(bytes(rel.record_tokens[row]))
                if ref is None:
                    continue
                values = out.setdefault(ref, set())
                for vertex in span[1:]:
                    pos = int(vertex) - rel.n_records
                    if 0 <= pos < rel.token_bytes.shape[0]:
                        values.add(bytes(rel.token_bytes[pos]))
        for token, refs in self._search_tail.items():
            for ref in refs:
                out.setdefault(ref, set()).add(bytes(token))
        return out

    @_serialized
    def write_index(self, *, protected_tokens=None) -> str:
        """Snapshot the current state as tensors, so the next open memory maps it
        instead of replaying the log."""
        self._check_writable()
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
        from .journal import LocalJournal, journal_identity
        log_anchor = (LocalJournal(self._records_path, store_id=self.store_id).anchor()
                      if journal_identity(self._records_path) is not None else None)
        RexIndex.write(self._index_path, materialized, blob_at, log_bytes, log_anchor=log_anchor)
        if self.search_policy is not None:
            from .protected_index import (
                build_search_relation,
                build_search_relation_from_tokens,
                save_search_relation,
            )
            if self._protected_labels():
                token_map = (self._protected_tokens_by_version()
                             if protected_tokens is None else protected_tokens)
                rows = [(rid, rec.version, token_map.get((rid, int(rec.version)), ()))
                        for rid in sorted(materialized) for rec in materialized[rid]]
                relation = build_search_relation_from_tokens(
                    rows, self.search_policy, kind=self._label_kind(),
                    keys=self.search_keys)
            else:
                rows = [(rid, rec) for rid in sorted(materialized)
                        for rec in materialized[rid]]
                relation = build_search_relation(rows, self.search_policy,
                                                 keys=self.search_keys)
            tmp_search = self._search_path + ".tmp"
            save_search_relation(tmp_search, relation)
            os.replace(tmp_search, self._search_path)
            self._search = relation
            self._search_tail = {}
            self._rebuild_search_records()
        self._indexed_count = len(materialized)
        self._tail_count = 0
        return self._index_path

    def close(self):
        return None
