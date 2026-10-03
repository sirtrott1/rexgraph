"""Common native store operations over StoreState and provider publication hooks.

Adapters own physical reads, immutable writes and publication arbitration. The
public selection, codec, history, mutation and collection behavior stays shared.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib

from .core import (ComplexRecord, RCStore, VersionConflictError,
                   _now, _read_selector, _QUERY_KEYS)


class NativeStateStore(RCStore):
    """Internal provider base; it has no independent storage or copy protocol."""
    _cache_corpus = True

    def _check_open(self):
        if self._closed:
            raise ValueError(f"{self.backend} store handle is closed")

    @property
    def blob_codec(self):
        return self.header.compression

    @property
    def header(self):
        return self._header

    def _load_store_identity(self):
        return self.header.identity

    def _serialize_payload(self, rex):
        from .engine import encode_native_payload
        return self._encode_payload(encode_native_payload(rex, self.blob_codec), object_type="RCDB.Rex")

    def put(self, *args, **kwargs):
        with self._scope(write=True):
            return super().put(*args, **kwargs)

    def put_prepared(self, *args, **kwargs):
        with self._scope(write=True):
            return super().put_prepared(*args, **kwargs)

    def commit_mutation(self, *args, **kwargs):
        with self._scope(write=True):
            return super().commit_mutation(*args, **kwargs)

    def next_version(self, id):
        with self._scope():
            return self._state.next_version(id)

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to,
                  search_terms=None, tx_time=None):
        with self._scope(write=True):
            now = _now() if tx_time is None else float(tx_time)
            record = ComplexRecord(id, sig, created=now, meta=meta, version=self._state.next_version(id),
                                   tx_from=now, valid_from=now if valid_from is None else valid_from,
                                   valid_to=valid_to)
            record, blob = self._record_payload(record, rex)
            digest = hashlib.sha256(blob).hexdigest()
            commit = self._load_commit_bytes(id, record.version)
            frame = self._state.prepare_put(record, blob_digest=digest,
                    commit_digest=None if commit is None else hashlib.sha256(commit).hexdigest())
            self._write_blob(digest, blob)
            self._publish(frame)
            return record

    def _selected(self, id, as_of=None, valid_at=None):
        return self._state.selected(id, as_of=as_of, valid_at=valid_at)

    def get_record(self, id, *, as_of=None, valid_at=None):
        with self._scope():
            return self._selected(id, as_of, valid_at)

    def _read_version(self, record, *, verify=True):
        change = self._state.change_for(record.id, record.version)
        if change is None:
            raise ValueError(f"published {self.backend} record has no engine transition")
        try:
            blob = self._read_blob(change.blob_digest)
        except FileNotFoundError as exc:
            raise ValueError(f"published {self.backend} record payload is missing") from exc
        if hashlib.sha256(blob).hexdigest() != change.blob_digest:
            raise ValueError(f"{self.backend} record payload differs from its published content address")
        if change.commit_digest is not None and self._load_commit_bytes(record.id, record.version) is None:
            raise ValueError(f"published {self.backend} mutation artifact is missing")
        return self._read_record_payload(record, blob, verify=verify)

    def get(self, id, *, as_of=None, valid_at=None, verify=True):
        with self._scope():
            record = self._selected(id, as_of, valid_at)
            return None if record is None else self._read_version(record, verify=verify)

    def get_version(self, id, version):
        with self._scope():
            record = self._state.record(id, version)
            return None if record is None else self._read_version(record)

    def _record_at_version(self, id, version):
        with self._scope():
            return self._state.record(id, version)

    def history(self, id):
        _read_selector(id, None, None, None)
        with self._scope():
            return self._state.history(id)

    def read_record(self, *args, **kwargs):
        with self._scope():
            return super().read_record(*args, **kwargs)

    def state_manifest(self):
        with self._scope():
            # The retained history includes a deletion's closed transaction
            # interval. Opaque store IDs, compression and physical journal
            # digests do not change this backend independent state identity.
            return super().state_manifest()

    @contextmanager
    def read_transaction(self):
        """Pin one provider snapshot across a complete dependency selection."""
        with self._scope():
            yield self

    @contextmanager
    def write_scope(self):
        with self._scope(write=True):
            yield self

    def corpus_snapshot(self, **kwargs):
        with self._scope():
            return super().corpus_snapshot(**kwargs)

    def list(self, limit=100, offset=0, *, as_of=None, valid_at=None, include_history=False):
        if type(limit) is not int or type(offset) is not int or limit < 0 or offset < 0:
            raise ValueError(f"{self.backend} collection bounds require nonnegative native integers")
        with self._scope():
            return self._state.select_records(limit=limit, offset=offset, as_of=as_of,
                                             valid_at=valid_at, include_history=include_history)

    def query(self, limit=100, *, as_of=None, valid_at=None, include_history=False, **predicate):
        if type(limit) is not int or limit < 0:
            raise ValueError(f"{self.backend} query limit requires a nonnegative native integer")
        if set(predicate)-_QUERY_KEYS:
            raise TypeError("unsupported query keys: "+", ".join(sorted(set(predicate)-_QUERY_KEYS))
                            +". Supported: "+", ".join(sorted(_QUERY_KEYS)))
        with self._scope():
            return self._state.select_records(limit=limit, as_of=as_of, valid_at=valid_at,
                                             include_history=include_history, predicate=predicate)

    def delete(self, id, *, tx_time=None, expected_version=None):
        _read_selector(id, None, tx_time, None)
        if expected_version is not None and (type(expected_version) is not int or expected_version < 0):
            raise ValueError("expected version requires a nonnegative native integer")
        with self._scope(write=True):
            self._claim_delete(id)
            current = self._state.current(id)
            actual = 0 if current is None else current.version
            if expected_version is not None and expected_version != actual:
                raise VersionConflictError(f"deletion expected version differs from current {self.backend} record")
            frame = self._state.prepare_delete(id, _now() if tx_time is None else tx_time)
            if frame is None:
                return False
            self._publish(frame)
            self._emit("rcdb.delete", id, frame.mutation.version, {})
            return True

    @property
    def change_cursor(self):
        with self._scope():
            return self._state.cursor

    def changes(self, after=None, *, limit=100):
        with self._scope():
            return self._state.changes(after, limit=limit)

    def cursor_for(self, change):
        with self._scope():
            return self._state.cursor_for(change)

    def tombstone(self, id):
        _read_selector(id, None, None, None)
        with self._scope():
            return self._state.tombstone(id)

    def commit_history(self, id):
        with self._scope():
            return [self._decode_commit(raw) for row in self._state.incarnation_history(id)
                    if (raw := self._load_commit_bytes(id, row.version)) is not None]

    def _commit_predecessor_reset(self, record):
        return self._state.change_for(record.id, record.version).previous_version == 0

    def verify_commits(self, id):
        with self._scope():
            return super().verify_commits(id)

    def close(self):
        with self._transaction_lock:
            self._closed = True
