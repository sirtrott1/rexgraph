"""Native object adapter: immutable checked changes and conditional head publication.

The head is the only publication point. Blobs, mutation artifacts and frames are
immutable content addressed objects; failed staging is inert. Retained history,
tombstones, codec inventories and cursors use the same StoreState as local/SQL.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import hmac

from rexgraph.value_codec import pack_value, unpack_value

from .core import PublicationUncertainError, VersionConflictError
from .checkpoint import CHECKPOINT_LIMIT, SEGMENT_LIMIT, ReplayCheckpoint, build_checkpoint
from .engine import ChangeCursor, StoreState
from .envelope import _RECORD_HEADER_LIMIT, _RECORD_PAYLOAD_LIMIT
from .header import BlobCodecSpec, HEADER_LIMIT, StoreHeader, record_codec_inventory
from .journal import FRAME_LIMIT, JournalFrame
from .object_publication import PublishedObject, publication_for
from .state_store import NativeStateStore
from .store_identity import StoreIdentity

HEAD_LIMIT = HEADER_LIMIT+CHECKPOINT_LIMIT+4096
_HEAD_MAGIC = b"RGOH1"
_HEAD_DOMAIN = b"rexgraph-object-head\x00\x01"
_BLOB_LIMIT = _RECORD_PAYLOAD_LIMIT+_RECORD_HEADER_LIMIT+1024*1024


@dataclass(frozen=True)
class ObjectHead:
    """Closed publication declaration; no provider tokens or credentials persist."""
    header: StoreHeader
    cursor: ChangeCursor
    checkpoint: ReplayCheckpoint | None = None
    epoch: int = 0

    def __post_init__(self):
        if not isinstance(self.header, StoreHeader) or not isinstance(self.cursor, ChangeCursor):
            raise TypeError("object head requires a declared store header and logical cursor")
        if (self.header.identity.backend != "object" or
                (self.cursor.store_id, self.cursor.header_digest) != (self.header.identity.id, self.header.digest)):
            raise ValueError("object head differs from its declared ownership or configuration")
        if type(self.epoch) is not int or not 0 <= self.epoch < 2**63:
            raise ValueError("object head maintenance epoch requires a bounded native integer")
        if self.checkpoint is not None and (not isinstance(self.checkpoint, ReplayCheckpoint)
                or self.checkpoint.header != self.header or self.checkpoint.cursor.sequence > self.cursor.sequence
                or (self.checkpoint.cursor.sequence == self.cursor.sequence and self.checkpoint.cursor != self.cursor)):
            raise ValueError("object replay checkpoint differs from its published head")

    def as_record(self):
        result = {"head_version": 1, "header_bytes": self.header.to_bytes(), "cursor": self.cursor.as_record()}
        if self.checkpoint is not None or self.epoch:
            result.update(head_version=2, checkpoint_bytes=None if self.checkpoint is None else self.checkpoint.to_bytes(),
                          epoch=self.epoch)
        return result

    def to_bytes(self):
        body = pack_value(self.as_record())
        if len(body) > HEAD_LIMIT-37:
            raise ValueError("object head exceeds its byte limit")
        return _HEAD_MAGIC+hashlib.sha256(_HEAD_DOMAIN+body).digest()+body

    @classmethod
    def from_bytes(cls, raw):
        if type(raw) is not bytes or not 37 < len(raw) <= HEAD_LIMIT or not raw.startswith(_HEAD_MAGIC):
            raise ValueError("invalid object head frame")
        body = raw[37:]
        if not hmac.compare_digest(raw[5:37], hashlib.sha256(_HEAD_DOMAIN+body).digest()):
            raise ValueError("object head digest mismatch")
        record = unpack_value(body)
        if type(record) is not dict or type(record.get("head_version")) is not int:
            raise ValueError("unknown object head declaration")
        fields = {"head_version", "header_bytes", "cursor"}
        if record["head_version"] == 1 and set(record) == fields:
            checkpoint, epoch = None, 0
        elif record["head_version"] == 2 and set(record) == fields | {"checkpoint_bytes", "epoch"}:
            checkpoint = (None if record["checkpoint_bytes"] is None else
                          ReplayCheckpoint.from_bytes(record["checkpoint_bytes"]))
            epoch = record["epoch"]
        else:
            raise ValueError("unknown object head declaration")
        result = cls(StoreHeader.from_bytes(record["header_bytes"]), ChangeCursor.from_record(record["cursor"]),
                     checkpoint, epoch)
        if result.to_bytes() != raw:
            raise ValueError("object head is not canonically encoded")
        return result


class NativeObjectStore(NativeStateStore):
    """Native RCStore over an explicit strong-read/CAS publication capability.

    File and memory fsspec providers are included. Remote providers must supply a
    publication_provider(fs, root) returning a bound ObjectPublication capability.
    The optional provider transaction reduces contention; CAS protects every head
    update even when no distributed writer lock exists. Conflicting writes refuse
    without retrying effectful mutation preparation. read_only=True never publishes.
    """
    backend = "object"

    def __init__(self, uri, *, publication_provider=None, compression=None, record_codecs=None,
                 read_only=False):
        from .objectstore import _fs_for
        if type(read_only) is not bool:
            raise TypeError("read_only must be a bool")
        if compression is not None and not isinstance(compression, BlobCodecSpec):
            raise TypeError("object compression requires a declared BlobCodecSpec")
        inventory = record_codec_inventory(record_codecs)
        self.uri, self.read_only = uri, read_only
        self.fs, self.root = _fs_for(uri)
        self._provider = publication_for(self.fs, self.root, publication_provider)
        self._closed, self._scope_depth, self._scope_write = False, 0, False
        self._pending_commits = {}
        import posixpath
        if any(self.fs.exists(posixpath.join(self.root, key)) for key in
               ("MANIFEST.json", "index.json", "index.log", "index.rexidx", "index.rexlog",
                "records.log", "records.journal", "blobs.pack")):
            raise ValueError("native object store cannot adopt a conflicting legacy or local layout")
        with nullcontext() if read_only else self._provider.transaction():
            value = self._read("store.head", HEAD_LIMIT)
            if value is None:
                if self._read("store.header", HEADER_LIMIT+64) is not None:
                    raise ValueError("authoritative object head is missing; explicit recovery required")
                if self._provider.has_objects():
                    raise ValueError("native object head is missing from existing data; migration or recovery required")
                if read_only:
                    raise ValueError("read-only native object source requires an existing head")
                header = StoreHeader(StoreIdentity.create(self.backend), compression or BlobCodecSpec(), inventory)
                value = self._create_head(ObjectHead(header, StoreState(header).cursor))
            head = ObjectHead.from_bytes(value.data)
            self._header, self._store_identity = head.header, head.header.identity
            if compression is not None and compression != self.blob_codec:
                raise ValueError("requested compression differs from the durable object header")
            if record_codecs is not None and inventory != self.header.record_codecs:
                raise ValueError("requested record codecs differ from the durable object header")
            anchor = self._read("store.header", HEADER_LIMIT+64)
            if anchor is None:
                if read_only or head.cursor.sequence != 0:
                    raise ValueError("immutable object ownership anchor is missing; explicit recovery required")
                # Genesis publishes identity in the same CAS as its initial head.
                # Concurrent initializers can finish this exact immutable anchor.
                self._write_immutable("store.header", self.header.to_bytes())
            elif anchor.data != self.header.to_bytes():
                raise ValueError("object ownership anchor differs from its head")
            self._state = StoreState(self.header)
            self._head_token, self._head_checkpoint, self._head_epoch = None, None, 0
            self._apply_head(value)

    def _check_writable(self):
        self._check_open()
        if self.read_only:
            raise PermissionError("native object store is read-only")

    def _read(self, key, limit):
        value = self._provider.read(key, limit=limit)
        if value is not None and (not isinstance(value, PublishedObject) or len(value.data) > limit):
            raise ValueError("object provider returned an undeclared or oversized read")
        return value

    def _acknowledged(self, result, raw):
        if not isinstance(result, PublishedObject) or result.data != raw:
            raise PublicationUncertainError("object provider returned an invalid publication acknowledgement; reopen and verify")
        return result

    def _create_head(self, head):
        raw = head.to_bytes()
        try:
            return self._acknowledged(self._provider.compare_and_swap("store.head", None, raw), raw)
        except VersionConflictError as exc:
            value = self._read("store.head", HEAD_LIMIT)
            if value is None:
                raise ValueError("competing object initializer left no readable head") from exc
            return value
        except Exception as exc:
            raise PublicationUncertainError("object initialization outcome is uncertain; reopen and verify") from exc

    def _write_immutable(self, key, raw):
        self._check_writable()
        try:
            result = self._provider.compare_and_swap(key, None, raw)
        except VersionConflictError as exc:
            existing = self._read(key, len(raw))
            if existing is None or existing.data != raw:
                raise ValueError("immutable object address already contains different bytes") from exc
        else:
            self._acknowledged(result, raw)

    def _apply_head(self, value):
        head = ObjectHead.from_bytes(value.data)
        if head.header != self.header:
            raise ValueError("object store identity or configuration changed beneath this handle")
        current = self._state.cursor
        changed = head.cursor != current
        if head.cursor.sequence < current.sequence:
            raise ValueError("object head regressed behind this handle's retained cursor")
        if head.epoch < self._head_epoch:
            raise ValueError("object head maintenance epoch regressed behind this handle")
        if head.checkpoint is not None and head.checkpoint.cursor.sequence > current.sequence:
            def read(reference):
                segment = self._read("replay/"+reference.digest, SEGMENT_LIMIT)
                if segment is None:
                    raise ValueError("published object replay segment is missing")
                return segment.data
            restored = head.checkpoint.restore(read)
            restored.check_cursor(current)
            self._state, current = restored, restored.cursor
        sequence, digest, tail = head.cursor.sequence, head.cursor.digest, []
        while sequence > current.sequence:
            raw = self._read("frames/"+digest, FRAME_LIMIT+64)
            if raw is None:
                raise ValueError("published object change frame is missing")
            frame = JournalFrame.from_bytes(raw.data)
            if (frame.digest, frame.sequence, frame.store_id) != (digest, sequence, self.store_id):
                raise ValueError("object change frame differs from its published address")
            tail.append(frame)
            sequence, digest = sequence-1, frame.previous
        if digest != current.digest:
            raise ValueError("object head does not extend this handle's retained history")
        for frame in reversed(tail):
            self._state.apply(frame)
        if self._state.cursor != head.cursor:
            raise ValueError("object replay differs from the published head")
        self._head_token = value.token
        self._head_checkpoint, self._head_epoch = head.checkpoint, head.epoch
        if changed:
            self._corpus_cache = None

    def _refresh(self):
        try:
            head = self._read("store.head", HEAD_LIMIT)
            anchor = self._read("store.header", HEADER_LIMIT+64)
            if head is None or anchor is None:
                raise ValueError("object ownership anchor or authoritative head is missing; explicit recovery required")
            if anchor.data != self.header.to_bytes():
                raise ValueError("immutable object configuration changed beneath this handle")
            self._apply_head(head)
        except BaseException:
            self._publication_uncertain = True
            raise

    @contextmanager
    def _scope(self, *, write=False):
        with self._transaction_lock:
            self._check_open()
            if self._publication_uncertain:
                raise PublicationUncertainError("object store state is uncertain; reopen and verify")
            if write:
                self._check_writable()
            if self._scope_depth:
                if write and not self._scope_write:
                    raise ValueError("cannot publish inside a pinned object read transaction")
                self._scope_depth += 1
                try:
                    yield
                finally:
                    self._scope_depth -= 1
            else:
                with self._provider.transaction() if write else nullcontext():
                    self._refresh()
                    self._scope_depth, self._scope_write = 1, write
                    try:
                        yield
                    except PublicationUncertainError:
                        self._publication_uncertain = True
                        raise
                    finally:
                        self._scope_depth, self._scope_write = 0, False
                        self._pending_commits.clear()

    def _publish(self, frame):
        self._check_writable()
        self._state.validate(frame)
        self._write_immutable("frames/"+frame.digest, frame.to_bytes())
        cursor = ChangeCursor(frame.store_id, self.header.digest, frame.sequence, frame.digest)
        raw = ObjectHead(self.header, cursor, self._head_checkpoint, self._head_epoch).to_bytes()
        try:
            result = self._provider.compare_and_swap("store.head", self._head_token, raw)
            self._acknowledged(result, raw)
        except VersionConflictError:
            raise
        except BaseException as exc:
            self._publication_uncertain = True
            if isinstance(exc, Exception):
                raise PublicationUncertainError("object head publication outcome is uncertain; reopen and verify") from exc
            raise
        try:
            self._state.apply(frame)
            self._head_token = result.token
        except BaseException as exc:
            self._publication_uncertain = True
            raise PublicationUncertainError("object head published but state did not finalize; reopen and verify") from exc
        self._corpus_cache = None

    def checkpoint(self, *, max_frames=256, target_bytes=8*1024*1024):
        """Coalesce retained checked changes; leave logical publications unchanged."""
        with self._scope(write=True):
            if self._head_checkpoint is not None:
                def read(reference):
                    segment = self._read("replay/"+reference.digest, SEGMENT_LIMIT)
                    if segment is None:
                        raise ValueError("published object replay segment is missing")
                    return segment.data
                self._head_checkpoint.restore(read)
            value = build_checkpoint(self._state, lambda reference, raw:
                    self._write_immutable("replay/"+reference.digest, raw),
                    previous=self._head_checkpoint, max_frames=max_frames, target_bytes=target_bytes)
            if value == self._head_checkpoint:
                return value
            head = ObjectHead(self.header, self._state.cursor, value, self._head_epoch+1)
            raw = head.to_bytes()
            try:
                published = self._acknowledged(self._provider.compare_and_swap("store.head", self._head_token, raw), raw)
            except VersionConflictError:
                raise
            except BaseException as exc:
                self._publication_uncertain = True
                if isinstance(exc, Exception):
                    raise PublicationUncertainError("object checkpoint publication is uncertain; reopen and verify") from exc
                raise
            self._head_token, self._head_checkpoint, self._head_epoch = published.token, value, head.epoch
            return value

    def _write_blob(self, digest, raw):
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("object payload differs from its declared content address")
        self._write_immutable("blobs/"+digest, raw)

    def _read_blob(self, digest):
        value = self._read("blobs/"+digest, _BLOB_LIMIT)
        if value is None:
            raise FileNotFoundError("published object payload is missing")
        return value.data

    def _store_commit_bytes(self, id, version, blob):
        with self._scope(write=True):
            if self._state.change_for(id, version) is not None:
                raise ValueError("cannot replace a published object mutation artifact")
            raw = bytes(blob)
            digest = hashlib.sha256(raw).hexdigest()
            previous = self._pending_commits.get((id, version))
            if previous is not None and previous != digest:
                raise ValueError("different object mutation artifact already staged in this transaction")
            self._write_immutable("commits/"+digest, raw)
            self._pending_commits[(id, version)] = digest

    def _load_commit_bytes(self, id, version):
        with self._scope():
            change = self._state.change_for(id, version)
            digest = (self._pending_commits.get((id, version)) if change is None else change.commit_digest)
            if digest is None:
                return None
            value = self._read("commits/"+digest, _BLOB_LIMIT)
            if value is None:
                raise ValueError("published or staged object mutation artifact is missing")
            if hashlib.sha256(value.data).hexdigest() != digest:
                raise ValueError("object mutation artifact differs from its published digest")
            return value.data

    def _delete_commit_bytes(self, id, version):
        with self._scope(write=True):
            if self._state.change_for(id, version) is not None:
                raise ValueError("cannot remove a published object mutation artifact")
            self._pending_commits.pop((id, version), None)
            # Uploaded bytes remain inert; retention/GC requires explicit policy.

    def stats(self):
        with self._scope():
            value = super().stats()
            value.update(journal_segments=self._state.cursor.sequence, native=True)
            return value

    def compact(self):
        """Refresh only. Ordinary maintenance never truncates authoritative history."""
        with self._scope():
            value = self.stats()
            return {"before": value, "after": dict(value), "history_retained": True}
