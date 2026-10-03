"""Canonical local RCDB adapter over the shared checked record state engine.

The append only journal is authoritative. Payloads and mutation artifacts are
published first; unreferenced staging residues never become records. Tombstones
retain audit history and version allocation. This format does not rewrite or
truncate logical history during a cache refresh or ordinary deletion.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

from rexgraph.io.publication import publication_lock, staged_publication

from .core import PublicationUncertainError
from .checkpoint import CHECKPOINT_LIMIT, SEGMENT_LIMIT, LocalReplayCheckpoint, build_checkpoint
from .state_store import NativeStateStore
from .engine import StoreState
from .envelope import _RECORD_HEADER_LIMIT, _RECORD_PAYLOAD_LIMIT
from .header import BlobCodecSpec, HEADER_LIMIT, StoreHeader
from .journal import LocalJournal, write_checked_journal
from .store_identity import StoreIdentity

_BLOB_LIMIT = _RECORD_PAYLOAD_LIMIT+_RECORD_HEADER_LIMIT+1024*1024


def _read_bytes(path, limit):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("local store storage requires regular files")
    with path.open("rb") as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError("local store storage exceeds its byte limit")
    return raw


class LocalStore(NativeStateStore):
    """Durable local provider over the shared native store operations.

    Compression and codec declarations persist in the immutable store header.
    Tombstones retain version allocation, history and claimed artifacts.
    """
    backend = "local"

    def __init__(self, root, *, compression=None, record_codecs=None):
        if compression is not None and not isinstance(compression, BlobCodecSpec):
            raise TypeError("local compression requires a declared BlobCodecSpec")
        from .header import record_codec_inventory
        inventory = record_codec_inventory(record_codecs)
        self.root = os.fspath(root)
        self._root = Path(root)
        if self._root.is_symlink():
            raise ValueError("local store root cannot be a symbolic link")
        self._root = self._root.absolute()
        self._header_path = self._root / "store.header"
        self._journal_path = self._root / "records.journal"
        self._blobs = self._root / "blobs"
        self._commits = self._root / "commits"
        self._replay = self._root / "replay"
        self._checkpoint_path = self._root / "replay.checkpoint"
        self._closed, self._scope_depth, self._scope_write = False, 0, False
        if not self._header_path.exists():
            self._initialize(compression or BlobCodecSpec(), inventory)
        with publication_lock(self._root):
            self._header = StoreHeader.from_bytes(_read_bytes(self._header_path, HEADER_LIMIT+64))
            if self.header.identity.backend != self.backend:
                raise ValueError("local store header declares another backend")
            if compression is not None and self.header.compression != compression:
                raise ValueError("requested compression differs from the durable store header")
            if record_codecs is not None and inventory != self.header.record_codecs:
                raise ValueError("requested record codecs differ from the durable store header")
            self._store_identity = self.header.identity
            for directory in (self._blobs, self._commits):
                if directory.is_symlink() or not directory.is_dir():
                    raise ValueError("local store payload directories are missing or invalid")
            if not self._journal_path.exists():
                raise ValueError("authoritative local store journal is missing; explicit recovery required")
            self._journal = LocalJournal(self._journal_path, store_id=self.store_id)
            cached = self._read_checkpoint()
            if cached is None:
                self._state = StoreState.replay(self.header, self._journal.frames())
            else:
                if cached.checkpoint.header != self.header:
                    raise ValueError("local replay cache belongs to another store or configuration")
                self._journal.check_anchor(cached.anchor)
                if self._prefix_digest(cached.anchor.byte_offset) != cached.prefix_digest:
                    raise ValueError("authoritative local journal prefix differs from its replay cache")
                self._state = cached.checkpoint.restore(self._read_segment)
                for frame in self._journal.frames(cached.anchor.byte_offset):
                    self._state.apply(frame)
            self._anchor = self._journal.anchor()


    def _initialize(self, compression, record_codecs):
        self._root.parent.mkdir(parents=True, exist_ok=True)
        with publication_lock(self._root.parent):
            if self._header_path.exists():
                return
            if self._root.is_symlink() or (self._root.exists() and not self._root.is_dir()):
                raise ValueError("local store root requires a directory")
            if self._root.exists() and any(self._root.iterdir()):
                raise ValueError("store header is missing from existing data; explicit migration or recovery required")
            header = StoreHeader(StoreIdentity.create(self.backend), compression, record_codecs)
            with staged_publication(self._root, directory=True) as staged:
                (staged / "store.header").write_bytes(header.to_bytes())
                (staged / "blobs").mkdir()
                (staged / "commits").mkdir()
                with (staged / "records.journal").open("wb") as stream:
                    write_checked_journal(stream, header.identity.id, ())


    def _refresh(self):
        stored = StoreHeader.from_bytes(_read_bytes(self._header_path, HEADER_LIMIT+64))
        if stored != self.header:
            raise ValueError("local store configuration changed beneath this handle")
        self._journal.check_anchor(self._anchor)
        # Physical framing is checked before applying any semantic changes.
        frames = tuple(self._journal.frames(self._anchor.byte_offset))
        try:
            for frame in frames:
                self._state.apply(frame)
            self._anchor = self._journal.anchor()
        except BaseException:
            self._publication_uncertain = True
            raise
        if frames:
            self._corpus_cache = None


    @contextmanager
    def _scope(self, *, write=False):
        with self._transaction_lock:
            if self._closed:
                raise ValueError("local store handle is closed")
            if self._publication_uncertain:
                raise PublicationUncertainError("local store state is uncertain; reopen and verify")
            if self._scope_depth:
                if write and not self._scope_write:
                    raise ValueError("cannot publish inside a pinned local read transaction")
                self._scope_depth += 1
                try:
                    yield
                finally:
                    self._scope_depth -= 1
            else:
                with publication_lock(self._root):
                    self._refresh()
                    self._scope_depth, self._scope_write = 1, write
                    try:
                        yield
                    finally:
                        self._scope_depth, self._scope_write = 0, False


    def _write_once(self, directory, digest, raw):
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("local payload differs from its declared content address")
        target = directory / digest
        if target.exists() or target.is_symlink():
            if _read_bytes(target, _BLOB_LIMIT) != raw:
                raise ValueError("local payload address already contains different bytes")
            return
        with staged_publication(target) as staged:
            staged.write_bytes(raw)


    def _publish(self, frame):
        self._state.validate(frame)
        try:
            self._journal.publish(frame)
        except PublicationUncertainError:
            self._publication_uncertain = True
            raise
        try:
            self._state.apply(frame)
            self._anchor = self._journal.anchor()
        except BaseException as exc:
            self._publication_uncertain = True
            raise PublicationUncertainError("journal published but local state did not finalize; reopen and verify") from exc
        self._corpus_cache = None


    def _commit_path(self, id, version):
        from rexgraph.value_codec import pack_value
        address = hashlib.sha256(pack_value((self.store_id, id, version))).hexdigest()
        return self._commits / address


    def _store_commit_bytes(self, id, version, blob):
        with self._scope(write=True):
            if self._state.change_for(id, version) is not None:
                raise ValueError("cannot replace a published local mutation artifact")
            path = self._commit_path(id, version)
            if path.exists() and _read_bytes(path, _BLOB_LIMIT) != blob:
                raise ValueError("unpublished mutation artifact already occupies this address; explicit recovery required")
            with staged_publication(path) as staged:
                staged.write_bytes(blob)


    def _load_commit_bytes(self, id, version):
        with self._scope():
            path = self._commit_path(id, version)
            try:
                raw = _read_bytes(path, _BLOB_LIMIT)
            except FileNotFoundError as exc:
                change = self._state.change_for(id, version)
                if change is not None and change.commit_digest is not None:
                    raise ValueError("published local mutation artifact is missing") from exc
                return None
            change = self._state.change_for(id, version)
            if change is not None and hashlib.sha256(raw).hexdigest() != change.commit_digest:
                raise ValueError("local mutation artifact differs from its published digest")
            return raw


    def _delete_commit_bytes(self, id, version):
        with self._scope(write=True):
            if self._state.change_for(id, version) is not None:
                raise ValueError("cannot remove a published local mutation artifact")
            path = self._commit_path(id, version)
            if path.is_symlink():
                raise ValueError("local mutation artifact cannot be a symbolic link")
            path.unlink(missing_ok=True)


    def _write_blob(self, digest, raw):
        self._write_once(self._blobs, digest, raw)

    def _read_blob(self, digest):
        return _read_bytes(self._blobs / digest, _BLOB_LIMIT)

    def _read_checkpoint(self):
        try:
            raw = _read_bytes(self._checkpoint_path, CHECKPOINT_LIMIT+4096)
        except FileNotFoundError:
            return None
        return LocalReplayCheckpoint.from_bytes(raw)

    def _read_segment(self, reference):
        if self._replay.is_symlink() or not self._replay.is_dir():
            raise ValueError("local replay segment directory is missing or invalid")
        try:
            return _read_bytes(self._replay / reference.digest, SEGMENT_LIMIT)
        except FileNotFoundError as exc:
            raise ValueError("published local replay segment is missing") from exc

    def _prefix_digest(self, size):
        self._journal._check_path()
        digest = hashlib.sha256(b"rexgraph-local-replay-prefix\x00\x01")
        with self._journal_path.open("rb") as stream:
            left = size
            while left:
                chunk = stream.read(min(left, 1024*1024))
                if not chunk:
                    raise ValueError("authoritative local journal prefix is incomplete")
                digest.update(chunk); left -= len(chunk)
        return digest.hexdigest()

    def checkpoint(self, *, max_frames=256, target_bytes=8*1024*1024):
        """Create a checked replay cache while retaining the authoritative journal."""
        with self._scope(write=True):
            # A live engine has checked the prefix previously. Maintenance must
            # also refuse physical prefix damage introduced since that read.
            last = self._state.initial_cursor()
            from .engine import ChangeCursor
            for frame in self._journal.frames():
                last = ChangeCursor(frame.store_id, self.header.digest, frame.sequence, frame.digest)
            if last != self._state.cursor:
                raise ValueError("local checkpoint differs from the authoritative journal head")
            cached = self._read_checkpoint()
            previous = None if cached is None else cached.checkpoint
            if cached is not None:
                self._journal.check_anchor(cached.anchor)
                if self._prefix_digest(cached.anchor.byte_offset) != cached.prefix_digest:
                    raise ValueError("authoritative local journal prefix differs from its replay cache")
                previous.restore(self._read_segment)
            if self._replay.is_symlink() or (self._replay.exists() and not self._replay.is_dir()):
                raise ValueError("local replay segment directory is invalid")
            self._replay.mkdir(exist_ok=True)
            value = build_checkpoint(self._state, lambda reference, raw:
                    self._write_once(self._replay, reference.digest, raw),
                    previous=previous, max_frames=max_frames, target_bytes=target_bytes)
            ref = LocalReplayCheckpoint(value, self._anchor, self._prefix_digest(self._anchor.byte_offset))
            raw = ref.to_bytes()
            if cached == ref:
                return value
            prepared = False
            try:
                with staged_publication(self._checkpoint_path) as staged:
                    staged.write_bytes(raw)
                    prepared = True
            except BaseException as exc:
                if prepared:
                    self._publication_uncertain = True
                    if isinstance(exc, Exception):
                        raise PublicationUncertainError("local checkpoint publication is uncertain; reopen and verify") from exc
                raise
            return value
