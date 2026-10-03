"""Closed, checked record journal frames shared by persistence adapters.

Physical log and segment adapters supply framing and durable publication. This
module defines the operation, exact metadata, ownership and sequence chain bytes;
it never interprets an unknown opcode as another operation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import os
from pathlib import Path
from numbers import Integral
import struct

from rexgraph._binary import Cursor, uint
from rexgraph.value_codec import pack_value, unpack_value

from .envelope import _hex_identity

FRAME_MAGIC = b"RGJF1"
FRAME_LIMIT = 64*1024*1024
GENESIS_DIGEST = "0"*64
_RECORD_FIELDS = {"id", "signature", "created", "meta", "version", "tx_from", "tx_to", "valid_from", "valid_to"}
_FIELDS = {"frame_version", "store_id", "sequence", "previous", "operation", "record_id", "record_bytes", "extra"}
JOURNAL_MAGIC = b"RGJL1"
_LENGTH = struct.Struct("<Q")
_MASK = 2**64-1


def exact_coordinates(extra):
    """Accept exact integer adapter values without coercing booleans or floats."""
    if extra is None:
        return None
    values = tuple(extra)
    if any(isinstance(value, bool) or not isinstance(value, Integral)
           or not -(2**63) <= value < 2**63 for value in values):
        raise ValueError("journal backend coordinates require exact int64 values")
    return tuple(int(value) for value in values)


class TornJournalError(ValueError):
    """A short final write; valid_end names the verified prefix for explicit repair."""
    def __init__(self, valid_end):
        self.valid_end = valid_end
        super().__init__(f"torn record journal after byte {valid_end}; explicit recovery required")


@dataclass(frozen=True)
class JournalFrame:
    """Immutable published change; accessing its record returns an owned value."""
    store_id: str
    sequence: int
    previous: str
    operation: str
    record_id: str
    record_bytes: bytes | None = None
    extra: tuple[int, ...] | None = None
    mutation: object | None = None

    def __post_init__(self):
        if self.mutation is not None:
            from .engine import RecordChange
            if not isinstance(self.mutation, RecordChange):
                raise TypeError("journal engine mutation must be a declared RecordChange")
        _hex_identity(self.store_id, 32, "journal store identity")
        _hex_identity(self.previous, 64, "journal predecessor")
        if type(self.sequence) is not int or not 0 < self.sequence < 2**63:
            raise ValueError("journal sequence requires a positive native integer")
        if type(self.record_id) is not str or not self.record_id:
            raise ValueError("journal record identity must be nonempty text")
        if type(self.operation) is not str or self.operation not in {"put", "delete"}:
            raise ValueError("unknown journal operation")
        if self.extra is not None and (type(self.extra) is not tuple or any(
                type(value) is not int or not -(2**63) <= value < 2**63 for value in self.extra)):
            raise ValueError("journal backend coordinates require exact int64 values")
        if self.operation == "delete":
            if self.record_bytes is not None or self.extra is not None:
                raise ValueError("journal delete cannot declare record payload or backend coordinates")
        else:
            if type(self.record_bytes) is not bytes or len(self.record_bytes) > FRAME_LIMIT:
                raise ValueError("journal put requires bounded native record metadata")
            record = self.record
            if record.id != self.record_id:
                raise ValueError("journal record differs from its declared address")
            if type(record.version) is not int or not 0 < record.version < 2**63:
                raise ValueError("journal record version requires a positive native integer")
            if record.envelope is not None:
                record.envelope.check_address(store_id=self.store_id, record_id=self.record_id,
                                              record_version=record.version)
                from .engine import metadata_digest
                if record.envelope.metadata_digest != metadata_digest(record):
                    raise ValueError("journal metadata differs from its record binding")

    @classmethod
    def put(cls, *, store_id, sequence, previous, record, extra=None, mutation=None):
        return cls(store_id, sequence, previous, "put", record.id,
                   pack_value(record.to_dict()), exact_coordinates(extra), mutation)

    @property
    def record(self):
        if self.record_bytes is None:
            return None
        from .core import ComplexRecord
        data = unpack_value(self.record_bytes)
        if type(data) is not dict or not _RECORD_FIELDS <= data.keys() or set(data)-_RECORD_FIELDS-{"envelope"}:
            raise ValueError("unknown or incomplete journal record declaration")
        return ComplexRecord.from_dict(data)

    def as_record(self):
        result = {"frame_version": 1, "store_id": self.store_id, "sequence": self.sequence,
                "previous": self.previous, "operation": self.operation, "record_id": self.record_id,
                "record_bytes": self.record_bytes, "extra": self.extra}
        if self.mutation is not None:
            result.update(frame_version=2, mutation=self.mutation.as_record())
        return result

    def _body(self):
        body = pack_value(self.as_record())
        if len(body) > FRAME_LIMIT:
            raise ValueError("journal frame exceeds its byte limit")
        return body

    @property
    def digest(self):
        # Native frame fields are frozen closed values, so their commitment is
        # stable. Application subclasses retain their existing dynamic behavior.
        from .engine import RecordChange
        native = type(self) is JournalFrame and (self.mutation is None or type(self.mutation) is RecordChange)
        cached = self.__dict__.get("_native_digest") if native else None
        if cached is None:
            cached = hashlib.sha256(b"rexgraph-record-journal\x00"+self._body()).hexdigest()
            if native:
                object.__setattr__(self, "_native_digest", cached)
        return cached

    def check_successor(self, *, store_id, sequence, previous):
        if (self.store_id, self.sequence, self.previous) != (store_id, sequence, previous):
            raise ValueError("journal store, sequence or predecessor differs from the expected chain")

    def to_bytes(self):
        body = self._body()
        digest = hashlib.sha256(b"rexgraph-record-journal\x00"+body).digest()
        return FRAME_MAGIC+uint(len(body))+digest+body

    @classmethod
    def from_bytes(cls, raw):
        if type(raw) is not bytes or len(raw) > FRAME_LIMIT+64:
            raise ValueError("journal frame exceeds its byte limit or is not bytes")
        cursor = Cursor(raw)
        if cursor.take(len(FRAME_MAGIC)) != FRAME_MAGIC:
            raise ValueError("unsupported journal frame format")
        size = cursor.uint()
        if not 0 < size <= FRAME_LIMIT:
            raise ValueError("invalid journal frame length")
        expected = cursor.take(32)
        body = cursor.take(size)
        cursor.finish()
        if not hmac.compare_digest(expected, hashlib.sha256(b"rexgraph-record-journal\x00"+body).digest()):
            raise ValueError("journal frame digest mismatch")
        record = unpack_value(body)
        if type(record) is not dict or type(record.get("frame_version")) is not int:
            raise ValueError("unknown journal frame declaration")
        version = record["frame_version"]
        if version == 1 and set(record) == _FIELDS:
            pass
        elif version == 2 and set(record) == _FIELDS | {"mutation"}:
            from .engine import RecordChange
            record["mutation"] = RecordChange.from_record(record["mutation"])
        else:
            raise ValueError("unknown journal frame declaration")
        return cls(**{key: value for key, value in record.items() if key != "frame_version"})


def _take(stream, size, *, valid_end):
    raw = stream.read(size)
    if len(raw) != size:
        raise TornJournalError(valid_end)
    return raw


def _stream_uint(stream, *, valid_end):
    raw = bytearray()
    for _ in range(10):
        byte = _take(stream, 1, valid_end=valid_end)[0]
        raw.append(byte)
        if byte < 128:
            return Cursor(bytes(raw)).uint()
    raise ValueError("oversized journal length declaration")


def _header(store_id):
    _hex_identity(store_id, 32, "journal store identity")
    body = pack_value({"journal_version": 1, "store_id": store_id})
    return JOURNAL_MAGIC+uint(len(body))+hashlib.sha256(b"rexgraph-journal-header\x00"+body).digest()+body


def _read_header(stream, *, store_id=None):
    stream.seek(0)
    magic = stream.read(len(JOURNAL_MAGIC))
    if not magic:
        return None, 0
    if len(magic) < len(JOURNAL_MAGIC) and JOURNAL_MAGIC.startswith(magic):
        raise TornJournalError(0)
    if magic != JOURNAL_MAGIC:
        raise ValueError("unsupported record journal format")
    size = _stream_uint(stream, valid_end=0)
    if not 0 < size <= 4096:
        raise ValueError("invalid record journal header length")
    digest = _take(stream, 32, valid_end=0)
    body = _take(stream, size, valid_end=0)
    if not hmac.compare_digest(digest, hashlib.sha256(b"rexgraph-journal-header\x00"+body).digest()):
        raise ValueError("record journal header digest mismatch")
    fields = unpack_value(body)
    if (type(fields) is not dict or set(fields) != {"journal_version", "store_id"}
            or type(fields["journal_version"]) is not int or fields["journal_version"] != 1):
        raise ValueError("unknown record journal header declaration")
    identity = fields["store_id"]
    _hex_identity(identity, 32, "journal store identity")
    if store_id is not None and identity != store_id:
        raise ValueError("record journal belongs to another store")
    return identity, stream.tell()


def journal_identity(path):
    """Read checked ownership without loading records; legacy logs return None."""
    journal = LocalJournal(path)
    journal._check_path()
    if not journal.path.exists():
        return None
    with journal.path.open("rb") as stream:
        magic = stream.read(len(JOURNAL_MAGIC))
        if magic and (magic == JOURNAL_MAGIC or JOURNAL_MAGIC.startswith(magic)):
            return _read_header(stream)[0]
    return None


def _block(frame):
    raw = frame.to_bytes()
    size = len(raw)
    return _LENGTH.pack(size)+_LENGTH.pack(size ^ _MASK)+raw+_LENGTH.pack(size)


def _read_block(stream, *, valid_end):
    first = stream.read(8)
    if not first:
        return None
    if len(first) != 8:
        raise TornJournalError(valid_end)
    size = _LENGTH.unpack(first)[0]
    inverse = _LENGTH.unpack(_take(stream, 8, valid_end=valid_end))[0]
    if size ^ inverse != _MASK or not 0 < size <= FRAME_LIMIT+64:
        raise ValueError("invalid record journal block length")
    frame = JournalFrame.from_bytes(_take(stream, size, valid_end=valid_end))
    footer = _LENGTH.unpack(_take(stream, 8, valid_end=valid_end))[0]
    if footer != size:
        raise ValueError("record journal block footer differs from its length")
    return frame


def _previous_at(stream, end, header_end):
    if end == header_end:
        return None
    if end < header_end+24:
        raise ValueError("journal cursor is not a complete block boundary")
    stream.seek(end-8)
    size = _LENGTH.unpack(_take(stream, 8, valid_end=header_end))[0]
    if not 0 < size <= FRAME_LIMIT+64 or end-size-24 < header_end:
        raise ValueError("journal cursor footer has an invalid block length")
    stream.seek(end-size-24)
    previous = _read_block(stream, valid_end=header_end)
    if stream.tell() != end:
        raise ValueError("journal cursor is not a complete block boundary")
    return previous


@dataclass(frozen=True)
class JournalStatus:
    store_id: str | None
    frame_count: int
    last_sequence: int
    last_digest: str
    valid_end: int
    size: int
    torn: bool


@dataclass(frozen=True)
class JournalAnchor:
    """Sealed snapshot's ownership and exact complete prefix commitment."""
    store_id: str
    sequence: int
    digest: str
    byte_offset: int

    def __post_init__(self):
        _hex_identity(self.store_id, 32, "journal store identity")
        _hex_identity(self.digest, 64, "journal anchor digest")
        if (type(self.sequence) is not int or not 0 <= self.sequence < 2**63
                or type(self.byte_offset) is not int or self.byte_offset <= 0
                or (self.sequence == 0 and self.digest != GENESIS_DIGEST)):
            raise ValueError("invalid journal snapshot anchor")

    def as_record(self):
        return {"anchor_version": 1, "store_id": self.store_id, "sequence": self.sequence,
                "digest": self.digest, "byte_offset": self.byte_offset}

    @classmethod
    def from_record(cls, record):
        if (type(record) is not dict or set(record) != {"anchor_version", "store_id", "sequence", "digest", "byte_offset"}
                or type(record["anchor_version"]) is not int or record["anchor_version"] != 1):
            raise ValueError("unknown journal snapshot anchor declaration")
        return cls(**{key: value for key, value in record.items() if key != "anchor_version"})


def write_checked_journal(stream, store_id, entries):
    """Encode a whole checked journal into owned staging storage, without locking.

    The caller owns publication and durability. ``entries`` carries the shared
    `(operation, record_id, record, extra)` adapter contract, in publication order.
    """
    header = _header(store_id)
    if stream.write(header) != len(header):
        raise OSError("short checked journal header write")
    previous, sequence = GENESIS_DIGEST, 0
    for operation, record_id, record, extra in entries:
        frame = JournalFrame(store_id, sequence+1, previous, operation, record_id,
                             None if record is None else pack_value(record.to_dict()), exact_coordinates(extra))
        block = _block(frame)
        if stream.write(block) != len(block):
            raise OSError("short checked journal block write")
        previous, sequence = frame.digest, frame.sequence
    return JournalAnchor(store_id, sequence, previous, stream.tell())


class LocalJournal:
    """Checked streaming local log with bounded tail work on each append.

    The directory publication lock arbitrates appends across POSIX processes
    and threads. This does not allocate record versions or protect whole store
    mutations; the record engine must hold its own transaction across those.
    """
    def __init__(self, path, *, store_id=None):
        self.path = Path(path)
        if store_id is not None:
            _hex_identity(store_id, 32, "journal store identity")
        self.store_id = store_id

    def _check_path(self):
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError("journal requires a regular file, not a symbolic link")

    def frames(self, start=0, *, allow_torn_tail=False):
        if type(start) is not int or start < 0:
            raise ValueError("journal cursor requires a nonnegative native byte offset")
        self._check_path()
        if not self.path.exists():
            if start:
                raise ValueError("journal cursor is beyond the published file")
            return
        try:
            with self.path.open("rb") as stream:
                identity, header_end = _read_header(stream, store_id=self.store_id)
                if identity is None:
                    if start:
                        raise ValueError("journal cursor is beyond the published file")
                    return
                if start in (0, header_end):
                    prior, sequence = GENESIS_DIGEST, 0
                    stream.seek(header_end)
                else:
                    if start > os.fstat(stream.fileno()).st_size:
                        raise ValueError("journal cursor is beyond the published file")
                    previous = _previous_at(stream, start, header_end)
                    if previous.store_id != identity:
                        raise ValueError("journal cursor predecessor belongs to another store")
                    prior, sequence = previous.digest, previous.sequence
                while True:
                    end = stream.tell()
                    frame = _read_block(stream, valid_end=end)
                    if frame is None:
                        return
                    frame.check_successor(store_id=identity, sequence=sequence+1, previous=prior)
                    yield frame
                    prior, sequence = frame.digest, frame.sequence
        except TornJournalError:
            if not allow_torn_tail:
                raise

    def inspect(self):
        self._check_path()
        if not self.path.exists():
            return JournalStatus(self.store_id, 0, 0, GENESIS_DIGEST, 0, 0, False)
        size = self.path.stat().st_size
        count, sequence, digest, end = 0, 0, GENESIS_DIGEST, 0
        identity = self.store_id
        try:
            with self.path.open("rb") as stream:
                identity, end = _read_header(stream, store_id=self.store_id)
                if identity is None:
                    return JournalStatus(identity, 0, 0, digest, 0, size, False)
                while True:
                    frame = _read_block(stream, valid_end=end)
                    if frame is None:
                        break
                    frame.check_successor(store_id=identity, sequence=sequence+1, previous=digest)
                    count, sequence, digest, end = count+1, frame.sequence, frame.digest, stream.tell()
        except TornJournalError as failure:
            return JournalStatus(identity, count, sequence, digest, failure.valid_end, size, True)
        return JournalStatus(identity, count, sequence, digest, end, size, False)

    def anchor(self):
        """Read only the checked header and final block under publication arbitration."""
        from rexgraph.io.publication import publication_lock
        self._check_path()
        with publication_lock(self.path.parent):
            with self.path.open("rb") as stream:
                identity, header_end = _read_header(stream, store_id=self.store_id)
                if identity is None:
                    raise ValueError("an empty file has no checked journal anchor")
                end = os.fstat(stream.fileno()).st_size
                previous = _previous_at(stream, end, header_end)
                if previous is not None and previous.store_id != identity:
                    raise ValueError("journal tail belongs to another store")
                return JournalAnchor(identity, 0 if previous is None else previous.sequence,
                                     GENESIS_DIGEST if previous is None else previous.digest, end)

    def check_anchor(self, anchor):
        """Validate a saved prefix without treating another format's offset as valid."""
        if not isinstance(anchor, JournalAnchor):
            raise TypeError("journal snapshot requires a declared anchor")
        self._check_path()
        with self.path.open("rb") as stream:
            identity, header_end = _read_header(stream, store_id=self.store_id)
            if identity != anchor.store_id or anchor.byte_offset > os.fstat(stream.fileno()).st_size:
                raise ValueError("journal snapshot anchor differs from its published store or length")
            previous = _previous_at(stream, anchor.byte_offset, header_end)
            actual = (0, GENESIS_DIGEST) if previous is None else (previous.sequence, previous.digest)
            if actual != (anchor.sequence, anchor.digest) or (previous is not None and previous.store_id != identity):
                raise ValueError("journal snapshot anchor differs from its published prefix")

    def append(self, operation, record_id, record=None, *, extra=None, mutation=None, expected=None):
        from rexgraph.io.publication import publication_lock
        if expected is not None and not isinstance(expected, JournalFrame):
            raise TypeError("journal publication requires a declared expected frame")
        self._check_path()
        with publication_lock(self.path.parent):
            self._check_path()
            if not self.path.exists() and self.store_id is None:
                raise ValueError("a new journal requires an explicit store identity")
            with self.path.open("a+b") as stream:
                identity, header_end = _read_header(stream, store_id=self.store_id)
                if identity is None:
                    if self.store_id is None:
                        raise ValueError("a new journal requires an explicit store identity")
                    identity = self.store_id
                stream.seek(0, os.SEEK_END)
                start = stream.tell()
                previous = _previous_at(stream, start, header_end) if start else None
                if previous is not None and previous.store_id != identity:
                    raise ValueError("journal tail belongs to another store")
                frame = JournalFrame(identity, 1 if previous is None else previous.sequence+1,
                                     GENESIS_DIGEST if previous is None else previous.digest,
                                     operation, record_id, None if record is None else pack_value(record.to_dict()),
                                     exact_coordinates(extra), mutation)
                if expected is not None and frame != expected:
                    raise ValueError("journal publication differs from its prepared transaction")
                payload = (_header(identity) if not start else b"")+_block(frame)
                stream.seek(start)
                try:
                    if stream.write(payload) != len(payload):
                        raise OSError("short checked record journal write")
                    stream.flush()
                    os.fsync(stream.fileno())
                    if not start and os.name != "nt":
                        descriptor = os.open(self.path.parent, os.O_RDONLY)
                        try:
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
                except BaseException:
                    try:
                        stream.seek(start)
                        stream.truncate()
                        stream.flush()
                        os.fsync(stream.fileno())
                    except BaseException as rollback:
                        from .core import PublicationUncertainError
                        raise PublicationUncertainError("checked journal publication rollback failed; recovery required") from rollback
                    raise
                return frame

    def publish(self, frame):
        """Publish an engine validated proposal, refusing a changed physical head."""
        if not isinstance(frame, JournalFrame):
            raise TypeError("journal publication requires a declared frame")
        return self.append(frame.operation, frame.record_id, frame.record, extra=frame.extra,
                           mutation=frame.mutation, expected=frame)

    def repair_torn_tail(self):
        """Explicitly remove only an incomplete final block after verifying its prefix."""
        from rexgraph.io.publication import publication_lock
        with publication_lock(self.path.parent):
            status = self.inspect()  # Complete frame corruption still raises.
            if status.torn:
                with self.path.open("r+b") as stream:
                    stream.truncate(status.valid_end)
                    stream.flush()
                    os.fsync(stream.fileno())
            return status
