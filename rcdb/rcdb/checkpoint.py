"""Canonical replay segments: checked frames, never unchecked derived state.

Logical addresses and change cursors survive checkpoint publication. Providers
own publication and physical anchors; the shared engine still validates every
restored transition. Segment references contain no URI or provider credentials.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac

from rexgraph.value_codec import pack_value, unpack_value

from .engine import ChangeCursor, StoreState
from .envelope import _hex_identity
from .header import StoreHeader
from .journal import FRAME_LIMIT, JournalAnchor, JournalFrame

SEGMENT_LIMIT = FRAME_LIMIT+2*1024*1024
CHECKPOINT_LIMIT = 4*1024*1024
_SEGMENT_MAGIC = b"RGRS1"
_CHECKPOINT_MAGIC = b"RGCP1"


def _encode(magic, body, limit):
    if len(body)+37 > limit:
        raise ValueError("replay declaration exceeds its byte limit")
    return magic+hashlib.sha256(magic+body).digest()+body


def _decode(raw, magic, limit):
    if type(raw) is not bytes or not 37 < len(raw) <= limit or not raw.startswith(magic):
        raise ValueError("invalid replay declaration frame")
    if not hmac.compare_digest(raw[5:37], hashlib.sha256(magic+raw[37:]).digest()):
        raise ValueError("replay declaration digest mismatch")
    return unpack_value(raw[37:])


@dataclass(frozen=True)
class ReplaySegmentRef:
    after: ChangeCursor
    end: ChangeCursor
    digest: str
    size: int

    def __post_init__(self):
        if not isinstance(self.after, ChangeCursor) or not isinstance(self.end, ChangeCursor):
            raise TypeError("replay segment requires declared logical cursors")
        if ((self.after.store_id, self.after.header_digest) != (self.end.store_id, self.end.header_digest)
                or not 0 < self.end.sequence-self.after.sequence <= 1024):
            raise ValueError("replay segment has inconsistent ownership or sequence bounds")
        _hex_identity(self.digest, 64, "replay segment digest")
        if type(self.size) is not int or not 37 < self.size <= SEGMENT_LIMIT:
            raise ValueError("replay segment requires a bounded native byte size")

    def as_record(self):
        return {"after": self.after.as_record(), "end": self.end.as_record(),
                "digest": self.digest, "size": self.size}

    @classmethod
    def from_record(cls, record):
        if type(record) is not dict or set(record) != {"after", "end", "digest", "size"}:
            raise ValueError("unknown replay segment reference")
        return cls(ChangeCursor.from_record(record["after"]), ChangeCursor.from_record(record["end"]),
                   record["digest"], record["size"])


@dataclass(frozen=True)
class ReplaySegment:
    after: ChangeCursor
    end: ChangeCursor
    frame_bytes: tuple[bytes, ...]

    def __post_init__(self):
        if not isinstance(self.after, ChangeCursor) or not isinstance(self.end, ChangeCursor):
            raise TypeError("replay segment requires declared logical cursors")
        if (type(self.frame_bytes) is not tuple or not 0 < len(self.frame_bytes) <= 1024
                or any(type(raw) is not bytes or not 0 < len(raw) <= FRAME_LIMIT+64 for raw in self.frame_bytes)
                or sum(map(len, self.frame_bytes)) > SEGMENT_LIMIT-4096
                or self.end.sequence-self.after.sequence != len(self.frame_bytes)
                or (self.after.store_id, self.after.header_digest) != (self.end.store_id, self.end.header_digest)):
            raise ValueError("replay segment has inconsistent frames, ownership or bounds")

    def to_bytes(self):
        return _encode(_SEGMENT_MAGIC, pack_value({"segment_version": 1,
            "after": self.after.as_record(), "end": self.end.as_record(),
            "frames": self.frame_bytes}), SEGMENT_LIMIT)

    @property
    def reference(self):
        raw = self.to_bytes()
        return ReplaySegmentRef(self.after, self.end, hashlib.sha256(raw).hexdigest(), len(raw))

    @classmethod
    def from_bytes(cls, raw, *, reference=None):
        if reference is not None:
            if not isinstance(reference, ReplaySegmentRef):
                raise TypeError("replay read requires a declared segment reference")
            if (type(raw) is not bytes or len(raw) != reference.size
                    or hashlib.sha256(raw).hexdigest() != reference.digest):
                raise ValueError("replay segment differs from its published content address")
        record = _decode(raw, _SEGMENT_MAGIC, SEGMENT_LIMIT)
        if (type(record) is not dict or set(record) != {"segment_version", "after", "end", "frames"}
                or type(record["segment_version"]) is not int or record["segment_version"] != 1):
            raise ValueError("unknown replay segment declaration")
        result = cls(ChangeCursor.from_record(record["after"]), ChangeCursor.from_record(record["end"]),
                     record["frames"])
        if result.to_bytes() != raw or (reference is not None and result.reference != reference):
            raise ValueError("noncanonical or inconsistent replay segment")
        return result

    def frames(self):
        cursor = self.after
        for raw in self.frame_bytes:
            frame = JournalFrame.from_bytes(raw)
            frame.check_successor(store_id=cursor.store_id, sequence=cursor.sequence+1, previous=cursor.digest)
            if frame.mutation is None or frame.mutation.header_digest != cursor.header_digest or frame.extra is not None:
                raise ValueError("replay segment differs from its declared record engine")
            cursor = ChangeCursor(frame.store_id, cursor.header_digest, frame.sequence, frame.digest)
            yield frame
        if cursor != self.end:
            raise ValueError("replay segment differs from its declared terminal cursor")


@dataclass(frozen=True)
class ReplayCheckpoint:
    header: StoreHeader
    segments: tuple[ReplaySegmentRef, ...]

    def __post_init__(self):
        if not isinstance(self.header, StoreHeader) or type(self.segments) is not tuple:
            raise TypeError("replay checkpoint requires a header and immutable segment references")
        if len(self.segments) > 4096:
            raise ValueError("replay checkpoint exceeds its segment limit")
        cursor = StoreState(self.header).cursor
        for reference in self.segments:
            if not isinstance(reference, ReplaySegmentRef) or reference.after != cursor:
                raise ValueError("replay checkpoint segment chain differs from its declared prefix")
            cursor = reference.end

    @property
    def cursor(self):
        return self.segments[-1].end if self.segments else StoreState(self.header).cursor

    def to_bytes(self):
        return _encode(_CHECKPOINT_MAGIC, pack_value({"checkpoint_version": 1,
            "header_bytes": self.header.to_bytes(),
            "segments": tuple(reference.as_record() for reference in self.segments)}), CHECKPOINT_LIMIT)

    @classmethod
    def from_bytes(cls, raw):
        record = _decode(raw, _CHECKPOINT_MAGIC, CHECKPOINT_LIMIT)
        if (type(record) is not dict or set(record) != {"checkpoint_version", "header_bytes", "segments"}
                or type(record["checkpoint_version"]) is not int or record["checkpoint_version"] != 1
                or type(record["segments"]) is not tuple or len(record["segments"]) > 4096):
            raise ValueError("unknown replay checkpoint declaration")
        result = cls(StoreHeader.from_bytes(record["header_bytes"]),
                     tuple(ReplaySegmentRef.from_record(value) for value in record["segments"]))
        if result.to_bytes() != raw:
            raise ValueError("replay checkpoint is not canonical")
        return result

    def restore(self, read):
        state = StoreState(self.header)
        for reference in self.segments:
            segment = ReplaySegment.from_bytes(read(reference), reference=reference)
            for frame in segment.frames():
                state.apply(frame)
        if state.cursor != self.cursor:
            raise ValueError("replay checkpoint differs from its restored state")
        return state


@dataclass(frozen=True)
class LocalReplayCheckpoint:
    """Bind the optional replay cache to exact authoritative journal prefix bytes."""
    checkpoint: ReplayCheckpoint
    anchor: JournalAnchor
    prefix_digest: str

    def __post_init__(self):
        if not isinstance(self.checkpoint, ReplayCheckpoint) or not isinstance(self.anchor, JournalAnchor):
            raise TypeError("local replay cache requires a checkpoint and physical journal anchor")
        cursor = self.checkpoint.cursor
        if (self.checkpoint.header.identity.backend != "local" or
                (cursor.store_id, cursor.sequence, cursor.digest) !=
                (self.anchor.store_id, self.anchor.sequence, self.anchor.digest)):
            raise ValueError("local replay cache differs from its authoritative journal anchor")
        _hex_identity(self.prefix_digest, 64, "local replay prefix digest")

    def to_bytes(self):
        return _encode(b"RGLC1", pack_value({"local_checkpoint_version": 1,
            "checkpoint_bytes": self.checkpoint.to_bytes(), "anchor": self.anchor.as_record(),
            "prefix_digest": self.prefix_digest}), CHECKPOINT_LIMIT+4096)

    @classmethod
    def from_bytes(cls, raw):
        record = _decode(raw, b"RGLC1", CHECKPOINT_LIMIT+4096)
        if (type(record) is not dict or set(record) != {"local_checkpoint_version", "checkpoint_bytes", "anchor", "prefix_digest"}
                or type(record["local_checkpoint_version"]) is not int or record["local_checkpoint_version"] != 1):
            raise ValueError("unknown local replay cache declaration")
        result = cls(ReplayCheckpoint.from_bytes(record["checkpoint_bytes"]),
                     JournalAnchor.from_record(record["anchor"]), record["prefix_digest"])
        if result.to_bytes() != raw:
            raise ValueError("local replay cache is not canonical")
        return result


def build_checkpoint(state, write, *, previous=None, max_frames=256, target_bytes=8*1024*1024):
    """Publish immutable suffix segments; the provider publishes the final descriptor.

    Existing published segment references remain in every later checkpoint, so
    readers of older checkpoint heads keep their complete prefix. A failed build
    leaves only inert staging; it does not change the engine cursor or history.
    """
    if not isinstance(state, StoreState):
        raise TypeError("checkpoint construction requires canonical store state")
    if type(max_frames) is not int or not 0 < max_frames <= 1024:
        raise ValueError("checkpoint frame target requires an integer from 1 to 1024")
    if type(target_bytes) is not int or not 4096 <= target_bytes <= SEGMENT_LIMIT-4096:
        raise ValueError("checkpoint byte target exceeds its declared bounds")
    if previous is not None:
        if not isinstance(previous, ReplayCheckpoint) or previous.header != state.header:
            raise ValueError("previous checkpoint differs from the canonical store header")
        state.check_cursor(previous.cursor)
    references = list(previous.segments) if previous is not None else []
    after = previous.cursor if previous is not None else state.initial_cursor()
    cursor, chunk, size = after, [], 0
    def publish():
        segment = ReplaySegment(after, cursor, tuple(chunk))
        raw, reference = segment.to_bytes(), segment.reference
        write(reference, raw)
        references.append(reference)
        # Refuse an oversized descriptor before it can become authoritative.
        ReplayCheckpoint(state.header, tuple(references)).to_bytes()
    for frame in state.changes(after, limit=2**63-1):
        raw = frame.to_bytes()
        if chunk and (len(chunk) >= max_frames or size+len(raw) > target_bytes):
            publish()
            after, chunk, size = cursor, [], 0
        frame.check_successor(store_id=cursor.store_id, sequence=cursor.sequence+1, previous=cursor.digest)
        cursor = ChangeCursor(frame.store_id, cursor.header_digest, frame.sequence, frame.digest)
        chunk.append(raw); size += len(raw)
    if chunk: publish()
    result = ReplayCheckpoint(state.header, tuple(references))
    if result.cursor != state.cursor:
        raise ValueError("checkpoint construction differs from the canonical store cursor")
    result.to_bytes()
    return result
