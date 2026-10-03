"""Pinned, paged native history replay through the existing copy boundary.

Each publication is durable independently. Resume verifies the destination's
actual checked prefix, including payloads, before advancing. Plans do not embed
graphs, import providers, transplant commit packages, or infer legacy tombstones.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac

from rexgraph.value_codec import pack_value, unpack_value
from .engine import ChangeCursor
from .envelope import _hex_identity
from .header import StoreHeader
from .journal import GENESIS_DIGEST
from .transfer import CopyReceipt


class _Sealed:
    _limit = 16*1024

    def _body(self):
        body = pack_value(self.as_record())
        if len(body) > self._limit-37:
            raise ValueError("migration declaration exceeds its byte limit")
        return body

    @property
    def digest(self):
        return hashlib.sha256(self._domain+self._body()).hexdigest()

    def to_bytes(self):
        body = self._body()
        return self._magic+hashlib.sha256(self._domain+body).digest()+body

    @classmethod
    def from_bytes(cls, raw):
        if type(raw) is not bytes or not 37 < len(raw) <= cls._limit or not raw.startswith(cls._magic):
            raise ValueError("invalid migration declaration frame")
        body = raw[37:]
        if not hmac.compare_digest(raw[5:37], hashlib.sha256(cls._domain+body).digest()):
            raise ValueError("migration declaration digest mismatch")
        value = cls.from_record(unpack_value(body))
        if value._body() != body:
            raise ValueError("migration declaration is not canonical")
        return value


def _record(value, fields):
    if (type(value) is not dict or set(value) != fields|{"format_version"}
            or type(value["format_version"]) is not int or value["format_version"] != 1):
        raise ValueError("unknown migration declaration")


@dataclass(frozen=True)
class MigrationPlan(_Sealed):
    source_head: ChangeCursor
    destination_start: ChangeCursor
    destination_policy_digest: str
    governed: bool = False
    actor: str = ""
    _magic = b"RGMN1"
    _domain = b"rexgraph-migration-plan\x00"

    def __post_init__(self):
        if not isinstance(self.source_head, ChangeCursor) or not isinstance(self.destination_start, ChangeCursor):
            raise TypeError("migration plan requires declared logical cursors")
        if self.source_head.store_id == self.destination_start.store_id or self.destination_start.sequence != 0:
            raise ValueError("migration requires a different, fresh native destination")
        _hex_identity(self.destination_policy_digest, 64, "migration policy digest")
        if type(self.governed) is not bool or type(self.actor) is not str or len(self.actor.encode("utf-8")) > 4096:
            raise ValueError("migration governance requires a boolean and bounded actor")
        if self.actor and not self.governed:
            raise ValueError("migration actor requires a governed destination")

    def as_record(self):
        return dict(format_version=1, source_head=self.source_head.as_record(),
                    destination_start=self.destination_start.as_record(),
                    destination_policy_digest=self.destination_policy_digest, governed=self.governed, actor=self.actor)

    @classmethod
    def from_record(cls, value):
        _record(value, {"source_head", "destination_start", "destination_policy_digest", "governed", "actor"})
        return cls(ChangeCursor.from_record(value["source_head"]), ChangeCursor.from_record(value["destination_start"]),
                   value["destination_policy_digest"], value["governed"], value["actor"])


@dataclass(frozen=True)
class MigrationProgress(_Sealed):
    plan_digest: str
    source_cursor: ChangeCursor
    destination_cursor: ChangeCursor
    _magic = b"RGMS1"
    _domain = b"rexgraph-migration-progress\x00"

    def __post_init__(self):
        _hex_identity(self.plan_digest, 64, "migration plan digest")
        if not isinstance(self.source_cursor, ChangeCursor) or not isinstance(self.destination_cursor, ChangeCursor):
            raise TypeError("migration progress requires declared logical cursors")
        if self.source_cursor.sequence != self.destination_cursor.sequence:
            raise ValueError("migration progress cursors describe different publication counts")

    def as_record(self):
        return dict(format_version=1, plan_digest=self.plan_digest, source_cursor=self.source_cursor.as_record(),
                    destination_cursor=self.destination_cursor.as_record())

    @classmethod
    def from_record(cls, value):
        _record(value, {"plan_digest", "source_cursor", "destination_cursor"})
        return cls(value["plan_digest"], ChangeCursor.from_record(value["source_cursor"]),
                   ChangeCursor.from_record(value["destination_cursor"]))


@dataclass(frozen=True)
class MigrationStepReceipt(_Sealed):
    progress: MigrationProgress
    operation: str
    record_id: str
    version: int
    copy: CopyReceipt | None = None
    _magic = b"RGMC1"
    _domain = b"rexgraph-migration-step\x00"
    _limit = 4*1024*1024

    def __post_init__(self):
        if not isinstance(self.progress, MigrationProgress) or self.progress.source_cursor.sequence == 0:
            raise ValueError("migration step requires published progress")
        if type(self.operation) is not str or self.operation not in {"put", "delete"}:
            raise ValueError("unknown migration operation")
        if type(self.record_id) is not str or not self.record_id or type(self.version) is not int or not 0 < self.version < 2**63:
            raise ValueError("migration step requires an exact record address")
        if self.operation == "delete":
            if self.copy is not None:
                raise ValueError("migration delete cannot declare a copied payload")
        elif (not isinstance(self.copy, CopyReceipt)
                or (self.copy.source_store_id, self.copy.source_record_id, self.copy.source_version) !=
                   (self.progress.source_cursor.store_id, self.record_id, self.version)
                or (self.copy.destination_store_id, self.copy.destination_record_id, self.copy.destination_version) !=
                   (self.progress.destination_cursor.store_id, self.record_id, self.version)
                or self.copy.source_digest != self.copy.destination_digest):
            raise ValueError("migration copy receipt differs from its published step")

    def as_record(self):
        return dict(format_version=1, progress=self.progress.as_record(), operation=self.operation,
                    record_id=self.record_id, version=self.version, copy=None if self.copy is None else self.copy.as_record())

    @classmethod
    def from_record(cls, value):
        _record(value, {"progress", "operation", "record_id", "version", "copy"})
        return cls(MigrationProgress.from_record(value["progress"]), value["operation"], value["record_id"],
                   value["version"], None if value["copy"] is None else CopyReceipt.from_record(value["copy"]))


@dataclass(frozen=True)
class MigrationBatch:
    progress: MigrationProgress
    receipts: tuple[MigrationStepReceipt, ...]
    complete: bool


def _genesis(cursor):
    return ChangeCursor(cursor.store_id, cursor.header_digest, 0, GENESIS_DIGEST)


def _native(store):
    if not isinstance(getattr(store, "header", None), StoreHeader):
        raise ValueError("checked migration requires a native record-state store")
    return store.change_cursor


def _policy(store):
    return store.transfer_policy_digest()


def _cursor(frame, header_digest):
    return ChangeCursor(frame.store_id, header_digest, frame.sequence, frame.digest)


def _pages(store, head, after=None, *, through=None):
    after = _genesis(head) if after is None else after
    end = head.sequence if through is None else through
    while after.sequence < end:
        page = store.changes(after, limit=min(100, end-after.sequence))
        if not page:
            raise ValueError("migration source checked prefix is incomplete")
        for frame in page:
            frame.check_successor(store_id=head.store_id, sequence=after.sequence+1, previous=after.digest)
            if frame.mutation is None or frame.mutation.header_digest != head.header_digest:
                raise ValueError("migration requires native checked history")
            after = _cursor(frame, head.header_digest)
            yield frame


def _projection(frame):
    change = frame.mutation
    value = dict(operation=frame.operation, id=frame.record_id, version=change.version,
                 previous_version=change.previous_version, tx_time=change.tx_time)
    if frame.operation == "put":
        record = frame.record
        value.update(created=record.created, tx_from=record.tx_from, valid_from=record.valid_from,
                     valid_to=record.valid_to, signature=record.signature, meta=record.meta,
                     object_type=record.envelope.object_type, object_digest=record.envelope.object_digest,
                     codec=record.envelope.codec, codec_version=record.envelope.codec_version)
    return pack_value(value)


def _check_policy_projection(dst, record):
    from .core import _priv
    if (pack_value(dst._stored_meta(_priv(record.meta))) != pack_value(record.meta)
            or pack_value(dst._stored_signature(record.signature)) != pack_value(record.signature)):
        raise ValueError("destination policy would change the checked source metadata or signature")


def plan_migration(src, dst, *, actor=""):
    """Pin the complete native source prefix and a fresh destination identity.

    The current mutation package contract cannot govern deletes or TemporalRex
    results; required commit destinations refuse such a plan before publication.
    """
    with dst.read_transaction():
        destination_start = _native(dst)
        policy_digest = _policy(dst)
        governed = bool(getattr(dst, "_require_commits", False))
    with src.read_transaction():
        source_head = _native(src)
    plan = MigrationPlan(source_head, destination_start, policy_digest, governed, actor)
    for frame in _pages(src, source_head):
        with dst.read_transaction():
            if _policy(dst) != policy_digest or _native(dst) != destination_start:
                raise ValueError("destination changed while planning migration")
            if frame.operation == "put":
                _check_policy_projection(dst, frame.record)
                dst.header.check_record_codec(frame.record.envelope.codec, frame.record.envelope.codec_version)
                if frame.record.envelope.codec != "rexgraph.safetensors":
                    from .codecs import record_codec
                    from .header import CodecRef
                    record_codec(CodecRef(frame.record.envelope.codec, frame.record.envelope.codec_version))
            if governed and (frame.operation == "delete" or frame.record.envelope.object_type != "RexGraph"):
                raise ValueError("current mutation policy cannot govern checked deletion or temporal replay")
    with dst.read_transaction():
        if _policy(dst) != policy_digest or _native(dst) != destination_start:
            raise ValueError("destination changed while planning migration")
    return plan


def _verify_prefix(src, dst, plan, destination_head):
    """Progress is evidence only after comparing the actual published prefix."""
    if destination_head.sequence > plan.source_head.sequence:
        raise ValueError("destination history extends beyond the pinned migration")
    after = _genesis(plan.source_head)
    sources = _pages(src, plan.source_head, through=destination_head.sequence)
    destinations = _pages(dst, destination_head)
    governed_ids = set()
    for source, destination in zip(sources, destinations, strict=True):
        if _projection(source) != _projection(destination):
            raise ValueError("destination checked prefix differs from the migration plan")
        if destination.operation == "put":
            snapshot = dst.read_record(destination.record_id, version=destination.mutation.version)
            if snapshot is None or snapshot.state_digest != source.record.envelope.object_digest:
                raise ValueError("destination copied payload differs from the checked source identity")
            if (destination.mutation.commit_digest is not None) != plan.governed:
                raise ValueError("destination checked prefix differs from migration governance")
            if plan.governed:
                with dst.read_transaction():
                    raw = dst._load_commit_bytes(destination.record_id, destination.mutation.version)
                    if raw is None or dst._decode_commit(raw).transition.actor != plan.actor:
                        raise ValueError("destination migration commit differs from its declared actor")
                governed_ids.add(destination.record_id)
        after = _cursor(source, plan.source_head.header_digest)
    for record_id in governed_ids:
        if not dst.verify_commits(record_id):
            raise ValueError("destination migration commit chain is invalid")
    return after


def migrate_batch(src, dst, plan, *, progress=None, limit=100):
    """Replay at most ``limit`` publications, including deletes, and return progress.

    Reopening and calling without progress recovers from the actual checked prefix.
    A saved progress frame is validated against both journals. A published prefix
    survives later refusal; existing publication uncertainty rules still apply.
    Source growth beyond the pinned head is excluded. Each record crosses solely
    through copy_record with its original clock and declared signature.
    """
    from .core import VersionConflictError, copy_record
    if not isinstance(plan, MigrationPlan):
        raise TypeError("checked migration requires a declared MigrationPlan")
    if type(limit) is not int or not 0 <= limit <= 1000:
        raise ValueError("migration batch limit requires a native integer from 0 to 1000")
    source_now, destination_now = _native(src), _native(dst)
    if ((source_now.store_id, source_now.header_digest) != (plan.source_head.store_id, plan.source_head.header_digest)
            or (destination_now.store_id, destination_now.header_digest) !=
               (plan.destination_start.store_id, plan.destination_start.header_digest)):
        raise ValueError("migration store or header ownership differs from its plan")
    if _policy(dst) != plan.destination_policy_digest:
        raise ValueError("destination policy differs from the migration plan")
    src.changes(plan.source_head, limit=0)
    dst.changes(plan.destination_start, limit=0)
    if progress is not None:
        if not isinstance(progress, MigrationProgress) or progress.plan_digest != plan.digest:
            raise ValueError("migration progress differs from its plan")
        if (progress.source_cursor.sequence > plan.source_head.sequence
                or progress.destination_cursor.sequence > destination_now.sequence):
            raise ValueError("migration progress is beyond the published prefix")
        src.changes(progress.source_cursor, limit=0)
        dst.changes(progress.destination_cursor, limit=0)
    source_after = _verify_prefix(src, dst, plan, destination_now)
    result = MigrationProgress(plan.digest, source_after, destination_now)
    receipts = []
    end = min(plan.source_head.sequence, source_after.sequence+limit)
    for frame in _pages(src, plan.source_head, source_after, through=end):
        copied = None
        if frame.operation == "put":
            copied = copy_record(src, dst, frame.record, governed=plan.governed, actor=plan.actor,
                                 expected_version=frame.mutation.previous_version if plan.governed else None,
                                 return_receipt=True, tx_time=frame.mutation.tx_time,
                                 expected_digest=frame.record.envelope.object_digest,
                                 expected_cursor=result.destination_cursor,
                                 expected_policy_digest=plan.destination_policy_digest, preserve_signature=True)
            if copied is None:
                raise ValueError("pinned source record payload is missing")
        else:
            with dst.write_scope():
                if dst.change_cursor != result.destination_cursor:
                    raise VersionConflictError("destination cursor changed before migration deletion")
                if _policy(dst) != plan.destination_policy_digest:
                    raise ValueError("destination policy differs from the migration plan")
                if not dst.delete(frame.record_id, tx_time=frame.mutation.tx_time,
                                  expected_version=frame.mutation.previous_version):
                    raise ValueError("destination record is missing for migration deletion")
        published = dst.changes(result.destination_cursor, limit=1)
        if len(published) != 1 or _projection(published[0]) != _projection(frame):
            raise ValueError("published destination change differs from the checked migration step")
        destination_after = _cursor(published[0], plan.destination_start.header_digest)
        source_after = _cursor(frame, plan.source_head.header_digest)
        result = MigrationProgress(plan.digest, source_after, destination_after)
        receipts.append(MigrationStepReceipt(result, frame.operation, frame.record_id, frame.mutation.version, copied))
    return MigrationBatch(result, tuple(receipts), source_after == plan.source_head)
