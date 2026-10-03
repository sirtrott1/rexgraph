"""Explicit, resumable available history migration from quiescent legacy sources.

This does not reconstruct purged history or pretend legacy layouts have native
change cursors. Every payload crosses the existing copy_record boundary. Plans
pin source bytes and available metadata; resume verifies the actual native
destination prefix rather than trusting a saved counter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sys

from rexgraph.value_codec import pack_value
from .engine import ChangeCursor
from .envelope import _hex_identity
from .legacy import source_fingerprint
from .migration import _Sealed, _check_policy_projection, _cursor, _native, _pages, _policy, _record
from .transfer import CopyReceipt

LEGACY_MIGRATION_LIMITATIONS = (
    "deleted_or_compacted_history_not_reconstructed",
    "original_transaction_order_and_tombstones_not_replayed",
    "destination_transaction_times_and_versions_reallocated",
    "source_commit_packages_not_transplanted",
    "unspecified_valid_start_resolved_at_destination",
    "unbound_legacy_payloads_have_no_original_ownership_proof",
)


@dataclass(frozen=True)
class LegacyRecordClaim:
    record_id: str
    source_version: int
    object_digest: str
    metadata_digest: str

    def __post_init__(self):
        if type(self.record_id) is not str or not self.record_id or len(self.record_id.encode("utf-8")) > 65536:
            raise ValueError("legacy claim requires a bounded literal record id")
        if type(self.source_version) is not int or not 0 < self.source_version < 2**63:
            raise ValueError("legacy claim version requires a positive native int64")
        _hex_identity(self.object_digest, 64, "legacy object digest")
        _hex_identity(self.metadata_digest, 64, "legacy metadata digest")

    def as_record(self):
        return dict(record_id=self.record_id, source_version=self.source_version,
                    object_digest=self.object_digest, metadata_digest=self.metadata_digest)

    @classmethod
    def from_record(cls, value):
        if type(value) is not dict or set(value) != {"record_id", "source_version", "object_digest", "metadata_digest"}:
            raise ValueError("unknown legacy record claim")
        return cls(**value)


@dataclass(frozen=True)
class LegacyMigrationPlan(_Sealed):
    source_store_id: str
    source_backend: str
    source_fingerprint: str
    records: tuple[LegacyRecordClaim, ...]
    destination_start: ChangeCursor
    destination_policy_digest: str
    governed: bool = False
    actor: str = ""
    limitations: tuple[str, ...] = LEGACY_MIGRATION_LIMITATIONS
    _magic = b"RGLM1"
    _domain = b"rexgraph-legacy-migration-plan\x00\x01"
    _limit = 16*1024*1024

    def __post_init__(self):
        _hex_identity(self.source_store_id, 32, "legacy source identity")
        _hex_identity(self.source_fingerprint, 64, "legacy source fingerprint")
        _hex_identity(self.destination_policy_digest, 64, "legacy migration policy digest")
        if type(self.source_backend) is not str or self.source_backend not in {"file", "rex", "object"}:
            raise ValueError("legacy migration supports file, rex and object sources")
        if (type(self.records) is not tuple or any(not isinstance(r, LegacyRecordClaim) for r in self.records)
                or [(r.record_id, r.source_version) for r in self.records] !=
                   sorted({(r.record_id, r.source_version) for r in self.records})):
            raise ValueError("legacy migration claims require unique, sorted source addresses")
        if (not isinstance(self.destination_start, ChangeCursor) or self.destination_start.sequence != 0
                or self.destination_start.store_id == self.source_store_id):
            raise ValueError("legacy migration requires a different, fresh native destination")
        if (type(self.governed) is not bool or type(self.actor) is not str or len(self.actor.encode("utf-8")) > 4096
                or self.actor and not self.governed):
            raise ValueError("invalid legacy migration governance")
        if type(self.limitations) is not tuple or self.limitations != LEGACY_MIGRATION_LIMITATIONS:
            raise ValueError("legacy migration must declare its available-history limitations")

    def as_record(self):
        return dict(format_version=1, source_store_id=self.source_store_id, source_backend=self.source_backend,
                    source_fingerprint=self.source_fingerprint, records=tuple(r.as_record() for r in self.records),
                    destination_start=self.destination_start.as_record(), destination_policy_digest=self.destination_policy_digest,
                    governed=self.governed, actor=self.actor, limitations=self.limitations)

    @classmethod
    def from_record(cls, value):
        _record(value, {"source_store_id", "source_backend", "source_fingerprint", "records", "destination_start",
                        "destination_policy_digest", "governed", "actor", "limitations"})
        if type(value["records"]) is not tuple:
            raise ValueError("legacy migration records require an ordered native tuple")
        return cls(value["source_store_id"], value["source_backend"], value["source_fingerprint"],
                   tuple(LegacyRecordClaim.from_record(r) for r in value["records"]),
                   ChangeCursor.from_record(value["destination_start"]), value["destination_policy_digest"],
                   value["governed"], value["actor"], value["limitations"])


@dataclass(frozen=True)
class LegacyMigrationBatch:
    plan_digest: str
    destination_cursor: ChangeCursor
    copied_versions: int
    receipts: tuple[CopyReceipt, ...]
    complete: bool

    def as_record(self):
        return dict(object_type="LegacyMigrationBatch", format_version=1, plan_digest=self.plan_digest,
                    destination_cursor=self.destination_cursor.as_record(), copied_versions=self.copied_versions,
                    receipts=tuple(r.as_record() for r in self.receipts), complete=self.complete,
                    scope="available_history", limitations=LEGACY_MIGRATION_LIMITATIONS)


def _source(src):
    from .core import FileStore
    from .rexstore import RexStore
    from .objectstore import ObjectStore
    if not isinstance(src, (FileStore, RexStore, ObjectStore)) or not src.read_only:
        raise ValueError("legacy migration requires a read-only FileStore, RexStore or ObjectStore")
    src._check_open()
    digest = src._source_fingerprint() if isinstance(src, ObjectStore) else source_fingerprint(src.root)
    if digest != src._legacy_source_fingerprint:
        raise ValueError("legacy source changed after its read-only snapshot opened; reopen before planning")
    return digest


def _metadata(record):
    return hashlib.sha256(b"rexgraph-legacy-record-metadata\x00"+pack_value(record.to_dict())).hexdigest()


def _selected(src, claim):
    snapshot = src.read_record(claim.record_id, version=claim.source_version)
    if snapshot is None or snapshot.state_digest != claim.object_digest or _metadata(snapshot.record) != claim.metadata_digest:
        raise ValueError("legacy source version differs from its pinned claim")
    return snapshot


def plan_legacy_migration(src, dst, *, actor=""):
    """Verify every available payload before publication and disclose lost semantics.

    The source must be quiescent; fingerprint checks detect changes but cannot
    supply a transaction over an external legacy writer. Old unowned layouts use
    a content derived source identity without modifying the source directory.
    """
    before = _source(src)
    with dst.read_transaction():
        start, policy = _native(dst), _policy(dst)
        governed = bool(getattr(dst, "_require_commits", False))
    records = []
    with src.read_transaction():
        owner = src.store_id
        for record in sorted(src.list(limit=sys.maxsize, include_history=True), key=lambda r: (r.id, r.version)):
            snapshot = src.read_record(record.id, version=record.version)
            if snapshot is None:
                raise ValueError("legacy published metadata has no readable payload")
            with dst.read_transaction():
                _check_policy_projection(dst, snapshot.record)
                if _native(dst) != start or _policy(dst) != policy:
                    raise ValueError("destination changed while planning legacy migration")
            if governed:
                from rexgraph.graph import RexGraph
                if type(snapshot.value) is not RexGraph:
                    raise ValueError("current mutation policy cannot govern legacy temporal records")
            records.append(LegacyRecordClaim(record.id, int(record.version), snapshot.state_digest, _metadata(snapshot.record)))
    if _source(src) != before:
        raise ValueError("legacy source changed while planning migration")
    plan = LegacyMigrationPlan(owner, src.backend, before, tuple(records), start, policy, governed, actor)
    plan.to_bytes()  # Enforce the closed declaration's bound before any write.
    with dst.read_transaction():
        if _native(dst) != start or _policy(dst) != policy:
            raise ValueError("destination changed while planning legacy migration")
    return plan


def _verify_prefix(src, dst, plan, head):
    if head.sequence > len(plan.records):
        raise ValueError("destination extends beyond the available-history plan")
    receipts, versions, governed_ids = [], {}, set()
    for claim, frame in zip(plan.records[:head.sequence], _pages(dst, head), strict=True):
        source = _selected(src, claim).record
        versions[claim.record_id] = versions.get(claim.record_id, 0)+1
        if (frame.operation != "put" or frame.record_id != claim.record_id
                or frame.mutation.version != versions[claim.record_id]
                or frame.record.envelope.object_digest != claim.object_digest
                or frame.record.envelope.codec != "rexgraph.safetensors"
                or pack_value(frame.record.signature) != pack_value(source.signature)
                or pack_value(frame.record.meta) != pack_value(source.meta)
                or frame.record.valid_from != (source.valid_from if source.valid_from is not None else frame.record.tx_from)
                or frame.record.valid_to != source.valid_to
                or (frame.mutation.commit_digest is not None) != plan.governed):
            raise ValueError("destination prefix differs from the available-history plan")
        snapshot = dst.read_record(claim.record_id, version=frame.mutation.version)
        if snapshot is None or snapshot.state_digest != claim.object_digest:
            raise ValueError("destination payload differs from the available-history claim")
        if plan.governed:
            with dst.read_transaction():
                raw = dst._load_commit_bytes(claim.record_id, frame.mutation.version)
                if raw is None or dst._decode_commit(raw).transition.actor != plan.actor:
                    raise ValueError("destination legacy migration commit differs from its actor")
            governed_ids.add(claim.record_id)
        receipts.append(CopyReceipt(plan.source_store_id, claim.record_id, claim.source_version, claim.object_digest,
                                    head.store_id, claim.record_id, frame.mutation.version, snapshot.state_digest))
    for rid in governed_ids:
        if not dst.verify_commits(rid):
            raise ValueError("destination legacy migration commit chain is invalid")
    return receipts


def migrate_legacy_batch(src, dst, plan, *, accept_loss=False, limit=100):
    """Copy available versions, resuming from the verified destination prefix.

    Explicit accept_loss=True acknowledges plan.limitations. Completion means
    this pinned available inventory was copied, not that deleted history was
    recovered. Receipts include earlier verified copies so a restart recovers
    the complete old to new version mapping. Individual writes are durable;
    a later refusal leaves the already published prefix intact.
    """
    from .core import copy_record
    if not isinstance(plan, LegacyMigrationPlan):
        raise TypeError("legacy migration requires a declared LegacyMigrationPlan")
    if type(accept_loss) is not bool or not accept_loss:
        raise ValueError("legacy migration requires explicit accept_loss=True for the declared limitations")
    if type(limit) is not int or not 0 <= limit <= 1000:
        raise ValueError("legacy migration batch limit requires a native integer from 0 to 1000")
    if _source(src) != plan.source_fingerprint or (src.store_id, src.backend) != (plan.source_store_id, plan.source_backend):
        raise ValueError("legacy source identity or bytes differ from the migration plan")
    head = _native(dst)
    if ((head.store_id, head.header_digest) != (plan.destination_start.store_id, plan.destination_start.header_digest)
            or _policy(dst) != plan.destination_policy_digest):
        raise ValueError("legacy destination ownership or policy differs from the migration plan")
    dst.changes(plan.destination_start, limit=0)
    receipts = _verify_prefix(src, dst, plan, head)
    for claim in plan.records[head.sequence:head.sequence+limit]:
        source = _selected(src, claim).record
        receipt = copy_record(src, dst, source, governed=plan.governed, actor=plan.actor,
                              expected_version=dst.next_version(claim.record_id)-1 if plan.governed else None,
                              return_receipt=True, expected_digest=claim.object_digest, expected_cursor=head,
                              expected_policy_digest=plan.destination_policy_digest, preserve_signature=True)
        if receipt is None:
            raise ValueError("pinned legacy source record is missing")
        published = dst.changes(head, limit=1)
        if len(published) != 1 or published[0].operation != "put" or published[0].record_id != claim.record_id:
            raise ValueError("published legacy migration step differs from its claim")
        head = _cursor(published[0], plan.destination_start.header_digest)
        receipts.append(receipt)
    if _source(src) != plan.source_fingerprint:
        raise ValueError("legacy source changed during migration; published prefix requires review")
    return LegacyMigrationBatch(plan.digest, head, head.sequence, tuple(receipts), head.sequence == len(plan.records))
