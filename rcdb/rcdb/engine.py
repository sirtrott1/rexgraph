"""Shared record semantics, independent of a backend's physical layout.

All storage adapters use the same ownership and immutable publication metadata
binding. Transaction closure is derived from subsequent publications and therefore
does not alter an earlier payload's envelope.
"""
from __future__ import annotations

import hashlib
import hmac
from bisect import bisect_right
from dataclasses import dataclass, replace
from fractions import Fraction
from heapq import nsmallest
from math import isfinite
from operator import attrgetter

from rexgraph.value_codec import pack_value

from .envelope import RECORD_MAGIC, RecordEnvelope, _hex_identity

COMPLEX_CODEC = "rexgraph.safetensors"
COMPLEX_CODEC_VERSION = 1


def encode_native_payload(value, compression):
    """One semantic state encoding path for providers using the shared engine."""
    from .core import _Prepared, decompress_blob
    from rexgraph.graph import TemporalRex
    from rexgraph.io.safetensors_bridge import state_to_safetensors_bytes, state_from_safetensors_bytes
    if isinstance(value, _Prepared):
        raw = decompress_blob(value.blob)
        state_from_safetensors_bytes(raw)
    else:
        if isinstance(value, TemporalRex):
            from rexgraph.temporal_state import to_temporal_state
            state = to_temporal_state(value)
        else:
            from rexgraph.state import to_state
            state = to_state(value)
        raw = state_to_safetensors_bytes(state)
    return compression.encode(raw)


def metadata_digest(record):
    fields = record.to_dict()
    fields.pop("envelope", None)
    fields.pop("tx_to")  # Derived when a successor becomes visible.
    return hashlib.sha256(b"rexgraph-record-metadata\x00"+pack_value(fields)).hexdigest()


def payload_state(payload, *, compression=None):
    from .core import decompress_blob
    from rexgraph.io.safetensors_bridge import state_from_safetensors_bytes
    raw = decompress_blob(payload) if compression is None else compression.decode(payload)
    state = state_from_safetensors_bytes(raw)
    if state.header["object_type"] == "RexGraph" and state.header.get("format_version") != 10:
        raise ValueError("prepared legacy graph payload requires explicit native-state migration")
    return state


def payload_identity(payload):
    state = payload_state(payload)
    return state.header["object_type"], state.header["digest"]


def _state_identity(state):
    kind = state.header["object_type"]
    return kind, state.header["digest"]


def bind_record(record, payload, *, store_id, identity=None, compression=None,
                codec=COMPLEX_CODEC, codec_version=COMPLEX_CODEC_VERSION):
    if identity is None:
        identity = _state_identity(payload_state(payload, compression=compression))
    kind, digest = identity
    envelope = RecordEnvelope(store_id, record.id, record.version, kind, digest,
                              codec, codec_version,
                              hashlib.sha256(payload).hexdigest(), len(payload), metadata_digest(record))
    return replace(record, envelope=envelope), envelope.to_bytes(payload)


def open_record(record, raw, *, store_id=None, compression=None, check_counts=None):
    """Verify ownership even when the caller skips expensive graph reconstruction checks."""
    if not raw.startswith(RECORD_MAGIC):
        if record.envelope is not None:
            raise ValueError("bound record payload was replaced by an unbound legacy payload")
        return raw, None  # Explicit compatibility for records published without a claim.
    envelope, payload = RecordEnvelope.from_bytes(raw)
    if record.envelope is None:
        raise ValueError("native record payload has no published metadata envelope")
    envelope.check_address(store_id=store_id, record_id=record.id, record_version=record.version)
    if envelope != record.envelope:
        raise ValueError("record payload envelope differs from its published metadata")
    if not hmac.compare_digest(envelope.metadata_digest, metadata_digest(record)):
        raise ValueError("record metadata digest differs from its published payload")
    if (envelope.codec, envelope.codec_version) != (COMPLEX_CODEC, COMPLEX_CODEC_VERSION):
        from .codecs import DecodedRecord, record_codec
        from .header import CodecRef
        from .core import decompress_blob
        codec = record_codec(CodecRef(envelope.codec, envelope.codec_version))
        raw = (decompress_blob(payload, max_output_bytes=codec.max_bytes) if compression is None
               else compression.decode(payload, max_output_bytes=codec.max_bytes))
        if check_counts is not None:
            check_counts(codec.cell_counts(raw))
        value = codec.open(raw)
        if envelope.object_type != codec.object_type or not hmac.compare_digest(envelope.object_digest, codec.identity(raw)):
            raise ValueError("record object identity differs from its declared native codec")
        return payload, DecodedRecord(value)
    state = payload_state(payload, compression=compression)
    kind, digest = _state_identity(state)
    if kind != envelope.object_type or not hmac.compare_digest(digest, envelope.object_digest):
        raise ValueError("record object identity differs from its payload envelope")
    return payload, state


def rebuild_record(payload, state, *, verify=True):
    """Materialize an opened record through the existing Core/codec readers."""
    from .codecs import DecodedRecord
    if isinstance(state, DecodedRecord):
        return state.value
    if state is None:
        from .core import deserialize_complex
        return deserialize_complex(payload, verify=verify)
    if state.header["object_type"] == "TemporalRex":
        from rexgraph.temporal_state import from_temporal_state
        return from_temporal_state(state)
    from rexgraph.state import from_state
    return from_state(state, verify=verify)


@dataclass(frozen=True)
class RecordChange:
    """Engine transition carried inside a checked journal frame.

    A deletion keeps the last allocated version. A subsequent put allocates its
    successor, preserving addresses while current state becomes absent between
    deletion and recreation. Blob and optional commit identities refer to exact
    stored bytes, including an encryption envelope when one is configured.
    """
    header_digest: str
    version: int
    previous_version: int
    tx_time: float
    blob_digest: str | None = None
    commit_digest: str | None = None

    def __post_init__(self):
        _hex_identity(self.header_digest, 64, "change store header digest")
        if (type(self.version) is not int or not 0 < self.version < 2**63
                or type(self.previous_version) is not int or not 0 <= self.previous_version < 2**63):
            raise ValueError("change versions require bounded native integers")
        if type(self.tx_time) is not float or not isfinite(self.tx_time):
            raise ValueError("change transaction time requires a finite native binary64 value")
        for name in ("blob_digest", "commit_digest"):
            value = getattr(self, name)
            if value is not None:
                _hex_identity(value, 64, "change "+name)
        if self.blob_digest is None and self.commit_digest is not None:
            raise ValueError("a deletion cannot publish a mutation artifact")

    def as_record(self):
        return {"change_version": 1, "header_digest": self.header_digest,
                "version": self.version, "previous_version": self.previous_version,
                "tx_time": self.tx_time, "blob_digest": self.blob_digest,
                "commit_digest": self.commit_digest}

    @classmethod
    def from_record(cls, value):
        if (type(value) is not dict or set(value) != {"change_version", "header_digest", "version",
                "previous_version", "tx_time", "blob_digest", "commit_digest"}
                or type(value["change_version"]) is not int or value["change_version"] != 1):
            raise ValueError("unknown engine change declaration")
        return cls(**{k: v for k, v in value.items() if k != "change_version"})


@dataclass(frozen=True)
class ChangeCursor:
    """Logical publication cursor; byte offsets and live record counts are unrelated."""
    store_id: str
    header_digest: str
    sequence: int
    digest: str

    def __post_init__(self):
        _hex_identity(self.store_id, 32, "cursor store identity")
        _hex_identity(self.header_digest, 64, "cursor store header digest")
        _hex_identity(self.digest, 64, "cursor change digest")
        if type(self.sequence) is not int or not 0 <= self.sequence < 2**63:
            raise ValueError("change cursor requires a nonnegative native sequence")
        from .journal import GENESIS_DIGEST
        if self.sequence == 0 and self.digest != GENESIS_DIGEST:
            raise ValueError("initial change cursor requires the genesis digest")

    def as_record(self):
        return {"cursor_version": 1, "store_id": self.store_id, "header_digest": self.header_digest,
                "sequence": self.sequence, "digest": self.digest}

    @classmethod
    def from_record(cls, value):
        if (type(value) is not dict or set(value) != {"cursor_version", "store_id", "header_digest", "sequence", "digest"}
                or type(value["cursor_version"]) is not int or value["cursor_version"] != 1):
            raise ValueError("unknown logical change cursor")
        return cls(**{k: v for k, v in value.items() if k != "cursor_version"})


def _closed_query_value(value, depth=0):
    """Values whose comparison cannot call application equality/iteration code."""
    if depth > 32:
        return False
    if type(value) in (type(None), bool, int, float, str, bytes, Fraction):
        return True
    if type(value) in (list, tuple, set, frozenset):
        return all(_closed_query_value(item, depth+1) for item in value)
    if type(value) is dict:
        return all(_closed_query_value(key, depth+1) and _closed_query_value(item, depth+1)
                   for key, item in value.items())
    return False


def _predicate_terms(family, row):
    """Conservative terms: None keeps unusual metadata in the residual scan.

    These are disposable projections of canonical metadata, never authority for
    a match. Restrict normalization to native values with stable semantics.
    """
    try:
        if family == "record_type":
            value = row.object_type
            return frozenset((value,)) if type(value) is str else None
        if family == "source":
            value = row.signature.get("source")
            if type(value) is str:
                return frozenset((value,))
            return frozenset() if _closed_query_value(value) else None
        if family == "labels":
            labels = row.meta.get("vertex_labels")
            sample = row.signature.get("labels_sample")
            if not _closed_query_value(labels) or not _closed_query_value(sample):
                return None
            values = labels or sample or []
            if not all(type(v) in (type(None), bool, int, float, str, bytes, Fraction) for v in values):
                return None
            return frozenset(str(v).lower() for v in values)
        values = row.signature.get("tags", [])
        if not _closed_query_value(values):
            return None
        terms = frozenset(values)
        return terms if all(type(v) is str for v in terms) else None
    except Exception:
        return None


class _PredicateCandidates:
    """Posting membership with a cheap upper bound for choosing a small seed."""
    def __init__(self, postings, unknown, require_all):
        self.postings, self.unknown, self.require_all = postings, unknown, require_all
        sizes = [len(p) for p in postings]
        self.estimate = len(unknown)+(min(sizes) if require_all else sum(sizes))

    def contains(self, address):
        return (address in self.unknown or
                (all(address in p for p in self.postings) if self.require_all
                 else any(address in p for p in self.postings)))

    def addresses(self):
        yield from self.unknown
        if self.require_all:
            for address in min(self.postings, key=len):
                if address not in self.unknown and self.contains(address):
                    yield address
        else:
            seen = set(self.unknown)
            for posting in self.postings:
                for address in posting:
                    if address not in seen:
                        seen.add(address)
                        yield address


class _PredicateIndex:
    """One lazily built field/snapshot index over literal native addresses."""
    def __init__(self, family):
        self.family, self.postings, self.unknown = family, {}, set()

    def add(self, row):
        address = (row.id, row.version)
        terms = _predicate_terms(self.family, row)
        if terms is None:
            self.unknown.add(address)
        else:
            for term in terms:
                self.postings.setdefault(term, set()).add(address)

    def remove(self, row):
        address = (row.id, row.version)
        terms = _predicate_terms(self.family, row)
        if terms is None:
            self.unknown.discard(address)
        else:
            for term in terms:
                posting = self.postings[term]
                posting.remove(address)
                if not posting:
                    del self.postings[term]

    def candidates(self, terms, require_all):
        return _PredicateCandidates([self.postings.get(term, ()) for term in terms],
                                    self.unknown, require_all)


class StoreState:
    """Shared version, time, deletion and replay rules, independent of physical I/O.

    Publication providers arbitrate transactions. This object validates a proposed
    frame before publication and applies it after durable success. Every read owns
    its metadata; committed frames stay immutable. Historical payloads survive a
    tombstone so exact version and earlier transaction time reads remain meaningful.
    """
    def __init__(self, header):
        from .header import StoreHeader
        if not isinstance(header, StoreHeader):
            raise TypeError("store state requires a declared StoreHeader")
        self.header = header
        self._header_digest = header.digest
        self._rows, self._highwater, self._last_time = {}, {}, {}
        self._tombstones, self._addresses, self._incarnation_start, self._changes = {}, {}, {}, []
        self._cursor = self.initial_cursor()
        self._predicate_indexes = {}
        self._validity_indexes = {}
        self._validity_catalog = None

    @property
    def cursor(self):
        return self._cursor

    def next_version(self, record_id):
        from .core import _read_selector
        _read_selector(record_id, None, None, None)
        value = self._highwater.get(record_id, 0)+1
        if value >= 2**63:
            raise ValueError("record version allocation exhausted")
        return value

    def current(self, record_id):
        rows = self._rows.get(record_id, ())
        return rows[-1].detached() if rows and rows[-1].tx_to is None else None

    def history(self, record_id):
        return [row.detached() for row in self._rows.get(record_id, ())]

    def record(self, record_id, version):
        """Owned metadata at one native address, without cloning its whole history."""
        from .core import _read_selector
        _read_selector(record_id, version, None, None)
        if version is None:
            raise ValueError("record version must be a positive integer")
        rows = self._rows.get(record_id, ())
        if version > len(rows):
            return None
        row = rows[version-1]
        if row.version != version:
            raise ValueError("native record allocation differs from its history address")
        return row.detached()

    def incarnation_history(self, record_id):
        if self.current(record_id) is None:
            return []
        rows = self.history(record_id)
        return rows[self._incarnation_start.get(record_id, 0):]

    @staticmethod
    def _selected_row(rows, as_of, valid_at):
        """Borrow one row internally; only the selected result is detached.

        Engine publication orders transaction starts, including equal ticks and
        deletion gaps. Validity only selection intentionally searches retained
        history, matching RCStore's existing bitemporal contract.
        """
        if not rows:
            return None
        if as_of is None and valid_at is None:
            return rows[-1] if rows[-1].tx_to is None else None
        if as_of is not None:
            position = bisect_right(rows, as_of, key=attrgetter("tx_from"))-1
            if position < 0:
                return None
            row = rows[position]
            if row.tx_to is not None and as_of >= row.tx_to:
                return None
            candidates = (row,)
        else:
            candidates = reversed(rows)
        for row in candidates:
            start = row.valid_from if row.valid_from is not None else row.tx_from
            if valid_at is None or (start <= valid_at and (row.valid_to is None or valid_at < row.valid_to)):
                return row
        return None

    def selected(self, record_id, *, as_of=None, valid_at=None):
        """Select owned literal metadata, with the established display alias fallback."""
        from .core import RCStore, _read_selector
        _read_selector(record_id, None, as_of, valid_at)
        rows = self._rows.get(record_id, ())
        if rows:
            row = self._select_lineage(record_id, rows, as_of, valid_at)
            return None if row is None else row.detached()
        alias = RCStore._split_versioned_id(record_id)
        if alias is None:
            return None
        if as_of is not None or valid_at is not None:
            raise ValueError("version display aliases cannot combine time selectors")
        return self.record(*alias)

    def _select_lineage(self, record_id, rows, as_of, valid_at):
        if as_of is not None or valid_at is None or type(valid_at) not in (int, float, Fraction):
            # Custom Real implementations retain their established comparison
            # behavior; do not turn their reverse scan into ordered map probes.
            return self._selected_row(rows, as_of, valid_at)
        if not rows:
            return None
        index = self._validity_indexes.get(record_id)
        if index is None:
            from ._validity import _ValidityIndex
            index = _ValidityIndex()
            for row in rows:
                index.add(row)
            self._validity_indexes[record_id] = index
        version = index.version_at(valid_at)
        return None if version is None else rows[version-1]

    def _record_rows(self, *, as_of=None, valid_at=None, include_history=False):
        from .core import _read_selector
        _read_selector("selector", None, as_of, valid_at)
        if type(include_history) is not bool:
            raise TypeError("include_history must be a bool")
        if include_history and (as_of is not None or valid_at is not None):
            raise ValueError("history collection cannot combine time selectors")
        if include_history:
            return (row for rows in self._rows.values() for row in rows)
        if as_of is None and type(valid_at) in (int, float, Fraction):
            return self._valid_rows(valid_at)
        return (row for record_id, rows in self._rows.items()
                if (row := self._select_lineage(record_id, rows, as_of, valid_at)) is not None)

    def _valid_rows(self, instant):
        # Generator construction stays inert for zero limit pages and queries
        # served by a narrower metadata posting index.
        if self._validity_catalog is None:
            from ._validity_catalog import _ValidityCatalog
            catalog = _ValidityCatalog()
            for record_id, rows in self._rows.items():
                index = self._validity_indexes.get(record_id)
                if index is None:
                    for row in rows:
                        catalog.add(row)
                else:
                    for start, end in index.coverage():
                        catalog.add_interval(record_id, start, end)
            self._validity_catalog = catalog
        for record_id in self._validity_catalog.ids_at(instant):
            row = self._select_lineage(record_id, self._rows[record_id], None, instant)
            if row is not None:
                yield row

    def select_records(self, *, limit=100, offset=0, as_of=None, valid_at=None,
                       include_history=False, predicate=None):
        """Bounded ordered metadata page; detach only records returned to callers.

        Filtering follows per lineage time selection. Borrowed rows never escape
        the provider's read scope or reach an application callback.
        """
        from .core import _matches, _QUERY_KEYS
        if type(limit) is not int or type(offset) is not int or limit < 0 or offset < 0:
            raise ValueError("native collection bounds require nonnegative native integers")
        predicate = {} if predicate is None else predicate
        if set(predicate)-_QUERY_KEYS:
            raise TypeError("unsupported query keys: "+", ".join(sorted(set(predicate)-_QUERY_KEYS)))
        rows = self._record_rows(as_of=as_of, valid_at=valid_at, include_history=include_history)
        if not limit:
            return []
        if predicate:
            # Rich caller values can run equality/iteration code. Preserve the
            # existing detached candidate boundary for those extensions rather
            # than passing them borrowed compound metadata during comparison.
            if not _closed_query_value(predicate):
                rows = (row.detached() for row in rows)
            else:
                indexed = self._indexed_rows(predicate, as_of, valid_at, include_history)
                if indexed is not None:
                    rows = indexed
            rows = (row for row in rows if _matches(row.signature, predicate, row.meta,
                    is_complex=row.is_complex, record_type=row.object_type))
        page = nsmallest(offset+limit, rows, key=lambda row: (-row.tx_from, row.id, -row.version))
        return [row.detached() for row in page[offset:]]

    def _indexed_rows(self, predicate, as_of, valid_at, include_history):
        history = include_history or as_of is not None or valid_at is not None
        plans = []
        for key, value in predicate.items():
            if value is None:
                continue
            if key in ("record_type", "source") and type(value) is str:
                family, terms, require_all = key, frozenset((value,)), True
            elif key in ("labels_any", "labels_all", "tags_any", "tags_all"):
                family, quantifier = key.split("_")
                try:
                    if family == "labels":
                        if not all(type(v) in (type(None), bool, int, float, str, bytes, Fraction) for v in value):
                            continue
                        terms = frozenset(str(v).lower() for v in value)
                    else:
                        terms = frozenset(value)
                        if not all(type(v) is str for v in terms):
                            continue
                except Exception:
                    continue
                require_all = quantifier == "all"
                if not terms:
                    if require_all:
                        continue  # Empty ALL still needs the ordinary metadata matcher.
                    return iter(())
            else:
                continue
            index_key = (family, history)
            index = self._predicate_indexes.get(index_key)
            if index is None:
                index = _PredicateIndex(family)
                for row in self._record_rows(include_history=history):
                    index.add(row)
                self._predicate_indexes[index_key] = index
            plans.append(index.candidates(terms, require_all))
        if not plans:
            return None
        seed = min(plans, key=lambda plan: plan.estimate)
        def candidates():
            selected_ids = set()
            for address in seed.addresses():
                if not all(plan.contains(address) for plan in plans if plan is not seed):
                    continue
                record_id, version = address
                rows = self._rows[record_id]
                if include_history:
                    yield rows[version-1]
                elif record_id not in selected_ids:
                    selected_ids.add(record_id)
                    row = self._select_lineage(record_id, rows, as_of, valid_at)
                    # Select from the complete lineage, not its matching versions.
                    if row is not None and all(plan.contains((row.id, row.version)) for plan in plans):
                        yield row
        return candidates()

    def records(self, *, as_of=None, valid_at=None, include_history=False):
        return [row.detached() for row in self._record_rows(
            as_of=as_of, valid_at=valid_at, include_history=include_history)]

    def tombstone(self, record_id):
        return self._tombstones.get(record_id)

    def change_for(self, record_id, version):
        return self._addresses.get((record_id, version))

    def cursor_for(self, frame):
        from .journal import JournalFrame
        if not isinstance(frame, JournalFrame):
            raise TypeError("change cursor requires a declared journal frame")
        cursor = ChangeCursor(frame.store_id, self._header_digest, frame.sequence, frame.digest)
        self.check_cursor(cursor)
        return cursor

    def changes(self, after=None, *, limit=100):
        if type(limit) is not int or limit < 0:
            raise ValueError("change limit requires a nonnegative native integer")
        after = self.initial_cursor() if after is None else after
        self.check_cursor(after)
        return tuple(self._changes[after.sequence:after.sequence+limit])

    def initial_cursor(self):
        from .journal import GENESIS_DIGEST
        return ChangeCursor(self.header.identity.id, self._header_digest, 0, GENESIS_DIGEST)

    def check_cursor(self, cursor):
        if not isinstance(cursor, ChangeCursor):
            raise TypeError("change feed requires a declared logical cursor")
        if (cursor.store_id != self.header.identity.id or cursor.header_digest != self._header_digest
                or cursor.sequence > len(self._changes)):
            raise ValueError("change cursor differs from this store's published history")
        expected = self.initial_cursor().digest if not cursor.sequence else self._changes[cursor.sequence-1].digest
        if not hmac.compare_digest(cursor.digest, expected):
            raise ValueError("change cursor differs from its published predecessor")

    def prepare_put(self, record, *, blob_digest, commit_digest=None):
        from .journal import JournalFrame
        current = self.current(record.id)
        mutation = RecordChange(self._header_digest, record.version, 0 if current is None else current.version,
                                record.tx_from, blob_digest, commit_digest)
        cursor = self.cursor
        frame = JournalFrame.put(store_id=cursor.store_id, sequence=cursor.sequence+1,
                                 previous=cursor.digest, record=record, mutation=mutation)
        self.validate(frame)
        return frame

    def prepare_delete(self, record_id, tx_time):
        from .core import _read_selector
        from .journal import JournalFrame
        _read_selector(record_id, None, tx_time, None)
        current = self.current(record_id)
        if current is None:
            return None
        cursor = self.cursor
        mutation = RecordChange(self._header_digest, current.version, current.version, float(tx_time))
        frame = JournalFrame(cursor.store_id, cursor.sequence+1, cursor.digest, "delete", record_id,
                             mutation=mutation)
        self.validate(frame)
        return frame

    def validate(self, frame):
        from .core import _write_interval
        from .journal import JournalFrame
        if not isinstance(frame, JournalFrame):
            raise TypeError("state transition requires a declared journal frame")
        cursor = self.cursor
        frame.check_successor(store_id=cursor.store_id, sequence=cursor.sequence+1, previous=cursor.digest)
        mutation = frame.mutation
        if mutation is None or mutation.header_digest != self._header_digest or frame.extra is not None:
            raise ValueError("journal change does not declare this store engine configuration")
        current = self.current(frame.record_id)
        actual = 0 if current is None else current.version
        if mutation.previous_version != actual:
            raise ValueError("journal change differs from the previous visible record version")
        if mutation.tx_time < self._last_time.get(frame.record_id, float("-inf")):
            raise ValueError("journal transaction time precedes this record's last change")
        if frame.operation == "delete":
            if current is None or mutation.version != current.version or mutation.blob_digest is not None:
                raise ValueError("invalid journal tombstone transition")
            return None
        record = frame.record
        if mutation.version != self.next_version(frame.record_id) or record.version != mutation.version:
            raise ValueError("journal publication does not allocate the next record version")
        if (record.tx_from != mutation.tx_time or type(record.tx_from) is not float
                or record.created != mutation.tx_time or type(record.created) is not float
                or record.tx_to is not None or record.envelope is None or mutation.blob_digest is None):
            raise ValueError("journal publication has inconsistent metadata, time or payload ownership")
        start, end = _write_interval(record.valid_from, record.valid_to)
        if type(record.valid_from) is not float or (record.valid_to is not None and type(record.valid_to) is not float):
            raise ValueError("journal validity coordinates require declared binary64 times")
        if (start, end) != (record.valid_from, record.valid_to):
            raise ValueError("invalid journal validity coordinates")
        self.header.check_record_codec(record.envelope.codec, record.envelope.codec_version)
        if (record.envelope.codec != COMPLEX_CODEC
                and record.signature.get("object_type") != record.envelope.object_type):
            raise ValueError("native record signature differs from its declared object type")
        return record

    def apply(self, frame):
        record = self.validate(frame)  # Refusal leaves state untouched.
        cursor = ChangeCursor(self.header.identity.id, self._header_digest, frame.sequence, frame.digest)
        rows = self._rows.get(frame.record_id)
        if rows and rows[-1].tx_to is None:
            for (family, history), index in self._predicate_indexes.items():
                if not history:
                    index.remove(rows[-1])
            rows[-1].tx_to = frame.mutation.tx_time
        if frame.operation == "put":
            if frame.mutation.previous_version == 0:
                self._incarnation_start[frame.record_id] = len(rows) if rows else 0
            self._rows.setdefault(frame.record_id, []).append(record)
            self._highwater[frame.record_id] = record.version
            self._addresses[(frame.record_id, record.version)] = frame.mutation
            self._tombstones.pop(frame.record_id, None)
            for index in self._predicate_indexes.values():
                index.add(record)
            validity = self._validity_indexes.get(record.id)
            if validity is not None:
                validity.add(record)
            if self._validity_catalog is not None:
                self._validity_catalog.add(record)
        else:
            self._tombstones[frame.record_id] = frame.mutation
        self._last_time[frame.record_id] = frame.mutation.tx_time
        self._changes.append(frame)
        self._cursor = cursor

    @classmethod
    def replay(cls, header, frames):
        state = cls(header)
        for frame in frames:
            state.apply(frame)
        return state
