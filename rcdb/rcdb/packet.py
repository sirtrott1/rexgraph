"""Portable, selected record versions using RCDB's existing native codecs.

A packet is an uncompressed representation of a checked snapshot, not a copy
of its encrypted/compressed storage bytes, journal or mutation package. It
preserves the selected source address, exact metadata and semantic identity.
Its owner is a claim by the sender; authenticate the transport separately.
Receiver publication goes through copy_record and allocates destination clocks
and versions under the destination's policy.
"""
from dataclasses import dataclass
import hashlib
import hmac
import math

from rexgraph._binary import Cursor, uint
from rexgraph.value_codec import pack_value, unpack_value
from .envelope import RecordEnvelope, _RECORD_HEADER_LIMIT, _RECORD_PAYLOAD_LIMIT
from .header import BlobCodecSpec, CodecRef

PACKET_VERSION = 1
PACKET_MAGIC = b"RGRX1"
PACKET_CONTENT_TYPE = "application/vnd.rexgraph.rcdb-record"
RECEIPT_CONTENT_TYPE = "application/vnd.rexgraph.rcdb-copy-receipt"
PACKET_LIMIT = _RECORD_PAYLOAD_LIMIT + 2*_RECORD_HEADER_LIMIT + 128
_DOMAIN = b"rexgraph-record-packet\x00\x01"
_FIELDS = {"id", "signature", "created", "meta", "version", "tx_from",
           "tx_to", "valid_from", "valid_to", "envelope"}


def _metadata(raw):
    from .core import ComplexRecord, _read_selector
    if type(raw) is not bytes or not 0 < len(raw) <= _RECORD_HEADER_LIMIT:
        raise ValueError("record packet metadata exceeds its byte limit")
    data = unpack_value(raw)
    if type(data) is not dict or set(data) != _FIELDS:
        raise ValueError("unknown or incomplete record packet metadata")
    for name in ("created", "tx_from", "tx_to", "valid_from", "valid_to"):
        value = data[name]
        if value is None and name not in {"created", "tx_from"}:
            continue
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("record packet times require finite native binary64 values")
    if data["tx_to"] is not None and data["tx_to"] < data["tx_from"]:
        raise ValueError("record packet transaction interval is reversed")
    if (data["valid_from"] is not None and data["valid_to"] is not None
            and data["valid_to"] < data["valid_from"]):
        raise ValueError("record packet validity interval is reversed")
    _read_selector(data["id"], data["version"], None, None)
    record = ComplexRecord.from_dict(data)
    if record.envelope is None or pack_value(record.to_dict()) != raw:
        raise ValueError("record packet requires canonical bound metadata")
    return record


def _counts(state):
    """Admission counts from checked payload state, never stored analytics."""
    from .codecs import DecodedRecord
    if isinstance(state, DecodedRecord):
        return {}
    if not isinstance(state.header, dict) or not isinstance(state.tensors, dict):
        raise ValueError("invalid state for cell admission")
    if state.header["object_type"] == "RexGraph":
        counts = {name: 0 for name in ("nV", "nE", "nF")}
        headers = [state.header]
        while headers:
            header = headers.pop()
            for name in counts:
                value = header.get(name)
                if type(value) is not int or value < 0:
                    raise ValueError("graph admission requires nonnegative native cell counts")
                counts[name] += value
            nested = header.get("nested", ())
            if not isinstance(nested, (tuple, list)) or any(
                    not isinstance(entry, dict) or not isinstance(entry.get("header"), dict) for entry in nested):
                raise ValueError("invalid nested state for cell admission")
            headers.extend(entry["header"] for entry in nested)
        return counts
    # Temporal admission conservatively counts carried checkpoints and births,
    # plus the highest vertex address, before rebuilding any history. T bounds
    # repeated empty snapshots too. This may exceed any one snapshot's counts.
    total = state.header.get("T")
    if state.header.get("object_type") != "TemporalRex" or type(total) is not int or total < 0:
        raise ValueError("invalid temporal state for cell admission")
    counts = {"nV": 0, "nE": 0, "nF": 0, "T": total}
    for name, tensor in state.tensors.items():
        import numpy as np
        if not isinstance(name, str) or not isinstance(tensor, np.ndarray):
            raise ValueError("invalid temporal tensor for cell admission")
        field = name.rsplit("/", 1)[-1]
        if field in {"boundary_idx", "born_cols", "mod_cols"} and tensor.size:
            counts["nV"] = max(counts["nV"], int(tensor.max())+1)
        if field == "boundary_ptr" or field == "born_offsets" and name.startswith("delta/"):
            counts["nE"] += max(0, tensor.size-1)
        if field == "B2_col_ptr" or field == "born_offsets" and name.startswith("face_delta/"):
            counts["nF"] += max(0, tensor.size-1)
    labels = state.header.get("vertex_labels") or []
    counts["nV"] = max(counts["nV"], max((len(v) for v in labels if v is not None), default=0))
    return counts


@dataclass(frozen=True)
class RecordPacket:
    metadata_bytes: bytes
    blob: bytes

    def __post_init__(self):
        from .engine import metadata_digest
        record = _metadata(self.metadata_bytes)
        envelope, payload = RecordEnvelope.from_bytes(self.blob)
        if envelope.to_bytes(payload) != self.blob:
            raise ValueError("record packet envelope is not canonical")
        if envelope != record.envelope or envelope.metadata_digest != metadata_digest(record):
            raise ValueError("record packet payload differs from its metadata binding")

    @property
    def record(self):
        return _metadata(self.metadata_bytes)

    @property
    def source_store_id(self):
        return self.record.envelope.store_id

    @property
    def state_digest(self):
        return self.record.envelope.object_digest

    @property
    def selection_digest(self):
        """Selected semantic record identity, independent of container ordering."""
        record = self.record
        metadata = record.to_dict()
        metadata.pop("envelope")
        reference = (record.envelope.codec, record.envelope.codec_version,
                     record.object_type, self.state_digest)
        body = (self.source_store_id, metadata, reference)
        return hashlib.sha256(b"rexgraph-selected-record\x00\x01"+pack_value(body)).hexdigest()

    def _body(self):
        return uint(len(self.metadata_bytes))+self.metadata_bytes+self.blob

    @property
    def digest(self):
        return hashlib.sha256(_DOMAIN+self._body()).hexdigest()

    def to_bytes(self):
        body = self._body()
        return PACKET_MAGIC+hashlib.sha256(_DOMAIN+body).digest()+body

    @classmethod
    def from_bytes(cls, raw):
        if type(raw) is not bytes or not 37 < len(raw) <= PACKET_LIMIT:
            raise ValueError("record packet exceeds its byte limit or is not bytes")
        cursor = Cursor(raw)
        if cursor.take(5) != PACKET_MAGIC:
            raise ValueError("unsupported record packet version")
        expected = cursor.take(32)
        if not hmac.compare_digest(hashlib.sha256(_DOMAIN+raw[37:]).digest(), expected):
            raise ValueError("record packet digest mismatch")
        size = cursor.uint()
        if not 0 < size <= _RECORD_HEADER_LIMIT:
            raise ValueError("invalid record packet metadata length")
        metadata = cursor.take(size)
        packet = cls(metadata, cursor.take(cursor.remaining))
        if packet.to_bytes() != raw:
            raise ValueError("record packet is not canonical")
        return packet

    @classmethod
    def from_snapshot(cls, snapshot, *, source_store_id):
        from .core import RecordSnapshot
        from .engine import bind_record, encode_native_payload
        if not isinstance(snapshot, RecordSnapshot):
            raise TypeError("record packet requires a selected RecordSnapshot")
        record = snapshot.record
        if record.envelope is not None:
            from .engine import metadata_digest
            record.envelope.check_address(store_id=source_store_id, record_id=record.id,
                                          record_version=record.version)
            if (snapshot.state_digest != record.envelope.object_digest
                    or metadata_digest(record) != record.envelope.metadata_digest):
                raise ValueError("record snapshot differs from its published binding")
        codec, version = "rexgraph.safetensors", 1
        if not record.is_complex:
            from .codecs import record_codec
            codec, version = record.envelope.codec, record.envelope.codec_version
            provider = record_codec(CodecRef(codec, version))
            payload = provider.encode(snapshot.value)
            provider.open(payload)
            identity = (provider.object_type, provider.identity(payload))
        else:
            from .engine import payload_state
            payload = encode_native_payload(snapshot.value, BlobCodecSpec())
            state = payload_state(payload, compression=BlobCodecSpec())
            identity = (state.header["object_type"], state.header["digest"])
        if identity[1] != snapshot.state_digest:
            raise ValueError("record snapshot changed before packet construction")
        record, blob = bind_record(record, payload, store_id=source_store_id, identity=identity,
                                   compression=BlobCodecSpec(), codec=codec, codec_version=version)
        return cls(pack_value(record.to_dict()), blob)

    def _open(self, *, check_counts=None):
        from .engine import open_record
        return open_record(self.record, self.blob, store_id=self.source_store_id,
                           compression=BlobCodecSpec(), check_counts=check_counts)

    def cell_counts(self):
        counts = {}
        _, state = self._open(check_counts=counts.update)
        from .codecs import DecodedRecord
        return counts if isinstance(state, DecodedRecord) else _counts(state)

    def snapshot(self, *, check_counts=None):
        from .core import RecordSnapshot
        from .engine import rebuild_record
        from .codecs import DecodedRecord
        payload, state = self._open(check_counts=check_counts)
        if check_counts is not None and not isinstance(state, DecodedRecord):
            check_counts(_counts(state))
        return RecordSnapshot(self.record, rebuild_record(payload, state), self.state_digest)

    def source(self, *, check_counts=None):
        """A selected, read only source for copy_record; no publication capability."""
        return _PacketSource(self, check_counts)


class _PacketSource:
    def __init__(self, packet, check_counts):
        self._packet, self._check_counts = packet, check_counts
        self.store_id = packet.source_store_id

    def read_record(self, record_id, *, version=None):
        record = self._packet.record
        if (record_id, version) != (record.id, record.version):
            return None
        return self._packet.snapshot(check_counts=self._check_counts)


def record_packet(store, record_id, *, version=None, as_of=None, valid_at=None):
    """Capture one checked version under the provider's ordinary read contract."""
    snapshot = store.read_record(record_id, version=version, as_of=as_of, valid_at=valid_at)
    return None if snapshot is None else RecordPacket.from_snapshot(snapshot, source_store_id=store.store_id)
