"""Installed record capabilities, using the shared Registry primitive.

Persisted CodecRefs never import providers. Applications explicitly install higher
package codecs; builtins use closed native codecs already owned by this stack.
"""
from dataclasses import dataclass
import hashlib
from threading import RLock

from rexgraph.registry import Registry
from rexgraph.value_codec import pack_value, unpack_value
from .header import CodecRef

VALUE_CODEC = CodecRef("rexgraph.value", 1)
PROVENANCE_CODEC = CodecRef("rexgraph.provenance", 1)
DECLARATION_CODEC = CodecRef("rexgraph.dataset-declaration", 1)
COPY_RECEIPT_CODEC = CodecRef("rcdb.copy-receipt", 1)
MIGRATION_PLAN_CODEC = CodecRef("rcdb.migration-plan", 1)
MIGRATION_PROGRESS_CODEC = CodecRef("rcdb.migration-progress", 1)
MIGRATION_STEP_CODEC = CodecRef("rcdb.migration-step", 1)
QUERY_RESULT_CODEC = CodecRef("rcql.query-result", 1)
_CODECS = Registry("record codec")
_LOCK = RLock()
_INSTALLED = False


@dataclass(frozen=True)
class RecordCodec:
    reference: CodecRef
    object_type: str
    encode: object
    decode: object
    max_bytes: int = 64*1024*1024
    admission: object = None

    def __post_init__(self):
        if not isinstance(self.reference, CodecRef):
            raise TypeError("record capability requires a declared CodecRef")
        if (type(self.object_type) is not str or not self.object_type
                or len(self.object_type.encode("utf-8")) > 256
                or self.object_type in {"RexGraph", "TemporalRex"}
                or self.reference.name == "rexgraph.safetensors"):
            raise ValueError("generic record codec requires a bounded non-complex object type")
        if not callable(self.encode) or not callable(self.decode):
            raise TypeError("record codec requires installed encode and decode capabilities")
        if type(self.max_bytes) is not int or not 0 < self.max_bytes <= 256*1024*1024:
            raise ValueError("record codec requires a bounded native payload limit")
        if self.admission is not None and not callable(self.admission):
            raise TypeError("record codec admission requires an installed callable")

    def cell_counts(self, raw):
        """Declare work before decoding; required when a socket budget is active."""
        self.identity(raw)
        if self.admission is None:
            raise ValueError("record codec has no declared cell admission capability")
        counts = self.admission(raw)
        if (type(counts) is not dict or any(type(k) is not str or type(v) is not int or v < 0
                                          for k, v in counts.items())):
            raise ValueError("record codec admission requires nonnegative native cell counts")
        return counts

    def identity(self, raw):
        if type(raw) is not bytes or len(raw) > self.max_bytes:
            raise ValueError("record codec payload exceeds its byte limit or is not bytes")
        declaration = (self.reference.name, self.reference.version, self.object_type)
        return hashlib.sha256(b"rexgraph-typed-record\x00\x01"+pack_value(declaration)+raw).hexdigest()

    def prepare(self, value):
        raw = self.encode(value)
        digest = self.identity(raw)
        # Establish canonical bytes before publication. A custom provider still
        # owns its declared semantic contract; byte canonicality cannot prove it.
        decoded = self.open(raw)
        return EncodedRecord(self.reference, self.object_type, raw, digest), decoded

    def open(self, raw):
        self.identity(raw)  # Apply bounds before invoking a provider.
        value = self.decode(raw)
        if self.encode(value) != raw:
            raise ValueError("record codec payload is not canonical")
        return value


@dataclass(frozen=True)
class EncodedRecord:
    reference: CodecRef
    object_type: str
    payload: bytes
    digest: str


@dataclass(frozen=True)
class DecodedRecord:
    value: object


def _key(reference):
    if not isinstance(reference, CodecRef):
        raise TypeError("record capability lookup requires a declared CodecRef")
    return f"{len(reference.name)}:{reference.name}:{reference.version}"


def _provenance_encode(value):
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise TypeError("provenance requires a closed string-keyed native record")
    return pack_value(value)


def _typed_encode(cls, value):
    if not isinstance(value, cls):
        raise TypeError(f"record codec requires {cls.__name__}")
    return value.to_bytes()


def _no_cells(raw):
    # These closed codecs decode values/declarations, never a graph or a reader.
    return {}


def _install_builtins():
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        from rexgraph.io.declaration import DatasetDeclaration
        from .transfer import CopyReceipt
        from .migration import MigrationPlan, MigrationProgress, MigrationStepReceipt
        codecs = [RecordCodec(VALUE_CODEC, "NativeValue", pack_value, unpack_value, admission=_no_cells),
                  RecordCodec(PROVENANCE_CODEC, "ProvenanceRecord", _provenance_encode, unpack_value, admission=_no_cells)]
        for reference, cls in ((DECLARATION_CODEC, DatasetDeclaration), (COPY_RECEIPT_CODEC, CopyReceipt),
                               (MIGRATION_PLAN_CODEC, MigrationPlan), (MIGRATION_PROGRESS_CODEC, MigrationProgress),
                               (MIGRATION_STEP_CODEC, MigrationStepReceipt)):
            codecs.append(RecordCodec(reference, cls.__name__, lambda v, cls=cls: _typed_encode(cls, v), cls.from_bytes, admission=_no_cells))
        for codec in codecs:
            _CODECS.register(_key(codec.reference), codec)
        _INSTALLED = True


def register_record_codec(codec, *, replace=False):
    """Install a trusted process capability; store headers still govern publication."""
    if not isinstance(codec, RecordCodec) or type(replace) is not bool:
        raise TypeError("record registration requires a declared RecordCodec and boolean")
    _install_builtins()
    with _LOCK:
        key = _key(codec.reference)
        existing = _CODECS.get(key)
        if existing is not None and existing != codec and not replace:
            raise ValueError("record codec is already installed; replacement must be explicit")
        _CODECS.register(key, codec)
    return codec.reference


def record_codec(reference):
    _install_builtins()
    with _LOCK:
        value = _CODECS.get(_key(reference))
        if value is None:
            raise ValueError("record codec has no declared installed capability")
        return value


def unregister_record_codec(reference):
    _install_builtins()
    with _LOCK:
        return _CODECS.unregister(_key(reference))


def available_record_codecs():
    _install_builtins()
    with _LOCK:
        return tuple(value.reference for _, value in _CODECS.items())
