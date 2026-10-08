"""
RCDB: a database of relational complexes.

Sits beside rexgraph rather than inside an application, so a store can be installed,
tested and reasoned about on its own. It interoperates with an application through
`configure_hooks`, which injects activity recording, request scoping, metadata privacy
and similarity scoring; with none of them set the store works alone.

The public surface is re exported here so a caller writes `from rcdb import MemoryStore`
rather than reaching into a submodule, and the submodules stay importable for anything
this list does not carry.
"""
from . import analytics, core, index, objectstore, protected_index, rexstore
from .packet import RecordPacket, record_packet
from .codecs import (COPY_RECEIPT_CODEC, DECLARATION_CODEC, MIGRATION_PLAN_CODEC,
                     MIGRATION_PROGRESS_CODEC, MIGRATION_STEP_CODEC, PROVENANCE_CODEC,
                     QUERY_RESULT_CODEC, VALUE_CODEC, RecordCodec, available_record_codecs,
                     register_record_codec, unregister_record_codec)
from .core import (
    ComplexRecord,
    StoredRecord,
    FileStore,
    MemoryStore,
    PublicationUncertainError,
    RCStore,
    RecordSnapshot,
    SQLStore,
    VersionConflictError,
    available_backends,
    cluster_complexes,
    compare,
    compress_blob,
    configure_hooks,
    copy_record,
    decompress_blob,
    default_store,
    default_store_uri,
    deserialize_complex,
    drift,
    find_similar,
    lineage,
    migrate,
    open_store,
    put_version,
    recommend_backend,
    register_backend,
    reset_default_store,
    serialize_complex,
    structural_signature,
    trajectory,
    trend_between,
    unregister_backend,
    version_if_changed,
)
from .objectstore import (
    ObjectStore,
    NativeObjectStore,
    open_object_store,
)
from .object_publication import (ObjectPublication, PublishedObject, LocalObjectPublication,
                                 MemoryObjectPublication)
from .protected_index import (
    IndexKeyProvider,
    IndexPolicy,
    SearchRelation,
    StaticIndexKeyProvider,
    build_search_relation,
    build_search_relation_from_tokens,
    load_search_relation,
    record_token,
    save_search_relation,
    term_token,
    version_record_token,
)
from .rexstore import (
    RexIndex,
    RexStore,
)
from .corpus import CorpusSnapshot
from .legacy_migration import (LEGACY_MIGRATION_LIMITATIONS, LegacyMigrationBatch, LegacyMigrationPlan,
                               LegacyRecordClaim, migrate_legacy_batch, plan_legacy_migration)
from .envelope import RecordEnvelope
from .store_identity import StoreIdentity
from .header import BlobCodecSpec, CodecRef, StoreHeader
from .engine import ChangeCursor, RecordChange, StoreState
from .checkpoint import ReplayCheckpoint, ReplaySegment, ReplaySegmentRef
from .retention import OrphanObject, RetentionPlan, RetentionPolicy
from .localstore import LocalStore
from .transfer import CopyReceipt
from .migration import MigrationBatch, MigrationPlan, MigrationProgress, MigrationStepReceipt, migrate_batch, plan_migration

#: Kept here rather than read back from installed metadata, so a source checkout reports
#: what it is. pyproject.toml has to match; a test enforces it.
__version__ = "1.3.1"

__all__ = [
    "OrphanObject", "RetentionPlan", "RetentionPolicy",
    "ReplayCheckpoint", "ReplaySegment", "ReplaySegmentRef",
    "LEGACY_MIGRATION_LIMITATIONS", "LegacyMigrationBatch", "LegacyMigrationPlan", "LegacyRecordClaim",
    "migrate_legacy_batch", "plan_legacy_migration",
    "StoredRecord", "RecordCodec", "available_record_codecs", "register_record_codec", "unregister_record_codec",
    "VALUE_CODEC", "PROVENANCE_CODEC", "DECLARATION_CODEC", "COPY_RECEIPT_CODEC",
    "MIGRATION_PLAN_CODEC", "MIGRATION_PROGRESS_CODEC", "MIGRATION_STEP_CODEC", "QUERY_RESULT_CODEC",
    "BlobCodecSpec",
    "CodecRef",
    "ChangeCursor",
    "RecordChange",
    "StoreHeader",
    "StoreState",
    "LocalStore",
    "CopyReceipt",
    "MigrationBatch",
    "MigrationPlan",
    "MigrationProgress",
    "MigrationStepReceipt",
    "migrate_batch",
    "plan_migration",
    "CorpusSnapshot",
    "ComplexRecord",
    "FileStore",
    "IndexKeyProvider",
    "IndexPolicy",
    "MemoryStore",
    "PublicationUncertainError",
    "ObjectStore",
    "NativeObjectStore",
    "open_object_store",
    "ObjectPublication",
    "PublishedObject",
    "LocalObjectPublication",
    "MemoryObjectPublication",
    "RCStore",
    "RecordSnapshot",
    "RecordPacket",
    "record_packet",
    "RecordEnvelope",
    "StoreIdentity",
    "VersionConflictError",
    "RexIndex",
    "RexStore",
    "SQLStore",
    "SearchRelation",
    "StaticIndexKeyProvider",
    "analytics",
    "available_backends",
    "build_search_relation",
    "build_search_relation_from_tokens",
    "cluster_complexes",
    "compare",
    "compress_blob",
    "configure_hooks",
    "copy_record",
    "core",
    "decompress_blob",
    "default_store",
    "default_store_uri",
    "deserialize_complex",
    "drift",
    "find_similar",
    "index",
    "lineage",
    "load_search_relation",
    "migrate",
    "objectstore",
    "open_store",
    "protected_index",
    "put_version",
    "recommend_backend",
    "record_token",
    "register_backend",
    "reset_default_store",
    "rexstore",
    "save_search_relation",
    "serialize_complex",
    "structural_signature",
    "term_token",
    "trajectory",
    "trend_between",
    "unregister_backend",
    "version_if_changed",
    "version_record_token",
]
