"""
RCDB: a database of relational complexes.

Sits beside rexgraph rather than inside an application, so a store can be installed,
tested and reasoned about on its own. It interoperates with an application through
`configure_hooks`, which injects activity recording, request scoping, metadata privacy
and similarity scoring; with none of them set the store works alone.

The public surface is re-exported here so a caller writes `from rcdb import MemoryStore`
rather than reaching into a submodule, and the submodules stay importable for anything
this list does not carry.
"""
from . import analytics, core, index, objectstore, protected_index, rexstore
from .core import (
    ComplexRecord,
    FileStore,
    MemoryStore,
    RCStore,
    SQLStore,
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
)
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

#: Kept here rather than read back from installed metadata, so a source checkout reports
#: what it is. pyproject.toml has to match; a test enforces it.
__version__ = "1.1.3"

__all__ = [
    "ComplexRecord",
    "FileStore",
    "IndexKeyProvider",
    "IndexPolicy",
    "MemoryStore",
    "ObjectStore",
    "RCStore",
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
