# rexgraph/io/




## `_compat`: Zarr v2/v3 and HDF5 Compatibility Layer

**File:** `_compat.py` (969 lines)

Foundation module for the IO layer. Provides backend detection (Zarr, HDF5,
scipy), shared type conversion, and storage helpers for arrays, complex
values, sparse CSR matrices, dicts, boolean masks, and strings in both Zarr
and HDF5 formats. Bridges Zarr v2/v3 API differences (create_dataset vs
create_array, numcodecs Blosc to v3 BloscCodec, compressor normalization).

---

### Backend Detection

- `HAS_ZARR`: True if zarr is importable
- `ZARR_V3`: True if zarr major version >= 3
- `HAS_HDF5`: True if h5py is importable
- `HAS_SCIPY`: True if scipy.sparse is importable

---

### Shared Type Conversion

- `to_native(v)` -> Python native

  Converts numpy scalars to Python int/float/bool, arrays to lists.
  NaN and Inf become 0.0.

- `json_default(o)` -> JSON-serializable

  JSON serializer fallback for numpy types. Pass as `json.dumps(obj, default=json_default)`.

- `NumpyJSONEncoder`: `json.JSONEncoder` subclass using `json_default`.

- `as_str(x)` -> str or passthrough

  Decodes bytes/np.bytes_ to str.

---

### Path Utilities

- `ensure_zarr_suffix(path)` -> str: appends .zarr if missing
- `rm_rf(path)`: removes file or directory tree

---

### Zarr Root Group

- `open_root_group(path, mode="r")`: opens a Zarr group (v2 or v3)
- `create_root_group(path, overwrite=True)`: creates a fresh Zarr group

---

### Zarr Compressor

- `default_zarr_compressor()` -> Blosc(zstd, clevel=3, bitshuffle) or None
- `normalize_zarr_compressor(comp)`: accepts None, numcodecs, v3 codec, or string shorthands ("blosc", "zstd", "none")

---

### Zarr Array Creation

- `g_create_array(group, name, data=None, dtype=None, shape=None, **kw)`

  Creates an array inside a Zarr group. Tries create_array() first, then
  create_dataset() for older v2. Normalizes kwargs across v2/v3 (chunks,
  compressor/compressors bridging).

---

### Zarr Complex Arrays

- `g_store_complex(group, name, arr, compressor=None, chunks=True)`

  Stores an ndarray. Complex arrays become a subgroup with real/imag
  datasets plus is_complex/dtype/shape attrs.

- `g_load_complex(group, name)` -> ndarray

  Loads and reassembles complex arrays from real+imag subgroup.

---

### Zarr Sparse CSR

- `g_store_sparse_csr(group, name, matrix, ...)`: stores scipy sparse as data/indices/indptr subgroup
- `g_load_sparse_csr(group, name, dense=False)`: loads sparse, optionally converting to dense

---

### Zarr Dict Storage

- `g_store_dict(group, name, data, ...)`: arrays become datasets, scalars become JSON attrs, nested dicts become sub-subgroups
- `g_load_dict(group, name)` -> dict: recursive reconstruction

---

### Zarr Boolean Masks

- `g_store_bool_masks(group, name, masks, ...)`: stores as uint8 (Zarr v3 bool issues)
- `g_load_bool_masks(group, name)` -> dict of bool arrays

---

### Zarr Ragged Strings

- `write_text_array(group, name, seq, ...)`: stores strings as ragged UTF-8 byte arrays (values + offsets)
- `read_text_array(group, name)` -> list of bytes: reads legacy fixed-width or ragged layout

---

### HDF5 Helpers

- `open_hdf5(path, mode="r")` -> h5py.File
- `h5_store_array(group, name, arr, ...)`: stores ndarray, complex -> subgroup
- `h5_load_array(group, name)` -> ndarray
- `h5_store_complex` / `h5_load_complex`: same as h5_store_array (API symmetry with Zarr)
- `h5_store_sparse_csr` / `h5_load_sparse_csr`: CSR subgroup with data/indices/indptr
- `h5_store_dict` / `h5_load_dict`: recursive dict storage
- `h5_store_bool_masks` / `h5_load_bool_masks`: uint8 boolean mask subgroup
- `h5_store_strings` / `h5_load_strings`: variable-length UTF-8 HDF5 datasets




## `_serialization`: Type-Aware NamedTuple Serialization

**File:** `_serialization.py` (597 lines)

Generic write/read for rexgraph NamedTuple types across any storage backend.
Inspects each field at runtime and dispatches to the correct helper:
arrays as datasets, scalars as attrs, dicts as subgroups with array/JSON
split, tuples and list-of-dicts as JSON, strings as attrs, None fields
recorded separately. Works through a thin StorageAdapter interface that
wraps Zarr groups, HDF5 groups, or .npy directories.

---

### StorageAdapter Interface

Abstract base with methods:

- `put_array(name, arr)` / `get_array(name)` -> NDArray or None
- `put_scalar(name, value)` / `get_scalar(name, default)` -> Any
- `put_json(name, obj)` / `get_json(name, default)` -> Any
- `put_string(name, s)` / `get_string(name, default)` -> str
- `has(name)` -> bool
- `subgroup(name)` -> StorageAdapter

---

### Adapter Implementations

- `ZarrAdapter(group, compressor=None, chunks=True)`: wraps a Zarr group.
  Uses `g_store_complex`/`g_load_complex` from `_compat` for arrays.

- `HDF5Adapter(group, compression="lzf", chunks=True)`: wraps an h5py group.
  Uses `h5_store_complex`/`h5_load_complex` from `_compat`.

- `NpyAdapter(directory)`: wraps a directory of `.npy` files with a
  `_meta.json` sidecar for scalars, strings, and JSON data. Used by the
  `.rex` bundle format.

---

### Field Classification

- `_classify_field(value)` -> str

  Determines storage strategy for a NamedTuple field value. Returns one of:
  "array", "scalar", "string", "dict", "tuple", "list_of_dict", "none".
  Lists/tuples of numeric values are detected and stored as arrays.

---

### NamedTuple Write/Read

- `write_namedtuple(adapter, name, obj)`

  Creates a subgroup and stores each field by its classified type. Records
  the type name for reconstruction and a `_none_fields` list for None values.

- `read_namedtuple(adapter, name, type_class=None)` -> NamedTuple or dict

  Reads fields back from storage. If type_class is given, constructs an
  instance. Otherwise resolves the type from the stored `_type_name` via
  the type registry. Fixes tuple fields that were stored as JSON lists.
  Falls back to a plain dict if the type cannot be resolved.

---

### Result Dict Write/Read

- `write_result_dict(adapter, name, data)`

  For non-NamedTuple dicts (like `analyze()` output). Arrays become datasets,
  nested dicts with arrays use `_write_dict_field`, everything else is
  JSON-encoded.

- `read_result_dict(adapter, name)` -> dict

  Reads arrays by key list and merges with JSON-stored scalar data.

---

### Type Registry

- `_resolve_type(type_name)` -> Type or None

  Lazily populates a registry by importing all NamedTuple types from
  `rexgraph.types` on first call.

- `register_type(cls)`: manually register a NamedTuple class for
  deserialization.




## `arrow_bridge`: Arrow/IPC Bridge

**File:** `arrow_bridge.py` (459 lines)

Zero-copy columnar export of RexGraph data through Apache Arrow, suitable
for interop with Polars, DuckDB, Spark, and any Arrow-compatible tool.
All pyarrow imports are lazy: the module can be imported without pyarrow
installed; ImportError is raised only when a function is called.

Complex arrays are split into `<name>__real` / `<name>__imag` columns.
2D array shapes are stored in Arrow schema metadata under `rex_array_meta`
so round-trip reshape is exact. Arrays of different lengths are padded to
uniform length for valid Arrow tables.

---

### Dict-of-Arrays <-> Arrow Table

- `arrays_to_arrow(arrays, metadata=None)` -> pyarrow.Table

  Converts a dict of NumPy arrays to an Arrow table. Each array becomes
  one column (or two for complex dtype). Shapes, dtypes, and complex flags
  are stored in schema metadata for exact reconstruction.

- `arrow_to_arrays(table)` -> dict of name -> ndarray

  Reconstructs original shapes, dtypes, and complex values from schema
  metadata. Unknown columns (not in metadata) pass through as-is.

---

### RexGraph <-> Arrow

- `rex_to_arrow(rex, include=None)` -> pyarrow.Table

  Exports a RexGraph as an Arrow table. Stores the minimal reconstruction
  data (boundary_ptr, boundary_idx, B2 arrays, w_E) plus graph metadata
  (nV, nE, nF, directed, dimension) in schema metadata. Optional `include`
  adds computed properties as extra columns.

- `arrow_to_rex(table)` -> RexGraph

  Reconstructs a RexGraph from an Arrow table created by `rex_to_arrow`.
  Reads graph metadata from schema metadata and passes arrays to the
  RexGraph constructor.

---

### IPC File I/O

- `write_arrow_ipc(arrays, path, metadata=None)`

  Writes a dict of arrays to an Arrow IPC file.

- `read_arrow_ipc(path)` -> dict of name -> ndarray

  Reads arrays back from an Arrow IPC file.

---

### Streaming Reads

- `read_arrow_batches(path, batch_rows=100_000)` -> Iterator

  Streams an Arrow IPC file as batches of arrays. Each yielded dict
  contains the same array names but fewer rows, for processing large
  exports without loading everything into memory.




## `safetensors_bridge`: ML Transport and Indexed Encryption

Safetensors export preserves the canonical `rex_state` tensors in a single
file and also supports `TemporalRex`, fingerprint/vector corpora, cache arrays,
and caller-owned extra tensors. Plain files retain the native safetensors layout
and remain compatible with earlier RexGraph releases.

Pass `encryption_properties=` to a writer and the matching opaque
`decryption_properties=` to a reader to use the authenticated indexed layout.
Core never accepts raw key bytes, imports an Agent KMS, or implements a cipher.
The caller-owned property exposes `authenticated_encryption = True` plus:

```python
seal(key_id: str, plaintext: bytes, aad: bytes) -> bytes
open(envelope: bytes, aad: bytes) -> bytes
# Optional fast path when key_id came from the authenticated inner manifest:
open_with(key_id: str, envelope: bytes, aad: bytes) -> bytes
```

`seal` must return a self-framing authenticated-encryption envelope. Configure
exact logical tensor names with `ContainerEncryptionConfig`:

```python
from rexgraph.io import (
    ContainerEncryptionConfig,
    SafetensorQuerySession,
    read_safetensor_tensor,
)

config = ContainerEncryptionConfig(
    footer_key="project-footer",
    tensor_keys={
        "grade-1": ["boundary_ptr", "boundary_idx"],
        "grade-2": ["B2_col_ptr", "B2_row_idx", "B2_vals"],
    },
    plaintext_tensors=["public/model_revision"],
)
# properties.configuration = config; properties owns KMS/envelope state
save_safetensors("graph.safetensors", rex, encryption_properties=properties)
rex2 = load_safetensors(
    "graph.safetensors", decryption_properties=properties
)["object"]
rows = read_safetensor_tensor(
    "graph.safetensors", "boundary_idx", index=slice(0, 100),
    decryption_properties=properties,
)
with SafetensorQuerySession(
    "graph.safetensors", decryption_properties=properties
) as query:
    selected = query.select(
        ["cochain/value"],
        where=("cochain/row_id", ">=", 1_000_000),
    )
```

The logical manifest is encrypted with `footer_key` by default. Setting
`plaintext_manifest=True` makes it visible but still authenticates it. Tensors
not explicitly mapped to another key or to the plaintext list fail closed to
`footer_key`; unknown names, overlaps, missing grade keys, incomplete chunk
inventories, and authentication failures are rejected. A random bundle id and
the tensor name, dtype, shape, byte range, chunk index, and total chunk count
are bound into associated data. Explicit plaintext tensors are bound by hashes
inside the authenticated manifest. The default 1 MiB chunks let
`read_safetensor_tensor` open only the requested first-axis region.

Each new member also carries fixed-length min, max, and null-count facts for
every chunk. A protected member's facts are sealed under that member's own key,
not the footer key. A reader that can open the manifest but cannot open a member
therefore learns no value distribution for that member. Statistics associated
with an explicitly public member are plaintext but remain authenticated. NaN
and NaT values count as null; integer and Boolean tensors have no null values.

`SafetensorQuerySession` authenticates the manifest once and reuses opened
statistics and data chunks. `where(name, operator, value)` returns matching row
positions. `select(names, where=(name, operator, value))` first uses statistics
to reject impossible predicate chunks, checks candidate chunks exactly, then
gathers result rows from only their touched chunks. Supported operators are
`==`, `!=`, `<`, `<=`, `>`, `>=`, `isnull`, and `notnull`. Predicates must be
one-dimensional, and every selected tensor must share their first axis. Older
encrypted files without statistics still read and query, but their predicate
tensor is scanned in full. Min/max pruning also cannot help an unclustered
predicate whose range overlaps every chunk.

When a property implements `open_with`, core uses it only for a key identifier
that came from the authenticated inner manifest. This resolves the one member
being read without trying or preloading keys for unrelated members. The outer
descriptor is not yet authenticated when the footer is opened, so it does not
qualify for this keyed fast path.

This protects container confidentiality and integrity, not row-level access,
identity authorization independent of key possession, or rollback. Ciphertext
and member sizes still disclose approximate complex shape and therefore can
suggest `nV`, `nE`, or `nF`; this version does not pad. Use an external trusted
version/digest anchor when rollback detection matters.

## `parquet_bridge`: Parquet Table Export/Import

**File:** `parquet_bridge.py` (915 lines)

Exports the mathematically meaningful structures of a relational complex as
columnar Parquet tables. Each table type maps to a specific part of the
algebraic/topological framework. All pyarrow imports are lazy. No pandas
dependency.

---

### Generic Parquet I/O

- `parquet_encryption_properties(crypto_factory, kms_connection_config, footer_key=..., column_keys=..., plaintext_footer=False)`: builds opaque PyArrow file-encryption properties from caller-owned KMS objects. Key values are identifiers, not raw keys. The encrypted footer is the safe default; a signed plaintext footer is explicit. Column keys protect columns inside one file, not rows; separate Rex grade tables need a caller-owned file/bundle key policy.
- `write_parquet(data, path, metadata=None, encryption_properties=None)`: writes a dict of equal-length 1D arrays; 2D arrays split into `{name}_0`, `{name}_1`, etc. An opaque PyArrow `FileEncryptionProperties` enables Parquet Modular Encryption; `None` preserves plaintext compatibility.
- `read_parquet(path, columns=None, decryption_properties=None)` -> dict: pushes the physical column projection into PyArrow, reassembles requested 2D arrays from split columns, and accepts optional opaque PyArrow `FileDecryptionProperties`.
- `read_parquet_batches(path, batch_rows=100_000, columns=None, decryption_properties=None)` -> Iterator: streaming reads with the same optional decryption properties.

Core does not ship a `WorkspaceKeyring` adapter for PyArrow. The adapter used in
RexGraph's integration measurement is test only. With PyArrow 24, every nonempty
projection of a file using distinct column keys also unwraps the first physical
column key even when that column's pages are not read. DuckDB 1.5.5 cannot open
that distinct-key PyArrow shape; its Arrow interoperability currently requires
one uniform key for the footer and every column. Treat Parquet column projection
as an I/O optimization, not as independent per-column authorization. The native
safetensors and `.rex` paths do not have this first-column coupling.

Every typed table writer/reader accepts the matching opaque encryption/decryption
property as a keyword argument, including character, vertex-character, and void
tables. Core does not accept raw keys or implement a KMS client; callers construct
their client and connection policy outside RexGraph.

---

### Boundary Table

- `write_boundary_table(rex, path)`: one row per (edge, boundary_vertex) pair. Handles all edge types: standard (2 rows), self-loop (2 rows, same vertex), branching (3+ rows), witness (1 row). Columns: edge_idx, vertex_idx, position.
- `read_boundary_table(path)` -> dict: reconstructs boundary_ptr/boundary_idx from the table.

---

### Edge Table

- `write_edge_table(rex, path, include=None)`: one row per edge. Columns: edge_idx, source, target, boundary_size, edge_type, weight. Optional include for Hodge components.
- `read_edge_table(path)` -> dict

---

### Vertex Table

- `write_vertex_table(rex, path, include=None)`: one row per vertex. Columns: vertex_idx, degree, x, y (spectral layout). Optional: fiedler_vector_L0, layout_3d.
- `read_vertex_table(path)` -> dict

---

### Face Table

- `write_face_table(rex, path)`: one row per nonzero in B2 CSC. Columns: face_idx, edge_idx, orientation (+/-1).
- `read_face_table(path)` -> dict: reconstructs B2_col_ptr, B2_row_idx, B2_vals.

---

### Persistence Table

- `write_persistence_table(result, path)`: from persistence diagram dict. Columns: birth, death, dim, birth_cell, death_cell, lifetime. Metadata stores essential pairs and Betti numbers.
- `read_persistence_table(path)` -> dict: includes betti tuple and essential array from metadata.

---

### Filtration Table

- `write_filtration_table(rex, filt_v, filt_e, filt_f, path, kind="")`: one row per cell. Columns: cell_idx, cell_dim, filtration_value.
- `read_filtration_table(path)` -> dict: splits back into filt_v, filt_e, filt_f by dimension.

---

### Metrics Table

- `write_metrics_table(metrics, path, index_name="cell_idx")`: generic per-cell numeric metrics. All arrays must have equal length.
- `read_metrics_table(path, exclude_index=True)` -> dict

---

### Character Tables

- `write_character_table(rex, path)` / `read_character_table(path)`: per-edge structural character chi. Columns: edge_idx, chi_0..chi_{nhats-1}.
- `write_vertex_character_table(rex, path)` / `read_vertex_character_table(path)`: per-vertex phi and kappa.

---

### Void Table

- `write_void_table(rex, path)` / `read_void_table(path)`: per-void triangle data. Columns: void_idx, eta, fills_beta, chi_void_0..chi_void_{nhats-1}. Empty table written when n_voids = 0.




## `sql_bridge`: SQL Database Bridge

**File:** `sql_bridge.py` (1024 lines)

Stores the same table types as `parquet_bridge` into SQL databases (SQLite,
PostgreSQL, or any SQLAlchemy-compatible backend). Each table maps to a
specific part of the algebraic/topological framework. All sqlalchemy and
pandas imports are lazy. Metadata is stored in companion `<table>_meta`
tables as single-row JSON.

Requires: `pip install sqlalchemy pandas`

---

### Engine Management

- `get_engine(conn_str)`: returns a SQLAlchemy engine. In-memory SQLite uses a shared StaticPool so multiple calls see the same database.

---

### Boundary Table

- `write_boundary_sql(rex, conn, table="boundary", if_exists="replace")`: one row per (edge, boundary_vertex) pair. Columns: edge_idx, vertex_idx, position.
- `read_boundary_sql(conn, table="boundary")` -> dict: reconstructs boundary_ptr/boundary_idx.

---

### Edge Table

- `write_edge_sql(rex, conn, table="edges", include=None, if_exists="replace")`: per-edge data with source, target, boundary_size, edge_type, edge_type_name, weight. Optional Hodge components.
- `read_edge_sql(conn, table="edges")` -> dict

---

### Vertex Table

- `write_vertex_sql(rex, conn, table="vertices", include=None, if_exists="replace")`: per-vertex data with degree, x, y (spectral layout). Optional: layout_3d, fiedler_vector_L0.
- `read_vertex_sql(conn, table="vertices")` -> dict

---

### Face Table

- `write_face_sql(rex, conn, table="faces", if_exists="replace")`: B2 operator as one row per nonzero. Columns: face_idx, edge_idx, orientation.
- `read_face_sql(conn, table="faces")` -> dict: reconstructs B2_col_ptr, B2_row_idx, B2_vals.

---

### Persistence Table

- `write_persistence_sql(result, conn, table="persistence", if_exists="replace")`: persistence pairs. Columns: birth, death, dim, birth_cell, death_cell, lifetime. Metadata stores essential pairs and Betti numbers.
- `read_persistence_sql(conn, table="persistence")` -> dict

---

### Filtration Table

- `write_filtration_sql(rex, filt_v, filt_e, filt_f, conn, table="filtration", kind="", if_exists="replace")`: one row per cell. Columns: cell_idx, cell_dim, filtration_value.
- `read_filtration_sql(conn, table="filtration")` -> dict: splits into filt_v, filt_e, filt_f.

---

### Temporal Table

- `write_temporal_sql(trex, conn, table="temporal", if_exists="replace")`: per-timestep Betti numbers, edge/face counts, Euler characteristic from a TemporalRex.
- `read_temporal_sql(conn, table="temporal")` -> dict: includes betti as T x 3 array.

---

### Metrics Table

- `write_metrics_sql(metrics, conn, table="metrics", cell_dim=0, if_exists="replace")`: generic per-cell numeric metrics with cell_dim column.
- `read_metrics_sql(conn, table="metrics", cell_dim=None, exclude_index=True)` -> dict

---

### Character and Void Tables

- `write_character_sql(rex, conn, table="character")` / `read_character_sql(conn, table="character")`: per-edge structural character chi.
- `write_vertex_character_sql(rex, conn, table="vertex_character")` / `read_vertex_character_sql(conn)`: per-vertex phi and kappa.
- `write_void_sql(rex, conn, table="void")` / `read_void_sql(conn, table="void")`: void complex data.

---

### Streaming

- `read_sql_batches(conn, table_or_query, chunksize=100_000)` -> Iterator: streams SQL results as batches of arrays via pandas chunked reads.




## `csv_loader`: CSV Edge List Loader with Column Classification

**File:** `csv_loader.py` (596 lines)

Loads CSV edge lists and automatically classifies each metadata column's
semantic role using a heuristic cascade: column name pattern matching,
value-set statistics (cardinality, average length, delimiter frequency,
numeric fraction), and value content scanning (positive/negative stems,
ordinal terms, identifier patterns). No heavy dependencies: uses only
csv, re, numpy, and collections.

---

### Column Roles

`ColumnRole` defines the taxonomy: TYPE (edge coloring), POLARITY
(flow sign), GROUPING (pathway/cluster), ORDINAL (high/medium/low),
NUMERIC (continuous), EVIDENCE (semicolon-delimited sources), REFERENCE
(PMIDs/accessions), DESCRIPTION (free text), UNKNOWN.

---

### Column Profiling

- `ColumnProfile` dataclass: per-column statistics: name, role, n_values,
  n_unique, avg_length, is_numeric, has_delimiter, unique_ratio, counts,
  numeric stats, positive/negative values for polarity, name_matched flag.
  Properties: is_categorical, is_freetext, is_delimited_list, is_binary.

- `classify_columns(meta)` -> dict of name -> ColumnProfile

  Profiles and classifies all metadata columns. Priority: name pattern match,
  then value heuristics, then statistical fallback.

---

### Weight Construction

- `build_weights(profiles, nE)` -> (w_E, negative_types)

  Constructs signed edge weight vector. Magnitude from first numeric column
  (default 1.0), ordinal columns scale multiplicatively (high=1.0,
  medium=0.6, low=0.3), polarity column applies sign (-|w| for negative
  types).

---

### Edge Attribute Assembly

- `build_edge_attrs(profiles)` -> dict

  Assembles the edge_attrs dict for `analyze()`. Ensures the type column
  is keyed as "type" if one exists.

---

### GraphData

`GraphData` dataclass: fully parsed and classified CSV data: sources,
targets, vertices, src_idx, tgt_idx, meta, profiles, edge_attrs, w_E,
negative_types, nV, nE. Methods:

- `summary()` -> str: human-readable column classification table
- `to_rex()` -> RexGraph: constructs graph with magnitude weights and polarity signs

---

### High-Level Loader

- `load_edge_csv(path, roles=None)` -> GraphData

  Loads a CSV edge list with full column role classification. Auto-detects
  source/target columns by name pattern. Manual role overrides via the
  `roles` dict. Returns GraphData ready for RexGraph construction and
  visualization.

  Supported source column names: source, src, from, head.
  Supported target column names: target, tgt, to, tail, dest.
  Delimiter detection: commas, tabs, pipes, semicolons via csv.Sniffer.




## `json_loader`: JSON Graph Loaders

**File:** `json_loader.py` (512 lines)

Loads graph data from common JSON interchange formats used in bioinformatics,
clinical research, and network science. Auto-detects format from JSON
structure. All loaders produce RexGraph via `from_graph()` or the constructor.
No heavy dependencies beyond json and numpy.

---

### Auto-Detection

- `load_json(path, format=None, threshold=0.0, directed=False)` -> RexGraph

  Loads any supported JSON format. Auto-detects from structure if format is
  None. Detection rules: "boundary_ptr" -> rexgraph native, "elements" ->
  cytoscape, "nodes"+"links" -> networkx, "edges" -> edge_list,
  "matrix"/"adjacency" or list-of-lists -> adjacency.

---

### Format-Specific Loaders

- `load_rexgraph_json(path)` -> RexGraph

  Native format from `RexGraph.to_json()`. Reads boundary_ptr, boundary_idx,
  B2 arrays, w_E. B2_vals reconstructed as ones if not stored.

- `load_edge_list_json(path, **kwargs)` -> RexGraph

  Accepts `{"edges": [{source, target, ...}]}` or a bare list of edge dicts.
  String vertex names mapped to integer indices. Metadata columns classified
  via `csv_loader.classify_columns` for weight/polarity extraction.

- `load_cytoscape_json(path, **kwargs)` -> RexGraph

  Cytoscape.js format: `{"elements": {"nodes": [...], "edges": [...]}}` or
  flat element list with "group" field. Reads weight/score from edge data.
  Negative weights produce edge signs.

- `load_networkx_json(path, **kwargs)` -> RexGraph

  NetworkX node-link format: `{"nodes": [{id}], "links": [{source, target}]}`.
  Reads weight/value from link data.

- `load_adjacency_json(path, threshold=0.0, **kwargs)` -> RexGraph

  Adjacency matrix: `{"matrix": [[...]]}`, `{"adjacency": [...], "labels": [...]}`,
  or bare list-of-lists. Delegates to `RexGraph.from_adjacency()`.

---

### Matrix CSV

- `load_matrix_csv(path, threshold=0.0, absolute=True, directed=False)` -> RexGraph

  Square matrix CSV (correlation matrices, gene expression). Auto-detects
  row/column labels. Threshold filters by |weight|. If absolute=True, uses
  |weight| as edge weight and sign(weight) as edge sign. Extracts edges from
  upper triangle for undirected graphs.




## `zarr_format`: Zarr-Based Storage

**File:** `zarr_format.py` (1284 lines)

Chunked, compressed Zarr stores for RexGraph, TemporalRex, NamedTuples,
and raw arrays. Works with both Zarr v2 and v3. On-disk layout mirrors the
algebraic structure: core reconstruction arrays at root, cache groups
(algebra, spectral, relational, topology, hodge, faces, field, wave, signal,
quotient, persistence, temporal, standard_metrics) as subgroups.

Requires: `pip install zarr`

---

### Simple Array I/O

- `save_zarr_array(arr, path)`: saves a NumPy array to a .zarr store
- `load_zarr_array(path)` -> ndarray

---

### RexZarrFormat Class

`RexZarrFormat(compressor="default", chunks=True, large_threshold=50000)`

Main class with configurable compression and chunking.

- `write(path, obj, cache=None)`: writes RexGraph, TemporalRex, or ndarray. Cache accepts "all" or a list of group names.
- `read(path)` -> RexGraph, TemporalRex, or ndarray: auto-detects object type from attrs.

Container API for multi-object stores:
- `write_to_group(path, name, obj, **kw)`: writes to /objects/\<name\>
- `read_from_group(path, name)` -> object
- `list_groups(path)` -> list of names

---

### Cache Groups

13 cache groups, each containing related computed properties:

- **algebra**: B1, B2, L0, L1, L2, L1_down, L1_up, overlap_adjacency, L_overlap
- **spectral**: spectral_bundle dict, eigenvalues, Fiedler vectors, layout
- **relational**: RL, evals/evecs, alpha constants, Lambda
- **topology**: Betti numbers, Euler characteristic, edge types, cycle basis
- **hodge**: Hodge decomposition components, rho
- **faces**: detected face data and metrics
- **field**: field operator M, eigendecomposition, mode classification
- **wave**: density matrices
- **signal**: perturbation results
- **quotient**: subcomplex masks, quotient operators, relative Betti
- **persistence**: diagrams, enrichment
- **temporal**: edge/face lifecycle, Betti matrix, BIOES
- **standard_metrics**: PageRank, betweenness, clustering, Louvain

---

### Convenience Functions

- `save_zarr(path, obj, cache=None, compressor="default")`: module-level save using default format instance
- `load_zarr(path)` -> object: module-level load




## `hdf5_format`: HDF5-Based Storage

**File:** `hdf5_format.py` (1230 lines)

Single-file counterpart to `zarr_format`. Same serialization surface
(RexGraph, TemporalRex, cache groups, NamedTuples) stored in a single .h5
file via h5py. On-disk layout mirrors the Zarr format with the same 13
cache groups. Compression via HDF5 filters (lzf default, gzip optional).

Requires: `pip install h5py`

---

### Simple Array I/O

- `save_hdf5_array(arr, path)`: saves a NumPy array to an .h5 file
- `load_hdf5_array(path)` -> ndarray

---

### RexHDF5Format Class

`RexHDF5Format(compression="lzf", compression_opts=None, chunks=True, large_threshold=50000)`

Main class with configurable compression and chunking.

- `write(path, obj, cache=None)`: writes RexGraph, TemporalRex, or ndarray to a single .h5 file. Cache accepts "all" or a list of group names.
- `read(path)` -> RexGraph, TemporalRex, or ndarray: auto-detects object type from attrs.

Container API for multi-object files:
- `write_to_group(path, name, obj, **kw)`: writes to /objects/\<name\> within the same .h5 file
- `read_from_group(path, name)` -> object
- `list_groups(path)` -> list of names

---

### Cache Groups

Same 13 cache groups as `zarr_format`: algebra, spectral, relational,
topology, hodge, faces, field, wave, signal, quotient, persistence,
temporal, standard_metrics. Each group writes the same set of computed
properties.

---

### Convenience Functions

- `save_hdf5(path, obj, cache=None, compression="lzf")`: module-level save using default format instance
- `load_hdf5(path)` -> object: module-level load




## `bundle`: RexGraph Bundle (.rex)

**File:** `bundle.py`

Portable relational complex package using only NumPy .npy files and JSON.
No zarr, h5py, scipy, or pyarrow required. A bundle is a self-contained
directory with individual .npy files for exact reconstruction, plus
optional precomputed cache. Memory-mappable for lazy/partial reads.

On-disk layout:

    my_graph.rex/
    +-- MANIFEST.json
    +-- boundary_ptr.npy, boundary_idx.npy
    +-- B2_col_ptr.npy, B2_row_idx.npy, B2_vals.npy
    +-- w_E.npy (if weighted)
    +-- cache/
        +-- layout.npy, eigenvalues_L0.npy, B1.npy, ...

---

### RexBundle Class

`RexBundle(root, manifest)`

- `manifest` -> dict: parsed MANIFEST.json
- `object_type` -> str: "RexGraph" or "TemporalRex"
- `path` -> Path

Construction:
- `RexBundle.from_graph(graph, cache=None)`: creates in-memory bundle spec (does not write to disk). Call `.save()` to persist.
- `RexBundle.load(path, mmap=False, decryption_properties=None)`: loads from a
  .rex directory and authenticates its complete encrypted inventory when present.
  `mmap=True` remains lazy for plaintext arrays.

Persistence:
- `save(path, encryption_properties=None)`: writes through an isolated staging
  directory and serializes publication (threads on every platform and processes
  through `flock` on POSIX), so writers cannot interleave two exports.

`save` treats its destination as an already authorized filesystem path and
creates missing parent directories. A request-facing caller must resolve and
validate that destination inside its bound workspace root before calling core;
bundle publication is not a path authorization boundary.

Reconstruction:
- `to_graph(allow_unsealed=False)` -> RexGraph. The opt-in is only for migrating a
  trusted bundle written before content digests existed; the default refuses an
  unsealed bundle.
- `to_temporal()` -> TemporalRex
- `to_object(allow_unsealed=False)` -> RexGraph or TemporalRex (auto-dispatch)

Array access:
- `bundle["boundary_ptr"]` -> ndarray: loads by name (root or cache/)
- `read_slice(name, index=None)` -> ndarray: decrypts only the selected
  first-axis chunks of a protected member
- `where(name, operator, value=None)` -> int64 ndarray: uses authenticated
  member statistics to prune impossible predicate chunks
- `select(names, where=(name, operator, value))` -> dict: checks a predicate
  exactly and gathers matching rows from named members
- `clear_query_cache()`: forgets opened chunks and statistics retained by bundle
  queries
- `"layout" in bundle` -> bool
- `list_arrays()` -> sorted list of array names
- `read_cache()` -> dict of all cached arrays + scalar cache from manifest

### Authenticated Encrypted Bundles

The same opaque `ContainerEncryptionConfig` and property contract used by the
safetensors bridge applies to `.rex` directories. With
`encryption_properties=None`, the existing named `.npy` layout is unchanged.
With a property, `MANIFEST.json` contains a small public envelope and the logical
manifest is encrypted by `footer_key` unless `plaintext_manifest=True` is
explicit. Generated storage paths do not reveal logical names. Protected members
are authenticated `.rexenc` chunk streams; names in `plaintext_tensors` remain
native `.npy` files whose dtype, shape, bytes, and presence are bound by the
authenticated manifest.

Opening an encrypted bundle authenticates the manifest before using its paths,
then rejects missing, extra, renamed, truncated, altered, or cross-bundle member
files. The inventory covers canonical Rex state, cache arrays, every temporal
snapshot, and every face snapshot. Members omitted from an exact caller policy
use `footer_key`; they never silently become public. `allow_unsealed=True` is a
legacy plaintext migration flag and cannot downgrade this check.

Cold `RexBundle.load` streams SHA-256 over every stored member, so it rejects a
same-length substitution before the member is requested. That eager validation
is O(total ciphertext bytes). After validation, logical-name lookup is O(1), and
`read_slice` decrypts only the selected member chunks. Deferring the hashes would
make initial open cheaper but could not detect an unrequested same-size swap.

Per-chunk query statistics use the same fixed-length format as safetensors. Facts
for a protected member are sealed under that member's key, so opening the footer
does not disclose value ranges for unauthorized grades. `where` and `select`
prune, verify, and gather through the already-open `RexBundle`; a legacy bundle
without statistics falls back to scanning its predicate member.

An explicitly public `.npy` remains memory-mappable. A protected member cannot
honestly be an `np.memmap`; indexed access under `mmap=True` raises with direction
to `read_slice`. The encryption claim remains confidentiality and authenticated
container integrity, not row authorization, identity independent of key
possession, or rollback detection. File/member sizes still disclose approximate
complex shape and may suggest `nV`, `nE`, or `nF`; there is no padding. An older
intact directory can only be detected with an external trusted monotonic anchor.

---

### Cache Groups

Same structure as zarr_format/hdf5_format: algebra, spectral, topology,
hodge. Each group writes specific computed properties as .npy files in the
cache/ subdirectory. Scalar values (Betti numbers, Euler characteristic,
Hodge percentages) are stored in MANIFEST.json under "cache_scalars".

---

### TemporalRex Bundles

Snapshots stored as numbered subdirectories:

    temporal.rex/snapshots/0/sources.npy, targets.npy
    temporal.rex/snapshots/1/...
    temporal.rex/face_snapshots/0/B2_col_ptr.npy, B2_row_idx.npy (optional)

---

### Convenience Functions

- `save_rex(path, obj, cache=None, encryption_properties=None)`: saves RexGraph
  or TemporalRex to a `.rex` bundle
- `load_rex(path, allow_unsealed=False, decryption_properties=None)` -> RexGraph
  or TemporalRex: loads from a
  sealed .rex bundle by default. Set `allow_unsealed=True` only while migrating a
  trusted pre-digest RexGraph bundle; wire and other container readers never expose
  this downgrade.

---

# Provenance and confidentiality

Twelve modules that answer one question in layers: what is this object, who says
so, and what may be learned from it. They compose upward. `manifest` fixes what a
digest is computed over, `security` seals and signs bytes, `transition` and
`commit` describe one change and place it in a lineage, `transport` frames the
result, and `mutation` binds all of it to a pair of endpoints. `catalog` and
`temporal_state` supply the identities those endpoints are stated in.
`partition_state`, `privacy`, `export` and `replication` are what you do with a
complex once it has one.

All are pure Python and import without `cryptography` installed; the AEAD and
signature paths raise only when called. `pyarrow` is likewise lazy in `export`.

**Identity here means the whole object, not its tensors.** The archived design
digested tensors alone, which meant two complexes differing in orientation,
channel semantics, cell metadata or sectioning had the same identity. Every
digest in this stack covers the full canonical header. That is a deliberate
break with the archived format and is why `TemporalState` v1 can never verify.

---

## `manifest`: Canonical Bytes for a Digest

**File:** `manifest.py` (55 lines)

Two objects agree on a digest only if they agree on what was hashed. This is that
agreement: one JSON encoding, sorted keys, no whitespace, no NaN, and a
length-prefixed framing so two different structures cannot serialize alike.

- `canonical_json(value)` -> bytes

  Sorted keys, `(",", ":")` separators, `ensure_ascii=False`, `allow_nan=False`.
  A float that cannot round-trip is an error rather than a token no other reader
  will accept.

- `manifest_digest(value, algorithm="sha256")` -> hex str

  Digest over `canonical_json(value)` behind a domain prefix and a version, so a
  manifest digest can never collide with a bare hash of the same bytes.

- `digest_parts(kind, parts, algorithm="sha256")` -> hex str

  For a sequence of `(name, value)` pairs, each length-prefixed. Used where the
  thing being identified is an ordered set of named digests rather than a document.

---

## `security`: Sealed Bytes and Detached Signatures

**File:** `security.py` (230 lines)

AES-256-GCM with an authenticated header, and Ed25519 signatures. Keys arrive
through a `KeyProvider` protocol and are never serialized: an envelope carries a
key IDENTIFIER, never key material.

- `encrypt_bytes(payload, key_id, keys, object_type)` -> bytes

  `ENVELOPE_MAGIC` + header length + canonical header + ciphertext. The header
  names the object type and key id and is bound as AAD, so a ciphertext cannot be
  relabelled as a different kind of object.

- `decrypt_bytes(blob, keys, max_header=...)` -> bytes
- `envelope_info(blob, max_header=...)` -> `EnvelopeInfo`

  Reads the header without the key, for a reader deciding whether it can open
  something before trying.

- `Ed25519Signer.generate(signer_id)`, `.sign(payload)`, `.verifier()`
- `Ed25519Verifier.verify(payload, signature)` -> bool

  Only the verifier verifies. `Ed25519Signer.verifier()` hands back the matching
  verifier; the signer itself has no verify.

  **Returns a bool rather than raising.** Every call site must branch on it: a
  bare call reads as success. The signer and verifier provide no domain
  separation of their own, so a caller signing more than one kind of record must
  put the kind inside the signed bytes. `TransitionCommit` and `CommitLink` do.

---

## `transition` and `commit`: One Change, and Its Place in a Lineage

**Files:** `transition.py` (67 lines), `commit.py` (55 lines)

Two records, deliberately separate. A `TransitionCommit` says a change happened
between two named states; a `CommitLink` says where that change sits in a chain.
Splitting them means a lineage can be re-signed without re-signing the change.

- `TransitionCommit(previous_state, delta_state, resulting_state, tx_time, actor, policy, signer_id, signature)`

  `previous_state` and `resulting_state` are object identities; `delta_state` is
  the identity of the evidence between them. `policy` is the digest of the policy
  in force, so a package cannot later be verified under a weaker one.

- `CommitLink(transition_digest, parent_digest, signer_id, signature)`

  `parent_digest` is `None` only at genesis, and that is checked rather than
  assumed.

Both embed their own `object_type` in `signing_bytes()`, so a transition
signature does not verify as a lineage signature.

---

## `transport`: A Self-Describing Frame

**File:** `transport.py` (127 lines)

`MAGIC` + header length + canonical header + payload, where the header carries
the object type, the payload size and its SHA-256.

- `pack(payload, object_type, metadata=None)` -> bytes
- `inspect(blob, max_header=...)` -> `TransportInfo`, without validating the payload
- `unpack(blob, verify=True)` -> `(payload, metadata)`

The digest here is a corruption check. It is unkeyed, so it says nothing about
authenticity on its own; that is what the signatures above are for.

---

## `temporal_state`: A TemporalRex as Verifiable Tensors

**File:** `temporal_state.py` (394 lines)

- `to_temporal_state(trex)` -> `TemporalState`
- `verify_temporal_state(state)` -> bool
- `from_temporal_state(state, verify=True, allow_legacy=False)` -> TemporalRex

**v1 is migration-only and can never verify.** Its header digest covered the
tensors alone, leaving `directed`, `general`, `T`, `checkpoint_times`, the
checkpoint threshold and the clock unsigned: an attacker could flip orientation
or rewrite history length, leave every tensor untouched, and every signature
still passed. v2 seals every header field. `verify_temporal_state` returns False
on a version mismatch before any digest work, so a v1 header cannot report
verified whatever its tensors say, and `from_temporal_state` refuses v1 unless
`allow_legacy=True` is passed explicitly. That gate sits outside the `verify`
flag, so `verify=False` does not slip past it.

Reconstruction validates rather than clamps: `T` non-negative, the booleans
actually boolean, checkpoints sorted, unique, in range and starting at zero, and
one finite non-decreasing time per step.

---

## `catalog`: What a File Is, and What an Object Is

**File:** `catalog.py` (484 lines)

- `object_digest(value)` -> hex str, over the full canonical `RexState` header
- `state_object_digest(state)` -> the same for an already-built state
- `FileCatalog(roots)`, `.entries()`, `.info(name)`

A catalog names files by a rooted RELATIVE path (`root0/sub/name`), never an
absolute one, refuses a name that escapes its root, and does not list symlinks.
The point is that a caller can be handed a catalog without being told where the
files are.

`object_digest` covers channel semantics, agent and cell metadata, nested
complexes, sectionings and Merkle state, not only the tensors. Two complexes with
identical tensors and different channels have different identities, which is what
makes an endpoint in a signed transition mean anything.

---

## `mutation`: A Change Bound to Its Endpoints

**File:** `mutation.py` (528 lines)

- `MutationPolicy(require_transition_signature, require_lineage_signature, allowed_signers)`
- `prepare_mutation(previous, resulting, tx_time, actor, policy, parent_digest, ...)` -> `MutationPackage`
- `verify_mutation(package, *, previous, policy=None, verifiers=None, parent_digest=...)` -> bool
- `mutation_to_bytes(package)` / `mutation_from_bytes(blob, allow_legacy=False)`

A v2 package carries ONE full canonical resulting `RexState`. The previous
endpoint is not carried, because in a chain it is the prior version the verifier
already holds; carrying it would pay 110% overhead to ship a second copy of
something on disk, against 55% for the result alone.

**`previous` is a required keyword.** Omitting it raises `TypeError`, and genesis
must pass `None` explicitly. This is structural rather than documentary: a
package verified in isolation can prove the carried result was signed, but cannot
prove the transition, and those two claims must not share a return value. Without
it, a genuine package presented against a store whose prior version is not the one
it was signed against would read as verified. `parent_digest` uses a sentinel, so
omitting the parent check is distinguishable from asserting genesis.

Identity is verified from the CARRIED canonical state, never from a temporal
reconstruction. The checkpoint and delta tuples drop `w_boundary`, graded duals,
signals, labels, agent metadata, cell metadata, nested complexes, sectionings and
Merkle state, so a reconstruction is structural EVIDENCE and not an identity.

A v1 package cannot verify, the writer refuses to emit one, and relabelling a v2
package as v1 fails on the temporal prefix rather than downgrading.

---

## `privacy`: Pseudonyms That Do Not Join

**File:** `privacy.py` (152 lines)

- `PrivacyProjection(fields, pseudonym_fields, scope, key_id)`
- `scoped_pseudonym(value, scope, key_id, keys)` -> base32 token
- `project_rows(rows, projection, keys)` -> projected rows

The same subject identifier under two different scopes produces two unrelated
tokens, so the same person in two studies cannot be joined. Scope and value are
each length-prefixed inside the HMAC message behind a domain prefix, so
`(scope_a, id)` cannot frame identically to `(scope_b, id')`.

**A missing key is a hard failure, not a fallback.** There is no plain-hash path:
one would make every pseudonym recomputable by anyone who can guess a subject id,
which is the entire property this module exists to provide.

Pseudonymisation is not de-identification. A token is stable within its scope,
which is what makes it useful and also what makes it linkable to other records in
that same scope.

---

## `export`: A Parquet Artifact With an Identity

**File:** `export.py` (253 lines)

- `parquet_bytes(data, metadata=None)` -> bytes, in memory, no temporary file
- `export_parquet(data, partition_digest, key_id=None, keys=None)` -> `(payload, ExportManifest)`
- `verify_export(payload, manifest, keys=None)` -> bool

`verify_export` recomputes SHA-256 over the payload it is handed and compares
with `hmac.compare_digest`, rather than trusting a stored digest, so a swapped
payload fails.

**An encrypted export is one AES-GCM envelope over the complete artifact.** It is
not Parquet modular column encryption, promises no per-column isolation, and
supports no projection or predicate pushdown: the whole thing is decrypted before
a single column can be read. That is the right shape for a handoff artifact,
whose recipient reads all of it, and the wrong shape for a working store. The
indexed containers in `safetensors_bridge` and `bundle` are where selective
reading lives.

Logical columns are sorted, so output is deterministic regardless of the order a
caller built the mapping. This diverges deliberately from the archived
caller-order bytes.

**An `ExportManifest` does not name the disclosure policy.** It carries the
partition digest, the schema and payload identity and whether the artifact is
encrypted, but there is no `PrivacyProjection` digest field. So if `project_rows`
was applied after partitioning, the payload's integrity still holds and the
authorising policy is still not named in the artifact. This is inherited from the
archived design rather than introduced here, and it means end-to-end policy
binding has to be asserted out of band until the field exists.

---

## `partition_state`: A Sub-Complex That Is Still a Complex

**File:** `partition_state.py` (273 lines)

- `build_rex_partition(rex, e_mask, *, f_mask=None, grade_masks=None, policy_digest="", closure="subcomplex")` -> `RexPartition`

  `e_mask` is positional and required: a partition of nothing is not a useful
  default. `closure` accepts only `"subcomplex"`, which is the parameter
  reserving room for a rule that is not downward closure, not a choice today.

Selecting cells is not enough: a selection that keeps a face without its edges is
not a complex and `B_k B_{k+1} = 0` will not hold on it. Closure propagates
downward from the top grade, so a selected cell brings its complete boundary at
every grade below it.

Grade one closes from the raw `boundary_ptr`/`boundary_idx` supports rather than
from matrix non-zeros, because a self-loop's signed column cancels to sparse zero
and counting non-zeros would drop a vertex the relation structurally contains.

Accepts the legacy `e_mask`/`f_mask` pair and explicit `grade_masks` for grade two
and above; the archived version rejected grades above two. Orientation, weights,
signs, heads, channels and boundary attribution are preserved by copy, not by
reference, so the partition and its source cannot mutate each other. An empty
grade is preserved as an empty grade rather than dropped, since dropping it
relabels every higher operator. Application metadata is deliberately omitted.
`policy_digest` records the projection that authorised the selection.

---

## `replication`: Applying a Chain Somewhere Else

**File:** `replication.py` (305 lines)

- `pack_replication(checkpoint, mutations, *, checkpoint_state="", checkpoint_commit="")`
  -> `(bytes, ReplicationManifest)`
- `unpack_replication(blob)` -> `(checkpoint_bytes, mutation_bytes, manifest)`

  The middle element is raw mutation BYTES, not decoded packages. Unpacking still
  decodes every mutation internally to validate its type and claimed ordering, then
  returns the original bytes so the caller controls when and where they are applied.
- `apply_replication(blob, *, checkpoint_loader, policy=None, verifiers=None)` -> `AppliedReplication`

`unpack_replication` checks the inventory: digests, counts, and that the lineage
is contiguous. `apply_replication` does the real work, and the division matters.
It matches the loaded checkpoint's identity against what the manifest claims
BEFORE applying anything, then walks the packages in order, calling
`verify_mutation` with the ACTUAL current graph and the exact prior link digest at
each step. Dropping an interior package therefore fails twice over: the lineage is
no longer contiguous, and the next package's `previous_state` no longer matches
what is on disk.

`checkpoint_loader` is a plain callable, which is how this module applies a chain
into a store without importing one.

**The transport chain is unkeyed SHA-256.** By default that is a corruption and
consistency check, not an authenticity boundary: resisting an adversary who
fabricates a whole self-consistent sequence requires the producer to have required
signatures and the verifier to hold real keys. A chain that WAS signed cannot be
downgraded, because the policy digest is bound into the transition.
