# RexGraph I/O API reference

See the [README](README.md) for installation and usage.

The I/O package reads declared records, stores native complexes and exports
columnar data. Native state preserves exact coefficients, identities and carried
components. Table exports have separate schemas from full graph snapshots.

## Installation and usage

`make install` includes the I/O dependencies. For an existing Core environment,
run `python -m pip install ".[io]"` from the repository root. Native RCBD bundles
need only the base Core dependencies. Encryption and signing use the `security`
extra.

```python
from rexgraph import RexGraph
from rexgraph.io import save_rcbd, load_rcbd

rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
save_rcbd("cycle.rcbd", rex)
restored = load_rcbd("cycle.rcbd")
assert tuple(restored.betti) == (1, 1, 0)
```

See the [installation guide](../../README.md#installation) for setup.

## Current native contracts

Native graph state is version 10 and is owned by Core's component codecs.
Container layout, SQL projection and query result transport reuse that state;
they do not select another mathematical representation. Exact columns preserve
integer and rational kinds, absence, identity, declared heads and shares.

HDF5 and Zarr writers stage complete output beside the destination before
publication. Appending a result or graph group verifies the old container,
copies it under the publication lock, writes and seals the copy, then publishes.
An encoding failure preserves the previous output. Directory replacement uses
locked renames with rollback; it is not a crash atomic filesystem exchange.

HDF5 and Zarr writers use container format version 3.0.0. Their content seal covers
groups, arrays and attributes, including derived caches and appended results.
Native graph identity remains its semantic state digest. A content digest
detects corruption; it does not authenticate an author. Legacy native graphs
remain readable. Consuming an unsealed legacy cache requires the explicit
`read_cache(path, allow_unsealed=True)` compatibility option.

`write_boundary_sql` publishes the declared native state and its boundary
projection in one SQL transaction. `reconstruct_from_sql` checks their agreement
before returning the complete graph, including isolated vertices and carried
state. Legacy tables without native state retain their older reconstruction
contract. Exact metric columns and batch reads use the same column schema as
the Arrow/Parquet adapters. SQLite supports rollback and caller owned
transactions; transaction behavior on other dialects depends on the driver.

RCQL's portable result codec is the shared query cache and System download
contract. See [RCQL](../../rcql/README.md#portable-exact-results).

## `_compat`: Zarr v2/v3 and HDF5 compatibility layer

File: `_compat.py`

Provides backend detection and shared storage helpers for arrays, complex
values, sparse CSR matrices, dictionaries, boolean masks and strings. It adapts
array creation and compression options to the Zarr v2 and v3 APIs.

### Backend detection

- `HAS_ZARR`: True if zarr is importable
- `ZARR_V3`: True if zarr major version >= 3
- `HAS_HDF5`: True if h5py is importable
- `HAS_SCIPY`: True if scipy.sparse is importable

### Shared type conversion

- `to_native(v)` -> Python native

  Converts numpy scalars to Python int/float/bool, arrays to lists.
  NaN and Inf become 0.0.

- `json_default(o)` -> JSON serializable

  JSON serializer fallback for numpy types. Pass as `json.dumps(obj, default=json_default)`.

- `NumpyJSONEncoder`: `json.JSONEncoder` subclass using `json_default`.

- `as_str(x)` -> str or passthrough

  Decodes bytes/np.bytes_ to str.

### Path utilities

- `ensure_zarr_suffix(path)` -> str: appends .zarr if missing
- `rm_rf(path)`: removes file or directory tree

### Zarr root group

- `open_root_group(path, mode="r")`: opens a Zarr group (v2 or v3)
- `create_root_group(path, overwrite=True)`: creates a fresh Zarr group

### Zarr compressor

- `default_zarr_compressor()` -> Blosc(zstd, clevel=3, bitshuffle) or None
- `normalize_zarr_compressor(comp)`: accepts None, numcodecs, v3 codec, or string shorthands ("blosc", "zstd", "none")

### Zarr array creation

- `g_create_array(group, name, data=None, dtype=None, shape=None, **kw)`

  Creates an array inside a Zarr group. Tries create_array() first, then
  create_dataset() for older v2. Normalizes kwargs across v2/v3 (chunks,
  compressor/compressors bridging).

### Zarr complex arrays

- `g_store_complex(group, name, arr, compressor=None, chunks=True)`

  Stores an ndarray. Complex arrays become a subgroup with real/imag
  datasets plus is_complex/dtype/shape attrs.

- `g_load_complex(group, name)` -> ndarray

  Loads and reassembles complex arrays from real+imag subgroup.

### Zarr sparse CSR

- `g_store_sparse_csr(group, name, matrix, ...)`: stores scipy sparse as data/indices/indptr subgroup
- `g_load_sparse_csr(group, name, dense=False)`: loads sparse, optionally converting to dense

### Zarr dict storage

- `g_store_dict(group, name, data, ...)`: arrays become datasets, scalars become JSON attrs, nested dicts become sub subgroups
- `g_load_dict(group, name)` -> dict: recursive reconstruction

### Zarr boolean masks

- `g_store_bool_masks(group, name, masks, ...)`: stores as uint8 (Zarr v3 bool issues)
- `g_load_bool_masks(group, name)` -> dict of bool arrays

### Zarr ragged strings

- `write_text_array(group, name, seq, ...)`: stores strings as ragged UTF-8 byte arrays (values + offsets)
- `read_text_array(group, name)` -> list of bytes: reads legacy fixed width or ragged layout

### HDF5 helpers

- `open_hdf5(path, mode="r")` -> h5py.File
- `h5_store_array(group, name, arr, ...)`: stores ndarray, complex -> subgroup
- `h5_load_array(group, name)` -> ndarray
- `h5_store_complex` / `h5_load_complex`: same as h5_store_array (API symmetry with Zarr)
- `h5_store_sparse_csr` / `h5_load_sparse_csr`: CSR subgroup with data/indices/indptr
- `h5_store_dict` / `h5_load_dict`: recursive dict storage
- `h5_store_bool_masks` / `h5_load_bool_masks`: uint8 boolean mask subgroup
- `h5_store_strings` / `h5_load_strings`: variable length UTF-8 HDF5 datasets

## `_serialization`: Type aware NamedTuple serialization

File: `_serialization.py`

Generic write/read for rexgraph NamedTuple types across any storage backend.
Inspects each field at runtime and dispatches to the correct helper:
arrays as datasets, scalars as attrs, dicts as subgroups with array/JSON
split, tuples and list of dicts as JSON, strings as attrs, None fields
recorded separately. Works through a thin StorageAdapter interface that
wraps Zarr groups, HDF5 groups, or .npy directories.

### StorageAdapter interface

Abstract base with methods:

- `put_array(name, arr)` / `get_array(name)` -> NDArray or None
- `put_scalar(name, value)` / `get_scalar(name, default)` -> Any
- `put_json(name, obj)` / `get_json(name, default)` -> Any
- `put_string(name, s)` / `get_string(name, default)` -> str
- `has(name)` -> bool
- `subgroup(name)` -> StorageAdapter

### Adapter implementations

- `ZarrAdapter(group, compressor=None, chunks=True)`: wraps a Zarr group.
  Uses `g_store_complex`/`g_load_complex` from `_compat` for arrays.

- `HDF5Adapter(group, compression="lzf", chunks=True)`: wraps an h5py group.
  Uses `h5_store_complex`/`h5_load_complex` from `_compat`.

- `NpyAdapter(directory)`: wraps a directory of `.npy` files with a
  `_meta.json` sidecar for scalars, strings, and JSON data. Used by the
  `.rcbd` bundle format.

### Field classification

- `_classify_field(value)` -> str

  Determines storage strategy for a NamedTuple field value. Returns one of:
  "array", "scalar", "string", "dict", "tuple", "list_of_dict", "none".
  Lists/tuples of numeric values are detected and stored as arrays.

### NamedTuple write/read

- `write_namedtuple(adapter, name, obj)`

  Creates a subgroup and stores each field by its classified type. Records
  the type name for reconstruction and a `_none_fields` list for None values.

- `read_namedtuple(adapter, name, type_class=None)` -> NamedTuple or dict

  Reads fields back from storage. If type_class is given, constructs an
  instance. Otherwise resolves the type from the stored `_type_name` via
  the type registry. Fixes tuple fields that were stored as JSON lists.
  Falls back to a plain dict if the type cannot be resolved.

### Result dict write/read

- `write_result_dict(adapter, name, data)`

  For non NamedTuple dicts (like `analyze()` output). Arrays become datasets,
  nested dicts with arrays use `_write_dict_field`, everything else is
  JSON encoded.

- `read_result_dict(adapter, name)` -> dict

  Reads arrays by key list and merges with JSON stored scalar data.

### Type registry

- `_resolve_type(type_name)` -> Type or None

  Lazily populates a registry by importing all NamedTuple types from
  `rexgraph.types` on first call.

- `register_type(cls)`: manually register a NamedTuple class for
  deserialization.

## `arrow_bridge`: Arrow/IPC bridge

File: `arrow_bridge.py`

Columnar export of RexGraph data through Apache Arrow, suitable
for interop with Polars, DuckDB, Spark, and any Arrow compatible tool.
All pyarrow imports are lazy: the module can be imported without pyarrow
installed; ImportError is raised only when a function is called.

Complex arrays are split into `<name>__real` / `<name>__imag` columns.
2D array shapes are stored in Arrow schema metadata under `rex_array_meta`
so round trip reshape is exact. Arrays of different lengths are padded to
uniform length for valid Arrow tables.

### Dict of arrays <-> Arrow table

- `arrays_to_arrow(arrays, metadata=None)` -> pyarrow.Table

  Converts a dict of NumPy arrays to an Arrow table. Each array becomes
  one column (or two for complex dtype). Shapes, dtypes, and complex flags
  are stored in schema metadata for exact reconstruction.

- `arrow_to_arrays(table)` -> dict of name -> ndarray

  Reconstructs original shapes, dtypes, and complex values from schema
  metadata. Unknown columns (not in metadata) pass through as is.

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

### IPC file I/O

- `write_arrow_ipc(arrays, path, metadata=None)`

  Writes a dict of arrays to an Arrow IPC file.

- `read_arrow_ipc(path)` -> dict of name -> ndarray

  Reads arrays back from an Arrow IPC file.

### Streaming reads

- `read_arrow_batches(path, batch_rows=100_000)` -> Iterator

  Streams an Arrow IPC file as batches of arrays. Each yielded dict
  contains the same array names but fewer rows, for processing large
  exports without loading everything into memory.

## `safetensors_bridge`: ML transport and indexed encryption

Safetensors export preserves the canonical `rex_state` tensors in a single
file and also supports `TemporalRex`, fingerprint/vector corpora, cache arrays,
and caller owned extra tensors. Plain files retain the native safetensors layout
and remain compatible with earlier RexGraph releases.

Writers accept encryption_properties and readers accept the matching
decryption_properties. The application supplies an authenticated encryption
provider whose property declares `authenticated_encryption = True` and exposes
these methods:

```text
seal(key_id: str, plaintext: bytes, aad: bytes) -> bytes
open(envelope: bytes, aad: bytes) -> bytes
# Optional fast path when key_id came from the authenticated inner manifest:
open_with(key_id: str, envelope: bytes, aad: bytes) -> bytes
```

`seal` must return a self framing authenticated encryption envelope. Configure
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
`read_safetensor_tensor` open only the requested first axis region.

Each new member also carries fixed length min, max, and null count facts for
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
one dimensional, and every selected tensor must share their first axis. Older
encrypted files without statistics still read and query, but their predicate
tensor is scanned in full. Min/max pruning also cannot help an unclustered
predicate whose range overlaps every chunk.

When a property implements `open_with`, core uses it only for a key identifier
that came from the authenticated inner manifest. This resolves the one member
being read without trying or preloading keys for unrelated members. The outer
descriptor is not yet authenticated when the footer is opened, so it does not
qualify for this keyed fast path.

This protects container confidentiality and integrity, not row level access,
identity authorization independent of key possession, or rollback. Ciphertext
and member sizes still disclose approximate complex shape and therefore can
suggest `nV`, `nE`, or `nF`; this version does not pad. Use an external trusted
version/digest anchor when rollback detection matters.

## `parquet_bridge`: Parquet table export/import

File: `parquet_bridge.py`

Exports boundaries, cell attributes, persistence and other graph measurements
as Parquet tables. It loads PyArrow when a function is called and does not
require pandas.

### Generic Parquet I/O

- `parquet_encryption_properties(crypto_factory, kms_connection_config, footer_key=..., column_keys=..., plaintext_footer=False)`: builds opaque PyArrow file encryption properties from caller owned KMS objects. Key values are identifiers, not raw keys. The encrypted footer is the safe default; a signed plaintext footer is explicit. Column keys protect columns inside one file, not rows; separate Rex grade tables need a caller owned file/bundle key policy.
- `write_parquet(data, path, metadata=None, encryption_properties=None)`: writes a dict of equal length 1D arrays; 2D arrays split into `{name}_0`, `{name}_1`, etc. An opaque PyArrow `FileEncryptionProperties` enables Parquet Modular Encryption; `None` preserves plaintext compatibility.
- `read_parquet(path, columns=None, decryption_properties=None)` -> dict: pushes the physical column projection into PyArrow, reassembles requested 2D arrays from split columns, and accepts optional opaque PyArrow `FileDecryptionProperties`.
- `read_parquet_batches(path, batch_rows=100_000, columns=None, decryption_properties=None)` -> Iterator: streaming reads with the same optional decryption properties.

Applications supply PyArrow key providers and encryption properties. Column
projection selects output fields, while the provider controls which keys the
reader can obtain; projection alone does not enforce per column access
permissions. Use the same compatible encryption policy when reading a Parquet
file with another engine.

Every typed table writer/reader accepts the matching opaque encryption/decryption
property as a keyword argument, including character, vertex character, and void
tables. Core does not accept raw keys or implement a KMS client; callers construct
their client and connection policy outside RexGraph.

### Boundary table

- `write_boundary_table(rex, path)`: one row per (edge, boundary_vertex) pair. Handles all edge types: standard (2 rows), self loop (2 rows, same vertex), branching (3+ rows), witness (1 row). Columns: edge_idx, vertex_idx, position.
- `read_boundary_table(path)` -> dict: reconstructs boundary_ptr/boundary_idx from the table.

### Edge table

- `write_edge_table(rex, path, include=None)`: one row per edge. Columns: edge_idx, source, target, boundary_size, edge_type, weight. Optional include for Hodge components.
- `read_edge_table(path)` -> dict

### Vertex table

- `write_vertex_table(rex, path, include=None)`: one row per vertex. Columns: vertex_idx, degree, x, y (spectral layout). Optional: fiedler_vector_L0, layout_3d.
- `read_vertex_table(path)` -> dict

### Face table

- `write_face_table(rex, path)`: one row per nonzero in B2 CSC. Columns: face_idx, edge_idx, orientation (+/-1).
- `read_face_table(path)` -> dict: reconstructs B2_col_ptr, B2_row_idx, B2_vals.

### Persistence table

- `write_persistence_table(result, path)`: from persistence diagram dict. Columns: birth, death, dim, birth_cell, death_cell, lifetime. Metadata stores essential pairs and Betti numbers.
- `read_persistence_table(path)` -> dict: includes betti tuple and essential array from metadata.

### Filtration table

- `write_filtration_table(rex, filt_v, filt_e, filt_f, path, kind="")`: one row per cell. Columns: cell_idx, cell_dim, filtration_value.
- `read_filtration_table(path)` -> dict: splits back into filt_v, filt_e, filt_f by dimension.

### Metrics table

- `write_metrics_table(metrics, path, index_name="cell_idx")`: generic per cell numeric metrics. All arrays must have equal length.
- `read_metrics_table(path, exclude_index=True)` -> dict

### Character tables

- `write_character_table(rex, path)` / `read_character_table(path)`: per edge structural character chi. Columns: edge_idx, chi_0..chi_{nhats-1}.
- `write_vertex_character_table(rex, path)` / `read_vertex_character_table(path)`: per vertex phi and kappa.

### Void table

- `write_void_table(rex, path)` / `read_void_table(path)`: per void triangle data. Columns: void_idx, eta, fills_beta, chi_void_0..chi_void_{nhats-1}. Empty table written when n_voids = 0.

## `sql_bridge`: SQL database bridge

File: `sql_bridge.py`

Stores the same table types as `parquet_bridge` into SQL databases (SQLite,
PostgreSQL, or any SQLAlchemy compatible backend). Each table maps to a
specific part of the algebraic/topological framework. All sqlalchemy and
pandas imports are lazy. Metadata is stored in companion `<table>_meta`
tables as single row JSON.

Requires: `pip install sqlalchemy pandas`

### Engine management

- `get_engine(conn_str)`: returns a SQLAlchemy engine. In memory SQLite uses a shared StaticPool so multiple calls see the same database.

### Boundary table

- `write_boundary_sql(rex, conn, table="boundary", if_exists="replace")`: one row per (edge, boundary_vertex) pair. Columns: edge_idx, vertex_idx, position.
- `read_boundary_sql(conn, table="boundary")` -> dict: reconstructs boundary_ptr/boundary_idx.

### Edge table

- `write_edge_sql(rex, conn, table="edges", include=None, if_exists="replace")`: per edge data with source, target, boundary_size, edge_type, edge_type_name, weight. Optional Hodge components.
- `read_edge_sql(conn, table="edges")` -> dict

### Vertex table

- `write_vertex_sql(rex, conn, table="vertices", include=None, if_exists="replace")`: per vertex data with degree, x, y (spectral layout). Optional: layout_3d, fiedler_vector_L0.
- `read_vertex_sql(conn, table="vertices")` -> dict

### Face table

- `write_face_sql(rex, conn, table="faces", if_exists="replace")`: B2 operator as one row per nonzero. Columns: face_idx, edge_idx, orientation.
- `read_face_sql(conn, table="faces")` -> dict: reconstructs B2_col_ptr, B2_row_idx, B2_vals.

### Persistence table

- `write_persistence_sql(result, conn, table="persistence", if_exists="replace")`: persistence pairs. Columns: birth, death, dim, birth_cell, death_cell, lifetime. Metadata stores essential pairs and Betti numbers.
- `read_persistence_sql(conn, table="persistence")` -> dict

### Filtration table

- `write_filtration_sql(rex, filt_v, filt_e, filt_f, conn, table="filtration", kind="", if_exists="replace")`: one row per cell. Columns: cell_idx, cell_dim, filtration_value.
- `read_filtration_sql(conn, table="filtration")` -> dict: splits into filt_v, filt_e, filt_f.

### Temporal table

- `write_temporal_sql(trex, conn, table="temporal", if_exists="replace")`: per timestep Betti numbers, edge/face counts, Euler characteristic from a TemporalRex.
- `read_temporal_sql(conn, table="temporal")` -> dict: includes betti as T x 3 array.

### Metrics table

- `write_metrics_sql(metrics, conn, table="metrics", cell_dim=0, if_exists="replace")`: generic per cell numeric metrics with cell_dim column.
- `read_metrics_sql(conn, table="metrics", cell_dim=None, exclude_index=True)` -> dict

### Character and void tables

- `write_character_sql(rex, conn, table="character")` / `read_character_sql(conn, table="character")`: per edge structural character chi.
- `write_vertex_character_sql(rex, conn, table="vertex_character")` / `read_vertex_character_sql(conn)`: per vertex phi and kappa.
- `write_void_sql(rex, conn, table="void")` / `read_void_sql(conn, table="void")`: void complex data.

### Streaming

- `read_sql_batches(conn, table_or_query, chunksize=100_000)` -> Iterator: streams SQL results as batches of arrays via pandas chunked reads.

## `csv_loader`: CSV edge list loader with column classification

File: `csv_loader.py`

Loads CSV edge lists and automatically classifies each metadata column's
semantic role using a heuristic cascade: column name pattern matching,
value set statistics (cardinality, average length, delimiter frequency,
numeric fraction), and value content scanning (positive/negative stems,
ordinal terms, identifier patterns). No heavy dependencies: uses only
csv, re, numpy, and collections.

### Column roles

`ColumnRole` defines the taxonomy: TYPE (edge coloring), POLARITY
(flow sign), GROUPING (pathway/cluster), ORDINAL (high/medium/low),
NUMERIC (continuous), EVIDENCE (semicolon delimited sources), REFERENCE
(PMIDs/accessions), DESCRIPTION (free text), UNKNOWN.

### Column profiling

- `ColumnProfile` dataclass: per column statistics: name, role, n_values,
  n_unique, avg_length, is_numeric, has_delimiter, unique_ratio, counts,
  numeric stats, positive/negative values for polarity, name_matched flag.
  Properties: is_categorical, is_freetext, is_delimited_list, is_binary.

- `classify_columns(meta)` -> dict of name -> ColumnProfile

  Profiles and classifies all metadata columns. Priority: name pattern match,
  then value heuristics, then statistical fallback.

### Weight construction

- `build_weights(profiles, nE)` -> (w_E, negative_types)

  Constructs signed edge weight vector. Magnitude from first numeric column
  (default 1.0), ordinal columns scale multiplicatively (high=1.0,
  medium=0.6, low=0.3), polarity column applies sign (-|w| for negative
  types).

### Edge attribute assembly

- `build_edge_attrs(profiles)` -> dict

  Assembles the edge_attrs dict for `analyze()`. Ensures the type column
  is keyed as "type" if one exists.

### GraphData

`GraphData` dataclass: fully parsed and classified CSV data: sources,
targets, vertices, src_idx, tgt_idx, meta, profiles, edge_attrs, w_E,
negative_types, nV, nE. Methods:

- `summary()` -> str: human readable column classification table
- `to_rex()` -> RexGraph: constructs graph with magnitude weights and polarity signs

### High level loader

- `load_edge_csv(path, roles=None)` -> GraphData

  Loads a CSV edge list with full column role classification. Auto detects
  source/target columns by name pattern. Manual role overrides via the
  `roles` dict. Returns GraphData ready for RexGraph construction and
  visualization.

  Supported source column names: source, src, from, head.
  Supported target column names: target, tgt, to, tail, dest.
  Delimiter detection: commas, tabs, pipes, semicolons via csv.Sniffer.

## `json_loader`: JSON graph loaders

File: `json_loader.py`

Loads graph data from common JSON interchange formats used in bioinformatics,
clinical research, and network science. Auto detects format from JSON
structure. All loaders produce RexGraph via `from_graph()` or the constructor.
No heavy dependencies beyond json and numpy.

### Auto detection

- `load_json(path, format=None, threshold=0.0, directed=False)` -> RexGraph

  Loads any supported JSON format. Auto detects from structure if format is
  None. Detection rules: "boundary_ptr" -> rexgraph native, "elements" ->
  cytoscape, "nodes"+"links" -> networkx, "edges" -> edge_list,
  "matrix"/"adjacency" or list of lists -> adjacency.

### Format specific loaders

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

  NetworkX node link format: `{"nodes": [{id}], "links": [{source, target}]}`.
  Reads weight/value from link data.

- `load_adjacency_json(path, threshold=0.0, **kwargs)` -> RexGraph

  Adjacency matrix: `{"matrix": [[...]]}`, `{"adjacency": [...], "labels": [...]}`,
  or bare list of lists. Delegates to `RexGraph.from_adjacency()`.

### Matrix CSV

- `load_matrix_csv(path, threshold=0.0, absolute=True, directed=False)` -> RexGraph

  Square matrix CSV (correlation matrices, gene expression). Auto detects
  row/column labels. Threshold filters by |weight|. If absolute=True, uses
  |weight| as edge weight and sign(weight) as edge sign. Extracts edges from
  upper triangle for undirected graphs.

## `zarr_format`: Zarr based storage

File: `zarr_format.py`

Chunked, compressed Zarr stores for RexGraph, TemporalRex, NamedTuples,
and raw arrays. Works with both Zarr v2 and v3. On disk layout mirrors the
algebraic structure: core reconstruction arrays at root, cache groups
(algebra, spectral, relational, topology, hodge, faces, field, wave, signal,
quotient, persistence, temporal, standard_metrics) as subgroups.

Requires: `pip install zarr`

### Simple array I/O

- `save_zarr_array(arr, path)`: saves a NumPy array to a .zarr store
- `load_zarr_array(path)` -> ndarray

### RexZarrFormat class

`RexZarrFormat(compressor="default", chunks=True, large_threshold=50000)`

Main class with configurable compression and chunking.

- `write(path, obj, cache=None)`: writes RexGraph, TemporalRex, or ndarray. Cache accepts "all" or a list of group names.
- `read(path)` -> RexGraph, TemporalRex, or ndarray: auto detects object type from attrs.

Container API for multi object stores:
- `write_to_group(path, name, obj, **kw)`: writes to /objects/\<name\>
- `read_from_group(path, name)` -> object
- `list_groups(path)` -> list of names

### Cache groups

13 cache groups, each containing related computed properties:

- algebra: B1, B2, L0, L1, L2, L1_down, L1_up, overlap_adjacency, L_overlap
- spectral: spectral_bundle dict, eigenvalues, Fiedler vectors, layout
- relational: RL, evals/evecs, alpha constants, Lambda
- topology: Betti numbers, Euler characteristic, edge types, cycle basis
- hodge: Hodge decomposition components, rho
- faces: detected face data and metrics
- field: field operator M, eigendecomposition, mode classification
- wave: density matrices
- signal: perturbation results
- quotient: subcomplex masks, quotient operators, relative Betti
- persistence: diagrams, enrichment
- temporal: edge/face lifecycle, Betti matrix, BIOES
- standard_metrics: PageRank, betweenness, clustering, Louvain

### Convenience functions

- `save_zarr(path, obj, cache=None, compressor="default")`: module level save using default format instance
- `load_zarr(path)` -> object: module level load

## `hdf5_format`: HDF5 storage

File: `hdf5_format.py`

Single file counterpart to `zarr_format`. Same serialization surface
(RexGraph, TemporalRex, cache groups, NamedTuples) stored in a single .h5
file via h5py. On disk layout mirrors the Zarr format with the same 13
cache groups. Compression via HDF5 filters (lzf default, gzip optional).

Requires: `pip install h5py`

### Simple array I/O

- `save_hdf5_array(arr, path)`: saves a NumPy array to an .h5 file
- `load_hdf5_array(path)` -> ndarray

### RexHDF5Format class

`RexHDF5Format(compression="lzf", compression_opts=None, chunks=True, large_threshold=50000)`

Main class with configurable compression and chunking.

- `write(path, obj, cache=None)`: writes RexGraph, TemporalRex, or ndarray to a single .h5 file. Cache accepts "all" or a list of group names.
- `read(path)` -> RexGraph, TemporalRex, or ndarray: auto detects object type from attrs.

Container API for multi object files:
- `write_to_group(path, name, obj, **kw)`: writes to /objects/\<name\> within the same .h5 file
- `read_from_group(path, name)` -> object
- `list_groups(path)` -> list of names

### Cache groups

Same 13 cache groups as `zarr_format`: algebra, spectral, relational,
topology, hodge, faces, field, wave, signal, quotient, persistence,
temporal, standard_metrics. Each group writes the same set of computed
properties.

### Convenience functions

- `save_hdf5(path, obj, cache=None, compression="lzf")`: module level save using default format instance
- `load_hdf5(path)` -> object: module level load

## `bundle`: Relational complex binary directory (.rcbd)

File: `bundle.py`

Portable relational complex package using only NumPy .npy files and JSON.
No zarr, h5py, scipy, or pyarrow required. A bundle is a self contained
directory with individual .npy files for exact reconstruction, plus
optional precomputed cache. Memory mappable for lazy/partial reads.

On disk layout:

    my_graph.rcbd/
    +-- MANIFEST.json
    +-- boundary_ptr.npy, boundary_idx.npy
    +-- B2_col_ptr.npy, B2_row_idx.npy, B2_vals.npy
    +-- w_E.npy (if weighted)
    +-- cache/
        +-- layout.npy, eigenvalues_L0.npy, B1.npy, ...

### RCBDBundle class

`RCBDBundle(root, manifest)`

- `manifest` -> dict: parsed MANIFEST.json
- `object_type` -> str: "RexGraph" or "TemporalRex"
- `path` -> Path

Construction:
- `RCBDBundle.from_graph(graph, cache=None)`: creates an in memory bundle spec (does not write to disk). Call `.save()` to persist.
- `RCBDBundle.load(path, mmap=False, decryption_properties=None)`: loads from an
  `.rcbd` directory and authenticates its complete encrypted inventory when present.
  `mmap=True` remains lazy for plaintext arrays.

Persistence:
- `save(path, encryption_properties=None)`: writes through an isolated staging
  directory and serializes publication (threads on every platform and processes
  through `flock` on POSIX), so writers cannot interleave two exports.

`save` treats its destination as an already authorized filesystem path and
creates missing parent directories. A request facing caller must resolve and
validate that destination inside its bound workspace root before calling core;
bundle publication is not a path authorization boundary.

Reconstruction:
- `to_graph(allow_unsealed=False)` -> RexGraph. The opt in is only for migrating a
  trusted bundle written before content digests existed; the default refuses an
  unsealed bundle.
- `to_temporal()` -> TemporalRex
- `to_object(allow_unsealed=False)` -> RexGraph or TemporalRex (auto dispatch)

Array access:
- `bundle["boundary_ptr"]` -> ndarray: loads by name (root or cache/)
- `read_slice(name, index=None)` -> ndarray: decrypts only the selected
  first axis chunks of a protected member
- `where(name, operator, value=None)` -> int64 ndarray: uses authenticated
  member statistics to prune impossible predicate chunks
- `select(names, where=(name, operator, value))` -> dict: checks a predicate
  exactly and gathers matching rows from named members
- `clear_query_cache()`: forgets opened chunks and statistics retained by bundle
  queries
- `"layout" in bundle` -> bool
- `list_arrays()` -> sorted list of array names
- `read_cache()` -> dict of all cached arrays + scalar cache from manifest

### Authenticated encrypted bundles

The same opaque `ContainerEncryptionConfig` and property contract used by the
safetensors bridge applies to `.rcbd` directories. With
`encryption_properties=None`, the existing named `.npy` layout is unchanged.
With a property, `MANIFEST.json` contains a small public envelope and the logical
manifest is encrypted by `footer_key` unless `plaintext_manifest=True` is
explicit. Generated storage paths do not reveal logical names. Protected members
are authenticated `.rexenc` chunk streams; names in `plaintext_tensors` remain
native `.npy` files whose dtype, shape, bytes, and presence are bound by the
authenticated manifest.

Opening an encrypted bundle authenticates the manifest before using its paths,
then rejects missing, extra, renamed, truncated, altered, or cross bundle member
files. The inventory covers canonical Rex state, cache arrays, every temporal
snapshot, and every face snapshot. Members omitted from an exact caller policy
use `footer_key`; they never silently become public. `allow_unsealed=True` is a
legacy plaintext migration flag and cannot downgrade this check.

Cold `RCBDBundle.load` streams SHA-256 over every stored member, so it rejects a
same length substitution before the member is requested. That eager validation
is O(total ciphertext bytes). After validation, logical name lookup is O(1), and
`read_slice` decrypts only the selected member chunks. Deferring the hashes would
make initial open cheaper but could not detect an unrequested same size swap.

Per chunk query statistics use the same fixed length format as safetensors. Facts
for a protected member are sealed under that member's key, so opening the footer
does not disclose value ranges for unauthorized grades. `where` and `select`
prune, verify, and gather through the already open `RCBDBundle`; a legacy bundle
without statistics falls back to scanning its predicate member.

Public .npy members support memory mapping. Protected members require
`read_slice`; accessing them as memory maps raises an error. Encryption
provides confidentiality and authenticated container integrity. It does not
provide row permissions or rollback detection. Unpadded file sizes reveal
approximate complex shape; detecting replacement with an older valid directory
requires an external trusted version record.

### Cache groups

Same structure as zarr_format/hdf5_format: algebra, spectral, topology,
hodge. Each group writes specific computed properties as .npy files in the
cache/ subdirectory. Scalar values (Betti numbers, Euler characteristic,
Hodge percentages) are stored in MANIFEST.json under "cache_scalars".

### TemporalRex bundles

Snapshots stored as numbered subdirectories:

    temporal.rcbd/snapshots/0/sources.npy, targets.npy
    temporal.rcbd/snapshots/1/...
    temporal.rcbd/face_snapshots/0/B2_col_ptr.npy, B2_row_idx.npy (optional)

### Convenience functions

- `save_rcbd(path, obj, cache=None, encryption_properties=None)`: saves RexGraph
  or TemporalRex to an `.rcbd` directory
- `load_rcbd(path, allow_unsealed=False, decryption_properties=None)` -> RexGraph
  or TemporalRex: loads from a sealed `.rcbd` directory by default. Set
  `allow_unsealed=True` only while migrating a
  trusted pre digest RexGraph bundle; wire and other container readers never expose
  this downgrade.

`RexBundle`, `save_rex`, and `load_rex` remain source compatibility aliases.
Omitting a suffix writes `.rcbd`; an explicit legacy `.rex` path is honored so
existing save/load pairs remain valid. Existing `.rex` directories with legacy
`rex-bundle` manifests remain readable.

## `rcbf`: Relational complex binary file (.rcbf)

`load_rcbf(path)` imports an RCBF stream into a RexGraph, then it can be
stored in RCDB or queried through RCQL like any other RexGraph source. The bridge
preserves C0 labels, primary C1 boundary spans (including branching relations),
C1 weights and signs, and C2 boundaries only when their coefficients and chain
condition are exact. It never applies a clique or star expansion.

RCBF stores a sequential stream, while RCBD stores an addressable directory.
The RCBF importer reconstructs the current grade 0 through grade 2 state and
excludes derived numeric products and temporal tails. A stream containing
temporal states requires `allow_current_snapshot=True` to select that current
state. Unsupported mixed relation directions and inexact grade 2 coefficients
are rejected.

## Provenance and confidentiality

These modules define canonical identities, encryption, signatures, mutation
artifacts and replication. Object identity covers the complete canonical state
header, including orientation, channel declarations, cell metadata and sections.
A tensor digest alone is insufficient to identify that state.

Cryptography is imported when encryption or signing is requested. Parquet export
loads pyarrow when called. The modules can be imported without these extras.

## `manifest`: Canonical bytes for a digest

File: `manifest.py`

Manifest encoding uses sorted JSON keys, no whitespace and no NaN values.
Length prefixes distinguish the parts of a framed digest.

- `canonical_json(value)` -> bytes

  Encodes JSON with sorted keys, compact separators and UTF-8 output.
  allow_nan=False rejects nonfinite float values.

- `manifest_digest(value, algorithm="sha256")` -> hex str

  Hashes `canonical_json(value)` with a domain prefix and version that
  distinguish it from a bare payload hash.

- `digest_parts(kind, parts, algorithm="sha256")` -> hex str

  Hashes an ordered sequence of named values, prefixing each name and value
  with its length.

## `security`: Sealed bytes and detached signatures

File: `security.py`

AES-256-GCM with an authenticated header, and Ed25519 signatures. Keys arrive
through a `KeyProvider` protocol and are never serialized: an envelope carries a
key identifier.

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

  `Ed25519Signer.verifier()` returns the verifier for that signer.

  Returns a boolean that the caller must check. Callers distinguish record
  kinds in the signed bytes. `TransitionCommit` and `CommitLink` include
  their kind for this purpose.

## `transition` and `commit`: changes and lineage

Files: `transition.py`, `commit.py`

`TransitionCommit` identifies a change between two states. `CommitLink`
identifies its position in a lineage. Their signatures can be renewed separately.

- `TransitionCommit(previous_state, delta_state, resulting_state, tx_time, actor, policy, signer_id, signature)`

  `previous_state` and `resulting_state` are object identities; `delta_state` is
  the identity of the evidence between them. `policy` is the digest of the policy
  in force, so a package cannot later be verified under a weaker one.

- `CommitLink(transition_digest, parent_digest, signer_id, signature)`

  parent_digest is None at genesis; verification checks the expected parent
  link.

Both embed their own `object_type` in `signing_bytes()`, so a transition
signature does not verify as a lineage signature.

## `transport`: A self describing frame

File: `transport.py`

`MAGIC` + header length + canonical header + payload, where the header carries
the object type, the payload size and its SHA-256.

- `pack(payload, object_type, metadata=None)` -> bytes
- `inspect(blob, max_header=...)` -> `TransportInfo`, without validating the payload
- `unpack(blob, verify=True)` -> `(payload, metadata)`

The unkeyed digest detects corruption. Authentication requires a signature
or an authenticated transport.

## `temporal_state`: A TemporalRex as verifiable tensors

File: `temporal_state.py`

- `to_temporal_state(trex)` -> `TemporalState`
- `verify_temporal_state(state)` -> bool
- `from_temporal_state(state, verify=True, allow_legacy=False)` -> TemporalRex

Temporal state v2 includes every header field in its content seal. Version 1
covers tensors only and is supported for explicit migration.
`verify_temporal_state` returns False for a version mismatch;
`from_temporal_state` accepts v1 only with `allow_legacy=True`, independently
of the verify flag.

Reconstruction requires nonnegative T, boolean flags, sorted unique checkpoints
in range beginning at zero, and one finite nondecreasing timestamp per step.
Invalid declarations raise an error.

## `catalog`: File and object metadata

File: `catalog.py`

- `object_digest(value)` -> hex str, over the full canonical `RexState` header
- `state_object_digest(state)` -> the same for an already built state
- `FileCatalog(roots)`, `.entries()`, `.info(name)`

A catalog exposes names relative to a root label, such as `root0/sub/name`,
while keeping the absolute root private. It rejects paths that escape the root
and excludes symlinks.

`object_digest` includes channel declarations, agent and cell metadata, nested
complexes, sections and Merkle state together with the tensors. A signed
transition uses this complete identity for each endpoint.

## `mutation`: A change bound to its endpoints

File: `mutation.py`

- `MutationPolicy(require_transition_signature, require_lineage_signature, allowed_signers)`
- `prepare_mutation(previous, resulting, tx_time, actor, policy, parent_digest, ...)` -> `MutationPackage`
- `verify_mutation(package, *, previous, policy=None, verifiers=None, parent_digest=...)` -> bool
- `mutation_to_bytes(package)` / `mutation_from_bytes(blob, allow_legacy=False)`

A v2 mutation package carries the complete resulting RexState. The caller
supplies the preceding state for transition verification.

`previous` is required; omitting it raises TypeError, and genesis requires an
explicit None. Verification checks the carried result against that preceding
state. The parent_digest sentinel distinguishes an omitted parent check from an
explicit genesis declaration.

Identity verification uses the carried canonical state, including metric
weights, graded boundaries, signals, labels, metadata, nested complexes,
sections and Merkle state. Temporal checkpoints and deltas record structural
changes and omit some of these components.

The writer emits v2 packages. Verification rejects v1 packages and checks that
the declared version agrees with the payload format.

## `privacy`: Scoped pseudonyms

File: `privacy.py`

- `PrivacyProjection(fields, pseudonym_fields, scope, key_id)`
- `scoped_pseudonym(value, scope, key_id, keys)` -> base32 token
- `project_rows(rows, projection, keys)` -> projected rows

The same subject identifier under two different scopes produces two unrelated
tokens, so the same person in two studies cannot be joined. Scope and value are
each length prefixed inside the HMAC message behind a domain prefix, so
`(scope_a, id)` cannot frame identically to `(scope_b, id')`.

The configured key is required to create or resolve a pseudonym; a missing key
raises an error.

Tokens are stable within their scope and can link records in that scope.
Pseudonymisation therefore retains a scoped identity and does not anonymize the
records.

## `export`: A Parquet artifact with an identity

File: `export.py`

- `parquet_bytes(data, metadata=None)` -> bytes, in memory, no temporary file
- `export_parquet(data, partition_digest, key_id=None, keys=None)` -> `(payload, ExportManifest)`
- `verify_export(payload, manifest, keys=None)` -> bool

`verify_export` recomputes SHA-256 from the supplied payload and compares it
with the manifest using hmac.compare_digest. A changed payload fails
verification.

An encrypted export wraps the complete artifact in an AES GCM envelope and
decrypts it before reading a column. Selective encrypted reads are available
through safetensors_bridge and bundle.

Column names are sorted before export, producing deterministic output
independently of the supplied mapping order.

ExportManifest records the partition digest, schema, payload identity and
encryption status. It has no PrivacyProjection digest field. Applications using
project_rows must retain and check the disclosure policy separately from the
export manifest.

## `partition_state`: Stored subcomplexes

The semantic implementation is `rexgraph.partition_state`; this I/O module is
a compatibility alias. `rexgraph.selection` provides its source bound facade.

- `build_rex_partition(rex, e_mask, *, v_mask=None, f_mask=None, grade_masks=None, policy_digest="", closure="subcomplex", carried_state="structural")` -> `RexPartition`

  `e_mask` is positional and required by this legacy mask interface. Empty
  selections are valid. `closure` accepts only `"subcomplex"`.

Restriction includes every selected cell's complete boundary, proceeding
downward through the grades to preserve boundary closure and `B_k B_(k+1) = 0`.

Grade one closes from the raw `boundary_ptr`/`boundary_idx` supports rather than
from matrix non zeros, because a self loop's signed column cancels to sparse zero
and counting non zeros would drop a vertex the relation structurally contains.

Accepts e_mask and f_mask, plus explicit grade_masks for grade two and above.
Orientation, weights, signs, heads, channels and boundary attribution are
copied so the partition and source own independent state. Empty grades retain
their original numbering. `carried_state="structural"` omits application
metadata, signals and sections; `"all"` carries registered component state.
policy_digest records the projection policy without granting authorization.

Public `Selection`, `Lineage`, `restrict` and `glue` share this builder and its
registered component transport. Lineage records requested cells separately
from closure added cells and checks both original and resulting bases. RCQL
result artifacts and query caches preserve these records. Legacy result
artifacts without original basis sizes retain their old members but cannot
claim certified lineage. Explicit empty C2 spaces use a sealed native header
declaration; older native readers refuse the unknown field.

## `replication`: Applying a transition chain

File: `replication.py`

- `pack_replication(checkpoint, mutations, *, checkpoint_state="", checkpoint_commit="")`
  -> `(bytes, ReplicationManifest)`
- `unpack_replication(blob)` -> `(checkpoint_bytes, mutation_bytes, manifest)`

  The middle element contains the original mutation bytes. Unpacking decodes
  each package to validate its type and ordering, then returns the bytes for
  application by the caller.

`unpack_replication` checks digests, counts and contiguous lineage.
`apply_replication` checks the loaded checkpoint's identity against the
manifest, then verifies each package against the current graph and preceding
link digest before applying it. A missing interior package breaks lineage and
endpoint agreement.

`checkpoint_loader` is a plain callable, which is how this module applies a chain
into a store without importing one.

The transport uses unkeyed SHA-256 for corruption and consistency checks.
Authenticated replication requires producer signatures and verifier keys. The
transition binds the signature policy digest, so a signed chain retains its
verification requirements.

## Declared reader capabilities

`rexgraph.io.readers` exposes `ReaderSpec`, `ReaderRegistry`, `ReaderParameter`,
`ParameterSchema`, `read_batches` and explicit `profile`. Extension, media type
and caller supplied bounded sniff prefix claims share one priority/refusal rule.
Equal priority matches name every competing reader. Naming a reader resolves the
choice explicitly. Resolution does not consume a stream or inspect an input path.

Reader options are closed typed declarations, checked before opening input.
Every emitted batch must match the requested RecordSchema and batch size. The
reader generator closes on completion, refusal or explicit consumer close.
CSV and TSV profiling observe bounded field inventories through the same parser;
profiles never apply inferred number rules or construct graphs. Other readers
need an explicitly installed profiler to offer that capability.

Trusted plugins export ReaderSpec values through the `rexgraph.readers` Python
entry point group. `registry.load_plugins()` is explicit and executes installed
code; declarations and wire payloads never invoke it. Registration publishes the
validated set together. Provider authors must increment ReaderSpec.version when
changing their declared contract; its digest does not authenticate code.

DatasetDeclaration seals its declared options and vertex domain and refuses
changed nested values. Native LocalStore directories are cataloged as one RCDB
entry, including partial/damaged native layouts. Catalog physical hashes include
ownership headers and journals; they are not transactional logical store hashes.
Core requires an injected loader to open an RCDB entry.
