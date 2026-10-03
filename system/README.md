# RexGraph System

System is the RexGraph observatory. It reads live Rex values through RCQL and does not reimplement the mathematics.

Run it with no initial source:

```bash
rexgraph-system
```

Load a Rex source at startup:

```bash
rexgraph-system --source main=graph.rcbd
```

Applications can also register live values with `system.register_source`.

`POST /api/query` returns a JSON preview by default. Send
`{"query": "FROM ... RETURN ...", "result_format": "native"}` to download a
lossless RCQL result with MIME type `application/vnd.rexgraph.rcql-result` and
filename `result.rgqr`. Load it with `rcql.Result.from_bytes(response.content)`.
The same codec serves query caches and preserves exact values, partitions,
carried fields, provenance and shared source references. Both response formats
execute through RCQL once; native download does not rerun a preview query.

Selection and partition previews show counts and bounded original cell addresses
at each grade, with explicit truncation flags. Partition lineage identifies the
source and result states, transport policy and parent records. Use the native
download for full selections and maps; it preserves them through the shared
RCQL codec. A modified partition is refused before preview or download.

The wheel includes the frontend and its licensed React runtime. Launching works
offline and requires no Agent installation. RCQL evaluates queries; the frontend
renders their results.

## File catalogs

Applications can also call `register_dataset(name, declaration, source,
registry=None, policy=None)` to pin input through the shared Core/RCQL declaration
path. Query it with `FROM DATASET(name)`. Dataset input is opened during explicit
registration; query text cannot open a path. `/api/query` accepts
`exactness="declared"` (default) or `"exact"`, forwarding the request to RCQL.
Exact output refusal occurs before returning preview or native bytes.

The source selector and query editor use registered DATASET, REX, CATALOG or RCDB
names. The numeric results selector applies the same declared/exact policy as the
API. Structural sources expose their panels, catalogs expose Files, and RCDB
sources with statistics expose the RCDB view. Source permissions apply to detail
inspection and panels. Native downloads retain full values; previews are bounded.

`/api/source?name=...` and `/api/catalog?name=...` accept literal registered names,
including slashes. `register_catalog(..., loaders=...)` accepts explicit kind
loaders; using catalogs does not require RCDB in the default installation.

System can register explicit local roots without exposing absolute paths to RCQL or the frontend:

```bash
rexgraph-system --catalog files=/data/rex
```

Catalogs index `.rcbd` bundles (including legacy `.rex` ones, which are identified by their manifest rather than their name), `.safetensors` files, and RCDB stores. Full content hashes are computed on demand so large model files do not slow catalog startup. Catalog names are relative to an opaque root label. Symlinks are not traversed. Search uses literal terms, not regular expressions or shell patterns.

```text
FROM CATALOG("files") RETURN FILES()
FROM CATALOG("files") RETURN SEARCH("corpus")
FROM CATALOG("files") RETURN FILE_HASH("root0/corpus.rcbd")
FROM FILE("files", "root0/corpus.rcbd") RETURN BETTI(1), STATE_HASH()
```
