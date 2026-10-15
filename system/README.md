# RexGraph System

System is a browser application for inspecting registered complexes, datasets,
file catalogs and RCDB stores. It executes queries through RCQL and displays
their results. Native result downloads preserve exact values and source
provenance for use in another application.

## Installation

From the repository root:

```sh
make install
make system
```

Open [http://127.0.0.1:8010](http://127.0.0.1:8010). System requires Core and
RCQL and can be installed independently of Agent. If those packages are already
installed, run `python -m pip install ./system` in the repository root. The
package includes the frontend and its licensed React runtime for offline use.
See the [installation guide](../README.md#installation) for environment setup.

## Opening a complex

Run `.venv/bin/rexgraph-system` with no initial source, or load a native graph
file at startup:

```sh
.venv/bin/rexgraph-system --source main=graph.rcbd
```

Applications can register live values with `system.register_source`. The source
selector lists registered complexes, datasets, catalogs and stores. Its panels
show the information supported by the selected source, subject to that source's
permissions.

## Query results

`POST /api/query` returns a JSON preview by default. Send
`{"query": "FROM ... RETURN ...", "result_format": "native"}` to download a
complete RCQL result with MIME type `application/vnd.rexgraph.rcql-result` and
filename `result.rgqr`. Load it with `rcql.Result.from_bytes(response.content)`.
Both response formats execute the query once.

Selection and partition previews show counts and a bounded list of original
cell addresses at each grade. Truncation flags identify incomplete previews.
Native downloads retain the full selections, maps, fields and source references.
Partition lineage records the original and resulting states, transport policy
and parent records; modified partitions are rejected before preview or download.

The API and numeric results selector accept `exactness="declared"` or
`"exact"`, using RCQL's arithmetic policy. An exact request is checked before
returning a preview or native payload. See the
[RCQL guide](../rcql/README.md#arithmetic-and-operator-contracts) for its scope.

## Registering datasets

Call `system.register_dataset(name, declaration, source, registry=None,
policy=None)` to register a dataset using Core's reader and relation mapping.
Registration opens the input and creates the binding; queries resolve its name
with `FROM DATASET(name)`.

## File catalogs

Register local roots with a catalog name:

```sh
.venv/bin/rexgraph-system --catalog files=/data/rex
```

Catalogs list `.rcbd` bundles, legacy `.rex` bundles, safetensors files and RCDB
stores. Bundle manifests determine their format. Names such as
`root0/corpus.rcbd` are relative to a root label, keeping absolute paths outside
the query and frontend. Catalogs exclude symlinks and search using literal
terms. Full file hashes are computed when requested.

```text
FROM CATALOG("files") RETURN FILES()
FROM CATALOG("files") RETURN SEARCH("corpus")
FROM CATALOG("files") RETURN FILE_HASH("root0/corpus.rcbd")
FROM FILE("files", "root0/corpus.rcbd") RETURN BETTI(1), STATE_HASH()
```

`/api/source?name=...` and `/api/catalog?name=...` accept literal registered
names, including slashes. Applications can supply explicit format loaders
through `system.register_catalog(..., loaders=...)`; the catalog interface
itself does not require RCDB.
