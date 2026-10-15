# RexGraph Agent

Agent provides the application server, analysis pipeline, worker orchestration,
data connectors and model tools for RexGraph. It depends on Core and RCDB.
Workers can be Python callables or services with an OpenAI compatible endpoint.

## Installation

From the repository root:

```sh
make install
make serve
```

The equivalent script is `sh install.sh`. Both install the standard profile
and all five repository packages in `.venv`. Open
[http://127.0.0.1:8000](http://127.0.0.1:8000) and use the admin token printed
on first startup. The HTTP API documentation is at `/docs`.

| Command | Agent dependencies |
| --- | --- |
| `make install-server` | Web application and API |
| `make install` | Server, schema tools, connectors, PDF/OCR bindings, training export and YAML |
| `make install-ml` | Standard profile plus PyTorch and Hugging Face tools |
| `make install-dev` | Standard profile and test tooling, installed as editable packages |
| `make install-all` | Declared runtime extras, including cloud and model integrations |
| `make install-dev-full` | Development dependencies for optional CPU feature tests |

Add an integration with `sh install.sh --extras trustgraph`, or combine extras
as `--extras langchain,langgraph`. Install Tesseract for scanned documents,
`llama-server` for managed local inference, and a compatible PyTorch build for
GPU training. The [installation guide](../README.md#installation) covers these
optional runtimes.

## Analysis

```python
from rexgraph import RexGraph
from agent.pipeline import AnalysisPipeline

rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
result = AnalysisPipeline(rex).run(depth="standard")
print(result["hodge"]["dim_H"])
```

The pipeline reports Hodge components and structural channel measurements.
Hodge decomposition separates gradient flow from vertex potentials, curl flow
supported by face boundaries, and harmonic flow in the Hodge operator's kernel.
The relationships and signal being analyzed determine how these measurements
are interpreted in an application.

`agent.auto.auto_rex` accepts edge tables, JSON, NumPy arrays, DataFrames and
text. The selected adapter determines which relationships are constructed from the
input. Core's [declared readers](../rexgraph/io/README.md) let an application
specify that mapping, source types, exact number conversion and provenance.

## Data and optional features

The standard profile includes common file readers, PDF extraction and Python
OCR bindings. Scanned documents also require the Tesseract executable. Model
weights, database credentials and external datasets are supplied for the
selected application. `make doctor` reports optional packages, external tools
and configured lexical files; `make verify` checks the installed software.

PDF rendering rejects incomplete page sets and stops its worker if conversion
exceeds five minutes. Temporary images are removed after processing or failure.
Model discovery checks configuration files and every indexed weight shard before
reporting a download as available. Model loaders validate file contents and the
required tokenizer or processor files when loading the model.

| Extra | Use |
| --- | --- |
| `molecules` | RDKit input and molecular interpretation |
| `analytics` | DuckDB and Polars views of stored signatures |
| `baselines` | scikit learn, LightGBM and XGBoost comparisons |
| `ml` | PyTorch and Hugging Face model tools |

Install additional dependencies with `sh install.sh --extras analytics,molecules`
from the repository root. The [dependency guide](../DEPENDENCIES.md) lists the
other extras and their requirements.

Lexical answerers use optional WordNet, Roget and Wiktionary resources. Configure
the data directory before starting Agent:

```sh
export REXGRAPH_LEXICAL_DIR=/path/to/lexical
make doctor
```

| Resource | Relative path under that directory | Individual override |
| --- | --- | --- |
| WordNet | `wordnet/english-wordnet-2024.xml.gz` | `REXGRAPH_WORDNET` |
| Roget | `thesaurus/rogets-pg22.txt` | `REXGRAPH_ROGET` |
| Wiktionary | `wiktionary/wiktionary-en.rexidx` | `REXGRAPH_WIKTIONARY` |

The default directory is `~/.local/share/rexgraph/lexical`, under `XDG_DATA_HOME`
when configured. `REXGRAPH_DATA_DIR` selects a RexGraph data root containing a
`lexical` directory. Existing data in the previous checkout location remains
usable when no new location is selected. Answerer constructors also accept
explicit paths. The Wiktionary file is the adapter's prepared index, rather
than a raw dump; [its source module](agent/adapters/wiktionary.py) documents
index construction. Installation does not download or convert these resources.

## Configuration and cache

Configure directories before starting Agent or its command line tools. On Linux
and macOS, configuration defaults to `~/.config/rexgraph` and the cache defaults
to `~/.cache/rexgraph`. `XDG_CONFIG_HOME` and `XDG_CACHE_HOME` change the parent
directories. Windows uses `APPDATA` for configuration and `LOCALAPPDATA` for
the cache.

| Setting | Purpose |
| --- | --- |
| `REXGRAPH_CONFIG_DIR` | Configuration, credentials, workspace records and journals |
| `REXGRAPH_CACHE_DIR` | Reusable analysis results and OCR server process files |
| `REXGRAPH_MODELS_DIR` | Explicit model downloads; defaults to `models` under the cache directory |
| `REXGRAPH_MODEL_SCAN_DIRS` | Local model discovery roots, separated by the platform path separator |
| `REXGRAPH_SESSION_DIR` | Saved analysis sessions |
| `REXGRAPH_NO_CACHE=1` | Disable reusable analysis and JSON result caching |
| `REXGRAPH_CACHE_MAX_ENTRIES` | Entries retained after pruning; defaults to 2000, with 0 disabling result caching |

Sessions default to `sessions` under the configuration directory. An existing
`~/.rexgraph-agent/sessions` directory remains the default when no configuration
or session directory override is set. An explicit directory setting isolates
session storage from that legacy location.

`REXGRAPH_MODEL_SCAN_DIRS` replaces the common model search locations. Additional
directories supplied through `REXGRAPH_MODEL_DIRS` or an explicit discovery call
are still included.

Session writes publish graph data before updating the history index. A failed
write raises an error and retains the previous history. A session loaded before
another writer appends history must be reloaded before it can write. Copied or
moved session directories retain access to their contained snapshots.

File, OCR and pipeline uploads use `REXGRAPH_MAX_UPLOAD_MB`, which defaults to
100 MiB. The limit covers the total file content in a pipeline request.
Invalid or nonpositive values use the default. Uploaded temporary files are
removed after processing, failed reads or cancellation.

Explicit RexGraph settings take precedence over platform defaults and support
`~` in paths. Hugging Face downloads use its separate cache settings,
including `HF_HOME` and `HF_HUB_CACHE`; model discovery reads those locations.

Analysis cache keys include input content, analysis parameters and the cache
format version. Stored complex data and its sidecar must have matching digests.
A missing, stale or damaged entry is recomputed. Pruning retains the most recently
written entries and runs on the first successful write in each process, then
every 128 writes. The count can temporarily exceed the configured limit between
sweeps. Pruning preserves model files and service configuration.

Chat reuses deterministic single document answers only when document state and
retrieval parameters match. Live model generation, lexical answers and queries
over mutable corpora or stores bypass the response cache.

## The Hive

`Hive.add_worker` registers a callable with a declared capability. Supported
capabilities are `generate`, `predict`, `score`, `embed`, `analyze` and `transform`.
Use `invoke` to call a named worker and `dispatch` to route a text request to a
worker with the `generate` capability.

```python
from agent.hive import Hive
from rexgraph import RexGraph

hive = Hive()

def analyze(data, **kwargs):
    graph = RexGraph.from_graph(data["sources"], data["targets"])
    return {"betti": tuple(graph.betti)}

hive.add_worker("structure", analyze, capability="analyze", specialties=["homology"])
answer = hive.invoke("structure", {"sources": [0, 1, 2], "targets": [1, 2, 0]})
print(answer)
```

`attach` connects to an existing OpenAI compatible endpoint. `spawn` starts a
managed llama.cpp server for a model file. `rexgraph-bee` exposes the same
endpoint contract to a responder that polls for requests. These use two
transports: an in process callable or an HTTP endpoint.

```python
hive.attach("chat", "http://127.0.0.1:8080", role="worker", specialties=["homology"])
reply = hive.dispatch("Explain the first Betti number of a cycle.")
```

The hive records interactions in an activity complex and uses declared
specialties and interaction history for routing. `hive.status()` returns
membership and health. `get_hive()` supplies the process singleton; create
separate `Hive` instances when tool registrations must be independent.

## Local inference

Install or build llama.cpp separately and make `llama-server` available on
`PATH`, or configure an existing binary through the local runtime. The RexGraph
installer compiles RexGraph's Core extensions.

```sh
.venv/bin/rexgraph-local --help
.venv/bin/rexgraph-hive --help
```

A managed server uses a local GGUF model file. An attached service can use
llama.cpp, vLLM or another implementation of the endpoint contract. Model files,
GPU drivers and external inference runtimes have their own installation steps.

## Models and connectors

The [model builder](agent/models/README.md) assembles MLP, CNN, language and
hypergraph models from `rexgraph.nn` components. Core's
[cochain models](../rexgraph/flow/README.md) use cells as their parameter space.
Feature models and cochain models use different optimizers and checkpoint
formats, as described in their respective guides.

[Connectors](agent/connectors/README.md) read declared source structure and
construct a relational complex. Optional live drivers support SQL, MongoDB,
Neo4j and cloud warehouses. Inspect the available schemes before configuring a
source:

```sh
.venv/bin/rexgraph-connect list
```

## Lifecycle and command line tools

A setup is a named configuration for paths, models, backend and thread settings.
Lifecycle operations use that setup and record their runs.

| Command | Purpose |
| --- | --- |
| `rexgraph-config` | Inspect and manage setups |
| `rexgraph-ops` | Serve, train, ingest, build, deploy, test and benchmark |
| `rexgraph-run` | Run the document analysis pipeline |
| `rexgraph-models` | Build and train model archetypes |
| `rexgraph-connect` | List, validate and ingest data sources |
| `rexgraph-deploy` | Write a container deployment bundle |
| `rexgraph-auth` | Manage authentication and workspace members |
| `rexgraph-courier` | Transfer selected records between stores or peers |

These executables are in `.venv/bin`. Use `--help` for their arguments.
`make run FILES='document.pdf'` runs the pipeline from the checkout.
[Benchmarks](agent/benchmarks/README.md) document measurements and their limits.

## Integrations

Optional extras install each integration's dependencies:

| Extra | Interface |
| --- | --- |
| `trustgraph` | Knowledge graph imports, structural retrieval measurements and enrichment triples |
| `huggingface` | Transformer attention analysis |
| `langchain` | Structural analysis tools for LangChain agents |
| `langgraph` | State graph construction and analysis |
| `warehouse` | SQLAlchemy dialects for Snowflake, BigQuery, Redshift and Databricks |
| `objectstore-cloud` | S3, Google Cloud Storage and Azure filesystem drivers |
| `oidc` | OIDC authentication support |

Standalone triple analysis needs no TrustGraph service:

```python
from agent.integrations.trustgraph_adapter import TrustGraphAdapter, SimpleTriple

adapter = TrustGraphAdapter()
rex, metadata = adapter.from_triples([
    SimpleTriple("compound", "targets", "protein"),
    SimpleTriple("protein", "participates_in", "process"),
])
```

## Server and API

`make serve` starts the Agent application. `rcf-server` is its installed console
entry point. `rexgraph-serve` starts the separate model/OCR inference service.

Analysis and session routes use `/api`. Application services use `/api/v1`.
Native artifact and record transfer uses `/rex/v1` through `agent.client.RexClient`.
The server includes its UI assets and licensed React runtime for offline use.

## Authentication

A fresh server enables authentication and prints a bootstrap admin token once.
Members receive bearer tokens with roles in named workspaces. Workspace admins
manage members and operations requiring an admin role; users have the permissions granted
to their workspace role. The activity journal records the acting identity.

```sh
.venv/bin/rexgraph-auth network-init
.venv/bin/rexgraph-auth member add --name alice --role user
.venv/bin/rexgraph-auth member list --workspace default
```

For a shared deployment, configure TLS with `REXGRAPH_TLS_CERT` and
`REXGRAPH_TLS_KEY`, or use a trusted reverse proxy. `rexgraph-deploy` writes
Compose configuration, environment settings and volume declarations. Copy its
`.env.example` to `.env` and configure the deployment before starting it.

## Deployment

The default application binds to `127.0.0.1:8000`. Configure `RCF_HOST` and
`RCF_PORT` for another bind address or port. Rate limits apply per server worker;
a proxy can enforce limits across several workers.

`make serve-https` generates a local certificate when needed. Use supplied
certificate and key files for a deployment with a trusted certificate.
`make serve-dev` enables source reload for an editable development installation.

The compute layer selects CPU, CUDA, ROCm or MPS according to the requested
backend and available runtime. An explicit GPU request raises an error when
the device or required runtime is unavailable.

## External readers

Dataset adapters, source clients and ontology mappings belong in external
packages. A reader can return `agent.adapters.EdgeConstruction`, retaining
primary relation supports, source identity and structured cell attributes.
Use `agent.auto.build_rex_from_edges` to construct the native object.

For file dispatch, register the external reader explicitly:

```python
from agent.adapters.formats import register_reader, read, unregister_reader
from my_readers import load_records

register_reader("my_records", load_records, extensions=(".records",))
try:
    construction = read("input.records", source_release=selected_release)
finally:
    unregister_reader("my_records")
```

Register dataset readers before using them. The reader interprets its source
data, Core constructs the complex, RCQL queries it and RCDB stores it.

## Stored record transfer

RCDB record packets retain a selected payload version, metadata, validity and
source identity. `agent.client.RexClient` transfers those packets through
`/rex/v1`, and couriers use the same `rcdb.copy_record` contract for local and
remote destinations. Destination workspace and integrity policies apply at
publication. A sender ledger retains acknowledgements and supports explicit
reconciliation when a checked receipt is available.

See the [record transfer guide](RECORD_TRANSFER.md) for version selection,
concurrency, uncertain outcomes and ledger retention, and the
[RCDB guide](../rcdb/README.md#portable-record-versions) for the packet contract.

## License

Agent is distributed under the [Apache License 2.0](LICENSE).
