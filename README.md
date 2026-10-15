# RexGraph

RexGraph is a Python library for working with relational complexes, with compiled
Cython kernels for sparse storage and numerical computation. It represents a
relationship as one cell, including relationships with several participants,
and records boundaries between cell grades: vertices at grade 0, relations at
grade 1, faces at grade 2 and higher cells at subsequent grades. This allows
an application to retain the structure of an observation, analyze flows through
it, and store the result while preserving relationships with several participants.

The library provides exact integer and rational calculations for declared
coefficients, ranks and supported operator actions, together with numerical
Hodge decomposition, propagation and model components. The repository also
contains a query language, versioned storage, a browser application and tools
for data ingestion and worker orchestration. The Core library can be
installed independently of these application packages.

## Installation

On Linux or macOS, clone the repository and run:

```sh
git clone https://github.com/sirtrott1/rexgraph.git
cd rexgraph
make install
make serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the Agent application.
The server prints an admin token on first startup; use that token to sign in.
If Make is unavailable, the same installation and launch sequence is:

```sh
sh install.sh
.venv/bin/python -I agent/run.py --no-browser
```

The installer creates `.venv` in the checkout, obtains Python, a C compiler and
OpenBLAS from `conda-forge`, and installs all five repository packages. It uses
an existing micromamba, mamba or conda executable, or downloads micromamba to
`~/.local/bin`. A new installation needs internet access, curl and tar, with Git
to obtain the source. Windows users can run these commands in WSL.

The standard installation includes the browser application, schema tools,
common database connectors, PDF readers, Python OCR bindings and training data
export. Installation compiles the Core extensions and checks numerical and exact
operations, packaged UI assets and dependency consistency before reporting
completion. Optional models, external data and service credentials are
configured separately according to the features an application uses.

### Installation profiles

Every platform profile installs Core, RCDB, RCQL, System and Agent. Choose the
profile for the work you plan to run:

| Command | Purpose |
| --- | --- |
| `make install` | Browser application, analysis and common data tools |
| `make install-core` | Mathematical library without the application packages |
| `make install-server` | Platform with the Agent server dependencies |
| `make install-ml` | Standard profile with PyTorch and Hugging Face model tools |
| `make install-all` | Declared application extras, including cloud integrations and numerical baselines |
| `make install-dev` | Editable source installation with the standard test dependencies |
| `make install-dev-full` | Development installation with the optional CPU model, TensorFlow, molecular, analytics and integration test dependencies |

The script accepts these profiles as `--profile core`, `server`, `standard`,
`ml`, `all`, `dev` or `dev-full`. Additional extras can be selected without
changing the main profile:

```sh
sh install.sh --profile core
sh install.sh --extras analytics,molecules
sh install.sh --extras trustgraph
```

Core, RCQL and base RCDB records work without SciPy. Agent's analysis pipeline
and Core's numerical neural components include it in their dependency profiles.
[DEPENDENCIES.md](DEPENDENCIES.md) describes the optional packages and
what each profile enables.

### System Python and custom locations

The pip installer uses Python 3.10 through 3.13 and Python's virtual environment
module. It requires Python headers, a C compiler and BLAS/LAPACK on the host.
For Debian or Ubuntu, install the build requirements before running it:

```sh
sudo apt install python3-venv python3-dev build-essential libopenblas-dev pkg-config
sh install.sh --installer pip
```

On macOS, this path requires Python and the Xcode command line tools; the build
uses Apple Accelerate for BLAS/LAPACK. Select a different interpreter or
installation directory when needed:

```sh
sh install.sh --installer pip --python python3.12 --venv /path/to/environment
make serve VENV_DIR=/path/to/environment
```

### Optional runtimes

Scanned documents require the Tesseract executable in addition to the Python
OCR bindings. Install it with `sudo apt install tesseract-ocr` on Debian or
Ubuntu, or `brew install tesseract` on macOS. Text files and PDFs with a text
layer can be read without a GPU OCR service.

Install GPU support with:

```sh
make install-gpu
```

The installer selects CUDA for NVIDIA GPUs or ROCm for AMD GPUs. AMD setup also
installs the ROCm development tools and builds the custom HIP kernels for the
detected GPU targets. ML profiles select CPU PyTorch on machines without a GPU.
GPU drivers must already be installed and accessible to the terminal running setup.

Vulkan sparse products, propagation and conjugate gradient solves are available
on hardware with float64 shader support. Sparse coefficients remain on the
device between products; vector updates run on the CPU. Select a backend when
needed, including Vulkan on AMD or NVIDIA hardware:

```sh
make install-gpu GPU_BACKEND=vulkan
make verify-gpu
make verify-gpu GPU_BACKEND=vulkan
```

`make verify-gpu` compares calculations with CPU results and checks required
native kernels. `make gpu-status` reports device access and runtime discovery.
Use `GPU_BACKEND=cpu` with `make install-ml` to select CPU PyTorch explicitly.
The script accepts the same selection as `sh install.sh --profile ml --gpu auto`.
`TORCH_INDEX_URL`, `ROCM_INDEX_URL`, `HIPCC` and `HIP_ARCH` override package and
compiler selection for an existing toolchain. `NATIVE_HIP=disabled` omits the
custom HIP library. Builds without device access can set `GPU_VERIFY=0` and
run `make verify-gpu` later in a terminal with GPU access.

Local language model inference uses a separately installed `llama-server`
executable and a local GGUF model file.
[Agent](agent/README.md#local-inference) documents how to attach an existing
service or start a managed one.

## Getting started with the library

Run `make python` to open the installed interpreter. An ordinary cycle can be
constructed from its source and target indices:

```python
from rexgraph import RexGraph

network = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
print(network.nV, network.nE)       # 3 vertices and 3 relations
print(tuple(network.betti))        # (1, 1, 0)
```

The Betti numbers count connected components, independent unfilled cycles and
closed two dimensional surfaces in this example. The cycle has one connected
component and one independent loop. Adding an explicit face that fills the
loop reduces the first Betti number to zero:

```python
import numpy as np

filled = RexGraph.from_simplicial(
    sources=np.array([0, 1, 2], dtype=np.int32),
    targets=np.array([1, 2, 0], dtype=np.int32),
    triangles=np.array([[0, 1, 2]], dtype=np.int32),
)
print(tuple(filled.betti))          # (1, 0, 0)
print(filled.chain_valid)          # True
```

A relation with several participants also occupies one cell. Its participant
shares and metric weight can be declared as fractions, so those inputs remain
exact:

```python
from fractions import Fraction
from rexgraph import Relations, VertexTable

relations = Relations.from_supports(
    [[0, 1, 2]],
    vertices=VertexTable(("head", "left", "right")),
    shares=[0, Fraction(1, 3), Fraction(2, 3)],
    weights=[Fraction(2, 5)],
    relation_ids=["measurement"],
)
branch = RexGraph.from_relations(relations)
print(branch.nV, branch.nE)         # 3 vertices and 1 relation
```

Use `.venv/bin/python -I example.py` to run a saved example against the installed
packages. The isolated interpreter prevents the checkout's source directories
from shadowing the compiled installation. Editable installations are provided
by the development profiles.

## Loading and processing data

RexGraph separates reading a file from declaring what its fields represent.
For an edge table, declare which columns identify participants and how numeric
weights are interpreted. Save the following table as `relations.csv`:

```csv
source,target,weight
A,B,1.0
B,C,0.5
C,A,0.25
```

The declaration below treats the participant labels as strings and the weights
as exact decimal values. `construct` creates one relation per row; a weight of
`0.5` is retained as the fraction `1/2`.

```python
from rexgraph import RelationSpec, NumberRule, ValueRules
from rexgraph.construction import construct
from rexgraph.io.readers import read_batches
from rexgraph.io.records import RecordField, RecordSchema

schema = RecordSchema((
    RecordField("source", "string"),
    RecordField("target", "string"),
    RecordField("weight", "number", rules=ValueRules(NumberRule.DECIMAL_EXACT)),
))
records = read_batches("relations.csv", reader="csv", schema=schema)
relations = construct(records, RelationSpec("pair", ("source", "target"), weight="weight"))
network = RexGraph.from_relations(relations)
print(network.relations.weight.values()[1])  # 1/2
```

The declared readers also support JSON, JSONL, Arrow IPC, Parquet, SQLite and
XLSX. A reusable dataset declaration can retain the schema, relation mapping
and source options for later reads. Native graph files retain the complete
complex, while a table describes the fields selected for import or export.
[I/O documentation](rexgraph/io/README.md) covers the format contracts and
streaming interfaces.

Agent provides `auto_rex` for common CSV, JSON, array, DataFrame and text inputs.
It chooses an adapter using the input type and recognizable fields. For
numerical tables or text, the selected adapter determines which relationships
are constructed; the source table alone does not specify that interpretation.
Use a declared schema and mapping when the application must control it.

## Analysis, queries and storage

Hodge decomposition separates a signal on relations into a gradient component
from vertex potentials, a curl component supported by filled boundaries, and a
harmonic component in the Hodge kernel. For a declared complex, the components
reconstruct the input signal:

```python
signal = np.array([1.0, 0.0, -1.0])
gradient, curl, harmonic = network.hodge(signal)
assert np.allclose(gradient + curl + harmonic, signal)
```

Core also provides exact homology, weighted operators, Green responses,
persistence, temporal identities and comparisons between complexes. Integer
and rational calculations preserve their coefficients; numerical solvers use
the tolerance and accuracy checks specified by their APIs. The
[Core guide](rexgraph/README.md) provides the API reference and examples.

RCQL evaluates typed queries against explicitly supplied data. This query
returns the relation count and the first Betti number:

```python
from rcql import run

result = run("FROM $network RETURN COUNT(CELLS(1)), BETTI(1)", sources={"network": network})
print(result.values)              # (3, 1)
```

RCDB stores complexes and typed records with versions and transaction history.
A local store can save the complex and read back its selected version:

```python
from contextlib import closing
from rcdb import LocalStore

with closing(LocalStore("records")) as store:
    record = store.put("network", network, analytics=False)
    restored = store.get_version("network", record.version)
    print(restored.nE)            # 3
```

For a portable graph file, use `save_rcbd("network.rcbd", network)` and
`load_rcbd("network.rcbd")` from `rexgraph.io`. RCBD, safetensors, HDF5 and Zarr
retain native state; CSV and JSON imports follow their declared source schemas.

## Applications and capabilities

The Agent browser application exposes uploads, analysis, stored records,
connectors and configured model tools. Its Python API can run the same analysis
pipeline and register workers with capabilities such as prediction, embedding,
analysis or transformation. Language model workers can be attached through an
OpenAI compatible endpoint. A worker can also be an ordinary Python callable.

| Package or feature | What it provides | Guide |
| --- | --- | --- |
| Core | Construction, homology, Hodge analysis, fields and propagation | [Core](rexgraph/README.md) |
| RCQL | Typed queries, plans, selections and mutations | [RCQL](rcql/README.md) |
| RCDB | Versioned storage, record queries and transfer | [RCDB](rcdb/README.md) |
| System | Browser inspection and query results over registered sources | [System](system/README.md) |
| Agent | Application server, analysis pipelines and worker orchestration | [Agent](agent/README.md) |
| Connectors | Structural reads from databases and other configured sources | [Connectors](agent/agent/connectors/README.md) |
| Models | PyTorch cochain models and assembled feature models | [Core models](rexgraph/flow/README.md), [Agent models](agent/agent/models/README.md) |

System provides a separate browser interface for inspecting a saved complex
and running RCQL queries:

```sh
make system ARGS='--source main=network.rcbd'
```

Open [http://127.0.0.1:8010](http://127.0.0.1:8010). The
[System guide](system/README.md) describes registered datasets and file catalogs.

### Configuration and storage

Run `make doctor` to list optional Python dependencies and external executables.
This reports installation status; `make verify`
checks compiled operators, packaged UI assets and dependency consistency.

Set `REXGRAPH_CONFIG_DIR`, `REXGRAPH_CACHE_DIR`, `REXGRAPH_MODELS_DIR` and
`REXGRAPH_SESSION_DIR` before starting Agent to choose its storage directories.
File paths and database URLs passed to the APIs select application data sources
and destinations. The [Agent guide](agent/README.md#configuration-and-cache)
describes directory defaults, overrides and cache controls.

## Development and testing

```sh
make install-dev
make verify
make test
```

Use `make install-dev-full` to exercise the optional CPU feature tests as well.
Tests use synthetic inputs in temporary storage.
GPU tests require compatible devices and runtimes. Read skipped tests alongside
their reported reasons:

```sh
make test TEST_ARGS='-rs'
```

The package targets are `test-rexgraph`, `test-agent`, `test-rcdb`, `test-rcql`
and `test-system`. Pass pytest options through `TEST_ARGS`, for example
`make test-rexgraph TEST_ARGS='-k boundary'`. The default compiler concurrency is
`BUILD_JOBS=2`; increase it only when the build machine has enough memory.

Regular installations need to be reinstalled after changing a checkout.
Editable installations use current Python source and rebuild changed Cython
extensions on import. `make hpc-setup` uses the shared installer and generates
SLURM templates; configure the cluster account, partition and paths before
submitting a job.

`make clean` removes Python bytecode and pytest caches from the checkout.
It preserves environments, editable build trees, package metadata and downloaded
models. Use `REXGRAPH_NO_CACHE=1` to run without reusing Agent analysis results.

## Documentation and support

The package guides above describe their APIs and data contracts. For lower
level implementations and measurements, see the
[compiled kernel reference](rexgraph/core/REFERENCE.md),
[I/O reference](rexgraph/io/REFERENCE.md) and
[benchmark guide](agent/agent/benchmarks/README.md).
Report reproducible failures or request features through
[GitHub Issues](https://github.com/sirtrott1/rexgraph/issues), including the
installation profile, interpreter version and relevant error output.

## Acknowledgements

RexGraph integrations build on these open source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp): Local LLM inference; the agent's native runtime for running quantized models on CPU and GPU.
- [TrustGraph](https://github.com/trustgraph-ai/trustgraph): Knowledge graph construction and management for RAG applications.
- [Hugging Face Transformers](https://github.com/huggingface/transformers): Pre trained transformer models and inference.
- [LangChain](https://github.com/langchain-ai/langchain): Framework for building applications with language models.
- [LangGraph](https://github.com/langchain-ai/langgraph): Stateful agent orchestration with cyclic computation graphs.
- [vLLM](https://github.com/vllm-project/vllm): High throughput LLM serving.

## License

RexGraph is distributed under the [Apache License 2.0](LICENSE).
