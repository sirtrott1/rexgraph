# Dependencies and installation profiles

RexGraph separates the Core library from its applications and optional
integrations. The base Core package requires NumPy and threadpoolctl, while its
compiled kernels link to BLAS/LAPACK and use OpenMP when enabled. The installer
provides the compiler and native libraries through conda, or uses the host
compiler and libraries with the pip installation path.

`make install` installs the standard application profile. `make install-core`
installs the library by itself, and `make install-ml` adds the numerical model
components. `make install-dev` supplies the standard development dependencies;
`make install-dev-full` adds the optional CPU features needed by the broader
test suites. The [README](README.md#installation) gives the installation commands.

## Package dependencies

| Package | Base requirements | Optional features |
| --- | --- | --- |
| Core | NumPy, threadpoolctl, compiled BLAS/LAPACK linkage | File formats, signing, SciPy exports, models, TensorFlow and molecular input |
| RCDB | Core, NumPy, safetensors | SQLAlchemy stores, fsspec stores, cryptography and older SciPy search APIs |
| RCQL | Core, NumPy | Supplied RCDB sources; SymPy for development reference tests |
| System | Core, RCQL, FastAPI, uvicorn | Registered RCDB sources |
| Agent | Core, RCDB, NumPy, SciPy, pandas | Server, connectors, OCR, model tools, analytics and integrations |

Core, RCQL and base RCDB record operations run without SciPy. Exact ranks,
boundary actions, native records and declared coefficient storage use the Core
implementations. Explicit SciPy matrix exports and some older numerical analysis
APIs require the `scipy` extra. An extra supplies dependencies; requesting an
operator or algorithm remains a separate API choice.

Core requires NumPy 2. Optional numerical and file adapters use compatible
dependency floors: [SciPy 1.13](https://docs.scipy.org/doc/scipy-1.13.0/release/1.13.0-notes.html),
[pandas 2.2.2](https://pandas.pydata.org/docs/whatsnew/v2.2.2.html),
[h5py 3.11](https://docs.h5py.org/en/stable/whatsnew/3.11.html) and
[Arrow 16](https://arrow.apache.org/blog/2024/04/20/16.0.0-release/).
These requirements prevent older binary adapters from being retained when
installing into an existing environment.

## Core extras

| Extra | Dependencies and purpose |
| --- | --- |
| `io` | safetensors, Zarr, HDF5, Arrow/Parquet, SQLAlchemy, pandas and XLSX readers |
| `security` | cryptography for encryption and signatures |
| `scipy`, `oracles` | SciPy compatibility exports and reference comparisons |
| `nn` | PyTorch and SciPy for the numerical neural and cochain model components |
| `tf` | TensorFlow for the TensorFlow relational layer |
| `molecules` | RDKit for molecular syntax and source interpretation |
| `dev` | pytest, lint/type tools, build tools and SciPy reference tests |
| `all` | `io` and `security` |

Core's `all` extra includes file formats and signing. Install model dependencies
and device runtimes through their corresponding extras and installation profiles. The native RCBD container uses
the base package, while HDF5, Zarr, Arrow, Parquet and XLSX require their format
dependencies.

## Agent extras

The standard profile combines the server, schema, connectors, OCR bindings,
training export and YAML configuration extras. It does not install a model
runtime or model weights.

| Extra | Dependencies and purpose |
| --- | --- |
| `ml` | Core neural components and Hugging Face training/analysis tools |
| `molecules` | Core's RDKit adapter |
| `analytics` | DuckDB and Polars for queries and table views over stored signatures |
| `baselines` | scikit learn, LightGBM and XGBoost for numerical comparisons over stored features |
| `langchain`, `langgraph`, `trustgraph` | The selected framework integration |
| `warehouse` | SQLAlchemy dialects for Snowflake, BigQuery, Redshift and Databricks |
| `objectstore-cloud` | S3, Google Cloud Storage and Azure filesystem drivers |
| `oidc` | OIDC support |
| `dev` | Standard server tests and test tooling |
| `test-full` | Standard tests plus CPU models, TensorFlow, molecular input, analytics and LangChain/LangGraph |
| `all` | Declared runtime application extras, including cloud and model integrations |

Select additional extras through the shared installer:

```sh
sh install.sh --extras analytics,molecules
sh install.sh --extras warehouse
```

Live database connections also need credentials and any driver required by the
chosen database URL. fsspec's base dependency supports local and memory object
stores; remote providers need their corresponding cloud driver. A generic remote
filesystem does not, by itself, establish RCDB's native atomic publication
contract.

## External tools and data

Tesseract is an executable installed through the host package manager. English
OCR also requires `eng.traineddata`; `tesseract --list-langs` must list `eng`.
The OCR setup command includes the English language package on Debian, Fedora
and Arch Linux. The Python OCR extra supplies its bindings and the PDF reader.
Local language model
inference uses `llama-server` and a selected GGUF file. GPU execution requires
compatible runtime libraries and accessible device nodes; installing PyTorch
alone does not supply a driver or make a hidden device accessible.

`make install-gpu` selects a CUDA wheel index from the NVIDIA driver and device
architecture, or installs the AMD ROCm Python SDK and device packages. AMD setup
builds the custom HIP kernels for all detected targets. The defaults use official
[PyTorch packages](https://pytorch.org/get-started/locally/) and the
[AMD stable package index](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).
`TORCH_INDEX_URL`, `ROCM_INDEX_URL`, `HIPCC` and `HIP_ARCH` provide explicit
overrides. `NATIVE_HIP=disabled` omits native HIP compilation.

Vulkan uses the optional `vulkan` Python bindings and the host Vulkan loader and
driver. The package includes the compiled sparse shader and its GLSL source, so
users do not need a shader compiler. Float64 shader support is required. The
Vulkan graph path uses GPU sparse products with CPU vector updates. PyTorch
training requires a CUDA or ROCm build for GPU execution. `make verify-gpu GPU_BACKEND=vulkan`
executes products, propagation and a solver residual check on the Vulkan device.

The separate OCR setup command accepts `TORCH_INDEX_URL` when selecting its
inference dependencies. Those dependencies must match the inference server.

ROCm inference setup accepts `VLLM_WHEEL_URL` for a selected vLLM wheel. Generic
vLLM installation can resolve a CUDA build on an incompatible Python or ROCm
combination. Select a wheel or use an existing platform installation according
to the [vLLM installation documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

`make doctor` reports installed optional packages and external executables.
`make verify` checks the compiled library and packaged
applications. Use `make doctor ARGS=--json` for a machine readable inventory.

## Test coverage and skipped cases

The standard development profile exercises the base mathematical, storage,
query and application contracts. Tests for optional model and molecular features
are collected when their dependencies are installed. Use the full development
profile to supply those Python dependencies:

```sh
make install-dev-full
make test TEST_ARGS='-rs'
```

TensorFlow comparisons run in separate CPU processes to keep TensorFlow and
PyTorch runtime initialization separate from the other tests.

Runtime thread controls preserve the original width of each loaded BLAS
library. Overlapping calls use the smallest active limit. OpenBLAS 0.3.28
with pthreads is restricted to one thread during these calls because resizing
that runtime can crash parallel triangular solves. Other BLAS libraries retain
their selected widths. [OpenBLAS 0.3.29](https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.29)
includes a fix for a null pointer failure in threaded builds.

GPU, multiple GPU and native HIP tests depend on actual hardware and a compatible
compiled runtime. Regression tests generate graph and document inputs in temporary
storage and isolate user configuration, caches, models and sessions. Tests for
optional file formats skip at the format boundary when the corresponding extra
is absent. Use the reported test name and skip reason to identify a missing device
or Python dependency.

For a Core installation that must exclude SciPy, the dedicated smoke test checks
native operations and deliberate refusal of compatibility exports:

```sh
.venv/bin/python -I scripts/smoke_no_scipy.py --require-absent
```
