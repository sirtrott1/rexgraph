# RexGraph

Relational complex analysis with Cython-accelerated internals.

RexGraph implements the Relational Complex Framework (RCF): a framework whose
primary object is the relational complex, a graded cell complex where edges are
primary and vertices are derived from edge boundaries. Signed boundary operators
B1 (vertices from edges) and B2 (edges from faces) satisfy the chain condition
B1 B2 = 0, and typed Laplacians decompose structure into topological, geometric,
frustration, and copath (coparticipation) channels. Structural character places
every edge and vertex on a simplex. Hodge theory, persistent homology, void
spectral theory, Dirac operators, fiber bundles, interfacing vectors,
cross-complex comparison, and quotient complexes are all computed through a
single `RexGraph` object backed by compiled Cython modules.

Boundary columns carry arbitrary signed arity: witness edges (one endpoint),
ordinary edges (two), and branching hyperedges (three or more) are all
first-class cells. Grade is a dimensional grading, not an arity constraint, so
hypergraphs, simplicial complexes, and cell complexes are the same object at
different gradings.

The computation is matrix-free and eigensolve-free on every live path. Betti
numbers come from exact integer rank (union-find and rational column reduction,
never a dense spectrum); the harmonic plane from a combinatorial cycle basis and
a low-rank projector; the heat, wave, Schrodinger, and field propagators from
Chebyshev sparse mat-vecs; Green's functions and the structural character from
block conjugate-gradient resolvents and LSQR pseudoinverses. Every sparse path is
checked against a retained dense oracle to machine precision. One compute layer
dispatches the same operators across CPU, OpenMP, CUDA, ROCm, and Apple MPS
(size-gated, with multi-GPU column tiling, multi-core fan-out, and a CPU fallback),
so the library runs unchanged on one core, thirty-two cores, an integrated GPU,
or many GPUs. On top of this `rexgraph.nn` adds a differentiable substrate: the
GreensCochain optimizer (reached through `make_optimizer("auto")`), relational
(propagator) attention, and Green-resolvent blocks, all autograd- and GPU-ready.

The `agent` package is a full platform over the core: a FastAPI server and web
UI, a multi-agent hive that orchestrates language models and custom workers as a
relational complex, a setups-driven lifecycle (serve, train, build, deploy,
finetune, ingest), local LLM inference through llama.cpp, connectors that turn
any database into a relational complex, and a model builder for custom ML
architectures. Integrations (TrustGraph knowledge cores and rex-RAG,
HuggingFace, LangChain, LangGraph) are one part of that ecosystem. It
auto-detects input types (triples, CSV, JSON, text, numpy arrays, pandas
DataFrames) and runs a unified analysis pipeline with harmonic mode diagnostics
(dim_H, frustration, coparticipation, health ratio, sigma-asymmetry).


## Install

Only git and curl are required on the host. The compilers, OpenBLAS, and Python
come from a conda-forge environment, so the build does not depend on system
BLAS packages whose names differ per distro.

### One command (recommended)

```bash
sh install.sh
```

Run from the repo root. It detects the OS, package manager, and conda frontend
(reusing mamba/micromamba/conda, or bootstrapping micromamba), creates the
`rexgraph` environment from `environment.yml`, builds the Cython core from the
repo root, and installs the agent. Override defaults through environment
variables:

```bash
EXTRAS=standard sh install.sh              # agent profile to install (see the table below)
RUN_CORE_TESTS=1 sh install.sh             # build, then run the core tests
NATIVE=1 sh install.sh                      # also build the native llama.cpp bindings
```

This also works on an HPC login node: it bootstraps micromamba into your home
directory (no miniforge or module loads) and builds in-environment.

### Manual (conda/mamba)

```bash
mamba env create -f environment.yml
mamba activate rexgraph
pip install -e . --no-build-isolation      # rexgraph Cython core, from the repo root
pip install -e ./agent                     # agent + integrations
```

The order matters, and will until the core is published. The agent requires
`rexgraph>=0.5.0`, which pip resolves from PyPI unless it is already installed, and
the core is not there yet. Installing the agent first fails with `No matching
distribution found for rexgraph>=0.5.0`, which reads like a missing release rather
than a missing step.

Minimal (core only, no I/O deps):

```bash
mamba env create -f environment-minimal.yml
mamba activate rexgraph
pip install -e . --no-build-isolation
```

### Manual (pip + system BLAS)

```bash
# Debian/Ubuntu
sudo apt install libopenblas-dev pkg-config

pip install .                                 # core only
pip install ".[io]"                           # + zarr, h5py, pyarrow, sqlalchemy, pandas
pip install ".[all]"                          # currently the same as [io]
pip install -e ".[dev]" --no-build-isolation  # editable dev install

pip install ./agent                           # CLI + integrations, no server
pip install "./agent[server]"                 # + web UI and API (light)
pip install "./agent[standard]"               # full local deployment (no torch, no cloud drivers)
pip install "./agent[ml]"                      # + LoRA fine-tuning and HuggingFace (torch)
pip install "./agent[all]"                     # everything that runs on CPU (large)
```

### What to install

The core (`.`) is only the relational complex math. Everything else is opt-in.

Core extras, from the repo root (e.g. `pip install ".[io]"`):

| Extra | Adds | Weight |
|-------|------|--------|
| `io` | zarr, h5py, arrow/parquet, SQL loaders | light |
| `all` | `io`, and nothing more today | light |
| `nn` | torch ML substrate (GreensCochain, propagators, relational attention) | heavy (torch) |
| `cuda` | GPU kernels (cupy) | heavy (CUDA toolkit) |
| `dev` | test, lint, type-check, build backend | contributors |

Agent profiles, from the repo root (e.g. `pip install "./agent[standard]"`):

| Profile | For | Weight |
|---------|-----|--------|
| `server` | web UI and API only | light |
| `standard` | full local deployment: UI, schema tools, connectors, OCR, training export, YAML. No torch, no cloud drivers | medium |
| `integrations` | LangChain, LangGraph, TrustGraph | medium |
| `ml` | LoRA fine-tuning and HuggingFace model analysis | heavy (torch) |
| `warehouse` | Snowflake / BigQuery / Redshift / Databricks drivers | heavy |
| `all` | everything that runs on CPU | large |

The granular extras behind the profiles (`schema`, `connectors`, `ocr`, `ocr-paddle`, `training`, `finetune`, `huggingface`, `langchain`, `langgraph`, `trustgraph`, `mistral`, `oidc`, `vllm`) can be combined directly, e.g. `pip install "./agent[server,ocr,trustgraph]"`. GPU-only extras (`vllm`, `ocr-got`) are never pulled by a profile.

### Verify

```bash
python -c "from rexgraph.graph import RexGraph; print('rexgraph OK')"
python -c "from agent.pipeline import AnalysisPipeline; print('agent OK')"
python -m pytest rexgraph/tests/
python -m pytest agent/tests/
```


## Agent Platform

Most work with RexGraph runs through the `agent` platform: the full application
layer over the core. It is a FastAPI server and web UI, a multi-agent hive that
orchestrates language models and custom workers as a relational complex, a
setups-driven lifecycle (serve, train, build, deploy, finetune, ingest), local
LLM inference through llama.cpp, connectors that turn any database into a
relational complex, a builder for custom ML architectures, and the integrations
(TrustGraph knowledge cores and rex-RAG, HuggingFace, LangChain, LangGraph).

```bash
pip install "./agent[standard]"     # full local deployment
rcf-server                          # server + web UI on http://127.0.0.1:8000
```

```python
# Orchestrate models and custom workers as a relational complex
from agent.hive import Hive
hive = Hive()
hive.attach("worker", "http://127.0.0.1:8080", role="worker", specialties=["hodge"])
hive.dispatch("What is the first Betti number of a torus?")   # routes + asks the best bee

# Turn a knowledge graph into a relational complex and score a retrieval (rex-RAG)
from agent.integrations.trustgraph_adapter import TrustGraphAdapter, SimpleTriple
adapter = TrustGraphAdapter()
rex, meta = adapter.from_triples([SimpleTriple("Drug_A", "treats", "Disease_X"),
                                  SimpleTriple("Drug_A", "targets", "Protein_Y")])
adapter.subgraph_confidence(rex, [0, 1, 2])   # structural trust of a retrieved subgraph
```

Full platform documentation: **[agent/README.md](agent/README.md)**, covering the hive,
the lifecycle, local inference, custom ML architectures, connectors, and the
integrations in depth.

The rest of this README covers the `rexgraph` core the platform is built on.


## Quick Start

### Core relational complex

```python
from rexgraph.graph import RexGraph
import numpy as np

rex = RexGraph.from_simplicial(
    sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
    targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
    triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32))

rex.betti                          # (1, 0, 1)
rex.chain_valid                    # True (B1 @ B2 = 0)
rex.euler_characteristic           # 2

# Relational Laplacian: RL = sum of trace-normalized typed Laplacians
RL = rex.relational_laplacian      # f64[nE, nE], tr(RL) = nhats
rex.nhats                          # 3 or 4 depending on copath availability

# Structural character: where each edge sits on the simplex
chi = rex.structural_character     # f64[nE, nhats], rows sum to 1
phi = rex.vertex_character         # f64[nV, nhats]
kappa = rex.coherence              # f64[nV] in [0, 1]

# Per-channel mixing times and anisotropy
times = rex.per_channel_mixing_times  # f64[nhats], ln(nE) / lambda_2 per hat

# Hodge decomposition
flow = np.ones(rex.nE, dtype=np.float64)
grad, curl, harm = rex.hodge(flow)
result = rex.hodge_full(flow)      # dict with energies, fractions, divergence, face curl

# Interfacing vector: map source vertices through typed response operators
iv_result = rex.interfacing_vector(
    target_indices=np.array([0, 1], dtype=np.int32),
    target_weights=np.array([1.0, 1.0]),
    target_signal=flow)
iv_result['sphere_pos']            # f64[4], unit sphere position (T, G, F, Sch)
iv_result['confidence']            # {'flag': 'CONFIDENT', 'reasons': []}

# Per-channel signal decomposition
psc = rex.primal_signal_character(flow)  # f64[nhats], energy fractions summing to 1
dipole = rex.face_void_dipole(flow)      # face vs void affinity

# Cross-complex comparison
from rexgraph.graph import cross_complex_bridge
bridge = cross_complex_bridge(rex_A, rex_B, labels_A, labels_B)
bridge['kappa']['correlation']     # Pearson correlation of coherence at shared vertices

# Typed face selection: same-type triangles become faces, cross-type become voids
rex_typed = rex.typed_face_selection(edge_type_labels)

# Character-based filtration: remove edges by decreasing chi, track Betti
filt = rex.quotient_filtration(channel=0, n_steps=20)
filt['transition_index']           # step with largest beta_1 drop

# Linkage complex from fiber bundle similarity
rex_link = rex.linkage_complex(sfb_threshold=0.85)

# Spectral
evals = rex.eigenvalues_L0         # vertex Laplacian spectrum
fiedler = rex.fiedler_vector_L0    # algebraic connectivity eigenvector
alpha_G, alpha_T = rex.coupling_constants

# Dirac operator on the full graded complex (nV + nE + nF) x (nV + nE + nF)
D = rex.dirac_operator
d_evals = rex.dirac_eigenvalues

# Field operator on (E, F): coupled edge-face dynamics
M, g, is_psd = rex.field_operator
f_evals, f_evecs, freqs = rex.field_eigen
modes = rex.classify_modes()       # edge / face / resonant per mode

# Fiber bundle similarity between vertices
S_phi = rex.phi_similarity         # f64[nV, nV] from vertex character
S_fb = rex.fiber_similarity        # f64[nV, nV] combined fiber + star

# Void spectral theory
vc = rex.void_complex              # Bvoid, Lvoid, eta, chi_void, fills_beta

# Persistent homology
fv, fe, ff = rex.filtration(kind="dimension")
dgm = rex.persistence(fv, fe, ff)
barcodes = rex.persistence_barcodes(dgm, dim=1)
H = rex.persistence_entropy(barcodes)

# Signal propagation
f_E = np.zeros(rex.nE, dtype=np.float64)
f_E[0] = 1.0
result = rex.analyze_perturbation(f_E, times=np.linspace(0, 5, 50))

# Quotient complex
sub, v_map, e_map = rex.subgraph(np.array([1,1,1,0,0,0], dtype=bool))
info = rex.quotient_analysis(v_mask, e_mask, f_mask, signal=flow)

# Query engine
imputed = rex.impute(observed_signal, observed_mask)
diag = rex.explain(dim=1, idx=3)
score = rex.propagate(source_signal, target_signal)

# Dynamics
diffused = rex.evolve_markov(flow, dim=1, t=1.0)
psi = rex.wave_state(dim=1)
outcome, collapsed = rex.measure(psi, dim=1)

# Dashboard
from rexgraph.analysis import analyze
data = analyze(rex, vertex_labels=["A","B","C","D"])
```

### Eigen-free / sparse computation

Every quantity above is computed with no dense eigensolve. The sparse modules
expose the same operators directly, for complexes too large to densify.

```python
from rexgraph import harmonic_sparse, scale_propagator as sp, sparse_character

rex.betti                                  # exact integer rank, never a spectrum

# Combinatorial harmonic basis of ker(L1): spanning-tree cycle basis, no eigh
H = harmonic_sparse.harmonic_basis(rex)    # sparse nE x dim_H

# Matrix-function propagators via Chebyshev sparse mat-vecs
heat = sp.heat_apply(rex.L1_sparse, flow, t=0.5)          # e^{-tL} flow
psi  = sp.schrodinger_apply(rex.L1_sparse, flow, t=0.5)   # e^{-iLt} flow (complex)

# Green's diagonal via block conjugate gradient; character via block-CG + LSQR
diag = sp.greens_diagonal(rex._rl4_sparse)                # diag(RL^-1)
char = sparse_character.compute_sparse_character(rex)     # chi / phi / kappa, scale-free
```

### Compute backends (CPU / GPU / multi-core / multi-GPU)

```python
from rexgraph import compute

compute.recommended_backend()              # 'cpu' | 'openmp' | 'cuda' | 'rocm' | 'mps'
compute.gpu_count()                        # usable GPUs (REXGRAPH_MAX_GPUS-capped)
compute.set_default_backend("rocm")        # or set REXGRAPH_BACKEND

# The propagators take an explicit backend; large column blocks tile across GPUs,
# and the CPU fan-out caps inner BLAS so it does not oversubscribe cores.
heat = sp.heat_apply(rex.L1_sparse, flow, t=0.5, backend="cuda")
```

### Differentiable substrate (`rexgraph.nn`)

```python
import rexgraph.nn as rnn

attn, name = rnn.build_attention(None, d=64, n_head=4)  # relational (propagator) attention
opt, label = rnn.make_optimizer("auto", attn, attn.parameters())
# "auto" is the router: GreensCochain for a relational-native model whose parameters are
# cochains on a complex, plain Adam for a standard feature-space model like this one.
# HodgeAdam / HodgeSGD remain reachable by name (method="hodge"/"hodgesgd") as back-compat.
dev = rnn.pick_device("auto")              # resolves through the compute backend
```

### Auto-detect any input format

```python
from agent.auto import auto_rex

rex = auto_rex("edges.csv")                # CSV file
rex = auto_rex("graph.json")               # JSON file
rex = auto_rex(adjacency_matrix)           # numpy array
rex = auto_rex(dataframe)                  # pandas DataFrame
rex = auto_rex("Proteins regulate genes.") # raw text
```

### Structural health diagnostic

```python
from agent.pipeline import AnalysisPipeline

pipe = AnalysisPipeline(rex)
result = pipe.run(depth="standard")
hodge = result["hodge"]

hodge["dim_H"]                     # oscillatory modes (beta_1)
hodge["health_ratio"]              # frustration / coparticipation
hodge["frustration_total"]         # harmonic tension from topology
hodge["coparticipation_total"]     # harmonic support from geometry
hodge["sigma_asymmetry_per_edge"]  # rigid vs adaptive per edge
hodge["harmonic_modes"]            # top edges per mode
```


## Structural Diagnostics

The analysis pipeline computes per-edge and aggregate metrics from
the Hodge decomposition and the four-channel structure.

Every edge signal decomposes uniquely into three orthogonal components:

**Gradient**: hierarchical structure determined by B_1. Face-independent.
Does not change when faces are added or removed. Measures directed
chain strength.

**Curl**: closed feedback loops determined by B_2. Each face is a triple
of entities where all three pairwise relationships form a stable cycle.

**Harmonic**: conserved relational flow on unfilled cycles. Measures
unresolved structural tension.

Derived metrics:

**dim(H)**: number of independent oscillatory modes.

**Frustration**: harmonic content projected onto the topology channel.
Tension that originates from graph topology. Cannot be reduced by
adding faces.

**Coparticipation**: harmonic content projected onto the geometry
channel. Measures how well the current face structure supports
relational dynamics.

**Health ratio**: frustration / coparticipation. Above 1.0 means the
structure works against itself.

**Sigma-asymmetry**: per-edge measure of topological rigidity (positive)
vs responsiveness to face changes (negative).


## Input Formats

| Format | Input | Notes |
|--------|-------|-------|
| TrustGraph flow | `adapter.from_flow("default")` | connected mode |
| TrustGraph core | `adapter.analyze_core("id")` | loads and analyzes |
| Triple list | `adapter.from_triples([...])` | standalone, no server |
| CSV string/file | `auto_rex("edges.csv")` | auto-detects columns |
| JSON string/file | `auto_rex("graph.json")` | edge list or adjacency |
| Numpy matrix | `auto_rex(array)` | adjacency or feature matrix |
| Pandas DataFrame | `auto_rex(df)` | source/target/weight columns |
| Raw text | `auto_rex("any text")` | built-in tokenizer, no deps |


## TrustGraph Integration

Full documentation: [agent/README.md](agent/README.md)

Requires `trustgraph-base>=2.4.0` for connected mode. Standalone mode
(analyzing triples directly) has no additional dependencies.

### Health snapshot

```python
from agent.integrations.trustgraph_adapter import TrustGraphAdapter

adapter = TrustGraphAdapter(url="http://localhost:8088/")
snap = adapter.health_snapshot(flow="production")
# {'status': 'healthy', 'dim_H': 3, 'health_ratio': 0.82,
#  'cost_multiplier': 1.07, 'nV': 450, 'nE': 1200}
```

### Query assessment

```python
result = adapter.assess_query(
    ["Metformin", "mTOR", "Cancer"],
    flow="default",
)
# {'entities_found': ['Metformin', 'mTOR', 'Cancer'],
#  'health_ratio': 0.82, 'adjusted_tokens': 175,
#  'per_entity': {'Metformin': {'connections': 4, 'local_harmonic_fraction': 0.55}, ...}}
```

### Context core health

```python
health = adapter.analyze_core("biomedical-v3")
print(health["health_summary"])
```

### Flow comparison

```python
comparison = adapter.compare_flows(["v1", "v2", "v3"])
print(comparison["comparison"]["most_stable"])
```

### Version tracking

```python
evolution = adapter.track_evolution(
    snapshots=["core-v1", "core-v2", "core-v3"]
)
print(evolution["trend"])  # "stabilizing", "fragmenting", or "stable"
```

### MCP tools

```python
tools = adapter.as_mcp_tool_definitions()
# 4 tools: analyze_flow, subgraph_confidence, predict_cost, compare_flows
```

### Standalone (no TrustGraph server)

```python
from agent.integrations.trustgraph_adapter import SimpleTriple

adapter = TrustGraphAdapter()
triples = [
    SimpleTriple("Drug_A", "treats", "Disease_X"),
    SimpleTriple("Drug_A", "targets", "Protein_Y"),
    SimpleTriple("Protein_Y", "involved_in", "Disease_X"),
]
rex, meta = adapter.from_triples(triples)
snap = adapter.health_snapshot(rex=rex, meta=meta)
result = adapter.assess_query(["Drug_A", "Disease_X"], rex=rex, meta=meta)
```


## HuggingFace Integration

Analyze transformer attention patterns for RCF axiom compliance.
Runs inference, captures attention at each layer, builds a relational
complex from each, and measures chain condition violation, equiweight
deviation, and channel specialization across the model's depth.

Requires `torch` and `transformers`: `pip install "rexgraph-agent[huggingface]"`

```python
from agent.integrations.huggingface_analyzer import analyze_transformer

report = analyze_transformer(
    model_name="mistralai/Mistral-7B-v0.1",
    text="The cat sat on the mat.",
    device="cuda",
)

report["per_layer_chain_violation"]   # ||B1 B2|| per layer
report["equiweight_deviation"]        # Dirac even/odd balance per layer
report["channel_specialization"]      # which channels each head uses
```

For pre-extracted attention weights (no model loading needed):

```python
from agent.integrations.huggingface_analyzer import quick_attention_analysis

result = quick_attention_analysis(
    attention_matrix,                  # (seq_len, seq_len) numpy array
    token_labels=["The", "cat", ...],
    threshold=0.05,
)
result["betti"]                        # topological structure of attention
result["chi_T"]                        # mean topology channel strength
result["kappa_mean"]                   # coherence of attention pattern
```

Individual functions for targeted measurements:

```python
from agent.integrations.huggingface_analyzer import (
    extract_attention_rex,       # attention matrix -> edge list for RexGraph
    measure_chain_condition,     # ||B1 B2|| on any boundary pair
    measure_equiweight,          # Dirac anticommutator deviation
)
```


## LangChain Integration

Four LangChain tools that give any agent access to structural
analysis during reasoning.

Requires `langchain-core`: `pip install "rexgraph-agent[langchain]"`

```python
from agent.integrations.langchain_tools import get_rex_tools

tools = get_rex_tools(rex)
# Returns: [RexConfidenceTool, RexAnalyzeTool, RexHodgeTool, RexExplainTool]

# Use with any LangChain agent
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)
```

**RexConfidenceTool**: Check structural confidence before answering.
Returns void_affinity, dipole_ratio, chain condition violation, and
coherence. If void_affinity > 0.5 or kappa_mean < 0.3, the structure
has gaps in that region and the agent should qualify its response.

**RexAnalyzeTool**: Full structural analysis on demand. Returns
the Hodge decomposition percentages, Betti numbers, and channel
character.

**RexHodgeTool**: Decompose a specific signal into gradient, curl,
and harmonic components. Reports the energy fraction in each.

**RexExplainTool**: Explain a specific topological feature (a Betti
generator, a void, a persistent cycle) in terms of which edges
and vertices participate.


## LangGraph Integration

Model an agent's state machine as a relational complex. Build
incrementally as the agent runs. Query the structural diagnostics
at any point to detect behavioral issues.

Requires `langgraph`: `pip install "rexgraph-agent[langgraph]"`

```python
from agent.integrations.langgraph_rex import RexStateGraph

rsg = RexStateGraph()

# Build the state graph as the agent runs
rsg.add_state("retrieve", metadata={"type": "action"})
rsg.add_state("reason", metadata={"type": "thinking"})
rsg.add_state("answer", metadata={"type": "output"})
rsg.add_transition("retrieve", "reason", weight=1.0)
rsg.add_transition("reason", "answer", weight=1.0)
rsg.add_transition("reason", "retrieve", weight=0.5)  # retry loop
rsg.log_visit("retrieve")
rsg.log_visit("reason")

# Structural analysis of agent behavior
analysis = rsg.analyze()
analysis["hodge"]                      # gradient/curl/harmonic of execution

# Should the agent continue or stop?
decision = rsg.should_continue(harmonic_threshold=0.4)
decision["recommendation"]             # "continue", "stop", or "caution"
decision["reason"]                     # why (based on harmonic content)

# Detect behavioral cycles
cycles = rsg.detect_cycles()
cycles["n_cycles"]                     # number of cyclic patterns
cycles["longest_cycle"]                # which states form the longest loop

# Decompose a specific execution path
path_info = rsg.decompose_path(["retrieve", "reason", "retrieve", "reason", "answer"])
path_info["pct_harmonic"]              # how much of the path is oscillatory

# Channel profile: what kind of computation is the agent doing?
profile = rsg.channel_profile()
profile["dominant_channel"]            # "T" (logical), "G" (associative), etc.

# Use as a LangGraph conditional edge
checker = rsg.as_langgraph_checker()
gate = rsg.as_langgraph_confidence_gate("reason", "answer")
```

The Hodge decomposition of the agent's state graph reveals:
gradient content means the agent is making progress (directed flow
from input to output), curl content means the agent is in a stable
feedback loop (retrieval-reasoning cycles that converge), and
harmonic content means the agent is stuck in an unresolved
oscillation (retrying without progress).


## I/O

Eight storage formats with automatic format detection on load.
Bundle (.rex) requires zero dependencies beyond numpy.

```python
from rexgraph.io import save_rex, load_rex, save_zarr, load_zarr, save_hdf5, load_hdf5
from rexgraph.io.json_loader import load_json
from rexgraph.io.csv_loader import load_edge_csv

save_rex("graph.rex", rex)                    # portable bundle
save_zarr("graph.zarr", rex, cache="all")     # chunked, compressed
save_hdf5("graph.h5", rex, cache="all")       # single file

rex = load_rex("graph.rex")
rex = load_zarr("graph.zarr")
rex = load_hdf5("graph.h5")
rex = load_json("graph.json")                 # auto-detects format
rex = load_edge_csv("edges.csv")              # column classification
```

| Format | Extension | Dependencies | Notes |
|--------|-----------|-------------|-------|
| Bundle | .rex | none | portable, memory-mappable, zero-dep |
| Zarr | .zarr | zarr | chunked, compressed, cloud-ready |
| HDF5 | .h5 | h5py | single file, HDF5 filters |
| Arrow IPC | .arrow | pyarrow | zero-copy interop with Polars/DuckDB |
| Parquet | .parquet | pyarrow | columnar per-edge/vertex/face tables |
| SQL | any DB | sqlalchemy, pandas | database storage |
| JSON | .json | none | Cytoscape, NetworkX, edge list, adjacency |
| CSV | .csv | none | edge lists with automatic column classification |

All serialization formats support RexGraph, TemporalRex, and cache
groups (algebra, spectral, relational, topology, hodge, harmonic,
faces, field, wave, signal, quotient, persistence, temporal,
standard_metrics). The harmonic cache group persists harmonic_basis,
frustration_per_edge, coparticipation_per_edge, sigma_asymmetry_per_edge,
and scalar health metrics.


## Architecture

Two layers sit under the `RexGraph` object. The **kernels** in `rexgraph.core`
build the structures and are optimized to run on sparse operators. The
**eigen-free modules** in `rexgraph` compute the spectral quantities without a
dense eigensolve, so results hold at any scale. Sparse is the default
everywhere: `RexGraph` does not build the dense relational bundle unless it is
asked for. Dense is materialized on demand for the low-level dense kernels, and
it is the exact reference the sparse path is tested against. `rexgraph.compute`
dispatches across CPU, GPU, or multiple GPUs.

```
rexgraph/                     the relational complex library
    graph.py                  RexGraph and TemporalRex; the single object, composes everything
    analysis.py               dashboard analysis pipeline
    rextypes.py               NamedTuples and enumerations
    compute.py                backend dispatch (CPU / OpenMP / CUDA / ROCm / MPS), multi-GPU, multi-core

    # eigen-free / scale-free layer (the sparse spectral path)
    scale_propagator.py       Chebyshev matrix-function propagators (heat/wave/schrodinger), block-CG Green's
    sparse_character.py       scale-free character / coherence (chi / phi / kappa) via block-CG + LSQR
    harmonic_sparse.py        combinatorial harmonic basis and low-rank projector
    graded_boundary.py        graded mixed-arity boundary builder, exact integer rank, union-find Betti
    field_propagator.py       matrix-free coupled (edge, face) field evolution
    dirac_propagator.py       sparse matrix-free graded Dirac operator
    sparse_interfacing.py     eigen-free interfacing-vector bundle
    dense_matrix.py           dense materialization and dense-only linear algebra

    core/                     39 Cython kernels (BLAS/LAPACK); full per-kernel reference in core/README.md
        boundaries, Laplacians, relational/typed channels, structural character, Hodge, fiber and
        linkage, spectral layout, field, wave, Dirac, transition dynamics, perturbation signals,
        persistence, quotients and relative homology, voids, curvature, temporal lifecycle, standard
        graph algorithms, joins, query, interfacing, cross-complex comparison, and holomorphic
        structure. Kernels are independent (composed only in graph.py) and optimized for sparse
        operators; the dense kernel path also serves as the exact reference for the eigen-free layer.

    io/                       storage and serialization
        bundle (.rex), Zarr, HDF5, Arrow/IPC, Parquet, SQL, JSON, CSV, SafeTensors; format auto-detection


agent/
    auto.py               Input auto-detection and dispatch
    pipeline.py           AnalysisPipeline (Hodge + harmonic diagnostics)
    engine.py             DecisionEngine (interpretation + stability assessment)
    session.py            Stateful analysis sessions

    adapters/
        text.py           Built-in text adapter (no external NLP deps)
        edge_list.py      Edge list adapter
        feature_matrix.py Feature matrix adapter

    integrations/
        trustgraph_adapter.py    TrustGraph (triples, flows, cores, MCP tools)
        trustgraph_pipeline.py   End-to-end TrustGraph pipeline
        huggingface_analyzer.py  Transformer attention analysis
        langgraph_rex.py         LangGraph state machine analysis
        langchain_tools.py       LangChain tool wrappers
        vllm_router.py           vLLM routing integration

    server/               FastAPI backend
    ui/                   Visualization engine
```

All Cython modules are independent. No Cython module imports another.
`RexGraph` exposes everything through `@cached_property` accessors
that lazily compute and cache results.


## Testing

```bash
python -m pytest rexgraph/tests/                    # rexgraph tests
python -m pytest rexgraph/tests/ -m "not slow"      # skip scale tests
python -m pytest rexgraph/tests/test_integration.py  # integration suite
python -m pytest agent/tests/                        # agent tests
```


## Compute backends

The eigen-free tower is matrix-free, so the same operators run on any backend
through the `rexgraph.compute` dispatch layer. It selects a backend
automatically (`recommended_backend()`), size-gates GPU work, tiles large column
blocks across multiple GPUs, fans CPU work across cores without oversubscribing
inner BLAS, and always falls back to CPU. No code change is needed to move
between one core, many cores, an integrated GPU, or several discrete GPUs.

```python
from rexgraph import compute
compute.recommended_backend()          # 'cpu' | 'openmp' | 'cuda' | 'rocm' | 'mps'
compute.set_default_backend("rocm")    # or export REXGRAPH_BACKEND=rocm
# per-call:  sp.heat_apply(L, x, t, backend="cuda")   /   REXGRAPH_MAX_GPUS caps devices
```

GPU acceleration works through a torch backend (CUDA or ROCm) out of the box.
A separate set of standalone CUDA kernels (sparse mat-vec, batched
eigendecomposition, PageRank, force-directed layout) can also be built via CMake
for cupy users:

```bash
cd rexgraph/cuda && mkdir build && cd build
cmake .. -DCUDA_ARCH="80;89;90"
make -j$(nproc)
```

Requires the CUDA toolkit and cupy. If absent, rexgraph runs on CPU.


## Acknowledgements

RexGraph integrations build on these open-source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp): Local LLM inference; the agent's native runtime for running quantized models on CPU and GPU.
- [TrustGraph](https://github.com/trustgraph-ai/trustgraph): Knowledge graph construction and management for RAG applications.
- [Hugging Face Transformers](https://github.com/huggingface/transformers): Pre-trained transformer models and inference.
- [LangChain](https://github.com/langchain-ai/langchain): Framework for building applications with language models.
- [LangGraph](https://github.com/langchain-ai/langgraph): Stateful agent orchestration with cyclic computation graphs.
- [vLLM](https://github.com/vllm-project/vllm): High-throughput LLM serving.


## License

Apache License 2.0
