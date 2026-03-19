# RexGraph

Relational complex analysis with Cython-accelerated internals.

RexGraph implements the Relational Complex Framework (RCF):
an edge-centric chain complex where edges are primitive and vertices
are derived from edge boundaries. Typed Laplacians decompose structure
into topological, geometric, frustration, and copath channels.
Structural character places every edge and vertex on a simplex.
Hodge theory, persistent homology, void spectral theory, Dirac
operators, fiber bundles, interfacing vectors, cross-complex
comparison, and quotient complexes are all computed through a single
`RexGraph` object backed by 33 Cython modules (25,246 lines) calling
BLAS/LAPACK directly.

## Install

Requires a C compiler, OpenBLAS (or any BLAS/LAPACK), and Python 3.10+.

### conda/mamba (recommended)

```bash
mamba env create -f environment.yml
mamba activate rexgraph
pip install -e . --no-build-isolation
```

Minimal (core only, no I/O or viz deps):

```bash
mamba env create -f environment-minimal.yml
mamba activate rexgraph
pip install -e . --no-build-isolation
```

### pip + system BLAS

```bash
# Debian/Ubuntu
sudo apt install libopenblas-dev pkg-config

pip install .                              # core only
pip install ".[io]"                        # + zarr, h5py, pyarrow, sqlalchemy, pandas
pip install ".[all]"                       # everything except CUDA
pip install -e ".[dev]" --no-build-isolation  # editable dev install
```

### HPC

```bash
bash setup_hpc.sh              # full install (loads gcc, sets up conda, builds in scratch)
bash setup_hpc.sh --minimal    # core only
bash setup_hpc.sh --test       # install + run tests
bash setup_hpc.sh --clean      # remove environment
```

### Verify

```bash
python -c "from rexgraph.graph import RexGraph; print('OK')"
python -m pytest tests/
```

## Quick start

```python
from rexgraph.graph import RexGraph
import numpy as np

rex = RexGraph.from_graph(
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

All serialization formats support RexGraph, TemporalRex, and 13 cache
groups (algebra, spectral, relational, topology, hodge, faces, field,
wave, signal, quotient, persistence, temporal, standard_metrics).

## Architecture

```
rexgraph/
    graph.py              RexGraph and TemporalRex (3801 lines)
    analysis.py           Dashboard analysis pipeline (1456 lines)
    types.py              NamedTuples and enumerations (1106 lines)

    core/                 33 Cython modules (25,246 lines)
        _common.pyx/.pxd    Shared types, memory limits, parallelization, CSR utilities
        _rex.pyx             Edge classification, subsumption embeddings (graph/hyper/simplicial)
        _boundary.pyx        Boundary operator B1 construction from CSR
        _laplacians.pyx      L0, L1, L2 Hodge Laplacians + full spectral bundle
        _overlap.pyx         Overlap Laplacian L_O (Jaccard-weighted)
        _relational.pyx      Relational Laplacian RL via trace-normalized hats
        _character.pyx       Structural character, hat eigen, mixing times, face-void dipole
        _fiber.pyx           Fiber bundle similarity, threshold graphs, linkage complex
        _hodge.pyx           Hodge decomposition g = grad + curl + harm
        _spectral.pyx        Spectral embedding, force-directed layout
        _linalg.pyx          Direct BLAS/LAPACK calls (dgemm, dsyevd, dgesdd)
        _sparse.pyx          DualCSR sparse matrix type, spmm kernels
        _field.pyx           Field operator M on (E, F), wave evolution
        _wave.pyx            Complex-amplitude wave mechanics, density matrices
        _dirac.pyx           Dirac operator, graded evolution, Born probabilities
        _state.pyx           Signal construction, energy decomposition
        _transition.pyx      Markov, Schrodinger, coupled RK4 evolution
        _signal.pyx          Perturbation analysis pipeline
        _persistence.pyx     Column reduction, barcodes, landscapes, distances
        _quotient.pyx        Subcomplex, quotient, relative homology, character filtration
        _temporal.pyx        Edge/face lifecycle, BIOES phase detection
        _standard.pyx        PageRank, betweenness, clustering, Louvain
        _void.pyx            Void boundary, harmonic content, void strain
        _rcfe.pyx            RCFE curvature, Bianchi identity, attributed curvature, strain equilibrium
        _hypermanifold.pyx   Manifold sequence M1 < M2 < M3, harmonic shadow
        _frustration.pyx     Signed graph frustration index
        _cycles.pyx          Cycle detection, adjacency list construction
        _faces.pyx           Face detection, metrics, typed/context face selection
        _joins.pyx           Chain complex inner/outer/left join
        _query.pyx           Signal imputation, cell explanation, spectral propagation
        _interfacing.pyx     Interfacing vector, response operators, channel scores, confidence
        _channels.pyx        Per-channel signal decomposition, spectral group scoring
        _cross_complex.pyx   Cross-complex alignment, kappa correlation, bridge analysis

    io/                   10 modules (8,541 lines)
        bundle.py            .rex portable format (numpy + json only)
        zarr_format.py       Chunked Zarr stores (v2 and v3)
        hdf5_format.py       Single-file HDF5 storage
        arrow_bridge.py      Arrow/IPC columnar export
        parquet_bridge.py    Parquet per-cell tables
        sql_bridge.py        SQLAlchemy database bridge
        csv_loader.py        CSV edge lists with column classification
        json_loader.py       JSON format auto-detection
        _serialization.py    Generic NamedTuple serialization
        _compat.py           Zarr v2/v3 and HDF5 compatibility layer

    viz/                  Dashboard generation
```

All inter-module composition happens in `graph.py`. No Cython module
imports another. `RexGraph` exposes everything through `@cached_property`
accessors that lazily compute and cache results.

## Testing

```bash
python -m pytest tests/                    # all tests
python -m pytest tests/ -m "not slow"      # skip 100K+ edge scale tests
python -m pytest tests/test_integration.py # integration suite
```

1669 tests across 33 core modules, 10 I/O modules, graph.py,
analysis.py, types.py, and scale tests from 10 to 500K edges.
Full suite runs in ~31 seconds on a single thread (AMD 5950X).
That time includes graph construction, spectral decomposition,
and all cached property computation for each test fixture, not
just assertion checks.

## CUDA (optional)

GPU-accelerated kernels for sparse matrix-vector multiply, batched
eigendecomposition, PageRank, and force-directed layout. Built
separately via CMake:

```bash
cd rexgraph/cuda && mkdir build && cd build
cmake .. -DCUDA_ARCH="80;89;90"
make -j$(nproc)
```

Requires CUDA toolkit and cupy. If absent, rexgraph falls back to CPU.

## License

GNU Affero General Public License v3.0 or later
