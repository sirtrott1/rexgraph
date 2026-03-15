# RexGraph

Relational complex analysis with Cython-accelerated internals.

RexGraph implements the Relational Complex Framework (RCF):
an edge-centric (nodes are edge boundaries) chain complex
with typed Laplacians; structural character decomposition;
analysis of signal propagation, dynamical systems, and fields;
void spectral theory; quotient complexes and subcomplex theory;
and algebraically rigorous operations that generalize relational algebra.

## Install

Requires a C compiler, OpenBLAS (or any BLAS/LAPACK), and Python 3.10+.

### With conda/mamba (recommended)

```bash
mamba env create -f environment.yml
mamba activate rexg
pip install -e . --no-build-isolation
```

### With pip + system BLAS

```bash
# Debian/Ubuntu
sudo apt install libopenblas-dev pkg-config

# Then
pip install .

# For development (editable)
pip install -e ".[dev]" --no-build-isolation
```

### Verify

```bash
python -c "from rexgraph.graph import RexGraph; print('OK')"
python -m pytest tests/
```

## Quick start

```python
from rexgraph.graph import RexGraph

rex = RexGraph.from_graph(sources, targets)

# Structural character per edge
chi = rex.structural_character          # f64[nE, 3] on the simplex

# Vertex character and coherence
phi = rex.vertex_character              # f64[nV, 3]
kappa = rex.coherence                   # f64[nV] in [0, 1]

# Betti numbers and Hodge decomposition
betti = rex.betti                       # (beta0, beta1, beta2)
grad, curl, harm = rex.hodge(flow)      # orthogonal decomposition

# Void spectral theory
vc = rex.void_complex                   # Bvoid, eta, chi^void

# Signal imputation
result = rex.impute(observed, mask)     # harmonic interpolation via RL

# Joins
joined = rex.inner_join(other, shared)  # chain complex intersection
```

## Architecture

```
rexgraph/
    core/           Cython math engine (28 modules, ~37K lines)
    io/             Zarr, HDF5, Arrow, Parquet, SQL bridges
    viz/            HTML dashboard generation
    graph.py        RexGraph class (lazy cached bundles)
    analysis.py     Dashboard data contract assembly
    types.py        Typed containers and enumerations
```

## License

GNU Affero General Public License v3.0 or later
