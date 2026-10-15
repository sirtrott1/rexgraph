# RexGraph compiled kernels

This directory contains the Cython implementations of sparse storage,
boundaries, numerical linear algebra, Hodge analysis, structural channels,
persistence and temporal operations. Python modules compose these kernels and
retain exact integer or rational declarations where the operation supports them.

## Building

The build project is the repository root. For a regular installation, run:

```sh
make install-core
make verify
```

For development, `make install-dev` installs editable packages and build tools.
The [installation guide](../../README.md#installation) covers Python, compiler
and BLAS/LAPACK requirements. `BUILD_JOBS=2` is the default compiler concurrency.
The installer verifies compiled imports and numerical and exact operations before
reporting completion.

OpenMP is enabled when the compiler and runtime support it. A build with OpenMP
disabled uses serial kernel paths. The default build preserves ordinary
floating point semantics and avoids CPU specific compiler options. The
installer's `--native` option selects CPU specific optimization and relaxed
floating point rules for a build intended for that machine.

## Python interfaces

Applications normally use `RexGraph` or typed operator modules. These retain
cell grade, ordered basis and arithmetic information around the kernel calls:

| Module | Purpose |
| --- | --- |
| `rexgraph.graph` | Construction and graph analysis |
| `rexgraph.graded_boundary` | Exact ranks and homology across grades |
| `rexgraph.linear_operator` | Boundary, coboundary and Hodge actions |
| `rexgraph.scale_propagator` | Matrix function actions and Green solves |
| `rexgraph.sparse_character` | Structural character using sparse computations |
| `rexgraph.harmonic_sparse` | Harmonic bases and projections |
| `rexgraph.compute` | Compute backend and thread selection |

```python
from rexgraph import RexGraph
from rexgraph.core import _common

rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
assert tuple(rex.betti) == (1, 1, 0)
print(_common.get_openmp_enabled())
```

Direct kernel calls have specific shape, dtype and memory contracts. Numerical
matrix and spectral routines use floating point arrays. Integer and rational
calculations use the typed exact operator interfaces, which retain those
coefficients through the requested action.

## Kernel groups

| Group | Operations |
| --- | --- |
| Sparse storage and boundaries | CSR construction, incidence, boundary composition and support traversal |
| Linear algebra | BLAS/LAPACK calls, sparse matrix actions and iterative solves |
| Hodge and spectral analysis | Lower and upper operators, decomposition, eigenvalues and projections |
| Structural channels | Signed incidence, unsigned G channel, frustration and coparticipation |
| Topological operations | Faces, selections, quotients, joins and persistence |
| Dynamics and temporal state | Propagation, field trajectories, changes and lifecycle measurements |

The [kernel reference](REFERENCE.md) describes individual functions, parameters,
return shapes, error codes and configuration. The [Core guide](../README.md)
provides application examples. GPU kernels and their device requirements are
optional and require separate verification on the selected device.

## Verification

```sh
make test-rexgraph TEST_ARGS='-rs'
```

Run this from the repository root after installing a development profile.
The runner checks which installed packages supply the runtime modules. Optional
feature and hardware tests report their skip reasons. See the
[dependency guide](../../DEPENDENCIES.md#test-coverage-and-skipped-cases) for the
full development profile and external requirements.
