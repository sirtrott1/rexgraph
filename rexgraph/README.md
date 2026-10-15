# RexGraph Core

Core constructs relational complexes and provides homology, Hodge analysis,
operator actions, propagation, persistence and native file formats. A relation
is one cell with declared participants and coefficients. Explicit boundaries
connect cells at successive grades, numbered 0 for vertices, 1 for relations
and 2 for faces. This allows pairwise links, shared relations
and filled boundaries to coexist in the same complex.

The Python API uses compiled Cython kernels for sparse storage and numerical
computation. Integer and rational paths retain exact coefficients and ranks;
numerical solvers expose their own tolerances and residuals.

## Installation

From the repository root:

```sh
make install-core
make python
```

Use `sh install.sh --profile core` when Make is unavailable. This installs the
Core library independently of Agent, RCDB, RCQL and System. `make install` includes
those application packages. The [installation guide](../README.md#installation)
describes the compiler, native libraries and optional dependencies.

## Constructing a complex

An ordinary graph uses one relation for each source and target pair:

```python
from rexgraph import RexGraph

rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
print(rex.nV, rex.nE)              # 3, 3
print(tuple(rex.betti))           # (1, 1, 0)
```

The cycle has one connected component and one unfilled loop. A face must be
declared to fill that loop. `from_simplicial` accepts triangle vertex indices;
parallel relations require explicit face edges when the boundary is ambiguous.
`chain_valid` checks that composing successive boundaries gives zero.

For a shared relation, declare its full participant support, tail shares and
metric weight. An explicit vertex table can also retain isolated entities:

```python
from fractions import Fraction
from rexgraph import Relations, VertexTable

relations = Relations.from_supports(
    [[0, 1, 2, 3]],
    vertices=VertexTable(("head", "a", "b", "c", "isolate")),
    shares=[0, Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)],
    weights=[Fraction(2, 3)],
    relation_ids=["assertion-1"],
)
shared = RexGraph.from_relations(relations)
assert shared.nV == 5 and shared.nE == 1
```

Declare participant shares, metric weights, signs, identifiers and attributes
separately. `Absent` records that a value was not supplied. A missing weight is
treated as one during calculation, while `relations.weight` retains its absence;
an explicitly supplied zero or one remains a present value.

| Constructor | Input |
| --- | --- |
| `from_graph` | Source and target arrays |
| `from_relations` | Declared participants, coefficients and identities |
| `from_hypergraph` | Compressed participant support arrays |
| `from_simplicial` | Pairwise relations and explicit triangles |
| `from_adjacency` | An adjacency matrix with a declared direction policy |

## Reading data

The declared readers separate file syntax from the meaning of its columns.
For a table, supply a `RecordSchema` and a `RelationSpec`; `construct` maps each
record into the common `Relations` carrier. Number rules specify whether a
decimal or binary source value remains exact. Numeric text is not inferred
from a field declared as a string.

The [root example](../README.md#loading-and-processing-data) reads a weighted
CSV table. The [I/O guide](io/README.md) covers JSON, JSONL, Arrow IPC, Parquet,
SQLite, XLSX and reusable dataset declarations.

## Signals and Hodge analysis

A signal assigns a value to each cell at one grade. For a relation signal,
Hodge decomposition returns gradient, curl and harmonic components:

```python
import numpy as np

signal = np.array([1.0, 0.0, -1.0])
gradient, curl, harmonic = rex.hodge(signal)
assert np.allclose(gradient + curl + harmonic, signal)
```

The gradient component comes from vertex potentials, the curl component from
face boundaries, and the harmonic component from the Hodge operator's kernel.
When a complex has no upper grade, its upper operator is exactly zero. The
relationships and signal supplied by the caller determine how to interpret
these components in an application.

`Cochain`, `Chain`, `Field` and `GradedState` retain the cell grade and optional
ordered basis alongside values. This information lets boundary, coboundary
and Hodge operators check which cell space those values belong to. Numerical
Green actions solve a specified system without forming an inverse; supported
rational actions can be requested explicitly.

## GPU execution

Run `make install-gpu` from the repository root to install the platform with
automatically selected CUDA or ROCm dependencies. AMD installation includes the
custom HIP kernels. A Core installation can enable the same dependencies with
`sh install.sh --profile core --gpu auto`.

Vulkan supports float64 sparse products and the propagation and solver paths
that use them. Select it explicitly for a numerical operation:

```python
from rexgraph.scale_propagator import heat_apply

smoothed = heat_apply(rex.L1, signal, t=0.2, backend="vulkan")
```

Vulkan retains sparse coefficients on the GPU between products and performs
vector updates on the CPU. CUDA and ROCm propagation can retain the recurrence
on the device; their automatic selection uses a size threshold to account for
transfer costs. An explicit Vulkan request requires a visible hardware GPU with
float64 shader support and reports an error when that requirement is unmet.
`make verify-gpu GPU_BACKEND=vulkan` checks numerical agreement with CPU results.

## Capabilities

| Capability | Main interfaces |
| --- | --- |
| Homology and boundaries | `betti`, `chain_valid`, `graded_boundary`, `linear_operator` |
| Signal decomposition | `hodge`, `hodge_full`, typed Hodge operators |
| Structural channels | `channel_operator`, `structural_character`, `vertex_character` |
| Local responses | `coherence_response`, `character_response`, `local_context` |
| Diffusion and waves | `evolve_markov`, `evolve_schrodinger`, field evolution methods |
| Persistence | `filtration`, `persistence`, barcode and diagram methods |
| Selection and edits | `Selection`, `restrict`, `subgraph`, insertion and deletion |
| Temporal structure | `TemporalRex`, snapshot identities and lifecycle measurements |
| Models | [Numerical cochain models](flow/README.md) and `rexgraph.nn` |
| Storage | [Native file formats and declared readers](io/README.md) |

The four structural channels are T, G, F and C. T measures signed boundary
incidence, G measures unsigned overlap with relation weights, F measures the
discrepancy between signed and unsigned incidence, and C measures coparticipation.
The geometry G channel belongs to the down sector, which reads a relation's
vertex boundary. Curl uses the up sector, which reads its face boundary.
The up operator at grade k and down operator at grade k+1 share their nonzero
spectrum and generally have different shapes.

Sparse operator APIs retain boundary factors and materialize matrices when
requested. Dense matrix properties and spectral methods remain available for
explicit numerical analysis. Choose those interfaces according to the size of
the complex and the required output. The [API reference](REFERENCE.md) lists
the methods, shapes and arithmetic contracts.

## Saving state and selecting cells

```python
from rexgraph.io import load_rcbd, save_rcbd

save_rcbd("cycle.rcbd", rex)
restored = load_rcbd("cycle.rcbd")
assert tuple(restored.betti) == tuple(rex.betti)
```

Native state retains declared coefficients, identities and registered
components. Table exports have their own schemas and do not automatically
represent a complete saved complex. Copies and restrictions use the same
component transport; a restriction drops an upper cell when its boundary loses
a participant. `Selection` and `restrict` provide explicit grade selections
with maps back to the original basis. See the
[state and selection contracts](REFERENCE.md#declared-data-and-exact-io).

## Examples and reference

See the [compiled kernel guide](core/README.md), [API reference](REFERENCE.md)
and [dependency guide](../DEPENDENCIES.md) for supported operations, file readers
and optional analysis packages. Core is distributed under the
[Apache License 2.0](../LICENSE).
