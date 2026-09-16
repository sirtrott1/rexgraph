# Runtime dependencies

Core, RCQL and the base RCDB record stores do not require SciPy. NumPy remains
required. Core numerical actions use its compiled sparse kernels and native
BLAS and LAPACK libraries. Exact boundary actions and rank use the existing
integer and rational routines. Removing SciPy does not make numerical actions
exact or remove the system libraries needed to compile the kernels.

## Installation profiles

Install local packages in dependency order, with Core first:

```sh
python -m pip install .
python -m pip install ./rcql ./rcdb
```

This base installation does not install SciPy. File formats beyond the native
record format are optional through `rexgraph[io]`. Signing uses
`rexgraph[security]`. Neither extra requires SciPy. Core `all` means `io` and
`security`; it does not enable reference oracles.

| Profile | Purpose |
| :-- | :-- |
| `rexgraph[scipy]` | Explicit SciPy matrix exports and older sparse analysis APIs |
| `rexgraph[oracles]` | The same dependency, requested for spectral and matrix reference comparisons |
| `rexgraph-rcdb[scipy]` | Older index matrix and document search operations |
| `rexgraph[dev]` | Test tools and SciPy for the full reference suite |
| `rexgraph-agent` | Declares SciPy directly because its pipeline and graph views use it |

An extra installs a dependency; it does not change operator dispatch or select
a dense algorithm. Reference APIs must still be requested explicitly. A full
Agent installation is not a SciPy free environment.
Oracle code remains part of the Core distribution. Some reference routines
use NumPy or Core LAPACK directly and need no SciPy extra.

## Native and optional surfaces

`RexGraph.from_cells` now constructs the whole boundary tower in native sparse
storage. `build_graded_boundaries(..., native=True)` exposes the same builder.
The default of that lower level builder remains its existing SciPy CSR output
contract. `graded_boundaries()` and `as_scipy()` are explicit compatibility
exports and require `rexgraph[scipy]`.

Native boundary actions, channels, exact ranks, homology, weighted Hodge and
Green actions are covered by `scripts/smoke_no_scipy.py`. RCQL execution and
RCDB record reads, edits, commits and native state serialization do not need
those compatibility exports. Importing the IO compatibility module no longer
loads SciPy just because it is installed. The compiled overlap module imports
its optional matrix dependency only inside the functions that use it; an
unused SciPy import in the compiled graded channel module was removed.

SciPy remains used in older field and Dirac propagators, scale propagation,
older partition analysis, Fiedler and some
harmonic analysis, explicit sparse basis outputs and matrix exports. RCDB
index matrix operators and Agent analysis also use it. The scalar accession
reading at one step and its exact counterpart use Core integer incidence
and rational accumulation without SciPy. These are real runtime APIs,
not all dense oracles. They are retained with explicit dependency profiles,
not removed or renamed as reference code.

The structural partition builder and its state serialization use native sparse
storage. RCQL `FACES`, `RESTRICT` and `PARTITION` use that builder without SciPy.
They do not call the older partition analysis APIs.

## Verification

The native gate has two modes. Its default blocks every SciPy import, including
the expected failure of an explicit matrix export. `--require-absent` also
checks that no SciPy distribution is installed. Run from a neutral directory
against installed wheels:

```sh
python -I /path/to/rexgraph/scripts/smoke_no_scipy.py --platform --operators --require-absent
python -m pip check
```

The platform workflow installs Core, RCQL and RCDB in a separate environment
without Agent or the oracle extras. It runs this gate and the RCQL and RCDB
lifecycle experiment there. A separate full environment includes the optional
analysis dependency for the complete reference suites.

