# RCQL

RCQL is a typed query language for relational complexes and registered data
sources. It evaluates construction, analysis, selection and operator expressions
through RexGraph, and provides explicit mutations for RCDB records. Queries
retain source identities, arithmetic declarations and execution provenance.

## Installation

From the repository root, `make install` installs RCQL with the application
packages. Use `make python` to open the installed interpreter. If Core is
already installed, run `python -m pip install ./rcql` from the repository root.
RCQL requires Core and NumPy; RCDB sources are supplied by the application.
See the [installation guide](../README.md#installation).

## Running a query

Register a complex under a name and refer to it in the query:

```python
from rexgraph import RexGraph
from rcql import run

network = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
result = run(
    "FROM $network RETURN COUNT(CELLS(1)), BETTI(1)",
    sources={"network": network},
)
assert result.values == (3, 1)
```

A query without `MATCH` returns one value for each `RETURN` expression.
`MATCH` iterates a finite cell selection and returns a sequence of rows.
The following query selects relations with more than two participants:

```python
from rexgraph import Relations, VertexTable

relations = Relations.from_supports(
    [[0, 1, 2], [1, 2]],
    vertices=VertexTable(("a", "b", "c")),
)
shared = RexGraph.from_relations(relations)
result = run(
    "FROM $graph MATCH e IN CELLS(1) WHERE ARITY(e) > 2 "
    "RETURN e.index, ARITY(e) ORDER BY e.index LIMIT 10",
    sources={"graph": shared},
)
assert result.values == (((0, 3),),)
```

`WHERE` requires a scalar boolean. `ORDER BY` requires compatible finite
scalars and uses a stable sort. `LIMIT` and `OFFSET` take nonnegative integers.
Selecting rows does not itself construct a subcomplex; restriction and
structural edits are explicit operators.

## Arithmetic and operator contracts

Integer and rational scalar expressions retain exact values:

```python
from fractions import Fraction

result = run("FROM $network RETURN SUM([1/3, 2/7]), MEAN([1, 2])",
             sources={"network": network}, exactness="exact")
assert result.values == (Fraction(13, 21), Fraction(3, 2))
```

The default `exactness="declared"` retains the arithmetic declared by each
source and operator. `exactness="exact"` checks that a supported finite read
query without MATCH returns no approximate or rounded numeric values. It
checks the result's arithmetic without converting a numerical algorithm. Exact channel and boundary actions
are requested through their operator arguments.

A Chain records coefficients for a combination of cells; a Cochain assigns
values to cells and evaluates those combinations. Boundary operators lower the
grade of a Chain, while coboundary operators raise the grade of a Cochain.
RCQL checks the grade and cell basis when combining typed values and operators,
including the identity and order of the cells. Explicit metrics determine
metric adjoints. The [operator reference](REFERENCE.md#operator-contracts)
covers Hodge, Green, Dirac, channel and metric operations.

## Capabilities

| Area | Queries and interfaces |
| --- | --- |
| Structure | Cell counts, supports, arity, boundaries and Betti numbers |
| Fields and channels | Typed actions, exact coefficients, Hodge components and channel measurements |
| Selection and edits | Restrictions, partitions, insertions and explicit structural changes |
| Registered data | Declared datasets, file catalogs and RCDB records |
| Plans and provenance | `EXPLAIN`, native plans and observed execution methods |
| Exact artifacts | Portable results and dependency checked query caches |
| Advanced declarations | Sheaf sections, type accessions, names and finite programs |
| Record mutations | Typed mutation requests with explicit publication and version guards |

Use `Executor` and `parse` to reuse a runtime, or the typed builder interfaces
to assemble queries in Python. The [API reference](REFERENCE.md) gives the
operator inventory, examples and requirements for advanced declarations.

## Registered sources and permissions

The application supplies source objects through `sources`. A declared dataset
combines a Core schema and relation mapping with a source file or stream.
A file catalog exposes selected roots under relative names, while an RCDB
binding exposes a supplied store. Query text resolves those registrations.
See [dataset and catalog examples](REFERENCE.md#local-artifacts).

A `BoundSource` can carry a `SourcePolicy`. Planning and execution enforce
operator permissions, including record history, identity and mutation
requirements. `EXPLAIN` checks the same type and permission contracts without
executing expression adapters; resolving a selected record can still read its
state to establish the source contract.

## Portable exact results

```python
from rcql import Result

payload = result.to_bytes()
restored = Result.from_bytes(payload)
assert restored.values == result.values
```

The result codec preserves exact values, plans, source references and
provenance. Graph results carry native sealed state, and supported Chain and
Cochain values retain their variance and basis. Decoding checks integrity
without rerunning the query. Values without a portable codec raise `TypeError`.
The default artifact limit is 64 MiB and can be configured with `max_bytes`.

System uses this codec for native result downloads, and the query cache uses
the same payload. See the [portable result reference](REFERENCE.md#portable-exact-results)
for the supported value inventory and cache identity rules.

## Record mutations

Mutation queries publish through RCDB's `commit_mutation` contract. They use
an explicit candidate state and can require an expected record version:

```text
FROM $db MUTATE "sample" SET state = $candidate COMMIT
```

The application binds `db` and `candidate` and supplies the required
permissions. `EXPLAIN` plans a mutation without publishing it. See the
[mutation reference](REFERENCE.md#mutations) and [RCDB guide](../rcdb/README.md)
for publication, history and concurrency behavior.

## Reference and testing

The [API reference](REFERENCE.md) covers operators and detailed mathematical
contracts. The [dependency guide](../DEPENDENCIES.md) lists development extras.
From the repository root, run `make test-rcql TEST_ARGS='-rs'` after installing
a development profile. RCQL is distributed under the [Apache License 2.0](LICENSE).
