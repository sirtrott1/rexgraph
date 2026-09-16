# RCQL

RCQL is the typed query and mutation layer for relational complexes.

## Install and run

Install from the repository root after configuring the compiler and BLAS as
described in the [library guide](../README.md).

```sh
python -m pip install .
python -m pip install ./rcql
```

The optional RCDB integration uses a supplied store. Install `./rcdb` when
using it. RCQL does not import Agent or System.

```python
from fractions import Fraction
from rexgraph.graph import RexGraph
from rcql import Executor, parse

graph = RexGraph.from_hypergraph(
    [0, 3, 5], [0, 1, 2, 1, 2],
    w_E=[Fraction(2, 3), Fraction(5, 7)],
)
runtime = Executor(sources={"graph": graph})
result = runtime.execute(parse(
    "FROM $graph MATCH e IN CELLS(1) WHERE ARITY(e) > 2 "
    "RETURN e.index, ARITY(e) ORDER BY e.index LIMIT 10"
))
assert result.values == (((0, 3),),)
assert runtime.execute(parse(
    "FROM $graph RETURN COUNT(CELLS(1)), SUM([1/3, 2/7]), MEAN([1, 2])"
)).values == (2, Fraction(13, 21), Fraction(3, 2))
```

Queries without MATCH return one value per RETURN expression. MATCH returns
one sequence of result rows. `EXPLAIN` checks the same types and permissions
without executing expression adapters. A source selector may still read the
selected stored state to establish its contract.

COUNT reads the length of a finite sequence or cell selection. SUM and MEAN
currently accept explicit real scalar sequences, not implicit tensor reductions.
Integer and rational coefficients remain exact. Numerical inputs use compensated
summation. An empty sum is zero; an empty mean is `NONE`.

WHERE requires a scalar boolean. ORDER BY requires compatible finite scalars
in each sort column. Sorting is stable; `NONE` comes first in ascending order
and last in descending order. LIMIT and OFFSET require nonnegative integers.
Selection does not construct or close a subcomplex.


Native accession responses repair RCDB
ratio overflow and remove SciPy from the default scalar reading.
Corpus fields now expose that reading through
visible store snapshots, with signature projection before scoring.

Conversation fields capture a bound Core TurnField
as TemporalRex and preview candidate text without appending it. Temporal
records retain term labels and primary turn identities through RCDB.

Chain symmetries certify explicit rational Euclidean
generators and construct ordered products through Core sparse maps.

Training partitions select declared cells
through Core exact closure, with source state checks and explicit authority.

Tensor manifests expose the existing bounded
Core metadata reader; tensor name search includes the complete header.

Resolvent group words retain ordered products
and inverses as native rational actions, without forming an operator matrix.

Triangular void readings enumerate missing faces
in an explicit pairwise region and compute exact independent fillings on demand.

Declared relation validation checks proposed
higher boundaries for exact closure and storage compatibility without attachment.

Lineage alignment aligns scalar C1 values
using native timeline identities and distinguishes measured zeros from absence.

Relative quotients retain exact branching
shares across the full stored tower, with a certified projection and lazy
relative homology. The result is a coordinate complex, not a renormalized Rex.

Canonical state replay documents APPLY_DELTA
and REPLICATE. Both use Core canonical packages and return owned state;
persistent publication remains an explicit RCDB commit.

The history experiment exercises native
branching relations, witnesses, temporal identities and RCDB version reopen
on Git commits. It also documents ACCESSION_DELTA and the difference between
an accession correspondence defect and a C1 temporal signal.


Rational operator actions describe Cayley
transforms, certified partial complex structures and rational rotations.
They use Core factored actions with explicit rational normalization limits.

Native participation and text overlap operate
directly on the primary tensor. `MARKOV_VIEW()` needs no execution mode flag.
Its action supports exact Q; `PAGERANK` reports a numerical fixed point error
estimate. `TEXT_OVERLAP_VIEW()` applies overlap on the original C1 relation
axis without constructing a sentence graph. Endpoint comparisons live in
the explicit Core oracle module, outside native RCQL execution.

Boundary differences and field correspondence defects
describe DIFF and the three FIELD_DELTA readings. These use Core exact sparse
operations with explicit union coordinates and correspondence contracts.

Primary document fields describe query mass,
coverage, stored section responses and structural closure. These use explicit
source coordinates and Core rational kernels, without a text similarity graph.
Their installed acceptance records the
native checks, storage roundtrips, exactness repairs and measured arithmetic cost.
The ratio execution repair removes the initial
accumulation slowdown using checked compiled integers and local promotion.
The critical family requires explicit weights
and channels and reports its numerical arithmetic. The
exact certificates cover adjugate actions
and supplied chain homotopy witnesses, with costs and proof scope stated.
The primary coordinates expose exact
column expansion, primary lift and full graded hyperslice selections.

Human query text and programmatic callers lower to the same typed AST. System and Agent can construct that AST directly without generating executable strings.

Text accepts exact rational literals (`-17/20`), finite decimal/scientific
notation, booleans, `NONE` and Unicode strings with JSON escaping. Fractions
remain exact in results and plan metadata; decimals remain approximate. See
literal contracts and verification.

Ordered `LET name = expression` clauses capture reusable native values between
FROM and RETURN. Builders use `let(name, expression)`, `ref(name)` and
`query(..., bindings=(...))`.

Calls accept declared named arguments, and expressions support nested lists,
checked record members and return aliases. A source alias qualifies calls on the
one bound source. See composition contracts:

```text
FROM $graph AS r
LET selected = r.CELLS(indices=[0, 2], grade=1)
RETURN INDICATOR(selected) AS signal, r.DESCRIBE().nE AS relations
```

## Native navigation and structural edits

`MATCH` iterates native `CELLS`, `FILES`/`SEARCH`, or `RCDB_LIST`/`RCDB_SEARCH`
values. It does not expand branching relations or close a selected subcomplex.
Scalar comparisons keep exact integers and fractions; `AND`, `OR`, and `NOT`
require booleans (operands are evaluated eagerly). Sorting is stable and explicit;
without `ORDER BY`, `LIMIT` stops iteration early. `Result.values[0]` is a tuple
of rows, each containing the returned expressions in order.

```text
FROM $graph MATCH e IN CELLS(1)
WHERE ARITY(e) >= 3
RETURN e, e.index, ARITY(e) ORDER BY e.index LIMIT 20

FROM $db LET old = RCDB_GET("sample")
MUTATE "sample" SET state=old, expected_version=1
REMOVE [2] ADD [[0,1,2,3]] COMMIT
```

`ADD` takes full C1 boundary supports. For weighted additions, a parameter can
carry `{"columns": [[0,1,2]], "weights": [Fraction(1,3)]}`, with optional
`signs` and `relation_ids`. `REMOVE` takes current C1 basis indices; each clause
addresses the preceding clause's result. Removed cofaces are dropped upward,
and metadata and section memberships are remapped. Inputs are independently
copied through the core codec; failed edits do not mutate the source value.
`SET state` remains explicit. These clauses are not generic arbitrary grade
construction, attribute assignment, or pairwise pattern syntax.

Standalone RCBD, legacy REX, safetensors, HDF5 and Zarr files are readable and
writable through a catalog. Static and temporal source aliases can return their
whole native value; `STATE_HASH()` covers the whole declared history as well.

```text
FROM FILE("files","root0/sample.rcbd") AS document RETURN document
FROM CATALOG("files") MUTATE "root0/sample.rcbd"
SET state=$candidate, expected_hash=$previous_file_hash COMMIT
```

The host passes returned values between statements. A file commit checks the
staged roundtrip's state identity and, when supplied, the physical `expected_hash`
under the publication lock. It retains the old state at the receipt's `backup`
path in a private `.rexgraph-write-*` directory (excluded from catalog discovery).
Regular file replacement is atomic; directory replacement is locked with rollback,
not crash atomic. File commits are not RCDB versions and reject RCDB validity/version
fields. RCDB remains the versioned, signed lineage workflow.

## Mathematical contracts

The native math modules are included in built wheels. See the
isolated packaging verification for build/import
provenance and installed package test coverage.


The native core's coupled field repair preserves
the defined coupling while supporting indefinite heat/wave evolution, arbitrary
initial velocity and sparse full SPD metrics. This is a numerical runtime
prerequisite, not a new field evolution grammar form or an exact exponential.

`CHARACTER(True)` returns exact rational diagonal character for raw G and for
normalized G with nonnegative relation weights. It does not require an eigenbasis
or a full channel matrix. `RANK` and `NULLITY` use exact elimination/structural
identities and refuse unsupported numeric matrices instead of silently estimating.

Exact `QUADRANCE` and `SPREAD` require integer or rational coefficient carriers;
both SPREAD operands are checked. Numerical complex inputs use a Hermitian inner
product. A query is typed before any algebraic rewrite, including under `EXPLAIN`.

## Exact sheaf sections

Exact sheaf comparison supports heterogeneous stalk dimensions and rectangular
incidence maps. `SECTION_CHECK($section)` checks compatibility without constructing
the all pair meeting graph; `GLUE($section)` retains detailed pair level residuals.
Neither computes sheaf cohomology nor performs the planned chain map complex merge.

```text
FROM PHRASE($section)
LET check = SECTION_CHECK(section=$section)
RETURN check.compatible, check.comparison_count, GLUE($section).ratio
```

## Operator inventory

The artifact operators expose Core digest,
lineage, transport and provider operations with explicit in memory contracts.

The partition operators expose stored face
containment and explicit downward closed restriction. They preserve primary
arity, identities and exact higher storage without requiring SciPy.

`FROM $graph RETURN SHOW_OPERATORS()` returns the current global inventory;
`SHOW_OPERATORS(25, 0)` paginates it. Python callers can use
`rcql.operator_inventory(limit=25, offset=0)` without loading the numerical stack.
Rows distinguish current expression contracts, FROM only forms and explicit
refusals.

Hodge, Green, winding and moment operators execute without SciPy imports.
The [dependency profiles](../DEPENDENCIES.md) make SciPy optional for Core and
RCDB. The homology readings add exact simple
and multiplicity dimensions at any carried grade, without harmonic bases.

## Native plans and execution provenance

Read queries now execute a typed native DAG. `EXPLAIN` includes that DAG, explicit
operator domain/codomain descriptors, checked predicates, and selected or deferred
method information. `Result.native_plan`, `Result.execution`, and
`Result.provenance` retain the plan, actual adapter observations, and per output
source/type/rewrite information. Existing result values and legacy plan strings
remain available.


## Channels, stars and scales

```text
FROM $graph RETURN APPLY(CHANNEL("F"), INDICATOR(CELL(1, 0)))
FROM $graph RETURN APPLY(CHANNEL("F"), INDICATOR(CELL(1, 0)), exact=true)
FROM $graph RETURN STAR_CHARACTER(CELL(0, 0), true)
FROM $graph RETURN SCALE_MOMENT(CHANNEL("G"), 1, true, true)
FROM $graph RETURN CHARACTER_ENERGY(CHANNEL("T"))
```

`CHANNEL` uses the bound Rex and does not trace normalize. G follows the source's
raw/normalized selection; F always references raw G. Actions are incidence factored.
T, raw G, F and C support exact rational full actions and transposes. Exact APPLY
retains the same C1 Field carrier as the numerical default, requires integer or
rational input coefficients, and performs no float matrix reconstruction.
Declared rational weights remain rational; a stored float weight means its exact
binary value through the existing source reader. Normalized G's potentially
irrational full action is not certified rational. See
exact channel actions.
`SCALE_MOMENT(operator, order, local=false, exact=false)` returns a trace or a
diagonal Cochain. Exact mode currently supports order zero and channel order one;
normalized-G diagonals remain rational even when its full action is numerical.
Higher numerical moments and energy use sparse products/row reductions, which may
fill. Star character is the incident edge mean, not a Green solve.


## Explicit metrics and Green variants

```text
FROM $graph RETURN MOMENT($u, $v, METRIC(1, $weights), true)
FROM $graph RETURN QUADRANCE($u, true, METRIC(1, $weights))
FROM $graph RETURN SPREAD($u, $v, true, METRIC(1, $weights))
FROM $graph RETURN GREEN_SOLVE(RESOLVENT(CHANNEL("F"), 0.5), $u)
FROM $graph RETURN INTEGRATE($omega, $chain, exact=true)
```

`INTEGRATE` evaluates a Cochain on a Chain with matching source, grade, ordered
basis and shape. This canonical bilinear dual pairing has no implicit metric
or complex conjugation; it is distinct from `MOMENT`. Exact mode uses Q and
the boundary/coboundary operations obey finite Stokes. See
integration contracts and tests.

Here `$u`, `$v` and `$weights` are bound C1 Cochains, not untyped arrays.
`METRIC(grade)` is identity; explicit weights must be strictly positive and on
the same ordered basis. Integer/Fraction contractions remain exact without
square root coordinates. `MOMENT` is signed/Hermitian, not its squared magnitude.
The existing exact flag positions in QUADRANCE and SPREAD are unchanged.

`RESOLVENT(L, alpha=1, tol=1e-10, maxiter=1000)` is the numerical matrix free
action `(I + alpha L)^-1` for a declared Euclidean symmetric PSD operator or a
native metric PSD Hodge down/up/sum. It is distinct from `GREEN()`'s C0
pseudoinverse. `GREEN_SOLVE` preserves vector/block shape and reports the observed
solver method. Weighted Hodge resolvents accept and return **Chains**; the existing
Euclidean Green actions retain their Cochain/Field contract.


`ADJOINT(operator, domain_metric=None, codomain_metric=None)` constructs
`M_domain^-1 operator* M_codomain`. The two metrics name the **original**
operator's endpoints; omitted metrics mean identity. For example:

```text
FROM $graph
LET a = ADJOINT(BOUNDARY(1), METRIC(1, $edge_weights), METRIC(0, $vertex_weights))
RETURN APPLY(a, $vertex_chain, exact=true)
```

This raises a Chain, not a dual Cochain. `ADJOINT(COBOUNDARY(k))` instead retains
cochain variance. `APPLY` preserves vector/block axes; its new optional `exact`
flag is supported by certified rational adjoints, weighted Hodge/Dirac and bracket
actions, not numerical Green solves. Ordinary `COBOUNDARY` remains the metric independent dual transpose.
No general adjoint is granted a Euclidean PSD certificate. See
adjoint contracts and acceptance.

`HODGE_DOWN`, `HODGE_UP`, `HODGE_SUM` and `HODGE_DIFFERENCE` expose the
positive diagonal **Chain** calculus. They compose native boundaries and metric
adjoints, support Q exact application and preserve empty sectors as exact zero.
For example, `APPLY(HODGE_SUM(1, metric=$m), $chain, exact=true)` returns a Chain.
The grade metric and lower/upper neighbor metrics default independently to
identity. Their metric self adjoint certificates do not imply Euclidean symmetry.
`RESOLVENT(HODGE_SUM(1, metric=$m))` uses the symmetric system
`M(I+alpha L)x=Mb`, with inverse diagonal preconditioning and measured residuals
in the original equation. No Gram matrix, inverse, eigenbasis or square root
coordinates are assembled. Hodge difference and Dirac are not granted a PSD
solver; exact solves and general metric solvers remain separate work. See
weighted Hodge and
metric Green contracts.

`DIRAC(metrics=None)` and `ANTI_DIRAC(metrics=None)` act on the full carried
Chain tower. `GRADED_CHAIN([$chain0, $chain1])` preserves separate grade components
and zero fills omitted grades; `GRADE_COMPONENT(state, grade)` extracts one Chain.
`APPLY(DIRAC(), GRADED_CHAIN([$chain]), exact=true)` uses the certified Q boundary
and metric adjoint factors without assembling a block matrix. Supplied metrics
are a list of grade bound positive diagonals; omitted grades use identity.
Dirac is metric self adjoint, anti Dirac metric skew adjoint, neither certified
PSD.

`COMMUTATOR(left, right)` and `ANTICOMMUTATOR(left, right)` construct ordinary
`AB-BA` and `AB+BA` actions on one canonical grade/variance or the full Chain
tower. The rightmost factor acts first; no product matrix or eigensystem is
assembled. Both factors must support exact action to use `exact=true`:

```text
FROM $graph
LET d = DIRAC()
LET a = ANTI_DIRAC()
LET state = GRADED_CHAIN([$chain])
RETURN APPLY(COMMUTATOR(d, a), state, exact=true),
       APPLY(ANTICOMMUTATOR(d, a), state, exact=true)
```

For a verified chain complex with the same grade metrics, these give twice
the Hodge down minus up action and zero, respectively. Execution evaluates the
factors; it never assumes the chain law from their names. Channel brackets use
the native numerical or newly certified rational actions for T/raw-G/F/C;
normalized G still has no general rational full action hook. No bracket is automatically certified PSD, including
the anticommutator of two PSD factors. Nested operand, metric, basis and source
contracts are retained.

## Overlapping type accessions

```text
FROM $graph RETURN ACCESS($x, $map, true)
FROM $graph RETURN MOMENT_TENSOR(ACCESS_TYPES($x, $maps, true), METRIC(1, $weights), true)
FROM $graph RETURN CO_RELATE(ACCESS($x, $p, true), ACCESS($x, $q, true), METRIC(1), true)
```

`$map` is an explicit `rexgraph.type_accession.TypeAccession`; `$maps` is an
ordered `AccessionFamily`. Their sparse maps measure fields from the same ambient
cell basis into ambient cells or explicitly named `CoordinateSpace` outputs.
They may overlap, mix coordinates and carry signed coefficients;
they are not assumed to be projectors or chain preserving subcomplexes. Results
retain their type labels separately from cell coordinates. MOMENT_TENSOR returns
the requested type by type matrix: diagonal quadrances and signed/Hermitian
cross type moments. Common ambient families use the existing diagonal metric.
Rectangular families can supply a `FamilyMetric` with explicit sparse
`TypeRealization` maps into that ambient metric. Both `MOMENT_TENSOR` and
`CO_RELATE` then use the coherent factored form; no cross blocks are assembled.
Alternatively, `CrossMetric(left_map, right_map, entries)` supplies an individual
ordered pairing for `CO_RELATE`, without claiming positivity or family coherence.
Equal coordinate counts never imply the same basis.

The rectangular coordinate extension covers unequal output dimensions and sparse
cross blocks. Coherent family metrics includes a runnable
rectangular moment tensor example. Exact chain maps
adds `CHAIN_MAP($declaration)` for full graded maps with explicit target
boundaries, plus sparse core composition and a bridge from type accessions.
Both chain laws and every commuting square are checked over Q. Name only
accession lookup, induced boundary solving, homotopies and temporal accession
transport remain future work.

## Local artifacts

RCQL reads registered `FileCatalog` sources. It does not expose a general filesystem primitive. File names are catalog relative names and the catalog resolves and rechecks them before each access.

```text
FROM CATALOG("files") RETURN FILES(), SEARCH("document")
FROM FILE("files", "root0/document.rex") RETURN DESCRIBE(), STATE_HASH()
```

RCDB stores use the same source model. Read operators include `RCDB_LIST`, `RCDB_SEARCH`, `RCDB_GET`, `RCDB_HISTORY`, `RCDB_COMMITS`, `RCDB_VERIFY`, `RCDB_HASH`, `RCDB_STATS`, `RCDB_SECURITY`, and `RCDB_STATE_HASH`. The last hashes the versioned logical history, including canonical payload identities, metadata, validity and commit identities, not a backend index. It requires read, history and identity permissions and reads every visible published payload.

`FROM RCDB("db")` resolves a bound store, never a URI. Record source transforms
and GET/HASH use RCDB's selected version snapshot contract. See
native RCDB reads for selectors, provenance, method
requirements and the per handle consistency boundary.

## Capabilities

A `BoundSource` carries a `SourcePolicy`. Signatures declare the requirements used by both EXPLAIN and execution; permissions are checked before an expression operator receives the underlying source. Derived `FILE` sources retain the catalog policy.

Exact record operations require `identity`. Mutation requires both `mutate` and `identity`. Commit history requires both `history` and `identity`. `RCDB_SECURITY` requires both `security` and `admin`, retaining the previous combined planner/executor requirements. Resolving an identity does not grant the `read` capability for subsequent Rex computations. The complete requirement table is in the operator contract note above.

A source policy can also restrict which structural signature fields appear in record lists and searches.

## Mutations

Typed mutation requests use `MutationQuery`. They do not share the ordinary read operator registry.

Text uses `FROM $db MUTATE "id" SET state = $candidate COMMIT`, with optional
actor, validity endpoints and expected version. `EXPLAIN` checks a typed native
input/commit plan without publishing. See mutation contracts
for the supported grammar, native DAG, version guard and execution provenance.

A mutation lowers to the RCDB `commit_mutation` path, which constructs a TemporalRex mutation package and applies the store integrity policy before publication. A store configured with `require_commits=True` rejects ordinary `put` updates and raw deletion.
