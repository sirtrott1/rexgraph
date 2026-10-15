# RCQL API reference

See the [README](README.md) for installation and usage.

RCQL evaluates typed queries and explicit mutations over relational complexes
and registered data sources. This reference describes operator inputs, returned
values and execution requirements.

## Install and run

From the repository root, `make install` installs RCQL with the other packages.
Run `make python` to use the installed interpreter. See the
[installation guide](../README.md#installation) for compiler and environment setup.

If Core is already installed in your environment, install RCQL with
`python -m pip install ./rcql` from the repository root. RCQL depends on Core;
RCDB integration accepts a supplied store. RCQL does not import Agent or System.

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

## Portable exact results

`Result.to_bytes()` and `Result.from_bytes(payload)` preserve exact values,
arithmetic declarations, aliases, plans, rewrites, provenance and execution
records. Graph results carry native sealed states. Chains and cochains retain
their variance and share the reconstructed source graph when they use the same
basis. Decoding validates integrity and reconstructs values without running a
query. The default artifact limit is 64 MiB; pass `max_bytes` to either method
to choose another limit.

The portable inventory includes the core closed scalar/tensor values, nested
lists/maps, static and temporal graphs, declared Relations, chains and cochains,
partitions with lineage, retained tensor fields and channels, sections, models,
spans, and finite program declarations and transformations. Shared graph and
FieldSource bindings retain their reference identity; detached sources remain
detached. No program executes during decoding. A live operator or another value
without a portable codec raises `TypeError`; request its explicit reading or
use its dedicated artifact contract. System's JSON response is a bounded
preview. Large integers use decimal text records, and absence and approximation
have distinct tags. Parquet export preserves exact coefficient columns and
partition lineage.

New query cache entries store the same `Result.to_bytes()` payload used for
downloads. Cache identity includes the implementation and declared input
identities; reuse checks dependency bindings before returning a result. Older
cache manifests retain a read compatibility path. Portable result framing and
graph identity do not certify a live operator that has no portable codec.

## Certified name transformations

ProgramTransformation stores an original name declaration, a candidate
declaration, input port and captured value mappings, and a certificate of their
boundary correspondence. The port and alias rules use NameRelation.rename and
NameRelation.named to transform an operation's interface.

```python
from rcql import Executor, NameRelation, parse
from rexgraph.graph import RexGraph

rex = RexGraph.from_cells([2, [[0, 1]]])
operation = NameRelation.operator("SUM")
runtime = Executor(sources={"r": rex}, params={"operation": operation})
result = runtime.execute(parse('''
FROM $r
LET t = TRANSFORM_NAME($operation, "port", ["values", "items"])
LET candidate = TRANSFORM_COMPILE(t, $operation, [[1/3, 2/7]])
RETURN TRANSFORM_VERIFY(t, $operation), NAME_APPLY(candidate, [[1/3, 2/7]])
'''))
```

TRANSFORM_COMPILE checks the original declaration identity and validates both
interfaces against the supplied source, arguments and permissions. It returns a
name declaration for later execution through NAME_APPLY, which checks the
inputs and permissions at the time of application.

TRANSFORM_VERIFY checks the substitution and exact chain map for the declared
interface rename. It retains the operation grades, repeated argument
occurrences and shared values. This certificate covers that correspondence;
program body equivalence, trace labels and undeclared metrics are outside its
scope.

`TRANSFORM_RECORD` and `TRANSFORM_READ` use ordinary native records and support
RCDB, `.rcbd` and `.safetensors`. Reload validates the declaration, certificate
and carrier. `TRANSFORM_TOPOLOGY` returns the candidate `RelationTopology`
record, including its exact persisted co relations. The transformation record
retains both endpoint towers and their correspondence.

Finite programs and pure recursive programs can carry transformations as values.
Result reuse includes their declaration identity. A stored transformation carries
no live source or permissions; native arguments and historical evidence remain
explicit execution bindings. The original declaration remains unchanged.

### Certified section readouts

`TRANSFORM_SECTION(family, left, right)` constructs a guarded replacement of
one linear readout by another. It reuses `ReadoutEquivalence` and checks both
`(left-right)x0 = 0` and `(left-right)N = 0` on the complete family `x = x0 + Nu`.
Matching only a particular solution is insufficient. Equality of readouts does
not imply that either readout is determined.

```text
FROM $r
LET t = TRANSFORM_SECTION($family, $left, $right)
LET original = TRANSFORM_SOURCE(t)
LET candidate = TRANSFORM_COMPILE(t, original, [$family, $left, $right])
RETURN NAME_APPLY(candidate, [$family, $left, $right])
```

The result is a `SectionImage`, retaining free directions and every field axis.
Use `SECTION_VALUE` only when that image is determined. The two operation
declarations call `SECTION_CERTIFIED_OBSERVE` with explicit left and right
selections. Both check the current family, map identities, source versions,
contributors and permissions. Compilation and application remain separate.

The stored transformation contains the readout claim, input digests and both
operation boundary records. Compilation and application require the live family
and maps and recompute the readout certificate. TRANSFORM_VERIFY checks the
stored declaration structure and reports the required live inputs.

The equivalence concerns readout values over that one family. It does not equate
the two maps on the whole coordinate space, erase their provenance, equate
`SectionImage` identities, or justify replacing an arbitrary program body.

### Specialization and composition

`TRANSFORM_SPECIALIZE(operation, bindings)` captures an explicit mapping of
open input names to finite values. It reuses `NameRelation.bind`. The candidate
agrees with the original when those inputs have exactly the declared values.
Defaults on the remaining inputs are preserved. Native sources and fields
remain explicit execution inputs, not serialized captures.

`TRANSFORM_COMPOSE(operation, following, ports)` reuses `NameRelation.then`.
It connects the first result to the listed input ports of the following name.
The first result is computed once, even when several ports use it. Captures
and intermediate names remain distinct. Unconnected input name collisions
are refused rather than silently renamed.

```text
FROM $r
LET t = TRANSFORM_COMPOSE($first, $following, ["input"])
LET candidate = TRANSFORM_COMPILE(t, $first, $arguments)
RETURN NAME_APPLY(candidate, $arguments)
```

Compilation validates the composed result type and effects without executing
either operation. Composition constructs the declared `G(F(x))`; it does not
assert that this result equals `F(x)`. The certificate retains both input
declarations, the connected ports, capture and stage mappings, and the source
and candidate operation towers. Each tower satisfies the chain law. A boundary
map between towers with different occurrences is not inferred from their shapes.

Construction uses the existing name methods. Verification checks the retained
bindings and stage substitutions independently, without expanding shared
expressions into a duplicated expression tree. The ordinary transformation
record, read, topology and source operations apply to both constructions.

## Certified finite program transformations

`TRANSFORM_PROGRAM(program, rule, arguments)` constructs a new finite `Program`
declaration. The original remains unchanged. Three rules are supported:

* `input` takes `[old, new]` and renames an open input and its parameter uses.
* `step` takes `[old, new]` and renames a step, its output references and exports.
* `specialize` takes a mapping of open inputs to finite captured values.

Captures retain their input type, grade and variance contracts. Exact integers
and fractions remain exact. Captured values cannot be overridden at execution.
Native sources and fields remain explicit bindings. These rules cannot rename
or capture a parameter used for source selection, including a historical cutoff.

```text
FROM $r
LET t = TRANSFORM_PROGRAM($program, "input", ["x", "value"])
LET candidate = TRANSFORM_PROGRAM_COMPILE(t, $program, $sources, $parameters)
RETURN candidate
```

Here `$sources` maps source names to native `Binding` objects. `$parameters`
contains exactly the candidate's remaining open inputs. Compilation checks
both declarations with the existing planner, including output types, effects,
source identities, historical evidence and contributor permissions. It does
not execute either program. `PROGRAM_RUN` is a separate request and validates
its own current bindings.

`TRANSFORM_VERIFY` checks the declaration certificate independently of the
construction methods. It retains source selector digests, typed port mappings,
step links, exports, capture digests and operator requirements. Specialization
claims correspondence only at the declared captured values. It does not establish
arbitrary program equivalence or equal trace labels.

TRANSFORM_PROGRAM_TOPOLOGY accepts the same four arguments as compilation and
returns the candidate's bound PlanTopology record. It inspects the static plan.
Programs containing dynamic MATCH can be transformed and executed, but this
static inspection rejects them.

`TRANSFORM_RECORD`, `TRANSFORM_READ` and `TRANSFORM_SOURCE` also support these
transformations. Schema two retains the finite endpoint declarations and their
certificate. Programs without captures keep schema one; programs with captures
use schema two. Both restore through the existing program owner. Finite meta
programs can return a transformed `Program`, with its declaration retained in
type checks, result reuse and recursive histories. Inspection or construction
never changes the program currently running.

## Exact program assembly

ProgramAssembly combines finite local programs, scalar coefficient spaces,
external inputs, links and exports. Each local input has one explicit binding,
links use preceding fragment exports, and shared external inputs require the
same type. Source selectors retain their names.

`PROGRAM_ASSEMBLY(name, fragments, coefficients, inputs, links, outputs)` uses
these tuple forms:

* Fragment: `(cell, Program)`.
* Coefficient: `(cell, input, local_coordinate)`.
* External input: `(cell, input, public_name)`.
* Link: `(cell, input, preceding_cell, export)`.
* Export: `(public_name, cell, export)`.

PROGRAM_GLUE(assembly, system, observation, observed) solves the declared exact
section system and returns a ProgramFamily of compatible coefficient
assignments. Local programs remain fixed; section coordinates supply their
scalar coefficients. Restriction maps follow the section incidences.

PROGRAM_FAMILY_OBSERVE(family) retains the coefficient observation used for
execution. PROGRAM_FAMILY_COMPILE(family, sources, parameters) requires every
needed coefficient to be determined, while unused coordinates may remain free.
Incompatible inputs raise the section contradiction exception with its witness.
Compilation neither executes a fragment nor chooses a free parameter.

The resulting `Program` retains linked input contracts and the source versions
used to determine its coefficients. Those sources must be bound again at
execution, under their contributor permission intersection and historical
evidence context. Schema three stores these contracts and dependencies.
Schemas one and two remain readable.

`PROGRAM_ASSEMBLY_RECORD` and `PROGRAM_ASSEMBLY_READ` persist the declaration.
`PROGRAM_FAMILY_RECORD` and `PROGRAM_FAMILY_READ` persist the section family
with its recipe. Family restoration requires the selected source and explicit
contributor bindings. RCDB, `.rcbd` and `.safetensors` use their existing state
codecs. `PROGRAM_RUN` remains the separate execution request.

## Program version comparison

`PROGRAM_COMPARE(old, new, matches)` compares two topology records of the same
kind: bound `PlanTopology` records or exact `RelationTopology` records.
`matches` contains one sequence of `(old_key, new_key)` pairs per retained
grade. Each sequence is an explicit partial bijection. Equal labels or array
shapes do not create implicit matches. Unmatched coordinates remain additions
or removals.

`PROGRAM_EVOLUTION_INFO` reports operation fields and endpoint declaration
changes, including retained captures, ports, effects, sources and outputs.
Grade two changes remain distinct where the endpoint topology retains them.
`PROGRAM_BOUNDARY_CHANGE(evolution, grade)` returns the Core factored action
`B_new J_grade - J_lower B_old`. It introduces no metric or eigensolve.

Captured constants can change while operation boundaries remain identical. A
zero boundary defect therefore describes boundary correspondence only.
PROGRAM_EVOLUTION_RECORD and PROGRAM_EVOLUTION_READ store both endpoint records
and the supplied occurrence matches and validate them on reload.

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

## Operator contracts

Built wheels include the Core operator implementations used by these contracts.

Core's numerical coupled field supports indefinite heat and wave evolution,
supplied initial velocity and sparse positive definite metrics. Propagation
applies numerical solvers and polynomial approximations; it does not evaluate
an exact matrix exponential.

CHARACTER(True) computes exact rational channel diagonals for raw G and for
normalized G with nonnegative relation weights. It reads them without
constructing an eigenbasis or full channel matrix. RANK and NULLITY use exact
elimination or structural identities and reject numeric matrices unsupported by
those methods.

Exact `QUADRANCE` and `SPREAD` require integer or rational coefficient carriers;
both SPREAD operands are checked. Numerical complex inputs use a Hermitian inner
product. A query is typed before any algebraic rewrite, including under `EXPLAIN`.

## Exact sheaf sections

Sheaf comparison accepts different stalk dimensions and rectangular incidence
maps. SECTION_CHECK checks compatibility directly on the section incidences.
GLUE returns the detailed residuals for pairwise comparisons. These operations
compare sections and do not compute sheaf cohomology or merge complexes.

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

`PARTITION` and `RESTRICT` default to `carried_state="structural"`, which omits
application state. `PARTITION(CELL(3,0), carried_state="all")` carries metadata,
signals, retained attachments and sectionings through Core component transport.
`TRAINING_PARTITION` accepts the same option in its explicit policy mapping;
the normalized policy digest binds the choice. All modes retain exact declared
shares and source metric presence, and all selected upper cells retain their full
downward closure.

Python `Selection` and `GradedCellPattern` parameters select several grades in
one `PARTITION`, `RESTRICT` or `QUOTIENT`. They must be bound to the current
source. `PARTITION(...).lineage` exposes declared read only state, sizes, requests,
maps, parent digests and content identity. Method calls are not query members.

`FROM $graph LET a=PARTITION(CELL(1,0)) LET b=PARTITION(CELL(1,1))
RETURN GLUE_PARTITIONS([a,b])` reconstructs their verified union in `$graph`'s
original basis. `carried_state="all"` requires all parts to carry application
state; modified or foreign parts are refused. This operation delegates to Core
`glue`; the existing `GLUE` operator retains its section contract. Portable result
artifacts and caches retain selections, shared source references and complete
lineage. A lineage content hash alone does not validate a derivation or authorize it.
`HASH(p.lineage,"lineage")`, `LINEAGE(p.lineage)` and `MANIFEST(p.lineage)` use
the complete lineage identity. For compatibility, the same operators on `p`
retain the older `PartitionState` summary and its distinct digest domain.

`FROM $graph RETURN SHOW_OPERATORS()` returns the current global inventory;
`SHOW_OPERATORS(25, 0)` paginates it. Python callers can use
`rcql.operator_inventory(limit=25, offset=0)` without loading the numerical stack.
Rows distinguish current expression contracts, FROM only forms and explicit
refusals.

Hodge, Green, winding and moment operators execute without SciPy imports.
The [dependency profiles](../DEPENDENCIES.md) make SciPy optional for Core and
RCDB. The homology readings add exact simple
and multiplicity dimensions at any carried grade, without harmonic bases.

## Ranking and participation walks

The participation walk uses relation incidence and weights to construct a
vertex transition operator. PageRank functions solve its fixed point, while
RESOLVENT_RANK applies a grade specific Hodge resolvent. The interfaces provide
the following calculations:

| Operator or function | Calculation |
| --- | --- |
| `PARTICIPATION_WALK()` | Vertex walk `U De^-1 U^T Dv^-1` through `U = abs(B1) W`, with relation weights as intensities |
| `PAGERANK_ITERATION(view)` | PageRank fixed point by power iteration on a participation walk |
| `PAGERANK_SOLVE(view)` | The same fixed point using rational arithmetic |
| `RESOLVENT_RANK(...)` | Hodge resolvent using the declared grade and metric |
| `TEXT_OVERLAP_VIEW()` | Raw overlap action `X^T X - diag(q)` on grade 1 relations |
| `rexgraph.core._standard` | Adjacency PageRank, betweenness, clustering, Louvain communities and Dijkstra shortest paths |

`MARKOV_VIEW`, `PAGERANK` and `PAGERANK_EXACT` were the former names; a query
that uses one is refused with the name that replaced it. In Python the former
names remain aliases: `rexgraph.markov.MarkovView` and `pagerank`,
`rexgraph.ranking_response.exact_pagerank` and `pagerank_delta`, and
`rexgraph.core._standard.pagerank`.

### Hodge resolvent ranking

```text
FROM $graph RETURN RESOLVENT_RANK(0, INDICATOR(CELL(0, 0)), 17/20, "walk")
FROM $graph RETURN RESOLVENT_RANK(1, damping=1/2)
FROM $graph RETURN RESOLVENT_RANK(1, calculus=$declared_calculus)
```

`RESOLVENT_RANK(grade, seed=None, damping=17/20, metric="declared", calculus=None,
metrics=None)` returns the exact Chain `(I + lam L_k)^-1 seed` with
`lam = damping / (1 - damping)`. Personalized PageRank solves
`(I - damping T) pi = (1 - damping) v`, which is this resolvent with `L = I - T`.
For loop free pair relations the walk metrics `M_0 = D^-1` and `M_1 = W^-1`, with
`D = diag(B_1 W B_1^T)`, make the grade zero Hodge operator equal to `I - A D^-1`,
so `RESOLVENT_RANK(0, seed, damping, "walk")` is exactly personalized PageRank on
the adjacency walk. The seed is nonnegative, normalized to unit mass and uniform
when omitted; damping is an exact rational in `[0, 1)`.

The selected grade and metric determine what the resolvent reads:

At grade one and above, RESOLVENT_RANK ranks relations and higher cells using
the next boundary operator B_(k+1). Harmonic components of the seed pass
through unchanged. Branching relations can retain differences between tail
participants as conserved directions, so a mass conserving response can have
signed entries.

The walk metric treats the declared grade one metric as resistance, with
conductance `W = M_1^-1`; the participation walk treats relation weights as
intensities. Uniformly scaling the weights leaves PAGERANK_SOLVE unchanged and
changes RESOLVENT_RANK.

Metric policies: `"declared"` uses the source's declared metrics (a RexGraph declares
its edge metric and the identity elsewhere); `"walk"` derives the grade zero metric as
above; `"degree"` derives every grade below the top the same way, from the top down;
`"completed"` replaces the ranked grade's metric with the completed field metric
`M_k (L_k + Pi^h)` of the complete field coordinates. A cell with no coface keeps its
declared metric under `"walk"` and `"degree"`. `metrics=[METRIC(k, $field), ...]`
replaces the policy's metric at those grades, so a metric can be read from a field,
and `calculus=$c` takes every grade's metric from a bound `NativeFieldCalculus` of
the same source.

### TextRank

TEXT_OVERLAP_VIEW computes raw overlap and omits TextRank's logarithmic length
weighting. In a document complex, sentences are grade 1 relations; use
DOCUMENT_FIELD for seed mass and coverage or RESOLVENT_RANK for grade 1
responses.

### Effective modes and the harmonic log

```text
FROM $graph RETURN EFFECTIVE_MODES(1), EFFECTIVE_MODES(1, "completed", "energy"), HARMONIC_LOG(1)
FROM $graph RETURN EFFECTIVE_MODES(1, calculus=$declared_calculus)
```

`EFFECTIVE_MODES(grade, sector="completed", weight="unit", calculus=None)` is the exact
count `tr(X)^2 / tr(X^2)` of one Hodge sector; `HARMONIC_LOG` is its float logarithm.
The sectors are `"completed"` (`L_k + w Pi^h`), `"hodge"` (`L_k`), `"down"` and `"up"`.
The completion weight is `"unit"` (`w = 1`, the complete field coordinates), `"mean"`
(the mean nonzero eigenvalue) or `"energy"` (the energy weighted mean eigenvalue, which
gives exactly `EFFECTIVE_MODES(k, "hodge") + BETTI(k)`). Only traces and the exact
Betti number are read.

## Native plans and execution provenance

Read queries execute a typed directed acyclic graph of operations. EXPLAIN
reports operator input and output types, checked conditions and selected or
deferred methods. Result.native_plan retains the plan, Result.execution records
the methods used, and Result.provenance identifies sources, types and rewrites
for each output. Result values and compatibility plan strings remain available.

## Channels, stars and scales

```text
FROM $graph RETURN APPLY(CHANNEL("F"), INDICATOR(CELL(1, 0)))
FROM $graph RETURN APPLY(CHANNEL("F"), INDICATOR(CELL(1, 0)), exact=true)
FROM $graph RETURN STAR_CHARACTER(CELL(0, 0), true)
FROM $graph RETURN SCALE_MOMENT(CHANNEL("G"), 1, true, true)
FROM $graph RETURN CHARACTER_ENERGY(CHANNEL("T"))
```

CHANNEL uses the source's coefficients without trace normalization. G follows
the raw or normalized source selection; F uses raw G. T, raw G, F and C support
exact rational actions and transposes through retained incidence factors. Exact
APPLY requires integer or rational input values and returns a grade 1 Field.
Declared fractions remain rational; source float weights are read as their
exact binary values. Normalized G can require irrational coefficients and does
not support a general exact rational action.

SCALE_MOMENT(operator, order, local=false, exact=false) returns a trace or
diagonal Cochain. Exact mode supports order zero and channel order one,
including normalized G diagonals. Higher numerical moments use sparse products
or row reductions and may introduce fill. Star character is the mean character
over incident relations.

## Explicit metrics and Green variants

```text
FROM $graph RETURN MOMENT($u, $v, METRIC(1, $weights), true)
FROM $graph RETURN QUADRANCE($u, true, METRIC(1, $weights))
FROM $graph RETURN SPREAD($u, $v, true, METRIC(1, $weights))
FROM $graph RETURN GREEN_SOLVE(RESOLVENT(CHANNEL("F"), 0.5), $u)
FROM $graph RETURN INTEGRATE($omega, $chain, exact=true)
```

INTEGRATE pairs a Cochain with a Chain on the same source, grade, ordered basis
and shape. It applies the bilinear pairing without an implicit metric or
complex conjugation. Exact mode uses rational arithmetic, and boundary and
coboundary actions satisfy finite Stokes.

Bind u, v and weights as grade 1 Cochains. METRIC(grade) supplies the identity
metric; explicit metric weights must be strictly positive and use the same
ordered basis. Integer and Fraction contractions remain exact. MOMENT returns
the signed or Hermitian contraction, including its sign or phase.

`RESOLVENT(L, alpha=1, tol=1e-10, maxiter=1000)` is the numerical matrix free
action `(I + alpha L)^-1` for a declared Euclidean symmetric PSD operator or a
native metric PSD Hodge down/up/sum. It is distinct from `GREEN()`'s C0
pseudoinverse. `GREEN_SOLVE` preserves vector/block shape and reports the observed
solver method. Weighted Hodge resolvents accept and return Chains; the existing
Euclidean Green actions retain their Cochain/Field contract.

`ADJOINT(operator, domain_metric=None, codomain_metric=None)` constructs
`M_domain^-1 operator* M_codomain`. The two metrics name the original
operator's endpoints; omitted metrics mean identity. For example:

```text
FROM $graph
LET a = ADJOINT(BOUNDARY(1), METRIC(1, $edge_weights), METRIC(0, $vertex_weights))
RETURN APPLY(a, $vertex_chain, exact=true)
```

Applying the adjoint of BOUNDARY raises a Chain's grade. The adjoint of
COBOUNDARY retains Cochain variance. APPLY preserves vector and block axes, and
exact=True supports rational adjoints, weighted Hodge and Dirac actions, and
brackets. Numerical Green solves use their solver tolerances. COBOUNDARY is the
metric independent dual transpose; a general adjoint has no automatic Euclidean
positive semidefinite certificate.

HODGE_DOWN, HODGE_UP, HODGE_SUM and HODGE_DIFFERENCE act on Chains with
positive diagonal metrics. They compose boundaries and metric adjoints, support
rational application and preserve empty sectors as exact zero. For example,
`APPLY(HODGE_SUM(1, metric=$m), $chain, exact=true)` returns a Chain. Metrics
at the selected grade and its neighbors default independently to identity.

These operators are self adjoint in their declared metric and need not be
Euclidean symmetric. RESOLVENT(HODGE_SUM(1, metric=$m)) solves `M(I+alpha
L)x=Mb` with inverse diagonal preconditioning and reports residuals in the
original equation. It applies retained factors without assembling a Gram
matrix, inverse or eigenbasis. The positive semidefinite resolvent solver does
not accept Hodge difference or Dirac.

`DIRAC(metrics=None)` and `ANTI_DIRAC(metrics=None)` act on the full carried
Chain tower. `GRADED_CHAIN([$chain0, $chain1])` preserves separate grade components
and zero fills omitted grades; `GRADE_COMPONENT(state, grade)` extracts one Chain.
`APPLY(DIRAC(), GRADED_CHAIN([$chain]), exact=true)` uses certified rational boundary
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

Rectangular coordinate maps support unequal output dimensions and sparse cross
blocks. CHAIN_MAP accepts a rexgraph.chain_map declaration containing graded
maps and target boundaries. Core composition and type accession conversion
check both chain laws and every commuting square with rational arithmetic.

## Local artifacts

Declared datasets are pinned explicitly by the application:

```python
from io import StringIO
from rcql import DatasetSource, run
from rexgraph import RelationSpec
from rexgraph.io.declaration import DatasetDeclaration
from rexgraph.io.records import RecordField, RecordSchema

declaration = DatasetDeclaration(
    "csv", RecordSchema((RecordField("a"), RecordField("b"))),
    RelationSpec("pair", ("a", "b")))
data = DatasetSource(declaration, StringIO("a,b\nx,y\n"))
result = run('FROM DATASET("edges") RETURN BETTI(0)',
             sources={"edges": data}, exactness="exact")
assert result.values == (1,)
```

DatasetSource uses Core's reader, construction and native state codecs. It owns
a snapshot; changing an input file or a previously materialized graph requires
a new binding. Its declaration/reader/state identity travels in SourceRef names
and cache keys. Policies are checked before materialization and retained by the
graph binding. Query text only resolves registered names, never paths. The
binding is a live capability; persist DatasetDeclaration and native records
through RCDB rather than serializing a live input handle.

`SourceSignature`, `SourceRegistry` and `SOURCES` provide the shared trusted source
classification registry. Explicit priorities preserve the existing source order;
equal priority matches refuse. Persisted names never load signature providers.

`run(text_or_query, sources=..., params=..., exactness="declared", cache=None)`
delegates to Executor and QueryCache. The default retains declared arithmetic.
`"exact"` accepts finite read only queries without MATCH, refuses effectful
operations before execution, and rejects approximate/rounded numeric outputs,
including mixed mappings, sequences and object arrays. It performs no numeric
conversion. Structural sources and descriptors retain their structural contract;
their annotations are not reinterpreted as numeric output. The request is retained
in provenance and cache identity. This policy does not change a numerical operator
into an exact implementation.

RCQL reads registered `FileCatalog` sources. It does not expose a general filesystem primitive. File names are catalog relative names and the catalog resolves and rechecks them before each access.

```text
FROM CATALOG("files") RETURN FILES(), SEARCH("document")
FROM FILE("files", "root0/document.rex") RETURN DESCRIBE(), STATE_HASH()
```

RCDB stores use the same source model. Read operators include `RCDB_LIST`, `RCDB_SEARCH`, `RCDB_GET`, `RCDB_HISTORY`, `RCDB_COMMITS`, `RCDB_VERIFY`, `RCDB_HASH`, `RCDB_STATS`, `RCDB_SECURITY`, and `RCDB_STATE_HASH`. The last hashes the versioned logical history, including canonical payload identities, metadata, validity and commit identities, not a backend index. It requires read, history and identity permissions and reads every visible published payload.

FROM RCDB("db") resolves a registered store name. Record selection, GET and
HASH use the selected version snapshot. The [RCDB
reference](../rcdb/REFERENCE.md) describes version selectors, provenance and
consistency within a store handle.

## Capabilities

A `BoundSource` carries a `SourcePolicy`. Signatures declare the requirements used by both EXPLAIN and execution; permissions are checked before an expression operator receives the underlying source. Derived `FILE` sources retain the catalog policy.

Exact record operations require identity permission. Mutation requires mutate
and identity; commit history requires history and identity. RCDB_SECURITY
requires security and admin. Rex computations also require read permission,
which identity inspection does not grant.

A source policy can also restrict which structural signature fields appear in record lists and searches.

## Mutations

Typed mutation requests use `MutationQuery`. They do not share the ordinary read operator registry.

Use `FROM $db MUTATE "id" SET state = $candidate COMMIT`, optionally supplying
an actor, validity endpoints and expected version. EXPLAIN checks the typed
input and commit plan without publishing. The plan retains its operations,
version guard and execution provenance.

A mutation lowers to the RCDB `commit_mutation` path, which constructs a TemporalRex mutation package and applies the store integrity policy before publication. A store configured with `require_commits=True` rejects ordinary `put` updates and raw deletion.
