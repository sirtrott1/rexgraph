# RexGraph Core API reference

See the [README](README.md) for installation and usage.

Core constructs relational complexes and provides boundary operators, homology,
Hodge decomposition, typed channels, exact fields and native I/O. `RexGraph`
exposes these through cached properties and operator actions. Compiled Cython
kernels supply sparse storage and numerical operations. Exact integer and
rational value containers preserve declared coefficients.

## Installation

From the repository root:

```sh
make install-core
make python
```

Use `sh install.sh --profile core` when Make is unavailable. To install the
application packages as well, use `make install`. The
[installation guide](../README.md#installation) covers prerequisites and profiles.
The Core build project is the repository root.

## Usage

```python
from rexgraph import RexGraph

rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
assert tuple(rex.betti) == (1, 1, 0)
assert rex.chain_valid
```

Sparse operators retain boundary factorizations and materialize matrices only
when requested. Exact ranks use integer or rational reduction. Numerical
propagation uses declared solver tolerances. Spectral properties and explicit
reference APIs return eigenvalues or dense matrices when requested.

Reference: [compiled kernels](core/README.md), [I/O](io/README.md),
[cochain models](flow/README.md) and [dependency profiles](../DEPENDENCIES.md).

## Declared data and exact I/O

Readers return typed records; `construct` maps declared fields to sparse
`Relations`. `RexGraph.from_relations` accepts those Relations in Core and
Agent. A branching relation remains one C1 cell. Head slots, tail shares,
metric weights, signs, IDs, types, vertex labels, aliases and source provenance
are independent declarations. An explicit `VertexTable` preserves isolated
vertices as well as participants.

```python
from fractions import Fraction as Q
from rexgraph import Relations, RexGraph, VertexTable

relations = Relations.from_supports(
    [[0, 1, 2, 3]],
    vertices=VertexTable(("head", "a", "b", "c", "isolate")),
    shares=[0, Q(1, 4), Q(1, 2), Q(1, 4)],
    weights=[Q(2, 3)], relation_ids=["assertion-1"],
)
rex = RexGraph.from_relations(relations)
assert rex.nV == 5 and rex.nE == 1
```

`Absent` has an explicit presence mask in `ExactArray`; it is distinct from
zero, one, `None` and `Approx`. A missing weight has a declared unit mathematical
view while its absence remains in `rex.relations.weight`. Source number rules
choose exact decimal, exact binary source values, rational JSON or typed
columnar values. Numeric text is never inferred from a string field.

```python
from rexgraph.construction import construct
from rexgraph.io.readers import read_batches
from rexgraph.io.records import RecordField, RecordSchema
from rexgraph import RelationSpec, NumberRule, ValueRules

schema = RecordSchema((
    RecordField("from", "string"), RecordField("to", "string"),
    RecordField("weight", "number", rules=ValueRules(NumberRule.DECIMAL_EXACT)),
))
records = read_batches("edges.csv", reader="csv", schema=schema)
relations = construct(records, RelationSpec("pair", ("from", "to"), weight="weight"))
rex = RexGraph.from_relations(relations)
```

The declared registry provides CSV, JSON, JSONL, Arrow IPC, Parquet, read only
SQLite and XLSX readers. Arrow/Parquet, HDF5, Zarr and XLSX use optional format
dependencies. `DatasetDeclaration` in `rexgraph.io.declaration` packages a reader,
schema, relation mapping, source options and optional vertex domain for reuse.
Row batches and record byte limits are configurable.

`rexgraph.state.to_state` writes format 10. The semantic record seals
dimensions, channel choices, component ownership, exact values and tensor
codecs together. Native readers reject tampering, unknown tensors and missing
components. Older state formats remain readable through the compatibility
decoder; `rexgraph.sealed_state.migrate_state` explicitly produces format 10.
RCBD, safetensors, HDF5, Zarr, Arrow IPC, wire transport and local RCDB backends
share this state contract. Cache data belongs to the container envelope.

Generic Parquet writers and exports accept `ExactArray` and integer/rational
object arrays. Typed coefficient structs preserve numeric kind, presence and
big integers through full reads, column projection and streaming batches.
An exact column containing an undeclared float is refused; callers choose a
source conversion rule before exporting it.

`copy()`, `subgraph`, functional insertion and deferred structural edits use the
same registered component transport. A restriction drops an upper cell when its
boundary loses a participant; it carries the remaining exact declarations,
identities, attributes, retained attachments, signals, section hierarchy,
provenance and embedding on their remapped bases. `subgraph` keeps vertices used
by surviving relations; `copy()` and insertion retain declared isolated vertices.
Copies own their mutable arrays and metadata. Retained model trees remain
immutable values.

Appends keep call order across pair and general arity batches. Omitted weights
remain absent with a unit mathematical view, while explicit zero and unit weights
retain presence. `set_cell_attrs(..., w_E=[Absent])` clears a declared weight;
setting it to `1` explicitly marks it present. New signal rows and embedding
positions have explicit absence until supplied; new cells are not automatically
assigned to existing sections. Native weight presence has its own sealed tensor.
Previous v10 relation records remain readable and migrate on the next write.

`Selection(rex, {0: [isolated_vertex], 3: [process]})` requests original cell
addresses across grades. `Selection.from_masks` checks binary masks before
conversion; `restrict(rex, selection, carried_state="all")` returns an owned
`RexPartition` through that same transport. Structural state is the default.
Its `lineage.cell_maps[k][i]` maps result cells to original cells;
`lineage.old_to_new[k][j]` maps original cells to the result or -1.
`lineage.verify(source, result)` reproduces the restriction and checks exact
state identity. Consecutive lineage records compose when intermediate states
and basis sizes agree. Digests identify content; they do not grant authorization.

`glue(parts, source=original)` reconstructs the union of verified restrictions
in the original basis, using the explicit original source for overlap. Edited
or foreign parts are refused. Mixed transport modes require an explicit policy;
omitted application state cannot be restored by choosing `all`. Portable
partitions contain lineage and their result, without a hidden original graph.
Older portable partitions remain readable but have no certified original basis.

An explicitly empty C2 space survives restriction and native I/O. Format 10
records it with the optional sealed `empty_face_grade` declaration when no higher
grade already implies that space. Earlier native readers refuse that new header
field. Earlier records remain readable; the legacy writer refuses this
unrepresentable empty grade declaration instead of losing it.

Simplicial, graded cell and adjacency imports use the common relation carrier.
Adjacency imports preserve the declared matrix size and every selected weight,
including near unit values. Triangles with ambiguous parallel relations require
explicit face edges. `to_dict` includes sealed native state alongside its legacy
primary projections; `from_dict` checks their agreement and still reads older
primary only dictionaries.

## `graph.py`: RexGraph and TemporalRex

### Construction

| Method | Description |
|---|---|
| `RexGraph(boundary_ptr, boundary_idx, ...)` | Direct constructor from CSR boundary |
| `RexGraph.from_graph(sources, targets, ...)` | Standard edge list |
| `RexGraph.from_simplicial(sources, targets, triangles)` | With filled faces |
| `RexGraph.from_hypergraph(hyperedge_ptr, hyperedge_idx)` | Relations from compressed support arrays |
| `RexGraph.from_adjacency(A, directed=False)` | From adjacency matrix |
| `RexGraph.from_dict(d)` | From serialization dict |

### Core properties

| Property | Returns |
|---|---|
| `nV`, `nE`, `nF` | Vertex, edge, face counts |
| `dimension` | Highest nonempty cell dimension |
| `boundary_ptr`, `boundary_idx` | CSR boundary operator |
| `sources`, `targets` | Standard edge endpoints (None for general) |
| `w_E` | Edge weights or None |
| `edge_types` | int8 per edge type: standard(0), self_loop(1), branching(2), witness(3) |
| `degree`, `in_degree`, `out_degree` | Vertex degrees |

### Boundary operators and homology

`rexgraph.column_expansion.ColumnExpansion` factors an exact boundary without
changing its primary cell space. `primary_lift` applies the retained factors.
`rexgraph.cell_neighborhood.hyperslice` returns typed incidence selections
through every carried grade.

| Property/Method | Returns |
|---|---|
| `B1`, `B2` | Dense numerical boundary views |
| `B1_sparse`, `B2_sparse` | Native sparse boundary carriers; B2 is None for an empty C2 grade |
| `B2_hodge`, `B2_hodge_sparse` | Dense and sparse views of C2 columns passing exact boundary closure |
| `chain_valid` | True if B1 @ B2 = 0 |
| `betti` | (b0, b1, b2) Betti numbers |
| `euler_characteristic` | nV - nE + nF |

### Laplacians and spectral

| Property | Returns |
|---|---|
| `L0`, `L1`, `L2` | Hodge Laplacians (dimensions 0, 1, 2) |
| `L_overlap` | Overlap Laplacian |
| `spectral_bundle` | Full spectral decomposition dict |
| `eigenvalues_L0` | L0 eigenvalues |
| `fiedler_vector_L0` | Second smallest eigenvector of L0 |
| `fiedler_overlap` | (value, vector) from L_overlap |
| `layout` | 2D spectral layout f64[nV, 2] |

### Relational

| Property | Returns |
|---|---|
| `relational_laplacian` (RL) | Relational Laplacian nE x nE |
| `coupling_constants` | (alpha_G, alpha_T) |
| `evals_RL1`, `evecs_RL1` | RL eigendecomposition |
| `structural_character` | Edge chi f64[nE, nhats] |
| `vertex_character` | Vertex phi f64[nV, nhats] |
| `coherence` | Vertex kappa f64[nV] |
| `local_coherence` | O(nnz) per vertex kappa_loc(v) from the star of v, no eigensolve |
| `inverse_centrality_ratio` | mu(v) = median(deg) / deg(v), f64[nV] |
| `per_channel_mixing_times` | mu_X = ln(nE) / lambda_2(hat_X), f64[nhats] |
| `energy_character` | Per edge energy diag(RL4^2)_e = row norms of RL4, no eigensolve |
| `star_character` | chi*(v) = mean of chi(e) over edges incident to v, f64[nV, nhats] |

### Sparse and eigen free operators

Sparse and eigen free forms of the spectral quantities, so large complexes
avoid densification and eigensolves.

| Property | Returns |
|---|---|
| `B1_dense`, `B2_dense` | Dense boundary matrices (the sparse CSR forms are the default) |
| `L0_sparse`, `L1_sparse`, `L2_sparse` | Hodge Laplacians as scipy CSR, no densification |
| `overlap_gramian`, `overlap_gramian_sparse` | Numerical weighted unsigned Gramian W\|B1\|^T\|B1\|W for the geometry G channel in the down sector |
| `greens_diagonal_eigenfree` | Numerical diag(RL4^-1) from block CG solves of RL4 X = I to a fixed tolerance |
| `harmonic_entropy` | Eigen free Renyi 2 harmonic log H2(RL4) = -log(tr(RL4^2)/tr(RL4)^2) |
| `harmonic_space` | Basis for ker(L1) |

### Typed cochains, graded operators, and Green actions

`Cochain` keeps a value array together with its grade and optional ordered cell
basis; `Field` attaches the operator that gives those values their meaning, and
`GradedState` collects cochains by grade. `boundary_operator` and
`coboundary_operator` expose the full `B_1, B_2, B_3, ...` tower, while
`down_laplacian`, `up_laplacian`, and `hodge_operator` retain their sparse boundary
factorizations and materialize a CSR product only through `as_scipy()`.

`GreenOperator.resolvent(L, alpha)` applies `(I + alpha L)^-1` without forming an
inverse. `vertex_green(rex)` applies the Moore Penrose action `(B_1 B_1^T)^+`;
it uses the factored deflated solve for pairwise complexes and a minimum norm
matrix free solve for branching relations, whose kernel is not generally spanned
by graph component indicators.

### Channel readings

`channel_operator(rex, channel)` retains the declared coefficients for exact
integer and rational actions. Use `apply(values, exact=True)` to request that
arithmetic. The matrix properties below are numerical compatibility views.
For branching relations, coefficients include their declared rational shares.

| Property | Returns |
|---|---|
| `g_channel` | Selected G (overlap) form: 'raw' (default, exact) or 'normalized' (opt in, takes a square root) |
| `g_channel_operator` | Selected numerical geometry G channel view: raw unsigned Gramian or normalized overlap |
| `frustration_exact` | SciPy sparse frustration view: off diagonal T minus G, with absolute row sums on the diagonal |
| `L_frustration` | Dense numerical view of `frustration_exact` |
| `L_frustration_weighted` | Inverse log degree weighted signed Gramian frustration (weighted variant) |
| `L_coPC` | Copath complex Laplacian L_C (line graph Hodge) |

### Scale free structure and moments

Local to global structure computed from closed walk moments rather than a
global eigenbasis.

| Property/Method | Returns |
|---|---|
| `vertex_scale_profile` | Local scale character from moments of closed walks of length k, per vertex |
| `scale_bridge` | Local to global structure across the scale profile |
| `heat_character(t)` | Chebyshev approximation of the heat response diagonal `diag(exp(-t RL4))` |
| `weighted_curvature_signature(...)` | Weighted curvature signature from moments |

### Self diagnostics

| Property | Returns |
|---|---|
| `character_varentropy` | The `H2 - H3` varentropy gap used to check character reliability |
| `character_reliability` | Reliability flag {H2, H3, shannon_est, gap} derived from the varentropy |

### Hodge decomposition

| Method | Returns |
|---|---|
| `hodge(g)` | (gradient, curl, harmonic) components |
| `hodge_full(g)` | Full dict with energies, fractions, spectral info |

### Interfacing and channel analysis

| Method | Description |
|---|---|
| `interfacing_vector(target_indices, target_weights, target_signal, ...)` | Full interfacing vector with confidence diagnostics |
| `primal_signal_character(psi)` | Energy decomposition across typed channels, f64[nhats] summing to 1 |
| `spectral_channel_score(source, target)` | Spectral propagation score through RL eigenmodes |
| `face_void_dipole(psi)` | Face vs void affinity of an edge signal |

### Face selection

| Method | Description |
|---|---|
| `typed_face_selection(edge_type_labels)` | New RexGraph with same type triangles as faces |
| `context_face_selection(context_matrix)` | New RexGraph with context covered triangles as faces |

### Filtration and linkage

| Method | Description |
|---|---|
| `quotient_filtration(channel, n_steps=20)` | Character based edge removal tracking Betti numbers |
| `linkage_complex(sfb_threshold=0.85)` | New RexGraph from thresholded fiber bundle similarity |

### Signal and state

| Method | Description |
|---|---|
| `signal(dim, values)` | Create edge/vertex/face signal |
| `signal_energy(g, dim)` | Quadratic form energy |
| `normalize(g, norm)` | L1/L2 normalization |
| `create_state(t)` | RexState with zero signals |
| `energy_kin_pot(f_E)` | (total, kinetic, potential) energy triple |
| `dirac_state(dim, idx)` | Delta signal at a single cell |
| `uniform_state(norm)` | Uniform (f0, f1, f2) |

### Perturbation analysis

| Method | Description |
|---|---|
| `edge_perturbation(edge_idx)` | Perturb single edge |
| `vertex_perturbation(vertex_idx)` | Perturb single vertex |
| `multi_edge_perturbation(edge_indices)` | Multi edge perturbation |
| `spectral_perturbation(mode_idx)` | Eigenvector based perturbation |
| `analyze_perturbation(...)` | Full perturbation trajectory analysis |
| `analyze_perturbation_field(...)` | Field based perturbation analysis |

### Dynamics

| Method | Description |
|---|---|
| `evolve_markov(g, dim, t)` | Continuous time Markov diffusion |
| `evolve_schrodinger(psi, dim, t)` | Schrodinger evolution |
| `evolve_coupled(state, dt, steps)` | Coupled 0-1-2 RK4 evolution |
| `evolve_field_wave(psi, dim, t, ...)` | Field based wave evolution |
| `evolve_field_trajectory(psi, dim, times)` | Field wave trajectory |

### Quantum / wave mechanics

| Method | Description |
|---|---|
| `wave_state(dim, amplitudes)` | Complex amplitude state |
| `measure(psi, dim)` | Projective measurement |
| `born_probabilities(psi)` | Born rule |
| `entanglement_entropy(psi, dim_A, dim_B)` | Bipartite entanglement |
| `measure_in_eigenbasis(psi, dim)` | Eigenbasis measurement |
| `graded_state(t, ...)` | Cross dimensional graded state |
| `graded_trajectory(times, ...)` | Graded evolution trajectory |

### Field theory

| Method/Property | Description |
|---|---|
| `field_operator` | (M, coupling, hermitian) field operator |
| `field_eigen` | (eigenvalues, eigenvectors, gaps) |
| `field_diffuse(F0, times)` | Diffusion through field operator |
| `field_wave_evolve(F0, dFdt0, times)` | Position and velocity trajectories from both initial conditions |
| `classify_modes()` | Mode classification dict |

### Dirac and graded operators

| Method/Property | Description |
|---|---|
| `dirac_operator` | Full Dirac operator matrix property |
| `dirac_eigenvalues` | Dirac spectrum |
| `canonical_collapse(vertex_idx)` | Graded to vertex projection |
| `born_graded(psi_re, psi_im)` | Per dimension Born probabilities |
| `energy_partition(psi_re, psi_im)` | Graded energy breakdown |

### Topological constructions

| Method | Description |
|---|---|
| `fill_cycle(cycle_edges)` | Add a face filling a cycle |
| `promote()` | Detect and fill all minimal 3 cycles |
| `face_data(...)` | Detected face data with metrics |
| `hypermanifold()` | Fiber bundle structure dict |
| `harmonic_shadow()` | Harmonic content across dimensions |
| `dimensional_subsumption()` | Tests if lower dims are captured by higher |

### Subcomplex and quotient

| Method | Description |
|---|---|
| `subcomplex(v_mask, e_mask, f_mask)` | Extract subcomplex |
| `quotient(v_mask, e_mask, f_mask)` | Quotient complex |
| `subcomplex_by_energy(f_E, ...)` | Energy based subcomplex |
| `star_of_vertex(v)` / `star_of_edge(e)` | Star neighborhoods |
| `validate_subcomplex(...)` | Check closure properties |
| `hyperslice(dim, idx)` | Cross section at a cell |
| `hyperslice_telescope(dim, idx, depth)` | Nested hyperslices |
| `edge_type_quotient(...)` | Quotient by edge type |
| `hyperslice_quotient(...)` | Quotient from hyperslice |
| `relative_cycle_basis(...)` | Relative homology basis |
| `connecting_homomorphism(...)` | Long exact sequence map |
| `restrict_signal(...)` / `lift_signal(...)` | Signal transfer |
| `restrict_field_state(...)` / `lift_field_state(...)` | Field state transfer |
| `congruence_classes(dim, mask)` | Equivalence classes |
| `quotient_hodge(...)` | Hodge decomposition on quotient |
| `quotient_analysis(...)` | Full quotient analysis dict |

### Graph operations

| Method | Description |
|---|---|
| `relation_supports()` | Each relation's boundary vertices, at any arity |
| `subgraph(edge_mask)` | Subcomplex over the kept relations, arity preserved |
| `partition_communities()` | Louvain community partition -> list of (sub_rex, v_map, e_map) |
| `insert_relations(supports)` | Add relations of any arity |
| `insert_edges(src, tgt)` | Add 2 ary relations (the pairwise case of the above) |
| `delete_edges(mask)` | Remove relations (the complement of `subgraph`) |
| `inner_join(other, shared)` / `outer_join(...)` / `left_join(...)` | Complex joins |

### Persistence

| Method | Description |
|---|---|
| `filtration(kind=...)` | Build filtration (vertex_sublevel, edge_sublevel, dimension, ...) |
| `persistence(filt_v, filt_e, filt_f)` | Persistence diagram dict |
| `persistence_barcodes(result, dim)` | Extract barcode array |
| `persistence_landscape(barcodes, grid, k_max)` | Landscape function |
| `persistence_distance(dgm1, dgm2, metric)` | Bottleneck/Wasserstein distance |
| `persistence_entropy(barcodes)` | Persistence entropy |
| `enrich_persistence(result)` | Add edge type and Hodge enrichment |

### Query engine

| Method | Description |
|---|---|
| `impute(signal, mask)` | Harmonic interpolation of missing values |
| `explain(dim, idx)` | Full diagnostic for a single cell |
| `propagate(source, target)` | Spectral propagation score |

### Demand driven (seed based) reads

These methods return quantities at a seed set or a selected neighborhood
without building the complete character or eigenbasis. Their sparse operator
actions still depend on the size and structure of the supplied complex.

| Method | Description |
|---|---|
| `character_response(seed, ...)` | Per vertex character phi(v,k) at just the seed vertices |
| `coherence_response(seed, ...)` | Coherence kappa at a seed set, by diffusion |
| `local_context(seed, ...)` | Bounded local context around seed vertices, by heat diffusion |
| `propagate_signal(f0, ...)` | Dynamic vertex propagator S0 = B1 RL+ B1^T applied to a signal |
| `effective_resistance(edge)` | R_eff(e) = b_e^T L0+ b_e for one edge |

### Agentic reading

| Method | Description |
|---|---|
| `agentic_reading(...)` | Context and structural measurements for the query's entities and relations |
| `explain_context(...)` | Contextual picture around query entities (vertices) and relations (edges) |

### Similarity and overlap

| Property | Returns |
|---|---|
| `overlap_similarity`, `overlap_pairs` | Normalized overlap similarity S in [0,1], and the k highest scoring edge pairs |
| `phi_similarity` | Vertex character similarity S_phi[i,j] = 1 - 0.5\|phi_i - phi_j\|_1 |
| `fiber_similarity` | Fiber bundle similarity S_fb[i,j] between vertices |

### Serialization

| Method | Returns |
|---|---|
| `to_json()` | JSON safe dict |
| `to_dict()` | Reconstruction dict |
| `from_dict(d)` | Class method: reconstruct from dict |

### Dashboard data

| Method | Description |
|---|---|
| `signal_dashboard_data(signal, ...)` | Full dashboard payload for signal visualization |
| `quotient_dashboard_data(...)` | Dashboard payload for quotient visualization |

### Standalone functions

| Function | Description |
|---|---|
| `cross_complex_bridge(rex_A, rex_B, labels_A, labels_B, ...)` | Cross complex bridge analysis between two relational complexes. Aligns by shared vertex labels, compares kappa correlation, void fractions, and optionally spectral channel scores. |

## TemporalRex

TemporalRex stores a sequence of pairwise or general boundary snapshots,
retaining relation identity across steps and storing changes as deltas.

### Properties and methods

| Method/Property | Description |
|---|---|
| `T` | Number of timesteps |
| `at(t)` | RexGraph snapshot at timestep t |
| `temporal_index()` | Build temporal index structure |
| `edge_lifecycle()` | (edge_ids, birth, death) arrays |
| `edge_metrics()` | (counts, born, died) per timestep |
| `face_lifecycle_data()` | Face lifecycle if face snapshots exist |
| `bioes(...)` | BIOES phase tagging (Begin/Inside/Outside/End/Single) |
| `bioes_energy(...)` | Energy ratio based BIOES |
| `bioes_joint(...)` | Joint structural + energy BIOES |
| `temporal_persistence(...)` | Persistence on final snapshot |
| `cascade_activation(...)` | Cascade activation from source edges |
| `cascade_wavefront(...)` | Wavefront propagation |

## `rextypes.py`: Typed containers and enumerations

Enumerations name the integer codes used by compiled kernels. NamedTuples
provide attribute access to returned dictionaries and tuples and are discovered
by rexgraph.io._serialization for storage.

### Enumerations (9)

| Enum | Values | Source |
|---|---|---|
| `EdgeType` | STANDARD(0), SELF_LOOP(1), BRANCHING(2), WITNESS(3) | `_rex.pyx` |
| `HodgeComponent` | GRADIENT(0), CURL(1), HARMONIC(2) | `_hodge.pyx` |
| `BioesTag` | BEGIN(0), INSIDE(1), OUTSIDE(2), END(3), SINGLE(4) | `_temporal.pyx` |
| `FaceEvent` | PERSIST(0), BORN(1), DIED(2), SPLIT(3), MERGE(4) | `_temporal.pyx` |
| `TransitionKind` | MARKOV(0), SCHRODINGER(1), DIFFERENTIAL(2), REWRITE(3) | `_transition.pyx` |
| `EnergyRegime` | KINETIC(0), CROSSOVER(1), POTENTIAL(2) | `_temporal.pyx` |
| `OperatorChannel` | TOPOLOGICAL(0), GEOMETRIC(1), FRUSTRATION(2), COPC(3) | `_relational.pyx` |
| `PredicateOp` | GT(0)..BETWEEN(6) | `_query.pyx` |
| `JoinType` | INNER(0), LEFT(1), OUTER(2), UNION(3) | `_joins.pyx` |

### NamedTuples (44)

Hodge: HodgeDecomposition, HodgeAnalysis

Spectral: SpectralBundle: RL (relational Laplacian from build_RL), hats,
nhats, hat_names, trace_values, chi (structural character), K1 (overlap matrix),
L_C (copath complex Laplacian), and `RL_1 = L1_down + alpha_G * L1_up`.

Relational: RCFBundle (RL, hats, nhats, chi, trace_values, hat_names),
VertexBundle (phi, chi_star, kappa), VoidComplex (tri_edges, void_indices),
RCFEResult

Energy: EnergyDecomposition, EnergyKinPot, WaveEnergy

Field: FieldOperatorResult, FieldEigenResult, ModeClassification

Signal: CascadeResult, FaceEmergenceResult, PerturbationResult, FieldPerturbationResult

Quotient: SubComplex, QuotientMaps, QuotientResult

Hyperslice: HypersliceVertex, HypersliceEdge, HypersliceFace

Persistence: PersistenceDiagram, Filtration, PersistenceEnrichment

Temporal: BioesEnergyResult, BioesResult

State: WaveState, SchrodingerState

Interfacing: InterfacingResult (rho, psi, scores, schrodinger, iv, sphere_pos, signal_magnitude, coverage, efficiency, confidence)

Channels: ChannelProfile (iv_T/G/F/Sch, pc_T/G/F, coverage, kappa_mean, efficiency)

Filtration: FiltrationResult (thresholds, beta0/1/2, n_edges_remaining, edges_removed_order, transition_index, transition_threshold)

Cross Complex: CrossComplexBridge (kappa, void, n_shared, channel=None)

Other: MeasurementResult, FaceData, CoupledEvolution, StandardMetrics, JoinResult, ImputeResult, CellExplanation, PropagationResult, RemovalStep, StructuralSummary

## `analysis.py`: Dashboard analysis pipeline

Produces the graph dashboard's JSON data from a RexGraph, optional vertex
labels and edge attributes. The analysis uses RexGraph properties, operator
modules and compiled kernels.

### `analyze(rex, *, vertex_labels=None, edge_attrs=None, negative_types=None, ...)`

Full structural analysis for the graph dashboard. Returns a dict with:

- meta: nV, nE, nF, canvas size, attribute metadata, type/weight columns
- vertices: per vertex: position, degree, role, partitions, Fiedler, divergence, PageRank, betweenness, clustering, community, face metrics
- edges: per edge: source/target, type, weight, Hodge components (raw + normalized), rho, L1_down/L1_up diagonal, face metrics, betweenness, E_kin/E_pot
- faces: per face: id, boundary, vertices, size, curl, concentration
- topology: chain validity, ranks, Betti, Euler, nF_hodge, self loop faces
- coupling: alpha_G, alpha_T, Fiedler values for L1, L_O, RL_1
- spectra: eigenvalue arrays for L0, L1_down, L1_full, L2, L_O, RL_1, RL, field M
- hodge: gradient/curl/harmonic norms and percentages
- energy: E_kin, E_pot, ratio, regime, top kinetic/potential edges
- analysis: topology summary, Hodge summary, coupling interpretation, energy regime, structural roles, partition splits, standard metrics (PageRank, betweenness, clustering, Louvain), face structure
- overlap: top overlap pairs with Jaccard similarity
- structural_character: per edge chi, per vertex phi + kappa, aggregate summary, per channel mixing times, mixing time anisotropy, inverse centrality ratio (when the relational modules are available)
- rcfe: curvature per edge, strain, weighted boundary residual check, attributed curvature, strain equilibrium
- void_complex: n_voids, n_potential, void strain, mean eta, fills_beta count, realization rate
- fiber_bundle: mean phi/fiber similarity, top vertex pairs by fiber similarity
- channels: primal signal character of the flow signal, face void dipole (when `_channels` module available)
- dirac: Dirac spectrum, zero mode count, Born probabilities per dimension, energy partition
- hypermanifold: manifold sequence with Betti/DOF per level, harmonic shadow, dimensional subsumption check
- perturbation: probe edge, cascade depth, activation count, energy/Hodge trajectories (when run_perturbation=True)
- diffusion: field diffusion norm decay, equilibrium index, vertex observables (when faces present)

### `analyze_signal(rex, *, probe_edges=None, n_steps=50, t_max=10.0, ...)`

Calls `analyze()` for base data, then appends signal specific data from
`rex.signal_dashboard_data()`: perturbation trajectories, field diffusion,
mode classification, BIOES tags, cascade activation.

### `analyze_quotient(rex, *, max_vertex_presets=8, ...)`

Calls `analyze()` for base data, then appends quotient presets from
`rex.quotient_dashboard_data()`: vertex star quotients, edge type
subcomplexes, energy regime subcomplexes with Hodge decomposition,
congruence classes, relative Betti numbers.

### `analyze_all(rex, *, probe_edges=None, signal_steps=50, max_vertex_presets=8, ...)`

Runs `analyze()` once, appends both signal and quotient data. More
efficient than calling `analyze_signal` + `analyze_quotient` separately
because the base analysis is shared.

### Helpers

- `_build_flow(rex, edge_attrs, negative_types, nE)`: signed flow signal from edge attributes. Magnitude from w_E or weight column, sign from polarity matching against negative stem patterns.
- `_classify_attributes(edge_attrs, nE)`: attribute metadata in dashboard format (numeric with min/max/mean, categorical with values/counts).
- `_partition_from_fiedler(vec)`: Fiedler bipartition labels ("A"/"B").
- `_vertex_roles(in_deg, out_deg, v_face_count, nV)`: structural role assignment (source/sink/hub/mediator/peripheral).
