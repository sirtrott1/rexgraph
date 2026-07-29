# rexgraph

**Orchestration layer for relational complexes.**

A k-rex is a graded cell complex in which edges are primary and vertices are
derived from their boundaries. `RexGraph` is the single object that composes the
39 Cython kernels in `rexgraph.core` through lazily-cached accessors. No kernel
imports another; all composition happens in `graph.py`.

Two layers sit under that object. The **kernels** in `rexgraph.core` build the
structures (boundaries, Laplacians, character, Hodge, persistence, and so on) and
are optimized to run on sparse operators. The **eigen-free modules** in
`rexgraph` (`scale_propagator`, `sparse_character`, `harmonic_sparse`,
`graded_boundary`, `field_propagator`, `dirac_propagator`, `sparse_interfacing`)
compute the spectral quantities without a dense eigensolve, so the same results
hold at any scale. Sparse is the default everywhere: `RexGraph` does not build
the dense relational bundle unless it is asked for. Dense is materialized on
demand for the low-level dense kernels, and it is the exact reference the sparse
path is tested against. The `rexgraph.compute` layer then dispatches across CPU,
GPU, or multiple GPUs.

---

## `graph.py`: RexGraph and TemporalRex

### Construction

| Method | Description |
|---|---|
| `RexGraph(boundary_ptr, boundary_idx, ...)` | Direct constructor from CSR boundary |
| `RexGraph.from_graph(sources, targets, ...)` | Standard edge list |
| `RexGraph.from_simplicial(sources, targets, triangles)` | With filled faces |
| `RexGraph.from_hypergraph(edges)` | Hyperedges as variable-length boundaries |
| `RexGraph.from_adjacency(A, directed=False)` | From adjacency matrix |
| `RexGraph.from_dict(d)` | From serialization dict |

### Core Properties

| Property | Returns |
|---|---|
| `nV`, `nE`, `nF` | Vertex, edge, face counts |
| `dimension` | Highest nonempty cell dimension |
| `boundary_ptr`, `boundary_idx` | CSR boundary operator |
| `sources`, `targets` | Standard-edge endpoints (None for general) |
| `w_E` | Edge weights or None |
| `edge_types` | int8 per-edge type: standard(0), self_loop(1), branching(2), witness(3) |
| `degree`, `in_degree`, `out_degree` | Vertex degrees |

### Boundary Operators and Homology

| Property/Method | Returns |
|---|---|
| `B1` | Vertex-edge boundary f64[nV, nE] |
| `B2` | Edge-face boundary (raw CSC) |
| `B2_hodge` | Edge-face boundary including self-loop faces |
| `chain_valid` | True if B1 @ B2 = 0 |
| `betti` | (b0, b1, b2) Betti numbers |
| `euler_characteristic` | nV - nE + nF |

### Laplacians and Spectral

| Property | Returns |
|---|---|
| `L0`, `L1`, `L2` | Hodge Laplacians (dimensions 0, 1, 2) |
| `L_overlap` | Overlap Laplacian |
| `spectral_bundle` | Full spectral decomposition dict |
| `eigenvalues_L0` | L0 eigenvalues |
| `fiedler_vector_L0` | Second-smallest eigenvector of L0 |
| `fiedler_overlap` | (value, vector) from L_overlap |
| `layout` | 2D spectral layout f64[nV, 2] |

### Relational (RCF v2)

| Property | Returns |
|---|---|
| `relational_laplacian` (RL) | Relational Laplacian nE x nE |
| `coupling_constants` | (alpha_G, alpha_T) |
| `evals_RL1`, `evecs_RL1` | RL eigendecomposition |
| `structural_character` | Edge chi f64[nE, nhats] |
| `vertex_character` | Vertex phi f64[nV, nhats] |
| `coherence` | Vertex kappa f64[nV] |
| `local_coherence` | O(nnz) per-vertex kappa_loc(v) from the star of v, no eigensolve |
| `inverse_centrality_ratio` | mu(v) = median(deg) / deg(v), f64[nV] |
| `per_channel_mixing_times` | mu_X = ln(nE) / lambda_2(hat_X), f64[nhats] |
| `energy_character` | Per-edge energy diag(RL4^2)_e = row-norms of RL4, no eigensolve |
| `star_character` | chi*(v) = mean of chi(e) over edges incident to v, f64[nV, nhats] |

### Sparse and eigen-free operators

Sparse and eigen-free forms of the spectral quantities, so large complexes
avoid densification and eigensolves.

| Property | Returns |
|---|---|
| `B1_dense`, `B2_dense` | Dense boundary matrices (the sparse CSR forms are the default) |
| `L0_sparse`, `L1_sparse`, `L2_sparse` | Hodge Laplacians as scipy CSR, no densification |
| `overlap_gramian`, `overlap_gramian_sparse` | Raw integer overlap Gramian K = \|B1\|^T \|B1\| (the canonical G channel) |
| `greens_diagonal_eigenfree` | diag(RL4^-1) exact via block-CG solves of RL4 X = I, no eigendecomposition |
| `harmonic_entropy` | Eigen-free Renyi-2 harmonic log H2(RL4) = -log(tr(RL4^2)/tr(RL4)^2) |
| `harmonic_space` | Basis for ker(L1) |

### Exact / integer channels

The typed channels are built from integer boundary counts, so the tower is
exact rather than float-accumulated.

| Property | Returns |
|---|---|
| `g_channel` | Selected G (overlap) form: 'normalized' (default) or 'raw' |
| `g_channel_operator` | The raw integer G operator the RL4 character consumes |
| `frustration_exact` | Doc-exact integer frustration channel F = T - G (Def 3.3), sparse |
| `L_frustration` | Frustration channel F = T - G, dense integer tower |
| `L_frustration_weighted` | Inverse-log-degree weighted signed-Gramian frustration (weighted variant) |
| `L_coPC` | Copath complex Laplacian L_C (line-graph Hodge) |

### Scale-free structure and moments

Local-to-global structure computed from closed-walk moments rather than a
global eigenbasis.

| Property/Method | Returns |
|---|---|
| `vertex_scale_profile` | Local scale character from closed-k-walk moments, per vertex |
| `scale_bridge` | Local-to-global structure across the scale profile |
| `heat_character(t)` | Scale-resolved character diag(e^{-t RL4}), the general-f heat propagator |
| `weighted_curvature_signature(...)` | Weighted geometric signature (curvature-weighted moments) |

### Self-diagnostics

| Property | Returns |
|---|---|
| `character_varentropy` | The H2-H3 varentropy gap, a self-diagnostic on character reliability |
| `character_reliability` | Reliability flag {H2, H3, shannon_est, gap} derived from the varentropy |

### Hodge Decomposition

| Method | Returns |
|---|---|
| `hodge(g)` | (gradient, curl, harmonic) components |
| `hodge_full(g)` | Full dict with energies, fractions, spectral info |

### Interfacing and Channel Analysis

| Method | Description |
|---|---|
| `interfacing_vector(target_indices, target_weights, target_signal, ...)` | Full interfacing vector with confidence diagnostics |
| `primal_signal_character(psi)` | Energy decomposition across typed channels, f64[nhats] summing to 1 |
| `spectral_channel_score(source, target)` | Spectral propagation score through RL eigenmodes |
| `face_void_dipole(psi)` | Face vs void affinity of an edge signal |

### Face Selection

| Method | Description |
|---|---|
| `typed_face_selection(edge_type_labels)` | New RexGraph with same-type triangles as faces |
| `context_face_selection(context_matrix)` | New RexGraph with context-covered triangles as faces |

### Filtration and Linkage

| Method | Description |
|---|---|
| `quotient_filtration(channel, n_steps=20)` | Character-based edge removal tracking Betti numbers |
| `linkage_complex(sfb_threshold=0.85)` | New RexGraph from thresholded fiber bundle similarity |

### Signal and State

| Method | Description |
|---|---|
| `signal(dim, values)` | Create edge/vertex/face signal |
| `signal_energy(g, dim)` | Quadratic form energy |
| `normalize(g, norm)` | L1/L2 normalization |
| `create_state(t)` | RexState with zero signals |
| `energy_kin_pot(f_E)` | (total, kinetic, potential) energy triple |
| `dirac_state(dim, idx)` | Delta signal at a single cell |
| `uniform_state(norm)` | Uniform (f0, f1, f2) |

### Perturbation Analysis

| Method | Description |
|---|---|
| `edge_perturbation(edge_idx)` | Perturb single edge |
| `vertex_perturbation(vertex_idx)` | Perturb single vertex |
| `multi_edge_perturbation(edge_indices)` | Multi-edge perturbation |
| `spectral_perturbation(mode_idx)` | Eigenvector-based perturbation |
| `analyze_perturbation(...)` | Full perturbation trajectory analysis |
| `analyze_perturbation_field(...)` | Field-based perturbation analysis |

### Dynamics

| Method | Description |
|---|---|
| `evolve_markov(g, dim, t)` | Continuous-time Markov diffusion |
| `evolve_schrodinger(psi, dim, t)` | Schrodinger evolution |
| `evolve_coupled(state, dt, steps)` | Coupled 0-1-2 RK4 evolution |
| `evolve_field_wave(psi, dim, t, ...)` | Field-based wave evolution |
| `evolve_field_trajectory(psi, dim, times)` | Field wave trajectory |

### Quantum / Wave Mechanics

| Method | Description |
|---|---|
| `wave_state(dim, amplitudes)` | Complex-amplitude state |
| `measure(psi, dim)` | Projective measurement |
| `born_probabilities(psi)` | Born rule |
| `entanglement_entropy(psi, dim_A, dim_B)` | Bipartite entanglement |
| `measure_in_eigenbasis(psi, dim)` | Eigenbasis measurement |
| `graded_state(t, ...)` | Cross-dimensional graded state |
| `graded_trajectory(times, ...)` | Graded evolution trajectory |

### Field Theory

| Method/Property | Description |
|---|---|
| `field_operator` | (M, coupling, hermitian) field operator |
| `field_eigen` | (eigenvalues, eigenvectors, gaps) |
| `field_diffuse(F0, times)` | Diffusion through field operator |
| `field_wave_evolve(psi, t, ...)` | Wave evolution on field |
| `classify_modes()` | Mode classification dict |

### Dirac and Graded Operators

| Method/Property | Description |
|---|---|
| `dirac_operator()` | Full Dirac operator matrix |
| `dirac_eigenvalues` | Dirac spectrum |
| `canonical_collapse(vertex_idx)` | Graded-to-vertex projection |
| `born_graded(psi_re, psi_im)` | Per-dimension Born probabilities |
| `energy_partition(psi_re, psi_im)` | Graded energy breakdown |

### Topological Constructions

| Method | Description |
|---|---|
| `fill_cycle(cycle_edges)` | Add a face filling a cycle |
| `promote()` | Detect and fill all minimal 3-cycles |
| `face_data(...)` | Detected face data with metrics |
| `hypermanifold()` | Fiber bundle structure dict |
| `harmonic_shadow()` | Harmonic content across dimensions |
| `dimensional_subsumption()` | Tests if lower dims are captured by higher |

### Subcomplex and Quotient

| Method | Description |
|---|---|
| `subcomplex(v_mask, e_mask, f_mask)` | Extract subcomplex |
| `quotient(v_mask, e_mask, f_mask)` | Quotient complex |
| `subcomplex_by_energy(f_E, ...)` | Energy-based subcomplex |
| `star_of_vertex(v)` / `star_of_edge(e)` | Star neighborhoods |
| `validate_subcomplex(...)` | Check closure properties |
| `hyperslice(dim, idx)` | Cross-section at a cell |
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

### Graph Operations

| Method | Description |
|---|---|
| `subgraph(vertex_mask)` | Induced subgraph |
| `partition_communities()` | Louvain community partition -> list of (sub_rex, v_map, e_map) |
| `insert_edges(...)` | Add edges (rewrite) |
| `delete_edges(mask)` | Remove edges (rewrite) |
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
| `enrich_persistence(result)` | Add edge-type and Hodge enrichment |

### Query Engine

| Method | Description |
|---|---|
| `impute(signal, mask)` | Harmonic interpolation of missing values |
| `explain(dim, idx)` | Full diagnostic for a single cell |
| `propagate(source, target)` | Spectral propagation score |

### Demand-driven (seed-based) reads

Localized quantities computed by diffusion from a seed set, without building
the global character or eigenbasis: the cost scales with the seed, not nE.

| Method | Description |
|---|---|
| `character_response(seed, ...)` | Per-vertex character phi(v,k) at just the seed vertices |
| `coherence_response(seed, ...)` | Coherence kappa at a seed set, by diffusion |
| `local_context(seed, ...)` | Bounded local context around seed vertices, by heat diffusion |
| `propagate_signal(f0, ...)` | Dynamic vertex propagator S0 = B1 RL+ B1^T applied to a signal |
| `effective_resistance(edge)` | R_eff(e) = b_e^T L0+ b_e for one edge |

### Agentic reading

| Method | Description |
|---|---|
| `agentic_reading(...)` | Decision-ready reading over a query's entities and relations |
| `explain_context(...)` | Contextual picture around query entities (vertices) and relations (edges) |

### Similarity and overlap

| Property | Returns |
|---|---|
| `overlap_similarity`, `overlap_pairs` | Normalized overlap similarity S in [0,1], and top-k edge pairs |
| `phi_similarity` | Vertex character similarity S_phi[i,j] = 1 - 0.5\|phi_i - phi_j\|_1 |
| `fiber_similarity` | Fiber bundle similarity S_fb[i,j] between vertices |

### Serialization

| Method | Returns |
|---|---|
| `to_json()` | JSON-safe dict |
| `to_dict()` | Reconstruction dict |
| `from_dict(d)` | Class method: reconstruct from dict |

### Dashboard Data

| Method | Description |
|---|---|
| `signal_dashboard_data(signal, ...)` | Full dashboard payload for signal visualization |
| `quotient_dashboard_data(...)` | Dashboard payload for quotient visualization |

---

### Standalone Functions

| Function | Description |
|---|---|
| `cross_complex_bridge(rex_A, rex_B, labels_A, labels_B, ...)` | Cross-complex bridge analysis between two relational complexes. Aligns by shared vertex labels, compares kappa correlation, void fractions, and optionally spectral channel scores. |

---

## TemporalRex

Wraps a sequence of (sources, targets) or (boundary_ptr, boundary_idx)
snapshots with continuous edge identity. Delta-encoded storage.

### Properties and Methods

| Method/Property | Description |
|---|---|
| `T` | Number of timesteps |
| `at(t)` | RexGraph snapshot at timestep t |
| `temporal_index()` | Build temporal index structure |
| `edge_lifecycle()` | (edge_ids, birth, death) arrays |
| `edge_metrics()` | (counts, born, died) per timestep |
| `face_lifecycle_data()` | Face lifecycle if face snapshots exist |
| `bioes(...)` | BIOES phase tagging (Begin/Inside/Outside/End/Single) |
| `bioes_energy(...)` | Energy-ratio based BIOES |
| `bioes_joint(...)` | Joint structural + energy BIOES |
| `temporal_persistence(...)` | Persistence on final snapshot |
| `cascade_activation(...)` | Cascade activation from source edges |
| `cascade_wavefront(...)` | Wavefront propagation |

---




## `rextypes.py`: Typed Containers and Enumerations

Enumerations mirror integer codes from the Cython layer. NamedTuples wrap
the dicts and tuples returned by the Cython layer for attribute access and
readability. Used by `rexgraph.io._serialization` for generic storage (auto-discovered
via `inspect.getmembers`).

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

**Hodge:** HodgeDecomposition, HodgeAnalysis

**Spectral:** SpectralBundle: RL (relational Laplacian from build_RL), hats,
nhats, hat_names, trace_values, chi (structural character), K1 (overlap matrix),
L_C (copath complex Laplacian), and RL_1 = L1 + alpha_G * L_O

**Relational:** RCFBundle (RL, hats, nhats, chi, trace_values, hat_names),
VertexBundle (phi, chi_star, kappa), VoidComplex (tri_edges, void_indices),
RCFEResult

**Energy:** EnergyDecomposition, EnergyKinPot, WaveEnergy

**Field:** FieldOperatorResult, FieldEigenResult, ModeClassification

**Signal:** CascadeResult, FaceEmergenceResult, PerturbationResult, FieldPerturbationResult

**Quotient:** SubComplex, QuotientMaps, QuotientResult

**Hyperslice:** HypersliceVertex, HypersliceEdge, HypersliceFace

**Persistence:** PersistenceDiagram, Filtration, PersistenceEnrichment

**Temporal:** BioesEnergyResult, BioesResult

**State:** WaveState, SchrodingerState

**Interfacing:** InterfacingResult (rho, psi, scores, schrodinger, iv, sphere_pos, signal_magnitude, coverage, efficiency, confidence)

**Channels:** ChannelProfile (iv_T/G/F/Sch, pc_T/G/F, coverage, kappa_mean, efficiency)

**Filtration:** FiltrationResult (thresholds, beta0/1/2, n_edges_remaining, edges_removed_order, transition_index, transition_threshold)

**Cross-Complex:** CrossComplexBridge (kappa, void, n_shared, channel=None)

**Other:** MeasurementResult, FaceData, CoupledEvolution, StandardMetrics, JoinResult, ImputeResult, CellExplanation, PropagationResult, RemovalStep, StructuralSummary

---




## `analysis.py`: Dashboard Analysis Pipeline

Connects RexGraph to the visualization dashboard JSON contract. Accepts a
rex with optional vertex labels and edge attributes, returns a dict whose
keys match what the dashboard templates read. All computation is delegated
to the Cython modules through RexGraph's cached bundles.

---

### `analyze(rex, *, vertex_labels=None, edge_attrs=None, negative_types=None, ...)`

Full structural analysis for the graph dashboard. Returns a dict with:

- **meta**: nV, nE, nF, canvas size, attribute metadata, type/weight columns
- **vertices**: per-vertex: position, degree, role, partitions, Fiedler, divergence, PageRank, betweenness, clustering, community, face metrics
- **edges**: per-edge: source/target, type, weight, Hodge components (raw + normalized), rho, L1_down/L1_up diagonal, face metrics, betweenness, E_kin/E_pot
- **faces**: per-face: id, boundary, vertices, size, curl, concentration
- **topology**: chain validity, ranks, Betti, Euler, nF_hodge, self-loop faces
- **coupling**: alpha_G, alpha_T, Fiedler values for L1, L_O, RL_1
- **spectra**: eigenvalue arrays for L0, L1_down, L1_full, L2, L_O, RL_1, RL, field M
- **hodge**: gradient/curl/harmonic norms and percentages
- **energy**: E_kin, E_pot, ratio, regime, top kinetic/potential edges
- **analysis**: topology summary, Hodge summary, coupling interpretation, energy regime, structural roles, partition splits, standard metrics (PageRank, betweenness, clustering, Louvain), face structure
- **overlap**: top overlap pairs with Jaccard similarity
- **structural_character**: per-edge chi, per-vertex phi + kappa, aggregate summary, per-channel mixing times, mixing time anisotropy, inverse centrality ratio (when RCF modules available)
- **rcfe**: curvature per edge, strain, Bianchi identity check, attributed curvature, strain equilibrium
- **void_complex**: n_voids, n_potential, void strain, mean eta, fills_beta count, realization rate
- **fiber_bundle**: mean phi/fiber similarity, top vertex pairs by fiber similarity
- **channels**: primal signal character of the flow signal, face-void dipole (when `_channels` module available)
- **dirac**: Dirac spectrum, zero-mode count, Born probabilities per dimension, energy partition
- **hypermanifold**: manifold sequence with Betti/DOF per level, harmonic shadow, dimensional subsumption check
- **perturbation**: probe edge, cascade depth, activation count, energy/Hodge trajectories (when run_perturbation=True)
- **diffusion**: field diffusion norm decay, equilibrium index, vertex observables (when faces present)

---

### `analyze_signal(rex, *, probe_edges=None, n_steps=50, t_max=10.0, ...)`

Calls `analyze()` for base data, then appends signal-specific data from
`rex.signal_dashboard_data()`: perturbation trajectories, field diffusion,
mode classification, BIOES tags, cascade activation.

---

### `analyze_quotient(rex, *, max_vertex_presets=8, ...)`

Calls `analyze()` for base data, then appends quotient presets from
`rex.quotient_dashboard_data()`: vertex star quotients, edge type
subcomplexes, energy regime subcomplexes with Hodge decomposition,
congruence classes, relative Betti numbers.

---

### `analyze_all(rex, *, probe_edges=None, signal_steps=50, max_vertex_presets=8, ...)`

Runs `analyze()` once, appends both signal and quotient data. More
efficient than calling `analyze_signal` + `analyze_quotient` separately
because the base analysis is shared.

---

### Helpers

- `_build_flow(rex, edge_attrs, negative_types, nE)`: signed flow signal from edge attributes. Magnitude from w_E or weight column, sign from polarity matching against negative stem patterns.
- `_classify_attributes(edge_attrs, nE)`: attribute metadata in dashboard format (numeric with min/max/mean, categorical with values/counts).
- `_partition_from_fiedler(vec)`: Fiedler bipartition labels ("A"/"B").
- `_vertex_roles(in_deg, out_deg, v_face_count, nV)`: structural role assignment (source/sink/hub/mediator/peripheral).
