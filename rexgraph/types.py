"""
rexgraph.types - Typed containers for rex chain complex data structures.

Enumerations mirror integer codes from the Cython layer so that
Python-level code (graph.py, analysis.py) can refer to them without
importing .pyx modules.

NamedTuples wrap the dicts and tuples returned by the Cython layer.
The core code returns raw structures for speed; callers can wrap
them in these types for attribute access and readability.

Enumerations:
    EdgeType - edge classification codes (0-3)
    HodgeComponent - Hodge decomposition component indices (0-2)
    BioesTag - BIOES sequence labels (0-4)
    FaceEvent - face tracking event codes (0-4)
    TransitionKind - transition operator types (0-3)
    EnergyRegime - E_kin/E_pot ratio classification (0-2)

NamedTuples:
    HodgeDecomposition, HodgeAnalysis - Hodge decomposition results
    SpectralBundle - eigendecomposition across all dimensions
    RexLaplacianResult - Relational Laplacian RL_k = L_k + alpha_G * L_O
    FieldOperatorResult - field operator M on (E, F)
    FieldEigenResult - eigendecomposition of M
    EnergyDecomposition - E_kin, E_pot, E_RL from transition
    EnergyKinPot - E_kin, E_pot, ratio from state
    WaveEnergy - kinetic, potential, total wave energy
    ModeClassification - edge/face/resonant mode labels
    CascadeResult - edge activation order from signal propagation
    FaceEmergenceResult - face activation times
    BioesEnergyResult - BIOES tags from energy ratio timeseries
    QuotientMaps - reindexing arrays for R/I
    QuotientResult - full quotient pipeline output
    SubComplex - boolean masks defining a subcomplex I
    HypersliceVertex, HypersliceEdge, HypersliceFace - hyperslice results
    PerturbationResult - full analyze_perturbation output
    FieldPerturbationResult - analyze_perturbation_field output
    WaveState - position + velocity for second-order ODE
    SchrodingerState - real + imaginary parts for unitary evolution
    PersistenceDiagram, Filtration, PersistenceEnrichment
    BioesResult - unified BIOES pipeline output
    MeasurementResult, FaceData, CoupledEvolution, StandardMetrics
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# Enumerations


class EdgeType(IntEnum):
    """Edge classification codes from _rex.classify_edges_general()."""

    STANDARD = 0
    SELF_LOOP = 1
    BRANCHING = 2
    WITNESS = 3


class HodgeComponent(IntEnum):
    """Hodge decomposition component indices."""

    GRADIENT = 0
    CURL = 1
    HARMONIC = 2


class BioesTag(IntEnum):
    """BIOES sequence labels for temporal phase detection.

    Mirrors DEF TAG_B/I/O/E/S in _temporal.pyx.
    """

    BEGIN = 0
    INSIDE = 1
    OUTSIDE = 2
    END = 3
    SINGLE = 4


class FaceEvent(IntEnum):
    """Face tracking event types across temporal snapshots.

    Mirrors DEF FACE_PERSIST/BORN/DIED/SPLIT/MERGE in _temporal.pyx.
    """

    PERSIST = 0
    BORN = 1
    DIED = 2
    SPLIT = 3
    MERGE = 4


class TransitionKind(IntEnum):
    """Transition operator types.

    Mirrors TRANS_* constants in _transition.pyx.
    """

    MARKOV = 0
    SCHRODINGER = 1
    DIFFERENTIAL = 2
    REWRITE = 3


class EnergyRegime(IntEnum):
    """E_kin/E_pot ratio classification for per-edge or per-timestep regimes.

    Mirrors DEF ENERGY_KINETIC/CROSSOVER/POTENTIAL in _temporal.pyx.
    """

    KINETIC = 0
    CROSSOVER = 1
    POTENTIAL = 2


# Hodge decomposition


class HodgeDecomposition(NamedTuple):
    """Hodge decomposition of an edge signal into three orthogonal components."""

    gradient: NDArray
    """Gradient component B_1^T phi in im(B_1^T), shape (nE,)."""

    curl: NDArray
    """Curl component B_2 psi in im(B_2), shape (nE,)."""

    harmonic: NDArray
    """Harmonic component eta in ker(L_1), shape (nE,)."""


class HodgeAnalysis(NamedTuple):
    """Full Hodge analysis with normalized components and energy fractions."""

    grad: NDArray
    """Raw gradient component, shape (nE,)."""
    curl: NDArray
    """Raw curl component, shape (nE,)."""
    harm: NDArray
    """Raw harmonic component, shape (nE,)."""
    grad_norm: NDArray
    """Self-normalized gradient, shape (nE,)."""
    curl_norm: NDArray
    """Self-normalized curl, shape (nE,)."""
    harm_norm: NDArray
    """Self-normalized harmonic, shape (nE,)."""
    flow_norm: NDArray
    """Self-normalized input flow, shape (nE,)."""
    rho: NDArray
    """Per-edge resistance ratio |eta_e| / |g_e|, shape (nE,)."""
    pct_grad: float
    """Gradient energy fraction."""
    pct_curl: float
    """Curl energy fraction."""
    pct_harm: float
    """Harmonic energy fraction."""
    divergence: NDArray
    """Vertex divergence B_1 g, shape (nV,)."""
    div_norm: NDArray
    """Normalized divergence, shape (nV,)."""
    face_curl: NDArray
    """Face curl B_2^T g, shape (nF,)."""
    face_curl_norm: NDArray
    """Normalized face curl, shape (nF,)."""


# Spectral bundle


class SpectralBundle(NamedTuple):
    """Eigendecomposition results from _laplacians.build_all_laplacians()."""

    # L0
    L0: object
    """Vertex Laplacian (dense or sparse)."""
    evals_L0: NDArray
    """Eigenvalues of L0, shape (k,)."""
    evecs_L0: NDArray
    """Eigenvectors of L0, shape (nV, k)."""
    fiedler_L0: float
    """Fiedler value (algebraic connectivity)."""
    fiedler_vec_L0: NDArray
    """Fiedler vector, shape (nV,)."""

    # L1
    L1_down: object
    L1_up: object
    L1_full: object
    evals_L1: Optional[NDArray]
    evecs_L1: Optional[NDArray]
    diag_L1_down: NDArray
    """Per-edge L1_down diagonal, shape (nE,)."""
    diag_L1_up: NDArray
    """Per-edge L1_up diagonal, shape (nE,)."""

    # L2
    L2: object
    evals_L2: Optional[NDArray]
    evecs_L2: Optional[NDArray]

    # Betti and topology
    beta0: int
    beta1: int
    beta2: int
    rank_B1: int
    rank_B2: int
    euler_char: int

    # Overlap block (present when L_O is provided)
    evals_L_O: Optional[NDArray]
    evecs_L_O: Optional[NDArray]
    fiedler_L_O: Optional[float]
    fiedler_vec_L_O: Optional[NDArray]
    alpha_G: Optional[float]
    alpha_T: float
    alpha_used: Optional[float]

    # Relational Laplacian (present when L_O is provided)
    RL1: object
    """RL_1 = L_1 + alpha_G * L_O."""
    evals_RL1: Optional[NDArray]
    evecs_RL1: Optional[NDArray]
    Lambda: object
    evals_Lambda: Optional[NDArray]
    evecs_Lambda: Optional[NDArray]


# Relational Laplacian


class RexLaplacianResult(NamedTuple):
    """Relational Laplacian RL_k = L_k + alpha_G * L_O at a single dimension."""

    RL: NDArray
    """The composite operator, shape (n, n)."""
    alpha_G: float
    """Coupling constant Fiedler(L_k) / Fiedler(L_O)."""
    spectral_gap: float
    """Fiedler(RL_k) - Fiedler(L_k)."""
    evals: NDArray
    """Eigenvalues of RL_k, shape (n,)."""
    evecs: NDArray
    """Eigenvectors of RL_k, shape (n, n)."""


# Field operator (from _field.pyx)


class FieldOperatorResult(NamedTuple):
    """Field operator M on (E, F) from build_field_operator()."""

    M: NDArray
    """Block matrix [[RL_1, -g*B_2], [-g*B_2^T, L_2]], shape (nE+nF, nE+nF)."""
    g: float
    """Coupling strength used."""
    is_psd: bool
    """Whether all eigenvalues are non-negative."""


class FieldEigenResult(NamedTuple):
    """Eigendecomposition of the field operator M."""

    evals: NDArray
    """Eigenvalues of M, shape (nE+nF,)."""
    evecs: NDArray
    """Eigenvectors of M, shape (nE+nF, nE+nF)."""
    freqs: NDArray
    """Mode frequencies sqrt(max(0, evals)), shape (nE+nF,)."""


# Energy types


class EnergyDecomposition(NamedTuple):
    """Energy decomposition under the Relational Laplacian.

    From _transition.energy_decomposition(), which takes (f_re, f_im)
    for complex signals (Schrodinger evolution).
    """

    E_kin: float
    """Topological energy <f|L_1|f>."""
    E_pot: float
    """Geometric energy <f|L_O|f>."""
    E_RL: float
    """Total Rex energy E_kin + alpha_G * E_pot."""


class EnergyKinPot(NamedTuple):
    """Kinetic/potential energy of a real edge signal.

    From _state.energy_kin_pot().
    """

    E_kin: float
    """Topological energy <f_E|L_1|f_E>."""
    E_pot: float
    """Geometric energy <f_E|L_O|f_E>."""
    ratio: float
    """E_kin / E_pot (inf-capped when E_pot is near zero)."""


class WaveEnergy(NamedTuple):
    """Wave energy on the (E, F) field.

    From _field.wave_energy(). KE + PE is conserved under the wave equation.
    """

    KE: float
    """Kinetic energy 0.5 * ||dF/dt||^2."""
    PE: float
    """Potential energy 0.5 * F^T M F."""
    total: float
    """KE + PE (conserved quantity)."""


# Mode classification (from _field.pyx)


class ModeClassification(NamedTuple):
    """Mode classification from field_eigendecomposition.

    From _field.classify_modes().
    """

    labels: NDArray
    """Per-mode label string: 'edge', 'face', or 'resonant', shape (n,)."""
    weights_E: NDArray
    """Edge-block weight per mode, shape (n,)."""
    weights_F: NDArray
    """Face-block weight per mode, shape (n,)."""
    n_resonant: int
    """Number of modes with weight in both E and F blocks."""


# Cascade and temporal (from _signal.pyx and _temporal.pyx)


class CascadeResult(NamedTuple):
    """Edge activation order from signal propagation.

    From _signal.cascade_from_edge().
    """

    activation_time: NDArray
    """First timestep each edge exceeds threshold, i32[nE]. -1 if never."""
    activation_order: NDArray
    """Edges sorted by activation time, i32[n_activated]."""
    activation_rank: NDArray
    """Rank of each edge in activation order, i32[nE]. -1 if never."""
    threshold: float
    """Activation threshold used."""


class FaceEmergenceResult(NamedTuple):
    """Face activation times from signal propagation.

    From _signal.face_emergence().
    """

    face_activation_time: NDArray
    """First timestep each face is fully activated, i32[nF]. -1 if never."""
    face_order: NDArray
    """Faces sorted by activation time, i32[n_activated]."""


class BioesEnergyResult(NamedTuple):
    """BIOES phase tags from energy ratio timeseries.

    From _temporal.compute_bioes_energy().
    """

    tags: NDArray
    """BIOES tags per timestep, i32[T]."""
    phase_start: NDArray
    """Phase start indices, i32[n_phases]."""
    phase_end: NDArray
    """Phase end indices, i32[n_phases]."""
    phase_regime: NDArray
    """EnergyRegime code per phase, i32[n_phases]."""
    log_ratios: NDArray
    """Per-timestep log(E_kin/E_pot), f64[T]."""
    crossover_times: NDArray
    """Timestep indices where ratio crosses 1, i32[n_crossings]."""


# Quotient complex (from _quotient.pyx)


class SubComplex(NamedTuple):
    """A subcomplex I of R defined by boolean masks."""

    v_mask: NDArray
    """Boolean mask over vertices (1 = in subcomplex), u8[nV]."""
    e_mask: NDArray
    """Boolean mask over edges, u8[nE]."""
    f_mask: NDArray
    """Boolean mask over faces, u8[nF]."""


class QuotientMaps(NamedTuple):
    """Reindexing arrays for the quotient complex R/I.

    From _quotient.quotient_maps().
    """

    v_reindex: NDArray
    """Old vertex -> new index, or -1 if collapsed, i32[nV]."""
    v_star: int
    """Basepoint index in the quotient, or -1 if v_mask is all zero."""
    e_reindex: NDArray
    """Old edge -> new index, or -1 if removed, i32[nE]."""
    f_reindex: NDArray
    """Old face -> new index, or -1 if removed, i32[nF]."""
    nV_quot: int
    nE_quot: int
    nF_quot: int


class QuotientResult(NamedTuple):
    """Full quotient pipeline output from _quotient.build_quotient()."""

    B1_quot: NDArray
    """Quotient boundary B_1, f64[nV', nE']."""
    B2_quot: NDArray
    """Quotient boundary B_2, f64[nE', nF']."""
    L1_quot: NDArray
    """Hodge Laplacian on quotient edges, f64[nE', nE']."""
    betti_rel: Tuple[int, int, int]
    """Relative Betti numbers (beta_0, beta_1, beta_2)."""
    chain_valid: bool
    """Whether B_1 B_2 = 0 holds on the quotient."""
    chain_error: float
    """Frobenius norm of B_1 B_2 on the quotient."""
    v_reindex: NDArray
    """Vertex reindexing, i32[nV]."""
    e_reindex: NDArray
    """Edge reindexing, i32[nE]."""
    f_reindex: NDArray
    """Face reindexing, i32[nF]."""
    v_star: int
    """Basepoint index in the quotient."""
    dims: Tuple[int, int, int]
    """Quotient dimensions (nV', nE', nF')."""
    LO_quot: Optional[NDArray]
    """Overlap Laplacian on quotient edges (present when LO is provided)."""
    RL1_quot: Optional[NDArray]
    """Relational Laplacian on quotient edges (present when LO is provided)."""


# Hyperslices


class HypersliceVertex(NamedTuple):
    """Hyperslice through a vertex (dim=0)."""

    above: NDArray
    """Incident edges, shape (deg,)."""
    lateral: NDArray
    """Neighbor vertices, shape (n_nbrs,)."""


class HypersliceEdge(NamedTuple):
    """Hyperslice through an edge (dim=1)."""

    below: NDArray
    """Boundary vertices [src, tgt], shape (2,)."""
    above: NDArray
    """Incident faces, shape (n_faces,)."""
    lateral: NDArray
    """Overlap-adjacent edges, shape (n_overlap,)."""


class HypersliceFace(NamedTuple):
    """Hyperslice through a face (dim=2)."""

    below: NDArray
    """Boundary edges, shape (n_bnd,)."""
    lateral: NDArray
    """Adjacent faces (sharing a boundary edge), shape (n_adj,)."""


# Signal pipeline results (from _signal.pyx)


class PerturbationResult(NamedTuple):
    """Full perturbation analysis from _signal.analyze_perturbation().

    Covers diffusion propagation, energy decomposition, cascade analysis,
    BIOES tagging, and Hodge decomposition at endpoints.
    """

    trajectory: NDArray
    """Edge signal over time, f64[T, nE]."""
    E_kin: NDArray
    """Topological energy per timestep, f64[T]."""
    E_pot: NDArray
    """Geometric energy per timestep, f64[T]."""
    ratio: NDArray
    """E_kin/E_pot per timestep, f64[T]."""
    norms: NDArray
    """Signal L2 norm per timestep, f64[T]."""
    Ekin_per_edge: NDArray
    """Per-edge kinetic energy at final timestep, f64[nE]."""
    Epot_per_edge: NDArray
    """Per-edge potential energy at final timestep, f64[nE]."""
    activation_time: NDArray
    """Per-edge first activation timestep, i32[nE]."""
    activation_order: NDArray
    """Edges sorted by activation time, i32[n_activated]."""
    activation_rank: NDArray
    """Rank of each edge, i32[nE]."""
    activation_threshold: float
    """Threshold used for activation detection."""
    face_activation_time: NDArray
    """Per-face activation timestep, i32[nF]."""
    face_order: NDArray
    """Faces sorted by activation time, i32[n_activated]."""
    bioes_tags: NDArray
    """BIOES phase tags per timestep, i32[T]."""
    phase_start: NDArray
    """Phase boundary start indices, i32[n_phases]."""
    phase_end: NDArray
    """Phase boundary end indices, i32[n_phases]."""
    phase_regime: NDArray
    """EnergyRegime code per phase, i32[n_phases]."""
    log_ratios: NDArray
    """Per-timestep log(E_kin/E_pot), f64[T]."""
    crossover_times: NDArray
    """Timestep indices where ratio crosses 1, i32[n_crossings]."""
    hodge_initial: dict
    """Hodge decomposition of initial edge signal."""
    hodge_final: dict
    """Hodge decomposition of final edge signal."""
    cascade_depth: NDArray
    """BFS topological distance from source, i32[nE]."""
    f_V_initial: NDArray
    """Derived vertex signal at t=0, f64[nV]."""
    f_V_final: NDArray
    """Derived vertex signal at t=T, f64[nV]."""
    alpha_G: float
    """Coupling constant used."""
    T: int
    """Number of timesteps."""
    nE: int
    """Number of edges."""


class FieldPerturbationResult(NamedTuple):
    """Field perturbation analysis from _signal.analyze_perturbation_field().

    Uses the full (E, F) field operator with diffusion or wave evolution.
    """

    field_trajectory: NDArray
    """Full field state over time, f64[T, nE+nF]."""
    edge_trajectory: NDArray
    """Edge block extracted, f64[T, nE]."""
    face_trajectory: NDArray
    """Face block extracted, f64[T, nF]."""
    vertex_trajectory: NDArray
    """Derived vertex signals, f64[T, nV]."""
    E_kin: NDArray
    """Topological energy from edge block, f64[T]."""
    E_pot: NDArray
    """Geometric energy from edge block, f64[T]."""
    ratio: NDArray
    """E_kin/E_pot per timestep, f64[T]."""
    norm_E: NDArray
    """Edge signal L2 norm per timestep, f64[T]."""
    norm_F: NDArray
    """Face signal L2 norm per timestep, f64[T]."""
    mode: str
    """Evolution mode: 'diffusion' or 'wave'."""
    nE: int
    nF: int
    T: int
    wave_KE: Optional[NDArray]
    """Wave kinetic energy (wave mode only), f64[T]."""
    wave_PE: Optional[NDArray]
    """Wave potential energy (wave mode only), f64[T]."""
    wave_total: Optional[NDArray]
    """Wave total energy (wave mode only, conserved), f64[T]."""
    velocity_trajectory: Optional[NDArray]
    """dF/dt trajectory (wave mode only), f64[T, nE+nF]."""


# Evolution state containers


class WaveState(NamedTuple):
    """Position + velocity state for the second-order wave equation.

    Used by _field.wave_evolve() and _field.field_rk4_step().
    """

    F: NDArray
    """Field position, f64[nE+nF]."""
    dFdt: NDArray
    """Field velocity, f64[nE+nF]."""
    t: float
    """Current time."""


class SchrodingerState(NamedTuple):
    """Real and imaginary parts of a unitary-evolved signal.

    From _transition.schrodinger_evolve_spectral().
    """

    f_re: NDArray
    """Real part, f64[n]."""
    f_im: NDArray
    """Imaginary part, f64[n]."""
    t: float
    """Current time."""


# Persistence


class PersistenceDiagram(NamedTuple):
    """Persistence diagram from a filtration."""

    pairs: NDArray
    """Finite pairs, shape (k, 5): [birth, death, dim, birth_cell, death_cell]."""
    essential: NDArray
    """Essential classes, shape (k, 3): [birth, inf, dim]."""
    betti: Tuple[int, int, int]
    """Betti numbers at the final filtration step."""
    order: NDArray
    """Filtration ordering of all cells, i64[N]."""


class Filtration(NamedTuple):
    """Filtration values on the chain complex."""

    filt_v: NDArray
    """Vertex filtration values, f64[nV]."""
    filt_e: NDArray
    """Edge filtration values, f64[nE]."""
    filt_f: NDArray
    """Face filtration values, f64[nF]."""


class PersistenceEnrichment(NamedTuple):
    """Rex-specific metadata for persistence pairs."""

    edge_type_annotations: NDArray
    """Edge type code (0-3) for each dim-1 pair's birth edge."""
    hodge_dominant: NDArray
    """Dominant Hodge component index (0/1/2) at birth."""
    hodge_fractions: NDArray
    """Energy fractions [grad^2, curl^2, harm^2] at birth edge, shape (k, 3)."""


# Temporal (unified BIOES)


class BioesResult(NamedTuple):
    """Unified BIOES pipeline from _temporal.compute_bioes_unified()."""

    unified_tags: NDArray
    """BIOES tags from joint phase detection, i32[T]."""
    per_dim_tags: NDArray
    """Independent BIOES tags per Betti dimension, i32[T, K]."""
    edge_counts: NDArray
    """Total edge count per timestep, i64[T]."""
    edge_born: NDArray
    """Edges born per timestep, i64[T]."""
    edge_died: NDArray
    """Edges died per timestep, i64[T]."""
    face_counts: NDArray
    """Total face count per timestep, i64[T]."""
    face_born: NDArray
    """Faces born per timestep, i64[T]."""
    face_died: NDArray
    """Faces died per timestep, i64[T]."""
    face_split: NDArray
    """Face split events per timestep, i64[T]."""
    face_merge: NDArray
    """Face merge events per timestep, i64[T]."""
    phase_start: NDArray
    """Phase start indices, i64[n_phases]."""
    phase_end: NDArray
    """Phase end indices, i64[n_phases]."""
    phase_betti: NDArray
    """Representative Betti numbers per phase, i64[n_phases, K]."""
    break_reasons: NDArray
    """Dimension index that triggered each phase break, i64[n_phases]."""


# Wave mechanics


class MeasurementResult(NamedTuple):
    """Result of a projective measurement on a quantum state."""

    outcome: int
    """Measured cell index."""
    collapsed: NDArray
    """Post-measurement state vector, shape (n,)."""


# Face data


class FaceData(NamedTuple):
    """Face analysis from _faces.build_face_data()."""

    faces: list
    """Per-face descriptors: [{id, boundary, vertices, size}, ...]."""
    vertex_face_count: NDArray
    """Number of faces incident to each vertex, i32[nV]."""
    metrics: dict
    """Structural metric arrays from compute_face_metrics()."""


# Coupled evolution


class CoupledEvolution(NamedTuple):
    """Result of coupled cross-dimensional diffusion via RK4."""

    y_final: NDArray
    """Final state vector (f_0, f_1, f_2), shape (nV+nE+nF,)."""
    trajectory: NDArray
    """State at each timestep, shape (n_steps+1, nV+nE+nF)."""
    times: NDArray
    """Time values, shape (n_steps+1,)."""


# Standard graph metrics


class StandardMetrics(NamedTuple):
    """Standard graph metrics from _standard.build_standard_metrics()."""

    pagerank: NDArray
    """PageRank scores, f64[nV]."""
    betweenness_v: NDArray
    """Vertex betweenness centrality, f64[nV]."""
    betweenness_e: NDArray
    """Edge betweenness centrality, f64[nE]."""
    btw_norm_v: NDArray
    """Normalized vertex betweenness, f64[nV]."""
    btw_norm_e: NDArray
    """Normalized edge betweenness, f64[nE]."""
    clustering: NDArray
    """Local clustering coefficient, f64[nV]."""
    community_labels: NDArray
    """Louvain community labels, i32[nV]."""
    n_communities: int
    """Number of communities detected."""
    modularity: float
    """Louvain modularity score."""


# RCF types (new in v2)


class OperatorChannel(IntEnum):
    """Channel indices for the relational Laplacian decomposition."""
    TOPOLOGICAL = 0
    GEOMETRIC = 1
    FRUSTRATION = 2
    COPC = 3


class PredicateOp(IntEnum):
    """Predicate operations for the query engine."""
    GT = 0
    GE = 1
    LT = 2
    LE = 3
    EQ = 4
    NE = 5
    BETWEEN = 6


class JoinType(IntEnum):
    """Join type for chain complex join operations."""
    INNER = 0
    LEFT = 1
    OUTER = 2
    UNION = 3


class RCFBundle(NamedTuple):
    """Relational Laplacian bundle from _relational.build_RL_from_laplacians()."""
    RL: NDArray
    """Relational Laplacian, f64[nE, nE]. tr(RL) = nhats."""
    hats: list
    """Trace-normalized hat operators, list of f64[nE, nE]."""
    nhats: int
    """Number of active hat operators."""
    chi: NDArray
    """Structural character per edge, f64[nE, nhats]. On the simplex."""
    trace_values: NDArray
    """Trace of each raw Laplacian before normalization, f64[nhats]."""
    hat_names: list
    """Human-readable names of each hat operator."""


class VertexBundle(NamedTuple):
    """Vertex character bundle from _character.build_character_bundle()."""
    phi: NDArray
    """Vertex character, f64[nV, nhats]. On the simplex."""
    chi_star: NDArray
    """Star character (mean of chi over incident edges), f64[nV, nhats]."""
    kappa: NDArray
    """Cross-dimensional coherence, f64[nV] in [0, 1]."""


class VoidComplex(NamedTuple):
    """Void spectral theory from _void.build_void_complex()."""
    Bvoid: object
    """Void boundary operator, f64[nE, n_voids] or None."""
    Lvoid: object
    """Void Laplacian, f64[nE, nE] or None."""
    n_voids: int
    """Number of void triangles."""
    n_potential: int
    """Total potential triangles (realized + void)."""
    eta: NDArray
    """Harmonic content per void, f64[n_voids] in [0, 1]."""
    chi_void: NDArray
    """Void character per void, f64[n_voids, nhats]."""
    fills_beta: NDArray
    """Whether filling this void decreases beta_1, i32[n_voids]."""
    void_strain: float
    """Total void strain S^void = tr(Lvoid)."""


class RCFEResult(NamedTuple):
    """RCFE curvature and conservation from _rcfe."""
    curvature: NDArray
    """RCFE curvature per edge, f64[nE]."""
    strain: float
    """Total strain S = sum C(e) * RL[e,e]."""
    bianchi_ok: bool
    """Whether Bianchi identity B1 diag(C) B2 = 0 holds."""
    bianchi_residual: float
    """Max absolute entry of B1 diag(C) B2."""


class JoinResult(NamedTuple):
    """Result of a chain complex join operation."""
    B1j: NDArray
    """Joined boundary operator, f64[nVj, nEj]."""
    B2j: NDArray
    """Joined face boundary operator, f64[nEj, nFj]."""
    nVj: int
    """Number of vertices in joined complex."""
    nEj: int
    """Number of edges in joined complex."""
    nFj: int
    """Number of faces in joined complex."""
    beta: tuple
    """Betti numbers (beta_0, beta_1, beta_2)."""
    chain_residual: float
    """Max |B1j @ B2j| (should be ~0)."""


class ImputeResult(NamedTuple):
    """Result of harmonic signal imputation."""
    imputed: NDArray
    """Full signal with imputed values, f64[nE]."""
    confidence: NDArray
    """Confidence per edge (1.0 for observed, lower for imputed), f64[nE]."""
    residual: float
    """Residual energy at observed positions."""
    n_observed: int
    """Number of observed edges."""
    n_imputed: int
    """Number of imputed edges."""


class CellExplanation(NamedTuple):
    """Full diagnostic for a single cell from _query.explain_*."""
    below: NDArray
    """Boundary cells (lower dimension), i32[k]."""
    above: NDArray
    """Co-boundary cells (higher dimension), i32[k]."""
    lateral: NDArray
    """Cells sharing a common boundary point, i32[k]."""
    chi: NDArray
    """Structural character at this cell, f64[nhats]."""
    dominant_channel: int
    """Index of dominant character channel."""
    kappa: float
    """Coherence at this cell (for vertices)."""


class PropagationResult(NamedTuple):
    """Result of spectral propagation through RL."""
    score: float
    """Overall propagation score."""
    typed_scores: NDArray
    """Per-channel propagation scores, f64[nhats]."""
    energy: float
    """Source energy in RL."""
    coverage: float
    """Fraction of spectral modes covered by source."""


class RemovalStep(NamedTuple):
    """One step in a sequential edge removal."""
    cell_removed: int
    """Index of the cell removed at this step."""
    beta: tuple
    """Betti numbers after removal."""
    fiedler: float
    """Fiedler value after removal."""
    components: int
    """Number of connected components."""
    faces_lost: int
    """Number of faces lost at this step."""
    chain_residual: float
    """Max |B1 @ B2| for residual complex."""


class StructuralSummary(NamedTuple):
    """Aggregate structural statistics."""
    mean_chi: NDArray
    """Mean chi per channel, f64[nhats]."""
    std_chi: NDArray
    """Std chi per channel, f64[nhats]."""
    mean_kappa: float
    """Mean coherence across vertices."""
    std_kappa: float
    """Std coherence."""
    dominant_dist: NDArray
    """Distribution of dominant channel per edge, i32[nhats]."""
    low_kappa_count: int
    """Number of vertices with kappa < 0.5."""
    n_voids: int
    """Number of void triangles."""
    void_strain: float
    """Total void strain."""
    mean_eta: float
    """Mean harmonic content of voids."""
