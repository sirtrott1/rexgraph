# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._signal - Signal processing pipeline on the rex chain complex.

End-to-end pipeline: perturbation -> propagation -> decomposition -> tagging.

This module orchestrates calls to _state, _field, _hodge, _temporal, _wave,
and _laplacians to produce a complete perturbation analysis. It is the
top-level analytical interface for the rex framework.

Field states live on (E, F) only. Vertex observables are derived via
f_V = B_1 f_E. All functions accept and return edge-primary data.

Pipeline stages:
    1. Perturbation construction (Dirac, vertex-derived, multi-edge, spectral)
    2. Propagation (diffusion, wave, Schrodinger)
    3. Energy decomposition (E_kin/E_pot per timestep, Hodge components)
    4. Cascade analysis (activation order, face emergence, topological depth)
    5. Temporal tagging (BIOES from energy ratio)
    6. Full pipeline (one-call entry point)
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt, log, exp
from libc.string cimport memcpy

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,
    get_EPSILON_NORM,
)

np.import_array()


# Helpers


def _safe_dot(A, x):
    """Matrix-vector product, dense or sparse."""
    return np.asarray(A.dot(x), dtype=np.float64)


def _ensure_1d(x):
    """Ensure array is 1-D f64."""
    return np.asarray(x, dtype=np.float64).ravel()


# Section 1: Perturbation construction


def build_edge_perturbation(Py_ssize_t nE, Py_ssize_t nF,
                            Py_ssize_t edge_idx,
                            double amplitude=1.0):
    """Create Dirac delta perturbation on a single edge.

    Returns (f_E, f_F) field state suitable for field_state_pack.
    Vertex activation is derived via B_1.

    Parameters
    ----------
    nE, nF : dimensions
    edge_idx : edge to perturb
    amplitude : signal magnitude

    Returns
    -------
    f_E : f64[nE]
    f_F : f64[nF]
    """
    cdef np.ndarray[f64, ndim=1] f_E = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f_F = np.zeros(nF, dtype=np.float64)
    if 0 <= edge_idx < nE:
        f_E[edge_idx] = amplitude
    return f_E, f_F


def build_vertex_perturbation(Py_ssize_t vertex_idx,
                              object B1_T,
                              Py_ssize_t nE,
                              Py_ssize_t nF):
    """Convert a vertex perturbation to edge signal via B_1^T.

    In the rex framework, "perturb vertex v" means "activate all
    edges incident to v". The edge signal is B_1^T delta_v.

    Parameters
    ----------
    vertex_idx : vertex to perturb
    B1_T : (nE, nV) - transpose of boundary operator
    nE, nF : dimensions

    Returns
    -------
    f_E : f64[nE]
    f_F : f64[nF]
    """
    cdef Py_ssize_t nV = B1_T.shape[1]
    cdef np.ndarray[f64, ndim=1] delta_v = np.zeros(nV, dtype=np.float64)
    delta_v[vertex_idx] = 1.0
    cdef np.ndarray[f64, ndim=1] f_E = np.asarray(B1_T.dot(delta_v),
                                                    dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f_F = np.zeros(nF, dtype=np.float64)
    return f_E, f_F


def build_multi_edge_perturbation(Py_ssize_t nE, Py_ssize_t nF,
                                  np.ndarray[i32, ndim=1] edge_indices,
                                  np.ndarray[f64, ndim=1] amplitudes):
    """Superposition of Dirac deltas on multiple edges.

    For modeling multiple simultaneous perturbations (e.g. a drug
    cocktail hitting several pathway interactions at once).

    Parameters
    ----------
    nE, nF : dimensions
    edge_indices : i32[k] - edges to perturb
    amplitudes : f64[k] - signal magnitudes

    Returns
    -------
    f_E : f64[nE]
    f_F : f64[nF]
    """
    cdef np.ndarray[f64, ndim=1] f_E = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f_F = np.zeros(nF, dtype=np.float64)
    cdef i32[::1] ei = edge_indices
    cdef f64[::1] av = amplitudes
    cdef Py_ssize_t k = edge_indices.shape[0], j

    for j in range(k):
        if 0 <= ei[j] < nE:
            f_E[ei[j]] += av[j]

    return f_E, f_F


def build_spectral_perturbation(Py_ssize_t nE, Py_ssize_t nF,
                                np.ndarray[f64, ndim=2] evecs_RL1,
                                Py_ssize_t mode_idx,
                                double amplitude=1.0):
    """Perturbation along a specific eigenmode of RL_1.

    Excites a single normal mode of the Relational Laplacian.
    Useful for probing topological vs geometric response: low modes
    are smooth (topological), high modes are rough (geometric).

    Parameters
    ----------
    nE, nF : dimensions
    evecs_RL1 : f64[nE, nE] - eigenvectors of RL_1 (columns)
    mode_idx : which eigenmode to excite (0 = lowest)
    amplitude : scale factor

    Returns
    -------
    f_E : f64[nE]
    f_F : f64[nF]
    """
    cdef np.ndarray[f64, ndim=1] f_E = amplitude * evecs_RL1[:, mode_idx].copy()
    cdef np.ndarray[f64, ndim=1] f_F = np.zeros(nF, dtype=np.float64)
    return f_E, f_F


# Section 2: Propagation engines


def propagate_diffusion(np.ndarray[f64, ndim=1] f_E,
                        object L_operator,
                        np.ndarray[f64, ndim=1] evals,
                        np.ndarray[f64, ndim=2] evecs,
                        np.ndarray[f64, ndim=1] times):
    """Heat equation on edges: f(t) = exp(-L t) f(0).

    Uses spectral decomposition for exact evolution.
    L_operator can be L_1, L_O, or RL_1 (any edge-space Laplacian).

    Parameters
    ----------
    f_E : f64[nE] - initial edge signal
    L_operator : (nE, nE) - Laplacian (not used directly, included for API)
    evals : f64[nE] - eigenvalues of L_operator
    evecs : f64[nE, nE] - eigenvectors
    times : f64[T] - timepoints

    Returns
    -------
    trajectory : f64[T, nE]
    """
    cdef Py_ssize_t nE = f_E.shape[0], T = times.shape[0]
    cdef Py_ssize_t k, j, step
    cdef f64[::1] fv = f_E
    cdef f64[:, ::1] ev = evecs
    cdef f64[::1] lv = evals, tv = times

    # Compute spectral coefficients once
    cdef np.ndarray[f64, ndim=1] coeffs = np.empty(nE, dtype=np.float64)
    cdef f64[::1] cv = coeffs
    cdef f64 s
    for k in range(nE):
        s = 0.0
        for j in range(nE):
            s += ev[j, k] * fv[j]
        cv[k] = s

    cdef np.ndarray[f64, ndim=2] traj = np.zeros((T, nE), dtype=np.float64)
    cdef f64[:, ::1] trajv = traj
    cdef f64 decay

    for step in range(T):
        for k in range(nE):
            if fabs(cv[k]) < 1e-15:
                continue
            decay = cv[k] * exp(-lv[k] * tv[step])
            for j in range(nE):
                trajv[step, j] += decay * ev[j, k]

    return traj


def propagate_diffusion_comparative(np.ndarray[f64, ndim=1] f_E,
                                    object L1, object LO, object RL1,
                                    np.ndarray[f64, ndim=1] evals_L1,
                                    np.ndarray[f64, ndim=2] evecs_L1,
                                    np.ndarray[f64, ndim=1] evals_LO,
                                    np.ndarray[f64, ndim=2] evecs_LO,
                                    np.ndarray[f64, ndim=1] evals_RL1,
                                    np.ndarray[f64, ndim=2] evecs_RL1,
                                    np.ndarray[f64, ndim=1] times):
    """Run diffusion under L_1, L_O, and RL_1 for comparison.

    Returns three trajectory arrays so the user can see how the same
    perturbation behaves under topological-only, geometric-only, and
    combined dynamics.

    Returns
    -------
    traj_L1 : f64[T, nE]
    traj_LO : f64[T, nE]
    traj_RL1 : f64[T, nE]
    """
    traj_L1 = propagate_diffusion(f_E, L1, evals_L1, evecs_L1, times)
    traj_LO = propagate_diffusion(f_E, LO, evals_LO, evecs_LO, times)
    traj_RL1 = propagate_diffusion(f_E, RL1, evals_RL1, evecs_RL1, times)
    return traj_L1, traj_LO, traj_RL1


# Section 3: Energy decomposition


def energy_trajectory(np.ndarray[f64, ndim=2] trajectory,
                      object L1,
                      object LO):
    """Compute E_kin and E_pot at each timestep of an edge trajectory.

    E_kin(t) = <f(t) | L_1 | f(t)>   (topological energy)
    E_pot(t) = <f(t) | L_O | f(t)>   (geometric energy)

    Parameters
    ----------
    trajectory : f64[T, nE]
    L1 : (nE, nE) - Hodge Laplacian
    LO : (nE, nE) - overlap Laplacian

    Returns
    -------
    E_kin : f64[T]
    E_pot : f64[T]
    ratio : f64[T] - E_kin / E_pot (capped at 1e15)
    E_total_norm : f64[T] - ||f(t)||^2 at each step
    """
    cdef Py_ssize_t T = trajectory.shape[0]
    cdef Py_ssize_t nE = trajectory.shape[1]
    cdef Py_ssize_t t, j

    cdef np.ndarray[f64, ndim=1] E_kin = np.empty(T, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] E_pot = np.empty(T, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] ratio = np.empty(T, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] norms = np.empty(T, dtype=np.float64)
    cdef f64[::1] ekv = E_kin, epv = E_pot, rv = ratio, nv = norms

    cdef np.ndarray[f64, ndim=1] f_t, L1f, LOf
    cdef f64 ek, ep, nm

    for t in range(T):
        f_t = trajectory[t].copy()
        L1f = _safe_dot(L1, f_t)
        LOf = _safe_dot(LO, f_t)

        ek = 0.0
        ep = 0.0
        nm = 0.0
        for j in range(nE):
            ek += f_t[j] * L1f[j]
            ep += f_t[j] * LOf[j]
            nm += f_t[j] * f_t[j]

        ekv[t] = ek
        epv[t] = ep
        nv[t] = nm

        if ep > 1e-15:
            rv[t] = ek / ep
        elif ek > 1e-15:
            rv[t] = 1e15
        else:
            rv[t] = 1.0

    return E_kin, E_pot, ratio, norms


def hodge_energy_decomposition(np.ndarray[f64, ndim=1] f_E,
                               object B1, object B2,
                               object L0, object L2,
                               object L1):
    """Hodge-decompose an edge signal and compute energy per component.

    Decomposes f_E = grad + curl + harm, then computes:
        E_grad = <grad | L_1 | grad>
        E_curl = <curl | L_1 | curl>
        E_harm = <harm | L_1 | harm>  (should be ~0 since L_1 harm = 0)

    Parameters
    ----------
    f_E : f64[nE]
    B1, B2 : boundary operators (B2 should be B2_hodge, self-loops filtered)
    L0, L2 : vertex and face Laplacians (or None, built internally by hodge)
    L1 : edge Hodge Laplacian

    Returns
    -------
    grad : f64[nE]
    curl : f64[nE]
    harm : f64[nE]
    E_grad : float
    E_curl : float
    E_harm : float
    pct_grad : float - fraction of ||f||^2 in gradient subspace
    pct_curl : float
    pct_harm : float
    """
    from rexgraph.core._hodge import hodge_decomposition, compute_energy_percentages

    grad, curl, harm = hodge_decomposition(B1, B2, f_E, L0, L2)

    # Energy per component
    cdef np.ndarray[f64, ndim=1] L1g = _safe_dot(L1, grad)
    cdef np.ndarray[f64, ndim=1] L1c = _safe_dot(L1, curl)
    cdef np.ndarray[f64, ndim=1] L1h = _safe_dot(L1, harm)

    cdef Py_ssize_t nE = f_E.shape[0], j
    cdef f64 eg = 0.0, ec = 0.0, eh = 0.0

    for j in range(nE):
        eg += grad[j] * L1g[j]
        ec += curl[j] * L1c[j]
        eh += harm[j] * L1h[j]

    pct_g, pct_c, pct_h = compute_energy_percentages(grad, curl, harm)

    return grad, curl, harm, eg, ec, eh, pct_g, pct_c, pct_h


def per_edge_energy_trajectory(np.ndarray[f64, ndim=2] trajectory,
                               object L1,
                               object LO):
    """Compute per-edge E_kin and E_pot at each timestep.

    E_kin_e(t) = f_e(t) * (L_1 f(t))_e
    E_pot_e(t) = f_e(t) * (L_O f(t))_e

    These per-edge values sum to the total E_kin and E_pot.

    Parameters
    ----------
    trajectory : f64[T, nE]
    L1, LO : edge Laplacians

    Returns
    -------
    Ekin_per_edge : f64[T, nE]
    Epot_per_edge : f64[T, nE]
    """
    cdef Py_ssize_t T = trajectory.shape[0]
    cdef Py_ssize_t nE = trajectory.shape[1]
    cdef Py_ssize_t t, j

    cdef np.ndarray[f64, ndim=2] Ek = np.empty((T, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] Ep = np.empty((T, nE), dtype=np.float64)
    cdef f64[:, ::1] ekv = Ek, epv = Ep

    cdef np.ndarray[f64, ndim=1] f_t, L1f, LOf

    for t in range(T):
        f_t = trajectory[t].copy()
        L1f = _safe_dot(L1, f_t)
        LOf = _safe_dot(LO, f_t)
        for j in range(nE):
            ekv[t, j] = f_t[j] * L1f[j]
            epv[t, j] = f_t[j] * LOf[j]

    return Ek, Ep


# Section 4: Cascade analysis


def cascade_from_edge(np.ndarray[f64, ndim=2] trajectory,
                      double threshold=-1.0):
    """Compute cascade activation order from an edge signal trajectory.

    Wraps the temporal cascade_edge_activation function with an
    automatic threshold based on peak signal magnitude.

    Parameters
    ----------
    trajectory : f64[T, nE] - edge signal trajectory (from propagation)
    threshold : float - activation threshold. If negative, auto-computed
        as 0.5% of peak signal magnitude across all timesteps.

    Returns
    -------
    activation_time : i32[nE] - first timestep each edge exceeds threshold (-1 = never)
    activation_order : i32[n_activated] - edges sorted by activation time
    activation_rank : i32[nE] - rank in activation order (-1 = never)
    threshold_used : float
    """
    cdef double thresh = threshold
    if thresh < 0:
        # Auto: 0.5% of peak magnitude
        peak = np.max(np.abs(trajectory))
        thresh = 0.005 * peak if peak > 1e-15 else 1e-10

    # Use absolute values for activation detection
    cdef np.ndarray[f64, ndim=2] abs_traj = np.abs(trajectory)

    from rexgraph.core._temporal import cascade_edge_activation
    act_time, act_order, act_rank = cascade_edge_activation(abs_traj, thresh)

    return act_time, act_order, act_rank, thresh


def face_emergence(np.ndarray[f64, ndim=2] trajectory,
                   object B2,
                   double threshold=-1.0):
    """Track when faces activate during signal propagation.

    A face is active at timestep t when the minimum |signal| across
    all its boundary edges exceeds the threshold. This models face
    "emergence" -- a higher-order structure becomes active when all
    its constituent edges are active.

    Parameters
    ----------
    trajectory : f64[T, nE]
    B2 : (nE, nF) - edge-face boundary (dense or sparse)
    threshold : float - auto if negative

    Returns
    -------
    face_activation_time : i32[nF] - first timestep face activates (-1 = never)
    face_order : i32[n_activated] - faces sorted by activation time
    """
    cdef Py_ssize_t T = trajectory.shape[0]
    cdef np.ndarray[f64, ndim=2] B2_d = np.asarray(B2, dtype=np.float64)
    cdef Py_ssize_t nE = B2_d.shape[0], nF = B2_d.shape[1]
    cdef Py_ssize_t t, f, e

    cdef double thresh = threshold
    if thresh < 0:
        peak = np.max(np.abs(trajectory))
        thresh = 0.005 * peak if peak > 1e-15 else 1e-10

    # For each face, find its boundary edges (nonzero entries in B2 column)
    # Then check when min(|signal|) across boundary edges exceeds threshold
    cdef np.ndarray[i32, ndim=1] face_act = np.full(nF, -1, dtype=np.int32)
    cdef i32[::1] fav = face_act
    cdef f64[:, ::1] b2v = B2_d

    cdef np.ndarray[f64, ndim=2] abs_traj = np.abs(trajectory)
    cdef f64[:, ::1] atv = abs_traj

    for f in range(nF):
        # Collect boundary edges of face f
        boundary_edges = []
        for e in range(nE):
            if fabs(b2v[e, f]) > 1e-15:
                boundary_edges.append(e)

        if len(boundary_edges) == 0:
            continue

        # Find first time all boundary edges exceed threshold
        for t in range(T):
            all_active = True
            for e_idx in boundary_edges:
                if atv[t, e_idx] < thresh:
                    all_active = False
                    break
            if all_active:
                fav[f] = <i32>t
                break

    # Build activation order
    activated = [(face_act[f], f) for f in range(nF) if face_act[f] >= 0]
    activated.sort()
    cdef np.ndarray[i32, ndim=1] face_order = np.array(
        [f for _, f in activated], dtype=np.int32)

    return face_act, face_order


def cascade_depth(np.ndarray[i32, ndim=1] activation_order,
                  np.ndarray[i32, ndim=1] edge_src,
                  np.ndarray[i32, ndim=1] edge_tgt,
                  Py_ssize_t nE):
    """Compute topological distance from perturbation source.

    Uses BFS on the edge adjacency graph (edges sharing a vertex)
    starting from the first-activated edge.

    depth 0 = perturbed edge
    depth 1 = star neighborhood (edges sharing a boundary vertex)
    depth 2 = star of star, etc.

    Parameters
    ----------
    activation_order : i32[n_activated] - edges sorted by activation time
    edge_src, edge_tgt : i32[nE] - edge endpoints
    nE : number of edges

    Returns
    -------
    depth : i32[nE] - topological depth from source (-1 if unreached)
    """
    if activation_order.shape[0] == 0:
        return np.full(nE, -1, dtype=np.int32)

    cdef i32[::1] es = edge_src, et = edge_tgt
    cdef Py_ssize_t e, e2

    # Build edge adjacency: edges sharing a vertex
    # vertex -> list of incident edges
    cdef i32 max_v = 0
    for e in range(nE):
        if es[e] > max_v: max_v = es[e]
        if et[e] > max_v: max_v = et[e]
    cdef Py_ssize_t nV = <Py_ssize_t>(max_v + 1)

    v2e = [[] for _ in range(nV)]
    for e in range(nE):
        v2e[es[e]].append(e)
        v2e[et[e]].append(e)

    # BFS from the first activated edge
    cdef np.ndarray[i32, ndim=1] depth = np.full(nE, -1, dtype=np.int32)
    cdef i32[::1] dv = depth

    cdef i32 seed = activation_order[0]
    dv[seed] = 0

    queue = [seed]
    cdef Py_ssize_t qi = 0
    cdef i32 cur_d

    while qi < len(queue):
        e = queue[qi]
        qi += 1
        cur_d = dv[e]

        # Neighbors: edges sharing vertex es[e] or et[e]
        for v in [es[e], et[e]]:
            for e2 in v2e[v]:
                if dv[e2] < 0:
                    dv[e2] = cur_d + 1
                    queue.append(e2)

    return depth


# Section 5: Temporal tagging


def tag_energy_phases(np.ndarray[f64, ndim=1] E_kin,
                      np.ndarray[f64, ndim=1] E_pot,
                      double ratio_tol=0.2,
                      Py_ssize_t min_phase_len=2,
                      double floor=1e-12):
    """BIOES tagging from energy ratio timeseries.

    Wraps _temporal.compute_bioes_energy for the signal pipeline.

    Parameters
    ----------
    E_kin : f64[T] - topological energy per timestep
    E_pot : f64[T] - geometric energy per timestep
    ratio_tol : log-ratio threshold for crossover band
    min_phase_len : minimum phase length for BIOES assignment
    floor : minimum energy value

    Returns
    -------
    tags : i32[T] - BIOES tags (0=B, 1=I, 2=O, 3=E, 4=S)
    phase_start : i32[n_phases]
    phase_end : i32[n_phases]
    phase_regime : i32[n_phases] (0=kinetic, 1=crossover, 2=potential)
    log_ratios : f64[T]
    crossover_times : i32[n_crossover]
    """
    from rexgraph.core._temporal import compute_bioes_energy
    return compute_bioes_energy(E_kin, E_pot, ratio_tol, min_phase_len, floor)


def tag_cascade_phases(np.ndarray[i32, ndim=1] activation_time,
                       Py_ssize_t T):
    """Assign temporal tags based on cascade wavefront.

    Each timestep is tagged by its cascade activity:
        0 = no new activations (quiet)
        1 = new edges activated (wavefront advancing)
        2 = peak activation step (most new edges in one step)

    Parameters
    ----------
    activation_time : i32[nE] - per-edge activation timestep (-1 = never)
    T : number of timesteps

    Returns
    -------
    step_tags : i32[T]
    new_per_step : i32[T] - count of newly activated edges per step
    """
    cdef Py_ssize_t nE = activation_time.shape[0]
    cdef i32[::1] atv = activation_time
    cdef Py_ssize_t e, t

    cdef np.ndarray[i32, ndim=1] new_counts = np.zeros(T, dtype=np.int32)
    cdef i32[::1] ncv = new_counts

    for e in range(nE):
        if 0 <= atv[e] < T:
            ncv[atv[e]] += 1

    # Find peak step
    cdef i32 peak_step = 0, peak_count = 0
    for t in range(T):
        if ncv[t] > peak_count:
            peak_count = ncv[t]
            peak_step = <i32>t

    cdef np.ndarray[i32, ndim=1] tags = np.zeros(T, dtype=np.int32)
    cdef i32[::1] tv = tags
    for t in range(T):
        if ncv[t] > 0:
            tv[t] = 1
    if peak_count > 0:
        tv[peak_step] = 2

    return tags, new_counts


# Section 6: Full pipeline


def analyze_perturbation(np.ndarray[f64, ndim=1] f_E,
                         np.ndarray[f64, ndim=1] f_F,
                         object L1,
                         object LO,
                         np.ndarray[f64, ndim=1] evals_RL1,
                         np.ndarray[f64, ndim=2] evecs_RL1,
                         object B1,
                         object B2,
                         np.ndarray[f64, ndim=1] times,
                         object L0=None,
                         object L2_op=None,
                         object RL1=None,
                         np.ndarray[i32, ndim=1] edge_src=None,
                         np.ndarray[i32, ndim=1] edge_tgt=None,
                         double alpha_G=1.0):
    """One-call perturbation analysis pipeline.

    1. Propagate f_E under RL_1 diffusion (spectral)
    2. Compute energy trajectory (E_kin, E_pot, ratio)
    3. Per-edge energy at each timestep
    4. Cascade activation order
    5. Face emergence times
    6. BIOES phase tags
    7. Hodge decomposition of initial and final states

    Parameters
    ----------
    f_E : f64[nE] - initial edge signal
    f_F : f64[nF] - initial face signal (usually zeros)
    L1 : (nE, nE) - Hodge Laplacian
    LO : (nE, nE) - overlap Laplacian
    evals_RL1, evecs_RL1 : eigendecomposition of RL_1
    B1 : (nV, nE) - vertex-edge boundary
    B2 : (nE, nF) - edge-face boundary (B2_hodge preferred)
    times : f64[T] - timepoints
    L0 : vertex Laplacian (optional, for Hodge)
    L2_op : face Laplacian (optional, for Hodge)
    RL1 : (nE, nE) - Relational Laplacian matrix (optional, built if needed)
    edge_src, edge_tgt : i32[nE] - edge endpoints (for cascade depth)
    alpha_G : coupling constant

    Returns
    -------
    result : dict with keys:
        trajectory : f64[T, nE]
        E_kin, E_pot, ratio, norms : f64[T] each
        Ekin_per_edge, Epot_per_edge : f64[T, nE] each
        activation_time, activation_order, activation_rank : arrays
        activation_threshold : float
        face_activation_time, face_order : arrays
        bioes_tags, phase_start, phase_end, phase_regime : arrays
        log_ratios : f64[T]
        crossover_times : i32 array
        hodge_initial : dict (grad, curl, harm, E_grad, E_curl, E_harm, pct)
        hodge_final : dict
        cascade_depth : i32[nE] or None
        f_V_initial : f64[nV] (derived vertex observable)
        f_V_final : f64[nV]
    """
    cdef Py_ssize_t nE = f_E.shape[0]
    cdef Py_ssize_t T = times.shape[0]

    # Step 1: Propagate under RL_1 diffusion
    trajectory = propagate_diffusion(f_E, None, evals_RL1, evecs_RL1, times)

    # Step 2: Energy trajectory
    E_kin, E_pot, ratio, norms = energy_trajectory(trajectory, L1, LO)

    # Step 3: Per-edge energy
    Ekin_pe, Epot_pe = per_edge_energy_trajectory(trajectory, L1, LO)

    # Step 4: Cascade
    act_time, act_order, act_rank, thresh = cascade_from_edge(trajectory)

    # Step 5: Face emergence
    face_act, face_ord = face_emergence(trajectory, B2)

    # Step 6: BIOES tags
    bioes = tag_energy_phases(E_kin, E_pot)
    tags, p_start, p_end, p_regime, log_ratios, crossover_t = bioes

    # Step 7: Hodge decomposition (initial and final)
    hodge_init = None
    hodge_fin = None
    try:
        g0, c0, h0, eg0, ec0, eh0, pg0, pc0, ph0 = hodge_energy_decomposition(
            f_E, B1, B2, L0, L2_op, L1)
        hodge_init = {
            "grad": g0, "curl": c0, "harm": h0,
            "E_grad": eg0, "E_curl": ec0, "E_harm": eh0,
            "pct_grad": pg0, "pct_curl": pc0, "pct_harm": ph0,
        }

        f_final = trajectory[-1] if T > 0 else f_E
        gf, cf, hf, egf, ecf, ehf, pgf, pcf, phf = hodge_energy_decomposition(
            f_final, B1, B2, L0, L2_op, L1)
        hodge_fin = {
            "grad": gf, "curl": cf, "harm": hf,
            "E_grad": egf, "E_curl": ecf, "E_harm": ehf,
            "pct_grad": pgf, "pct_curl": pcf, "pct_harm": phf,
        }
    except Exception:
        pass  # Hodge may fail if B2 is missing or incompatible

    # Derived vertex observables
    f_V_init = np.asarray(B1.dot(f_E), dtype=np.float64)
    f_V_final = np.asarray(B1.dot(trajectory[-1] if T > 0 else f_E),
                           dtype=np.float64)

    # Cascade depth
    depth = None
    if edge_src is not None and edge_tgt is not None and act_order.shape[0] > 0:
        depth = cascade_depth(act_order, edge_src, edge_tgt, nE)

    return {
        "trajectory": trajectory,
        "E_kin": E_kin,
        "E_pot": E_pot,
        "ratio": ratio,
        "norms": norms,
        "Ekin_per_edge": Ekin_pe,
        "Epot_per_edge": Epot_pe,
        "activation_time": act_time,
        "activation_order": act_order,
        "activation_rank": act_rank,
        "activation_threshold": thresh,
        "face_activation_time": face_act,
        "face_order": face_ord,
        "bioes_tags": tags,
        "phase_start": p_start,
        "phase_end": p_end,
        "phase_regime": p_regime,
        "log_ratios": log_ratios,
        "crossover_times": crossover_t,
        "hodge_initial": hodge_init,
        "hodge_final": hodge_fin,
        "cascade_depth": depth,
        "f_V_initial": f_V_init,
        "f_V_final": f_V_final,
        "alpha_G": alpha_G,
        "T": T,
        "nE": nE,
    }


def analyze_perturbation_field(np.ndarray[f64, ndim=1] f_E,
                               np.ndarray[f64, ndim=1] f_F,
                               np.ndarray[f64, ndim=2] M_field,
                               np.ndarray[f64, ndim=1] evals_M,
                               np.ndarray[f64, ndim=2] evecs_M,
                               np.ndarray[f64, ndim=1] freqs_M,
                               object L1,
                               object LO,
                               object B1,
                               np.ndarray[f64, ndim=1] times,
                               Py_ssize_t nE,
                               Py_ssize_t nF,
                               str mode="diffusion"):
    """Perturbation analysis using the full (E, F) field operator.

    Propagates the packed field state F = [f_E, f_F] under the coupled
    field operator M from _field.pyx, then extracts per-dimension
    energy and cascade information.

    Parameters
    ----------
    f_E : f64[nE]
    f_F : f64[nF]
    M_field : f64[nE+nF, nE+nF] - field operator
    evals_M, evecs_M, freqs_M : eigendecomposition of M
    L1, LO : edge Laplacians (for E_kin/E_pot decomposition)
    B1 : boundary operator (for vertex derivation)
    times : f64[T]
    nE, nF : dimensions
    mode : 'diffusion' or 'wave'

    Returns
    -------
    result : dict with keys:
        field_trajectory : f64[T, nE+nF]
        edge_trajectory : f64[T, nE] (extracted edge block)
        face_trajectory : f64[T, nF] (extracted face block)
        vertex_trajectory : f64[T, nV] (derived via B_1)
        E_kin, E_pot, ratio : f64[T]
        norm_E, norm_F : f64[T] per-dimension signal norms
        wave_KE, wave_PE, wave_total : f64[T] (wave mode only)
    """
    cdef Py_ssize_t T = times.shape[0], n = nE + nF

    # Pack field state
    cdef np.ndarray[f64, ndim=1] F0 = np.concatenate([f_E, f_F])

    from rexgraph.core._field import (
        field_diffusion_trajectory, wave_evolve_trajectory,
        wave_energy, wave_dimensional_energy,
    )

    cdef np.ndarray[f64, ndim=2] field_traj
    cdef np.ndarray[f64, ndim=2] vel_traj = None

    if mode == "wave":
        field_traj, vel_traj = wave_evolve_trajectory(
            F0, evals_M, evecs_M, freqs_M, times)
    else:
        field_traj = field_diffusion_trajectory(F0, evals_M, evecs_M, times)

    # Extract edge and face blocks
    edge_traj = field_traj[:, :nE].copy()
    face_traj = field_traj[:, nE:].copy()

    # Derive vertex trajectory
    from rexgraph.core._field import derive_vertex_trajectory
    vertex_traj = derive_vertex_trajectory(field_traj, B1, nE)

    # Energy decomposition on edge block
    E_kin, E_pot, ratio, norms = energy_trajectory(edge_traj, L1, LO)

    # Per-dimension norms
    cdef np.ndarray[f64, ndim=1] norm_E = np.empty(T, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] norm_F = np.empty(T, dtype=np.float64)
    cdef Py_ssize_t t, j
    cdef f64 se, sf

    for t in range(T):
        se = 0.0
        sf = 0.0
        for j in range(nE):
            se += edge_traj[t, j] * edge_traj[t, j]
        for j in range(nF):
            sf += face_traj[t, j] * face_traj[t, j]
        norm_E[t] = sqrt(se)
        norm_F[t] = sqrt(sf)

    result = {
        "field_trajectory": field_traj,
        "edge_trajectory": edge_traj,
        "face_trajectory": face_traj,
        "vertex_trajectory": vertex_traj,
        "E_kin": E_kin,
        "E_pot": E_pot,
        "ratio": ratio,
        "norm_E": norm_E,
        "norm_F": norm_F,
        "mode": mode,
        "nE": nE,
        "nF": nF,
        "T": T,
    }

    # Wave-specific: total energy conservation
    if mode == "wave" and vel_traj is not None:
        wave_KE = np.empty(T, dtype=np.float64)
        wave_PE = np.empty(T, dtype=np.float64)
        wave_total = np.empty(T, dtype=np.float64)
        for t in range(T):
            ke, pe, tot = wave_energy(field_traj[t], vel_traj[t], M_field)
            wave_KE[t] = ke
            wave_PE[t] = pe
            wave_total[t] = tot
        result["wave_KE"] = wave_KE
        result["wave_PE"] = wave_PE
        result["wave_total"] = wave_total
        result["velocity_trajectory"] = vel_traj

    return result
