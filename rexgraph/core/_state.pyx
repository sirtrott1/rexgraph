# rexgraph/core/_state.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._state - Rex state representation and signal operations.

A rex state consists of per-dimension signals f0 (vertices), f1 (edges),
f2 (faces). This module provides norms, normalization, packing/unpacking,
differencing, projection, energy computation, stochastic validation,
and state construction helpers.

In the rex framework, edges are the primitive elements and vertices
are boundaries of edges. The field state (f_E, f_F) lives on edges
and faces; vertex observables are derived via f_V = B_1 f_E.

Energy decomposition under the Relational Laplacian RL_1 = L_1 + alpha_G * L_O:
    E_kin = <f_E | L_1 | f_E>    (topological energy from Hodge Laplacian)
    E_pot = <f_E | L_O | f_E>    (geometric energy from overlap Laplacian)
    E_tot = E_kin + alpha_G * E_pot

Includes RexState class for managing signal evolution and energy.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
cimport cython
from rexgraph.core._common cimport i32, i64, f64
from libc.math cimport fabs, sqrt
from libc.string cimport memcpy

np.import_array()

cdef enum:
    _NORM_L1 = 0
    _NORM_L2 = 1
    _NORM_LINF = 2

# Python-visible constants for use as default arguments
NORM_L1 = _NORM_L1
NORM_L2 = _NORM_L2
NORM_LINF = _NORM_LINF

# Lazy import of _transition (avoids import-order issues during parallel builds)
cdef object _transition_mod = None

cdef object _get_transition():
    global _transition_mod
    if _transition_mod is None:
        from rexgraph.core import _transition
        _transition_mod = _transition
    return _transition_mod


# Signal Norms & Helpers

def signal_norm_l1(np.ndarray[f64, ndim=1] signal):
    """L1 norm: sum |f_i|. Used for probability distributions."""
    cdef f64[::1] sv = signal
    cdef Py_ssize_t n = signal.shape[0], j
    cdef f64 total = 0.0
    for j in range(n):
        total += fabs(sv[j])
    return total

def signal_norm_l2(np.ndarray[f64, ndim=1] signal):
    """L2 norm: sqrt(sum f_i^2). Used for amplitude vectors."""
    cdef f64[::1] sv = signal
    cdef Py_ssize_t n = signal.shape[0], j
    cdef f64 total = 0.0
    for j in range(n):
        total += sv[j] * sv[j]
    return sqrt(total)

def signal_norm_linf(np.ndarray[f64, ndim=1] signal):
    """L-infinity norm: max |f_i|."""
    cdef f64[::1] sv = signal
    cdef Py_ssize_t n = signal.shape[0], j
    cdef f64 mx = 0.0, v
    for j in range(n):
        v = fabs(sv[j])
        if v > mx: mx = v
    return mx

def signal_norm(np.ndarray[f64, ndim=1] signal, int norm_type=NORM_L2):
    """Compute norm of signal. norm_type: 0=L1, 1=L2, 2=Linf."""
    if norm_type == NORM_L1: return signal_norm_l1(signal)
    if norm_type == NORM_L2: return signal_norm_l2(signal)
    return signal_norm_linf(signal)

def normalize_l1(np.ndarray[f64, ndim=1] signal):
    """Normalize to L1=1 (probability distribution). Returns copy."""
    cdef f64 n = signal_norm_l1(signal)
    if n < 1e-15: return signal.copy()
    return signal / n

def normalize_l2(np.ndarray[f64, ndim=1] signal):
    """Normalize to L2=1 (unit amplitude). Returns copy."""
    cdef f64 n = signal_norm_l2(signal)
    if n < 1e-15: return signal.copy()
    return signal / n

def normalize_signal(np.ndarray[f64, ndim=1] signal, int norm_type=NORM_L2):
    """Normalize signal by specified norm type."""
    if norm_type == NORM_L1: return normalize_l1(signal)
    if norm_type == NORM_L2: return normalize_l2(signal)
    return signal.copy()


# State Packing & Unpacking

def pack_state(np.ndarray[f64, ndim=1] f0,
               np.ndarray[f64, ndim=1] f1,
               np.ndarray[f64, ndim=1] f2):
    """Pack per-dimension signals into a single flat vector.

    Returns
    -------
    flat : f64[nV + nE + nF]
    sizes : int32[3] [nV, nE, nF]
    """
    cdef Py_ssize_t nV = f0.shape[0], nE = f1.shape[0], nF = f2.shape[0]
    cdef Py_ssize_t total = nV + nE + nF
    cdef np.ndarray[f64, ndim=1] flat = np.empty(total, dtype=np.float64)

    if nV > 0: memcpy(&flat[0], &f0[0], nV * sizeof(f64))
    if nE > 0: memcpy(&flat[nV], &f1[0], nE * sizeof(f64))
    if nF > 0: memcpy(&flat[nV + nE], &f2[0], nF * sizeof(f64))

    sizes = np.array([nV, nE, nF], dtype=np.int32)
    return flat, sizes

def unpack_state(np.ndarray[f64, ndim=1] flat,
                 np.ndarray[i32, ndim=1] sizes):
    """Unpack flat vector back to per-dimension signals."""
    cdef i32[::1] sz = sizes
    cdef Py_ssize_t nV = sz[0], nE = sz[1], nF = sz[2]
    cdef np.ndarray[f64, ndim=1] f0 = np.empty(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f1 = np.empty(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f2 = np.empty(nF, dtype=np.float64)

    if nV > 0: memcpy(&f0[0], &flat[0], nV * sizeof(f64))
    if nE > 0: memcpy(&f1[0], &flat[nV], nE * sizeof(f64))
    if nF > 0: memcpy(&f2[0], &flat[nV + nE], nF * sizeof(f64))

    return f0, f1, f2


# Field State Packing (Edge + Face only, V-as-boundary)

def field_state_pack(np.ndarray[f64, ndim=1] f_E,
                     np.ndarray[f64, ndim=1] f_F):
    """Pack edge and face signals into a field state vector.

    In the rex framework, vertices are boundaries of edges, not
    independent degrees of freedom. The field state lives on
    (E, F) only; vertex observables are derived via f_V = B_1 f_E.

    Returns
    -------
    flat : f64[nE + nF]
    sizes : int32[2] [nE, nF]
    """
    cdef Py_ssize_t nE = f_E.shape[0], nF = f_F.shape[0]
    cdef np.ndarray[f64, ndim=1] flat = np.empty(nE + nF, dtype=np.float64)

    if nE > 0: memcpy(&flat[0], &f_E[0], nE * sizeof(f64))
    if nF > 0: memcpy(&flat[nE], &f_F[0], nF * sizeof(f64))

    sizes = np.array([nE, nF], dtype=np.int32)
    return flat, sizes

def field_state_unpack(np.ndarray[f64, ndim=1] flat,
                       np.ndarray[i32, ndim=1] sizes):
    """Unpack field state vector back to edge and face signals.

    Returns
    -------
    f_E : f64[nE]
    f_F : f64[nF]
    """
    cdef i32[::1] sz = sizes
    cdef Py_ssize_t nE = sz[0], nF = sz[1]
    cdef np.ndarray[f64, ndim=1] f_E = np.empty(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f_F = np.empty(nF, dtype=np.float64)

    if nE > 0: memcpy(&f_E[0], &flat[0], nE * sizeof(f64))
    if nF > 0: memcpy(&f_F[0], &flat[nE], nF * sizeof(f64))

    return f_E, f_F

def field_state_vertex_observable(np.ndarray[f64, ndim=1] f_E, object B1):
    """Derive vertex observable from edge signal via boundary operator.

    f_V = B_1 f_E

    This is the correct way to obtain vertex signals in the rex
    framework. The result is the divergence of edge flow at each
    vertex, not an independently evolved quantity.

    Parameters
    ----------
    f_E : f64[nE] - edge signal
    B1 : (nV, nE) matrix (dense or sparse) - boundary operator

    Returns
    -------
    f_V : f64[nV]
    """
    return np.asarray(B1.dot(f_E), dtype=np.float64)


# State Differencing

def state_diff(np.ndarray[f64, ndim=1] state_a,
               np.ndarray[f64, ndim=1] state_b):
    """Compute state delta: diff = state_b - state_a."""
    cdef Py_ssize_t n = state_a.shape[0], j
    cdef np.ndarray[f64, ndim=1] diff = np.empty(n, dtype=np.float64)
    cdef f64[::1] dv = diff, av = state_a, bv = state_b
    for j in range(n):
        dv[j] = bv[j] - av[j]
    return diff

def state_apply_diff(np.ndarray[f64, ndim=1] state,
                     np.ndarray[f64, ndim=1] diff):
    """Apply delta to state: result = state + diff."""
    cdef Py_ssize_t n = state.shape[0], j
    cdef np.ndarray[f64, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef f64[::1] rv = result, sv = state, dv = diff
    for j in range(n):
        rv[j] = sv[j] + dv[j]
    return result


# Energy Computation

def energy_kin_pot(np.ndarray[f64, ndim=1] f_E, object L1, object LO):
    """Compute kinetic and potential energy of an edge signal.

    E_kin = <f_E | L_1 | f_E>  (topological energy)
    E_pot = <f_E | L_O | f_E>  (geometric energy)

    Parameters
    ----------
    f_E : f64[nE] - edge signal
    L1 : (nE, nE) - Hodge Laplacian (dense or sparse)
    LO : (nE, nE) - overlap Laplacian (dense or sparse)

    Returns
    -------
    E_kin : float
    E_pot : float
    ratio : float (E_kin / E_pot, inf if E_pot ~ 0)
    """
    cdef np.ndarray[f64, ndim=1] L1f = np.asarray(L1.dot(f_E), dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] LOf = np.asarray(LO.dot(f_E), dtype=np.float64)

    cdef f64[::1] fv = f_E, l1v = L1f, lov = LOf
    cdef Py_ssize_t n = f_E.shape[0], j
    cdef f64 ek = 0.0, ep = 0.0

    for j in range(n):
        ek += fv[j] * l1v[j]
        ep += fv[j] * lov[j]

    cdef f64 ratio
    if ep > 1e-15:
        ratio = ek / ep
    elif ek > 1e-15:
        ratio = 1e15  # effectively infinity
    else:
        ratio = 1.0  # both zero

    return ek, ep, ratio


# State Construction Helpers

def uniform_state(Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF,
                  int norm_type=NORM_L1):
    """Create uniform state at all dimensions."""
    cdef f64 v0, v1, v2
    if norm_type == NORM_L1:
        v0 = 1.0 / <f64>nV if nV > 0 else 0.0
        v1 = 1.0 / <f64>nE if nE > 0 else 0.0
        v2 = 1.0 / <f64>nF if nF > 0 else 0.0
    else:
        v0 = 1.0 / sqrt(<f64>nV) if nV > 0 else 0.0
        v1 = 1.0 / sqrt(<f64>nE) if nE > 0 else 0.0
        v2 = 1.0 / sqrt(<f64>nF) if nF > 0 else 0.0

    cdef np.ndarray[f64, ndim=1] f0 = np.full(nV, v0, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f1 = np.full(nE, v1, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f2 = np.full(nF, v2, dtype=np.float64)
    return f0, f1, f2

def dirac_state(Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF,
                Py_ssize_t dim, Py_ssize_t idx):
    """Create Dirac delta state: all zeros except 1.0 at (dim, idx)."""
    cdef np.ndarray[f64, ndim=1] f0 = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f1 = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f2 = np.zeros(nF, dtype=np.float64)
    if dim == 0 and idx < nV: f0[idx] = 1.0
    elif dim == 1 and idx < nE: f1[idx] = 1.0
    elif dim == 2 and idx < nF: f2[idx] = 1.0
    return f0, f1, f2

def dirac_edge(Py_ssize_t nE, Py_ssize_t nF, Py_ssize_t edge_idx):
    """Create Dirac delta on a single edge (field state, no V).

    For perturbation analysis: activates one edge in the (E, F) field.
    Vertex activation is derived via B_1.
    """
    cdef np.ndarray[f64, ndim=1] f_E = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f_F = np.zeros(nF, dtype=np.float64)
    if edge_idx < nE: f_E[edge_idx] = 1.0
    return f_E, f_F

def vertex_perturbation_to_edges(Py_ssize_t vertex_idx, object B1_T,
                                 Py_ssize_t nE, Py_ssize_t nF):
    """Convert a vertex perturbation to edge signal via B_1^T.

    In the rex framework, "activate vertex v" means "activate all
    edges incident to v". The edge signal is B_1^T delta_v.

    Parameters
    ----------
    vertex_idx : int - vertex to perturb
    B1_T : (nE, nV) matrix - transpose of boundary operator
    nE, nF : dimensions

    Returns
    -------
    f_E : f64[nE] - edge signal (B_1^T delta_v)
    f_F : f64[nF] - zero face signal
    """
    cdef Py_ssize_t nV = B1_T.shape[1]
    cdef np.ndarray[f64, ndim=1] delta_v = np.zeros(nV, dtype=np.float64)
    delta_v[vertex_idx] = 1.0
    cdef np.ndarray[f64, ndim=1] f_E = np.asarray(B1_T.dot(delta_v),
                                                    dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f_F = np.zeros(nF, dtype=np.float64)
    return f_E, f_F

def random_state(Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF,
                 int norm_type=NORM_L1):
    """Create random state, normalized by norm_type."""
    cdef np.ndarray[f64, ndim=1] f0 = np.abs(np.random.randn(nV))
    cdef np.ndarray[f64, ndim=1] f1 = np.abs(np.random.randn(nE))
    cdef np.ndarray[f64, ndim=1] f2 = np.abs(np.random.randn(nF))
    f0 = normalize_signal(f0, norm_type)
    f1 = normalize_signal(f1, norm_type)
    f2 = normalize_signal(f2, norm_type)
    return f0, f1, f2


# Rex State Class

cdef class RexState:
    """Container for signals on the 2-rex chain complex.

    Attributes
    ----------
    f0 : f64[nV] (vertex signals, derived from f1 via B_1)
    f1 : f64[nE] (edge signals, primary)
    f2 : f64[nF] (face signals)
    t  : double  (current time of state)

    Energy decomposition under RL_1 = L_1 + alpha_G * L_O:
        E_kin = <f1 | L_1 | f1>  (topological)
        E_pot = <f1 | L_O | f1>  (geometric)
    """
    cdef public np.ndarray f0
    cdef public np.ndarray f1
    cdef public np.ndarray f2
    cdef public double t

    # Cache for energy components
    cdef public double _last_E_kin
    cdef public double _last_E_pot
    cdef public double _last_E_tot
    cdef bint _energy_dirty

    def __init__(self, nV, nE, nF, t=0.0):
        self.f0 = np.zeros(nV, dtype=np.float64)
        self.f1 = np.zeros(nE, dtype=np.float64)
        self.f2 = np.zeros(nF, dtype=np.float64)
        self.t = t
        self._energy_dirty = True
        self._last_E_kin = 0.0
        self._last_E_pot = 0.0
        self._last_E_tot = 0.0

    @property
    def shapes(self):
        return (self.f0.shape[0], self.f1.shape[0], self.f2.shape[0])

    def set_f0(self, data):
        self.f0[:] = data
        self._energy_dirty = True

    def set_f1(self, data):
        self.f1[:] = data
        self._energy_dirty = True

    def set_f2(self, data):
        self.f2[:] = data
        self._energy_dirty = True

    cpdef void update_energy(self, object L1, object LO, double alpha):
        """Recompute energy components under RL_1 = L_1 + alpha_G * L_O.

        E_kin = <f1 | L1 | f1>  (topological stress)
        E_pot = <f1 | LO | f1>  (geometric mass)
        E_tot = E_kin + alpha * E_pot
        """
        cdef double Ek, Ep, Et
        cdef np.ndarray[f64, ndim=1] zeros = np.zeros_like(self.f1)

        Ek, Ep, Et = _get_transition().energy_decomposition(self.f1, zeros, L1, LO, alpha)

        self._last_E_kin = Ek
        self._last_E_pot = Ep
        self._last_E_tot = Et
        self._energy_dirty = False

    @property
    def energy(self):
        """Return cached total energy."""
        if self._energy_dirty:
            return float('nan')
        return self._last_E_tot

    @property
    def E_kin(self):
        """Topological (kinetic) energy: <f1 | L_1 | f1>."""
        if self._energy_dirty: return float('nan')
        return self._last_E_kin

    @property
    def E_pot(self):
        """Geometric (potential) energy: <f1 | L_O | f1>."""
        if self._energy_dirty: return float('nan')
        return self._last_E_pot

    def derive_vertex_signal(self, object B1):
        """Derive f0 from f1 via B_1 (V-as-boundary semantics).

        Sets f0 = B_1 @ f1. This replaces independent vertex evolution
        with the derived vertex observable.
        """
        self.f0 = np.asarray(B1.dot(self.f1), dtype=np.float64)

    def evolve_coupled(self, system, double dt, Py_ssize_t n_steps=1):
        """Evolve state using coupled cross-dimensional equations.

        Updates f0, f1, and f2 simultaneously via RK4 integration.

        Parameters
        ----------
        system : RexGraph (wrapper)
            Must provide L0, L1, L2, LO, B1_dense, B2_dense, and alphas.
        dt : float
        n_steps : int
        """
        cdef object L0 = system.L0
        cdef object L1 = system.L1_full
        cdef object L2 = system.L2
        cdef object LO = system.LO
        cdef object B1 = system.B1_dense
        cdef object B2 = system.B2_dense

        cdef double a0 = system.alpha0
        cdef double a1 = system.alpha_T
        cdef double a2 = system.alpha2
        cdef double aG = system.alpha_G

        sizes = np.array(self.shapes, dtype=np.int32)
        _tr = _get_transition()

        def deriv_func(y, t):
            return _tr.coupled_derivative(
                y, sizes, L0, L1, L2, LO, B1, B2,
                a0, a1, a2, aG
            )

        operator_data = {"derivative_func": deriv_func}

        nf0, nf1, nf2 = _tr.apply_transition(
            _tr.TRANS_DIFFERENTIAL,
            self.f0, self.f1, self.f2,
            1, operator_data, dt, self.t, n_steps
        )

        self.f0 = nf0
        self.f1 = nf1
        self.f2 = nf2
        self.t += dt
        self._energy_dirty = True

    def evolve_schrodinger(self, system, double dt):
        """Unitary evolution of the edge field f1 via Relational Laplacian.

        Uses RL_1 = L_1 + alpha_G * L_O as the evolution operator.
        """
        cdef object operator_data = {}

        if system.evals_RL1 is not None:
            operator_data["evals"] = system.evals_RL1
            operator_data["evecs"] = system.evecs_RL1
        elif system.rex_laplacian is not None:
            operator_data["L"] = system.rex_laplacian
        else:
            operator_data["L"] = system.L1_full

        _tr = _get_transition()
        nf0, nf1, nf2 = _tr.apply_transition(
            _tr.TRANS_SCHRODINGER,
            self.f0, self.f1, self.f2,
            1, operator_data, dt, self.t
        )

        self.f1 = nf1
        self.t += dt
        self._energy_dirty = True

    def evolve_diffusion(self, system, double dt, int dim=0):
        """Simple diffusion on a specific dimension.

        dim=0 (f0 via L0), dim=1 (f1 via L1), dim=2 (f2 via L2).
        """
        cdef object L
        if dim == 0: L = system.L0
        elif dim == 1: L = system.L1_full
        elif dim == 2: L = system.L2
        else: raise ValueError("Invalid dim")

        operator_data = {"L": L}
        _tr = _get_transition()

        nf0, nf1, nf2 = _tr.apply_transition(
            _tr.TRANS_MARKOV,
            self.f0, self.f1, self.f2,
            dim, operator_data, dt, self.t, 1
        )

        if dim == 0: self.f0 = nf0
        elif dim == 1: self.f1 = nf1
        elif dim == 2: self.f2 = nf2

        self.t += dt
        self._energy_dirty = True
