# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._wave - Complex-amplitude wave mechanics on the rex chain complex.

Complex-valued counterpart to _state.pyx and _transition.pyx. Operates on
complex amplitudes psi_k in C^n under exp(-i L t) evolution, where L is
any Laplacian (L_0, L_1, L_2, L_O, or the Relational Laplacian RL_1).

Complex state operations - normalization, inner products, Born probabilities,
    fidelity, phase extraction.
Information theory - Shannon, von Neumann, Renyi entropy; participation ratio;
    KL divergence.
Wave evolution - Schrodinger propagation (spectral, RK4, Trotter-Suzuki).
    Includes field_schrodinger_evolve for coupled (V+E+F) evolution.
Interference - superposition, fringe visibility, coherence measures.
Entanglement - tensor product, partial trace, Schmidt decomposition, PPT.
Decoherence - dephasing, amplitude damping, depolarizing, Lindblad.
Measurement - Born sampling, projective collapse, eigenbasis measurement.
Density matrices - pure-to-density, mixed state, purity, von Neumann entropy.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, log2, cos, sin, exp, fabs

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,

    get_EPSILON_NORM,
)

np.import_array()

ctypedef double complex c128


# Complex state operations

def normalize_c128(np.ndarray[np.complex128_t, ndim=1] psi):
    """L2-normalize complex amplitude vector in place. Returns original norm."""
    cdef c128[::1] a = psi
    cdef Py_ssize_t n = psi.shape[0], i
    cdef f64 norm_sq = 0.0, norm, inv_norm, ar, ai

    for i in range(n):
        ar = a[i].real; ai = a[i].imag
        norm_sq += ar * ar + ai * ai

    norm = sqrt(norm_sq)
    if norm > get_EPSILON_NORM():
        inv_norm = 1.0 / norm
        for i in range(n):
            a[i] = a[i] * inv_norm
    return norm


def norm_c128(np.ndarray[np.complex128_t, ndim=1] psi):
    """Compute L2 norm of complex vector."""
    cdef c128[::1] a = psi
    cdef Py_ssize_t n = psi.shape[0], i
    cdef f64 s = 0.0, ar, ai
    for i in range(n):
        ar = a[i].real; ai = a[i].imag
        s += ar * ar + ai * ai
    return sqrt(s)


def inner_product(np.ndarray[np.complex128_t, ndim=1] psi,
                  np.ndarray[np.complex128_t, ndim=1] phi):
    """Inner product <psi|phi> = sum_i conj(psi_i) * phi_i."""
    cdef c128[::1] a = psi, b = phi
    cdef Py_ssize_t n = psi.shape[0], i
    cdef c128 acc = 0.0
    for i in range(n):
        acc = acc + a[i].conjugate() * b[i]
    return acc


def born_probabilities(np.ndarray[np.complex128_t, ndim=1] psi):
    """Born rule: P(i) = |psi_i|^2. Returns f64[n]."""
    cdef c128[::1] a = psi
    cdef Py_ssize_t n = psi.shape[0], i
    cdef np.ndarray[f64, ndim=1] p = np.empty(n, dtype=np.float64)
    cdef f64[::1] pv = p
    cdef f64 ar, ai
    for i in range(n):
        ar = a[i].real; ai = a[i].imag
        pv[i] = ar * ar + ai * ai
    return p


def fidelity_pure(np.ndarray[np.complex128_t, ndim=1] psi,
                  np.ndarray[np.complex128_t, ndim=1] phi):
    """Fidelity |<psi|phi>|^2 between pure states."""
    cdef c128 ip = inner_product(psi, phi)
    return ip.real * ip.real + ip.imag * ip.imag


def trace_distance_pure(np.ndarray[np.complex128_t, ndim=1] psi,
                        np.ndarray[np.complex128_t, ndim=1] phi):
    """Trace distance sqrt(1 - F) for pure states."""
    cdef f64 F = fidelity_pure(psi, phi)
    if F > 1.0: F = 1.0
    if F < 0.0: F = 0.0
    return sqrt(1.0 - F)


def extract_phases(np.ndarray[np.complex128_t, ndim=1] psi,
                   Py_ssize_t ref=0):
    """Extract relative phases w.r.t. reference component."""
    cdef c128[::1] a = psi
    cdef Py_ssize_t n = psi.shape[0], i
    cdef np.ndarray[f64, ndim=1] phases = np.empty(n, dtype=np.float64)
    cdef f64[::1] pv = phases
    cdef f64 ref_phase = np.angle(psi[ref])
    for i in range(n):
        pv[i] = np.angle(psi[i]) - ref_phase
    return phases


def apply_phase_gate(np.ndarray[np.complex128_t, ndim=1] psi,
                     np.ndarray[f64, ndim=1] phases):
    """Apply diagonal phase rotation: out_i = exp(i*phases_i) * psi_i."""
    cdef c128[::1] a = psi
    cdef f64[::1] ph = phases
    cdef Py_ssize_t n = psi.shape[0], i
    cdef np.ndarray[np.complex128_t, ndim=1] out = np.empty(n, dtype=np.complex128)
    cdef c128[::1] ov = out
    cdef f64 c, s, ar, ai
    for i in range(n):
        c = cos(ph[i]); s = sin(ph[i])
        ar = a[i].real; ai = a[i].imag
        ov[i] = (c * ar - s * ai) + (s * ar + c * ai) * 1j
    return out


# Information theory

def shannon_entropy(np.ndarray[np.complex128_t, ndim=1] psi):
    """Shannon entropy H = -sum p_i log2(p_i) from Born probabilities."""
    cdef c128[::1] a = psi
    cdef Py_ssize_t n = psi.shape[0], i
    cdef f64 total = 0.0, inv, pi, ent = 0.0, ar, ai
    for i in range(n):
        ar = a[i].real; ai = a[i].imag
        total += ar * ar + ai * ai
    if total < get_EPSILON_NORM(): return 0.0
    inv = 1.0 / total
    for i in range(n):
        ar = a[i].real; ai = a[i].imag
        pi = (ar * ar + ai * ai) * inv
        if pi > get_EPSILON_NORM():
            ent -= pi * log2(pi)
    return ent


def renyi_entropy(np.ndarray[f64, ndim=1] probs, double alpha):
    """Renyi entropy H_alpha = log2(sum p_i^alpha) / (1 - alpha)."""
    cdef f64[::1] p = probs
    cdef Py_ssize_t n = probs.shape[0], i
    cdef f64 s = 0.0, pi
    if fabs(alpha - 1.0) < get_EPSILON_NORM():
        s = 0.0
        for i in range(n):
            pi = p[i]
            if pi > get_EPSILON_NORM(): s -= pi * log2(pi)
        return s
    for i in range(n):
        pi = p[i]
        if pi > get_EPSILON_NORM():
            s += pi ** alpha
    if s < get_EPSILON_NORM(): return 0.0
    return log2(s) / (1.0 - alpha)


def participation_ratio(np.ndarray[np.complex128_t, ndim=1] psi):
    """PR = 1 / sum |psi_i|^4. Measures signal delocalization."""
    cdef c128[::1] a = psi
    cdef Py_ssize_t n = psi.shape[0], i
    cdef f64 ipr = 0.0, pi, ar, ai
    for i in range(n):
        ar = a[i].real; ai = a[i].imag
        pi = ar * ar + ai * ai
        ipr += pi * pi
    if ipr < get_EPSILON_NORM(): return <f64>n
    return 1.0 / ipr


def signal_purity(np.ndarray[np.complex128_t, ndim=1] psi):
    """Purity = sum |psi_i|^4 (=1 for Dirac delta, =1/n for uniform)."""
    cdef c128[::1] a = psi
    cdef Py_ssize_t n = psi.shape[0], i
    cdef f64 s = 0.0, pi, ar, ai
    for i in range(n):
        ar = a[i].real; ai = a[i].imag
        pi = ar * ar + ai * ai
        s += pi * pi
    return s


def kl_divergence(np.ndarray[f64, ndim=1] p,
                  np.ndarray[f64, ndim=1] q):
    """KL divergence D(p||q) = sum p_i log2(p_i / q_i) in bits."""
    cdef f64[::1] pv = p, qv = q
    cdef Py_ssize_t n = p.shape[0], i
    cdef f64 s = 0.0, pi, qi
    for i in range(n):
        pi = pv[i]; qi = qv[i]
        if pi > get_EPSILON_NORM():
            if qi < get_EPSILON_NORM(): return np.inf
            s += pi * log2(pi / qi)
    return s


def linear_entropy(np.ndarray[np.complex128_t, ndim=1] psi):
    """Linear entropy S_L = 1 - sum |psi_i|^4."""
    return 1.0 - signal_purity(psi)


# Wave evolution

def schrodinger_spectral(np.ndarray[np.complex128_t, ndim=1] psi,
                         np.ndarray[f64, ndim=1] evals,
                         np.ndarray[f64, ndim=2] evecs,
                         double t):
    """
    Schrodinger evolution: psi(t) = exp(-i L t) psi(0) via spectral.

    Since L is real symmetric with eigenpairs (lambda_j, v_j):
      psi(t) = sum_j exp(-i lambda_j t) * <v_j|psi(0)> * v_j

    Parameters
    ----------
    psi : complex128[n]
    evals : f64[k]
    evecs : f64[n, k]
    t : float

    Returns
    -------
    psi_t : complex128[n]
    """
    cdef Py_ssize_t n = psi.shape[0], nk = evals.shape[0], j, i
    cdef np.ndarray[np.complex128_t, ndim=1] out = np.zeros(n, dtype=np.complex128)
    cdef c128[::1] ov = out, pv = psi
    cdef f64[::1] ev = evals
    cdef f64[:, ::1] vv = evecs
    cdef c128 coeff
    cdef f64 c_val, s_val, vij

    for j in range(nk):
        coeff = 0.0
        for i in range(n):
            coeff = coeff + vv[i, j] * pv[i]
        c_val = cos(ev[j] * t)
        s_val = sin(ev[j] * t)
        for i in range(n):
            vij = vv[i, j]
            ov[i] = ov[i] + (c_val - s_val * 1j) * coeff * vij

    return out


def schrodinger_spectral_trajectory(np.ndarray[np.complex128_t, ndim=1] psi,
                                     np.ndarray[f64, ndim=1] evals,
                                     np.ndarray[f64, ndim=2] evecs,
                                     np.ndarray[f64, ndim=1] times):
    """Evolve through multiple times. Returns complex128[nT, n]."""
    cdef Py_ssize_t n = psi.shape[0], nT = times.shape[0], step
    cdef np.ndarray[np.complex128_t, ndim=2] traj = np.empty((nT, n), dtype=np.complex128)
    for step in range(nT):
        traj[step] = schrodinger_spectral(psi, evals, evecs, times[step])
    return traj


def rk4_step_complex(np.ndarray[np.complex128_t, ndim=1] psi,
                     np.ndarray[f64, ndim=2] L,
                     double dt):
    """
    Single RK4 step for Schrodinger: d|psi>/dt = -i L |psi>.
    Dense Laplacian, complex state. Uses malloc for temp arrays.
    """
    cdef Py_ssize_t n = psi.shape[0], i, j
    cdef c128[::1] pv = psi
    cdef f64[:, ::1] Lv = L
    cdef c128 acc

    cdef c128* k1 = <c128*>malloc(n * sizeof(c128))
    cdef c128* k2 = <c128*>malloc(n * sizeof(c128))
    cdef c128* k3 = <c128*>malloc(n * sizeof(c128))
    cdef c128* k4 = <c128*>malloc(n * sizeof(c128))
    cdef c128* tmp = <c128*>malloc(n * sizeof(c128))
    if k1 == NULL or k2 == NULL or k3 == NULL or k4 == NULL or tmp == NULL:
        if k1 != NULL: free(k1)
        if k2 != NULL: free(k2)
        if k3 != NULL: free(k3)
        if k4 != NULL: free(k4)
        if tmp != NULL: free(tmp)
        raise MemoryError()

    cdef f64 dt2 = dt * 0.5, dt6 = dt / 6.0

    cdef np.ndarray[np.complex128_t, ndim=1] out = np.empty(n, dtype=np.complex128)
    cdef c128[::1] ov = out

    try:
        for i in range(n):
            acc = 0.0
            for j in range(n): acc = acc + Lv[i, j] * pv[j]
            k1[i] = -1j * acc

        for i in range(n): tmp[i] = pv[i] + dt2 * k1[i]

        for i in range(n):
            acc = 0.0
            for j in range(n): acc = acc + Lv[i, j] * tmp[j]
            k2[i] = -1j * acc

        for i in range(n): tmp[i] = pv[i] + dt2 * k2[i]

        for i in range(n):
            acc = 0.0
            for j in range(n): acc = acc + Lv[i, j] * tmp[j]
            k3[i] = -1j * acc

        for i in range(n): tmp[i] = pv[i] + dt * k3[i]

        for i in range(n):
            acc = 0.0
            for j in range(n): acc = acc + Lv[i, j] * tmp[j]
            k4[i] = -1j * acc

        for i in range(n):
            ov[i] = pv[i] + dt6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
    finally:
        free(k1)
        free(k2)
        free(k3)
        free(k4)
        free(tmp)

    return out


def rk4_integrate_complex(np.ndarray[np.complex128_t, ndim=1] psi0,
                          np.ndarray[f64, ndim=2] L,
                          double t0, double t1,
                          Py_ssize_t n_steps):
    """Integrate Schrodinger from t0 to t1 with n_steps of RK4."""
    cdef f64 dt = (t1 - t0) / <f64>n_steps
    cdef Py_ssize_t n = psi0.shape[0], step
    cdef np.ndarray[np.complex128_t, ndim=2] traj = np.empty((n_steps + 1, n), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] cur = psi0.copy()
    traj[0] = cur
    for step in range(n_steps):
        cur = rk4_step_complex(cur, L, dt)
        traj[step + 1] = cur
    return cur, traj


def field_schrodinger_evolve(np.ndarray[np.complex128_t, ndim=1] psi_E,
                             np.ndarray[np.complex128_t, ndim=1] psi_F,
                             np.ndarray[f64, ndim=1] evals_RL1,
                             np.ndarray[f64, ndim=2] evecs_RL1,
                             evals_L2, evecs_L2,
                             double t,
                             B1=None):
    """Schrodinger evolution on the rex field (E, F).

    In the rex framework, vertices are boundaries of edges, not
    independent degrees of freedom. The field state lives on edges
    and faces:
        psi_E(t) = exp(-i RL_1 t) psi_E(0)
        psi_F(t) = exp(-i L_2 t) psi_F(0)

    Vertex observables are derived from the edge state via the
    boundary map: psi_V(t) = B_1 psi_E(t). This is returned as a
    convenience when B_1 is provided.

    The edge tier uses the Relational Laplacian RL_1 = L_1 + alpha_G * L_O.
    Cross-dimensional coupling (via B_1, B_2) is not included in the
    evolution itself; use the field operator in _field.pyx for fully
    coupled dynamics.

    Parameters
    ----------
    psi_E : complex128[nE]
    psi_F : complex128[nF]
    evals_RL1, evecs_RL1 : Relational Laplacian spectrum
    evals_L2, evecs_L2 : face Laplacian spectrum (or None)
    t : float
    B1 : boundary operator (nV, nE) or None
        If provided, vertex observables psi_V = B_1 psi_E are returned.

    Returns
    -------
    psi_E_t : complex128[nE]
    psi_F_t : complex128[nF]
    psi_V_t : complex128[nV] or None
        Vertex projection via B_1. None if B1 not provided.
    """
    psi_E_t = schrodinger_spectral(psi_E, evals_RL1, evecs_RL1, t)

    if evals_L2 is not None and evecs_L2 is not None and psi_F.shape[0] > 0:
        psi_F_t = schrodinger_spectral(psi_F, evals_L2, evecs_L2, t)
    else:
        psi_F_t = psi_F.copy()

    psi_V_t = None
    if B1 is not None:
        psi_V_t = B1.dot(psi_E_t.real) + 1j * B1.dot(psi_E_t.imag)

    return psi_E_t, psi_F_t, psi_V_t


def field_schrodinger_trajectory(psi_E, psi_F,
                                 evals_RL1, evecs_RL1,
                                 evals_L2, evecs_L2,
                                 np.ndarray[f64, ndim=1] times,
                                 B1=None):
    """Evolve rex field through multiple timepoints.

    Vertex observables are derived at each timepoint via B_1 psi_E(t)
    when B1 is provided.

    Returns
    -------
    traj_E : complex128[nT, nE]
    traj_F : complex128[nT, nF]
    traj_V : complex128[nT, nV] or None
    """
    cdef Py_ssize_t nT = times.shape[0], step
    cdef Py_ssize_t nE = psi_E.shape[0]
    cdef Py_ssize_t nF = psi_F.shape[0]

    traj_E = np.empty((nT, nE), dtype=np.complex128)
    traj_F = np.empty((nT, nF), dtype=np.complex128)
    traj_V = None

    if B1 is not None:
        nV = B1.shape[0] if hasattr(B1, 'shape') else B1.nrow
        traj_V = np.empty((nT, nV), dtype=np.complex128)

    for step in range(nT):
        et, ft, vt = field_schrodinger_evolve(
            psi_E, psi_F,
            evals_RL1, evecs_RL1,
            evals_L2, evecs_L2,
            times[step], B1=B1)
        traj_E[step] = et
        traj_F[step] = ft
        if traj_V is not None and vt is not None:
            traj_V[step] = vt

    return traj_E, traj_F, traj_V


def trotter_step(np.ndarray[np.complex128_t, ndim=1] psi,
                 np.ndarray[f64, ndim=1] diag,
                 np.ndarray[f64, ndim=2] L_off,
                 double dt):
    """
    Trotter-Suzuki step for split operator L = L_diag + L_off.

    exp(-iLdt) ~ exp(-iL_diag dt/2) exp(-iL_off dt) exp(-iL_diag dt/2)

    Useful for Laplacians where diagonal = degree, off-diagonal = coupling.
    """
    cdef c128[::1] pv = psi
    cdef f64[::1] dv = diag
    cdef Py_ssize_t n = psi.shape[0], i
    cdef f64 dt2 = dt * 0.5, c, s

    cdef np.ndarray[np.complex128_t, ndim=1] tmp = np.empty(n, dtype=np.complex128)
    cdef c128[::1] tv = tmp
    for i in range(n):
        c = cos(dv[i] * dt2); s = sin(dv[i] * dt2)
        tv[i] = (c - s * 1j) * pv[i]

    tmp = rk4_step_complex(tmp, L_off, dt)
    tv = tmp

    cdef np.ndarray[np.complex128_t, ndim=1] out = np.empty(n, dtype=np.complex128)
    cdef c128[::1] ov = out
    for i in range(n):
        c = cos(dv[i] * dt2); s = sin(dv[i] * dt2)
        ov[i] = (c - s * 1j) * tv[i]

    return out


# Interference and superposition

def superpose(list psi_list, np.ndarray[np.complex128_t, ndim=1] weights):
    """
    Superpose multiple wave states: Psi = sum_k w_k * psi_k.
    Returns unnormalized superposition.
    """
    cdef Py_ssize_t n_states = len(psi_list), n, k, i
    cdef c128[::1] wv = weights

    n = psi_list[0].shape[0]
    cdef np.ndarray[np.complex128_t, ndim=1] out = np.zeros(n, dtype=np.complex128)
    cdef c128[::1] ov = out

    cdef c128[::1] psi_k_v
    for k in range(n_states):
        psi_k_v = psi_list[k]
        for i in range(n):
            ov[i] = ov[i] + wv[k] * psi_k_v[i]

    return out


def interference_pattern(np.ndarray[np.complex128_t, ndim=1] psi1,
                         np.ndarray[np.complex128_t, ndim=1] psi2,
                         c128 w1, c128 w2):
    """
    Interference probabilities: P(i) = |w1*psi1_i + w2*psi2_i|^2.

    The interference term is 2 Re(w1* w2 * conj(psi1_i) * psi2_i).
    """
    cdef c128[::1] a = psi1, b = psi2
    cdef Py_ssize_t n = psi1.shape[0], i
    cdef np.ndarray[f64, ndim=1] p = np.empty(n, dtype=np.float64)
    cdef f64[::1] pv = p
    cdef c128 s
    cdef f64 sr, si
    for i in range(n):
        s = w1 * a[i] + w2 * b[i]
        sr = s.real; si = s.imag
        pv[i] = sr * sr + si * si
    return p


def classical_mixture(np.ndarray[np.complex128_t, ndim=1] psi1,
                      np.ndarray[np.complex128_t, ndim=1] psi2,
                      f64 w1, f64 w2):
    """Classical mixture: P(i) = w1*|psi1_i|^2 + w2*|psi2_i|^2 (no interference)."""
    cdef c128[::1] a = psi1, b = psi2
    cdef Py_ssize_t n = psi1.shape[0], i
    cdef np.ndarray[f64, ndim=1] p = np.empty(n, dtype=np.float64)
    cdef f64[::1] pv = p
    cdef f64 ar, ai, br, bi
    for i in range(n):
        ar = a[i].real; ai = a[i].imag
        br = b[i].real; bi = b[i].imag
        pv[i] = w1 * (ar*ar + ai*ai) + w2 * (br*br + bi*bi)
    return p


def interference_term(np.ndarray[np.complex128_t, ndim=1] psi1,
                      np.ndarray[np.complex128_t, ndim=1] psi2):
    """
    Pure interference: I(i) = 2 Re(conj(psi1_i) * psi2_i).
    P_quantum = P_classical + I.
    """
    cdef c128[::1] a = psi1, b = psi2
    cdef Py_ssize_t n = psi1.shape[0], i
    cdef np.ndarray[f64, ndim=1] iterm = np.empty(n, dtype=np.float64)
    cdef f64[::1] iv = iterm
    cdef c128 prod
    for i in range(n):
        prod = a[i].conjugate() * b[i]
        iv[i] = 2.0 * prod.real
    return iterm


def visibility(np.ndarray[np.complex128_t, ndim=1] psi1,
               np.ndarray[np.complex128_t, ndim=1] psi2):
    """
    Fringe visibility V = 2|<psi1|psi2>| / (1 + |<psi1|psi2>|^2).
    V=1 for identical states (full interference), V=0 for orthogonal (none).
    """
    cdef c128 ov = inner_product(psi1, psi2)
    cdef f64 abs_ov = sqrt(ov.real * ov.real + ov.imag * ov.imag)
    cdef f64 fid = abs_ov * abs_ov
    if fid > 1.0: fid = 1.0
    return 2.0 * abs_ov / (1.0 + fid)


def coherence_measure(np.ndarray[np.complex128_t, ndim=2] rho):
    """
    l1-norm coherence: C = sum_{i!=j} |rho_{ij}|.
    Measures total off-diagonal weight (quantum coherence).
    """
    cdef c128[:, ::1] R = rho
    cdef Py_ssize_t n = rho.shape[0], i, j
    cdef f64 s = 0.0, rr, ri
    for i in range(n):
        for j in range(n):
            if i != j:
                rr = R[i, j].real; ri = R[i, j].imag
                s += sqrt(rr * rr + ri * ri)
    return s


# Cross-dimensional entanglement

def tensor_product(np.ndarray[np.complex128_t, ndim=1] psi_A,
                   np.ndarray[np.complex128_t, ndim=1] psi_B):
    """
    Tensor product |psi_A> (x) |psi_B>.

    """
    cdef c128[::1] a = psi_A, b = psi_B
    cdef Py_ssize_t dA = psi_A.shape[0], dB = psi_B.shape[0], i, j
    cdef np.ndarray[np.complex128_t, ndim=1] out = np.empty(dA * dB, dtype=np.complex128)
    cdef c128[::1] ov = out
    for i in range(dA):
        for j in range(dB):
            ov[i * dB + j] = a[i] * b[j]
    return out


def partial_trace_A(np.ndarray[np.complex128_t, ndim=2] rho,
                    Py_ssize_t dim_A, Py_ssize_t dim_B):
    """
    Trace out subsystem A: rho_B = Tr_A(rho).

    """
    cdef c128[:, ::1] R = rho
    cdef np.ndarray[np.complex128_t, ndim=2] out = np.zeros((dim_B, dim_B), dtype=np.complex128)
    cdef c128[:, ::1] O = out
    cdef Py_ssize_t i, j, k
    cdef c128 acc
    for j in range(dim_B):
        for k in range(dim_B):
            acc = 0.0
            for i in range(dim_A):
                acc = acc + R[i * dim_B + j, i * dim_B + k]
            O[j, k] = acc
    return out


def partial_trace_B(np.ndarray[np.complex128_t, ndim=2] rho,
                    Py_ssize_t dim_A, Py_ssize_t dim_B):
    """
    Trace out subsystem B: rho_A = Tr_B(rho).

    """
    cdef c128[:, ::1] R = rho
    cdef np.ndarray[np.complex128_t, ndim=2] out = np.zeros((dim_A, dim_A), dtype=np.complex128)
    cdef c128[:, ::1] O = out
    cdef Py_ssize_t i, j, k
    cdef c128 acc
    for i in range(dim_A):
        for j in range(dim_A):
            acc = 0.0
            for k in range(dim_B):
                acc = acc + R[i * dim_B + k, j * dim_B + k]
            O[i, j] = acc
    return out


def partial_transpose_A(np.ndarray[np.complex128_t, ndim=2] rho,
                        Py_ssize_t dim_A, Py_ssize_t dim_B):
    """
    Partial transpose over A: rho^{T_A}_{(i,j),(k,l)} = rho_{(k,j),(i,l)}.
    Negative eigenvalues of rho^{T_A} certify entanglement (PPT criterion).
    """
    cdef c128[:, ::1] R = rho
    cdef np.ndarray[np.complex128_t, ndim=2] out = np.empty((dim_A * dim_B, dim_A * dim_B), dtype=np.complex128)
    cdef c128[:, ::1] O = out
    cdef Py_ssize_t i, j, k, l
    for i in range(dim_A):
        for j in range(dim_B):
            for k in range(dim_A):
                for l in range(dim_B):
                    O[i * dim_B + j, k * dim_B + l] = R[k * dim_B + j, i * dim_B + l]
    return out


def entanglement_entropy(np.ndarray[np.complex128_t, ndim=1] psi,
                         Py_ssize_t dim_A, Py_ssize_t dim_B):
    """
    Entanglement entropy via Schmidt decomposition.
    S = -sum_i lambda_i^2 log2(lambda_i^2) where lambda_i are Schmidt values.

    """
    cdef np.ndarray[np.complex128_t, ndim=2] M = psi.reshape(dim_A, dim_B)
    # Use numpy SVD for complex inputs; _linalg.svd only handles real f64
    if np.any(np.imag(M) != 0):
        _U, s_vals, _Vt = np.linalg.svd(np.asarray(M), full_matrices=False)
    else:
        from rexgraph.core._linalg import svd as _lp_svd
        _U, s_vals, _Vt = _lp_svd(np.asarray(M, dtype=np.float64))

    cdef np.ndarray[f64, ndim=1] sv = np.asarray(s_vals, dtype=np.float64)
    cdef f64[::1] svv = sv
    cdef Py_ssize_t r = sv.shape[0], i
    cdef f64 p, ent = 0.0
    for i in range(r):
        p = svv[i] * svv[i]
        if p > get_EPSILON_NORM():
            ent -= p * log2(p)
    return ent


def schmidt_decomposition(np.ndarray[np.complex128_t, ndim=1] psi,
                          Py_ssize_t dim_A, Py_ssize_t dim_B):
    """
    Schmidt decomposition: |psi> = sum_i lambda_i |a_i> (x) |b_i>.

    Returns
    -------
    schmidt_values : f64[r]
    vectors_A : complex128[dim_A, r]
    vectors_B : complex128[r, dim_B]
    """
    M = psi.reshape(dim_A, dim_B)
    # Use numpy SVD for complex inputs; _linalg.svd only handles real f64
    if np.any(np.imag(M) != 0):
        U, s, Vh = np.linalg.svd(np.asarray(M), full_matrices=True)
    else:
        from rexgraph.core._linalg import svd as _lp_svd
        U, s, Vh = _lp_svd(np.asarray(M, dtype=np.float64))
    return np.asarray(s, dtype=np.float64), U, Vh


# Decoherence channels

def dephasing_channel(np.ndarray[np.complex128_t, ndim=2] rho,
                      double gamma, double dt):
    """
    Dephasing: off-diagonals decay as rho[i,j] *= exp(-gamma*|i-j|*dt).

    """
    cdef c128[:, ::1] R = rho
    cdef Py_ssize_t n = rho.shape[0], i, j
    cdef f64 decay
    for i in range(n):
        for j in range(n):
            if i != j:
                decay = exp(-gamma * fabs(<f64>(i - j)) * dt)
                R[i, j] = R[i, j] * decay


def amplitude_damping(np.ndarray[np.complex128_t, ndim=2] rho,
                      double gamma, double dt):
    """
    Amplitude damping on 2-level subspaces.
    Models irreversible decay toward ground state of each pair.
    """
    cdef c128[:, ::1] R = rho
    cdef Py_ssize_t n = rho.shape[0], i
    cdef f64 p = 1.0 - exp(-gamma * dt)
    cdef f64 sqrt_1_p = sqrt(1.0 - p)
    cdef f64 transfer
    for i in range(0, n - 1, 2):
        transfer = p * R[i + 1, i + 1].real
        R[i, i] = R[i, i] + transfer
        R[i + 1, i + 1] = R[i + 1, i + 1] * (1.0 - p)
        R[i, i + 1] = R[i, i + 1] * sqrt_1_p
        R[i + 1, i] = R[i + 1, i] * sqrt_1_p


def depolarizing_channel(np.ndarray[np.complex128_t, ndim=2] rho,
                         double p):
    """
    Depolarizing: rho -> (1-p)*rho + (p/d)*I.
    p=0 is pure quantum, p=1 is maximally mixed (complete decoherence).
    """
    cdef c128[:, ::1] R = rho
    cdef Py_ssize_t n = rho.shape[0], i, j
    cdef f64 coeff = 1.0 - p
    cdef f64 add_diag = p / <f64>n
    for i in range(n):
        for j in range(n):
            R[i, j] = R[i, j] * coeff
        R[i, i] = R[i, i] + add_diag


def lindblad_step(np.ndarray[np.complex128_t, ndim=2] rho,
                  np.ndarray[np.complex128_t, ndim=2] H,
                  list lindblad_ops,
                  double dt):
    """
    Single Euler step of the Lindblad master equation:

    drho/dt = -i[H, rho] + sum_k (Lk rho Lk^dag - 0.5 {Lk^dag Lk, rho})

    Here H is a Hermitian operator and Lk are Lindblad jump operators
    (not to be confused with Hodge Laplacians).
    """
    cdef Py_ssize_t n = rho.shape[0]

    cdef np.ndarray[np.complex128_t, ndim=2] Hrho = H @ rho
    cdef np.ndarray[np.complex128_t, ndim=2] rhoH = rho @ H
    cdef np.ndarray[np.complex128_t, ndim=2] drho = -1j * (Hrho - rhoH)

    cdef Py_ssize_t k
    for k in range(len(lindblad_ops)):
        Lk = lindblad_ops[k]
        Lk_dag = Lk.conj().T
        LdL = Lk_dag @ Lk
        drho = drho + Lk @ rho @ Lk_dag - 0.5 * (LdL @ rho + rho @ LdL)

    return rho + dt * drho


# Measurement and collapse

def born_sample(np.ndarray[np.complex128_t, ndim=1] psi,
                Py_ssize_t n_samples):
    """
    Sample cell indices from Born probability distribution.
    Returns int64[n_samples].
    """
    probs = born_probabilities(psi)
    return np.random.choice(psi.shape[0], size=n_samples, p=probs / probs.sum())


def projective_collapse(np.ndarray[np.complex128_t, ndim=1] psi,
                        np.ndarray[np.complex128_t, ndim=2] projector):
    """
    Projective measurement: |psi_out> = P|psi> / ||P|psi>||.
    Returns (collapsed_state, measurement_probability).
    """
    cdef c128[::1] pv = psi
    cdef c128[:, ::1] P = projector
    cdef Py_ssize_t n = psi.shape[0], i, j
    cdef np.ndarray[np.complex128_t, ndim=1] out = np.empty(n, dtype=np.complex128)
    cdef c128[::1] ov = out
    cdef c128 acc
    cdef f64 norm_sq = 0.0, inv_norm, sr, si

    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc = acc + P[i, j] * pv[j]
        ov[i] = acc
        sr = acc.real; si = acc.imag
        norm_sq += sr * sr + si * si

    if norm_sq > get_EPSILON_NORM():
        inv_norm = 1.0 / sqrt(norm_sq)
        for i in range(n):
            ov[i] = ov[i] * inv_norm

    return out, norm_sq


def measure_in_eigenbasis(np.ndarray[np.complex128_t, ndim=1] psi,
                          np.ndarray[f64, ndim=2] evecs):
    """
    Measure in the eigenbasis of a Laplacian.

    Returns
    -------
    outcome : int
    probability : float
    collapsed : complex128[n]
    """
    cdef Py_ssize_t n = psi.shape[0], k = evecs.shape[1], j, i
    cdef f64[:, ::1] vv = evecs
    cdef c128[::1] pv = psi

    cdef np.ndarray[f64, ndim=1] probs = np.empty(k, dtype=np.float64)
    cdef f64[::1] pbs = probs
    cdef c128 coeff
    for j in range(k):
        coeff = 0.0
        for i in range(n):
            coeff = coeff + vv[i, j] * pv[i]
        pbs[j] = coeff.real * coeff.real + coeff.imag * coeff.imag

    total = probs.sum()
    if total > get_EPSILON_NORM():
        probs_n = probs / total
    else:
        probs_n = np.ones(k, dtype=np.float64) / <f64>k
    outcome = np.random.choice(k, p=probs_n)

    cdef np.ndarray[np.complex128_t, ndim=1] collapsed = np.empty(n, dtype=np.complex128)
    cdef c128[::1] cv = collapsed
    for i in range(n):
        cv[i] = vv[i, outcome] + 0j

    return outcome, probs[outcome], collapsed


# Density matrix operations

def pure_to_density(np.ndarray[np.complex128_t, ndim=1] psi):
    """Construct density matrix rho = |psi><psi|."""
    cdef c128[::1] v = psi
    cdef Py_ssize_t n = psi.shape[0], i, j
    cdef np.ndarray[np.complex128_t, ndim=2] rho = np.empty((n, n), dtype=np.complex128)
    cdef c128[:, ::1] R = rho
    for i in range(n):
        for j in range(n):
            R[i, j] = v[i] * v[j].conjugate()
    return rho


def density_from_ensemble(list states,
                          np.ndarray[f64, ndim=1] weights):
    """
    Mixed state from ensemble: rho = sum_k w_k |psi_k><psi_k|.
    """
    cdef Py_ssize_t n_states = len(states), k
    cdef f64[::1] wv = weights
    cdef Py_ssize_t n = states[0].shape[0]
    cdef np.ndarray[np.complex128_t, ndim=2] rho = np.zeros((n, n), dtype=np.complex128)

    for k in range(n_states):
        rho = rho + wv[k] * pure_to_density(states[k])
    return rho


def density_trace(np.ndarray[np.complex128_t, ndim=2] rho):
    """Trace of density matrix."""
    cdef c128[:, ::1] R = rho
    cdef Py_ssize_t n = rho.shape[0], i
    cdef c128 acc = 0.0
    for i in range(n):
        acc = acc + R[i, i]
    return acc


def density_purity(np.ndarray[np.complex128_t, ndim=2] rho):
    """Purity Tr(rho^2). = 1 for pure, = 1/d for maximally mixed."""
    cdef c128[:, ::1] R = rho
    cdef Py_ssize_t n = rho.shape[0], i, j
    cdef c128 acc = 0.0
    for i in range(n):
        for j in range(n):
            acc = acc + R[i, j] * R[j, i]
    return acc.real


def von_neumann_entropy(np.ndarray[np.complex128_t, ndim=2] rho):
    """
    Von Neumann entropy S = -Tr(rho log2 rho).
    Computed via eigenvalues of the density matrix.
    """
    # Use numpy eigh for complex (Hermitian) inputs; _linalg.eigh only handles real f64
    if np.any(np.imag(rho) != 0):
        evals = np.linalg.eigvalsh(np.asarray(rho))
    else:
        from rexgraph.core._linalg import eigh as _lp_eigh
        evals = _lp_eigh(np.asarray(rho, dtype=np.float64))[0]
    cdef np.ndarray[f64, ndim=1] ev = np.real(evals).astype(np.float64)
    cdef f64[::1] evv = ev
    cdef Py_ssize_t n = ev.shape[0], i
    cdef f64 s = 0.0, p
    for i in range(n):
        p = evv[i]
        if p > get_EPSILON_NORM():
            s -= p * log2(p)
    return s


def fidelity_mixed(np.ndarray[np.complex128_t, ndim=2] rho,
                   np.ndarray[np.complex128_t, ndim=2] sigma):
    """
    Fidelity between mixed states: F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.
    """
    from scipy.linalg import sqrtm
    sqrt_rho = sqrtm(rho)
    inner = sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    return np.real(np.trace(inner)) ** 2


# ═══ RCF dynamics operators (from DRCT/I-SPY2 case studies) ═══

def amplitude_graded_projection(np.ndarray[f64, ndim=2] B1,
                                 np.ndarray[f64, ndim=2] B2,
                                 np.ndarray[f64, ndim=1] amplitudes,
                                 int nV, int nE, int nF):
    """Amplitude-based graded projection onto the chain complex.

    Unlike the standard delta-vertex projection (Def 4.5), this uses
    continuous vertex amplitudes with geometric-mean edge coupling:

        psi_0(v) = a_v
        psi_0(nV+e) = sqrt(a_i * a_j) * sign(B1[i,e])
        psi_0(nV+nE+f) = B2^T * psi_E

    Normalized to unit norm. Used in the I-SPY2 tumor trajectory analysis.

    Returns f64[nV + nE + nF] (the graded state vector).
    """
    cdef int NG = nV + nE + nF
    cdef np.ndarray[f64, ndim=1] psi = np.zeros(NG, dtype=np.float64)
    cdef f64[::1] pv = psi, av = amplitudes
    cdef f64[:, ::1] b1v = B1, b2v = B2
    cdef int v, e, f
    cdef f64 ai, aj, sgn, nm

    # Vertex component
    for v in range(nV):
        pv[v] = av[v]

    # Edge component: geometric mean coupling
    cdef int vi, vj
    for e in range(nE):
        vi = -1; vj = -1
        for v in range(nV):
            if fabs(b1v[v, e]) > 0.5:
                if vi < 0: vi = v
                else: vj = v; break
        if vi >= 0 and vj >= 0:
            ai = av[vi]; aj = av[vj]
            sgn = b1v[vi, e]
            pv[nV + e] = sqrt(fabs(ai * aj)) * (-1.0 if sgn < 0 else 1.0)

    # Face component: B2^T * psi_E
    for f in range(nF):
        for e in range(nE):
            pv[nV + nE + f] += b2v[e, f] * pv[nV + e]

    # Normalize
    nm = 0
    for v in range(NG):
        nm += pv[v] * pv[v]
    nm = sqrt(nm)
    if nm > 1e-10:
        for v in range(NG):
            pv[v] /= nm

    return psi


def lagrangian_step(np.ndarray[f64, ndim=1] amplitudes,
                     np.ndarray[f64, ndim=1] prev_amplitudes,
                     np.ndarray[i32, ndim=1] sources,
                     np.ndarray[i32, ndim=1] targets,
                     np.ndarray[f64, ndim=1] edge_weights,
                     int nV, int nE,
                     f64 dt, f64 H0=1.0):
    """One step of Lagrangian dynamics.

    T = (1/2) sum_v (da_v/dt)^2     (kinetic energy)
    V = -H0 * sum_e a_i * a_j       (coupling potential)
    L = T - V                        (Lagrangian)

    Returns dict with T, V, L.
    """
    cdef f64[::1] av = amplitudes, pv = prev_amplitudes, wv = edge_weights
    cdef i32[::1] sv = sources, tv = targets
    cdef int v, e
    cdef f64 da, dt_safe, T_val, V_val

    dt_safe = dt if dt > 0.001 else 0.001

    T_val = 0
    for v in range(nV):
        da = (av[v] - pv[v]) / dt_safe
        T_val += 0.5 * da * da

    V_val = 0
    for e in range(nE):
        V_val -= H0 * av[sv[e]] * av[tv[e]]

    return {'T': float(T_val), 'V': float(V_val), 'L': float(T_val - V_val)}


def harmonic_basis_extract(np.ndarray[f64, ndim=1] evals,
                            np.ndarray[f64, ndim=2] evecs,
                            int n, f64 tol=1e-10):
    """Extract harmonic eigenvectors (eigenvalue < tol) as a basis.

    Returns f64[n_harm, n] where each row is a harmonic eigenvector.
    """
    mask = evals < tol
    if not np.any(mask):
        return np.zeros((0, n), dtype=np.float64)
    return evecs[:, mask].T.copy()


def harmonic_project(np.ndarray[f64, ndim=1] signal,
                      np.ndarray[f64, ndim=2] basis):
    """Project signal onto harmonic basis.

    proj = sum_k <signal, phi_k> phi_k
    """
    if basis.shape[0] == 0:
        return np.zeros_like(signal)
    coeffs = basis @ signal
    return basis.T @ coeffs


def face_partition(np.ndarray[f64, ndim=2] B1,
                    np.ndarray[f64, ndim=2] B2,
                    np.ndarray[i32, ndim=1] sources,
                    np.ndarray[i32, ndim=1] targets,
                    int probe_vertex,
                    int nV, int nE, int nF):
    """Partition faces into probe-incident (psi) and scaffold.

    A face is probe-incident if any of its boundary edges touch probe_vertex.

    Returns dict with psi_faces (i32[]), scaffold_faces (i32[]).
    """
    cdef i32[::1] sv = sources, tv = targets
    psi_faces = []
    scaffold_faces = []

    for f in range(nF):
        touches = False
        for e in range(nE):
            if fabs(B2[e, f]) > 0.5:
                if sv[e] == probe_vertex or tv[e] == probe_vertex:
                    touches = True
                    break
        if touches:
            psi_faces.append(f)
        else:
            scaffold_faces.append(f)

    return {
        'psi_faces': np.array(psi_faces, dtype=np.int32),
        'scaffold_faces': np.array(scaffold_faces, dtype=np.int32),
    }


def action_integral(np.ndarray[f64, ndim=1] lagrangian_values,
                     np.ndarray[f64, ndim=1] dt_values):
    """Cumulative action integral: A = sum L_i * dt_i.

    Returns f64[nT] where A[t] = sum_{i<=t} L_i * dt_i.
    """
    cdef int nT = lagrangian_values.shape[0]
    cdef np.ndarray[f64, ndim=1] action = np.zeros(nT, dtype=np.float64)
    cdef f64[::1] av = action, lv = lagrangian_values, dv = dt_values
    cdef int t
    cdef f64 cumul = 0

    for t in range(nT):
        cumul += lv[t] * dv[t]
        av[t] = cumul

    return action
