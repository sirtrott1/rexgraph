# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._field - Cross-dimensional field dynamics on the rex chain complex.

Implements the coupled field operator and wave equation on the (E, F)
field state, where edges and faces are the independent degrees of freedom
and vertices are derived via f_V = B_1 f_E.

Field Operator
--------------
The field operator M couples edges and faces through the boundary map B_2:

    M = [[ RL_1,   -g * B_2 ],
         [-g * B_2^T,   L_2  ]]

where:
    RL_1 = L_1 + alpha_G * L_O   (Relational Laplacian on edges)
    L_2 = B_2^T B_2              (face Laplacian)
    g = coupling strength between edge and face tiers
    B_2 = edge-face boundary operator (nE x nF)

M is PSD when g is small enough. The default auto-coupling
g = 1 / max(||B_2||_F, 1) stays in the PSD regime for typical complexes.

Vertex observables are always derived from the edge component:
    f_V(t) = B_1 f_E(t)

This is the correct field operator for the rex framework, not the
(V, E, F) block matrix. The coupled_derivative in _transition.pyx
provides a backward-compatible V+E+F ODE for historical reasons, but
the mathematically correct operator lives on (E, F) only.

Wave Equation
-------------
The coupled wave equation on (E, F):
    d^2 F/dt^2 = -M F

Solutions are superpositions of normal modes with frequencies
omega_k = sqrt(lambda_k) where lambda_k are eigenvalues of M.
Total energy (KE + PE) is exactly conserved.

Heat Equation
-------------
The coupled diffusion on (E, F):
    dF/dt = -M F

Solutions decay exponentially: F(t) = exp(-M t) F(0).

All functions are stateless: arrays in, arrays out.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport fabs, sqrt, cos, sin, exp
from libc.string cimport memcpy

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,
    get_EPSILON_NORM,
)

np.import_array()


# Section 1: Field operator construction


def _safe_dot(A, x):
    """Matrix-vector product, dense or sparse."""
    return np.asarray(A.dot(x), dtype=np.float64)


def build_field_operator(object RL1,
                         object L2,
                         object B2,
                         double g=-1.0):
    """Build the coupled field operator on (E, F) space.

    M = [[ RL_1,      -g * B_2     ],
         [-g * B_2^T,     L_2      ]]

    Vertices are not part of the field operator. Vertex observables
    are derived from the edge block via f_V = B_1 f_E.

    Parameters
    ----------
    RL1 : (nE, nE) - Relational Laplacian on edges
    L2 : (nF, nF) - face Laplacian
    B2 : (nE, nF) - edge-face boundary operator
    g : float - coupling strength. If negative, auto-computed as
        1 / max(||B_2||_F, 1) to stay in PSD regime.

    Returns
    -------
    M : f64[nE+nF, nE+nF] - field operator (dense)
    g_used : float - coupling strength actually used
    is_psd : bool - True if minimum eigenvalue >= -epsilon
    """
    cdef Py_ssize_t nE, nF, n, i, j

    cdef np.ndarray[f64, ndim=2] RL1_d = np.asarray(RL1, dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] B2_d = np.asarray(B2, dtype=np.float64)
    nE = RL1_d.shape[0]
    nF = B2_d.shape[1]
    n = nE + nF

    # Auto-compute coupling if not specified
    cdef double g_used = g
    cdef double b2_frob = 0.0
    cdef f64[:, ::1] b2v = B2_d
    if g_used < 0.0:
        for i in range(nE):
            for j in range(nF):
                b2_frob += b2v[i, j] * b2v[i, j]
        b2_frob = sqrt(b2_frob)
        g_used = 1.0 / (b2_frob if b2_frob > 1.0 else 1.0)

    # Assemble dense block matrix
    cdef np.ndarray[f64, ndim=2] M = np.zeros((n, n), dtype=np.float64)
    cdef f64[:, ::1] mv = M
    cdef f64[:, ::1] rl1v = RL1_d

    # Top-left: RL_1 (nE x nE)
    for i in range(nE):
        memcpy(&mv[i, 0], &rl1v[i, 0], nE * sizeof(f64))

    # Bottom-right: L_2 (nF x nF)
    cdef np.ndarray[f64, ndim=2] L2_d = np.asarray(L2, dtype=np.float64)
    cdef f64[:, ::1] l2v = L2_d
    for i in range(nF):
        memcpy(&mv[nE + i, nE], &l2v[i, 0], nF * sizeof(f64))

    # Off-diagonal: -g * B_2 and -g * B_2^T
    cdef f64 neg_g = -g_used
    for i in range(nE):
        for j in range(nF):
            mv[i, nE + j] = neg_g * b2v[i, j]
            mv[nE + j, i] = neg_g * b2v[i, j]

    # PSD check via minimum eigenvalue
    cdef np.ndarray[f64, ndim=1] evals_check
    from rexgraph.core._linalg import eigh as _lp_eigh
    evals_check = _lp_eigh(np.asarray(M, dtype=np.float64))[0]
    cdef bint is_psd = evals_check[0] >= -get_EPSILON_NORM()

    return M, g_used, is_psd


def field_operator_matvec(np.ndarray[f64, ndim=1] F,
                          object RL1,
                          object L2,
                          object B2,
                          double g,
                          Py_ssize_t nE,
                          Py_ssize_t nF):
    """Apply the field operator M @ F without building the dense matrix.

    For large complexes where the dense (nE+nF)^2 matrix is too expensive.
    Uses operator-vector products directly.

    Parameters
    ----------
    F : f64[nE + nF] - field state vector
    RL1, L2, B2 : operators (dense or sparse)
    g : coupling strength
    nE, nF : dimensions

    Returns
    -------
    MF : f64[nE + nF]
    """
    cdef np.ndarray[f64, ndim=1] f_E = F[:nE]
    cdef np.ndarray[f64, ndim=1] f_F = F[nE:]

    # Edge block: RL_1 f_E - g B_2 f_F
    cdef np.ndarray[f64, ndim=1] MF_E = _safe_dot(RL1, f_E)
    if nF > 0:
        MF_E = MF_E - g * _safe_dot(B2, f_F)

    # Face block: -g B_2^T f_E + L_2 f_F
    cdef np.ndarray[f64, ndim=1] MF_F
    if nF > 0:
        MF_F = _safe_dot(L2, f_F)
        MF_F = MF_F - g * np.asarray(B2.T.dot(f_E), dtype=np.float64)
    else:
        MF_F = np.zeros(0, dtype=np.float64)

    return np.concatenate([MF_E, MF_F])


# Section 2: Eigendecomposition and frequencies


def field_eigendecomposition(np.ndarray[f64, ndim=2] M):
    """Eigendecomposition of the field operator.

    Parameters
    ----------
    M : f64[n, n] - field operator (symmetric, PSD)

    Returns
    -------
    evals : f64[n] - eigenvalues (sorted ascending)
    evecs : f64[n, n] - eigenvectors as columns
    freqs : f64[n] - frequencies omega_k = sqrt(max(lambda_k, 0))
    """
    cdef np.ndarray[f64, ndim=1] evals
    cdef np.ndarray[f64, ndim=2] evecs

    from rexgraph.core._linalg import eigh as _lp_eigh
    evals, evecs = _lp_eigh(np.asarray(M, dtype=np.float64))

    # Clean near-zero eigenvalues
    cdef f64[::1] ev = evals
    cdef Py_ssize_t n = evals.shape[0], k
    cdef double eps = get_EPSILON_NORM()
    for k in range(n):
        if fabs(ev[k]) < eps:
            ev[k] = 0.0

    # Frequencies
    cdef np.ndarray[f64, ndim=1] freqs = np.empty(n, dtype=np.float64)
    cdef f64[::1] fv = freqs
    for k in range(n):
        fv[k] = sqrt(ev[k]) if ev[k] > 0.0 else 0.0

    return evals, evecs, freqs


def field_spectral_coefficients(np.ndarray[f64, ndim=1] F,
                                np.ndarray[f64, ndim=2] evecs):
    """Project field state onto eigenbasis of M.

    c_k = evecs[:, k]^T @ F

    Parameters
    ----------
    F : f64[n] - field state
    evecs : f64[n, n] - eigenvectors as columns

    Returns
    -------
    coeffs : f64[n]
    """
    cdef Py_ssize_t n = F.shape[0], k, j
    cdef np.ndarray[f64, ndim=1] coeffs = np.empty(n, dtype=np.float64)
    cdef f64[::1] cv = coeffs, fv = F
    cdef f64[:, ::1] ev = evecs
    cdef f64 s

    for k in range(n):
        s = 0.0
        for j in range(n):
            s += ev[j, k] * fv[j]
        cv[k] = s

    return coeffs


# Section 3: Wave evolution


def wave_evolve(np.ndarray[f64, ndim=1] F0,
                np.ndarray[f64, ndim=1] evals,
                np.ndarray[f64, ndim=2] evecs,
                np.ndarray[f64, ndim=1] freqs,
                double t):
    """Exact spectral wave evolution on the field.

    The wave equation d^2F/dt^2 = -M F with F(0) = F0, dF/dt(0) = 0
    has solution:
        F(t) = sum_k c_k cos(omega_k t) v_k
        dF/dt(t) = -sum_k c_k omega_k sin(omega_k t) v_k

    Parameters
    ----------
    F0 : f64[n] - initial field state (E+F packed)
    evals : f64[n] - eigenvalues of M
    evecs : f64[n, n] - eigenvectors
    freqs : f64[n] - frequencies sqrt(evals)
    t : float - time

    Returns
    -------
    Ft : f64[n] - field state at time t
    dFdt : f64[n] - field velocity at time t
    """
    cdef Py_ssize_t n = F0.shape[0], k, j
    cdef f64[::1] fv = F0
    cdef f64[:, ::1] ev = evecs
    cdef f64[::1] wv = freqs

    # Compute spectral coefficients c_k = v_k^T F0
    cdef f64 *c = <f64 *>malloc(n * sizeof(f64))
    if c == NULL:
        raise MemoryError("wave_evolve: malloc failed")

    cdef f64 s
    for k in range(n):
        s = 0.0
        for j in range(n):
            s += ev[j, k] * fv[j]
        c[k] = s

    # Reconstruct F(t) and dF/dt(t)
    cdef np.ndarray[f64, ndim=1] Ft = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] dFdt = np.zeros(n, dtype=np.float64)
    cdef f64[::1] ftv = Ft, dftv = dFdt
    cdef f64 cos_wt, sin_wt, ck_cos, ck_sin

    for k in range(n):
        if fabs(c[k]) < 1e-15:
            continue
        cos_wt = cos(wv[k] * t)
        sin_wt = sin(wv[k] * t)
        ck_cos = c[k] * cos_wt
        ck_sin = -c[k] * wv[k] * sin_wt
        for j in range(n):
            ftv[j] += ck_cos * ev[j, k]
            dftv[j] += ck_sin * ev[j, k]

    free(c)
    return Ft, dFdt


def wave_evolve_trajectory(np.ndarray[f64, ndim=1] F0,
                           np.ndarray[f64, ndim=1] evals,
                           np.ndarray[f64, ndim=2] evecs,
                           np.ndarray[f64, ndim=1] freqs,
                           np.ndarray[f64, ndim=1] times):
    """Evolve field under wave equation at multiple timepoints.

    Reuses spectral coefficients across all times for O(n * T) work
    after the initial O(n^2) coefficient computation.

    Returns
    -------
    traj : f64[T, n] - field state trajectory
    vel : f64[T, n] - velocity trajectory
    """
    cdef Py_ssize_t n = F0.shape[0], T = times.shape[0]
    cdef Py_ssize_t k, j, step
    cdef f64[::1] fv = F0
    cdef f64[:, ::1] ev = evecs
    cdef f64[::1] wv = freqs, tv = times

    # Compute coefficients once
    cdef f64 *c = <f64 *>malloc(n * sizeof(f64))
    if c == NULL:
        raise MemoryError("wave_evolve_trajectory: malloc failed")

    cdef f64 s
    for k in range(n):
        s = 0.0
        for j in range(n):
            s += ev[j, k] * fv[j]
        c[k] = s

    cdef np.ndarray[f64, ndim=2] traj = np.zeros((T, n), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] vel = np.zeros((T, n), dtype=np.float64)
    cdef f64[:, ::1] trajv = traj, velv = vel
    cdef f64 t, cos_wt, sin_wt, ck_cos, ck_sin

    for step in range(T):
        t = tv[step]
        for k in range(n):
            if fabs(c[k]) < 1e-15:
                continue
            cos_wt = cos(wv[k] * t)
            sin_wt = sin(wv[k] * t)
            ck_cos = c[k] * cos_wt
            ck_sin = -c[k] * wv[k] * sin_wt
            for j in range(n):
                trajv[step, j] += ck_cos * ev[j, k]
                velv[step, j] += ck_sin * ev[j, k]

    free(c)
    return traj, vel


# Section 4: Energy and conservation


def wave_energy(np.ndarray[f64, ndim=1] F,
                np.ndarray[f64, ndim=1] dFdt,
                np.ndarray[f64, ndim=2] M):
    """Compute wave energy components.

    KE = 0.5 ||dF/dt||^2     (kinetic energy of field motion)
    PE = 0.5 F^T M F         (potential energy from field operator)
    Total = KE + PE           (conserved under wave evolution)

    Parameters
    ----------
    F : f64[n] - field state (E+F packed)
    dFdt : f64[n] - field velocity
    M : f64[n, n] - field operator

    Returns
    -------
    KE, PE, total : float
    """
    cdef f64[::1] fv = F, dv = dFdt
    cdef Py_ssize_t n = F.shape[0], j

    # KE = 0.5 ||dFdt||^2
    cdef f64 ke = 0.0
    for j in range(n):
        ke += dv[j] * dv[j]
    ke *= 0.5

    # PE = 0.5 F^T M F
    cdef np.ndarray[f64, ndim=1] MF = np.asarray(M.dot(F), dtype=np.float64)
    cdef f64[::1] mfv = MF
    cdef f64 pe = 0.0
    for j in range(n):
        pe += fv[j] * mfv[j]
    pe *= 0.5

    return ke, pe, ke + pe


def field_energy_kin_pot(np.ndarray[f64, ndim=1] F,
                        object L1,
                        object LO,
                        Py_ssize_t nE):
    """Compute E_kin and E_pot from the edge component of a field state.

    Extracts f_E from the packed field vector F = [f_E, f_F] and computes
    the topological-geometric energy decomposition on edges only.

    E_kin = <f_E | L_1 | f_E>   (topological energy from Hodge Laplacian)
    E_pot = <f_E | L_O | f_E>   (geometric energy from overlap Laplacian)

    This is the field-state analog of _state.energy_kin_pot. It takes L1
    and LO (not RL1), matching the _state interface exactly.

    Parameters
    ----------
    F : f64[nE + nF] - packed field state
    L1 : (nE, nE) - Hodge Laplacian on edges
    LO : (nE, nE) - overlap Laplacian on edges
    nE : int - number of edges (to slice F)

    Returns
    -------
    E_kin : float - <f_E | L_1 | f_E>
    E_pot : float - <f_E | L_O | f_E>
    ratio : float - E_kin / E_pot (inf if E_pot ~ 0)
    """
    cdef np.ndarray[f64, ndim=1] f_E = F[:nE].copy()
    cdef np.ndarray[f64, ndim=1] L1f = _safe_dot(L1, f_E)
    cdef np.ndarray[f64, ndim=1] LOf = _safe_dot(LO, f_E)

    cdef f64[::1] fv = f_E, l1v = L1f, ov = LOf
    cdef Py_ssize_t j
    cdef f64 ek = 0.0, ep = 0.0

    for j in range(nE):
        ek += fv[j] * l1v[j]
        ep += fv[j] * ov[j]

    cdef f64 ratio
    if ep > 1e-15:
        ratio = ek / ep
    elif ek > 1e-15:
        ratio = 1e15
    else:
        ratio = 1.0

    return ek, ep, ratio


def wave_dimensional_energy(np.ndarray[f64, ndim=1] F,
                            np.ndarray[f64, ndim=1] dFdt,
                            Py_ssize_t nE,
                            Py_ssize_t nF):
    """Split field energy into edge and face components.

    KE_E = 0.5 ||dF_E/dt||^2
    KE_F = 0.5 ||dF_F/dt||^2
    PE uses quadratic form on each block (approximate for off-diagonal).
    ||F_E||^2 and ||F_F||^2 give signal magnitude per dimension.

    Parameters
    ----------
    F : f64[nE + nF] - field state
    dFdt : f64[nE + nF] - velocity
    nE, nF : dimensions

    Returns
    -------
    dict with keys: norm_E, norm_F, ke_E, ke_F
    """
    cdef f64[::1] fv = F, dv = dFdt
    cdef Py_ssize_t j
    cdef f64 ne = 0.0, nf = 0.0, ke_e = 0.0, ke_f = 0.0

    for j in range(nE):
        ne += fv[j] * fv[j]
        ke_e += dv[j] * dv[j]

    for j in range(nE, nE + nF):
        nf += fv[j] * fv[j]
        ke_f += dv[j] * dv[j]

    return {
        "norm_E": sqrt(ne),
        "norm_F": sqrt(nf),
        "ke_E": 0.5 * ke_e,
        "ke_F": 0.5 * ke_f,
    }


# Section 5: Mode classification and resonance


def classify_modes(np.ndarray[f64, ndim=1] evals,
                   np.ndarray[f64, ndim=2] evecs,
                   Py_ssize_t nE,
                   Py_ssize_t nF,
                   double threshold=0.1):
    """Classify eigenmodes by dimensional weight.

    For each eigenmode v_k of M, compute the fraction of squared weight
    in the edge block vs face block:
        w_E(k) = ||v_k[:nE]||^2 / ||v_k||^2
        w_F(k) = ||v_k[nE:]||^2 / ||v_k||^2

    A mode is edge-dominated if w_E > 1 - threshold,
    face-dominated if w_F > 1 - threshold,
    and EF-resonant if both w_E > threshold and w_F > threshold.

    Resonant modes are the ones that transfer energy between dimensions.

    Parameters
    ----------
    evals : f64[n]
    evecs : f64[n, n]
    nE, nF : dimensions
    threshold : float

    Returns
    -------
    labels : i32[n] - 0=edge, 1=face, 2=EF-resonant
    weights_E : f64[n] - edge weight fraction per mode
    weights_F : f64[n] - face weight fraction per mode
    n_resonant : int
    """
    cdef Py_ssize_t n = evals.shape[0], k, j
    cdef f64[:, ::1] ev = evecs

    cdef np.ndarray[i32, ndim=1] labels = np.empty(n, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] wE = np.empty(n, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] wF = np.empty(n, dtype=np.float64)
    cdef i32[::1] lv = labels
    cdef f64[::1] wev = wE, wfv = wF

    cdef f64 se, sf, total
    cdef Py_ssize_t n_res = 0

    for k in range(n):
        se = 0.0
        sf = 0.0
        for j in range(nE):
            se += ev[j, k] * ev[j, k]
        for j in range(nE, nE + nF):
            sf += ev[j, k] * ev[j, k]

        total = se + sf
        if total < 1e-15:
            wev[k] = 0.5
            wfv[k] = 0.5
            lv[k] = 2
            n_res += 1
            continue

        wev[k] = se / total
        wfv[k] = sf / total

        if wev[k] > 1.0 - threshold:
            lv[k] = 0   # edge-dominated
        elif wfv[k] > 1.0 - threshold:
            lv[k] = 1   # face-dominated
        else:
            lv[k] = 2   # EF-resonant
            n_res += 1

    return labels, wE, wF, n_res


def resonance_frequencies(np.ndarray[f64, ndim=1] freqs,
                          np.ndarray[i32, ndim=1] labels):
    """Extract frequencies of EF-resonant modes.

    These are the frequencies at which energy transfers between
    the edge and face tiers of the chain complex.

    Returns
    -------
    res_freqs : f64[n_resonant]
    res_indices : i32[n_resonant]
    """
    cdef Py_ssize_t n = freqs.shape[0], k, count = 0
    cdef i32[::1] lv = labels
    cdef f64[::1] fv = freqs

    # Count resonant modes
    for k in range(n):
        if lv[k] == 2:
            count += 1

    cdef np.ndarray[f64, ndim=1] rf = np.empty(count, dtype=np.float64)
    cdef np.ndarray[i32, ndim=1] ri = np.empty(count, dtype=np.int32)
    cdef f64[::1] rfv = rf
    cdef i32[::1] riv = ri
    cdef Py_ssize_t idx = 0

    for k in range(n):
        if lv[k] == 2:
            rfv[idx] = fv[k]
            riv[idx] = <i32>k
            idx += 1

    return rf, ri


# Section 6: Diffusion (heat equation)


def field_diffusion_spectral(np.ndarray[f64, ndim=1] F0,
                             np.ndarray[f64, ndim=1] evals,
                             np.ndarray[f64, ndim=2] evecs,
                             double t):
    """Heat equation on the field: F(t) = exp(-M t) F(0).

    F(t) = sum_k c_k exp(-lambda_k t) v_k

    Parameters
    ----------
    F0 : f64[n] - initial field state
    evals : f64[n] - eigenvalues of M
    evecs : f64[n, n] - eigenvectors
    t : float

    Returns
    -------
    Ft : f64[n] - diffused field state
    """
    cdef Py_ssize_t n = F0.shape[0], k, j
    cdef f64[::1] fv = F0
    cdef f64[:, ::1] ev = evecs
    cdef f64[::1] lv = evals

    # Compute coefficients
    cdef f64 *c = <f64 *>malloc(n * sizeof(f64))
    if c == NULL:
        raise MemoryError("field_diffusion_spectral: malloc failed")

    cdef f64 s
    for k in range(n):
        s = 0.0
        for j in range(n):
            s += ev[j, k] * fv[j]
        c[k] = s

    # Reconstruct with exponential decay
    cdef np.ndarray[f64, ndim=1] Ft = np.zeros(n, dtype=np.float64)
    cdef f64[::1] ftv = Ft
    cdef f64 decay

    for k in range(n):
        if fabs(c[k]) < 1e-15:
            continue
        decay = c[k] * exp(-lv[k] * t)
        for j in range(n):
            ftv[j] += decay * ev[j, k]

    free(c)
    return Ft


def field_diffusion_trajectory(np.ndarray[f64, ndim=1] F0,
                               np.ndarray[f64, ndim=1] evals,
                               np.ndarray[f64, ndim=2] evecs,
                               np.ndarray[f64, ndim=1] times):
    """Diffuse field through multiple timepoints.

    Returns
    -------
    traj : f64[T, n] - diffused field trajectory
    """
    cdef Py_ssize_t n = F0.shape[0], T = times.shape[0]
    cdef Py_ssize_t k, j, step
    cdef f64[::1] fv = F0
    cdef f64[:, ::1] ev = evecs
    cdef f64[::1] lv = evals, tv = times

    # Compute coefficients once
    cdef f64 *c = <f64 *>malloc(n * sizeof(f64))
    if c == NULL:
        raise MemoryError("field_diffusion_trajectory: malloc failed")

    cdef f64 s
    for k in range(n):
        s = 0.0
        for j in range(n):
            s += ev[j, k] * fv[j]
        c[k] = s

    cdef np.ndarray[f64, ndim=2] traj = np.zeros((T, n), dtype=np.float64)
    cdef f64[:, ::1] trajv = traj
    cdef f64 decay

    for step in range(T):
        for k in range(n):
            if fabs(c[k]) < 1e-15:
                continue
            decay = c[k] * exp(-lv[k] * tv[step])
            for j in range(n):
                trajv[step, j] += decay * ev[j, k]

    free(c)
    return traj


# Section 7: Vertex observables (derived, not independent)


def derive_vertex_trajectory(np.ndarray[f64, ndim=2] traj_EF,
                             object B1,
                             Py_ssize_t nE):
    """Derive vertex observables from a field trajectory.

    For each timepoint, f_V(t) = B_1 f_E(t) where f_E is the
    first nE components of the packed field state.

    Parameters
    ----------
    traj_EF : f64[T, nE+nF] - field trajectory
    B1 : (nV, nE) - boundary operator
    nE : int

    Returns
    -------
    traj_V : f64[T, nV]
    """
    cdef Py_ssize_t T = traj_EF.shape[0], step
    nV = B1.shape[0] if hasattr(B1, 'shape') else B1.nrow

    cdef np.ndarray[f64, ndim=2] traj_V = np.empty((T, nV), dtype=np.float64)

    for step in range(T):
        f_E = traj_EF[step, :nE]
        traj_V[step] = np.asarray(B1.dot(f_E), dtype=np.float64)

    return traj_V


def derive_vertex_state(np.ndarray[f64, ndim=1] F,
                        object B1,
                        Py_ssize_t nE):
    """Derive vertex observable from a single field state.

    f_V = B_1 f_E where f_E = F[:nE].
    """
    return np.asarray(B1.dot(F[:nE]), dtype=np.float64)


# Section 8: RK4 integration for field dynamics


def field_rk4_step(np.ndarray[f64, ndim=1] F,
                   np.ndarray[f64, ndim=1] dFdt,
                   object RL1,
                   object L2,
                   object B2,
                   double g,
                   Py_ssize_t nE,
                   Py_ssize_t nF,
                   double dt):
    """Single RK4 step for the second-order wave equation.

    Rewrites d^2F/dt^2 = -M F as first-order system:
        d/dt [F, V] = [V, -M F]
    where V = dF/dt.

    For large systems where spectral decomposition is too expensive.

    Returns
    -------
    F_new : f64[n]
    dFdt_new : f64[n]
    """
    cdef Py_ssize_t n = nE + nF

    def accel(np.ndarray[f64, ndim=1] pos):
        return -field_operator_matvec(pos, RL1, L2, B2, g, nE, nF)

    # RK4 on the (position, velocity) system
    cdef np.ndarray[f64, ndim=1] k1v = dFdt.copy()
    cdef np.ndarray[f64, ndim=1] k1a = accel(F)

    cdef np.ndarray[f64, ndim=1] k2v = dFdt + 0.5 * dt * k1a
    cdef np.ndarray[f64, ndim=1] k2a = accel(F + 0.5 * dt * k1v)

    cdef np.ndarray[f64, ndim=1] k3v = dFdt + 0.5 * dt * k2a
    cdef np.ndarray[f64, ndim=1] k3a = accel(F + 0.5 * dt * k2v)

    cdef np.ndarray[f64, ndim=1] k4v = dFdt + dt * k3a
    cdef np.ndarray[f64, ndim=1] k4a = accel(F + dt * k3v)

    cdef np.ndarray[f64, ndim=1] F_new = F + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    cdef np.ndarray[f64, ndim=1] dFdt_new = dFdt + (dt / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)

    return F_new, dFdt_new


def field_diffusion_rk4_step(np.ndarray[f64, ndim=1] F,
                             object RL1,
                             object L2,
                             object B2,
                             double g,
                             Py_ssize_t nE,
                             Py_ssize_t nF,
                             double dt):
    """Single RK4 step for heat equation dF/dt = -M F.

    For large systems where eigendecomposition is impractical.

    Returns
    -------
    F_new : f64[n]
    """
    def deriv(np.ndarray[f64, ndim=1] f):
        return -field_operator_matvec(f, RL1, L2, B2, g, nE, nF)

    cdef np.ndarray[f64, ndim=1] k1 = deriv(F)
    cdef np.ndarray[f64, ndim=1] k2 = deriv(F + 0.5 * dt * k1)
    cdef np.ndarray[f64, ndim=1] k3 = deriv(F + 0.5 * dt * k2)
    cdef np.ndarray[f64, ndim=1] k4 = deriv(F + dt * k3)

    return F + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
