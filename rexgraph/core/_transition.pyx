# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._transition - Transition operators on the rex chain complex.

Markov - discrete and continuous stochastic diffusion on k-cells.
Schrodinger - unitary evolution via Hodge Laplacians (real cos/sin split).
Differential - ODE integration (RK4) with coupled cross-dimensional dynamics.
Rewrite - signal resizing after structural mutation of the rex topology.

All operators are stateless: (state_arrays, operator_data) -> state_arrays.

Energy decomposition:
    E_kin = <f|L_1|f>     topological (boundary deviation)
    E_pot = <f|L_O|f>     geometric (overlap deviation)
    E_RL  = E_kin + alpha_G * E_pot

Coupled derivative uses RL_1 = L_1 + alpha_G * L_O for the edge
tier, with boundary operators B_1 and B_2 driving cross-dimensional
coupling. B_2 should be B2_hodge (self-loop faces filtered).
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt, exp, cos, sin

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,
)

np.import_array()

# Transition type constants
TRANS_MARKOV = 0
TRANS_SCHRODINGER = 1
TRANS_DIFFERENTIAL = 2
TRANS_REWRITE = 3


# Internal helpers

cdef inline np.ndarray _safe_dot(object A, np.ndarray x):
    """Polymorphic A @ x for dense ndarray or sparse matrix."""
    return A.dot(x)


# Markov diffusion

def markov_vertex_step(np.ndarray[f64, ndim=1] p,
                       np.ndarray[f64, ndim=2] W):
    """One discrete Markov step on vertex signals.

    p_new = W @ p where W is column-stochastic (D^{-1}A from B_1).
    """
    return W.dot(p)


def markov_edge_step(np.ndarray[f64, ndim=1] p,
                     object W_O):
    """One discrete Markov step on edge signals via overlap adjacency."""
    return W_O.dot(p)


def markov_face_step(np.ndarray[f64, ndim=1] p,
                     object W_F):
    """One discrete Markov step on face signals via face adjacency."""
    return W_F.dot(p)


def markov_multistep(np.ndarray[f64, ndim=1] p,
                     object W,
                     Py_ssize_t n_steps):
    """Apply n discrete Markov steps: p_new = W^n @ p.

    Returns
    -------
    final : f64[n]
    trajectory : f64[n_steps+1, n]
    """
    cdef Py_ssize_t n = p.shape[0], step
    cdef np.ndarray[f64, ndim=2] traj = np.empty((n_steps + 1, n), dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] cur = p.copy()
    traj[0] = cur
    for step in range(n_steps):
        cur = W.dot(cur)
        traj[step + 1] = cur
    return cur, traj


def markov_continuous_expm(np.ndarray[f64, ndim=1] p,
                           object L,
                           double t):
    """Continuous-time Markov: p(t) = exp(-L*t) @ p(0).

    Uses scipy expm. L is the Laplacian (generator = -L for diffusion).
    """
    from scipy.linalg import expm
    if hasattr(L, 'toarray'):
        L_dense = np.asarray(L.toarray() if hasattr(L, 'toarray') else L, dtype=np.float64)
    else:
        L_dense = L
    cdef np.ndarray[f64, ndim=2] E = expm(-L_dense * t)
    return E.dot(p)


@cython.boundscheck(False)
@cython.wraparound(False)
def markov_continuous_spectral(np.ndarray[f64, ndim=1] p,
                               np.ndarray[f64, ndim=1] evals,
                               np.ndarray[f64, ndim=2] evecs,
                               double t):
    """Continuous-time Markov via spectral decomposition.

    p(t) = V diag(exp(-lambda_k * t)) V^T p(0)

    Precomputes all mode coefficients c_k = v_k^T p(0) and decay
    factors d_k = exp(-lambda_k * t), then accumulates in one pass.

    Parameters
    ----------
    p : f64[n]
    evals : f64[k]
    evecs : f64[n, k]
    t : float

    Returns
    -------
    p_new : f64[n]
    """
    cdef Py_ssize_t n = p.shape[0], k = evals.shape[0], j, i
    cdef np.ndarray[f64, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef f64[::1] rv = result
    cdef f64[:] pv = p, ev = evals
    cdef f64[:, :] vv = evecs
    cdef f64 coeff, scale

    for j in range(k):
        coeff = 0.0
        for i in range(n):
            coeff += vv[i, j] * pv[i]
        scale = exp(-ev[j] * t) * coeff
        for i in range(n):
            rv[i] += scale * vv[i, j]

    return result


def build_vertex_transition_matrix(np.ndarray[f64, ndim=2] L0):
    """Column-stochastic transition matrix from L_0.

    W = I - D^{-1} L_0 where D = diag(L_0).
    Isolated vertices (d=0) get self-loop (W[i,i] = 1).

    Returns
    -------
    W : f64[nV, nV]
    """
    cdef Py_ssize_t n = L0.shape[0], i, j
    cdef np.ndarray[f64, ndim=2] W = np.zeros((n, n), dtype=np.float64)
    cdef f64[:, :] wv = W, lv = L0
    cdef f64 d

    for i in range(n):
        d = lv[i, i]
        if d > 1e-15:
            for j in range(n):
                if i != j:
                    wv[i, j] = -lv[i, j] / d
                else:
                    wv[i, i] = 0.0
        else:
            wv[i, i] = 1.0

    return W


@cython.boundscheck(False)
@cython.wraparound(False)
def build_lazy_transition_matrix(np.ndarray[f64, ndim=2] W, double lazy=0.5):
    """Lazy random walk: W_lazy = lazy * I + (1 - lazy) * W.

    Ensures aperiodicity. Fused single-pass, no temporaries.
    """
    cdef Py_ssize_t n = W.shape[0], i, j
    cdef np.ndarray[f64, ndim=2] WL = np.empty((n, n), dtype=np.float64)
    cdef f64[:, ::1] wlv = WL
    cdef f64[:, :] wv = W
    cdef double one_minus = 1.0 - lazy
    for i in range(n):
        for j in range(n):
            wlv[i, j] = one_minus * wv[i, j]
        wlv[i, i] += lazy
    return WL


# Schrodinger (unitary) evolution

@cython.boundscheck(False)
@cython.wraparound(False)
def schrodinger_evolve_spectral(np.ndarray[f64, ndim=1] f,
                                np.ndarray[f64, ndim=1] evals,
                                np.ndarray[f64, ndim=2] evecs,
                                double t):
    """Unitary evolution: f(t) = exp(-i L_k t) f(0) via spectral.

    Since L_k is real symmetric, exp(-iLt) splits into cos and sin.
    Tracks real and imaginary parts separately.

    Returns
    -------
    f_real : f64[n]
    f_imag : f64[n]
    """
    cdef Py_ssize_t n = f.shape[0], k = evals.shape[0], j, i
    cdef np.ndarray[f64, ndim=1] f_re = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f_im = np.zeros(n, dtype=np.float64)
    cdef f64[::1] fr = f_re, fi = f_im
    cdef f64[:] fv = f, ev = evals
    cdef f64[:, :] vv = evecs
    cdef f64 coeff, c_val, s_val

    for j in range(k):
        coeff = 0.0
        for i in range(n):
            coeff += vv[i, j] * fv[i]
        c_val = cos(ev[j] * t)
        s_val = sin(ev[j] * t)
        for i in range(n):
            fr[i] += c_val * coeff * vv[i, j]
            fi[i] -= s_val * coeff * vv[i, j]

    return f_re, f_im


def schrodinger_evolve_expm(np.ndarray[f64, ndim=1] f,
                            object L,
                            double t):
    """Unitary evolution via matrix exponential (dense, complex).

    Returns complex result as (real, imag) pair.
    """
    from scipy.linalg import expm
    cdef np.ndarray L_c
    if hasattr(L, 'toarray'):
        L_c = np.asarray(L.toarray() if hasattr(L, 'toarray') else L, dtype=np.complex128)
    else:
        L_c = L.astype(np.complex128)

    cdef np.ndarray U = expm(-1j * L_c * t)
    cdef np.ndarray result = U.dot(f.astype(np.complex128))
    return np.real(result).astype(np.float64), np.imag(result).astype(np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def quadratic_form(np.ndarray[f64, ndim=1] f_re,
                   np.ndarray[f64, ndim=1] f_im,
                   object L):
    """Compute <f|L|f> = Re(f)^T L Re(f) + Im(f)^T L Im(f).

    Works with dense or sparse L.
    """
    cdef np.ndarray[f64, ndim=1] Lf_re = _safe_dot(L, f_re)
    cdef np.ndarray[f64, ndim=1] Lf_im = _safe_dot(L, f_im)

    cdef f64[:] fr = f_re, fi = f_im, lr = Lf_re, li = Lf_im
    cdef Py_ssize_t n = f_re.shape[0], j
    cdef f64 energy = 0.0
    for j in range(n):
        energy += fr[j] * lr[j] + fi[j] * li[j]
    return energy


def kinetic_energy(np.ndarray[f64, ndim=1] f_re,
                   np.ndarray[f64, ndim=1] f_im,
                   object L1):
    """Topological energy E_kin = <f|L_1|f>."""
    return quadratic_form(f_re, f_im, L1)


def potential_energy(np.ndarray[f64, ndim=1] f_re,
                     np.ndarray[f64, ndim=1] f_im,
                     object LO):
    """Geometric energy E_pot = <f|L_O|f>."""
    if LO is None:
        return 0.0
    return quadratic_form(f_re, f_im, LO)


def energy_decomposition(np.ndarray[f64, ndim=1] f_re,
                         np.ndarray[f64, ndim=1] f_im,
                         object L1,
                         object LO,
                         double alpha_G):
    """Energy decomposition under the Relational Laplacian.

    E_kin  = <f|L_1|f>     (topological deviation)
    E_pot  = <f|L_O|f>     (geometric deviation)
    E_RL   = E_kin + alpha_G * E_pot

    Parameters
    ----------
    f_re, f_im : f64[nE]
    L1 : edge Hodge Laplacian
    LO : overlap Laplacian (or None)
    alpha_G : coupling constant

    Returns
    -------
    E_kin, E_pot, E_RL : float
    """
    cdef double Ek = kinetic_energy(f_re, f_im, L1)
    cdef double Ep = 0.0
    if LO is not None:
        Ep = potential_energy(f_re, f_im, LO)

    return Ek, Ep, Ek + alpha_G * Ep


def schrodinger_multistep(np.ndarray[f64, ndim=1] f,
                          np.ndarray[f64, ndim=1] evals,
                          np.ndarray[f64, ndim=2] evecs,
                          np.ndarray[f64, ndim=1] times):
    """Evolve through multiple timepoints, recording trajectory.

    Returns
    -------
    traj_re : f64[len(times), n]
    traj_im : f64[len(times), n]
    """
    cdef Py_ssize_t nT = times.shape[0], n = f.shape[0], step
    cdef np.ndarray[f64, ndim=2] tr = np.empty((nT, n), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] ti = np.empty((nT, n), dtype=np.float64)

    for step in range(nT):
        f_re, f_im = schrodinger_evolve_spectral(f, evals, evecs, times[step])
        tr[step] = f_re
        ti[step] = f_im

    return tr, ti


# Differential (ODE)

def rk4_step(np.ndarray[f64, ndim=1] y, double t, double dt,
             derivative_func):
    """Single RK4 integration step.

    derivative_func(y, t) -> dy/dt as f64[n].
    """
    cdef np.ndarray[f64, ndim=1] k1, k2, k3, k4
    k1 = derivative_func(y, t)
    k2 = derivative_func(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivative_func(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivative_func(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate(np.ndarray[f64, ndim=1] y0, double t0, double t1,
                  Py_ssize_t n_steps, derivative_func):
    """Integrate ODE from t0 to t1 using n_steps of RK4.

    Returns
    -------
    y_final : f64[n]
    trajectory : f64[n_steps+1, n]
    times : f64[n_steps+1]
    """
    cdef double dt = (t1 - t0) / <f64>n_steps
    cdef Py_ssize_t n = y0.shape[0], step
    cdef np.ndarray[f64, ndim=2] traj = np.empty((n_steps + 1, n), dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] times = np.empty(n_steps + 1, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] y = y0.copy()
    cdef double t = t0

    traj[0] = y; times[0] = t
    for step in range(n_steps):
        y = rk4_step(y, t, dt, derivative_func)
        t += dt
        traj[step + 1] = y; times[step + 1] = t

    return y, traj, times


def diffusion_derivative(np.ndarray[f64, ndim=1] f,
                         object L,
                         double diffusion_rate=1.0):
    """Heat equation derivative: df/dt = -rate * L @ f."""
    return -diffusion_rate * _safe_dot(L, f)


def coupled_derivative(np.ndarray[f64, ndim=1] flat_state,
                       np.ndarray[i32, ndim=1] sizes,
                       object L0,
                       object L1,
                       object L2,
                       object L_O,
                       B1_dense, B2_dense,
                       double alpha0=1.0,
                       double alpha1=1.0,
                       double alpha2=1.0,
                       double alpha_G=0.0):
    """Coupled cross-dimensional diffusion with Relational Laplacian.

    df0/dt = -alpha0 * L_0 @ f0
    df1/dt = -(alpha1 * L_1 + alpha_G * L_O) @ f1 + B_1^T @ f0
    df2/dt = -alpha2 * L_2 @ f2 + B_2^T @ f1

    B_2 should be B2_hodge (self-loop faces filtered) so that the
    chain complex is exact. The edge-tier operator is the Rex
    Laplacian RL_1 = alpha1 * L_1 + alpha_G * L_O.

    Parameters
    ----------
    flat_state : f64[nV + nE + nF]
    sizes : i32[3] = [nV, nE, nF]
    L0, L1, L2, L_O : operators (dense or sparse)
    B1_dense, B2_dense : boundary matrices (or None)
    alpha0, alpha1, alpha2 : per-tier diffusion rates
    alpha_G : geometric coupling constant

    Returns
    -------
    f64[nV + nE + nF]
    """
    cdef i32[:] sz = sizes
    cdef Py_ssize_t nV = sz[0], nE = sz[1], nF = sz[2]

    cdef np.ndarray[f64, ndim=1] f0 = flat_state[:nV]
    cdef np.ndarray[f64, ndim=1] f1 = flat_state[nV:nV + nE]
    cdef np.ndarray[f64, ndim=1] f2 = flat_state[nV + nE:]

    # Vertex tier: simple diffusion
    cdef np.ndarray[f64, ndim=1] df0 = -alpha0 * _safe_dot(L0, f0)

    # Edge tier: Relational Laplacian RL_1 = alpha1 * L_1 + alpha_G * L_O
    cdef np.ndarray[f64, ndim=1] df1
    if L_O is not None and alpha_G != 0.0:
        df1 = -(alpha1 * _safe_dot(L1, f1) + alpha_G * _safe_dot(L_O, f1))
    else:
        df1 = -alpha1 * _safe_dot(L1, f1)

    if B1_dense is not None:
        df1 = df1 + B1_dense.T.dot(f0)

    # Face tier
    cdef np.ndarray[f64, ndim=1] df2
    if nF > 0:
        df2 = -alpha2 * _safe_dot(L2, f2)
        if B2_dense is not None:
            df2 = df2 + B2_dense.T.dot(f1)
    else:
        df2 = np.zeros(0, dtype=np.float64)

    return np.concatenate([df0, df1, df2])


# Rewrite (signal resizing)

def rewrite_insert_edges_i32(np.ndarray[f64, ndim=1] f0,
                              np.ndarray[f64, ndim=1] f1,
                              np.ndarray[f64, ndim=1] f2,
                              Py_ssize_t nV_old, Py_ssize_t nE_old,
                              Py_ssize_t nV_new, Py_ssize_t nE_new,
                              double default_vertex_val=0.0,
                              double default_edge_val=0.0):
    """Resize state signals after edge insertion.

    New vertices/edges get default values. Faces unchanged.
    """
    cdef Py_ssize_t j
    cdef np.ndarray[f64, ndim=1] new_f0 = np.full(nV_new, default_vertex_val, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] new_f1 = np.full(nE_new, default_edge_val, dtype=np.float64)
    cdef f64[:] nf0 = new_f0, nf1 = new_f1, of0 = f0, of1 = f1

    for j in range(nV_old):
        if j < nV_new: nf0[j] = of0[j]
    for j in range(nE_old):
        if j < nE_new: nf1[j] = of1[j]

    return new_f0, new_f1, f2.copy()


def rewrite_delete_edges_i32(np.ndarray[f64, ndim=1] f0,
                              np.ndarray[f64, ndim=1] f1,
                              np.ndarray[f64, ndim=1] f2,
                              np.ndarray[i32, ndim=1] vertex_map,
                              np.ndarray[i32, ndim=1] edge_map,
                              Py_ssize_t nV_new, Py_ssize_t nE_new):
    """Resize state signals after edge deletion with reindexing.

    vertex_map[old] = new_idx or -1 (removed).
    edge_map[old] = new_idx or -1 (deleted).
    """
    cdef Py_ssize_t j, nV_old = f0.shape[0], nE_old = f1.shape[0]
    cdef np.ndarray[f64, ndim=1] new_f0 = np.zeros(nV_new, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] new_f1 = np.zeros(nE_new, dtype=np.float64)
    cdef f64[:] nf0 = new_f0, nf1 = new_f1, of0 = f0, of1 = f1
    cdef i32[:] vm = vertex_map, em = edge_map

    for j in range(nV_old):
        if vm[j] >= 0:
            nf0[vm[j]] = of0[j]
    for j in range(nE_old):
        if em[j] >= 0:
            nf1[em[j]] = of1[j]

    return new_f0, new_f1, f2.copy()


def rewrite_add_faces(np.ndarray[f64, ndim=1] f2,
                      Py_ssize_t n_new_faces,
                      double default_face_val=0.0):
    """Extend face signal for newly added faces."""
    cdef np.ndarray[f64, ndim=1] ext = np.full(n_new_faces, default_face_val, dtype=np.float64)
    return np.concatenate([f2, ext])


def rewrite_remove_faces(np.ndarray[f64, ndim=1] f2,
                         np.ndarray[i32, ndim=1] keep_mask):
    """Remove faces by boolean mask (1=keep, 0=remove)."""
    cdef Py_ssize_t nF = f2.shape[0], j
    cdef Py_ssize_t n_keep = 0
    cdef i32[:] km = keep_mask
    for j in range(nF):
        if km[j]: n_keep += 1
    cdef np.ndarray[f64, ndim=1] new_f2 = np.empty(n_keep, dtype=np.float64)
    cdef f64[:] nf = new_f2, of = f2
    cdef Py_ssize_t pos = 0
    for j in range(nF):
        if km[j]:
            nf[pos] = of[j]; pos += 1
    return new_f2


# Transition dispatch

def apply_transition(int trans_type,
                     np.ndarray[f64, ndim=1] f0,
                     np.ndarray[f64, ndim=1] f1,
                     np.ndarray[f64, ndim=1] f2,
                     int target_dim,
                     operator_data,
                     double dt=1.0,
                     double t=0.0,
                     Py_ssize_t n_steps=1):
    """Dispatch for applying any transition type.

    Parameters
    ----------
    trans_type : int
        TRANS_MARKOV (0), TRANS_SCHRODINGER (1), TRANS_DIFFERENTIAL (2),
        TRANS_REWRITE (3).
    f0, f1, f2 : current state signals
    target_dim : which cell dimension the operator targets (0/1/2)
    operator_data : dict with operator-specific arrays
        MARKOV: {"W": transition_matrix} or
                {"L": laplacian, "evals": ..., "evecs": ...}
        SCHRODINGER: {"evals": ..., "evecs": ...} or {"L": laplacian}
        DIFFERENTIAL: {"derivative_func": callable}
        REWRITE: not supported here, use rewrite_* directly
    dt : time step
    t : current time
    n_steps : number of steps (Markov discrete / RK4)

    Returns
    -------
    new_f0, new_f1, new_f2 : updated state signals
        Untouched dimensions are returned as-is (no copy).
    """
    cdef np.ndarray[f64, ndim=1] signal

    if target_dim == 0: signal = f0
    elif target_dim == 1: signal = f1
    elif target_dim == 2: signal = f2
    else: raise ValueError("target_dim must be 0, 1, or 2")

    cdef np.ndarray[f64, ndim=1] new_signal

    if trans_type == TRANS_MARKOV:
        if "W" in operator_data:
            if n_steps == 1:
                new_signal = operator_data["W"].dot(signal)
            else:
                new_signal, _ = markov_multistep(signal, operator_data["W"], n_steps)
        elif "evals" in operator_data:
            new_signal = markov_continuous_spectral(
                signal, operator_data["evals"], operator_data["evecs"], dt)
        elif "L" in operator_data:
            new_signal = markov_continuous_expm(signal, operator_data["L"], dt)
        else:
            raise ValueError("MARKOV requires 'W', 'L', or 'evals'+'evecs'")

    elif trans_type == TRANS_SCHRODINGER:
        if "evals" in operator_data:
            f_re, f_im = schrodinger_evolve_spectral(
                signal, operator_data["evals"], operator_data["evecs"], dt)
        elif "L" in operator_data:
            f_re, f_im = schrodinger_evolve_expm(signal, operator_data["L"], dt)
        else:
            raise ValueError("SCHRODINGER requires 'L' or 'evals'+'evecs'")
        new_signal = f_re
        operator_data["_last_imag"] = f_im

    elif trans_type == TRANS_DIFFERENTIAL:
        deriv = operator_data["derivative_func"]
        new_signal, _, _ = rk4_integrate(signal, t, t + dt, n_steps, deriv)

    elif trans_type == TRANS_REWRITE:
        raise ValueError("Use rewrite_* functions directly for structural mutation")

    else:
        raise ValueError("Unknown transition type: %d" % trans_type)

    # Return untouched signals as-is (no unnecessary copies)
    if target_dim == 0: return new_signal, f1, f2
    elif target_dim == 1: return f0, new_signal, f2
    else: return f0, f1, new_signal
