# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._hodge - Hodge decomposition of edge signals.

Decomposes an edge signal g in R^m into three mutually orthogonal
components:

    g = B_1^T phi + B_2 psi + eta
        (gradient)   (curl)   (harmonic)

where B_1^T phi lies in im(B_1^T), B_2 psi lies in im(B_2), and eta
lies in ker(L_1). Orthogonality holds when B_1 B_2 = 0 (the chain
complex condition). If self-loop faces are present, they must be
filtered from B_2 before calling this module; see graph.py B2_hodge.

Potentials are recovered via pseudoinverse:
    phi = L_0^+ (B_1 g),   psi = L_2^+ (B_2^T g)

Energy orthogonality: ||g||^2 = ||grad||^2 + ||curl||^2 + ||harm||^2.

Dense path (small dimensions): numpy lstsq (LAPACK dgelsd).
Sparse path (large dimensions): scipy lsqr (iterative).

Provides:
    build_flow_signal - oriented edge signal from weights and types
    normalize_signal - scale to [-1, 1] by max absolute value
    compute_divergence - vertex divergence B_1 g
    compute_face_curl - face curl B_2^T g
    compute_rho - per-edge harmonic resistance ratio
    compute_energy_percentages - energy fractions of each component
    hodge_decomposition - decompose into gradient, curl, harmonic
    build_hodge - full analysis with all derived quantities
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,
    can_allocate_dense_f64,
    should_use_dense_eigen,
    should_use_dense_matmul,
    get_EPSILON_NORM,
)

from libc.math cimport fabs, sqrt

np.import_array()


# Signal construction

@cython.boundscheck(False)
@cython.wraparound(False)
def build_flow_signal(np.ndarray[f64, ndim=1] weights,
                      np.ndarray[i32, ndim=1] edge_type_indices=None,
                      np.ndarray[np.uint8_t, ndim=1] negative_type_mask=None):
    """Oriented edge flow signal from weights and types.

    For each edge j, flow[j] = weights[j] * sign[j], where sign is -1
    if the edge type is marked negative in the mask, +1 otherwise.

    Parameters
    ----------
    weights : f64[nE]
        Edge weights (magnitudes).
    edge_type_indices : i32[nE] or None
        Index into the type array for each edge. If None, all positive.
    negative_type_mask : uint8[n_types] or None
        1 if the type is negative, 0 otherwise.

    Returns
    -------
    f64[nE]
        Oriented flow signal.
    """
    cdef Py_ssize_t nE = weights.shape[0]
    cdef np.ndarray[f64, ndim=1] flow = np.empty(nE, dtype=np.float64)
    cdef f64[::1] fv = flow, wv = weights
    cdef Py_ssize_t j

    if edge_type_indices is None or negative_type_mask is None:
        for j in range(nE):
            fv[j] = wv[j]
        return flow

    cdef i32[::1] ti = edge_type_indices
    cdef np.uint8_t[::1] nm = negative_type_mask

    for j in range(nE):
        if nm[ti[j]]:
            fv[j] = -wv[j]
        else:
            fv[j] = wv[j]

    return flow


# Signal normalization

@cython.boundscheck(False)
@cython.wraparound(False)
def normalize_signal(np.ndarray[f64, ndim=1] x):
    """Scale to [-1, 1] by dividing by max absolute value.

    Returns zeros if the signal is all-zero.

    Parameters
    ----------
    x : f64[n]

    Returns
    -------
    f64[n]
    """
    cdef double mx = 0.0
    cdef f64[::1] xv = x
    cdef Py_ssize_t i, n = x.shape[0]

    for i in range(n):
        if fabs(xv[i]) > mx:
            mx = fabs(xv[i])

    if mx < get_EPSILON_NORM():
        return np.zeros(n, dtype=np.float64)

    cdef np.ndarray[f64, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef f64[::1] ov = out
    cdef double inv_mx = 1.0 / mx

    for i in range(n):
        ov[i] = xv[i] * inv_mx

    return out


# Vertex divergence and face curl

def compute_divergence(B1, np.ndarray[f64, ndim=1] flow):
    """Vertex divergence B_1 g.

    Parameters
    ----------
    B1 : DualCSR, shape (nV, nE).
    flow : f64[nE]

    Returns
    -------
    f64[nV]
    """
    from rexgraph.core._sparse import matvec
    return matvec(B1, flow)


def compute_face_curl(B2, np.ndarray[f64, ndim=1] flow):
    """Face curl B_2^T g.

    Parameters
    ----------
    B2 : DualCSR, shape (nE, nF).
    flow : f64[nE]

    Returns
    -------
    f64[nF]
    """
    from rexgraph.core._sparse import rmatvec
    return rmatvec(B2, flow)


# Per-edge resistance ratio

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_rho(np.ndarray[f64, ndim=1] harm,
                np.ndarray[f64, ndim=1] flow):
    """Per-edge resistance ratio rho(e) = |eta_e| / |g_e|.

    Fraction of each edge's flow that is harmonic. Zero where the
    original flow is zero.

    Parameters
    ----------
    harm : f64[nE]
        Harmonic component.
    flow : f64[nE]
        Original flow signal.

    Returns
    -------
    f64[nE]
        In [0, 1].
    """
    cdef Py_ssize_t nE = flow.shape[0]
    cdef np.ndarray[f64, ndim=1] rho = np.zeros(nE, dtype=np.float64)
    cdef f64[::1] rv = rho, hv = harm, fv = flow
    cdef Py_ssize_t j
    cdef double af

    for j in range(nE):
        af = fabs(fv[j])
        if af > get_EPSILON_NORM():
            rv[j] = fabs(hv[j]) / af

    return rho


# Energy decomposition

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_energy_percentages(np.ndarray[f64, ndim=1] grad,
                               np.ndarray[f64, ndim=1] curl,
                               np.ndarray[f64, ndim=1] harm):
    """Energy partition: ||g||^2 = ||grad||^2 + ||curl||^2 + ||harm||^2.

    Parameters
    ----------
    grad, curl, harm : f64[nE]

    Returns
    -------
    pct_grad, pct_curl, pct_harm : float
        Energy fractions summing to 1.0 (or all 0.0 if total is zero).
    """
    cdef Py_ssize_t nE = grad.shape[0]
    cdef f64[::1] gv = grad, cv = curl, hv = harm
    cdef double eg = 0.0, ec = 0.0, eh = 0.0, total
    cdef Py_ssize_t j

    for j in range(nE):
        eg += gv[j] * gv[j]
        ec += cv[j] * cv[j]
        eh += hv[j] * hv[j]

    total = eg + ec + eh
    if total < 1e-30:
        return 0.0, 0.0, 0.0

    return eg / total, ec / total, eh / total


# Orthogonality verification

@cython.boundscheck(False)
@cython.wraparound(False)
def check_orthogonality(np.ndarray[f64, ndim=1] grad,
                        np.ndarray[f64, ndim=1] curl,
                        np.ndarray[f64, ndim=1] harm):
    """Inner products between Hodge components.

    When B_1 B_2 = 0, all three inner products should be near machine
    precision. Large values indicate that the chain complex condition
    is violated, likely because self-loop faces were not filtered
    from B_2.

    Returns
    -------
    dict
        grad_curl, grad_harm, curl_harm: absolute inner products.
        max_inner: largest of the three.
        orthogonal: True if max_inner < 1e-6.
    """
    cdef Py_ssize_t nE = grad.shape[0]
    cdef f64[::1] gv = grad, cv = curl, hv = harm
    cdef double gc = 0.0, gh = 0.0, ch = 0.0
    cdef Py_ssize_t j

    for j in range(nE):
        gc += gv[j] * cv[j]
        gh += gv[j] * hv[j]
        ch += cv[j] * hv[j]

    gc = fabs(gc)
    gh = fabs(gh)
    ch = fabs(ch)

    cdef double mx = gc
    if gh > mx:
        mx = gh
    if ch > mx:
        mx = ch

    return {
        'grad_curl': gc,
        'grad_harm': gh,
        'curl_harm': ch,
        'max_inner': mx,
        'orthogonal': mx < 1e-6,
    }


# Hodge decomposition, dense path

def _hodge_dense(B1, B2, np.ndarray[f64, ndim=1] flow, L0_mat, L2_mat):
    """Dense Hodge decomposition via lstsq (LAPACK dgelsd).

    Parameters
    ----------
    B1 : DualCSR, shape (nV, nE).
    B2 : DualCSR or None, shape (nE, nF).
    flow : f64[nE]
    L0_mat : ndarray (nV, nV)
    L2_mat : ndarray (nF, nF) or None

    Returns
    -------
    grad, curl, harm : f64[nE]
    """
    from rexgraph.core._sparse import matvec, rmatvec

    cdef Py_ssize_t nE = flow.shape[0]

    # phi = L_0^+ B_1 g, grad = B_1^T phi
    rhs_grad = matvec(B1, flow)
    from rexgraph.core._linalg import lstsq as _lp_lstsq
    phi, _ = _lp_lstsq(np.asarray(L0_mat, dtype=np.float64), np.asarray(rhs_grad, dtype=np.float64))
    grad = rmatvec(B1, phi)

    # psi = L_2^+ B_2^T g, curl = B_2 psi
    cdef bint has_faces = (B2 is not None and B2.ncol > 0
                           and L2_mat is not None and L2_mat.shape[0] > 0)
    if has_faces:
        rhs_curl = rmatvec(B2, flow)
        psi, _ = _lp_lstsq(np.asarray(L2_mat, dtype=np.float64), np.asarray(rhs_curl, dtype=np.float64))
        curl = matvec(B2, psi)
    else:
        curl = np.zeros(nE, dtype=np.float64)

    harm = flow - grad - curl

    return (np.asarray(grad, dtype=np.float64),
            np.asarray(curl, dtype=np.float64),
            np.asarray(harm, dtype=np.float64))


# Hodge decomposition, sparse path

def _hodge_sparse(B1, B2, np.ndarray[f64, ndim=1] flow, L0_sp, L2_sp):
    """Sparse Hodge decomposition via lsqr (iterative).

    Parameters
    ----------
    B1 : DualCSR, shape (nV, nE).
    B2 : DualCSR or None, shape (nE, nF).
    flow : f64[nE]
    L0_sp : scipy.sparse (nV, nV)
    L2_sp : scipy.sparse (nF, nF) or None

    Returns
    -------
    grad, curl, harm : f64[nE]
    """
    from rexgraph.core._sparse import to_scipy_csr
    # scipy.sparse.linalg.lsqr - kept for sparse Hodge path
    from scipy.sparse.linalg import lsqr as _lsqr

    cdef Py_ssize_t nE = flow.shape[0]

    sp_B1 = to_scipy_csr(B1)

    rhs_grad = sp_B1 @ flow
    phi = _lsqr(L0_sp.astype(np.float64, copy=False),
                rhs_grad.astype(np.float64))[0]
    grad = sp_B1.T @ phi

    cdef bint has_faces = (B2 is not None and B2.ncol > 0
                           and L2_sp is not None and L2_sp.shape[0] > 0)
    if has_faces:
        sp_B2 = to_scipy_csr(B2)
        rhs_curl = sp_B2.T @ flow
        psi = _lsqr(L2_sp.astype(np.float64, copy=False),
                     rhs_curl.astype(np.float64))[0]
        curl = sp_B2 @ psi
    else:
        curl = np.zeros(nE, dtype=np.float64)

    harm = flow - grad - curl

    return (np.asarray(grad, dtype=np.float64),
            np.asarray(curl, dtype=np.float64),
            np.asarray(harm, dtype=np.float64))


# Hodge decomposition entry point

def hodge_decomposition(B1, B2, np.ndarray[f64, ndim=1] flow,
                        L0=None, L2=None):
    """Decompose edge signal into gradient, curl, and harmonic.

    B_2 should have self-loop faces filtered out so that B_1 B_2 = 0
    holds exactly. When this condition holds, the three components are
    mutually orthogonal and their energies sum to ||g||^2.

    Parameters
    ----------
    B1 : DualCSR, shape (nV, nE).
    B2 : DualCSR or None, shape (nE, nF_hodge).
        Exclude self-loop faces for exact orthogonality.
    flow : f64[nE]
        Edge signal to decompose.
    L0 : ndarray or scipy.sparse or None
        Vertex Laplacian. Built internally if None.
    L2 : ndarray or scipy.sparse or None
        Face Laplacian. Built internally if None.

    Returns
    -------
    grad : f64[nE]
        Gradient component B_1^T phi, in im(B_1^T).
    curl : f64[nE]
        Curl component B_2 psi, in im(B_2).
    harm : f64[nE]
        Harmonic residual, in ker(L_1).
    """
    cdef Py_ssize_t nE = flow.shape[0]
    cdef Py_ssize_t nV = B1.nrow
    cdef Py_ssize_t nF = B2.ncol if B2 is not None else 0

    cdef Py_ssize_t max_lap_dim = nV
    if nF > max_lap_dim:
        max_lap_dim = nF
    cdef bint use_dense = should_use_dense_matmul(max_lap_dim)

    if L0 is None:
        if use_dense:
            from rexgraph.core._sparse import spmm_AAt_dense_f64
            L0 = spmm_AAt_dense_f64(B1)
        else:
            from rexgraph.core._sparse import to_scipy_csr
            sp_B1 = to_scipy_csr(B1)
            L0 = sp_B1 @ sp_B1.T

    if L2 is None and nF > 0:
        if use_dense:
            from rexgraph.core._sparse import spmm_AtA_dense_f64
            L2 = spmm_AtA_dense_f64(B2)
        else:
            from rexgraph.core._sparse import to_scipy_csr
            sp_B2 = to_scipy_csr(B2)
            L2 = sp_B2.T @ sp_B2

    if use_dense:
        if L0 is not None and not isinstance(L0, np.ndarray):
            L0 = np.asarray(L0.toarray() if hasattr(L0, 'toarray') else L0, dtype=np.float64)
        if L2 is not None and not isinstance(L2, np.ndarray):
            L2 = np.asarray(L2.toarray() if hasattr(L2, 'toarray') else L2, dtype=np.float64)
        return _hodge_dense(B1, B2, flow, L0, L2)
    else:
        if L0 is not None and isinstance(L0, np.ndarray):
            from scipy.sparse import csr_matrix
            L0 = csr_matrix(L0)
        if L2 is not None and isinstance(L2, np.ndarray):
            from scipy.sparse import csr_matrix
            L2 = csr_matrix(L2)
        return _hodge_sparse(B1, B2, flow, L0, L2)


# Full Hodge analysis

def build_hodge(B1, B2,
                np.ndarray[f64, ndim=1] flow,
                L0=None, L2=None):
    """Hodge decomposition with all derived quantities.

    Parameters
    ----------
    B1 : DualCSR, shape (nV, nE).
    B2 : DualCSR or None, shape (nE, nF_hodge).
        Exclude self-loop faces for exact orthogonality.
    flow : f64[nE]
        Edge signal to decompose.
    L0 : ndarray or scipy.sparse or None
        Precomputed L_0. Built if None.
    L2 : ndarray or scipy.sparse or None
        Precomputed L_2. Built if None.

    Returns
    -------
    dict
        grad, curl, harm : f64[nE]
            Raw decomposition components.
        grad_norm, curl_norm, harm_norm, flow_norm : f64[nE]
            Components divided by their max absolute value.
        rho : f64[nE]
            Per-edge harmonic resistance ratio |eta_e| / |g_e|.
        pct_grad, pct_curl, pct_harm : float
            Energy fractions summing to 1.0.
        divergence, div_norm : f64[nV]
            Vertex divergence B_1 g and its normalization.
        face_curl : f64[nF]
            Face curl B_2^T g.
        orthogonality : dict
            Inner products between components. When B_1 B_2 = 0
            (self-loop faces filtered), max_inner is near machine
            precision (~1e-15).
    """
    result = {}

    grad, curl, harm = hodge_decomposition(B1, B2, flow, L0=L0, L2=L2)

    result['flow'] = flow
    result['grad'] = grad
    result['curl'] = curl
    result['harm'] = harm

    result['flow_norm'] = normalize_signal(flow)
    result['grad_norm'] = normalize_signal(grad)
    result['curl_norm'] = normalize_signal(curl)
    result['harm_norm'] = normalize_signal(harm)

    result['rho'] = compute_rho(harm, flow)

    pct_g, pct_c, pct_h = compute_energy_percentages(grad, curl, harm)
    result['pct_grad'] = pct_g
    result['pct_curl'] = pct_c
    result['pct_harm'] = pct_h

    div = compute_divergence(B1, flow)
    result['divergence'] = div
    result['div_norm'] = normalize_signal(div)

    if B2 is not None and B2.ncol > 0:
        result['face_curl'] = compute_face_curl(B2, flow)
    else:
        result['face_curl'] = np.empty(0, dtype=np.float64)

    result['orthogonality'] = check_orthogonality(grad, curl, harm)

    return result
