# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._hodge: Hodge decomposition of edge signals.

Decomposes an edge signal g in R^m into three mutually orthogonal
components:

    g = B_1^T phi + B_2 psi + eta
        (gradient)   (curl)   (harmonic)

where B_1^T phi lies in im(B_1^T), B_2 psi lies in im(B_2), and eta
lies in ker(L_1). Orthogonality holds when B_1 B_2 = 0 (the chain
complex condition). If self loop faces are present, they must be
filtered from B_2 before calling this module; see graph.py B2_hodge.

Potentials are recovered via pseudoinverse:
    phi = L_0^+ (B_1 g),   psi = L_2^+ (B_2^T g)

Energy orthogonality: ||g||^2 = ||grad||^2 + ||curl||^2 + ||harm||^2.

Native execution uses LSQR on the boundary factors. The dense routine is a
reference oracle and is never selected by the public decomposition.

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

    weights : f64[nE]
        Edge weights (magnitudes).
    edge_type_indices : i32[nE] or None
        Index into the type array for each edge. If None, all positive.
    negative_type_mask : uint8[n_types] or None
        1 if the type is negative, 0 otherwise.

    Returns

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

    Returns zeros if the signal is all zero.

    Parameters

    x : f64[n]

    Returns

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

    B1 : DualCSR, shape (nV, nE).
    flow : f64[nE]

    Returns

    f64[nV]
    """
    from rexgraph.core._sparse import matvec
    return matvec(B1, flow)


def compute_face_curl(B2, np.ndarray[f64, ndim=1] flow):
    """Face curl B_2^T g.

    Parameters

    B2 : DualCSR, shape (nE, nF).
    flow : f64[nE]

    Returns

    f64[nF]
    """
    from rexgraph.core._sparse import rmatvec
    return rmatvec(B2, flow)


# Per edge resistance ratio

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_rho(np.ndarray[f64, ndim=1] harm,
                np.ndarray[f64, ndim=1] flow):
    """Per edge resistance ratio rho(e) = |eta_e| / |g_e|.

    Local amplitude ratio. It can exceed one when the components cancel
    on an edge. Zero where the original flow is zero.

    Parameters

    harm : f64[nE]
        Harmonic component.
    flow : f64[nE]
        Original flow signal.

    Returns

    f64[nE]
        Nonnegative; not bounded above by one.
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

    grad, curl, harm : f64[nE]

    Returns

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
    precision. Large values indicate that the chain condition
    is violated, likely because self loop faces were not filtered
    from B_2.

    Returns

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

    Returns the potentials alongside the components. phi and psi are the
    coordinates the components are built from, so a caller wanting to work in
    the Hodge spaces rather than in the edge space reads them here instead of
    solving the same systems again.

    Parameters

    B1 : DualCSR, shape (nV, nE).
    B2 : DualCSR or None, shape (nE, nF).
    flow : f64[nE]
    L0_mat : ndarray (nV, nV)
    L2_mat : ndarray (nF, nF) or None

    Returns

    grad, curl, harm : f64[nE]
    phi : f64[nV]
    psi : f64[nF], empty when there are no faces
    """
    from rexgraph.core._sparse import matvec, rmatvec

    cdef Py_ssize_t nE = flow.shape[0]

    # phi = L_0^+ B_1 g, grad = B_1^T phi. L_0 is singular -- its kernel is the component
    # indicators -- so this is the harmonic complement inverse, (L + Pi_h)^-1 - Pi_h,
    # rather than a least squares that has to rediscover that null space numerically.
    rhs_grad = matvec(B1, flow)
    phi = _dense_psd_pinv_apply(np.asarray(L0_mat, dtype=np.float64),
                                np.asarray(rhs_grad, dtype=np.float64))
    grad = rmatvec(B1, phi)

    # psi = L_2^+ B_2^T g, curl = B_2 psi
    cdef bint has_faces = (B2 is not None and B2.ncol > 0
                           and L2_mat is not None and L2_mat.shape[0] > 0)
    if has_faces:
        rhs_curl = rmatvec(B2, flow)
        psi = _dense_psd_pinv_apply(np.asarray(L2_mat, dtype=np.float64),
                                    np.asarray(rhs_curl, dtype=np.float64))
        curl = matvec(B2, psi)
    else:
        curl = np.zeros(nE, dtype=np.float64)
        psi = np.zeros(0, dtype=np.float64)

    harm = flow - grad - curl

    return (np.asarray(grad, dtype=np.float64),
            np.asarray(curl, dtype=np.float64),
            np.asarray(harm, dtype=np.float64),
            np.asarray(phi, dtype=np.float64),
            np.asarray(psi, dtype=np.float64))


# Hodge decomposition, sparse path

def _hodge_sparse(B1, B2, np.ndarray[f64, ndim=1] flow, L0_sp, L2_sp):
    """Compatibility entry point for the native Hodge decomposition."""
    return hodge_decomposition(B1, B2, flow, L0=L0_sp, L2=L2_sp, potentials=True)


def _stable_norm(values):
    scale = float(np.max(np.abs(values), initial=0.0))
    return 0.0 if scale == 0 else scale * float(np.linalg.norm(values / scale))


def least_squares(B, values, *, transpose=False, tol=1e-12, maxiter=2000,
                  return_info=False):
    """Minimum norm numerical least squares using native boundary actions.

    LSQR uses Golub Kahan bidiagonalization, starting at zero without damping
    or right preconditioning. Each iterate is in im(A^T). This fixes the
    Euclidean minimum norm convention even for rectangular or singular A.
    Both residual tests are recomputed from the returned iterate. Failure to
    meet either test raises; no unconverged coefficients are returned.

    Algorithm: LSQR, as published by Paige and Saunders in 1982.
    https://web.stanford.edu/group/SOL/software/lsqr/
    """
    from numbers import Integral
    from rexgraph.native_sparse import as_native
    from rexgraph.linear_operator import _numeric_array
    if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("least squares tolerance must lie strictly between zero and one")
    if isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, Integral) or maxiter < 1:
        raise ValueError("least squares maxiter must be a positive integer")
    A = as_native(B)
    if transpose:
        A = A.T
    block = _numeric_array(values, operation="least squares")
    if np.iscomplexobj(block):
        raise TypeError("least squares requires real coefficients")
    one = block.ndim == 1
    if one:
        block = block[:, None]
    if block.ndim != 2 or block.shape[0] != A.shape[0]:
        raise ValueError("least squares RHS must match the matrix row axis")
    scale = float(np.max(np.abs(A.data), initial=0.0))
    A = A.with_data(A.data / scale) if scale else A
    bound = _stable_norm(A.data)
    out = np.zeros((A.shape[1], block.shape[1]))
    observations = []
    for column in range(block.shape[1]):
        rhs = block[:, column]
        rhs_scale = float(np.max(np.abs(rhs), initial=0.0))
        rhs = rhs / rhs_scale if rhs_scale else rhs
        bnorm = _stable_norm(rhs)
        x = np.zeros(A.shape[1])
        u = rhs.copy()
        beta = bnorm
        if beta:
            u /= beta
        v = A.transpose_apply(u)
        alpha = _stable_norm(v)
        if alpha:
            v /= alpha
        w = v.copy()
        phi_bar, rho_bar = beta, alpha
        converged = bnorm == 0 or bound == 0 or alpha <= tol * bound
        relative, normal = (0.0 if bnorm == 0 else 1.0), (alpha / bound if bound else 0.0)
        iterations = 0
        for iteration in range(int(maxiter)):
            if converged:
                break
            u = A.apply(v) - alpha * u
            beta = _stable_norm(u)
            if beta:
                u /= beta
            v = A.transpose_apply(u) - beta * v
            alpha = _stable_norm(v)
            if alpha:
                v /= alpha
            rho = float(np.hypot(rho_bar, beta))
            if rho == 0:
                break
            cosine, sine = rho_bar / rho, beta / rho
            theta, rho_bar = sine * alpha, -cosine * alpha
            phi, phi_bar = cosine * phi_bar, sine * phi_bar
            x += (phi / rho) * w
            w = v - (theta / rho) * w
            residual = A.apply(x) - rhs
            rnorm = _stable_norm(residual)
            relative = rnorm / bnorm
            normal = (_stable_norm(A.transpose_apply(residual)) / bound / rnorm
                      if rnorm and bound else 0.0)
            iterations = iteration + 1
            converged = relative <= tol or normal <= tol
        if not converged or not np.all(np.isfinite(x)):
            raise ArithmeticError("native LSQR did not converge to the requested residual tolerance")
        # Divide and multiply in this order only when both remain representable.
        # longdouble keeps a finite final result from overflowing its scale ratio.
        with np.errstate(over='ignore', invalid='ignore'):
            result = (x.astype(np.longdouble) * np.longdouble(rhs_scale)
                      / np.longdouble(scale)) if scale else x
            out[:, column] = result
        if not np.all(np.isfinite(out[:, column])):
            raise FloatingPointError("least squares solution is outside float64")
        observations.append({"iterations": iterations, "relative_residual": relative,
                             "normal_residual": normal})
    result = out[:, 0] if one else out
    info = {"kernel": "native-lsqr", "tol": float(tol), "maxiter": int(maxiter),
            "columns": observations, "status": "observed"}
    return (result, info) if return_info else result


# Hodge decomposition entry point

cdef _dense_psd_pinv_apply(A, b):
    """`A^+ b` for a dense symmetric PSD `A`, without a spectrum where one is avoidable.

    Three readings in order, each exact for the case it claims:

    1. `A` positive definite -- `A^+ = A^-1`, and one Cholesky both proves it and solves.
    2. `A` a Laplacian -- its kernel is the component indicators, so the harmonic-
       complement inverse `(A + Pi_h)^-1 - Pi_h` applies. The frame is VERIFIED against
       `A` first: a component frame that is not actually in the kernel would deflate the
       wrong subspace, which is a wrong answer rather than a slow one.
    3. Anything else -- SVD least squares, which is the minimum norm solution and so the
       pseudoinverse action for a singular operator whose kernel is not known.
    """
    import scipy.sparse as _sp

    from rexgraph.core._linalg import harmonic_pinv_matvec, lstsq as _lp_lstsq
    from rexgraph.sparse_interfacing import _component_projector
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
    if A.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    from rexgraph.core._linalg import spd_solve
    solved = spd_solve(A, b)
    if solved is not None:
        return solved
    project = _component_projector(_sp.csr_matrix(A))
    probe = project(np.eye(A.shape[0], dtype=np.float64))
    if not np.any(A @ probe):
        # Exactly zero: for a Laplacian built from integer incidence the component
        # indicators are in the kernel exactly, so this needs no tolerance.
        return harmonic_pinv_matvec(A, project, b)
    phi, _rank = _lp_lstsq(A, b)
    return phi


def hodge_decomposition(B1, B2, np.ndarray[f64, ndim=1] flow,
                        L0=None, L2=None, bint potentials=False):
    """Decompose edge signal into gradient, curl, and harmonic.

    B_2 should have self loop faces filtered out so that B_1 B_2 = 0
    holds exactly. When this condition holds, the three components are
    mutually orthogonal and their energies sum to ||g||^2.

    Parameters

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

    grad : f64[nE]
        Gradient component B_1^T phi, in im(B_1^T).
    curl : f64[nE]
        Curl component B_2 psi, in im(B_2).
    harm : f64[nE]
        Harmonic residual, in ker(L_1).
    phi : f64[nV], only when `potentials` is set
        The vertex potential the gradient is built from.
    psi : f64[nF], only when `potentials` is set
        The face potential the curl is built from, empty without faces.
    """
    cdef Py_ssize_t nE = flow.shape[0]
    cdef Py_ssize_t nV = B1.nrow
    cdef Py_ssize_t nF = B2.ncol if B2 is not None else 0

    from rexgraph.native_sparse import NativeSparse
    lower = NativeSparse(B1)
    if lower.shape[1] != nE or (B2 is not None and B2.nrow != nE):
        raise ValueError("Hodge boundary axes must match the edge signal")
    phi = (least_squares(lower, flow, transpose=True) if L0 is None else
           least_squares(L0, lower.apply(flow)))
    grad = lower.transpose_apply(phi)
    if nF:
        upper = NativeSparse(B2)
        psi = (least_squares(upper, flow) if L2 is None else
               least_squares(L2, upper.transpose_apply(flow)))
        curl = upper.apply(psi)
    else:
        psi, curl = np.zeros(0), np.zeros(nE)
    out = (grad, curl, flow - grad - curl, phi, psi)
    return out if potentials else out[:3]


# Full Hodge analysis

def build_hodge(B1, B2,
                np.ndarray[f64, ndim=1] flow,
                L0=None, L2=None):
    """Hodge decomposition with all derived quantities.

    Parameters

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
