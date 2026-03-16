# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._dirac - Dirac operator on the graded cell space.

The Dirac operator D = d + d* acts on the graded vector space
V = R^nV + R^nE + R^nF. It encodes the full chain complex structure
in a single symmetric matrix.

D^2 = blkdiag(L0, L1, L2) by the chain condition B1*B2 = 0.
Schrodinger evolution exp(-iDt) preserves ||Psi||^2 exactly.
The face component vanishes under canonical vertex collapse.

Reference: RCFE Foundations, Sections 4, 8, 9.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt, cos, sin, exp
from libc.string cimport memset, memcpy

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    get_EPSILON_NORM,
)

from rexgraph.core._linalg cimport (
    lp_eigh,
    bl_gemm_nn, bl_gemm_nt, bl_gemm_tn,
)

np.import_array()


def build_dirac_operator(np.ndarray[f64, ndim=2] B1,
                          np.ndarray[f64, ndim=2] B2):
    """Build the Dirac operator D = d + d* on the graded cell space.

    D is the (nV + nE + nF) x (nV + nE + nF) real symmetric matrix:

        D = [[ 0,       B1,      0     ],
             [ B1^T,    0,       B2    ],
             [ 0,       B2^T,    0     ]]

    Parameters
    ----------
    B1 : f64[nV, nE] - vertex-edge boundary operator
    B2 : f64[nE, nF] - edge-face boundary operator

    Returns
    -------
    D : f64[N, N] where N = nV + nE + nF
    sizes : (nV, nE, nF)
    """
    cdef int nV = B1.shape[0]
    cdef int nE = B1.shape[1]
    cdef int nF = B2.shape[1]
    cdef int N = nV + nE + nF

    cdef np.ndarray[f64, ndim=2] D = np.zeros((N, N), dtype=np.float64)
    cdef f64[:, ::1] dv = D
    cdef f64[:, ::1] b1v = B1, b2v = B2
    cdef int i, j

    # B1 block: D[0:nV, nV:nV+nE] = B1
    for i in range(nV):
        for j in range(nE):
            dv[i, nV + j] = b1v[i, j]
            dv[nV + j, i] = b1v[i, j]  # symmetric: B1^T

    # B2 block: D[nV:nV+nE, nV+nE:N] = B2
    for i in range(nE):
        for j in range(nF):
            dv[nV + i, nV + nE + j] = b2v[i, j]
            dv[nV + nE + j, nV + i] = b2v[i, j]  # symmetric: B2^T

    return D, (nV, nE, nF)


def dirac_eigen(np.ndarray[f64, ndim=2] D):
    """Eigendecomposition of the Dirac operator.

    Returns (evals, evecs) sorted by eigenvalue magnitude.
    Eigenvalues can be positive or negative (D is not PSD).
    """
    cdef int N = D.shape[0]
    cdef np.ndarray[f64, ndim=2] D_F = np.asfortranarray(D.copy())
    cdef np.ndarray[f64, ndim=1] evals = np.empty(N, dtype=np.float64)
    lp_eigh(&D_F[0, 0], &evals[0], N)
    cdef np.ndarray[f64, ndim=2] evecs = np.ascontiguousarray(D_F)
    return evals, evecs


def verify_d_squared(np.ndarray[f64, ndim=2] D,
                      np.ndarray[f64, ndim=2] L0,
                      np.ndarray[f64, ndim=2] L1,
                      np.ndarray[f64, ndim=2] L2,
                      int nV, int nE, int nF,
                      double tol=1e-10):
    """Verify D^2 = blkdiag(L0, L1, L2).

    The off-diagonal blocks of D^2 vanish because B1*B2 = 0.
    Returns (is_valid, max_error).
    """
    cdef int N = nV + nE + nF
    cdef np.ndarray[f64, ndim=2] D2 = D @ D

    # Build expected blkdiag
    cdef np.ndarray[f64, ndim=2] expected = np.zeros((N, N), dtype=np.float64)
    expected[:nV, :nV] = L0
    expected[nV:nV+nE, nV:nV+nE] = L1
    if nF > 0:
        expected[nV+nE:, nV+nE:] = L2

    cdef double max_err = float(np.max(np.abs(D2 - expected)))
    return max_err < tol, max_err


def schrodinger_evolve(np.ndarray[f64, ndim=1] evals,
                        np.ndarray[f64, ndim=2] evecs,
                        np.ndarray[f64, ndim=1] psi0,
                        double t):
    """Evolve graded state under Schrodinger equation: Psi(t) = exp(-iDt) Psi(0).

    Uses spectral decomposition: Psi(t) = sum_j exp(-i*lambda_j*t) <Psi0, phi_j> phi_j

    Since D is real symmetric and Psi0 is real, the result has real and
    imaginary parts:
        Psi_re(t) = sum_j cos(lambda_j * t) * c_j * phi_j
        Psi_im(t) = sum_j -sin(lambda_j * t) * c_j * phi_j

    where c_j = <Psi0, phi_j>.

    Parameters
    ----------
    evals : f64[N] - Dirac eigenvalues
    evecs : f64[N, N] - Dirac eigenvectors (columns)
    psi0 : f64[N] - initial graded state
    t : float - time

    Returns
    -------
    psi_re : f64[N] - real part of Psi(t)
    psi_im : f64[N] - imaginary part of Psi(t)
    """
    cdef int N = evals.shape[0]
    cdef np.ndarray[f64, ndim=1] psi_re = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] psi_im = np.zeros(N, dtype=np.float64)
    cdef f64[::1] rv = psi_re, iv = psi_im
    cdef f64[::1] ev = evals, p0 = psi0
    cdef f64[:, ::1] ecv = evecs
    cdef int j, k
    cdef f64 cj, cos_lt, sin_lt, lam

    for j in range(N):
        # c_j = <Psi0, phi_j>
        cj = 0
        for k in range(N):
            cj += p0[k] * ecv[k, j]

        if fabs(cj) < 1e-15:
            continue

        lam = ev[j]
        cos_lt = cos(lam * t)
        sin_lt = sin(lam * t)

        for k in range(N):
            rv[k] += cos_lt * cj * ecv[k, j]
            iv[k] -= sin_lt * cj * ecv[k, j]

    return psi_re, psi_im


def schrodinger_trajectory(np.ndarray[f64, ndim=1] evals,
                            np.ndarray[f64, ndim=2] evecs,
                            np.ndarray[f64, ndim=1] psi0,
                            np.ndarray[f64, ndim=1] times):
    """Evolve graded state at multiple timepoints.

    Returns
    -------
    traj_re : f64[T, N] - real parts
    traj_im : f64[T, N] - imaginary parts
    born : f64[T, N] - Born probability |Psi_k(t)|^2 per cell
    """
    cdef int N = evals.shape[0]
    cdef int T = times.shape[0]

    # Precompute spectral coefficients
    cdef np.ndarray[f64, ndim=1] coeffs = evecs.T @ psi0
    cdef f64[::1] cv = coeffs, ev = evals, tv = times

    cdef np.ndarray[f64, ndim=2] traj_re = np.zeros((T, N), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] traj_im = np.zeros((T, N), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] born = np.zeros((T, N), dtype=np.float64)
    cdef f64[:, ::1] trv = traj_re, tiv = traj_im, bv = born
    cdef f64[:, ::1] ecv = evecs

    cdef int ti, j, k
    cdef f64 cos_lt, sin_lt, lam, cj, re_k, im_k

    for ti in range(T):
        for j in range(N):
            cj = cv[j]
            if fabs(cj) < 1e-15:
                continue
            lam = ev[j]
            cos_lt = cos(lam * tv[ti])
            sin_lt = sin(lam * tv[ti])
            for k in range(N):
                trv[ti, k] += cos_lt * cj * ecv[k, j]
                tiv[ti, k] -= sin_lt * cj * ecv[k, j]

        # Born probability
        for k in range(N):
            re_k = trv[ti, k]
            im_k = tiv[ti, k]
            bv[ti, k] = re_k * re_k + im_k * im_k

    return traj_re, traj_im, born


def canonical_collapse(np.ndarray[f64, ndim=2] B1,
                        int nV, int nE, int nF,
                        int vertex_idx):
    """Canonical graded projection upon observing vertex v.

    P = (delta_v, B1^T delta_v, 0, ...) / ||...||

    The face component is exactly zero because B2^T B1^T delta_v
    = (B1 B2)^T delta_v = 0 by the chain condition.

    Parameters
    ----------
    B1 : f64[nV, nE]
    nV, nE, nF : dimensions
    vertex_idx : which vertex is observed

    Returns
    -------
    psi_collapsed : f64[N] - normalized graded state
    """
    cdef int N = nV + nE + nF
    cdef np.ndarray[f64, ndim=1] psi = np.zeros(N, dtype=np.float64)
    cdef f64[::1] pv = psi
    cdef f64[:, ::1] b1v = B1
    cdef int e
    cdef f64 nm

    # Vertex component: delta_v
    pv[vertex_idx] = 1.0

    # Edge component: B1^T delta_v = B1[v, :]
    for e in range(nE):
        pv[nV + e] = b1v[vertex_idx, e]

    # Face component: zero (B2^T B1^T delta_v = 0 by chain condition)

    # Normalize
    nm = 0
    for e in range(N):
        nm += pv[e] * pv[e]
    nm = sqrt(nm)
    if nm > 1e-15:
        for e in range(N):
            pv[e] /= nm

    return psi


def born_graded(np.ndarray[f64, ndim=1] psi_re,
                np.ndarray[f64, ndim=1] psi_im,
                int nV, int nE, int nF):
    """Born probability per cell and per dimension from graded state.

    Returns
    -------
    per_cell : f64[N] - |Psi_k|^2 per cell
    per_dim : f64[3] - total probability in V, E, F sectors
    """
    cdef int N = nV + nE + nF
    cdef np.ndarray[f64, ndim=1] per_cell = np.empty(N, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] per_dim = np.zeros(3, dtype=np.float64)
    cdef f64[::1] pcv = per_cell, pdv = per_dim
    cdef f64[::1] rv = psi_re, iv = psi_im
    cdef int k
    cdef f64 p

    for k in range(N):
        p = rv[k] * rv[k] + iv[k] * iv[k]
        pcv[k] = p
        if k < nV:
            pdv[0] += p
        elif k < nV + nE:
            pdv[1] += p
        else:
            pdv[2] += p

    return per_cell, per_dim


def energy_partition(np.ndarray[f64, ndim=1] psi_re,
                      np.ndarray[f64, ndim=1] psi_im,
                      int nV, int nE, int nF):
    """Fraction of total energy in each dimensional sector.

    Returns f64[3] with V, E, F fractions summing to 1.
    """
    _, per_dim = born_graded(psi_re, psi_im, nV, nE, nF)
    cdef f64 total = per_dim[0] + per_dim[1] + per_dim[2]
    if total > 1e-15:
        per_dim[0] /= total
        per_dim[1] /= total
        per_dim[2] /= total
    return per_dim
