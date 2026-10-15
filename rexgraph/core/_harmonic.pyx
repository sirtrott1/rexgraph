# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""Dense harmonic reference operations for bounded numerical comparisons.

The module constructs a basis of ker(L1) from supplied boundary matrices,
forms component projectors, and evaluates products and coordinate transforms
on that basis. Runtime sparse analysis uses rexgraph.harmonic_sparse.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

ctypedef np.float64_t f64


def harmonic_basis(np.ndarray[f64, ndim=2] B1,
                   np.ndarray[f64, ndim=2] B2):
    """
    Extract the orthonormal harmonic basis from boundary operators.

    Parameters

    B1 : (nV, nE) float64
        Grade-1 boundary operator.
    B2 : (nE, nF) float64
        Grade-2 boundary operator.

    Returns

    harm_basis : (nE, dim_H) float64
        Columns are orthonormal harmonic basis vectors.
    evals : (nE,) float64
        All eigenvalues of L_1, for diagnostics.
    """
    cdef int nE = B1.shape[1]
    cdef np.ndarray[f64, ndim=2] L1 = B1.T @ B1 + B2 @ B2.T
    cdef np.ndarray[f64, ndim=1] evals
    cdef np.ndarray[f64, ndim=2] evecs

    evals, evecs = np.linalg.eigh(L1)

    cdef int j
    cdef int dim_H = 0
    for j in range(nE):
        if evals[j] < 1e-10:
            dim_H += 1

    cdef np.ndarray[f64, ndim=2] hb = evecs[:, :dim_H]
    return hb, evals


def harmonic_projectors(np.ndarray[f64, ndim=2] B1,
                        np.ndarray[f64, ndim=2] B2):
    """
    Compute the three Hodge projectors: P_grad, P_curl, P_harm.

    Parameters

    B1 : (nV, nE) float64
    B2 : (nE, nF) float64

    Returns

    dict with keys 'P_harm', 'P_grad', 'P_curl', 'harm_basis', 'dim_H'.
    """
    cdef int nE = B1.shape[1]
    cdef np.ndarray[f64, ndim=2] hb
    cdef np.ndarray[f64, ndim=1] evals

    hb, evals = harmonic_basis(B1, B2)
    cdef int dim_H = hb.shape[1]

    cdef np.ndarray[f64, ndim=2] P_harm = hb @ hb.T
    cdef np.ndarray[f64, ndim=2] B1T_B1 = B1 @ B1.T
    cdef np.ndarray[f64, ndim=2] P_grad = B1.T @ np.linalg.pinv(B1T_B1) @ B1
    cdef np.ndarray[f64, ndim=2] P_curl = np.eye(nE) - P_grad - P_harm

    return {
        'P_harm': P_harm,
        'P_grad': P_grad,
        'P_curl': P_curl,
        'harm_basis': hb,
        'dim_H': dim_H,
    }


def harmonic_product_table(np.ndarray[f64, ndim=2] harm_basis):
    """
    Compute the Hadamard product multiplication table on H.

    Parameters

    harm_basis : (nE, dim_H) float64
        Orthonormal harmonic basis.

    Returns

    dict with mult_table (dim_H, dim_H, dim_H), closure matrix,
    commutativity and associativity violations.
    """
    cdef int nE = harm_basis.shape[0]
    cdef int dim_H = harm_basis.shape[1]
    cdef int i, j, m, a, b, c

    if dim_H == 0:
        return {'dim_H': 0, 'error': 'trivial harmonic subspace'}

    cdef np.ndarray[f64, ndim=2] P_harm = harm_basis @ harm_basis.T
    cdef np.ndarray[f64, ndim=3] mult = np.zeros((dim_H, dim_H, dim_H), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] closure = np.zeros((dim_H, dim_H), dtype=np.float64)

    cdef np.ndarray[f64, ndim=1] product, projected, coords
    cdef double total_sq, proj_sq

    for i in range(dim_H):
        for j in range(dim_H):
            product = harm_basis[:, i] * harm_basis[:, j]
            projected = P_harm @ product
            coords = harm_basis.T @ projected
            for m in range(dim_H):
                mult[i, j, m] = coords[m]
            total_sq = 0.0
            proj_sq = 0.0
            for e in range(nE):
                total_sq += product[e] * product[e]
                proj_sq += projected[e] * projected[e]
            closure[i, j] = proj_sq / total_sq if total_sq > 1e-30 else 0.0

    # Commutativity check
    cdef double max_comm = 0.0, d
    for i in range(dim_H):
        for j in range(i + 1, dim_H):
            d = 0.0
            for m in range(dim_H):
                d += (mult[i, j, m] - mult[j, i, m]) ** 2
            d = sqrt(d)
            if d > max_comm:
                max_comm = d

    # Associativity check (sample first 4 basis vectors)
    cdef double max_assoc = 0.0
    cdef int n_check = min(dim_H, 4)
    cdef np.ndarray[f64, ndim=1] ab, ab_H, abc_L, bc, bc_H, abc_R
    cdef np.ndarray[f64, ndim=1] coords_L, coords_R

    for a in range(n_check):
        for b in range(n_check):
            for c in range(n_check):
                ab = harm_basis[:, a] * harm_basis[:, b]
                ab_H = P_harm @ ab
                abc_L = P_harm @ (ab_H * harm_basis[:, c])
                coords_L = harm_basis.T @ abc_L

                bc = harm_basis[:, b] * harm_basis[:, c]
                bc_H = P_harm @ bc
                abc_R = P_harm @ (harm_basis[:, a] * bc_H)
                coords_R = harm_basis.T @ abc_R

                d = 0.0
                for m in range(dim_H):
                    d += (coords_L[m] - coords_R[m]) ** 2
                d = sqrt(d)
                if d > max_assoc:
                    max_assoc = d

    # Nilpotency
    cdef list nilpotency = []
    cdef np.ndarray[f64, ndim=1] h_sq, proj_h
    cdef double norm_sq, norm_proj
    for i in range(dim_H):
        h_sq = harm_basis[:, i] ** 2
        proj_h = P_harm @ h_sq
        norm_sq = 0.0
        norm_proj = 0.0
        for e in range(nE):
            norm_sq += h_sq[e] * h_sq[e]
            norm_proj += proj_h[e] * proj_h[e]
        nilpotency.append(sqrt(norm_proj) / sqrt(norm_sq) if norm_sq > 1e-30 else 0.0)

    return {
        'dim_H': dim_H,
        'mult_table': mult,
        'closure': closure,
        'mean_closure': float(np.mean(closure)),
        'min_closure': float(np.min(closure)),
        'max_closure': float(np.max(closure)),
        'commutative': max_comm < 1e-10,
        'max_commutativity_violation': max_comm,
        'max_associativity_violation': max_assoc,
        'associative': max_assoc < 1e-6,
        'nilpotency': nilpotency,
    }


def harmonic_channel_character(np.ndarray[f64, ndim=2] harm_basis,
                               np.ndarray[f64, ndim=2] chi):
    """
    Compute the channel character of the harmonic subspace.

    Parameters

    harm_basis : (nE, dim_H) float64
    chi : (nE, 4) float64
        Per-edge structural character on Delta^3.

    Returns

    chi_H : (4,) float64
        Average channel character across harmonic basis vectors.
    isotropic : bool
        Whether all four channels are approximately equal.
    """
    cdef int dim_H = harm_basis.shape[1]
    cdef int nE = harm_basis.shape[0]
    cdef np.ndarray[f64, ndim=1] chi_H = np.zeros(4, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] h_sq
    cdef double h_sum
    cdef int h_idx, e, c

    for h_idx in range(dim_H):
        h_sq = harm_basis[:, h_idx] ** 2
        h_sum = 0.0
        for e in range(nE):
            h_sum += h_sq[e]
        if h_sum > 1e-30:
            for e in range(nE):
                h_sq[e] /= h_sum
            for c in range(4):
                for e in range(nE):
                    chi_H[c] += h_sq[e] * chi[e, c]

    if dim_H > 0:
        for c in range(4):
            chi_H[c] /= dim_H

    cdef double std = np.std(chi_H)
    return chi_H, std < 0.02


def harmonic_encode(np.ndarray[f64, ndim=1] data_coords,
                    np.ndarray[f64, ndim=2] harm_basis):
    """
    Encode a coordinate vector into the harmonic subspace.

    Parameters

    data_coords : (dim_H,) float64
        Coordinates in the harmonic basis.
    harm_basis : (nE, dim_H) float64
        Orthonormal harmonic basis.

    Returns

    harm_vec : (nE,) float64
        Edge signal in the harmonic subspace.
    """
    return harm_basis @ data_coords


def harmonic_decode(np.ndarray[f64, ndim=1] harm_vec,
                    np.ndarray[f64, ndim=2] harm_basis):
    """
    Decode a harmonic vector back to coordinates.

    Parameters

    harm_vec : (nE,) float64
        Edge signal (possibly with gradient/curl noise).
    harm_basis : (nE, dim_H) float64
        Orthonormal harmonic basis.

    Returns

    coords : (dim_H,) float64
        Coordinates in the harmonic basis (noise annihilated).
    """
    return harm_basis.T @ harm_vec


def harmonic_leakage(np.ndarray[f64, ndim=1] signal,
                     np.ndarray[f64, ndim=2] P_harm,
                     np.ndarray[f64, ndim=2] P_grad,
                     np.ndarray[f64, ndim=2] P_curl):
    """
    Measure the Hodge decomposition of a signal and the leakage
    between subspaces.

    Parameters

    signal : (nE,) float64
    P_harm, P_grad, P_curl : (nE, nE) float64

    Returns

    dict with norms, percentages, and cross leakage (should be ~0).
    """
    cdef np.ndarray[f64, ndim=1] h = P_harm @ signal
    cdef np.ndarray[f64, ndim=1] g = P_grad @ signal
    cdef np.ndarray[f64, ndim=1] c = P_curl @ signal

    cdef double nh = np.linalg.norm(h)
    cdef double ng = np.linalg.norm(g)
    cdef double nc = np.linalg.norm(c)
    cdef double total = nh*nh + ng*ng + nc*nc

    # Cross leakage: harmonic content in gradient/curl projections
    cdef double h_in_g = np.linalg.norm(P_harm @ g)
    cdef double h_in_c = np.linalg.norm(P_harm @ c)

    return {
        'harm_norm': nh,
        'grad_norm': ng,
        'curl_norm': nc,
        'harm_pct': nh*nh / total if total > 1e-30 else 0.0,
        'grad_pct': ng*ng / total if total > 1e-30 else 0.0,
        'curl_pct': nc*nc / total if total > 1e-30 else 0.0,
        'harm_in_grad': h_in_g,
        'harm_in_curl': h_in_c,
    }
