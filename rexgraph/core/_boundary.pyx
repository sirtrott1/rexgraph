# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._boundary - Chain complex construction for the 2-rex.

Assembles the boundary operators B_1 (nV x nE) and B_2 (nE x nF)
that define the chain complex, and computes Betti numbers from
Laplacian eigenvalues.

Provides:
    build_B1 - signed vertex-edge incidence matrix
    build_B2_from_cycles - signed edge-face boundary from cycle data
    build_B2_from_dense - B_2 from a dense matrix
    verify_chain_complex - check B_1 B_2 = 0 via sparse matvec
    count_zero_eigenvalues - count near-zero eigenvalues
    rank_from_eigenvalues - matrix rank from Laplacian spectrum
    betti_from_eigenvalues - all Betti numbers from Laplacian spectra
    betti_numbers - unified interface with SVD fallback
    compute_rank - SVD-based rank (slow, use spectral path instead)
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,

    MAX_INT32_NNZ,

    should_use_dense_eigen,
    can_allocate_dense_f64,
    get_eigen_dense_limit,
)

from libc.math cimport fabs

np.import_array()


# B_1 construction

@cython.boundscheck(False)
@cython.wraparound(False)
def build_B1_i32(Py_ssize_t nV, Py_ssize_t nE,
                 np.ndarray[i32, ndim=1] sources,
                 np.ndarray[i32, ndim=1] targets):
    """
    Assemble B_1 (nV x nE) from oriented edge endpoints.

    For each edge j: B_1[sources[j], j] = -1, B_1[targets[j], j] = +1.

    Parameters
    ----------
    nV : int
        Number of vertices (row dimension).
    nE : int
        Number of edges (column dimension).
    sources, targets : int32[nE]
        Tail and head vertex indices for each edge.

    Returns
    -------
    DualCSR
        B1 as a dual CSR/CSC matrix with float64 values.
    """
    from rexgraph.core._sparse import dual_from_coo

    cdef Py_ssize_t nnz = 2 * nE
    cdef np.ndarray[i32, ndim=1] rows = np.empty(nnz, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] cols = np.empty(nnz, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] vals = np.empty(nnz, dtype=np.float64)

    cdef i32[::1] r = rows, c = cols
    cdef i32[::1] s = sources, t = targets
    cdef f64[::1] v = vals
    cdef Py_ssize_t j, k

    k = 0
    for j in range(nE):
        r[k] = s[j]; c[k] = <i32>j; v[k] = -1.0; k += 1
        r[k] = t[j]; c[k] = <i32>j; v[k] = 1.0; k += 1

    return dual_from_coo(rows, cols, vals, nV, nE)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_B1_i64(Py_ssize_t nV, Py_ssize_t nE,
                 np.ndarray[i64, ndim=1] sources,
                 np.ndarray[i64, ndim=1] targets):
    """Assemble B_1 (nV x nE). int64 variant."""
    from rexgraph.core._sparse import dual_from_coo

    cdef Py_ssize_t nnz = 2 * nE
    cdef np.ndarray[i64, ndim=1] rows = np.empty(nnz, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] cols = np.empty(nnz, dtype=np.int64)
    cdef np.ndarray[f64, ndim=1] vals = np.empty(nnz, dtype=np.float64)

    cdef i64[::1] r = rows, c = cols
    cdef i64[::1] s = sources, t = targets
    cdef f64[::1] v = vals
    cdef Py_ssize_t j, k

    k = 0
    for j in range(nE):
        r[k] = s[j]; c[k] = <i64>j; v[k] = -1.0; k += 1
        r[k] = t[j]; c[k] = <i64>j; v[k] = 1.0; k += 1

    return dual_from_coo(rows, cols, vals, nV, nE)


def build_B1(Py_ssize_t nV, Py_ssize_t nE, sources, targets):
    """
    Assemble B_1 (nV x nE). Dispatches on index dtype.
    """
    if not isinstance(sources, np.ndarray):
        sources = np.asarray(sources)
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets)

    if sources.dtype == np.int64 or targets.dtype == np.int64 or max(nV, nE) >= MAX_INT32_NNZ:
        return build_B1_i64(nV, nE,
                            sources.astype(np.int64, copy=False),
                            targets.astype(np.int64, copy=False))
    return build_B1_i32(nV, nE,
                        sources.astype(np.int32, copy=False),
                        targets.astype(np.int32, copy=False))


# B_2 construction

@cython.boundscheck(False)
@cython.wraparound(False)
def build_B2_from_cycles_i32(Py_ssize_t nE,
                              np.ndarray[i32, ndim=1] cycle_edges,
                              np.ndarray[f64, ndim=1] cycle_signs,
                              np.ndarray[i32, ndim=1] cycle_lengths):
    """
    Assemble B_2 (nE x nF) from flat cycle data.

    cycle_edges and cycle_signs are concatenated across faces.
    Face f occupies cycle_lengths[f] consecutive positions.

    Parameters
    ----------
    nE : int
        Number of edges (row dimension of B2).
    cycle_edges : int32[total_boundary_edges]
        Edge indices in each face boundary, concatenated.
    cycle_signs : float64[total_boundary_edges]
        Orientation signs (+/-1.0) for each boundary edge.
    cycle_lengths : int32[nF]
        Number of boundary edges per face.

    Returns
    -------
    DualCSR
        B2 as a dual CSR/CSC matrix with float64 values.
    """
    from rexgraph.core._sparse import dual_from_coo

    cdef Py_ssize_t nF = cycle_lengths.shape[0]
    cdef Py_ssize_t nnz = cycle_edges.shape[0]

    cdef np.ndarray[i32, ndim=1] rows = np.empty(nnz, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] cols = np.empty(nnz, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] vals = np.empty(nnz, dtype=np.float64)

    cdef i32[::1] r = rows, c = cols
    cdef i32[::1] ce = cycle_edges, cl = cycle_lengths
    cdef f64[::1] v = vals, cs = cycle_signs
    cdef Py_ssize_t f, k, offset

    offset = 0
    for f in range(nF):
        for k in range(cl[f]):
            r[offset + k] = ce[offset + k]
            c[offset + k] = <i32>f
            v[offset + k] = cs[offset + k]
        offset += cl[f]

    return dual_from_coo(rows, cols, vals, nE, nF)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_B2_from_cycles_i64(Py_ssize_t nE,
                              np.ndarray[i64, ndim=1] cycle_edges,
                              np.ndarray[f64, ndim=1] cycle_signs,
                              np.ndarray[i64, ndim=1] cycle_lengths):
    """Assemble B_2 (nE x nF) from flat cycle data. int64 variant."""
    from rexgraph.core._sparse import dual_from_coo

    cdef Py_ssize_t nF = cycle_lengths.shape[0]
    cdef Py_ssize_t nnz = cycle_edges.shape[0]

    cdef np.ndarray[i64, ndim=1] rows = np.empty(nnz, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] cols = np.empty(nnz, dtype=np.int64)
    cdef np.ndarray[f64, ndim=1] vals = np.empty(nnz, dtype=np.float64)

    cdef i64[::1] r = rows, c = cols
    cdef i64[::1] ce = cycle_edges, cl = cycle_lengths
    cdef f64[::1] v = vals, cs = cycle_signs
    cdef Py_ssize_t f, k, offset

    offset = 0
    for f in range(nF):
        for k in range(cl[f]):
            r[offset + k] = ce[offset + k]
            c[offset + k] = <i64>f
            v[offset + k] = cs[offset + k]
        offset += cl[f]

    return dual_from_coo(rows, cols, vals, nE, nF)


def build_B2_from_cycles(Py_ssize_t nE, cycle_edges, cycle_signs, cycle_lengths):
    """
    Assemble B_2 (nE x nF) from flat cycle arrays. Dispatches on dtype.
    """
    if not isinstance(cycle_edges, np.ndarray):
        cycle_edges = np.asarray(cycle_edges)
    if not isinstance(cycle_signs, np.ndarray):
        cycle_signs = np.asarray(cycle_signs, dtype=np.float64)
    if not isinstance(cycle_lengths, np.ndarray):
        cycle_lengths = np.asarray(cycle_lengths)

    if cycle_edges.dtype == np.int64 or nE >= MAX_INT32_NNZ:
        return build_B2_from_cycles_i64(
            nE,
            cycle_edges.astype(np.int64, copy=False),
            cycle_signs.astype(np.float64, copy=False),
            cycle_lengths.astype(np.int64, copy=False))
    return build_B2_from_cycles_i32(
        nE,
        cycle_edges.astype(np.int32, copy=False),
        cycle_signs.astype(np.float64, copy=False),
        cycle_lengths.astype(np.int32, copy=False))


def build_B2_from_dense(Py_ssize_t nE, Py_ssize_t nF,
                        np.ndarray[f64, ndim=2] matrix):
    """
    Assemble B2 from a dense matrix. Entries rounded to nearest integer.

    Accepts either (nE, nF) or (nF, nE) orientation; if matrix.shape[0] == nF
    and matrix.shape[1] == nE, it is transposed to (nE, nF) before assembly.
    """
    from rexgraph.core._sparse import from_dense_f64

    cdef np.ndarray[f64, ndim=2] M
    if matrix.shape[0] == nF and matrix.shape[1] == nE and nE != nF:
        M = np.ascontiguousarray(matrix.T)
    else:
        M = matrix

    cdef Py_ssize_t i, j
    cdef f64 entry
    for i in range(nE):
        for j in range(nF):
            entry = M[i, j]
            if fabs(entry) < 0.5:
                M[i, j] = 0.0
            elif entry > 0:
                M[i, j] = 1.0
            else:
                M[i, j] = -1.0

    return from_dense_f64(M, tol=0.5)


# Chain complex verification

def verify_chain_complex(B1, B2, double tol=1e-10):
    """
    Verify the chain complex condition B1 B2 = 0.

    Iterates over columns of B2, applies B1 via sparse matvec, and
    checks that the result is zero within tolerance. This avoids
    materializing the nV x nF product matrix.

    Parameters
    ----------
    B1 : DualCSR
        Shape (nV, nE).
    B2 : DualCSR
        Shape (nE, nF).
    tol : float
        Absolute tolerance for the zero check.

    Returns
    -------
    ok : bool
        True if max|B1 B2| < tol.
    max_error : float
        Maximum absolute entry in B1 B2.
    """
    from rexgraph.core._sparse import matvec, col_entries

    cdef Py_ssize_t nF = B2.ncol
    cdef Py_ssize_t nE = B2.nrow
    cdef Py_ssize_t nV = B1.nrow
    cdef double max_err = 0.0
    cdef double entry_abs
    cdef Py_ssize_t f, i

    cdef np.ndarray[f64, ndim=1] col_vec = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] result

    for f in range(nF):
        col_vec[:] = 0.0
        indices, values = col_entries(B2, f)
        for i in range(len(indices)):
            col_vec[indices[i]] = values[i]

        result = matvec(B1, col_vec)

        for i in range(nV):
            entry_abs = fabs(result[i])
            if entry_abs > max_err:
                max_err = entry_abs

    return (max_err < tol), max_err


# Spectral Betti numbers

@cython.boundscheck(False)
@cython.wraparound(False)
def count_zero_eigenvalues(np.ndarray[f64, ndim=1] evals,
                           double tol=1e-10):
    """
    Count eigenvalues below tolerance. Expects sorted ascending input.

    Parameters
    ----------
    evals : f64[k]
        Eigenvalues sorted ascending, already cleaned.
    tol : float
        Threshold below which an eigenvalue is considered zero.

    Returns
    -------
    int
        Number of eigenvalues <= tol.
    """
    cdef f64[::1] ev = evals
    cdef Py_ssize_t n = evals.shape[0]
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t i

    for i in range(n):
        if ev[i] <= tol:
            count += 1
        else:
            break

    return count


def rank_from_eigenvalues(np.ndarray[f64, ndim=1] evals,
                          Py_ssize_t full_dim,
                          double tol=1e-10):
    """
    Compute matrix rank from Laplacian eigenvalues.

    rank(A) = full_dim - count_zero(evals) where L = A^T A or A A^T.

    Parameters
    ----------
    evals : f64[k]
        Eigenvalues of the corresponding Laplacian, sorted ascending.
    full_dim : int
        The row or column dimension of the original operator.
        For rank(B_1) via L_0: full_dim = nV.
        For rank(B_2) via L_2: full_dim = nF.
    tol : float
        Zero threshold.

    Returns
    -------
    int
        Numerical rank.
    """
    cdef Py_ssize_t n_zero = count_zero_eigenvalues(evals, tol)
    return full_dim - n_zero


def betti_from_eigenvalues(evals_L0, evals_L1, evals_L2,
                           Py_ssize_t nV, Py_ssize_t nE,
                           Py_ssize_t nF,
                           double tol=1e-10):
    """
    Compute all Betti numbers from pre-computed Laplacian eigenvalues.

    Runs in O(k) total. No matrix factorization, no SVD, no dense
    conversion.

    beta_k = dim ker(L_k), so each Betti number is just a count
    of near-zero eigenvalues. Cross-checks via the Euler relation.

    Parameters
    ----------
    evals_L0 : f64[] or None
        Eigenvalues of L_0. If None, beta_0 not computed.
    evals_L1 : f64[] or None
        Eigenvalues of L_1. If None, beta_1 not computed.
    evals_L2 : f64[] or None
        Eigenvalues of L_2. If None, beta_2 not computed.
    nV, nE, nF : int
        Dimensions of the chain complex.
    tol : float
        Zero threshold for eigenvalues.

    Returns
    -------
    dict
        Keys: 'beta0', 'beta1', 'beta2', 'rank_B1', 'rank_B2',
              'euler_char', 'euler_check'.
        euler_check is True if beta_0 - beta_1 + beta_2 == nV - nE + nF.
    """
    result = {}

    cdef Py_ssize_t beta0 = -1, beta1 = -1, beta2 = -1
    cdef Py_ssize_t rank_B1 = -1, rank_B2 = -1

    if evals_L0 is not None:
        beta0 = count_zero_eigenvalues(
            np.asarray(evals_L0, dtype=np.float64), tol)
        rank_B1 = nV - beta0
        result['beta0'] = beta0
        result['rank_B1'] = rank_B1

    if evals_L2 is not None and nF > 0:
        beta2 = count_zero_eigenvalues(
            np.asarray(evals_L2, dtype=np.float64), tol)
        rank_B2 = nF - beta2
        result['beta2'] = beta2
        result['rank_B2'] = rank_B2
    elif nF == 0:
        beta2 = 0
        rank_B2 = 0
        result['beta2'] = 0
        result['rank_B2'] = 0

    if evals_L1 is not None:
        beta1 = count_zero_eigenvalues(
            np.asarray(evals_L1, dtype=np.float64), tol)
        result['beta1'] = beta1

    # Euler relation cross-check
    if beta0 >= 0 and beta1 >= 0 and beta2 >= 0:
        chi_topo = nV - nE + nF
        chi_betti = beta0 - beta1 + beta2
        result['euler_char'] = chi_topo
        result['euler_check'] = (chi_topo == chi_betti)

        if rank_B1 >= 0 and rank_B2 >= 0:
            beta1_from_ranks = nE - rank_B1 - rank_B2
            result['beta1_rank_check'] = (beta1 == beta1_from_ranks)

    return result


# Legacy rank computation (SVD fallback)


def compute_rank(M, str method="auto", double tol=1e-10):
    """
    Numerical rank via SVD. Slow for large matrices; prefer
    betti_from_eigenvalues() when Laplacian eigenvalues are available.

    Parameters
    ----------
    M : DualCSR or CSRMatrix
        Sparse matrix whose rank is to be computed.
    method : {"auto", "dense", "sparse"}
    tol : float
        Singular values below tol are treated as zero.

    Returns
    -------
    int
        Numerical rank.
    """
    from rexgraph.core._sparse import to_dense_f64, to_scipy_csr

    cdef Py_ssize_t nrow = M.nrow, ncol = M.ncol
    cdef Py_ssize_t min_dim = min(nrow, ncol)

    if method == "auto":
        # Check both dimension limit AND actual allocation size.
        # A 2000x2M matrix has min_dim=2000 (small) but 32 GB allocation.
        if should_use_dense_eigen(min_dim) and can_allocate_dense_f64(nrow, ncol):
            method = "dense"
        else:
            method = "sparse"

    if method == "dense":
        D = to_dense_f64(M)
        from rexgraph.core._linalg import svd as _lp_svd
        _, s, _ = _lp_svd(np.asarray(D, dtype=np.float64))
        return int(np.sum(s > tol))

    # scipy.sparse.linalg.svds replaced by dense SVD via _linalg
    sp = to_scipy_csr(M)

    if min_dim <= 1:
        from rexgraph.core._linalg import svd as _lp_svd
        _, s, _ = _lp_svd(np.asarray(sp.toarray(), dtype=np.float64))
        return int(np.sum(s > tol))

    cdef Py_ssize_t k = min(min_dim - 1, M.nnz)
    if k <= 0:
        return 0
    from scipy.sparse.linalg import svds
    s = svds(sp.astype(np.float64), k=k, return_singular_vectors=False)
    cdef int count = int(np.sum(s > tol))

    if count == k:
        from rexgraph.core._linalg import svd as _lp_svd
        _, s_full, _ = _lp_svd(np.asarray(sp.toarray(), dtype=np.float64))
        return int(np.sum(s_full > tol))

    return count


def betti_numbers(B1, B2=None, evals_L0=None, evals_L1=None, evals_L2=None):
    """
    Compute Betti numbers of the chain complex.

    Uses the spectral path if eigenvalue arrays are provided,
    otherwise falls back to SVD-based rank computation.

    Parameters
    ----------
    B1 : DualCSR
        Shape (nV, nE).
    B2 : DualCSR or None
        Shape (nE, nF). Omit for a 1-rex.
    evals_L0 : ndarray or None
        Eigenvalues of L_0 (from build_L0).
    evals_L1 : ndarray or None
        Eigenvalues of L_1 (from build_all_laplacians).
    evals_L2 : ndarray or None
        Eigenvalues of L_2 (from build_L2).

    Returns
    -------
    tuple of int
        (beta_0, beta_1) or (beta_0, beta_1, beta_2).
    """
    cdef Py_ssize_t n = B1.nrow, m = B1.ncol
    cdef Py_ssize_t f = B2.ncol if B2 is not None else 0

    # Spectral path
    cdef bint have_L0 = evals_L0 is not None
    cdef bint have_L1 = evals_L1 is not None
    cdef bint have_L2 = evals_L2 is not None or B2 is None

    if have_L0 and have_L1 and have_L2:
        info = betti_from_eigenvalues(
            evals_L0, evals_L1,
            evals_L2 if evals_L2 is not None else None,
            n, m, f)
        b0 = info['beta0']
        b1 = info['beta1']
        if B2 is None:
            return (b0, b1)
        return (b0, b1, info['beta2'])

    # SVD fallback
    if have_L0:
        b0 = count_zero_eigenvalues(
            np.asarray(evals_L0, dtype=np.float64))
        rank_B1 = n - b0
    else:
        import warnings
        warnings.warn(
            "betti_numbers: computing rank(B1) via SVD. "
            "Pass evals_L0 from build_L0() for O(k) spectral path.",
            stacklevel=2)
        rank_B1 = compute_rank(B1)
        b0 = n - rank_B1

    if B2 is None:
        return (b0, m - rank_B1)

    if have_L2:
        b2 = count_zero_eigenvalues(
            np.asarray(evals_L2, dtype=np.float64))
        rank_B2 = f - b2
    else:
        import warnings
        warnings.warn(
            "betti_numbers: computing rank(B2) via SVD. "
            "Pass evals_L2 from build_L2() for O(k) spectral path.",
            stacklevel=2)
        rank_B2 = compute_rank(B2)
        b2 = f - rank_B2

    b1 = m - rank_B1 - rank_B2
    return (b0, b1, b2)
