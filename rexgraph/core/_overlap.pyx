# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._overlap - Overlap Laplacian L_O on the edge set.

Builds L_O in R^{m x m}, capturing geometric similarity between edges.
Two edges are similar when they share boundary vertices. The similarity
is normalized by overlap degree (total shared-vertex count per edge),
which guarantees L_O is PSD with eigenvalues in [0, 1].

The formula is:

    K = |B_1|^T W |B_1|               unsigned Gramian
    d_ov = rowsums(K)                  overlap degree
    S = diag(d_ov)^{-1/2} K diag(d_ov)^{-1/2}   normalized similarity
    L_O = I - S                        overlap Laplacian

K_ij counts the weighted number of vertices shared by edges i and j.
W = diag(w_v) holds optional per-vertex weights (default: uniform).

L_O is PSD with eigenvalues in [0, 1] because D_ov - K is diagonally
dominant (nonneg diagonal, nonneg off-diagonal), so K <= D_ov in the
Loewner order, giving S <= I after congruence by D_ov^{-1/2}.

The Fiedler value of L_O enters the coupling constant
alpha_G = fiedler(L_1) / fiedler(L_O) of the Relational Laplacian.

Algorithm: vertex-driven pair enumeration in O(sum deg^2) time. For
each vertex v with degree d, the d^2 pairs of incident edges all share
v and contribute w_v to K. COO triples are accumulated then summed into
CSR via scipy. Normalization is O(nnz(K)).

Dense path (nE <= dense_limit): returns ndarray[nE, nE].
Sparse path (nE > dense_limit): returns scipy CSR.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,
    can_allocate_dense_f64,
    should_use_dense_matmul,
    get_EPSILON_DIV,
)

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport sqrt

np.import_array()

try:
    from scipy.sparse import eye as _speye, diags as _spdiags, triu as _sptriu, coo_matrix as _coo_matrix
    _HAS_SCIPY_SPARSE = True
except ImportError:
    _HAS_SCIPY_SPARSE = False


# Vertex-to-edge CSR

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _build_v2e_csr(
    const i32[::1] sources,
    const i32[::1] targets,
    Py_ssize_t nV,
    Py_ssize_t nE,
    i32* vptr,
    i32* vidx,
) noexcept nogil:
    """Build vertex-to-edge CSR index.

    For each vertex v, vptr[v]..vptr[v+1] lists edge indices incident
    to v. Caller allocates vptr[nV+1] and vidx[2*nE].

    Returns 0 on success, -1 on allocation failure.
    """
    cdef Py_ssize_t i
    cdef i32 u, v_node

    memset(vptr, 0, (nV + 1) * sizeof(i32))
    for i in range(nE):
        vptr[sources[i] + 1] += 1
        vptr[targets[i] + 1] += 1

    # Prefix sum
    for i in range(nV):
        vptr[i + 1] += vptr[i]

    # Fill edge indices; cursor tracks insert position per vertex
    cdef i32* cursor = <i32*>malloc(nV * sizeof(i32))
    if cursor == NULL:
        return -1

    for i in range(nV):
        cursor[i] = vptr[i]

    for i in range(nE):
        u = sources[i]
        vidx[cursor[u]] = <i32>i
        cursor[u] += 1

        v_node = targets[i]
        vidx[cursor[v_node]] = <i32>i
        cursor[v_node] += 1

    free(cursor)
    return 0


# Gramian K = |B_1|^T W |B_1| via vertex-driven enumeration

@cython.boundscheck(False)
@cython.wraparound(False)
cdef i64 _count_gramian_nnz(
    const i32* vptr,
    Py_ssize_t nV,
) noexcept nogil:
    """Count total COO entries for the Gramian (including duplicates).

    Each vertex v with degree d contributes d^2 entries. Duplicates at
    the same (i, j) are summed during COO-to-CSR conversion.
    """
    cdef Py_ssize_t i
    cdef i64 total = 0
    cdef i32 deg

    for i in range(nV):
        deg = vptr[i + 1] - vptr[i]
        total += <i64>deg * <i64>deg

    return total


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_gramian_coo(
    const i32* vptr,
    const i32* vidx,
    const f64* vertex_weights,
    Py_ssize_t nV,
    i32* rows,
    i32* cols,
    f64* vals,
) noexcept nogil:
    """Fill COO triples for K = |B_1|^T W |B_1|.

    For each vertex v, all d^2 pairs (e_i, e_j) of incident edges
    contribute vertex_weights[v] to K[e_i, e_j]. When edges share
    multiple vertices, duplicate (i, j) entries arise and are summed
    during COO-to-CSR conversion.
    """
    cdef Py_ssize_t v, j, k
    cdef i64 ptr = 0
    cdef i32 e1, e2
    cdef f64 w

    for v in range(nV):
        w = vertex_weights[v]
        if w <= 0.0:
            continue

        for j in range(vptr[v], vptr[v + 1]):
            e1 = vidx[j]
            for k in range(vptr[v], vptr[v + 1]):
                e2 = vidx[k]
                rows[ptr] = e1
                cols[ptr] = e2
                vals[ptr] = w
                ptr += 1


# Dense Gramian fill (avoids COO allocation for small graphs)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_gramian_dense(
    const i32* vptr,
    const i32* vidx,
    const f64* vertex_weights,
    Py_ssize_t nV,
    f64* K,
    Py_ssize_t nE,
) noexcept nogil:
    """Fill dense K[nE, nE] via vertex-driven enumeration.

    Row-major layout: K[e1, e2] at K[e1 * nE + e2].
    """
    cdef Py_ssize_t v, j, k
    cdef i32 e1, e2
    cdef f64 w

    for v in range(nV):
        w = vertex_weights[v]
        if w <= 0.0:
            continue

        for j in range(vptr[v], vptr[v + 1]):
            e1 = vidx[j]
            for k in range(vptr[v], vptr[v + 1]):
                e2 = vidx[k]
                K[e1 * nE + e2] += w


# Overlap-degree normalization: L_O = I - D_ov^{-1/2} K D_ov^{-1/2}

def _normalize_sparse(K_csr, Py_ssize_t nE):
    """Apply overlap-degree normalization to sparse K.

    Returns (L_O, S, d_ov) where L_O and S are CSR, d_ov is ndarray.
    """
    speye, spdiags = _speye, _spdiags

    cdef np.ndarray[f64, ndim=1] d_ov = np.asarray(K_csr.sum(axis=1)).ravel()

    cdef f64 eps = get_EPSILON_DIV()
    cdef np.ndarray[f64, ndim=1] inv_sqrt_d = np.empty(nE, dtype=np.float64)
    cdef f64[::1] iv = inv_sqrt_d
    cdef f64[::1] dv = d_ov
    cdef Py_ssize_t i

    for i in range(nE):
        if dv[i] > eps:
            iv[i] = 1.0 / sqrt(dv[i])
        else:
            iv[i] = 0.0

    D_inv_sqrt = spdiags(inv_sqrt_d)
    S = D_inv_sqrt @ K_csr @ D_inv_sqrt
    L_O = speye(nE, format='csr') - S

    return L_O, S, d_ov


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _normalize_dense(
    const f64* K,
    f64* L_O,
    f64* S,
    f64* d_ov,
    Py_ssize_t nE,
) noexcept nogil:
    """Apply overlap-degree normalization to dense K.

    Reads K, writes L_O, S, and d_ov. Row-major layout.
    """
    cdef Py_ssize_t i, j
    cdef f64 eps = 1e-12
    cdef f64 inv_i, inv_j, s_val

    # Row sums -> overlap degree
    for i in range(nE):
        d_ov[i] = 0.0
        for j in range(nE):
            d_ov[i] += K[i * nE + j]

    # S_ij = K_ij / sqrt(d_i * d_j); L_O = I - S
    for i in range(nE):
        inv_i = 1.0 / sqrt(d_ov[i]) if d_ov[i] > eps else 0.0

        for j in range(nE):
            inv_j = 1.0 / sqrt(d_ov[j]) if d_ov[j] > eps else 0.0

            s_val = K[i * nE + j] * inv_i * inv_j
            S[i * nE + j] = s_val

            if i == j:
                L_O[i * nE + j] = 1.0 - s_val
            else:
                L_O[i * nE + j] = -s_val


# Public API

def build_L_O(
    Py_ssize_t nV,
    Py_ssize_t nE,
    sources,
    targets,
    str method="auto",
    vertex_weights=None,
):
    """Overlap Laplacian L_O = I - D_ov^{-1/2} K D_ov^{-1/2}.

    Builds the unsigned Gramian K = |B_1|^T W |B_1| via vertex-driven
    pair enumeration, then normalizes by overlap degree (row sums of K).

    Parameters
    ----------
    nV, nE : int
        Vertex and edge counts.
    sources, targets : array-like of int32
        Edge endpoint arrays (length nE).
    method : {"auto", "dense", "sparse"}
        "auto" picks dense when nE^2 fits the dense allocation budget.
    vertex_weights : ndarray[nV] of float64, optional
        Per-vertex weights for K = |B_1|^T W |B_1|. Default: uniform.

    Returns
    -------
    L_O : ndarray[nE, nE] or scipy.sparse.csr_matrix
        Overlap Laplacian. Symmetric PSD, eigenvalues in [0, 1].
    """
    if not isinstance(sources, np.ndarray):
        sources = np.asarray(sources, dtype=np.int32)
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets, dtype=np.int32)

    cdef i32[::1] s = sources.astype(np.int32, copy=False)
    cdef i32[::1] t = targets.astype(np.int32, copy=False)

    cdef np.ndarray[f64, ndim=1] W
    if vertex_weights is not None:
        W = np.ascontiguousarray(vertex_weights, dtype=np.float64)
    else:
        W = np.ones(nV, dtype=np.float64)
    cdef f64[::1] w_view = W

    if method == "auto":
        method = "dense" if should_use_dense_matmul(nE) else "sparse"

    # Vertex-to-edge CSR
    cdef np.ndarray[i32, ndim=1] vptr_arr = np.empty(nV + 1, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] vidx_arr = np.empty(2 * nE, dtype=np.int32)

    cdef int rc
    with nogil:
        rc = _build_v2e_csr(s, t, nV, nE, &vptr_arr[0], &vidx_arr[0])
    if rc != 0:
        raise MemoryError("v2e CSR cursor allocation failed")

    if method == "dense":
        return _build_dense(nV, nE, vptr_arr, vidx_arr, W)

    return _build_sparse(nV, nE, vptr_arr, vidx_arr, W)


cdef object _build_dense(
    Py_ssize_t nV,
    Py_ssize_t nE,
    np.ndarray[i32, ndim=1] vptr_arr,
    np.ndarray[i32, ndim=1] vidx_arr,
    np.ndarray[f64, ndim=1] W,
):
    """Dense path: fill K in-place, normalize to L_O."""
    cdef np.ndarray[f64, ndim=2] K = np.zeros((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] L_O = np.empty((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] S = np.empty((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] d_ov = np.empty(nE, dtype=np.float64)

    with nogil:
        _fill_gramian_dense(
            &vptr_arr[0], &vidx_arr[0], &W[0], nV, &K[0, 0], nE,
        )
        _normalize_dense(&K[0, 0], &L_O[0, 0], &S[0, 0], &d_ov[0], nE)

    return L_O


cdef object _build_sparse(
    Py_ssize_t nV,
    Py_ssize_t nE,
    np.ndarray[i32, ndim=1] vptr_arr,
    np.ndarray[i32, ndim=1] vidx_arr,
    np.ndarray[f64, ndim=1] W,
):
    """Sparse path: COO -> CSR -> normalized Laplacian."""
    cdef i64 total_pairs
    with nogil:
        total_pairs = _count_gramian_nnz(&vptr_arr[0], nV)

    cdef np.ndarray[i32, ndim=1] coo_rows = np.empty(total_pairs, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] coo_cols = np.empty(total_pairs, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] coo_vals = np.empty(total_pairs, dtype=np.float64)

    with nogil:
        _fill_gramian_coo(
            &vptr_arr[0], &vidx_arr[0], &W[0], nV,
            &coo_rows[0], &coo_cols[0], &coo_vals[0],
        )

    coo_matrix = _coo_matrix
    K = coo_matrix(
        (coo_vals, (coo_rows, coo_cols)),
        shape=(nE, nE),
    ).tocsr()

    L_O, S, d_ov = _normalize_sparse(K, nE)
    return L_O


# Overlap adjacency and top-k pairs

def build_overlap_adjacency(
    Py_ssize_t nV,
    Py_ssize_t nE,
    sources,
    targets,
    vertex_weights=None,
):
    """Return overlap similarity S and overlap degree d_ov (dense).

    Used by _laplacians for the Relational Laplacian
    RL_1 = L_1 + alpha_G * L_O.

    Parameters
    ----------
    nV, nE : int
        Vertex and edge counts.
    sources, targets : array-like of int32
        Edge endpoint arrays.
    vertex_weights : ndarray[nV] of float64, optional
        Per-vertex weights. Default: uniform.

    Returns
    -------
    S : ndarray[nE, nE], float64
        Overlap similarity (symmetric, nonneg, entries in [0, 1]).
    d_ov : ndarray[nE], float64
        Overlap degree (row sums of the Gramian K).
    """
    if not isinstance(sources, np.ndarray):
        sources = np.asarray(sources, dtype=np.int32)
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets, dtype=np.int32)

    cdef i32[::1] s = sources.astype(np.int32, copy=False)
    cdef i32[::1] t = targets.astype(np.int32, copy=False)

    cdef np.ndarray[f64, ndim=1] W
    if vertex_weights is not None:
        W = np.ascontiguousarray(vertex_weights, dtype=np.float64)
    else:
        W = np.ones(nV, dtype=np.float64)

    # Vertex-to-edge CSR
    cdef np.ndarray[i32, ndim=1] vptr_arr = np.empty(nV + 1, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] vidx_arr = np.empty(2 * nE, dtype=np.int32)

    cdef int rc
    with nogil:
        rc = _build_v2e_csr(s, t, nV, nE, &vptr_arr[0], &vidx_arr[0])
    if rc != 0:
        raise MemoryError("v2e CSR cursor allocation failed")

    cdef np.ndarray[f64, ndim=2] K = np.zeros((nE, nE), dtype=np.float64)
    with nogil:
        _fill_gramian_dense(
            &vptr_arr[0], &vidx_arr[0], &W[0], nV, &K[0, 0], nE,
        )

    cdef np.ndarray[f64, ndim=2] S_arr = np.empty((nE, nE), dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] d_ov = np.empty(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] L_O_tmp = np.empty((nE, nE), dtype=np.float64)

    with nogil:
        _normalize_dense(&K[0, 0], &L_O_tmp[0, 0], &S_arr[0, 0], &d_ov[0], nE)

    return S_arr, d_ov


def build_overlap_pairs(
    Py_ssize_t nV,
    Py_ssize_t nE,
    sources,
    targets,
    Py_ssize_t topk=30,
    vertex_weights=None,
):
    """Top-k most similar edge pairs by overlap similarity.

    Parameters
    ----------
    nV, nE : int
        Vertex and edge counts.
    sources, targets : array-like of int32
        Edge endpoint arrays.
    topk : int
        Maximum pairs to return.
    vertex_weights : ndarray[nV] of float64, optional
        Per-vertex weights. Default: uniform.

    Returns
    -------
    list of dict
        Each dict: {edge_i, edge_j, similarity, shared}.
        Sorted by descending similarity.
    """
    speye, triu = _speye, _sptriu

    L_O = build_L_O(nV, nE, sources, targets,
                     method="sparse", vertex_weights=vertex_weights)
    S = speye(nE, format='csr') - L_O
    S_upper = triu(S, k=1)

    data = S_upper.data
    rows_arr, cols_arr = S_upper.nonzero()

    if len(data) == 0:
        return []

    indices = np.argsort(-data)
    if len(indices) > topk:
        indices = indices[:topk]

    cdef np.ndarray[i32, ndim=1] s_arr = np.asarray(sources, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] t_arr = np.asarray(targets, dtype=np.int32)

    result = []
    for idx in indices:
        ei = int(rows_arr[idx])
        ej = int(cols_arr[idx])

        verts_i = {int(s_arr[ei]), int(t_arr[ei])}
        verts_j = {int(s_arr[ej]), int(t_arr[ej])}
        shared = len(verts_i & verts_j)

        result.append({
            "edge_i": ei,
            "edge_j": ej,
            "similarity": float(data[idx]),
            "shared": shared,
        })

    return result
