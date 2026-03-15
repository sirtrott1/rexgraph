# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._frustration - Frustration Laplacian L_SG.

Signed weighted Gramian K_s = B1^T W B1 where W = diag(1/log(deg+e)).
Frustration Laplacian L_SG = D_{|K_off|} - K_off, where K_off is K_s
with diagonal zeroed.

Same vertex-driven pair enumeration as _overlap.pyx but with different
weights. Dense and sparse construction paths with adaptive selection.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    can_allocate_dense_f64,
    should_use_dense_matmul,
    get_EPSILON_DIV,
)

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport log, fabs, sqrt, exp

np.import_array()


# Vertex weights: w(v) = 1 / log(deg(v) + e)

@cython.boundscheck(False)
@cython.wraparound(False)
def build_vertex_weights(Py_ssize_t nV, Py_ssize_t nE,
                         np.ndarray[i32, ndim=1] sources,
                         np.ndarray[i32, ndim=1] targets):
    """Inverse-log-degree vertex weights for the signed Gramian."""
    cdef np.ndarray[f64, ndim=1] deg = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] w = np.empty(nV, dtype=np.float64)
    cdef i32[::1] sv = sources, tv = targets
    cdef f64[::1] dv = deg, wv = w
    cdef Py_ssize_t e, v
    cdef f64 d

    for e in range(nE):
        dv[sv[e]] += 1.0
        dv[tv[e]] += 1.0

    cdef f64 E_VAL = exp(1.0)
    for v in range(nV):
        d = dv[v]
        wv[v] = 1.0 / log(d + E_VAL) if d + E_VAL > 1.0 else 1.0

    return w


@cython.boundscheck(False)
@cython.wraparound(False)
def build_vertex_weights_i64(Py_ssize_t nV, Py_ssize_t nE,
                              np.ndarray[i64, ndim=1] sources,
                              np.ndarray[i64, ndim=1] targets):
    """int64 variant."""
    cdef np.ndarray[f64, ndim=1] deg = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] w = np.empty(nV, dtype=np.float64)
    cdef i64[::1] sv = sources, tv = targets
    cdef f64[::1] dv = deg, wv = w
    cdef Py_ssize_t e, v
    cdef f64 d, E_VAL = exp(1.0)

    for e in range(nE):
        dv[sv[e]] += 1.0
        dv[tv[e]] += 1.0

    for v in range(nV):
        d = dv[v]
        wv[v] = 1.0 / log(d + E_VAL) if d + E_VAL > 1.0 else 1.0

    return w


# Dense signed Gramian

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _build_v2e_csr_i32(
    const i32[::1] sources,
    const i32[::1] targets,
    Py_ssize_t nV, Py_ssize_t nE,
    i32* vptr, i32* vidx
) noexcept nogil:
    """Build vertex-to-edge CSR. Caller allocates vptr[nV+1], vidx[2*nE]."""
    cdef Py_ssize_t e
    cdef i32 u, v

    memset(vptr, 0, (nV + 1) * sizeof(i32))
    for e in range(nE):
        vptr[sources[e] + 1] += 1
        vptr[targets[e] + 1] += 1

    cdef Py_ssize_t i
    for i in range(1, nV + 1):
        vptr[i] += vptr[i - 1]

    cdef i32* pos = <i32*>malloc(nV * sizeof(i32))
    if pos == NULL:
        return -1
    for i in range(nV):
        pos[i] = vptr[i]

    for e in range(nE):
        u = sources[e]
        vidx[pos[u]] = <i32>e
        pos[u] += 1
        v = targets[e]
        vidx[pos[v]] = <i32>e
        pos[v] += 1

    free(pos)
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def build_signed_gramian_dense(Py_ssize_t nV, Py_ssize_t nE,
                                np.ndarray[i32, ndim=1] sources,
                                np.ndarray[i32, ndim=1] targets,
                                np.ndarray[f64, ndim=1] signs,
                                np.ndarray[f64, ndim=1] vertex_weights):
    """K_s = B1^T diag(w) B1 with edge signs applied.

    K_s[i,j] = sum_{v in boundary(i) & boundary(j)} w(v) * sign(i) * sign(j)
    Diagonal: K_s[i,i] = sum_{v in boundary(i)} w(v)
    """
    cdef np.ndarray[f64, ndim=2] Ks = np.zeros((nE, nE), dtype=np.float64)
    cdef f64[:, ::1] kv = Ks
    cdef i32[::1] sv = sources, tv = targets
    cdef f64[::1] sgn = signs, wt = vertex_weights
    cdef Py_ssize_t v, j, k, lo, hi
    cdef i32 ei, ej
    cdef f64 wv

    # Build v2e CSR
    cdef i32* vptr = <i32*>malloc((nV + 1) * sizeof(i32))
    cdef i32* vidx = <i32*>malloc(2 * nE * sizeof(i32))
    if vptr == NULL or vidx == NULL:
        if vptr != NULL: free(vptr)
        if vidx != NULL: free(vidx)
        raise MemoryError()

    _build_v2e_csr_i32(sources, targets, nV, nE, vptr, vidx)

    # Vertex-driven pair enumeration
    for v in range(nV):
        wv = wt[v]
        lo = vptr[v]
        hi = vptr[v + 1]
        for j in range(lo, hi):
            ei = vidx[j]
            # Diagonal
            kv[ei, ei] += wv
            # Off-diagonal pairs
            for k in range(j + 1, hi):
                ej = vidx[k]
                kv[ei, ej] += wv * sgn[ei] * sgn[ej]
                kv[ej, ei] += wv * sgn[ei] * sgn[ej]

    free(vptr)
    free(vidx)
    return Ks


@cython.boundscheck(False)
@cython.wraparound(False)
def build_L_SG_dense(Py_ssize_t nV, Py_ssize_t nE,
                      np.ndarray[i32, ndim=1] sources,
                      np.ndarray[i32, ndim=1] targets,
                      np.ndarray[f64, ndim=1] signs,
                      np.ndarray[f64, ndim=1] vertex_weights):
    """Frustration Laplacian L_SG = D_{|K_off|} - K_off.

    K_off = K_s with diagonal zeroed.
    D is the diagonal of row sums of |K_off|.
    """
    cdef np.ndarray[f64, ndim=2] Ks = build_signed_gramian_dense(
        nV, nE, sources, targets, signs, vertex_weights)
    cdef f64[:, ::1] kv = Ks
    cdef Py_ssize_t i, j
    cdef f64 row_sum

    # Zero diagonal, compute D
    cdef np.ndarray[f64, ndim=2] L = np.zeros((nE, nE), dtype=np.float64)
    cdef f64[:, ::1] lv = L

    for i in range(nE):
        row_sum = 0.0
        for j in range(nE):
            if i != j:
                lv[i, j] = -kv[i, j]
                row_sum += fabs(kv[i, j])
        lv[i, i] = row_sum

    return L


def build_L_SG_sparse(Py_ssize_t nV, Py_ssize_t nE, sources, targets, signs, vertex_weights):
    """Sparse path: construct dense then caller converts if needed."""
    return build_L_SG_dense(nV, nE,
                             np.asarray(sources, dtype=np.int32),
                             np.asarray(targets, dtype=np.int32),
                             np.asarray(signs, dtype=np.float64),
                             np.asarray(vertex_weights, dtype=np.float64))


def build_L_SG(Py_ssize_t nV, Py_ssize_t nE, sources, targets,
               signs=None, method="auto"):
    """Frustration Laplacian L_SG.

    Parameters
    ----------
    nV, nE : int
    sources, targets : int array[nE]
    signs : float array[nE], optional
        Edge signs (+1/-1). Default: all +1.
    method : str
        "auto", "dense", or "sparse".

    Returns
    -------
    ndarray or scipy CSR
    """
    src = np.asarray(sources, dtype=np.int32)
    tgt = np.asarray(targets, dtype=np.int32)

    if signs is None:
        sgn = np.ones(nE, dtype=np.float64)
    else:
        sgn = np.asarray(signs, dtype=np.float64)

    wt = build_vertex_weights(nV, nE, src, tgt)

    if method == "auto":
        method = "dense"  # sparse handled by caller

    if method == "dense":
        return build_L_SG_dense(nV, nE, src, tgt, sgn, wt)
    else:
        return build_L_SG_sparse(nV, nE, src, tgt, sgn, wt)


def frustration_rate(np.ndarray[f64, ndim=1] signs,
                     np.ndarray[i32, ndim=1] edge_types,
                     Py_ssize_t nE, Py_ssize_t n_types):
    """Fraction of negative-signed edges per type."""
    cdef np.ndarray[f64, ndim=1] rates = np.zeros(n_types, dtype=np.float64)
    cdef np.ndarray[i32, ndim=1] total = np.zeros(n_types, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] neg = np.zeros(n_types, dtype=np.int32)
    cdef i32[::1] tp = edge_types, tot = total, ng = neg
    cdef f64[::1] sg = signs, rt = rates
    cdef Py_ssize_t e
    cdef i32 t

    for e in range(nE):
        t = tp[e]
        tot[t] += 1
        if sg[e] < 0:
            ng[t] += 1

    for t in range(n_types):
        if tot[t] > 0:
            rt[t] = <f64>ng[t] / <f64>tot[t]

    return rates
