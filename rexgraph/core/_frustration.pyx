# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._frustration: Frustration Laplacian L_SG.

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
    """K_s = B1^T diag(w) B1 - signed boundary Gramian.

    K_s[i,j] = sum_{v in boundary(i) & boundary(j)} w(v) * B1[v,i] * B1[v,j]

    where B1[v,e] = -1 if v is source of e, +1 if v is target.
    The edge signs array further multiplies each edge's B1 column.

    Diagonal: K_s[i,i] = sum_{v in boundary(i)} w(v) (always positive).
    Off-diagonal: sign depends on boundary orientations at the shared vertex.
    """
    cdef np.ndarray[f64, ndim=2] Ks = np.zeros((nE, nE), dtype=np.float64)
    cdef f64[:, ::1] kv = Ks
    cdef i32[::1] sv = sources, tv = targets
    cdef f64[::1] sgn = signs, wt = vertex_weights
    cdef Py_ssize_t v, j, k, lo, hi
    cdef i32 ei, ej
    cdef f64 wv, bi, bj

    # Build v2e CSR
    cdef i32* vptr = <i32*>malloc((nV + 1) * sizeof(i32))
    cdef i32* vidx = <i32*>malloc(2 * nE * sizeof(i32))
    # Store boundary signs: bsign[idx] = B1[v, e] for the v2e entry
    cdef f64* bsign = <f64*>malloc(2 * nE * sizeof(f64))
    if vptr == NULL or vidx == NULL or bsign == NULL:
        if vptr != NULL: free(vptr)
        if vidx != NULL: free(vidx)
        if bsign != NULL: free(bsign)
        raise MemoryError()

    # Build v2e CSR with boundary signs
    # First pass: count edges per vertex
    cdef Py_ssize_t e
    cdef i32 u
    memset(vptr, 0, (nV + 1) * sizeof(i32))
    for e in range(nE):
        vptr[sv[e] + 1] += 1
        vptr[tv[e] + 1] += 1
    cdef Py_ssize_t i
    for i in range(1, nV + 1):
        vptr[i] += vptr[i - 1]

    # Second pass: fill indices and boundary signs
    cdef i32* pos = <i32*>malloc(nV * sizeof(i32))
    if pos == NULL:
        free(vptr); free(vidx); free(bsign)
        raise MemoryError()
    for i in range(nV):
        pos[i] = vptr[i]

    for e in range(nE):
        # Source vertex: B1[src, e] = -1
        u = sv[e]
        vidx[pos[u]] = <i32>e
        bsign[pos[u]] = -1.0 * sgn[e]
        pos[u] += 1
        # Target vertex: B1[tgt, e] = +1
        u = tv[e]
        vidx[pos[u]] = <i32>e
        bsign[pos[u]] = +1.0 * sgn[e]
        pos[u] += 1

    free(pos)

    # Vertex-driven pair enumeration
    for v in range(nV):
        wv = wt[v]
        lo = vptr[v]
        hi = vptr[v + 1]
        for j in range(lo, hi):
            ei = vidx[j]
            # Diagonal: always positive (B1[v,e]^2 = 1)
            kv[ei, ei] += wv
            # Off-diagonal pairs: use boundary signs
            bi = bsign[j]
            for k in range(j + 1, hi):
                ej = vidx[k]
                bj = bsign[k]
                kv[ei, ej] += wv * bi * bj
                kv[ej, ei] += wv * bi * bj

    free(vptr)
    free(vidx)
    free(bsign)
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
    """REAL sparse frustration Laplacian - K_s = Bs^T W Bs assembled as a SPARSE matmul
    (Bs the sign-scaled signed incidence: Bs[v,e] = B1[v,e]·sign(e) = -sign(e) at the
    source, +sign(e) at the target), so K_s has the O(Σ deg²) line-graph sparsity and is
    never the dense nE×nE Gramian. Then L_SG = D_{|K_off|} − K_off with K_off = K_s
    off-diagonal. Returns scipy CSR. Equals build_L_SG_dense exactly."""
    import scipy.sparse as _sp
    src = np.asarray(sources, dtype=np.int64)
    tgt = np.asarray(targets, dtype=np.int64)
    sgn = np.asarray(signs, dtype=np.float64)
    w = np.asarray(vertex_weights, dtype=np.float64)
    eidx = np.arange(nE, dtype=np.int64)
    Bs = _sp.csr_matrix(
        (np.concatenate([-sgn, sgn]),
         (np.concatenate([src, tgt]), np.concatenate([eidx, eidx]))),
        shape=(nV, nE))                                  # signed, sign-scaled incidence
    Ks = (Bs.T @ (_sp.diags(w) @ Bs)).tocsr()            # nE × nE, sparse matmul
    Koff = (Ks - _sp.diags(Ks.diagonal())).tocsr()       # zero the diagonal
    D = np.asarray(np.abs(Koff).sum(axis=1), dtype=np.float64).ravel()
    return (_sp.diags(D) - Koff).tocsr()


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
