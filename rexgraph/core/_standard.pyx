# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._standard - Classical graph algorithms on the 1-skeleton.

Operates on the undirected graph underlying a relational complex.
Input is a symmetric CSR adjacency from _cycles.build_symmetric_adjacency.

PageRank - power iteration, O(nE) per step.
Betweenness - vertex and edge betweenness centrality, O(nV * nE).
Clustering - local clustering coefficient via sorted neighbor intersection.
Louvain - modularity-based community detection, O(nE) per pass.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,
)

from libc.math cimport fabs, sqrt

np.import_array()


# PageRank

@cython.boundscheck(False)
@cython.wraparound(False)
def pagerank_i32(np.ndarray[i32, ndim=1] adj_ptr,
                 np.ndarray[i32, ndim=1] adj_idx,
                 np.ndarray[f64, ndim=1] adj_wt,
                 Py_ssize_t nV,
                 Py_ssize_t nE,
                 double damping=0.85,
                 int max_iter=100,
                 double tol=1e-8):
    """
    PageRank via power iteration on the undirected 1-skeleton.

    Weighted damped random walk. Converges when L1 change < tol.

    Parameters
    ----------
    adj_ptr : i32[nV+1]
        CSR row pointers of the symmetric adjacency.
    adj_idx : i32[2*nE]
        Neighbor vertex indices.
    adj_wt : f64[2*nE]
        Edge weights for each adjacency entry.
    nV, nE : int
    damping : float, default 0.85
    max_iter : int, default 100
    tol : float, default 1e-8

    Returns
    -------
    f64[nV]
        PageRank scores summing to 1.0.
    """
    if nV == 0:
        return np.empty(0, dtype=np.float64)

    cdef i32[::1] ap = adj_ptr, ai = adj_idx
    cdef f64[::1] aw = adj_wt

    cdef np.ndarray[f64, ndim=1] inv_wdeg_arr = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] inv_wdeg = inv_wdeg_arr
    cdef Py_ssize_t v, k
    cdef double wdeg

    for v in range(nV):
        wdeg = 0.0
        for k in range(ap[v], ap[v + 1]):
            wdeg += aw[k]
        if wdeg > 1e-15:
            inv_wdeg[v] = 1.0 / wdeg

    cdef np.ndarray[f64, ndim=1] r0 = np.full(nV, 1.0 / <double>nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] r1 = np.empty(nV, dtype=np.float64)
    cdef f64[::1] rc = r0, rn = r1
    cdef f64[::1] tmp

    cdef double base = (1.0 - damping) / <double>nV
    cdef double diff, contrib
    cdef int it
    cdef Py_ssize_t u

    for it in range(max_iter):
        diff = 0.0
        for v in range(nV):
            contrib = 0.0
            for k in range(ap[v], ap[v + 1]):
                u = ai[k]
                contrib += rc[u] * aw[k] * inv_wdeg[u]
            rn[v] = base + damping * contrib
            diff += fabs(rn[v] - rc[v])

        tmp = rc; rc = rn; rn = tmp

        if diff < tol:
            break

    if rc is r0:
        return r0
    return r1


@cython.boundscheck(False)
@cython.wraparound(False)
def pagerank_i64(np.ndarray[i64, ndim=1] adj_ptr,
                 np.ndarray[i64, ndim=1] adj_idx,
                 np.ndarray[f64, ndim=1] adj_wt,
                 Py_ssize_t nV,
                 Py_ssize_t nE,
                 double damping=0.85,
                 int max_iter=100,
                 double tol=1e-8):
    """PageRank via power iteration. int64 index variant."""
    if nV == 0:
        return np.empty(0, dtype=np.float64)

    cdef i64[::1] ap = adj_ptr, ai = adj_idx
    cdef f64[::1] aw = adj_wt

    cdef np.ndarray[f64, ndim=1] inv_wdeg_arr = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] inv_wdeg = inv_wdeg_arr
    cdef Py_ssize_t v, k
    cdef double wdeg

    for v in range(nV):
        wdeg = 0.0
        for k in range(ap[v], ap[v + 1]):
            wdeg += aw[k]
        if wdeg > 1e-15:
            inv_wdeg[v] = 1.0 / wdeg

    cdef np.ndarray[f64, ndim=1] r0 = np.full(nV, 1.0 / <double>nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] r1 = np.empty(nV, dtype=np.float64)
    cdef f64[::1] rc = r0, rn = r1
    cdef f64[::1] tmp

    cdef double base = (1.0 - damping) / <double>nV
    cdef double diff, contrib
    cdef int it
    cdef Py_ssize_t u

    for it in range(max_iter):
        diff = 0.0
        for v in range(nV):
            contrib = 0.0
            for k in range(ap[v], ap[v + 1]):
                u = <Py_ssize_t>ai[k]
                contrib += rc[u] * aw[k] * inv_wdeg[u]
            rn[v] = base + damping * contrib
            diff += fabs(rn[v] - rc[v])

        tmp = rc; rc = rn; rn = tmp
        if diff < tol:
            break

    if rc is r0:
        return r0
    return r1


def pagerank(adj_ptr, adj_idx, adj_wt, Py_ssize_t nV, Py_ssize_t nE,
             double damping=0.85, int max_iter=100, double tol=1e-8):
    """Dispatch PageRank by index type."""
    if adj_ptr.dtype == np.int64:
        return pagerank_i64(adj_ptr, adj_idx, adj_wt, nV, nE,
                            damping, max_iter, tol)
    return pagerank_i32(adj_ptr, adj_idx, adj_wt, nV, nE,
                        damping, max_iter, tol)


# Betweenness centrality

@cython.boundscheck(False)
@cython.wraparound(False)
def betweenness_i32(np.ndarray[i32, ndim=1] adj_ptr,
                    np.ndarray[i32, ndim=1] adj_idx,
                    np.ndarray[i32, ndim=1] adj_edge,
                    Py_ssize_t nV,
                    Py_ssize_t nE,
                    Py_ssize_t max_sources=0):
    """
    Vertex and edge betweenness centrality via BFS dependency
    accumulation. Vertex betweenness normalized by
    (nV-1)(nV-2)/2; edge betweenness by nV(nV-1)/2.

    Parameters
    ----------
    adj_ptr : i32[nV+1]
    adj_idx : i32[2*nE]
    adj_edge : i32[2*nE]
        Maps each adjacency entry to the original edge index.
    nV, nE : int
    max_sources : int, default 0
        If > 0, sample this many source vertices and rescale.
        0 means use all vertices (exact computation).

    Returns
    -------
    bc_v : f64[nV]
        Normalized vertex betweenness.
    bc_e : f64[nE]
        Normalized edge betweenness.
    """
    if nV <= 1:
        return (np.zeros(nV, dtype=np.float64),
                np.zeros(nE, dtype=np.float64))

    cdef i32[::1] ap = adj_ptr, ai = adj_idx, ae = adj_edge

    cdef np.ndarray[f64, ndim=1] bc_v_arr = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] bc_e_arr = np.zeros(nE, dtype=np.float64)
    cdef f64[::1] bc_v = bc_v_arr, bc_e = bc_e_arr

    cdef np.ndarray[i32, ndim=1] dist_arr = np.full(nV, -1, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] sigma_arr = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] delta_arr = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[i32, ndim=1] queue_arr = np.empty(nV, dtype=np.int32)
    cdef i32[::1] dist = dist_arr
    cdef f64[::1] sigma = sigma_arr, delta = delta_arr
    cdef i32[::1] queue = queue_arr

    cdef Py_ssize_t n_sources = nV
    if max_sources > 0 and max_sources < nV:
        n_sources = max_sources

    cdef Py_ssize_t s, v, w, k, qi, q_len
    cdef double coeff

    for s in range(n_sources):
        if s == 0:
            for v in range(nV):
                dist[v] = -1
                sigma[v] = 0.0
                delta[v] = 0.0
        else:
            for qi in range(q_len):
                v = queue[qi]
                dist[v] = -1
                sigma[v] = 0.0
                delta[v] = 0.0

        dist[s] = 0
        sigma[s] = 1.0
        queue[0] = <i32>s
        q_len = 1
        qi = 0

        while qi < q_len:
            v = queue[qi]; qi += 1
            for k in range(ap[v], ap[v + 1]):
                w = ai[k]
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    sigma[w] = sigma[v]
                    queue[q_len] = <i32>w; q_len += 1
                elif dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]

        for qi in range(q_len - 1, 0, -1):
            w = queue[qi]
            if sigma[w] < 1e-15:
                continue
            for k in range(ap[w], ap[w + 1]):
                v = ai[k]
                if dist[v] == dist[w] - 1:
                    coeff = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    delta[v] += coeff
                    bc_e[ae[k]] += coeff
            if w != s:
                bc_v[w] += delta[w]

    cdef double scale_v, scale_e, src_scale
    scale_v = (<double>nV - 1.0) * (<double>nV - 2.0)
    if scale_v > 0.0:
        scale_v = 1.0 / scale_v
    else:
        scale_v = 0.0

    scale_e = <double>nV * (<double>nV - 1.0)
    if scale_e > 0.0:
        scale_e = 1.0 / scale_e
    else:
        scale_e = 0.0

    if n_sources < nV:
        src_scale = <double>nV / <double>n_sources
    else:
        src_scale = 1.0

    for v in range(nV):
        bc_v[v] *= scale_v * src_scale
    for k in range(<Py_ssize_t>nE):
        # Halve: each edge counted in both directions
        bc_e[k] *= 0.5 * scale_e * src_scale

    return bc_v_arr, bc_e_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def betweenness_i64(np.ndarray[i64, ndim=1] adj_ptr,
                    np.ndarray[i64, ndim=1] adj_idx,
                    np.ndarray[i64, ndim=1] adj_edge,
                    Py_ssize_t nV,
                    Py_ssize_t nE,
                    Py_ssize_t max_sources=0):
    """Vertex and edge betweenness centrality. int64 index variant."""
    if nV <= 1:
        return (np.zeros(nV, dtype=np.float64),
                np.zeros(nE, dtype=np.float64))

    cdef i64[::1] ap = adj_ptr, ai = adj_idx, ae = adj_edge

    cdef np.ndarray[f64, ndim=1] bc_v_arr = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] bc_e_arr = np.zeros(nE, dtype=np.float64)
    cdef f64[::1] bc_v = bc_v_arr, bc_e = bc_e_arr

    cdef np.ndarray[i32, ndim=1] dist_arr = np.full(nV, -1, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] sigma_arr = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] delta_arr = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[i64, ndim=1] queue_arr = np.empty(nV, dtype=np.int64)
    cdef i32[::1] dist = dist_arr
    cdef f64[::1] sigma = sigma_arr, delta = delta_arr
    cdef i64[::1] queue = queue_arr

    cdef Py_ssize_t n_sources = nV
    if max_sources > 0 and max_sources < nV:
        n_sources = max_sources

    cdef Py_ssize_t s, v, w, k, qi, q_len
    cdef double coeff

    for s in range(n_sources):
        if s == 0:
            for v in range(nV):
                dist[v] = -1
                sigma[v] = 0.0
                delta[v] = 0.0
        else:
            for qi in range(q_len):
                v = <Py_ssize_t>queue[qi]
                dist[v] = -1
                sigma[v] = 0.0
                delta[v] = 0.0

        dist[s] = 0
        sigma[s] = 1.0
        queue[0] = <i64>s
        q_len = 1
        qi = 0

        while qi < q_len:
            v = <Py_ssize_t>queue[qi]; qi += 1
            for k in range(ap[v], ap[v + 1]):
                w = <Py_ssize_t>ai[k]
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    sigma[w] = sigma[v]
                    queue[q_len] = <i64>w; q_len += 1
                elif dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]

        for qi in range(q_len - 1, 0, -1):
            w = <Py_ssize_t>queue[qi]
            if sigma[w] < 1e-15:
                continue
            for k in range(ap[w], ap[w + 1]):
                v = <Py_ssize_t>ai[k]
                if dist[v] == dist[w] - 1:
                    coeff = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    delta[v] += coeff
                    bc_e[<Py_ssize_t>ae[k]] += coeff
            if w != s:
                bc_v[w] += delta[w]

    cdef double scale_v, scale_e, src_scale
    scale_v = (<double>nV - 1.0) * (<double>nV - 2.0)
    scale_v = 1.0 / scale_v if scale_v > 0.0 else 0.0
    scale_e = <double>nV * (<double>nV - 1.0)
    scale_e = 1.0 / scale_e if scale_e > 0.0 else 0.0

    if n_sources < nV:
        src_scale = <double>nV / <double>n_sources
    else:
        src_scale = 1.0

    for v in range(nV):
        bc_v[v] *= scale_v * src_scale
    for k in range(<Py_ssize_t>nE):
        bc_e[k] *= 0.5 * scale_e * src_scale

    return bc_v_arr, bc_e_arr


def betweenness(adj_ptr, adj_idx, adj_edge,
                Py_ssize_t nV, Py_ssize_t nE,
                Py_ssize_t max_sources=0):
    """Dispatch betweenness centrality by index type."""
    if adj_ptr.dtype == np.int64:
        return betweenness_i64(adj_ptr, adj_idx, adj_edge, nV, nE, max_sources)
    return betweenness_i32(adj_ptr, adj_idx, adj_edge, nV, nE, max_sources)


# Clustering coefficient

@cython.boundscheck(False)
@cython.wraparound(False)
def clustering_i32(np.ndarray[i32, ndim=1] adj_ptr,
                   np.ndarray[i32, ndim=1] adj_idx,
                   Py_ssize_t nV):
    """
    Local clustering coefficient for each vertex.

    Triangles counted via two-pointer merge over sorted neighbor
    lists. C(v) = 2*T(v) / (deg(v) * (deg(v) - 1)) for deg >= 2.

    Parameters
    ----------
    adj_ptr : i32[nV+1]
    adj_idx : i32[2*nE]
        Sorted neighbor indices within each row.
    nV : int

    Returns
    -------
    f64[nV]
        Clustering coefficient in [0, 1]. Zero for deg < 2.
    """
    if nV == 0:
        return np.empty(0, dtype=np.float64)

    cdef i32[::1] ap = adj_ptr, ai = adj_idx

    cdef np.ndarray[i32, ndim=1] tri_arr = np.zeros(nV, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] cc_arr = np.zeros(nV, dtype=np.float64)
    cdef i32[::1] tri = tri_arr
    cdef f64[::1] cc = cc_arr

    cdef Py_ssize_t u, v, k, pu, pv, eu, ev, deg_u, deg_v
    cdef i32 nu, nv, w

    for u in range(nV):
        eu = ap[u + 1]
        for k in range(ap[u], eu):
            v = ai[k]
            if v <= u:
                continue
            pu = ap[u]; ev = ap[v + 1]; pv = ap[v]
            while pu < eu and pv < ev:
                nu = ai[pu]; nv = ai[pv]
                if nu == nv:
                    tri[u] += 1
                    tri[v] += 1
                    tri[nu] += 1
                    pu += 1; pv += 1
                elif nu < nv:
                    pu += 1
                else:
                    pv += 1

    for v in range(nV):
        deg_v = ap[v + 1] - ap[v]
        if deg_v >= 2:
            cc[v] = (2.0 * <double>tri[v]) / (<double>deg_v * (<double>deg_v - 1.0))

    return cc_arr


@cython.boundscheck(False)
@cython.wraparound(False)
def clustering_i64(np.ndarray[i64, ndim=1] adj_ptr,
                   np.ndarray[i64, ndim=1] adj_idx,
                   Py_ssize_t nV):
    """Local clustering coefficient. int64 index variant."""
    if nV == 0:
        return np.empty(0, dtype=np.float64)

    cdef i64[::1] ap = adj_ptr, ai = adj_idx

    cdef np.ndarray[i32, ndim=1] tri_arr = np.zeros(nV, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] cc_arr = np.zeros(nV, dtype=np.float64)
    cdef i32[::1] tri = tri_arr
    cdef f64[::1] cc = cc_arr

    cdef Py_ssize_t u, v, k, pu, pv, deg_v
    cdef i64 eu, ev, nu, nv

    for u in range(nV):
        eu = ap[u + 1]
        for k in range(ap[u], eu):
            v = <Py_ssize_t>ai[k]
            if v <= u:
                continue
            pu = <Py_ssize_t>ap[u]; ev = ap[v + 1]; pv = <Py_ssize_t>ap[v]
            while pu < <Py_ssize_t>eu and pv < <Py_ssize_t>ev:
                nu = ai[pu]; nv = ai[pv]
                if nu == nv:
                    tri[u] += 1
                    tri[v] += 1
                    tri[<Py_ssize_t>nu] += 1
                    pu += 1; pv += 1
                elif nu < nv:
                    pu += 1
                else:
                    pv += 1

    for v in range(nV):
        deg_v = <Py_ssize_t>(ap[v + 1] - ap[v])
        if deg_v >= 2:
            cc[v] = (2.0 * <double>tri[v]) / (<double>deg_v * (<double>deg_v - 1.0))

    return cc_arr


def clustering(adj_ptr, adj_idx, Py_ssize_t nV):
    """Dispatch clustering coefficient by index type."""
    if adj_ptr.dtype == np.int64:
        return clustering_i64(adj_ptr, adj_idx, nV)
    return clustering_i32(adj_ptr, adj_idx, nV)


# Louvain community detection

@cython.boundscheck(False)
@cython.wraparound(False)
def louvain_i32(np.ndarray[i32, ndim=1] adj_ptr,
                np.ndarray[i32, ndim=1] adj_idx,
                np.ndarray[f64, ndim=1] adj_wt,
                Py_ssize_t nV,
                Py_ssize_t nE,
                int max_passes=20):
    """
    Louvain community detection via modularity optimization.

    For each vertex, evaluates modularity gain of moving to each
    neighbor's community. Repeats until no improvement.

    Parameters
    ----------
    adj_ptr : i32[nV+1]
    adj_idx : i32[2*nE]
    adj_wt : f64[2*nE]
    nV, nE : int
    max_passes : int, default 20

    Returns
    -------
    labels : i32[nV]
        Community label for each vertex.
    n_communities : int
    modularity : float
    """
    if nV == 0:
        return np.empty(0, dtype=np.int32), 0, 0.0
    if nV == 1:
        return np.zeros(1, dtype=np.int32), 1, 0.0

    cdef f64[::1] aw = adj_wt
    cdef i32[::1] ap = adj_ptr, ai = adj_idx
    cdef double W2 = 0.0  # 2W
    cdef Py_ssize_t k
    for k in range(2 * nE):
        W2 += aw[k]
    if W2 < 1e-15:
        labels = np.arange(nV, dtype=np.int32)
        return labels, nV, 0.0
    cdef double inv_W2 = 1.0 / W2

    cdef np.ndarray[i32, ndim=1] comm_arr = np.arange(nV, dtype=np.int32)
    cdef i32[::1] comm = comm_arr

    cdef np.ndarray[f64, ndim=1] sigma_tot_arr = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] sigma_tot = sigma_tot_arr

    cdef np.ndarray[f64, ndim=1] ki_arr = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] ki = ki_arr
    cdef Py_ssize_t v
    for v in range(nV):
        for k in range(ap[v], ap[v + 1]):
            ki[v] += aw[k]
        sigma_tot[v] = ki[v]

    cdef np.ndarray[f64, ndim=1] comm_wt_arr = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] comm_wt = comm_wt_arr

    cdef np.ndarray[i32, ndim=1] touched_arr = np.empty(nV, dtype=np.int32)
    cdef i32[::1] touched = touched_arr

    cdef int p, improved
    cdef Py_ssize_t u, n_touched
    cdef i32 ci, cj, best_c
    cdef double dq, best_dq, ki_v
    cdef double w_to_ci, w_to_best

    for p in range(max_passes):
        improved = 0

        for v in range(nV):
            ci = comm[v]
            ki_v = ki[v]
            n_touched = 0

            for k in range(ap[v], ap[v + 1]):
                u = ai[k]
                cj = comm[u]
                if comm_wt[cj] == 0.0 and cj != ci:
                    touched[n_touched] = cj
                    n_touched += 1
                comm_wt[cj] += aw[k]

            w_to_ci = comm_wt[ci]


            best_c = ci
            best_dq = 0.0

            for k in range(n_touched):
                cj = touched[k]
                dq = ((comm_wt[cj] - w_to_ci) * inv_W2
                      + ki_v * (sigma_tot[ci] - sigma_tot[cj] - ki_v)
                      * inv_W2 * inv_W2)
                if dq > best_dq:
                    best_dq = dq
                    best_c = cj

            if best_c != ci:
                sigma_tot[ci] -= ki_v
                sigma_tot[best_c] += ki_v
                comm[v] = best_c
                improved = 1

            comm_wt[ci] = 0.0
            for k in range(n_touched):
                comm_wt[touched[k]] = 0.0

        if not improved:
            break

    cdef np.ndarray[i32, ndim=1] remap_arr = np.full(nV, -1, dtype=np.int32)
    cdef i32[::1] remap = remap_arr
    cdef i32 n_comm = 0
    for v in range(nV):
        ci = comm[v]
        if remap[ci] == -1:
            remap[ci] = n_comm
            n_comm += 1
        comm[v] = remap[ci]

    # Compute final modularity
    cdef np.ndarray[f64, ndim=1] s_in = np.zeros(n_comm, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] s_tot = np.zeros(n_comm, dtype=np.float64)
    cdef f64[::1] sin_v = s_in, stot_v = s_tot

    for v in range(nV):
        ci = comm[v]
        stot_v[ci] += ki[v]
        for k in range(ap[v], ap[v + 1]):
            u = ai[k]
            if comm[u] == ci:
                sin_v[ci] += aw[k]

    cdef double Q = 0.0
    cdef Py_ssize_t c
    for c in range(n_comm):
        Q += sin_v[c] / W2 - (stot_v[c] * inv_W2) * (stot_v[c] * inv_W2)

    return comm_arr, <int>n_comm, Q


@cython.boundscheck(False)
@cython.wraparound(False)
def louvain_i64(np.ndarray[i64, ndim=1] adj_ptr,
                np.ndarray[i64, ndim=1] adj_idx,
                np.ndarray[f64, ndim=1] adj_wt,
                Py_ssize_t nV,
                Py_ssize_t nE,
                int max_passes=20):
    """Louvain community detection. int64 index variant."""
    if nV == 0:
        return np.empty(0, dtype=np.int32), 0, 0.0
    if nV == 1:
        return np.zeros(1, dtype=np.int32), 1, 0.0

    cdef f64[::1] aw = adj_wt
    cdef i64[::1] ap = adj_ptr, ai = adj_idx
    cdef double W2 = 0.0
    cdef Py_ssize_t k
    for k in range(2 * nE):
        W2 += aw[k]
    if W2 < 1e-15:
        return np.arange(nV, dtype=np.int32), nV, 0.0
    cdef double inv_W2 = 1.0 / W2

    cdef np.ndarray[i32, ndim=1] comm_arr = np.arange(nV, dtype=np.int32)
    cdef i32[::1] comm = comm_arr

    cdef np.ndarray[f64, ndim=1] sigma_tot_arr = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] sigma_tot = sigma_tot_arr

    cdef np.ndarray[f64, ndim=1] ki_arr = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] ki = ki_arr
    cdef Py_ssize_t v
    for v in range(nV):
        for k in range(ap[v], ap[v + 1]):
            ki[v] += aw[k]
        sigma_tot[v] = ki[v]

    cdef np.ndarray[f64, ndim=1] comm_wt_arr = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] comm_wt = comm_wt_arr

    cdef np.ndarray[i32, ndim=1] touched_arr = np.empty(nV, dtype=np.int32)
    cdef i32[::1] touched = touched_arr

    cdef int p, improved
    cdef Py_ssize_t u, n_touched
    cdef i32 ci, cj, best_c
    cdef double dq, best_dq, ki_v
    cdef double w_to_ci

    for p in range(max_passes):
        improved = 0

        for v in range(nV):
            ci = comm[v]
            ki_v = ki[v]
            n_touched = 0

            for k in range(ap[v], ap[v + 1]):
                u = <Py_ssize_t>ai[k]
                cj = comm[u]
                if comm_wt[cj] == 0.0 and cj != ci:
                    touched[n_touched] = cj
                    n_touched += 1
                comm_wt[cj] += aw[k]

            w_to_ci = comm_wt[ci]

            best_c = ci
            best_dq = 0.0

            for k in range(n_touched):
                cj = touched[k]
                dq = ((comm_wt[cj] - w_to_ci) * inv_W2
                      + ki_v * (sigma_tot[ci] - sigma_tot[cj] - ki_v)
                      * inv_W2 * inv_W2)
                if dq > best_dq:
                    best_dq = dq
                    best_c = cj

            if best_c != ci:
                sigma_tot[ci] -= ki_v
                sigma_tot[best_c] += ki_v
                comm[v] = best_c
                improved = 1

            comm_wt[ci] = 0.0
            for k in range(n_touched):
                comm_wt[touched[k]] = 0.0

        if not improved:
            break

    cdef np.ndarray[i32, ndim=1] remap_arr = np.full(nV, -1, dtype=np.int32)
    cdef i32[::1] remap = remap_arr
    cdef i32 n_comm = 0
    for v in range(nV):
        ci = comm[v]
        if remap[ci] == -1:
            remap[ci] = n_comm
            n_comm += 1
        comm[v] = remap[ci]

    cdef np.ndarray[f64, ndim=1] s_in = np.zeros(n_comm, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] s_tot = np.zeros(n_comm, dtype=np.float64)
    cdef f64[::1] sin_v = s_in, stot_v = s_tot

    for v in range(nV):
        ci = comm[v]
        stot_v[ci] += ki[v]
        for k in range(ap[v], ap[v + 1]):
            u = <Py_ssize_t>ai[k]
            if comm[u] == ci:
                sin_v[ci] += aw[k]

    cdef double Q = 0.0
    cdef Py_ssize_t c
    for c in range(n_comm):
        Q += sin_v[c] / W2 - (stot_v[c] * inv_W2) * (stot_v[c] * inv_W2)

    return comm_arr, <int>n_comm, Q


def louvain(adj_ptr, adj_idx, adj_wt, Py_ssize_t nV, Py_ssize_t nE,
            int max_passes=20):
    """Dispatch Louvain by index type."""
    if adj_ptr.dtype == np.int64:
        return louvain_i64(adj_ptr, adj_idx, adj_wt, nV, nE, max_passes)
    return louvain_i32(adj_ptr, adj_idx, adj_wt, nV, nE, max_passes)


# Pearson correlation

@cython.boundscheck(False)
@cython.wraparound(False)
def safe_correlation(np.ndarray[f64, ndim=1] a,
                     np.ndarray[f64, ndim=1] b):
    """
    Pearson correlation with zero-variance guard.

    Returns 0.0 if either signal has zero variance or n < 2.

    Parameters
    ----------
    a, b : f64[n]

    Returns
    -------
    float
    """
    cdef Py_ssize_t n = a.shape[0]
    if n < 2:
        return 0.0

    cdef f64[::1] av = a, bv = b
    cdef double sa = 0.0, sb = 0.0
    cdef double sa2 = 0.0, sb2 = 0.0, sab = 0.0
    cdef double ai, bi
    cdef double nn = <double>n
    cdef double dx, dy, denom
    cdef Py_ssize_t i

    for i in range(n):
        ai = av[i]; bi = bv[i]
        sa += ai;   sb += bi
        sa2 += ai * ai
        sb2 += bi * bi
        sab += ai * bi

    dx = nn * sa2 - sa * sa
    dy = nn * sb2 - sb * sb

    if dx < 1e-30 or dy < 1e-30:
        return 0.0

    denom = sqrt(dx * dy)
    return (nn * sab - sa * sb) / denom


# Build adjacency weights from edge weights

@cython.boundscheck(False)
@cython.wraparound(False)
def build_adj_weights_i32(np.ndarray[i32, ndim=1] adj_edge,
                          np.ndarray[f64, ndim=1] edge_weights,
                          Py_ssize_t nnz):
    """
    Map edge weights to adjacency entries.

    For each adjacency entry k, adj_wt[k] = edge_weights[adj_edge[k]].
    The adjacency is symmetric (2*nE entries), and each edge appears
    twice with the same weight.

    Parameters
    ----------
    adj_edge : i32[nnz]
        Edge index for each adjacency entry.
    edge_weights : f64[nE]
        Weight of each edge.
    nnz : int
        Length of adj_edge (= 2*nE).

    Returns
    -------
    f64[nnz]
    """
    cdef np.ndarray[f64, ndim=1] wt = np.empty(nnz, dtype=np.float64)
    cdef f64[::1] wv = wt, ew = edge_weights
    cdef i32[::1] ae = adj_edge
    cdef Py_ssize_t k

    for k in range(nnz):
        wv[k] = ew[ae[k]]

    return wt


@cython.boundscheck(False)
@cython.wraparound(False)
def build_adj_weights_i64(np.ndarray[i64, ndim=1] adj_edge,
                          np.ndarray[f64, ndim=1] edge_weights,
                          Py_ssize_t nnz):
    """Map edge weights to adjacency entries. int64 variant."""
    cdef np.ndarray[f64, ndim=1] wt = np.empty(nnz, dtype=np.float64)
    cdef f64[::1] wv = wt, ew = edge_weights
    cdef i64[::1] ae = adj_edge
    cdef Py_ssize_t k

    for k in range(nnz):
        wv[k] = ew[<Py_ssize_t>ae[k]]

    return wt


def build_adj_weights(adj_edge, edge_weights):
    """Dispatch adjacency weight construction by index type."""
    cdef Py_ssize_t nnz = adj_edge.shape[0]
    if adj_edge.dtype == np.int64:
        return build_adj_weights_i64(adj_edge, edge_weights, nnz)
    return build_adj_weights_i32(adj_edge, edge_weights, nnz)


# Combined builder

def build_standard_metrics(adj_ptr, adj_idx, adj_edge, adj_wt,
                           Py_ssize_t nV, Py_ssize_t nE,
                           double damping=0.85,
                           int pagerank_iter=100,
                           Py_ssize_t btw_max_sources=0,
                           int louvain_max_passes=20):
    """
    Compute all standard graph metrics.

    Parameters
    ----------
    adj_ptr : I[nV+1]
    adj_idx : I[2*nE]
    adj_edge : I[2*nE]
    adj_wt : f64[2*nE]
    nV, nE : int
    damping : float, default 0.85
    pagerank_iter : int, default 100
    btw_max_sources : int, default 0 (all vertices)
    louvain_max_passes : int, default 20

    Returns
    -------
    dict
        pagerank : f64[nV]
        betweenness_v : f64[nV]
        betweenness_e : f64[nE]
        btw_norm_v : f64[nV]
        btw_norm_e : f64[nE]
        clustering : f64[nV]
        community_labels : i32[nV]
        n_communities : int
        modularity : float
    """
    result = {}

    # PageRank
    result['pagerank'] = pagerank(adj_ptr, adj_idx, adj_wt, nV, nE,
                                  damping, pagerank_iter)

    # Betweenness
    bc_v, bc_e = betweenness(adj_ptr, adj_idx, adj_edge, nV, nE,
                             btw_max_sources)
    result['betweenness_v'] = bc_v
    result['betweenness_e'] = bc_e

    # Normalized betweenness
    cdef double mx_v = 0.0, mx_e = 0.0
    cdef f64[::1] bv = bc_v, be = bc_e
    cdef Py_ssize_t i
    for i in range(nV):
        if bv[i] > mx_v:
            mx_v = bv[i]
    for i in range(nE):
        if be[i] > mx_e:
            mx_e = be[i]

    cdef np.ndarray[f64, ndim=1] bn_v = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] bn_e = np.zeros(nE, dtype=np.float64)
    cdef f64[::1] bnv = bn_v, bne = bn_e
    if mx_v > 1e-15:
        for i in range(nV):
            bnv[i] = bv[i] / mx_v
    if mx_e > 1e-15:
        for i in range(nE):
            bne[i] = be[i] / mx_e
    result['btw_norm_v'] = bn_v
    result['btw_norm_e'] = bn_e

    # Clustering
    result['clustering'] = clustering(adj_ptr, adj_idx, nV)

    # Louvain
    labels, n_comm, Q = louvain(adj_ptr, adj_idx, adj_wt, nV, nE,
                                louvain_max_passes)
    result['community_labels'] = labels
    result['n_communities'] = n_comm
    result['modularity'] = Q

    return result
