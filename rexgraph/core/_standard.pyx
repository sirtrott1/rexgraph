# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._standard: Classical graph algorithms on the 1 skeleton.

Operates on the undirected graph underlying a relational complex.
Input is a symmetric CSR adjacency from _cycles.build_symmetric_adjacency.

PageRank - power iteration, O(nE) per step.
Betweenness - vertex and edge betweenness centrality, O(nV * nE).
Clustering - local clustering coefficient via sorted neighbor intersection.
Louvain - modularity based community detection, O(nE) per pass.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,
)

from libc.math cimport fabs, sqrt, isfinite
from libc.float cimport DBL_MIN

np.import_array()


# PageRank

ctypedef fused rank_index:
    i32
    i64


def _markov_weights(np.ndarray[rank_index, ndim=1] ptr,
                    np.ndarray[rank_index, ndim=1] idx,
                    np.ndarray[f64, ndim=1] weights, Py_ssize_t n):
    cdef np.ndarray[f64, ndim=1] probabilities = np.zeros(weights.shape[0])
    cdef np.ndarray[np.uint8_t, ndim=1] dangling = np.zeros(n, dtype=np.uint8)
    cdef Py_ssize_t v, k
    cdef double scale, total, inverse_scale, inverse_total
    for v in range(n):
        if ptr[v+1] < ptr[v] or ptr[v+1] > len(idx):
            raise ValueError("invalid Markov CSR pointers")
        total = 0
        for k in range(ptr[v], ptr[v+1]):
            if idx[k] < 0 or idx[k] >= n:
                raise ValueError("invalid Markov CSR index")
            if not isfinite(weights[k]) or weights[k] < 0:
                raise ValueError("Markov weights must be finite and nonnegative")
            total += weights[k]
        if total == 0:
            dangling[v] = 1
            continue
        if isfinite(total) and total >= DBL_MIN:
            inverse_total = 1/total
            for k in range(ptr[v], ptr[v+1]):
                probabilities[k] = weights[k]*inverse_total
            continue
        scale = 0
        for k in range(ptr[v], ptr[v+1]):
            if weights[k] > scale:
                scale = weights[k]
        total = 0
        if scale >= DBL_MIN:
            inverse_scale = 1/scale
            for k in range(ptr[v], ptr[v+1]):
                total += weights[k]*inverse_scale
            inverse_total = 1/total
            for k in range(ptr[v], ptr[v+1]):
                probabilities[k] = (weights[k]*inverse_scale)*inverse_total
        else:
            for k in range(ptr[v], ptr[v+1]):
                total += weights[k]/scale
            inverse_total = 1/total
            for k in range(ptr[v], ptr[v+1]):
                probabilities[k] = (weights[k]/scale)*inverse_total
    return probabilities, dangling


def markov_weights(adj_ptr, adj_idx, adj_wt, nV):
    """Validate CSR and normalize outgoing mass without a magnitude threshold."""
    from numbers import Integral
    if isinstance(nV, (bool, np.bool_)) or not isinstance(nV, Integral) or nV < 0:
        raise ValueError("nV must be a nonnegative integer")
    ptr, idx, wt = map(np.asarray, (adj_ptr, adj_idx, adj_wt))
    if (ptr.ndim != 1 or idx.ndim != 1 or wt.ndim != 1
            or ptr.dtype not in (np.dtype('int32'), np.dtype('int64'))
            or idx.dtype != ptr.dtype):
        raise TypeError("Markov CSR requires matching int32 or int64 indices and vector weights")
    if len(ptr) != nV+1 or ptr[0] != 0 or ptr[len(ptr)-1] != len(idx) or len(wt) != len(idx):
        raise ValueError("invalid Markov CSR coordinates")
    if wt.dtype.kind not in "fiu":
        raise TypeError("Markov weights must be real numbers")
    converted = np.ascontiguousarray(wt, dtype=np.float64)
    if wt.dtype.itemsize > 8 and np.any((wt != 0) & (converted == 0)):
        raise ValueError("Markov weights underflow float64")
    return _markov_weights(np.ascontiguousarray(ptr), np.ascontiguousarray(idx),
                           converted, nV)


cdef void _rank_step(const rank_index[::1] ptr, const rank_index[::1] idx,
                     const f64[::1] probabilities, const np.uint8_t[::1] dangling,
                     const f64[::1] seed, const f64[::1] current, f64[::1] output,
                     double damping, Py_ssize_t n) noexcept:
    cdef Py_ssize_t v, k
    cdef double mass = 0
    for v in range(n):
        if dangling[v]:
            mass += current[v]
    for v in range(n):
        output[v] = (1-damping)*seed[v] + damping*mass/n
    for v in range(n):
        mass = damping*current[v]
        for k in range(ptr[v], ptr[v+1]):
            output[idx[k]] += mass*probabilities[k]


def _pagerank_iter(np.ndarray[rank_index, ndim=1] ptr,
                   np.ndarray[rank_index, ndim=1] idx,
                   np.ndarray[f64, ndim=1] probabilities,
                   np.ndarray[np.uint8_t, ndim=1] dangling,
                   np.ndarray[f64, ndim=1] seed,
                   double damping, int max_iter, double tol, bint report, action=None):
    cdef Py_ssize_t n = len(seed), v
    cdef np.ndarray[f64, ndim=1] current_array = seed.copy()
    cdef np.ndarray[f64, ndim=1] next_array = np.empty(n)
    cdef f64[::1] current = current_array, nxt = next_array, temporary
    cdef const rank_index[::1] rows = ptr, columns = idx
    cdef const f64[::1] transition = probabilities, restart = seed
    cdef const f64[::1] applied
    cdef const np.uint8_t[::1] absent = dangling
    cdef double diff = 0, residual = 0
    cdef int iteration = 0
    if n:
        for iteration in range(1, max_iter+1):
            if action is None:
                _rank_step[rank_index](rows, columns, transition, absent, restart, current, nxt, damping, n)
            else:
                applied = action(np.asarray(current))
                if len(applied) != n:
                    raise ValueError("PageRank action changed its vertex axis")
                for v in range(n):
                    nxt[v] = damping*applied[v] + (1-damping)*restart[v]
            diff = 0
            for v in range(n):
                diff += fabs(nxt[v]-current[v])
            temporary = current
            current = nxt
            nxt = temporary
            if damping*diff <= tol*(1-damping):
                break
        if report:
            if action is None:
                _rank_step[rank_index](rows, columns, transition, absent, restart, current, nxt, damping, n)
            else:
                applied = action(np.asarray(current))
                if len(applied) != n:
                    raise ValueError("PageRank action changed its vertex axis")
                for v in range(n):
                    nxt[v] = damping*applied[v] + (1-damping)*restart[v]
            for v in range(n):
                residual += fabs(nxt[v]-current[v])
    # Memoryview identity is not ndarray identity. Return the actual current buffer.
    if not report:
        return np.asarray(current), None
    return np.asarray(current), {
        "iterations": iteration, "residual_l1": residual,
        "error_bound_l1": residual/(1-damping),
        "converged": bool(residual <= tol*(1-damping)),
        "kernel": "native-pagerank" if action is None else "native-tensor-pagerank", "dangling": "uniform",
    }


def pagerank_controls(damping=0.85, max_iter=100, tol=1e-8):
    """Validate the numerical contraction and iteration policy without solving."""
    from numbers import Integral, Real
    for name, value in (("damping", damping), ("tol", tol)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(float(value)):
            raise ValueError(name + " must be a finite real scalar")
    damping, tol = float(damping), float(tol)
    if not 0 <= damping < 1 or not 0 < tol < 1:
        raise ValueError("PageRank requires 0<=damping<1 and 0<tol<1")
    if (isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral)
            or not 0 < max_iter < np.iinfo(np.int32).max):
        raise ValueError("max_iter must be a positive integer below INT32_MAX")
    return damping, int(max_iter), tol


def _pagerank_seed(nV, seed):
    """Normalize the restart once for both selected transition policies."""
    if seed is None:
        seed = np.full(nV, 1.0/nV) if nV else np.empty(0)
    else:
        seed = np.asarray(seed)
        if (seed.shape != (nV,) or seed.dtype.kind not in "fiu"
                or np.any(~np.isfinite(seed)) or np.any(seed < 0)):
            raise ValueError("PageRank seed must be a finite nonnegative vertex vector")
        seed = np.ascontiguousarray(seed, dtype=np.float64)
        if np.any(~np.isfinite(seed)):
            raise ValueError("PageRank seed must be representable in float64")
        if nV:
            scale = np.max(seed)
            if scale == 0:
                raise ValueError("PageRank seed must have positive mass")
            seed = seed/scale
            seed = seed/seed.sum()
    return seed


def _pagerank_action(action, nV, damping, max_iter, tol, seed=None):
    """Internal solver for the certified stochastic action built by Core."""
    damping, max_iter, tol = pagerank_controls(damping, max_iter, tol)
    seed = _pagerank_seed(nV, seed)
    empty = np.empty(0, dtype=np.int64)
    return _pagerank_iter(empty, empty, np.empty(0), np.empty(0, dtype=np.uint8),
                          seed, damping, max_iter, tol, True, action)


def pagerank(adj_ptr, adj_idx, adj_wt, nV, nE,
             damping=0.85, max_iter=100, tol=1e-8, *, seed=None, report=False):
    """Weighted outgoing walk with uniform dangling mass and explicit restart.

    report=True returns a residual and contraction error bound with the scores.
    The legacy array return remains available even after iteration exhaustion.
    """
    from numbers import Integral
    damping, max_iter, tol = pagerank_controls(damping, max_iter, tol)
    if isinstance(nE, (bool, np.bool_)) or not isinstance(nE, Integral) or nE < 0:
        raise ValueError("nE must be a nonnegative integer")
    if not isinstance(report, (bool, np.bool_)):
        raise TypeError("report must be boolean")
    probabilities, dangling = markov_weights(adj_ptr, adj_idx, adj_wt, nV)
    seed = _pagerank_seed(nV, seed)
    result, info = _pagerank_iter(np.ascontiguousarray(adj_ptr), np.ascontiguousarray(adj_idx),
                                   probabilities, dangling, seed, damping, max_iter, tol, report)
    return (result, info) if report else result


def pagerank_i32(np.ndarray[i32, ndim=1] adj_ptr,
                 np.ndarray[i32, ndim=1] adj_idx,
                 np.ndarray[f64, ndim=1] adj_wt, nV, nE,
                 damping=0.85, max_iter=100, tol=1e-8, *, seed=None, report=False):
    return pagerank(adj_ptr, adj_idx, adj_wt, nV, nE, damping, max_iter, tol, seed=seed, report=report)


def pagerank_i64(np.ndarray[i64, ndim=1] adj_ptr,
                 np.ndarray[i64, ndim=1] adj_idx,
                 np.ndarray[f64, ndim=1] adj_wt, nV, nE,
                 damping=0.85, max_iter=100, tol=1e-8, *, seed=None, report=False):
    return pagerank(adj_ptr, adj_idx, adj_wt, nV, nE, damping, max_iter, tol, seed=seed, report=report)


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

    adj_ptr : i32[nV+1]
    adj_idx : i32[2*nE]
    adj_edge : i32[2*nE]
        Maps each adjacency entry to the original edge index.
    nV, nE : int
    max_sources : int, default 0
        If > 0, sample this many source vertices and rescale.
        0 means use all vertices (exact computation).

    Returns

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

    Triangles counted via two pointer merge over sorted neighbor
    lists. C(v) = 2*T(v) / (deg(v) * (deg(v) - 1)) for deg >= 2.

    Parameters

    adj_ptr : i32[nV+1]
    adj_idx : i32[2*nE]
        Sorted neighbor indices within each row.
    nV : int

    Returns

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
                    if nu > v:                       # count each triangle once, at its smallest edge (apex > v)
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
                    if nu > v:                       # count each triangle once, at its smallest edge (apex > v)
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

    adj_ptr : i32[nV+1]
    adj_idx : i32[2*nE]
    adj_wt : f64[2*nE]
    nV, nE : int
    max_passes : int, default 20

    Returns

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
    Pearson correlation with zero variance guard.

    Returns 0.0 if either signal has zero variance or n < 2.

    Parameters

    a, b : f64[n]

    Returns

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

    adj_edge : i32[nnz]
        Edge index for each adjacency entry.
    edge_weights : f64[nE]
        Weight of each edge.
    nnz : int
        Length of adj_edge (= 2*nE).

    Returns

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


def weighted_shortest_path(np.ndarray[i32, ndim=1] src,
                           np.ndarray[i32, ndim=1] tgt,
                           np.ndarray[f64, ndim=1] weights,
                           int n_vertices,
                           int source, int target):
    """
    Dijkstra shortest path on the 1 skeleton with edge weights.

    Parameters

    src, tgt : (nE,) int32
        Edge endpoints.
    weights : (nE,) float64
        Non-negative edge weights (lower = cheaper to traverse).
    n_vertices : int
        Total number of vertices.
    source, target : int
        Start and end vertex indices.

    Returns

    dict with 'distance' (float), 'path' (list of vertex indices),
    'found' (bool).
    """
    import heapq

    cdef int nE = src.shape[0]

    # Build adjacency list
    adj = [[] for _ in range(n_vertices)]
    cdef int e
    for e in range(nE):
        adj[src[e]].append((tgt[e], weights[e]))
        adj[tgt[e]].append((src[e], weights[e]))

    # Dijkstra
    cdef np.ndarray[f64, ndim=1] dist = np.full(n_vertices, np.inf, dtype=np.float64)
    prev = [-1] * n_vertices
    dist[source] = 0.0

    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[target] == np.inf:
        return {'distance': float('inf'), 'path': [], 'found': False}

    # Reconstruct path
    path = []
    v = target
    while v != -1:
        path.append(v)
        v = prev[v]
    path.reverse()

    return {'distance': float(dist[target]), 'path': path, 'found': True}
