# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._cycles - Deterministic fundamental cycle basis.

Computes a fundamental cycle basis for the 1-skeleton of a rex via
tree-cotree decomposition, yielding the face set and data needed to
build B_2.

Algorithm:
    1. Build symmetric undirected adjacency in CSR form.
    2. BFS spanning forest with sorted neighbor traversal (deterministic).
    3. Identify beta_1 = m - n + beta_0 cotree edges.
    4. For each cotree edge (u, v), trace the cycle through the
       tree path u -> LCA(u,v) -> v.
    5. Assign orientation signs: +1 if traversal agrees with the
       edge's source->target direction, -1 otherwise.

Output format matches build_B2_from_cycles in _boundary.pyx:
    cycle_edges   - concatenated edge indices
    cycle_signs   - orientation signs (+/-1.0)
    cycle_lengths - boundary length of each face
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,

    MAX_INT32_NNZ,

    UnionFind, uf_init, uf_find, uf_union, uf_free,
    UnionFind64, uf64_init, uf64_find, uf64_union, uf64_free,
)

np.import_array()

cdef enum:
    _SENTINEL = -1


# BFS queue (nogil, malloc-backed)

cdef struct _Queue:
    i32* data
    Py_ssize_t capacity
    Py_ssize_t head
    Py_ssize_t tail
    Py_ssize_t size

cdef inline int _q_init(_Queue* q, Py_ssize_t cap) noexcept nogil:
    q.data = <i32*>malloc(cap * sizeof(i32))
    if q.data == NULL: return -1
    q.capacity = cap
    q.head = 0
    q.tail = 0
    q.size = 0
    return 0

cdef inline void _q_free(_Queue* q) noexcept nogil:
    if q.data != NULL: free(q.data)
    q.data = NULL
    q.capacity = 0
    q.head = 0
    q.tail = 0
    q.size = 0

cdef inline void _q_clear(_Queue* q) noexcept nogil:
    q.head = 0
    q.tail = 0
    q.size = 0

cdef inline void _q_push(_Queue* q, i32 val) noexcept nogil:
    q.data[q.tail] = val
    q.tail += 1
    if q.tail >= q.capacity: q.tail = 0
    q.size += 1

cdef inline i32 _q_pop(_Queue* q) noexcept nogil:
    cdef i32 val = q.data[q.head]
    q.head += 1
    if q.head >= q.capacity: q.head = 0
    q.size -= 1
    return val

cdef inline bint _q_empty(_Queue* q) noexcept nogil:
    return q.size == 0


cdef struct _Queue64:
    i64* data
    Py_ssize_t capacity
    Py_ssize_t head
    Py_ssize_t tail
    Py_ssize_t size

cdef inline int _q64_init(_Queue64* q, Py_ssize_t cap) noexcept nogil:
    q.data = <i64*>malloc(cap * sizeof(i64))
    if q.data == NULL: return -1
    q.capacity = cap
    q.head = 0
    q.tail = 0
    q.size = 0
    return 0

cdef inline void _q64_free(_Queue64* q) noexcept nogil:
    if q.data != NULL: free(q.data)
    q.data = NULL
    q.capacity = 0
    q.head = 0
    q.tail = 0
    q.size = 0

cdef inline void _q64_clear(_Queue64* q) noexcept nogil:
    q.head = 0
    q.tail = 0
    q.size = 0

cdef inline void _q64_push(_Queue64* q, i64 val) noexcept nogil:
    q.data[q.tail] = val
    q.tail += 1
    if q.tail >= q.capacity: q.tail = 0
    q.size += 1

cdef inline i64 _q64_pop(_Queue64* q) noexcept nogil:
    cdef i64 val = q.data[q.head]
    q.head += 1
    if q.head >= q.capacity: q.head = 0
    q.size -= 1
    return val

cdef inline bint _q64_empty(_Queue64* q) noexcept nogil:
    return q.size == 0


# Paired insertion sort

cdef inline void _isort_paired_i32(i32* keys, i32* vals, Py_ssize_t n) noexcept nogil:
    """Sort keys[0..n) ascending, permuting vals in parallel."""
    cdef Py_ssize_t i, j
    cdef i32 kk, vv
    for i in range(1, n):
        kk = keys[i]
        vv = vals[i]
        j = i - 1
        while j >= 0 and keys[j] > kk:
            keys[j+1] = keys[j]
            vals[j+1] = vals[j]
            j -= 1
        keys[j+1] = kk
        vals[j+1] = vv


cdef inline void _isort_paired_i64(i64* keys, i64* vals, Py_ssize_t n) noexcept nogil:
    """Sort keys[0..n) ascending, permuting vals in parallel (int64 variant)."""
    cdef Py_ssize_t i, j
    cdef i64 kk, vv
    for i in range(1, n):
        kk = keys[i]
        vv = vals[i]
        j = i - 1
        while j >= 0 and keys[j] > kk:
            keys[j+1] = keys[j]
            vals[j+1] = vals[j]
            j -= 1
        keys[j+1] = kk
        vals[j+1] = vv


# Symmetric adjacency construction

@cython.boundscheck(False)
@cython.wraparound(False)
def build_symmetric_adjacency_i32(Py_ssize_t nV, Py_ssize_t nE,
                                   np.ndarray[i32, ndim=1] sources,
                                   np.ndarray[i32, ndim=1] targets):
    """
    Build undirected CSR adjacency from directed edge arrays.

    For each directed edge j, stores both directions. Neighbors
    within each row are sorted by vertex index for deterministic
    BFS traversal.

    Returns
    -------
    adj_ptr : int32[nV + 1]
        CSR row pointers.
    adj_idx : int32[2 * nE]
        Neighbor vertex indices, sorted within each row.
    adj_edge : int32[2 * nE]
        Edge index corresponding to each adjacency entry.
    """
    cdef Py_ssize_t j, v, nnz = 2 * nE
    cdef i32[::1] s = sources, t = targets

    cdef np.ndarray[i32, ndim=1] deg = np.zeros(nV, dtype=np.int32)
    cdef i32[::1] d = deg
    for j in range(nE):
        d[s[j]] += 1
        d[t[j]] += 1

    # Prefix sum
    cdef np.ndarray[i32, ndim=1] ptr = np.empty(nV + 1, dtype=np.int32)
    cdef i32[::1] p = ptr
    p[0] = 0
    for v in range(nV): p[v+1] = p[v] + d[v]

    cdef np.ndarray[i32, ndim=1] idx = np.empty(nnz, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] eidx = np.empty(nnz, dtype=np.int32)
    cdef i32[::1] ai = idx, ae = eidx
    cdef np.ndarray[i32, ndim=1] cursor = ptr[:nV].copy()
    cdef i32[::1] cur = cursor
    cdef Py_ssize_t pos

    for j in range(nE):
        pos = cur[s[j]]
        ai[pos] = t[j]
        ae[pos] = <i32>j
        cur[s[j]] += 1
        pos = cur[t[j]]
        ai[pos] = s[j]
        ae[pos] = <i32>j
        cur[t[j]] += 1

    # Sort rows for deterministic traversal
    cdef Py_ssize_t row_start, row_len
    with nogil:
        for v in range(nV):
            row_start = p[v]
            row_len = p[v+1] - row_start
            if row_len > 1:
                _isort_paired_i32(&ai[row_start], &ae[row_start], row_len)

    return ptr, idx, eidx


@cython.boundscheck(False)
@cython.wraparound(False)
def build_symmetric_adjacency_i64(Py_ssize_t nV, Py_ssize_t nE,
                                   np.ndarray[i64, ndim=1] sources,
                                   np.ndarray[i64, ndim=1] targets):
    """Build undirected CSR adjacency from directed edge arrays (int64 variant)."""
    cdef Py_ssize_t j, v, nnz = 2 * nE
    cdef i64[::1] s = sources, t = targets

    cdef np.ndarray[i64, ndim=1] deg = np.zeros(nV, dtype=np.int64)
    cdef i64[::1] d = deg
    for j in range(nE):
        d[s[j]] += 1
        d[t[j]] += 1

    cdef np.ndarray[i64, ndim=1] ptr = np.empty(nV + 1, dtype=np.int64)
    cdef i64[::1] p = ptr
    p[0] = 0
    for v in range(nV): p[v+1] = p[v] + d[v]

    cdef np.ndarray[i64, ndim=1] idx = np.empty(nnz, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] eidx = np.empty(nnz, dtype=np.int64)
    cdef i64[::1] ai = idx, ae = eidx
    cdef np.ndarray[i64, ndim=1] cursor = ptr[:nV].copy()
    cdef i64[::1] cur = cursor
    cdef Py_ssize_t pos

    for j in range(nE):
        pos = cur[s[j]]
        ai[pos] = t[j]
        ae[pos] = <i64>j
        cur[s[j]] += 1
        pos = cur[t[j]]
        ai[pos] = s[j]
        ae[pos] = <i64>j
        cur[t[j]] += 1

    cdef Py_ssize_t row_start, row_len
    with nogil:
        for v in range(nV):
            row_start = p[v]
            row_len = p[v+1] - row_start
            if row_len > 1:
                _isort_paired_i64(&ai[row_start], &ae[row_start], row_len)

    return ptr, idx, eidx


def build_symmetric_adjacency(Py_ssize_t nV, Py_ssize_t nE, sources, targets):
    """Build undirected CSR adjacency. Dispatches on index dtype."""
    if not isinstance(sources, np.ndarray): sources = np.asarray(sources)
    if not isinstance(targets, np.ndarray): targets = np.asarray(targets)
    if sources.dtype == np.int64 or targets.dtype == np.int64 or max(nV, nE) >= MAX_INT32_NNZ:
        return build_symmetric_adjacency_i64(
            nV, nE, sources.astype(np.int64, copy=False), targets.astype(np.int64, copy=False))
    return build_symmetric_adjacency_i32(
        nV, nE, sources.astype(np.int32, copy=False), targets.astype(np.int32, copy=False))


# BFS spanning forest

cdef int _bfs_spanning_forest(
    const i32* adj_ptr,
    const i32* adj_idx,
    const i32* adj_edge,
    Py_ssize_t nV,
    Py_ssize_t nE,
    i32* parent,
    i32* parent_edge,
    i32* depth,
    i32* is_tree,
) noexcept nogil:
    """
    Deterministic BFS spanning forest.

    Visits vertices in sorted neighbor order (guaranteed by adjacency
    construction). Produces a spanning tree per connected component.

    Returns the number of connected components (beta_0).
    """
    cdef _Queue q
    cdef Py_ssize_t v, j, start, end
    cdef i32 u, nbr, eidx
    cdef Py_ssize_t n_components = 0

    if _q_init(&q, nV) != 0: return -1

    for v in range(nV):
        parent[v] = _SENTINEL
        parent_edge[v] = _SENTINEL
        depth[v] = _SENTINEL
    memset(is_tree, 0, nE * sizeof(i32))

    for v in range(nV):
        if depth[v] != _SENTINEL: continue

        n_components += 1
        depth[v] = 0
        parent[v] = <i32>v
        _q_push(&q, <i32>v)

        while not _q_empty(&q):
            u = _q_pop(&q)
            start = adj_ptr[u]
            end = adj_ptr[u + 1]

            for j in range(start, end):
                nbr = adj_idx[j]
                eidx = adj_edge[j]

                if depth[nbr] == _SENTINEL:
                    depth[nbr] = depth[u] + 1
                    parent[nbr] = u
                    parent_edge[nbr] = eidx
                    is_tree[eidx] = 1
                    _q_push(&q, nbr)

    _q_free(&q)
    return <int>n_components


@cython.boundscheck(False)
@cython.wraparound(False)
def bfs_spanning_forest(np.ndarray[i32, ndim=1] adj_ptr,
                        np.ndarray[i32, ndim=1] adj_idx,
                        np.ndarray[i32, ndim=1] adj_edge,
                        Py_ssize_t nV, Py_ssize_t nE):
    """
    BFS spanning forest with deterministic traversal order.

    Returns
    -------
    parent : int32[nV]
        Parent vertex for each vertex (-1 sentinel unused
        roots
        have parent[v] == v).
    parent_edge : int32[nV]
        Edge index connecting each vertex to its parent (-1 for roots).
    depth : int32[nV]
        BFS depth from the component root.
    is_tree : int32[nE]
        1 for spanning tree edges, 0 for cotree (non-tree) edges.
    n_components : int
        Number of connected components (beta_0).
    """
    cdef np.ndarray[i32, ndim=1] par = np.empty(nV, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] par_e = np.empty(nV, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] dep = np.empty(nV, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] itree = np.zeros(nE, dtype=np.int32)

    cdef i32[::1] p_ptr = adj_ptr, p_idx = adj_idx, p_edge = adj_edge
    cdef i32[::1] p_par = par, p_pe = par_e, p_dep = dep, p_it = itree

    cdef int nc
    with nogil:
        nc = _bfs_spanning_forest(&p_ptr[0], &p_idx[0], &p_edge[0],
                                  nV, nE, &p_par[0], &p_pe[0],
                                  &p_dep[0], &p_it[0])
    if nc < 0:
        raise MemoryError("BFS queue allocation failed")

    return par, par_e, dep, itree, nc


cdef int _bfs_spanning_forest_i64(
    const i64* adj_ptr,
    const i64* adj_idx,
    const i64* adj_edge,
    Py_ssize_t nV,
    Py_ssize_t nE,
    i64* parent,
    i64* parent_edge,
    i64* depth,
    i64* is_tree,
) noexcept nogil:
    """Deterministic BFS spanning forest (int64 variant)."""
    cdef _Queue64 q
    cdef Py_ssize_t v, j, start, end
    cdef i64 u, nbr, eidx
    cdef Py_ssize_t n_components = 0

    if _q64_init(&q, nV) != 0: return -1

    for v in range(nV):
        parent[v] = _SENTINEL
        parent_edge[v] = _SENTINEL
        depth[v] = _SENTINEL
    memset(is_tree, 0, nE * sizeof(i64))

    for v in range(nV):
        if depth[v] != _SENTINEL: continue
        n_components += 1
        depth[v] = 0
        parent[v] = <i64>v
        _q64_push(&q, <i64>v)

        while not _q64_empty(&q):
            u = _q64_pop(&q)
            start = adj_ptr[u]
            end = adj_ptr[u + 1]
            for j in range(start, end):
                nbr = adj_idx[j]
                eidx = adj_edge[j]
                if depth[nbr] == _SENTINEL:
                    depth[nbr] = depth[u] + 1
                    parent[nbr] = u
                    parent_edge[nbr] = eidx
                    is_tree[eidx] = 1
                    _q64_push(&q, nbr)

    _q64_free(&q)
    return <int>n_components


# Cycle tracing via LCA

cdef Py_ssize_t _trace_cycle(
    i32 edge_j,
    const i32* sources,
    const i32* targets,
    const i32* parent,
    const i32* parent_edge,
    const i32* depth,
    i32* out_edges,
    f64* out_signs,
) noexcept nogil:
    """
    Trace the fundamental cycle closed by cotree edge edge_j.

    The cycle consists of:
      (a) the path from u up to LCA(u, v) in the spanning tree,
      (b) the path from LCA(u, v) down to v (reversed),
      (c) the cotree edge (v, u) itself, closing the loop.

    For each edge in the cycle, the orientation sign is:
      +1 if the cycle traverses the edge from sources[e] to targets[e],
      -1 if the cycle traverses from targets[e] to sources[e].

    Returns the cycle length (number of edges).
    """
    cdef i32 u = sources[edge_j], v = targets[edge_j]
    cdef i32 a, b, lca
    cdef Py_ssize_t len_a = 0, len_b = 0, cycle_len, i, k

    cdef i32* path_a_edges = NULL
    cdef i32* path_a_from = NULL
    cdef i32* path_b_edges = NULL
    cdef i32* path_b_from = NULL

    cdef Py_ssize_t max_depth = depth[u] + depth[v] + 2
    path_a_edges = <i32*>malloc(max_depth * sizeof(i32))
    path_a_from = <i32*>malloc(max_depth * sizeof(i32))
    path_b_edges = <i32*>malloc(max_depth * sizeof(i32))
    path_b_from = <i32*>malloc(max_depth * sizeof(i32))

    if (path_a_edges == NULL or path_a_from == NULL or
        path_b_edges == NULL or path_b_from == NULL):
        if path_a_edges != NULL: free(path_a_edges)
        if path_a_from != NULL: free(path_a_from)
        if path_b_edges != NULL: free(path_b_edges)
        if path_b_from != NULL: free(path_b_from)
        return 0

    a = u
    b = v
    while depth[a] > depth[b]:
        path_a_edges[len_a] = parent_edge[a]
        path_a_from[len_a] = a
        len_a += 1
        a = parent[a]
    while depth[b] > depth[a]:
        path_b_edges[len_b] = parent_edge[b]
        path_b_from[len_b] = b
        len_b += 1
        b = parent[b]

    while a != b:
        path_a_edges[len_a] = parent_edge[a]
        path_a_from[len_a] = a
        len_a += 1
        a = parent[a]

        path_b_edges[len_b] = parent_edge[b]
        path_b_from[len_b] = b
        len_b += 1
        b = parent[b]

    lca = a

    # Assemble cycle
    cycle_len = len_a + len_b + 1
    k = 0

    for i in range(len_a):
        out_edges[k] = path_a_edges[i]
        if path_a_from[i] == sources[path_a_edges[i]]:
            out_signs[k] = 1.0
        else:
            out_signs[k] = -1.0
        k += 1

    for i in range(len_b - 1, -1, -1):
        out_edges[k] = path_b_edges[i]
        if parent[path_b_from[i]] == sources[path_b_edges[i]]:
            out_signs[k] = 1.0
        else:
            out_signs[k] = -1.0
        k += 1

    out_edges[k] = edge_j
    if v == sources[edge_j]:
        out_signs[k] = 1.0
    else:
        out_signs[k] = -1.0
    k += 1

    free(path_a_edges)
    free(path_a_from)
    free(path_b_edges)
    free(path_b_from)

    return cycle_len


cdef Py_ssize_t _trace_cycle_i64(
    i64 edge_j,
    const i64* sources,
    const i64* targets,
    const i64* parent,
    const i64* parent_edge,
    const i64* depth,
    i64* out_edges,
    f64* out_signs,
) noexcept nogil:
    """Trace fundamental cycle closed by cotree edge (int64 variant)."""
    cdef i64 u = sources[edge_j], v = targets[edge_j]
    cdef i64 a, b, lca
    cdef Py_ssize_t len_a = 0, len_b = 0, cycle_len, i, k

    cdef i64* path_a_edges = NULL
    cdef i64* path_a_from = NULL
    cdef i64* path_b_edges = NULL
    cdef i64* path_b_from = NULL

    cdef Py_ssize_t max_depth = depth[u] + depth[v] + 2
    path_a_edges = <i64*>malloc(max_depth * sizeof(i64))
    path_a_from = <i64*>malloc(max_depth * sizeof(i64))
    path_b_edges = <i64*>malloc(max_depth * sizeof(i64))
    path_b_from = <i64*>malloc(max_depth * sizeof(i64))

    if (path_a_edges == NULL or path_a_from == NULL or
        path_b_edges == NULL or path_b_from == NULL):
        if path_a_edges != NULL: free(path_a_edges)
        if path_a_from != NULL: free(path_a_from)
        if path_b_edges != NULL: free(path_b_edges)
        if path_b_from != NULL: free(path_b_from)
        return 0

    a = u
    b = v
    while depth[a] > depth[b]:
        path_a_edges[len_a] = parent_edge[a]
        path_a_from[len_a] = a
        len_a += 1
        a = parent[a]
    while depth[b] > depth[a]:
        path_b_edges[len_b] = parent_edge[b]
        path_b_from[len_b] = b
        len_b += 1
        b = parent[b]

    while a != b:
        path_a_edges[len_a] = parent_edge[a]
        path_a_from[len_a] = a
        len_a += 1
        a = parent[a]
        path_b_edges[len_b] = parent_edge[b]
        path_b_from[len_b] = b
        len_b += 1
        b = parent[b]

    lca = a
    cycle_len = len_a + len_b + 1
    k = 0

    for i in range(len_a):
        out_edges[k] = path_a_edges[i]
        if path_a_from[i] == sources[path_a_edges[i]]:
            out_signs[k] = 1.0
        else:
            out_signs[k] = -1.0
        k += 1

    for i in range(len_b - 1, -1, -1):
        out_edges[k] = path_b_edges[i]
        if parent[path_b_from[i]] == sources[path_b_edges[i]]:
            out_signs[k] = 1.0
        else:
            out_signs[k] = -1.0
        k += 1

    out_edges[k] = edge_j
    if v == sources[edge_j]:
        out_signs[k] = 1.0
    else:
        out_signs[k] = -1.0
    k += 1

    free(path_a_edges)
    free(path_a_from)
    free(path_b_edges)
    free(path_b_from)
    return cycle_len


# Cycle basis computation

@cython.boundscheck(False)
@cython.wraparound(False)
def find_fundamental_cycles_i32(Py_ssize_t nV, Py_ssize_t nE,
                                 np.ndarray[i32, ndim=1] sources,
                                 np.ndarray[i32, ndim=1] targets):
    """
    Compute a fundamental cycle basis for the 1-skeleton.

    Parameters
    ----------
    nV : int
        Number of vertices.
    nE : int
        Number of edges.
    sources, targets : int32[nE]
        Tail and head vertex indices for each directed edge.

    Returns
    -------
    cycle_edges : int32[sum(lengths)]
        Concatenated edge indices for all fundamental cycles.
    cycle_signs : float64[sum(lengths)]
        Orientation signs (+/-1.0) for each edge in each cycle.
    cycle_lengths : int32[nF]
        Number of edges in the boundary of each face.
    nF : int
        Number of fundamental cycles (= beta_1).
    n_components : int
        Number of connected components (= beta_0).
    """
    if nV == 0 or nE == 0:
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32), 0, 0)

    adj_ptr, adj_idx, adj_edge = build_symmetric_adjacency_i32(nV, nE, sources, targets)

    cdef i32[::1] ap = adj_ptr, ai = adj_idx, ae = adj_edge
    cdef i32[::1] sv = sources, tv = targets

    # BFS spanning forest
    cdef np.ndarray[i32, ndim=1] par = np.empty(nV, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] par_e = np.empty(nV, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] dep = np.empty(nV, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] itree = np.zeros(nE, dtype=np.int32)
    cdef i32[::1] p_par = par, p_pe = par_e, p_dep = dep, p_it = itree

    cdef int nc
    with nogil:
        nc = _bfs_spanning_forest(&ap[0], &ai[0], &ae[0], nV, nE,
                                  &p_par[0], &p_pe[0], &p_dep[0], &p_it[0])
    if nc < 0:
        raise MemoryError("BFS spanning forest allocation failed")

    cdef Py_ssize_t beta1 = nE - nV + nc
    if beta1 <= 0:
        return (np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32), 0, nc)

    cdef np.ndarray[i32, ndim=1] cotree = np.empty(beta1, dtype=np.int32)
    cdef i32[::1] ct = cotree
    cdef Py_ssize_t ci = 0, j
    for j in range(nE):
        if p_it[j] == 0:
            ct[ci] = <i32>j
            ci += 1
            if ci >= beta1: break

    cdef Py_ssize_t max_depth = 0
    for j in range(nV):
        if p_dep[j] > max_depth: max_depth = p_dep[j]
    cdef Py_ssize_t buf_size = 2 * max_depth + 3

    cdef i32* tmp_edges = <i32*>malloc(buf_size * sizeof(i32))
    cdef f64* tmp_signs = <f64*>malloc(buf_size * sizeof(f64))
    if tmp_edges == NULL or tmp_signs == NULL:
        if tmp_edges != NULL: free(tmp_edges)
        if tmp_signs != NULL: free(tmp_signs)
        raise MemoryError("Cycle tracing buffer allocation failed")

    cdef np.ndarray[i32, ndim=1] clens = np.empty(beta1, dtype=np.int32)
    cdef i32[::1] cl = clens
    cdef Py_ssize_t total_len = 0, clen

    with nogil:
        for j in range(beta1):
            clen = _trace_cycle(ct[j], &sv[0], &tv[0],
                                &p_par[0], &p_pe[0], &p_dep[0],
                                tmp_edges, tmp_signs)
            cl[j] = <i32>clen
            total_len += clen

    cdef np.ndarray[i32, ndim=1] out_edges = np.empty(total_len, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] out_signs = np.empty(total_len, dtype=np.float64)
    cdef i32[::1] oe = out_edges
    cdef f64[::1] os = out_signs

    cdef Py_ssize_t offset = 0, k
    with nogil:
        for j in range(beta1):
            clen = _trace_cycle(ct[j], &sv[0], &tv[0],
                                &p_par[0], &p_pe[0], &p_dep[0],
                                tmp_edges, tmp_signs)
            for k in range(clen):
                oe[offset + k] = tmp_edges[k]
                os[offset + k] = tmp_signs[k]
            offset += clen

    free(tmp_edges)
    free(tmp_signs)

    return out_edges, out_signs, clens, int(beta1), nc


@cython.boundscheck(False)
@cython.wraparound(False)
def find_fundamental_cycles_i64(Py_ssize_t nV, Py_ssize_t nE,
                                 np.ndarray[i64, ndim=1] sources,
                                 np.ndarray[i64, ndim=1] targets):
    """Compute fundamental cycle basis (int64 variant)."""
    if nV == 0 or nE == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64), 0, 0)

    adj_ptr, adj_idx, adj_edge = build_symmetric_adjacency_i64(nV, nE, sources, targets)

    cdef i64[::1] ap = adj_ptr, ai = adj_idx, ae = adj_edge
    cdef i64[::1] sv = sources, tv = targets

    cdef np.ndarray[i64, ndim=1] par = np.empty(nV, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] par_e = np.empty(nV, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] dep = np.empty(nV, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] itree = np.zeros(nE, dtype=np.int64)
    cdef i64[::1] p_par = par, p_pe = par_e, p_dep = dep, p_it = itree

    cdef int nc
    with nogil:
        nc = _bfs_spanning_forest_i64(&ap[0], &ai[0], &ae[0], nV, nE,
                                       &p_par[0], &p_pe[0], &p_dep[0], &p_it[0])
    if nc < 0:
        raise MemoryError("BFS spanning forest allocation failed")

    cdef Py_ssize_t beta1 = nE - nV + nc
    if beta1 <= 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64), 0, nc)

    cdef np.ndarray[i64, ndim=1] cotree = np.empty(beta1, dtype=np.int64)
    cdef i64[::1] ct = cotree
    cdef Py_ssize_t ci = 0, j
    for j in range(nE):
        if p_it[j] == 0:
            ct[ci] = <i64>j
            ci += 1
            if ci >= beta1: break

    cdef Py_ssize_t max_depth = 0
    for j in range(nV):
        if p_dep[j] > max_depth: max_depth = p_dep[j]
    cdef Py_ssize_t buf_size = 2 * max_depth + 3

    cdef i64* tmp_edges = <i64*>malloc(buf_size * sizeof(i64))
    cdef f64* tmp_signs = <f64*>malloc(buf_size * sizeof(f64))
    if tmp_edges == NULL or tmp_signs == NULL:
        if tmp_edges != NULL: free(tmp_edges)
        if tmp_signs != NULL: free(tmp_signs)
        raise MemoryError("Cycle tracing buffer allocation failed")

    cdef np.ndarray[i64, ndim=1] clens = np.empty(beta1, dtype=np.int64)
    cdef i64[::1] cl = clens
    cdef Py_ssize_t total_len = 0, clen

    with nogil:
        for j in range(beta1):
            clen = _trace_cycle_i64(ct[j], &sv[0], &tv[0],
                                     &p_par[0], &p_pe[0], &p_dep[0],
                                     tmp_edges, tmp_signs)
            cl[j] = <i64>clen
            total_len += clen

    cdef np.ndarray[i64, ndim=1] out_edges = np.empty(total_len, dtype=np.int64)
    cdef np.ndarray[f64, ndim=1] out_signs = np.empty(total_len, dtype=np.float64)
    cdef i64[::1] oe = out_edges
    cdef f64[::1] os = out_signs

    cdef Py_ssize_t offset = 0, k
    with nogil:
        for j in range(beta1):
            clen = _trace_cycle_i64(ct[j], &sv[0], &tv[0],
                                     &p_par[0], &p_pe[0], &p_dep[0],
                                     tmp_edges, tmp_signs)
            for k in range(clen):
                oe[offset + k] = tmp_edges[k]
                os[offset + k] = tmp_signs[k]
            offset += clen

    free(tmp_edges)
    free(tmp_signs)

    return out_edges, out_signs, clens, int(beta1), nc


# Dispatchers

def find_fundamental_cycles(Py_ssize_t nV, Py_ssize_t nE, sources, targets):
    """
    Compute a fundamental cycle basis for the 1-skeleton.

    Parameters
    ----------
    nV, nE : int
        Vertex and edge counts.
    sources, targets : array-like of int
        Tail and head vertex indices for each directed edge.

    Returns
    -------
    cycle_edges : int32[sum(lengths)]
        Concatenated edge indices for all fundamental cycles.
    cycle_signs : float64[sum(lengths)]
        Orientation signs (+/-1.0) for each boundary edge.
    cycle_lengths : int32[nF]
        Boundary length of each face.
    nF : int
        Number of fundamental cycles (= beta_1).
    n_components : int
        Number of connected components (= beta_0).
    """
    if not isinstance(sources, np.ndarray):
        sources = np.asarray(sources)
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets)

    if sources.dtype == np.int64 or targets.dtype == np.int64 or max(nV, nE) >= MAX_INT32_NNZ:
        return find_fundamental_cycles_i64(
            nV, nE,
            sources.astype(np.int64, copy=False),
            targets.astype(np.int64, copy=False))
    return find_fundamental_cycles_i32(
        nV, nE,
        sources.astype(np.int32, copy=False),
        targets.astype(np.int32, copy=False))


def cycle_space_dimension(Py_ssize_t nV, Py_ssize_t nE, sources, targets):
    """
    Compute beta_1 = m - n + beta_0 without tracing cycles.

    Uses union-find for component counting.
    """
    if not isinstance(sources, np.ndarray):
        sources = np.asarray(sources)
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets)

    cdef Py_ssize_t nc, j
    cdef Py_ssize_t beta1
    cdef i64[::1] s64, t64
    cdef UnionFind64 uf64
    cdef i32[::1] s32, t32
    cdef UnionFind uf32

    if sources.dtype == np.int64 or targets.dtype == np.int64 or max(nV, nE) >= MAX_INT32_NNZ:
        sv64 = sources.astype(np.int64, copy=False)
        tv64 = targets.astype(np.int64, copy=False)
        s64 = sv64
        t64 = tv64
        if uf64_init(&uf64, nV) != 0:
            raise MemoryError("Union-Find allocation failed")
        with nogil:
            for j in range(nE):
                uf64_union(&uf64, s64[j], t64[j])
        nc = uf64.n_components
        uf64_free(&uf64)
    else:
        sv32 = sources.astype(np.int32, copy=False)
        tv32 = targets.astype(np.int32, copy=False)
        s32 = sv32
        t32 = tv32
        if uf_init(&uf32, nV) != 0:
            raise MemoryError("Union-Find allocation failed")
        with nogil:
            for j in range(nE):
                uf_union(&uf32, s32[j], t32[j])
        nc = uf32.n_components
        uf_free(&uf32)

    beta1 = nE - nV + nc
    return max(0, beta1)


def build_adjacency_and_forest(Py_ssize_t nV, Py_ssize_t nE, sources, targets):
    """
    Build symmetric adjacency and BFS spanning forest.

    Exposes intermediate results for inspection and testing.

    Returns
    -------
    adj_ptr, adj_idx, adj_edge : int32 arrays
        Symmetric CSR adjacency with sorted rows.
    parent, parent_edge, depth, is_tree : int32 arrays
        BFS spanning forest data.
    n_components : int
        Number of connected components.
    """
    if not isinstance(sources, np.ndarray):
        sources = np.asarray(sources, dtype=np.int32)
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets, dtype=np.int32)

    s = sources.astype(np.int32, copy=False)
    t = targets.astype(np.int32, copy=False)

    adj_ptr, adj_idx, adj_edge = build_symmetric_adjacency_i32(nV, nE, s, t)
    par, par_e, dep, itree, nc = bfs_spanning_forest(adj_ptr, adj_idx, adj_edge, nV, nE)

    return adj_ptr, adj_idx, adj_edge, par, par_e, dep, itree, nc


def verify_cycles_in_kernel(Py_ssize_t nV, Py_ssize_t nE,
                            sources, targets,
                            cycle_edges, cycle_signs, cycle_lengths,
                            double tol=1e-10):
    """
    Verify that every cycle lies in ker(B1).

    For each cycle, constructs the signed edge vector and checks that
    B1 @ vector = 0. Returns (ok, max_error).
    """
    if not isinstance(sources, np.ndarray):
        sources = np.asarray(sources, dtype=np.int32)
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets, dtype=np.int32)

    cdef Py_ssize_t nF = len(cycle_lengths)
    cdef np.ndarray[f64, ndim=1] vec = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] result = np.zeros(nV, dtype=np.float64)
    cdef f64[::1] vv = vec, rv = result
    cdef i32[::1] sv = sources.astype(np.int32, copy=False)
    cdef i32[::1] tv = targets.astype(np.int32, copy=False)
    cdef i32[::1] ce = np.asarray(cycle_edges, dtype=np.int32)
    cdef f64[::1] cs = np.asarray(cycle_signs, dtype=np.float64)
    cdef i32[::1] cl = np.asarray(cycle_lengths, dtype=np.int32)

    cdef double max_err = 0.0, abs_val
    cdef Py_ssize_t f, k, offset, eidx
    cdef i32 src, tgt

    offset = 0
    for f in range(nF):
        for k in range(nE): vv[k] = 0.0
        for k in range(cl[f]):
            vv[ce[offset + k]] = cs[offset + k]

        for k in range(nV): rv[k] = 0.0
        for k in range(nE):
            if vv[k] != 0.0:
                src = sv[k]
                tgt = tv[k]
                rv[src] += -1.0 * vv[k]
                rv[tgt] += 1.0 * vv[k]

        for k in range(nV):
            abs_val = rv[k] if rv[k] >= 0 else -rv[k]
            if abs_val > max_err: max_err = abs_val

        offset += cl[f]

    return (max_err < tol), max_err
