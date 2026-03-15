# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._persistence - Persistent homology on the rex chain complex.

Given a 2-rex and a filtration function on cells, computes persistence
pairs tracking birth and death of homological features.

Filtration sources include vertex/edge/face signals, Hodge components,
Laplacian eigenvectors, Jaccard overlap, temporal appearance order,
and spectral layout distances.

Column reduction (left-to-right) over Z/2 or Z coefficients. The
combined boundary matrix merges B1 and B2 into a single operator D
indexed by (filtration_value, dimension, cell_index).
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport fabs, sqrt, log2, INFINITY

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,

    get_EPSILON_NORM,
)

np.import_array()

# Finite sentinel for filtration comparisons and "never dies" pairs.
cdef double _INF = 1e308


# Filtration construction

def filtration_sublevel_vertex(np.ndarray[f64, ndim=1] f0,
                               np.ndarray[i32, ndim=1] boundary_ptr,
                               np.ndarray[i32, ndim=1] boundary_idx,
                               np.ndarray[i32, ndim=1] B2_col_ptr,
                               np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Sublevel filtration from a vertex signal f0.

    f(v) = f0(v)
    f(e) = max over boundary vertices of f0
    f(f) = max over boundary edges of f(e)

    Uses general boundary so branching edges take the max over all
    boundary vertices.

    Parameters
    ----------
    f0 : f64[nV]
        Vertex signal.
    boundary_ptr : i32[nE+1]
        General edge boundary CSR pointer.
    boundary_idx : i32[nnz]
        Boundary vertex indices.
    B2_col_ptr, B2_row_idx : CSC of B2

    Returns
    -------
    filt_v : f64[nV], filt_e : f64[nE], filt_f : f64[nF]
    """
    cdef Py_ssize_t nV = f0.shape[0]
    cdef Py_ssize_t nE = boundary_ptr.shape[0] - 1
    cdef Py_ssize_t nF = B2_col_ptr.shape[0] - 1 if B2_col_ptr.shape[0] > 1 else 0
    cdef f64[::1] fv = f0
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx
    cdef Py_ssize_t e, f, j
    cdef f64 mx

    cdef np.ndarray[f64, ndim=1] filt_v = f0.copy()

    cdef np.ndarray[f64, ndim=1] filt_e = np.empty(nE, dtype=np.float64)
    cdef f64[::1] fev = filt_e
    for e in range(nE):
        mx = -_INF
        for j in range(bp[e], bp[e + 1]):
            if fv[bi[j]] > mx:
                mx = fv[bi[j]]
        fev[e] = mx if mx > -_INF else 0.0

    cdef np.ndarray[f64, ndim=1] filt_f = np.empty(nF, dtype=np.float64)
    cdef f64[::1] ffv = filt_f
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    for f in range(nF):
        mx = -_INF
        for j in range(cp[f], cp[f + 1]):
            if fev[ri[j]] > mx:
                mx = fev[ri[j]]
        ffv[f] = mx

    return filt_v, filt_e, filt_f


def filtration_sublevel_edge(np.ndarray[f64, ndim=1] f1,
                             Py_ssize_t nV,
                             np.ndarray[i32, ndim=1] v2e_ptr,
                             np.ndarray[i32, ndim=1] v2e_idx,
                             np.ndarray[i32, ndim=1] B2_col_ptr,
                             np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Sublevel filtration from an edge signal f1.

    f(v) = min over incident edges of f1(e)
    f(e) = f1(e)
    f(f) = max_{e in boundary(f)} f1(e)

    """
    cdef Py_ssize_t nE = f1.shape[0]
    cdef Py_ssize_t nF = B2_col_ptr.shape[0] - 1 if B2_col_ptr.shape[0] > 1 else 0
    cdef f64[::1] fe = f1
    cdef i32[::1] vep = v2e_ptr, vei = v2e_idx
    cdef Py_ssize_t v, e, f, j
    cdef f64 mn, mx

    cdef np.ndarray[f64, ndim=1] filt_v = np.full(nV, _INF, dtype=np.float64)
    cdef f64[::1] fvv = filt_v
    for v in range(nV):
        mn = _INF
        for j in range(vep[v], vep[v + 1]):
            if fe[vei[j]] < mn:
                mn = fe[vei[j]]
        fvv[v] = mn

    cdef np.ndarray[f64, ndim=1] filt_e = f1.copy()

    cdef np.ndarray[f64, ndim=1] filt_f = np.empty(nF, dtype=np.float64)
    cdef f64[::1] ffv = filt_f
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    for f in range(nF):
        mx = -_INF
        for j in range(cp[f], cp[f + 1]):
            if fe[ri[j]] > mx:
                mx = fe[ri[j]]
        ffv[f] = mx

    return filt_v, filt_e, filt_f


def filtration_sublevel_face(np.ndarray[f64, ndim=1] f2,
                             Py_ssize_t nV, Py_ssize_t nE,
                             np.ndarray[i32, ndim=1] boundary_ptr,
                             np.ndarray[i32, ndim=1] boundary_idx,
                             np.ndarray[i32, ndim=1] B2_col_ptr,
                             np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Sublevel filtration from a face signal f2.

    f(f) = f2(f)
    f(e) = min over incident faces of f2(f)
    f(v) = min over incident edges of f(e)
    """
    cdef Py_ssize_t nF = f2.shape[0]
    cdef f64[::1] ff = f2
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx
    cdef Py_ssize_t f, j, e, v

    cdef np.ndarray[f64, ndim=1] filt_e = np.full(nE, _INF, dtype=np.float64)
    cdef f64[::1] fev = filt_e
    for f in range(nF):
        for j in range(cp[f], cp[f + 1]):
            e = ri[j]
            if ff[f] < fev[e]:
                fev[e] = ff[f]

    cdef np.ndarray[f64, ndim=1] filt_v = np.full(nV, _INF, dtype=np.float64)
    cdef f64[::1] fvv = filt_v
    for e in range(nE):
        for j in range(bp[e], bp[e + 1]):
            v = bi[j]
            if fev[e] < fvv[v]:
                fvv[v] = fev[e]

    return filt_v, filt_e, f2.copy()


def filtration_hodge_component(np.ndarray[f64, ndim=1] grad,
                               np.ndarray[f64, ndim=1] curl,
                               np.ndarray[f64, ndim=1] harmonic,
                               Py_ssize_t nV,
                               np.ndarray[i32, ndim=1] v2e_ptr,
                               np.ndarray[i32, ndim=1] v2e_idx,
                               np.ndarray[i32, ndim=1] B2_col_ptr,
                               np.ndarray[i32, ndim=1] B2_row_idx,
                               Py_ssize_t component=2):
    """
    Filtration based on Hodge decomposition component magnitude.

    component: 0=gradient, 1=curl, 2=harmonic (default).
    Edges with small |component| enter first (they lack that component).
    Edges with large |component| enter last (they are dominated by it).

    """
    cdef np.ndarray[f64, ndim=1] f1
    if component == 0:
        f1 = np.abs(grad)
    elif component == 1:
        f1 = np.abs(curl)
    else:
        f1 = np.abs(harmonic)

    return filtration_sublevel_edge(f1, nV, v2e_ptr, v2e_idx,
                                    B2_col_ptr, B2_row_idx)


def filtration_spectral(np.ndarray[f64, ndim=1] eigenvector,
                        Py_ssize_t nV,
                        np.ndarray[i32, ndim=1] boundary_ptr,
                        np.ndarray[i32, ndim=1] boundary_idx,
                        np.ndarray[i32, ndim=1] v2e_ptr,
                        np.ndarray[i32, ndim=1] v2e_idx,
                        np.ndarray[i32, ndim=1] B2_col_ptr,
                        np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Filtration from a Laplacian eigenvector (e.g., Fiedler vector of L0).

    Uses absolute value of the eigenvector as the filter function.
    Nodes near zero-crossings (partition boundary) enter first.
    """
    cdef np.ndarray[f64, ndim=1] f0 = np.abs(eigenvector)
    return filtration_sublevel_vertex(f0, boundary_ptr, boundary_idx,
                                      B2_col_ptr, B2_row_idx)


def filtration_rips(np.ndarray[f64, ndim=2] positions,
                    np.ndarray[i32, ndim=1] boundary_ptr,
                    np.ndarray[i32, ndim=1] boundary_idx,
                    np.ndarray[i32, ndim=1] B2_col_ptr,
                    np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Vietoris-Rips filtration from vertex positions (e.g., spectral layout).

    f(v) = 0 (all vertices present at start)
    f(e) = max pairwise distance among ALL boundary vertices of e
    f(f) = max_{e in boundary(f)} f(e)

    For branching edges, uses the diameter of the boundary point set.
    """
    cdef Py_ssize_t nV = positions.shape[0]
    cdef Py_ssize_t nE = boundary_ptr.shape[0] - 1
    cdef Py_ssize_t nF = B2_col_ptr.shape[0] - 1 if B2_col_ptr.shape[0] > 1 else 0
    cdef f64[:, ::1] pos = positions
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx
    cdef Py_ssize_t e, f, j, k, d, ndim = positions.shape[1]
    cdef Py_ssize_t a, b
    cdef f64 dist, dx, mx, max_dist

    cdef np.ndarray[f64, ndim=1] filt_v = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] filt_e = np.empty(nE, dtype=np.float64)
    cdef f64[::1] fev = filt_e

    for e in range(nE):
        max_dist = 0.0
        for j in range(bp[e], bp[e + 1]):
            a = bi[j]
            for k in range(j + 1, bp[e + 1]):
                b = bi[k]
                dist = 0.0
                for d in range(ndim):
                    dx = pos[a, d] - pos[b, d]
                    dist += dx * dx
                dist = sqrt(dist)
                if dist > max_dist:
                    max_dist = dist
        fev[e] = max_dist

    cdef np.ndarray[f64, ndim=1] filt_f = np.empty(nF, dtype=np.float64)
    cdef f64[::1] ffv = filt_f
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    for f in range(nF):
        mx = -_INF
        for j in range(cp[f], cp[f + 1]):
            if fev[ri[j]] > mx:
                mx = fev[ri[j]]
        ffv[f] = mx

    return filt_v, filt_e, filt_f


def filtration_temporal(list snapshot_sources,
                        list snapshot_targets,
                        Py_ssize_t nV, Py_ssize_t nE,
                        np.ndarray[i32, ndim=1] sources,
                        np.ndarray[i32, ndim=1] targets,
                        np.ndarray[i32, ndim=1] B2_col_ptr,
                        np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Temporal filtration: f(sigma) = first time step where sigma appears.

    Maps directly from _temporal.pyx snapshot data.
    Edges are identified by (src, tgt) pairs across snapshots.
    """
    cdef Py_ssize_t T = len(snapshot_sources), t, e, j
    cdef i32[::1] src = sources, tgt = targets

    edge_first = {}
    vertex_first = {}

    for t in range(T):
        src_t = snapshot_sources[t]
        tgt_t = snapshot_targets[t]
        for j in range(len(src_t)):
            s = int(src_t[j])
            g = int(tgt_t[j])
            key = (s, g)
            if key not in edge_first:
                edge_first[key] = t
            if s not in vertex_first:
                vertex_first[s] = t
            if g not in vertex_first:
                vertex_first[g] = t

    cdef np.ndarray[f64, ndim=1] filt_v = np.full(nV, <f64>T, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] filt_e = np.full(nE, <f64>T, dtype=np.float64)
    cdef f64[::1] fvv = filt_v, fev = filt_e

    for v in range(nV):
        if v in vertex_first:
            fvv[v] = <f64>vertex_first[v]

    for e in range(nE):
        key = (int(src[e]), int(tgt[e]))
        if key in edge_first:
            fev[e] = <f64>edge_first[key]

    cdef Py_ssize_t nF = B2_col_ptr.shape[0] - 1 if B2_col_ptr.shape[0] > 1 else 0
    cdef np.ndarray[f64, ndim=1] filt_f = np.empty(nF, dtype=np.float64)
    cdef f64[::1] ffv = filt_f
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef f64 mx
    for f in range(nF):
        mx = -_INF
        for j in range(cp[f], cp[f + 1]):
            if fev[ri[j]] > mx:
                mx = fev[ri[j]]
        ffv[f] = mx

    return filt_v, filt_e, filt_f


def filtration_temporal_general(list snapshots,
                                Py_ssize_t nV, Py_ssize_t nE,
                                np.ndarray[i32, ndim=1] boundary_ptr,
                                np.ndarray[i32, ndim=1] boundary_idx,
                                np.ndarray[i32, ndim=1] B2_col_ptr,
                                np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Temporal filtration for general boundary snapshots.

    Each snapshot is (boundary_ptr, boundary_idx).
    Edge identity = sorted tuple of boundary vertices.
    Handles branching edges, self-loops, witness edges.

    Parameters
    ----------
    snapshots : list of (boundary_ptr, boundary_idx) per timestep
    nV, nE : vertex/edge count of the canonical rex
    boundary_ptr, boundary_idx : general boundary of the canonical rex
    B2_col_ptr, B2_row_idx : B2 CSC of the canonical rex

    Returns
    -------
    filt_v, filt_e, filt_f : f64 arrays
    """
    cdef Py_ssize_t T = len(snapshots), t, e, j
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx

    edge_first = {}
    vertex_first = {}

    for t in range(T):
        bp_t, bi_t = snapshots[t]
        nE_t = bp_t.shape[0] - 1
        for e_t in range(nE_t):
            bverts = []
            for j in range(bp_t[e_t], bp_t[e_t + 1]):
                v = int(bi_t[j])
                bverts.append(v)
                if v not in vertex_first:
                    vertex_first[v] = t
            bverts.sort()
            key = tuple(bverts)
            if key not in edge_first:
                edge_first[key] = t

    cdef np.ndarray[f64, ndim=1] filt_v = np.full(nV, <f64>T, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] filt_e = np.full(nE, <f64>T, dtype=np.float64)
    cdef f64[::1] fvv = filt_v, fev = filt_e

    for v in range(nV):
        if v in vertex_first:
            fvv[v] = <f64>vertex_first[v]

    for e in range(nE):
        bverts = []
        for j in range(bp[e], bp[e + 1]):
            bverts.append(int(bi[j]))
        bverts.sort()
        key = tuple(bverts)
        if key in edge_first:
            fev[e] = <f64>edge_first[key]

    cdef Py_ssize_t nF = B2_col_ptr.shape[0] - 1 if B2_col_ptr.shape[0] > 1 else 0
    cdef np.ndarray[f64, ndim=1] filt_f = np.empty(nF, dtype=np.float64)
    cdef f64[::1] ffv = filt_f
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef f64 mx
    for f in range(nF):
        mx = -_INF
        for j in range(cp[f], cp[f + 1]):
            if fev[ri[j]] > mx:
                mx = fev[ri[j]]
        ffv[f] = mx

    return filt_v, filt_e, filt_f


def filtration_dimension(Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF):
    """
    Canonical dimension filtration: vertices=0, edges=1, faces=2.

    """
    return (np.zeros(nV, dtype=np.float64),
            np.ones(nE, dtype=np.float64),
            np.full(nF, 2.0, dtype=np.float64))


# Boundary matrix construction

def build_filtration_order(np.ndarray[f64, ndim=1] filt_v,
                           np.ndarray[f64, ndim=1] filt_e,
                           np.ndarray[f64, ndim=1] filt_f):
    """
    Sort all cells by (filtration_value, dimension, original_index).

    Returns
    -------
    order : i64[N]
        Permutation of [0, N) where N = nV + nE + nF.
    cell_dim : i32[N]
        Dimension of each cell in sorted order.
    cell_idx : i32[N]
        Original index within its dimension.
    filt_vals : f64[N]
        Filtration value of each cell in sorted order.

    Convention: cells 0..nV-1 are vertices, nV..nV+nE-1 are edges,
    nV+nE..nV+nE+nF-1 are faces in the pre-sorted ordering.
    """
    cdef Py_ssize_t nV = filt_v.shape[0], nE = filt_e.shape[0]
    cdef Py_ssize_t nF = filt_f.shape[0], N = nV + nE + nF
    cdef Py_ssize_t i

    cdef np.ndarray[f64, ndim=1] all_filt = np.empty(N, dtype=np.float64)
    cdef np.ndarray[i32, ndim=1] all_dim = np.empty(N, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] all_idx = np.empty(N, dtype=np.int32)

    cdef f64[::1] afv = all_filt
    cdef i32[::1] adv = all_dim, aiv = all_idx

    for i in range(nV):
        afv[i] = filt_v[i]
        adv[i] = 0
        aiv[i] = <i32>i

    for i in range(nE):
        afv[nV + i] = filt_e[i]
        adv[nV + i] = 1
        aiv[nV + i] = <i32>i

    for i in range(nF):
        afv[nV + nE + i] = filt_f[i]
        adv[nV + nE + i] = 2
        aiv[nV + nE + i] = <i32>i

    sort_keys = np.lexsort((all_idx.astype(np.int64),
                            all_dim.astype(np.int64),
                            all_filt))
    order = sort_keys.astype(np.int64)

    return order, all_dim[order].copy(), all_idx[order].copy(), all_filt[order].copy()


def build_boundary_matrix(np.ndarray[i64, ndim=1] order,
                          np.ndarray[i32, ndim=1] cell_dim,
                          np.ndarray[i32, ndim=1] cell_idx,
                          Py_ssize_t nV, Py_ssize_t nE,
                          np.ndarray[i32, ndim=1] boundary_ptr,
                          np.ndarray[i32, ndim=1] boundary_idx,
                          np.ndarray[i32, ndim=1] B2_col_ptr,
                          np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Build the combined boundary matrix D as a list of columns.

    Uses general boundary (boundary_ptr/boundary_idx) for all edge types.
    D is stored as a list of sorted row-index sets. Over Z/2, only
    membership matters, not signs.

    Returns
    -------
    boundary_cols : list of list of int
        D[j] = sorted boundary indices.
    """
    cdef Py_ssize_t N = order.shape[0], j, k, row
    cdef i32[::1] cdim = cell_dim, cidx = cell_idx
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx

    pos_map_v = {}
    pos_map_e = {}

    for j in range(N):
        if cdim[j] == 0:
            pos_map_v[int(cidx[j])] = j
        elif cdim[j] == 1:
            pos_map_e[int(cidx[j])] = j

    boundary_cols = [[] for _ in range(N)]

    for j in range(N):
        if cdim[j] == 0:
            pass

        elif cdim[j] == 1:
            e = int(cidx[j])
            rows = set()
            for k in range(bp[e], bp[e + 1]):
                v = int(bi[k])
                if v in pos_map_v:
                    pos = pos_map_v[v]
                    if pos in rows:
                        rows.discard(pos)
                    else:
                        rows.add(pos)
            boundary_cols[j] = sorted(rows)

        elif cdim[j] == 2:
            f = int(cidx[j])
            rows = set()
            if cp.shape[0] > f + 1:
                for k in range(cp[f], cp[f + 1]):
                    edge_orig = int(ri[k])
                    if edge_orig in pos_map_e:
                        pos = pos_map_e[edge_orig]
                        if pos in rows:
                            rows.discard(pos)
                        else:
                            rows.add(pos)
            boundary_cols[j] = sorted(rows)

    return boundary_cols


# Column reduction

def reduce_boundary_matrix_mod2(list boundary_cols):
    """
    Standard left-to-right column reduction over Z/2.

    The persistence algorithm: for each column j (left to right),
    if another column j' < j has the same pivot (lowest nonzero row),
    XOR column j' into column j. Repeat until pivot is unique or column
    is zero.

    Parameters
    ----------
    boundary_cols : list of list of int
        Sparse columns (sorted row indices).

    Returns
    -------
    reduced : list of list of int
        Reduced boundary matrix columns.
    pivot_to_col : dict
        Maps pivot_row -> column_index.
    """
    cdef Py_ssize_t N = len(boundary_cols), j
    cdef Py_ssize_t pivot_row

    reduced = [set(col) for col in boundary_cols]
    pivot_to_col = {}

    for j in range(N):
        while True:
            if not reduced[j]:
                break

            pivot_row = max(reduced[j])  # Lowest = max in our convention

            if pivot_row not in pivot_to_col:
                pivot_to_col[pivot_row] = j
                break
            else:
                other = pivot_to_col[pivot_row]
                reduced[j] = reduced[j] ^ reduced[other]

    reduced_lists = [sorted(s) for s in reduced]
    return reduced_lists, pivot_to_col


def reduce_boundary_matrix(list boundary_cols,
                           np.ndarray[i32, ndim=1] cell_dim):
    """
    Column reduction over Z (with signs tracked).

    Uses integer coefficients and proper sign arithmetic.
    Slower than Z/2 but preserves orientation information.

    Returns
    -------
    reduced : list of list of (int, int)
        (row_idx, coefficient) pairs.
    pivot_to_col : dict
    """
    cdef Py_ssize_t N = len(boundary_cols), j
    cdef Py_ssize_t pivot_row, other

    cols = []
    for j in range(N):
        col_dict = {}
        for r in boundary_cols[j]:
            col_dict[r] = col_dict.get(r, 0) + 1
        cols.append(col_dict)

    pivot_to_col = {}

    for j in range(N):
        while True:
            cols[j] = {r: c for r, c in cols[j].items() if c != 0}

            if not cols[j]:
                break

            pivot_row = max(cols[j].keys())

            if pivot_row not in pivot_to_col:
                pivot_to_col[pivot_row] = j
                break
            else:
                other = pivot_to_col[pivot_row]
                c_j = cols[j][pivot_row]
                c_o = cols[other][pivot_row]
                new_col = {}
                for r, c in cols[j].items():
                    new_col[r] = new_col.get(r, 0) + c_o * c
                for r, c in cols[other].items():
                    new_col[r] = new_col.get(r, 0) - c_j * c
                cols[j] = {r: c for r, c in new_col.items() if c != 0}

    reduced = [sorted(col.items()) for col in cols]
    return reduced, pivot_to_col


# Persistence pair extraction

def extract_persistence_pairs(dict pivot_to_col,
                              np.ndarray[i32, ndim=1] cell_dim,
                              np.ndarray[f64, ndim=1] filt_vals,
                              np.ndarray[i32, ndim=1] cell_idx,
                              Py_ssize_t N):
    """
    Extract persistence pairs (birth, death, dimension) from reduced matrix.

    A column j that reduces to zero: cell j is a creator (birth).
    A column j with pivot at row i: cell j kills the class born at i.

    Returns
    -------
    pairs : f64[n_pairs, 5]
        Each row: [birth_value, death_value, dimension, birth_cell_idx, death_cell_idx]
        birth_cell_idx and death_cell_idx are original indices in their dimension.
    essential : f64[n_ess, 3]
        Classes that never die: [birth_value, inf, dimension]
    """
    destroyer_set = set(pivot_to_col.values())
    creator_killed = set(pivot_to_col.keys())

    pairs_list = []
    essential_list = []

    cdef i32[::1] cdim = cell_dim
    cdef f64[::1] fvals = filt_vals
    cdef i32[::1] cidx = cell_idx
    cdef Py_ssize_t j

    for j in range(N):
        if j in destroyer_set:
            continue

        if j in creator_killed:
            continue

        pass

    for creator_pos, destroyer_pos in pivot_to_col.items():
        b_val = float(fvals[creator_pos])
        d_val = float(fvals[destroyer_pos])
        dim = int(cdim[creator_pos])
        b_idx = int(cidx[creator_pos])
        d_idx = int(cidx[destroyer_pos])
        pairs_list.append([b_val, d_val, dim, b_idx, d_idx])

    paired_as_creator = set(pivot_to_col.keys())
    paired_as_destroyer = set(pivot_to_col.values())

    for j in range(N):
        if j not in paired_as_creator and j not in paired_as_destroyer:
            essential_list.append([float(fvals[j]), _INF, int(cdim[j])])

    pairs = np.array(pairs_list, dtype=np.float64) if pairs_list else np.zeros((0, 5), dtype=np.float64)
    essential = np.array(essential_list, dtype=np.float64) if essential_list else np.zeros((0, 3), dtype=np.float64)

    return pairs, essential


def persistence_diagram(np.ndarray[f64, ndim=1] filt_v,
                        np.ndarray[f64, ndim=1] filt_e,
                        np.ndarray[f64, ndim=1] filt_f,
                        np.ndarray[i32, ndim=1] boundary_ptr,
                        np.ndarray[i32, ndim=1] boundary_idx,
                        np.ndarray[i32, ndim=1] B2_col_ptr,
                        np.ndarray[i32, ndim=1] B2_row_idx):
    """
    Full persistence diagram computation from filtration values.


    Returns
    -------
    dict with keys:
      'pairs'     : f64[n, 5] - [birth, death, dim, birth_cell, death_cell]
      'essential' : f64[n, 3] - [birth, inf, dim]
      'betti'     : (beta0, beta1, beta2) at the final filtration step
      'order'     : i64[N] - filtration ordering
    """
    cdef Py_ssize_t nV = filt_v.shape[0]
    cdef Py_ssize_t nE = filt_e.shape[0]
    cdef Py_ssize_t nF = filt_f.shape[0]

    order, cell_dim, cell_idx, filt_vals = build_filtration_order(
        filt_v, filt_e, filt_f)

    boundary_cols = build_boundary_matrix(
        order, cell_dim, cell_idx, nV, nE,
        boundary_ptr, boundary_idx, B2_col_ptr, B2_row_idx)

    reduced, pivot_to_col = reduce_boundary_matrix_mod2(boundary_cols)

    N = nV + nE + nF
    pairs, essential = extract_persistence_pairs(
        pivot_to_col, cell_dim, filt_vals, cell_idx, N)

    b0 = b1 = b2 = 0
    for i in range(essential.shape[0]):
        d = int(essential[i, 2])
        if d == 0: b0 += 1
        elif d == 1: b1 += 1
        elif d == 2: b2 += 1

    return {
        'pairs': pairs,
        'essential': essential,
        'betti': (b0, b1, b2),
        'order': order,
        'cell_dim': cell_dim,
        'cell_idx': cell_idx,
        'filt_vals': filt_vals,
    }


def persistence_barcodes(np.ndarray[f64, ndim=2] pairs,
                         np.ndarray[f64, ndim=2] essential,
                         Py_ssize_t target_dim=-1):
    """
    Extract barcodes (birth, death) for a specific dimension.

    Parameters
    ----------
    pairs : f64[n, 5]
        From persistence_diagram.
    essential : f64[n, 3]
        From persistence_diagram.
    target_dim : int
        Dimension to filter (-1 = all).

    Returns
    -------
    barcodes : f64[k, 2]
        (birth, death) pairs, sorted by persistence.
    """
    bars = []

    for i in range(pairs.shape[0]):
        if target_dim >= 0 and int(pairs[i, 2]) != target_dim:
            continue
        bars.append([pairs[i, 0], pairs[i, 1]])

    for i in range(essential.shape[0]):
        if target_dim >= 0 and int(essential[i, 2]) != target_dim:
            continue
        bars.append([essential[i, 0], essential[i, 1]])

    if not bars:
        return np.zeros((0, 2), dtype=np.float64)

    result = np.array(bars, dtype=np.float64)
    persistence = result[:, 1] - result[:, 0]
    persistence[np.isinf(persistence)] = 1e300
    order = np.argsort(-persistence)
    return result[order]


# Diagram distances

def bottleneck_distance(np.ndarray[f64, ndim=2] dgm1,
                        np.ndarray[f64, ndim=2] dgm2):
    """
    Approximate bottleneck distance between two persistence diagrams.

    Uses a greedy matching heuristic (not optimal, but O(n^2) vs O(n^3)
    for the exact Hungarian algorithm). Sufficient for most applications.

    Parameters
    ----------
    dgm1, dgm2 : f64[n, 2]
        (birth, death) pairs, finite entries only.

    Returns
    -------
    distance : float
    """
    cdef Py_ssize_t n1 = dgm1.shape[0], n2 = dgm2.shape[0]
    cdef Py_ssize_t i, j

    if n1 == 0 and n2 == 0:
        return 0.0

    diag_costs_1 = np.array([(dgm1[i, 1] - dgm1[i, 0]) / 2.0
                              for i in range(n1) if not np.isinf(dgm1[i, 1])])
    diag_costs_2 = np.array([(dgm2[i, 1] - dgm2[i, 0]) / 2.0
                              for i in range(n2) if not np.isinf(dgm2[i, 1])])

    finite_1 = dgm1[~np.isinf(dgm1[:, 1])] if n1 > 0 else np.zeros((0, 2))
    finite_2 = dgm2[~np.isinf(dgm2[:, 1])] if n2 > 0 else np.zeros((0, 2))

    cdef Py_ssize_t m1 = finite_1.shape[0], m2 = finite_2.shape[0]

    if m1 == 0 and m2 == 0:
        return 0.0

    used1 = set()
    used2 = set()
    cdef f64 max_cost = 0.0, cost, best_cost, d_b, d_d

    for _ in range(min(m1, m2)):
        best_i = -1
        best_j = -1
        best_cost = _INF
        for i in range(m1):
            if i in used1:
                continue
            for j in range(m2):
                if j in used2:
                    continue
                d_b = fabs(finite_1[i, 0] - finite_2[j, 0])
                d_d = fabs(finite_1[i, 1] - finite_2[j, 1])
                cost = d_b if d_b > d_d else d_d
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
                    best_j = j
        if best_i >= 0:
            used1.add(best_i)
            used2.add(best_j)
            if best_cost > max_cost:
                max_cost = best_cost

    for i in range(m1):
        if i not in used1:
            cost = (finite_1[i, 1] - finite_1[i, 0]) / 2.0
            if cost > max_cost:
                max_cost = cost

    for j in range(m2):
        if j not in used2:
            cost = (finite_2[j, 1] - finite_2[j, 0]) / 2.0
            if cost > max_cost:
                max_cost = cost

    return max_cost


def wasserstein_distance(np.ndarray[f64, ndim=2] dgm1,
                         np.ndarray[f64, ndim=2] dgm2,
                         f64 p=2.0):
    """
    Approximate Wasserstein-p distance between persistence diagrams.

    Uses greedy matching (same caveat as bottleneck_distance).

    Returns sum_{matched pairs} max(|b1-b2|, |d1-d2|)^p + diagonal costs.
    """
    cdef Py_ssize_t n1 = dgm1.shape[0], n2 = dgm2.shape[0]

    if n1 == 0 and n2 == 0:
        return 0.0

    finite_1 = dgm1[~np.isinf(dgm1[:, 1])] if n1 > 0 else np.zeros((0, 2))
    finite_2 = dgm2[~np.isinf(dgm2[:, 1])] if n2 > 0 else np.zeros((0, 2))

    cdef Py_ssize_t m1 = finite_1.shape[0], m2 = finite_2.shape[0]
    cdef f64 total = 0.0, cost, best_cost, d_b, d_d
    cdef Py_ssize_t i, j

    used1 = set()
    used2 = set()

    for _ in range(min(m1, m2)):
        best_i = -1
        best_j = -1
        best_cost = _INF
        for i in range(m1):
            if i in used1:
                continue
            for j in range(m2):
                if j in used2:
                    continue
                d_b = fabs(finite_1[i, 0] - finite_2[j, 0])
                d_d = fabs(finite_1[i, 1] - finite_2[j, 1])
                cost = d_b if d_b > d_d else d_d
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
                    best_j = j
        if best_i >= 0:
            used1.add(best_i)
            used2.add(best_j)
            total += best_cost ** p

    for i in range(m1):
        if i not in used1:
            total += ((finite_1[i, 1] - finite_1[i, 0]) / 2.0) ** p

    for j in range(m2):
        if j not in used2:
            total += ((finite_2[j, 1] - finite_2[j, 0]) / 2.0) ** p

    return total ** (1.0 / p)


# Rex-specific enrichment

def enrich_pairs_edge_type(np.ndarray[f64, ndim=2] pairs,
                           np.ndarray[np.uint8_t, ndim=1] edge_types):
    """
    Annotate dim-1 persistence pairs with edge type at birth and death.

    Type codes: 0=standard, 1=self-loop, 2=branching, 3=witness.

    Returns
    -------
    annotations : i32[n_pairs, 2]
        [birth_edge_type, death_edge_type].
        -1 for non-edge cells.
    """
    cdef Py_ssize_t n = pairs.shape[0], i
    cdef np.ndarray[i32, ndim=2] ann = np.full((n, 2), -1, dtype=np.int32)
    cdef np.uint8_t[::1] et = edge_types
    cdef i32 b_idx, d_idx

    for i in range(n):
        if int(pairs[i, 2]) == 1:  # dim-1 pair
            b_idx = <i32>pairs[i, 3]
            d_idx = <i32>pairs[i, 4]
            if b_idx >= 0 and b_idx < edge_types.shape[0]:
                ann[i, 0] = <i32>et[b_idx]
            if d_idx >= 0 and d_idx < edge_types.shape[0]:
                ann[i, 1] = <i32>et[d_idx]

    return ann


def enrich_pairs_hodge(np.ndarray[f64, ndim=2] pairs,
                       np.ndarray[f64, ndim=1] grad_energy,
                       np.ndarray[f64, ndim=1] curl_energy,
                       np.ndarray[f64, ndim=1] harm_energy):
    """
    Annotate dim-1 pairs with the dominant Hodge component at the birth edge.

    Parameters
    ----------
    pairs : f64[n, 5]
    grad_energy, curl_energy, harm_energy : f64[nE]
        Per-edge energy in each Hodge component (from _hodge.pyx decomposition).

    Returns
    -------
    dominant : i32[n]
        0=gradient, 1=curl, 2=harmonic, -1=non-edge.
    fractions : f64[n, 3]
        [grad_frac, curl_frac, harm_frac].
    """
    cdef Py_ssize_t n = pairs.shape[0], i
    cdef np.ndarray[i32, ndim=1] dom = np.full(n, -1, dtype=np.int32)
    cdef np.ndarray[f64, ndim=2] frac = np.zeros((n, 3), dtype=np.float64)
    cdef f64 g, c, h, total
    cdef i32 b_idx

    for i in range(n):
        if int(pairs[i, 2]) != 1:
            continue
        b_idx = <i32>pairs[i, 3]
        if b_idx < 0:
            continue
        g = grad_energy[b_idx]
        c = curl_energy[b_idx]
        h = harm_energy[b_idx]
        total = g + c + h
        if total > get_EPSILON_NORM():
            frac[i, 0] = g / total
            frac[i, 1] = c / total
            frac[i, 2] = h / total
        if g >= c and g >= h:
            dom[i] = 0
        elif c >= g and c >= h:
            dom[i] = 1
        else:
            dom[i] = 2

    return dom, frac


def persistence_entropy(np.ndarray[f64, ndim=2] barcodes):
    """
    Persistence entropy: Shannon entropy of normalized barcode lengths.

    H = -sum_i (l_i / L) log2(l_i / L)  where l_i = death_i - birth_i, L = sum l_i.


    Parameters
    ----------
    barcodes : f64[n, 2]
        (birth, death) finite pairs only.

    Returns
    -------
    entropy : float
    """
    cdef Py_ssize_t n = barcodes.shape[0], i
    if n == 0:
        return 0.0

    cdef f64 L = 0.0, li, p, ent = 0.0

    lengths = []
    for i in range(n):
        if not np.isinf(barcodes[i, 1]):
            li = barcodes[i, 1] - barcodes[i, 0]
            if li > get_EPSILON_NORM():
                lengths.append(li)
                L += li

    if L < get_EPSILON_NORM() or len(lengths) == 0:
        return 0.0

    for li in lengths:
        p = li / L
        if p > get_EPSILON_NORM():
            ent -= p * log2(p)

    return ent


# Relative persistence

def relative_persistence(np.ndarray[f64, ndim=1] filt_v,
                         np.ndarray[f64, ndim=1] filt_e,
                         np.ndarray[f64, ndim=1] filt_f,
                         np.ndarray[i32, ndim=1] boundary_ptr,
                         np.ndarray[i32, ndim=1] boundary_idx,
                         np.ndarray[i32, ndim=1] B2_col_ptr,
                         np.ndarray[i32, ndim=1] B2_row_idx,
                         np.ndarray[np.uint8_t, ndim=1] v_mask,
                         np.ndarray[np.uint8_t, ndim=1] e_mask,
                         np.ndarray[np.uint8_t, ndim=1] f_mask):
    """
    Persistent relative homology H_*(R, I) across a filtration.

    Restricts filtration to surviving cells (R/I) and computes
    persistence on the quotient complex.


    Parameters
    ----------
    filt_v, filt_e, filt_f : filtration values on full complex
    boundary_ptr, boundary_idx : general edge boundary representation
    B2_col_ptr, B2_row_idx : B2 in CSC
    v_mask, e_mask, f_mask : subcomplex masks (from _quotient.pyx)

    Returns
    -------
    Same as persistence_diagram, but computed on R/I.
    """
    filt_v_q = filt_v[~v_mask.astype(bool)]
    filt_e_q = filt_e[~e_mask.astype(bool)]
    filt_f_q = filt_f[~f_mask.astype(bool)] if f_mask.shape[0] > 0 else np.zeros(0, dtype=np.float64)

    surviving_edges = np.where(~e_mask.astype(bool))[0]
    surviving_faces = np.where(~f_mask.astype(bool))[0] if f_mask.shape[0] > 0 else np.array([], dtype=np.intp)

    v_old_to_new = np.full(filt_v.shape[0], -1, dtype=np.int32)
    has_collapsed = False
    cdef i32 v_count = 0
    for i in range(filt_v.shape[0]):
        if v_mask[i]:
            has_collapsed = True
        else:
            v_old_to_new[i] = v_count
            v_count += 1

    cdef i32 v_star = v_count if has_collapsed else -1
    if has_collapsed:
        min_collapsed = np.min(filt_v[v_mask.astype(bool)]) if np.any(v_mask) else 0.0
        filt_v_q = np.append(filt_v_q, min_collapsed)

    nV_q = filt_v_q.shape[0]
    nE_q = filt_e_q.shape[0]
    nF_q = filt_f_q.shape[0]

    new_bp = [0]
    new_bi = []
    for idx in range(nE_q):
        e_orig = surviving_edges[idx]
        for j in range(boundary_ptr[e_orig], boundary_ptr[e_orig + 1]):
            v_orig = boundary_idx[j]
            if v_mask[v_orig]:
                v_new = v_star
            else:
                v_new = v_old_to_new[v_orig]
            if v_new >= 0:
                new_bi.append(v_new)
        new_bp.append(len(new_bi))

    bp_q = np.array(new_bp, dtype=np.int32)
    bi_q = np.array(new_bi, dtype=np.int32) if new_bi else np.zeros(0, dtype=np.int32)

    e_old_to_new = np.full(filt_e.shape[0], -1, dtype=np.int32)
    for idx in range(nE_q):
        e_old_to_new[surviving_edges[idx]] = idx

    new_col_ptr = [0]
    new_row_idx = []
    for idx in range(nF_q):
        f_orig = surviving_faces[idx]
        for j in range(B2_col_ptr[f_orig], B2_col_ptr[f_orig + 1]):
            e_orig = B2_row_idx[j]
            e_new = e_old_to_new[e_orig]
            if e_new >= 0:
                new_row_idx.append(e_new)
        new_col_ptr.append(len(new_row_idx))

    B2q_cp = np.array(new_col_ptr, dtype=np.int32)
    B2q_ri = np.array(new_row_idx, dtype=np.int32) if new_row_idx else np.zeros(0, dtype=np.int32)

    return persistence_diagram(filt_v_q, filt_e_q, filt_f_q,
                               bp_q, bi_q, B2q_cp, B2q_ri)


def persistence_landscape(np.ndarray[f64, ndim=2] barcodes,
                          np.ndarray[f64, ndim=1] grid,
                          Py_ssize_t k_max=5):
    """
    Compute persistence landscape functions Lambda_k(t) on a grid.

    The k-th landscape function at t is the k-th largest value of
    min(t - birth_i, death_i - t) over all bars (birth_i, death_i).


    Parameters
    ----------
    barcodes : f64[n, 2]
        (birth, death) finite pairs.
    grid : f64[G]
        Evaluation points.
    k_max : int
        Number of landscape functions.

    Returns
    -------
    landscapes : f64[k_max, G]
    """
    cdef Py_ssize_t n = barcodes.shape[0], G = grid.shape[0]
    cdef Py_ssize_t i, j, k
    cdef f64 t, b, d, val

    cdef np.ndarray[f64, ndim=2] result = np.zeros((k_max, G), dtype=np.float64)

    if n == 0:
        return result

    cdef np.ndarray[f64, ndim=1] vals = np.empty(n, dtype=np.float64)
    cdef f64[::1] vv = vals

    for j in range(G):
        t = grid[j]
        for i in range(n):
            b = barcodes[i, 0]
            d = barcodes[i, 1]
            if np.isinf(d):
                d = 1e300
            if t <= b or t >= d:
                vv[i] = 0.0
            else:
                val = t - b
                if d - t < val:
                    val = d - t
                vv[i] = val

        if n <= k_max:
            sorted_vals = np.sort(vals)[::-1]
            for k in range(min(n, k_max)):
                result[k, j] = sorted_vals[k]
        else:
            top_k = np.partition(vals, -k_max)[-k_max:]
            top_k.sort()
            for k in range(k_max):
                result[k, j] = top_k[k_max - 1 - k]

    return result
