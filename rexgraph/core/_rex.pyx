# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._rex - Structural operations for the relational complex.

Array-level kernels for constructing, modifying, querying, and
projecting rex structures at all dimension levels.

Every function has i32 and i64 typed variants plus a dispatcher
that auto-selects by dtype.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,

    safe_free,
)

from libc.stdlib cimport malloc, free
from libc.math cimport fabs

np.import_array()


# Edge classification
cdef enum:
    _EDGE_STANDARD  = 0   # |supp(d1(e))| = 2
    _EDGE_SELF_LOOP = 1   # |supp(d1(e))| = 1, multiplicity 2
    _EDGE_BRANCHING = 2   # |supp(d1(e))| >= 3
    _EDGE_WITNESS   = 3   # |supp(d1(e))| = 1, multiplicity 1

# Python-accessible constants
EDGE_STANDARD  = _EDGE_STANDARD
EDGE_SELF_LOOP = _EDGE_SELF_LOOP
EDGE_BRANCHING = _EDGE_BRANCHING
EDGE_WITNESS   = _EDGE_WITNESS


def classify_edges_standard_i32(Py_ssize_t nE,
                                np.ndarray[i32, ndim=1] sources,
                                np.ndarray[i32, ndim=1] targets):
    """Classify 2-boundary edges: STANDARD(0) or SELF_LOOP(1)."""
    cdef np.ndarray[i32, ndim=1] out = np.empty(nE, dtype=np.int32)
    cdef i32[::1] ov = out, sv = sources, tv = targets
    cdef Py_ssize_t j
    for j in range(nE):
        ov[j] = _EDGE_SELF_LOOP if sv[j] == tv[j] else _EDGE_STANDARD
    return out


def classify_edges_standard_i64(Py_ssize_t nE,
                                np.ndarray[i64, ndim=1] sources,
                                np.ndarray[i64, ndim=1] targets):
    """Classify 2-boundary edges. int64 variant."""
    cdef np.ndarray[i32, ndim=1] out = np.empty(nE, dtype=np.int32)
    cdef i32[::1] ov = out
    cdef i64[::1] sv = sources, tv = targets
    cdef Py_ssize_t j
    for j in range(nE):
        ov[j] = _EDGE_SELF_LOOP if sv[j] == tv[j] else _EDGE_STANDARD
    return out


def classify_edges_standard(Py_ssize_t nE, sources, targets):
    """Auto-dispatch by dtype."""
    if sources.dtype == np.int64:
        return classify_edges_standard_i64(nE, sources, targets)
    return classify_edges_standard_i32(nE, sources, targets)


def classify_edges_general_i32(Py_ssize_t nE,
                               np.ndarray[i32, ndim=1] boundary_ptr,
                               np.ndarray[i32, ndim=1] boundary_idx):
    """
    Classify edges with arbitrary boundary sizes .
    Returns (edge_types[nE], boundary_sizes[nE]).
    """
    cdef np.ndarray[i32, ndim=1] out = np.empty(nE, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] bsz = np.empty(nE, dtype=np.int32)
    cdef i32[::1] ov = out, bs = bsz, bp = boundary_ptr, bi = boundary_idx
    cdef Py_ssize_t j, k, start, end, sz, n_unique, ii, jj
    cdef i32 key
    cdef Py_ssize_t max_bnd = 0
    for j in range(nE):
        sz = bp[j + 1] - bp[j]
        if sz > max_bnd: max_bnd = sz
    if max_bnd == 0: max_bnd = 1
    cdef i32* tmp = <i32*>malloc(max_bnd * sizeof(i32))
    if tmp == NULL: raise MemoryError()
    try:
        for j in range(nE):
            start = bp[j]
            end = bp[j + 1]
            sz = end - start
            if sz == 0:
                ov[j] = _EDGE_WITNESS
                bs[j] = 0
                continue
            for k in range(sz): tmp[k] = bi[start + k]
            for ii in range(1, sz):
                key = tmp[ii]
                jj = ii - 1
                while jj >= 0 and tmp[jj] > key:
                    tmp[jj + 1] = tmp[jj]
                    jj -= 1
                tmp[jj + 1] = key
            n_unique = 1
            for k in range(1, sz):
                if tmp[k] != tmp[k - 1]: n_unique += 1
            bs[j] = <i32>n_unique
            if n_unique == 1:
                ov[j] = _EDGE_WITNESS if sz == 1 else _EDGE_SELF_LOOP
            elif n_unique == 2:
                ov[j] = _EDGE_STANDARD
            else:
                ov[j] = _EDGE_BRANCHING
    finally:
        free(tmp)
    return out, bsz


def classify_edges_general_i64(Py_ssize_t nE,
                               np.ndarray[i64, ndim=1] boundary_ptr,
                               np.ndarray[i64, ndim=1] boundary_idx):
    """Classify edges with arbitrary boundary sizes. int64 variant."""
    cdef np.ndarray[i32, ndim=1] out = np.empty(nE, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] bsz = np.empty(nE, dtype=np.int32)
    cdef i32[::1] ov = out, bs = bsz
    cdef i64[::1] bp = boundary_ptr, bi = boundary_idx
    cdef Py_ssize_t j, k, start, end, sz, n_unique, ii, jj
    cdef i64 key64
    cdef Py_ssize_t max_bnd = 0
    for j in range(nE):
        sz = <Py_ssize_t>(bp[j + 1] - bp[j])
        if sz > max_bnd: max_bnd = sz
    if max_bnd == 0: max_bnd = 1
    cdef i64* tmp = <i64*>malloc(max_bnd * sizeof(i64))
    if tmp == NULL: raise MemoryError()
    try:
        for j in range(nE):
            start = <Py_ssize_t>bp[j]
            end = <Py_ssize_t>bp[j + 1]
            sz = end - start
            if sz == 0:
                ov[j] = _EDGE_WITNESS
                bs[j] = 0
                continue
            for k in range(sz): tmp[k] = bi[start + k]
            for ii in range(1, sz):
                key64 = tmp[ii]
                jj = ii - 1
                while jj >= 0 and tmp[jj] > key64:
                    tmp[jj + 1] = tmp[jj]
                    jj -= 1
                tmp[jj + 1] = key64
            n_unique = 1
            for k in range(1, sz):
                if tmp[k] != tmp[k - 1]: n_unique += 1
            bs[j] = <i32>n_unique
            if n_unique == 1:
                ov[j] = _EDGE_WITNESS if sz == 1 else _EDGE_SELF_LOOP
            elif n_unique == 2:
                ov[j] = _EDGE_STANDARD
            else:
                ov[j] = _EDGE_BRANCHING
    finally:
        free(tmp)
    return out, bsz


def classify_edges_general(Py_ssize_t nE, boundary_ptr, boundary_idx):
    """Auto-dispatch by dtype."""
    if boundary_ptr.dtype == np.int64:
        return classify_edges_general_i64(nE, boundary_ptr, boundary_idx)
    return classify_edges_general_i32(nE, boundary_ptr, boundary_idx)


def classify_edges(nE, sources=None, targets=None,
                   boundary_ptr=None, boundary_idx=None):
    """Classify edges by boundary type. Auto-dispatches."""
    if boundary_ptr is not None:
        return classify_edges_general(nE, boundary_ptr, boundary_idx)
    return classify_edges_standard(nE, sources, targets)


# Vertex derivation

def derive_vertex_set_i32(Py_ssize_t nE,
                          np.ndarray[i32, ndim=1] sources,
                          np.ndarray[i32, ndim=1] targets):
    """Derive vertex set from edges. Returns (nV, degree, in_deg, out_deg)."""
    cdef i32[::1] sv = sources, tv = targets
    cdef Py_ssize_t j
    cdef i32 mx = 0
    for j in range(nE):
        if sv[j] > mx: mx = sv[j]
        if tv[j] > mx: mx = tv[j]
    cdef Py_ssize_t nV = <Py_ssize_t>(mx + 1)
    cdef np.ndarray[i32, ndim=1] degree = np.zeros(nV, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] in_deg = np.zeros(nV, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] out_deg = np.zeros(nV, dtype=np.int32)
    cdef i32[::1] dv = degree, iv = in_deg, ov = out_deg
    for j in range(nE):
        ov[sv[j]] += 1
        iv[tv[j]] += 1
        dv[sv[j]] += 1
        dv[tv[j]] += 1
    return nV, degree, in_deg, out_deg


def derive_vertex_set_i64(Py_ssize_t nE,
                          np.ndarray[i64, ndim=1] sources,
                          np.ndarray[i64, ndim=1] targets):
    """Derive vertex set. int64 variant."""
    cdef i64[::1] sv = sources, tv = targets
    cdef Py_ssize_t j
    cdef i64 mx = 0
    for j in range(nE):
        if sv[j] > mx: mx = sv[j]
        if tv[j] > mx: mx = tv[j]
    cdef Py_ssize_t nV = <Py_ssize_t>(mx + 1)
    cdef np.ndarray[i32, ndim=1] degree = np.zeros(nV, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] in_deg = np.zeros(nV, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] out_deg = np.zeros(nV, dtype=np.int32)
    cdef i32[::1] dv = degree, iv = in_deg, ov = out_deg
    for j in range(nE):
        ov[<Py_ssize_t>sv[j]] += 1
        iv[<Py_ssize_t>tv[j]] += 1
        dv[<Py_ssize_t>sv[j]] += 1
        dv[<Py_ssize_t>tv[j]] += 1
    return nV, degree, in_deg, out_deg


def derive_vertex_set(nE, sources, targets):
    """Auto-dispatch by dtype."""
    if sources.dtype == np.int64:
        return derive_vertex_set_i64(nE, sources, targets)
    return derive_vertex_set_i32(nE, sources, targets)


# CSR incidence structures

def build_vertex_to_edge_csr_i32(Py_ssize_t nV, Py_ssize_t nE,
                                  np.ndarray[i32, ndim=1] sources,
                                  np.ndarray[i32, ndim=1] targets):
    """CSR: vertex to incident edge indices."""
    cdef i32[::1] sv = sources, tv = targets
    cdef Py_ssize_t j, v, pos
    cdef np.ndarray[i32, ndim=1] deg = np.zeros(nV, dtype=np.int32)
    cdef i32[::1] d = deg
    for j in range(nE):
        d[sv[j]] += 1
        d[tv[j]] += 1
    cdef np.ndarray[i32, ndim=1] vptr = np.empty(nV + 1, dtype=np.int32)
    cdef i32[::1] vp = vptr
    vp[0] = 0
    for v in range(nV): vp[v + 1] = vp[v] + d[v]
    cdef Py_ssize_t nnz = vp[nV]
    cdef np.ndarray[i32, ndim=1] vidx = np.empty(nnz, dtype=np.int32)
    cdef i32[::1] vi = vidx
    cdef np.ndarray[i32, ndim=1] cur = vptr[:nV].copy()
    cdef i32[::1] c = cur
    for j in range(nE):
        pos = c[sv[j]]
        vi[pos] = <i32>j
        c[sv[j]] = <i32>(pos + 1)
        pos = c[tv[j]]
        vi[pos] = <i32>j
        c[tv[j]] = <i32>(pos + 1)
    return vptr, vidx


def build_vertex_to_edge_csr_i64(Py_ssize_t nV, Py_ssize_t nE,
                                  np.ndarray[i64, ndim=1] sources,
                                  np.ndarray[i64, ndim=1] targets):
    """CSR: vertex to incident edge indices. int64 variant."""
    cdef i64[::1] sv = sources, tv = targets
    cdef Py_ssize_t j, v, pos
    cdef np.ndarray[i64, ndim=1] deg = np.zeros(nV, dtype=np.int64)
    cdef i64[::1] d = deg
    for j in range(nE):
        d[<Py_ssize_t>sv[j]] += 1
        d[<Py_ssize_t>tv[j]] += 1
    cdef np.ndarray[i64, ndim=1] vptr = np.empty(nV + 1, dtype=np.int64)
    cdef i64[::1] vp = vptr
    vp[0] = 0
    for v in range(nV): vp[v + 1] = vp[v] + d[v]
    cdef Py_ssize_t nnz = <Py_ssize_t>vp[nV]
    cdef np.ndarray[i64, ndim=1] vidx = np.empty(nnz, dtype=np.int64)
    cdef i64[::1] vi = vidx
    cdef np.ndarray[i64, ndim=1] cur = vptr[:nV].copy()
    cdef i64[::1] c = cur
    for j in range(nE):
        pos = <Py_ssize_t>c[<Py_ssize_t>sv[j]]
        vi[pos] = <i64>j
        c[<Py_ssize_t>sv[j]] = <i64>(pos + 1)
        pos = <Py_ssize_t>c[<Py_ssize_t>tv[j]]
        vi[pos] = <i64>j
        c[<Py_ssize_t>tv[j]] = <i64>(pos + 1)
    return vptr, vidx


def build_vertex_to_edge_csr(nV, nE, sources, targets):
    """Auto-dispatch by dtype."""
    if sources.dtype == np.int64:
        return build_vertex_to_edge_csr_i64(nV, nE, sources, targets)
    return build_vertex_to_edge_csr_i32(nV, nE, sources, targets)


def build_edge_to_face_csr_i32(Py_ssize_t nE, Py_ssize_t nF,
                                np.ndarray[i32, ndim=1] B2_col_ptr,
                                np.ndarray[i32, ndim=1] B2_row_idx):
    """CSR: edge to incident face indices (B2 CSC transpose)."""
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef Py_ssize_t f, k, e, pos, nnz = cp[nF]
    cdef np.ndarray[i32, ndim=1] deg = np.zeros(nE, dtype=np.int32)
    cdef i32[::1] d = deg
    for f in range(nF):
        for k in range(cp[f], cp[f + 1]):
            d[ri[k]] += 1
    cdef np.ndarray[i32, ndim=1] eptr = np.empty(nE + 1, dtype=np.int32)
    cdef i32[::1] ep = eptr
    ep[0] = 0
    for e in range(nE): ep[e + 1] = ep[e] + d[e]
    cdef np.ndarray[i32, ndim=1] fidx = np.empty(nnz, dtype=np.int32)
    cdef i32[::1] fi = fidx
    cdef np.ndarray[i32, ndim=1] cur = eptr[:nE].copy()
    cdef i32[::1] c = cur
    for f in range(nF):
        for k in range(cp[f], cp[f + 1]):
            e = ri[k]
            pos = c[e]
            fi[pos] = <i32>f
            c[e] = <i32>(pos + 1)
    return eptr, fidx


def build_edge_to_face_csr_i64(Py_ssize_t nE, Py_ssize_t nF,
                                np.ndarray[i64, ndim=1] B2_col_ptr,
                                np.ndarray[i64, ndim=1] B2_row_idx):
    """CSR: edge to incident face indices. int64 variant."""
    cdef i64[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef Py_ssize_t f, k, e, pos, nnz = <Py_ssize_t>cp[nF]
    cdef np.ndarray[i64, ndim=1] deg = np.zeros(nE, dtype=np.int64)
    cdef i64[::1] d = deg
    for f in range(nF):
        for k in range(<Py_ssize_t>cp[f], <Py_ssize_t>cp[f + 1]):
            d[<Py_ssize_t>ri[k]] += 1
    cdef np.ndarray[i64, ndim=1] eptr = np.empty(nE + 1, dtype=np.int64)
    cdef i64[::1] ep = eptr
    ep[0] = 0
    for e in range(nE): ep[e + 1] = ep[e] + d[e]
    cdef np.ndarray[i64, ndim=1] fidx = np.empty(nnz, dtype=np.int64)
    cdef i64[::1] fi = fidx
    cdef np.ndarray[i64, ndim=1] cur = eptr[:nE].copy()
    cdef i64[::1] c = cur
    for f in range(nF):
        for k in range(<Py_ssize_t>cp[f], <Py_ssize_t>cp[f + 1]):
            e = <Py_ssize_t>ri[k]
            pos = <Py_ssize_t>c[e]
            fi[pos] = <i64>f
            c[e] = <i64>(pos + 1)
    return eptr, fidx


def build_edge_to_face_csr(nE, nF, B2_col_ptr, B2_row_idx):
    """Auto-dispatch by dtype."""
    if B2_col_ptr.dtype == np.int64:
        return build_edge_to_face_csr_i64(nE, nF, B2_col_ptr, B2_row_idx)
    return build_edge_to_face_csr_i32(nE, nF, B2_col_ptr, B2_row_idx)


# Branching edge expansion

def clique_expand_branching_i32(Py_ssize_t nE,
                                np.ndarray[i32, ndim=1] boundary_ptr,
                                np.ndarray[i32, ndim=1] boundary_idx,
                                np.ndarray[i32, ndim=1] edge_types):
    """
    Clique-expand branching edges: k boundary points produce C(k,2) standard
    edges at weight 1/(k-1). Returns (src, tgt, weights, parent_edge).
    """
    cdef i32[::1] bp = boundary_ptr, bi = boundary_idx, et = edge_types
    cdef Py_ssize_t j, k, a, b, start, end, sz, n_out, pos, n_uniq
    cdef f64 w
    n_out = 0
    for j in range(nE):
        if et[j] == _EDGE_WITNESS:
            continue
        elif et[j] == _EDGE_BRANCHING:
            start = bp[j]
            end = bp[j + 1]
            k = 1
            for a in range(start + 1, end):
                if bi[a] != bi[a - 1]: k += 1
            n_out += k * (k - 1) // 2
        else:
            n_out += 1
    cdef np.ndarray[i32, ndim=1] ns = np.empty(n_out, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] nt = np.empty(n_out, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] nw = np.empty(n_out, dtype=np.float64)
    cdef np.ndarray[i32, ndim=1] pe = np.empty(n_out, dtype=np.int32)
    cdef i32[::1] nsv = ns, ntv = nt, pev = pe
    cdef f64[::1] nwv = nw
    cdef Py_ssize_t max_bnd = 1
    for j in range(nE):
        sz = bp[j + 1] - bp[j]
        if sz > max_bnd: max_bnd = sz
    cdef i32* uniq = <i32*>malloc(max_bnd * sizeof(i32))
    if uniq == NULL: raise MemoryError()
    pos = 0
    try:
        for j in range(nE):
            if et[j] == _EDGE_WITNESS: continue
            start = bp[j]
            end = bp[j + 1]
            sz = end - start
            if et[j] == _EDGE_BRANCHING:
                uniq[0] = bi[start]
                n_uniq = 1
                for k in range(start + 1, end):
                    if bi[k] != bi[k - 1]:
                        uniq[n_uniq] = bi[k]
                        n_uniq += 1
                w = 1.0 / <f64>(n_uniq - 1)
                for a in range(n_uniq):
                    for b in range(a + 1, n_uniq):
                        nsv[pos] = uniq[a]
                        ntv[pos] = uniq[b]
                        nwv[pos] = w
                        pev[pos] = <i32>j
                        pos += 1
            else:
                nsv[pos] = bi[start]
                ntv[pos] = bi[start + 1] if sz > 1 else bi[start]
                nwv[pos] = 1.0
                pev[pos] = <i32>j
                pos += 1
    finally:
        free(uniq)
    return ns, nt, nw, pe


def clique_expand_branching_i64(Py_ssize_t nE,
                                np.ndarray[i64, ndim=1] boundary_ptr,
                                np.ndarray[i64, ndim=1] boundary_idx,
                                np.ndarray[i32, ndim=1] edge_types):
    """Clique-expand branching edges. int64 variant."""
    cdef i64[::1] bp = boundary_ptr, bi = boundary_idx
    cdef i32[::1] et = edge_types
    cdef Py_ssize_t j, k, a, b, start, end, sz, n_out, pos, n_uniq
    cdef f64 w
    n_out = 0
    for j in range(nE):
        if et[j] == _EDGE_WITNESS:
            continue
        elif et[j] == _EDGE_BRANCHING:
            start = <Py_ssize_t>bp[j]
            end = <Py_ssize_t>bp[j + 1]
            k = 1
            for a in range(start + 1, end):
                if bi[a] != bi[a - 1]: k += 1
            n_out += k * (k - 1) // 2
        else:
            n_out += 1
    cdef np.ndarray[i64, ndim=1] ns = np.empty(n_out, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] nt = np.empty(n_out, dtype=np.int64)
    cdef np.ndarray[f64, ndim=1] nw = np.empty(n_out, dtype=np.float64)
    cdef np.ndarray[i64, ndim=1] pe = np.empty(n_out, dtype=np.int64)
    cdef i64[::1] nsv = ns, ntv = nt, pev = pe
    cdef f64[::1] nwv = nw
    cdef Py_ssize_t max_bnd = 1
    for j in range(nE):
        sz = <Py_ssize_t>(bp[j + 1] - bp[j])
        if sz > max_bnd: max_bnd = sz
    cdef i64* uniq = <i64*>malloc(max_bnd * sizeof(i64))
    if uniq == NULL: raise MemoryError()
    pos = 0
    try:
        for j in range(nE):
            if et[j] == _EDGE_WITNESS: continue
            start = <Py_ssize_t>bp[j]
            end = <Py_ssize_t>bp[j + 1]
            sz = end - start
            if et[j] == _EDGE_BRANCHING:
                uniq[0] = bi[start]
                n_uniq = 1
                for k in range(start + 1, end):
                    if bi[k] != bi[k - 1]:
                        uniq[n_uniq] = bi[k]
                        n_uniq += 1
                w = 1.0 / <f64>(n_uniq - 1)
                for a in range(n_uniq):
                    for b in range(a + 1, n_uniq):
                        nsv[pos] = uniq[a]
                        ntv[pos] = uniq[b]
                        nwv[pos] = w
                        pev[pos] = <i64>j
                        pos += 1
            else:
                nsv[pos] = bi[start]
                ntv[pos] = bi[start + 1] if sz > 1 else bi[start]
                nwv[pos] = 1.0
                pev[pos] = <i64>j
                pos += 1
    finally:
        free(uniq)
    return ns, nt, nw, pe


def clique_expand_branching(nE, boundary_ptr, boundary_idx, edge_types):
    """Auto-dispatch by dtype."""
    if boundary_ptr.dtype == np.int64:
        return clique_expand_branching_i64(nE, boundary_ptr, boundary_idx, edge_types)
    return clique_expand_branching_i32(nE, boundary_ptr, boundary_idx, edge_types)


# Hyperslice queries

def hyperslice_vertex_i32(i32 v,
                          np.ndarray[i32, ndim=1] v2e_ptr,
                          np.ndarray[i32, ndim=1] v2e_idx,
                          np.ndarray[i32, ndim=1] sources,
                          np.ndarray[i32, ndim=1] targets):
    """Hyperslice(v): above=incident edges, lateral=neighbors."""
    cdef i32[::1] vp = v2e_ptr, vi = v2e_idx, sv = sources, tv = targets
    cdef Py_ssize_t start = vp[v], end = vp[v + 1], deg = end - start, k
    cdef Py_ssize_t n_nbrs = 0
    cdef np.ndarray[i32, ndim=1] above = np.empty(deg, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] nbrs = np.empty(deg, dtype=np.int32)
    cdef i32[::1] ab = above, nb = nbrs
    cdef i32 e, other
    for k in range(deg):
        e = vi[start + k]
        ab[k] = e
        other = tv[e] if sv[e] == v else sv[e]
        nb[n_nbrs] = other
        n_nbrs += 1
    return above, np.unique(nbrs[:n_nbrs])


def hyperslice_vertex_i64(i64 v,
                          np.ndarray[i64, ndim=1] v2e_ptr,
                          np.ndarray[i64, ndim=1] v2e_idx,
                          np.ndarray[i64, ndim=1] sources,
                          np.ndarray[i64, ndim=1] targets):
    """Hyperslice(v). int64 variant."""
    cdef i64[::1] vp = v2e_ptr, vi = v2e_idx, sv = sources, tv = targets
    cdef Py_ssize_t start = <Py_ssize_t>vp[v], end = <Py_ssize_t>vp[v + 1]
    cdef Py_ssize_t deg = end - start, k, n_nbrs = 0
    cdef np.ndarray[i64, ndim=1] above = np.empty(deg, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] nbrs = np.empty(deg, dtype=np.int64)
    cdef i64[::1] ab = above, nb = nbrs
    cdef i64 e, other
    for k in range(deg):
        e = vi[start + k]
        ab[k] = e
        other = tv[<Py_ssize_t>e] if sv[<Py_ssize_t>e] == v else sv[<Py_ssize_t>e]
        nb[n_nbrs] = other
        n_nbrs += 1
    return above, np.unique(nbrs[:n_nbrs])


def hyperslice_edge_i32(i32 e,
                        np.ndarray[i32, ndim=1] sources,
                        np.ndarray[i32, ndim=1] targets,
                        np.ndarray[i32, ndim=1] e2f_ptr,
                        np.ndarray[i32, ndim=1] e2f_idx,
                        np.ndarray[i32, ndim=1] v2e_ptr,
                        np.ndarray[i32, ndim=1] v2e_idx):
    """Hyperslice(e): below=d1(e), above=d2inv(e), lateral=overlap."""
    cdef i32[::1] sv = sources, tv = targets
    cdef i32[::1] ep = e2f_ptr, ei = e2f_idx, vp = v2e_ptr, vi = v2e_idx
    cdef Py_ssize_t k
    cdef np.ndarray[i32, ndim=1] below = np.array([sv[e], tv[e]], dtype=np.int32)
    cdef Py_ssize_t f_start = ep[e], f_end = ep[e + 1], n_faces = f_end - f_start
    cdef np.ndarray[i32, ndim=1] above = np.empty(n_faces, dtype=np.int32)
    cdef i32[::1] ab = above
    for k in range(n_faces): ab[k] = ei[f_start + k]
    cdef i32 s = sv[e], t = tv[e]
    cdef Py_ssize_t s_s = vp[s], s_e = vp[s + 1], t_s = vp[t], t_e = vp[t + 1]
    cdef Py_ssize_t max_lat = (s_e - s_s) + (t_e - t_s)
    cdef np.ndarray[i32, ndim=1] lb = np.empty(max_lat, dtype=np.int32)
    cdef i32[::1] lbv = lb
    cdef Py_ssize_t n_lat = 0
    cdef i32 ej
    for k in range(s_s, s_e):
        ej = vi[k]
        if ej != e:
            lbv[n_lat] = ej
            n_lat += 1
    for k in range(t_s, t_e):
        ej = vi[k]
        if ej != e:
            lbv[n_lat] = ej
            n_lat += 1
    return below, above, np.unique(lb[:n_lat])


def hyperslice_edge_i64(i64 e,
                        np.ndarray[i64, ndim=1] sources,
                        np.ndarray[i64, ndim=1] targets,
                        np.ndarray[i64, ndim=1] e2f_ptr,
                        np.ndarray[i64, ndim=1] e2f_idx,
                        np.ndarray[i64, ndim=1] v2e_ptr,
                        np.ndarray[i64, ndim=1] v2e_idx):
    """Hyperslice(e). int64 variant."""
    cdef i64[::1] sv = sources, tv = targets
    cdef i64[::1] ep = e2f_ptr, ei = e2f_idx, vp = v2e_ptr, vi = v2e_idx
    cdef Py_ssize_t k, ev = <Py_ssize_t>e
    cdef np.ndarray[i64, ndim=1] below = np.array([sv[ev], tv[ev]], dtype=np.int64)
    cdef Py_ssize_t f_start = <Py_ssize_t>ep[ev], f_end = <Py_ssize_t>ep[ev + 1]
    cdef Py_ssize_t n_faces = f_end - f_start
    cdef np.ndarray[i64, ndim=1] above = np.empty(n_faces, dtype=np.int64)
    cdef i64[::1] ab = above
    for k in range(n_faces): ab[k] = ei[f_start + k]
    cdef i64 s = sv[ev], t = tv[ev]
    cdef Py_ssize_t s_s = <Py_ssize_t>vp[<Py_ssize_t>s]
    cdef Py_ssize_t s_e = <Py_ssize_t>vp[<Py_ssize_t>s + 1]
    cdef Py_ssize_t t_s = <Py_ssize_t>vp[<Py_ssize_t>t]
    cdef Py_ssize_t t_e = <Py_ssize_t>vp[<Py_ssize_t>t + 1]
    cdef Py_ssize_t max_lat = (s_e - s_s) + (t_e - t_s)
    cdef np.ndarray[i64, ndim=1] lb = np.empty(max_lat, dtype=np.int64)
    cdef i64[::1] lbv = lb
    cdef Py_ssize_t n_lat = 0
    cdef i64 ej
    for k in range(s_s, s_e):
        ej = vi[k]
        if ej != e:
            lbv[n_lat] = ej
            n_lat += 1
    for k in range(t_s, t_e):
        ej = vi[k]
        if ej != e:
            lbv[n_lat] = ej
            n_lat += 1
    return below, above, np.unique(lb[:n_lat])


def hyperslice_face_i32(i32 f, Py_ssize_t nF,
                        np.ndarray[i32, ndim=1] B2_col_ptr,
                        np.ndarray[i32, ndim=1] B2_row_idx,
                        np.ndarray[i32, ndim=1] e2f_ptr,
                        np.ndarray[i32, ndim=1] e2f_idx):
    """Hyperslice(f): below=boundary edges, lateral=adjacent faces."""
    cdef i32[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef i32[::1] ep = e2f_ptr, ei = e2f_idx
    cdef Py_ssize_t start = cp[f], end = cp[f + 1], n_bnd = end - start, k, k2
    cdef np.ndarray[i32, ndim=1] below = np.empty(n_bnd, dtype=np.int32)
    cdef i32[::1] bl = below
    for k in range(n_bnd): bl[k] = ri[start + k]
    cdef Py_ssize_t max_lat = 0
    cdef i32 e
    for k in range(n_bnd):
        e = ri[start + k]
        max_lat += (ep[e + 1] - ep[e])
    cdef np.ndarray[i32, ndim=1] lb = np.empty(max_lat, dtype=np.int32)
    cdef i32[::1] lbv = lb
    cdef Py_ssize_t n_lat = 0
    cdef i32 fj
    for k in range(n_bnd):
        e = ri[start + k]
        for k2 in range(ep[e], ep[e + 1]):
            fj = ei[k2]
            if fj != f:
                lbv[n_lat] = fj
                n_lat += 1
    return below, np.unique(lb[:n_lat])


def hyperslice_face_i64(i64 f, Py_ssize_t nF,
                        np.ndarray[i64, ndim=1] B2_col_ptr,
                        np.ndarray[i64, ndim=1] B2_row_idx,
                        np.ndarray[i64, ndim=1] e2f_ptr,
                        np.ndarray[i64, ndim=1] e2f_idx):
    """Hyperslice(f). int64 variant."""
    cdef i64[::1] cp = B2_col_ptr, ri = B2_row_idx
    cdef i64[::1] ep = e2f_ptr, ei = e2f_idx
    cdef Py_ssize_t fv = <Py_ssize_t>f
    cdef Py_ssize_t start = <Py_ssize_t>cp[fv], end = <Py_ssize_t>cp[fv + 1]
    cdef Py_ssize_t n_bnd = end - start, k, k2
    cdef np.ndarray[i64, ndim=1] below = np.empty(n_bnd, dtype=np.int64)
    cdef i64[::1] bl = below
    for k in range(n_bnd): bl[k] = ri[start + k]
    cdef Py_ssize_t max_lat = 0
    cdef i64 e
    for k in range(n_bnd):
        e = ri[start + k]
        max_lat += <Py_ssize_t>(ep[<Py_ssize_t>e + 1] - ep[<Py_ssize_t>e])
    cdef np.ndarray[i64, ndim=1] lb = np.empty(max_lat, dtype=np.int64)
    cdef i64[::1] lbv = lb
    cdef Py_ssize_t n_lat = 0
    cdef i64 fj
    for k in range(n_bnd):
        e = ri[start + k]
        for k2 in range(<Py_ssize_t>ep[<Py_ssize_t>e], <Py_ssize_t>ep[<Py_ssize_t>e + 1]):
            fj = ei[k2]
            if fj != f:
                lbv[n_lat] = fj
                n_lat += 1
    return below, np.unique(lb[:n_lat])


# Edge insertion and deletion

def insert_edges_i32(Py_ssize_t nV, Py_ssize_t nE,
                     np.ndarray[i32, ndim=1] sources,
                     np.ndarray[i32, ndim=1] targets,
                     np.ndarray[i32, ndim=1] new_sources,
                     np.ndarray[i32, ndim=1] new_targets):
    """Insert edges, extend vertex set. Returns (src, tgt, nV_new)."""
    cdef Py_ssize_t n_new = new_sources.shape[0], nE_out = nE + n_new, j
    cdef np.ndarray[i32, ndim=1] a_s = np.empty(nE_out, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] a_t = np.empty(nE_out, dtype=np.int32)
    cdef i32[::1] asv = a_s, atv = a_t
    cdef i32[::1] os = sources, ot = targets, ns = new_sources, nt = new_targets
    cdef i32 mx = <i32>(nV - 1)
    for j in range(nE):
        asv[j] = os[j]
        atv[j] = ot[j]
    for j in range(n_new):
        asv[nE + j] = ns[j]
        atv[nE + j] = nt[j]
        if ns[j] > mx: mx = ns[j]
        if nt[j] > mx: mx = nt[j]
    return a_s, a_t, <Py_ssize_t>(mx + 1)


def insert_edges_i64(Py_ssize_t nV, Py_ssize_t nE,
                     np.ndarray[i64, ndim=1] sources,
                     np.ndarray[i64, ndim=1] targets,
                     np.ndarray[i64, ndim=1] new_sources,
                     np.ndarray[i64, ndim=1] new_targets):
    """Insert edges. int64 variant."""
    cdef Py_ssize_t n_new = new_sources.shape[0], nE_out = nE + n_new, j
    cdef np.ndarray[i64, ndim=1] a_s = np.empty(nE_out, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] a_t = np.empty(nE_out, dtype=np.int64)
    cdef i64[::1] asv = a_s, atv = a_t
    cdef i64[::1] os = sources, ot = targets, ns = new_sources, nt = new_targets
    cdef i64 mx = <i64>(nV - 1)
    for j in range(nE):
        asv[j] = os[j]
        atv[j] = ot[j]
    for j in range(n_new):
        asv[nE + j] = ns[j]
        atv[nE + j] = nt[j]
        if ns[j] > mx: mx = ns[j]
        if nt[j] > mx: mx = nt[j]
    return a_s, a_t, <Py_ssize_t>(mx + 1)


def insert_edges(nV, nE, sources, targets, new_sources, new_targets):
    """Auto-dispatch by dtype."""
    if sources.dtype == np.int64:
        return insert_edges_i64(nV, nE, sources, targets, new_sources, new_targets)
    return insert_edges_i32(nV, nE, sources, targets, new_sources, new_targets)


def delete_edges_i32(Py_ssize_t nV, Py_ssize_t nE,
                     np.ndarray[i32, ndim=1] sources,
                     np.ndarray[i32, ndim=1] targets,
                     np.ndarray[i32, ndim=1] delete_mask):
    """
    Delete edges (mask=1), enforce vertex lifecycle.
    Returns (new_src, new_tgt, nV_new, v_map[nV], e_map[nE]).
    """
    cdef i32[::1] sv = sources, tv = targets, dm = delete_mask
    cdef Py_ssize_t j, v, pos
    cdef Py_ssize_t nE_keep = 0
    for j in range(nE):
        if dm[j] == 0: nE_keep += 1
    cdef np.ndarray[i32, ndim=1] em = np.full(nE, -1, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] ks = np.empty(nE_keep, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] kt = np.empty(nE_keep, dtype=np.int32)
    cdef i32[::1] emv = em, ksv = ks, ktv = kt
    pos = 0
    for j in range(nE):
        if dm[j] == 0:
            emv[j] = <i32>pos
            ksv[pos] = sv[j]
            ktv[pos] = tv[j]
            pos += 1
    cdef np.ndarray[i32, ndim=1] alive = np.zeros(nV, dtype=np.int32)
    cdef i32[::1] al = alive
    for j in range(nE_keep):
        al[ksv[j]] = 1
        al[ktv[j]] = 1
    cdef np.ndarray[i32, ndim=1] vm = np.full(nV, -1, dtype=np.int32)
    cdef i32[::1] vmv = vm
    cdef Py_ssize_t nV_new = 0
    for v in range(nV):
        if al[v]:
            vmv[v] = <i32>nV_new
            nV_new += 1
    cdef np.ndarray[i32, ndim=1] ns = np.empty(nE_keep, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] nt = np.empty(nE_keep, dtype=np.int32)
    cdef i32[::1] nsv = ns, ntv = nt
    for j in range(nE_keep):
        nsv[j] = vmv[ksv[j]]
        ntv[j] = vmv[ktv[j]]
    return ns, nt, nV_new, vm, em


def delete_edges_i64(Py_ssize_t nV, Py_ssize_t nE,
                     np.ndarray[i64, ndim=1] sources,
                     np.ndarray[i64, ndim=1] targets,
                     np.ndarray[i32, ndim=1] delete_mask):
    """Delete edges. int64 variant."""
    cdef i64[::1] sv = sources, tv = targets
    cdef i32[::1] dm = delete_mask
    cdef Py_ssize_t j, v, pos
    cdef Py_ssize_t nE_keep = 0
    for j in range(nE):
        if dm[j] == 0: nE_keep += 1
    cdef np.ndarray[i64, ndim=1] em = np.full(nE, -1, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] ks = np.empty(nE_keep, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] kt = np.empty(nE_keep, dtype=np.int64)
    cdef i64[::1] emv = em, ksv = ks, ktv = kt
    pos = 0
    for j in range(nE):
        if dm[j] == 0:
            emv[j] = <i64>pos
            ksv[pos] = sv[j]
            ktv[pos] = tv[j]
            pos += 1
    cdef np.ndarray[i32, ndim=1] alive = np.zeros(nV, dtype=np.int32)
    cdef i32[::1] al = alive
    for j in range(nE_keep):
        al[<Py_ssize_t>ksv[j]] = 1
        al[<Py_ssize_t>ktv[j]] = 1
    cdef np.ndarray[i64, ndim=1] vm = np.full(nV, -1, dtype=np.int64)
    cdef i64[::1] vmv = vm
    cdef Py_ssize_t nV_new = 0
    for v in range(nV):
        if al[v]:
            vmv[v] = <i64>nV_new
            nV_new += 1
    cdef np.ndarray[i64, ndim=1] ns = np.empty(nE_keep, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] nt = np.empty(nE_keep, dtype=np.int64)
    cdef i64[::1] nsv = ns, ntv = nt
    for j in range(nE_keep):
        nsv[j] = vmv[<Py_ssize_t>ksv[j]]
        ntv[j] = vmv[<Py_ssize_t>ktv[j]]
    return ns, nt, nV_new, vm, em


def delete_edges(nV, nE, sources, targets, delete_mask):
    """Auto-dispatch by dtype."""
    if sources.dtype == np.int64:
        return delete_edges_i64(nV, nE, sources, targets, delete_mask)
    return delete_edges_i32(nV, nE, sources, targets, delete_mask)


# Dimensional projection

def project_to_dimension(Py_ssize_t target_dim,
                         Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF,
                         Py_ssize_t rank_B1, Py_ssize_t rank_B2):
    """Betti numbers for projection to target_dim. Returns (betti[3], euler)."""
    cdef Py_ssize_t b0, b1, b2, chi
    if target_dim >= 2:
        b0 = nV - rank_B1
        b1 = nE - rank_B1 - rank_B2
        b2 = nF - rank_B2
        chi = nV - nE + nF
    elif target_dim == 1:
        b0 = nV - rank_B1
        b1 = nE - rank_B1
        b2 = 0
        chi = nV - nE
    else:
        b0 = nV
        b1 = 0
        b2 = 0
        chi = nV
    return np.array([b0, b1, b2], dtype=np.int64), chi


def betti_deltas(Py_ssize_t nV, Py_ssize_t nE, Py_ssize_t nF,
                 Py_ssize_t rank_B1, Py_ssize_t rank_B2):
    """Betti deltas across projection steps."""
    return {
        "2to1": np.array([0, rank_B2, -(nF - rank_B2)], dtype=np.int64),
        "1to0": np.array([rank_B1, -(nE - rank_B1), 0], dtype=np.int64),
    }


# Subsumption embeddings

def from_graph_i32(Py_ssize_t nV, np.ndarray[i32, ndim=1] src,
                   np.ndarray[i32, ndim=1] tgt):
    """Simple graph to 1-rex."""
    cdef Py_ssize_t nE = src.shape[0], j
    cdef np.ndarray[i32, ndim=1] et = np.zeros(nE, dtype=np.int32)
    cdef i32[::1] sv = src, tv = tgt, ev = et
    for j in range(nE):
        if sv[j] == tv[j]: ev[j] = _EDGE_SELF_LOOP
    return src.copy(), tgt.copy(), et, nV


def from_graph_i64(Py_ssize_t nV, np.ndarray[i64, ndim=1] src,
                   np.ndarray[i64, ndim=1] tgt):
    """Simple graph to 1-rex. int64 variant."""
    cdef Py_ssize_t nE = src.shape[0], j
    cdef np.ndarray[i32, ndim=1] et = np.zeros(nE, dtype=np.int32)
    cdef i64[::1] sv = src, tv = tgt
    cdef i32[::1] ev = et
    for j in range(nE):
        if sv[j] == tv[j]: ev[j] = _EDGE_SELF_LOOP
    return src.copy(), tgt.copy(), et, nV


def from_graph(nV, src, tgt):
    """Auto-dispatch by dtype."""
    if src.dtype == np.int64: return from_graph_i64(nV, src, tgt)
    return from_graph_i32(nV, src, tgt)


def from_hypergraph_i32(Py_ssize_t nV,
                        np.ndarray[i32, ndim=1] hedge_ptr,
                        np.ndarray[i32, ndim=1] hedge_idx):
    """Hypergraph to branching 1-rex."""
    cdef Py_ssize_t nH = hedge_ptr.shape[0] - 1, j, sz
    cdef np.ndarray[i32, ndim=1] et = np.empty(nH, dtype=np.int32)
    cdef i32[::1] hp = hedge_ptr, ev = et
    for j in range(nH):
        sz = hp[j + 1] - hp[j]
        if sz <= 1:
            ev[j] = _EDGE_WITNESS
        elif sz == 2:
            ev[j] = _EDGE_STANDARD
        else:
            ev[j] = _EDGE_BRANCHING
    return hedge_ptr.copy(), hedge_idx.copy(), et


def from_hypergraph_i64(Py_ssize_t nV,
                        np.ndarray[i64, ndim=1] hedge_ptr,
                        np.ndarray[i64, ndim=1] hedge_idx):
    """Hypergraph to branching 1-rex. int64 variant."""
    cdef Py_ssize_t nH = hedge_ptr.shape[0] - 1, j, sz
    cdef np.ndarray[i32, ndim=1] et = np.empty(nH, dtype=np.int32)
    cdef i64[::1] hp = hedge_ptr
    cdef i32[::1] ev = et
    for j in range(nH):
        sz = <Py_ssize_t>(hp[j + 1] - hp[j])
        if sz <= 1:
            ev[j] = _EDGE_WITNESS
        elif sz == 2:
            ev[j] = _EDGE_STANDARD
        else:
            ev[j] = _EDGE_BRANCHING
    return hedge_ptr.copy(), hedge_idx.copy(), et


def from_hypergraph(nV, hedge_ptr, hedge_idx):
    """Auto-dispatch by dtype."""
    if hedge_ptr.dtype == np.int64:
        return from_hypergraph_i64(nV, hedge_ptr, hedge_idx)
    return from_hypergraph_i32(nV, hedge_ptr, hedge_idx)


def from_simplicial_2complex_i32(Py_ssize_t nV,
                                  np.ndarray[i32, ndim=1] edge_src,
                                  np.ndarray[i32, ndim=1] edge_tgt,
                                  np.ndarray[i32, ndim=1] tri_e0,
                                  np.ndarray[i32, ndim=1] tri_e1,
                                  np.ndarray[i32, ndim=1] tri_e2,
                                  np.ndarray[f64, ndim=1] tri_s0,
                                  np.ndarray[f64, ndim=1] tri_s1,
                                  np.ndarray[f64, ndim=1] tri_s2):
    """Simplicial 2-complex to 2-rex B2 in CSC."""
    cdef Py_ssize_t nT = tri_e0.shape[0], nnz = 3 * nT, t
    cdef np.ndarray[i32, ndim=1] cp = np.empty(nT + 1, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] ri = np.empty(nnz, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] vv = np.empty(nnz, dtype=np.float64)
    cdef i32[::1] cpv = cp, riv = ri
    cdef f64[::1] vvv = vv
    cdef i32[::1] e0 = tri_e0, e1 = tri_e1, e2 = tri_e2
    cdef f64[::1] s0 = tri_s0, s1 = tri_s1, s2 = tri_s2
    for t in range(nT):
        cpv[t] = <i32>(3 * t)
        riv[3*t] = e0[t]
        riv[3*t+1] = e1[t]
        riv[3*t+2] = e2[t]
        vvv[3*t] = s0[t]
        vvv[3*t+1] = s1[t]
        vvv[3*t+2] = s2[t]
    cpv[nT] = <i32>nnz
    return cp, ri, vv


def from_simplicial_2complex_i64(Py_ssize_t nV,
                                  np.ndarray[i64, ndim=1] edge_src,
                                  np.ndarray[i64, ndim=1] edge_tgt,
                                  np.ndarray[i64, ndim=1] tri_e0,
                                  np.ndarray[i64, ndim=1] tri_e1,
                                  np.ndarray[i64, ndim=1] tri_e2,
                                  np.ndarray[f64, ndim=1] tri_s0,
                                  np.ndarray[f64, ndim=1] tri_s1,
                                  np.ndarray[f64, ndim=1] tri_s2):
    """Simplicial 2-complex to 2-rex. int64 variant."""
    cdef Py_ssize_t nT = tri_e0.shape[0], nnz = 3 * nT, t
    cdef np.ndarray[i64, ndim=1] cp = np.empty(nT + 1, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] ri = np.empty(nnz, dtype=np.int64)
    cdef np.ndarray[f64, ndim=1] vv = np.empty(nnz, dtype=np.float64)
    cdef i64[::1] cpv = cp, riv = ri
    cdef f64[::1] vvv = vv
    cdef i64[::1] e0 = tri_e0, e1 = tri_e1, e2 = tri_e2
    cdef f64[::1] s0 = tri_s0, s1 = tri_s1, s2 = tri_s2
    for t in range(nT):
        cpv[t] = <i64>(3 * t)
        riv[3*t] = e0[t]
        riv[3*t+1] = e1[t]
        riv[3*t+2] = e2[t]
        vvv[3*t] = s0[t]
        vvv[3*t+1] = s1[t]
        vvv[3*t+2] = s2[t]
    cpv[nT] = <i64>nnz
    return cp, ri, vv


def from_simplicial_2complex(nV, edge_src, edge_tgt,
                              tri_e0, tri_e1, tri_e2,
                              tri_s0, tri_s1, tri_s2):
    """Auto-dispatch by dtype."""
    if tri_e0.dtype == np.int64:
        return from_simplicial_2complex_i64(nV, edge_src, edge_tgt,
                                             tri_e0, tri_e1, tri_e2, tri_s0, tri_s1, tri_s2)
    return from_simplicial_2complex_i32(nV, edge_src, edge_tgt,
                                         tri_e0, tri_e1, tri_e2, tri_s0, tri_s1, tri_s2)


# Generalized k-chain

def build_Bk_from_cells_i32(Py_ssize_t n_lower,
                             np.ndarray[i32, ndim=1] cell_ptr,
                             np.ndarray[i32, ndim=1] cell_idx,
                             np.ndarray[f64, ndim=1] cell_signs):
    """Build Bk in CSC from k-cells. Returns (col_ptr, row_idx, vals, n_cells)."""
    cdef Py_ssize_t n_cells = cell_ptr.shape[0] - 1
    return cell_ptr.copy(), cell_idx.copy(), cell_signs.copy(), n_cells


def build_Bk_from_cells_i64(Py_ssize_t n_lower,
                             np.ndarray[i64, ndim=1] cell_ptr,
                             np.ndarray[i64, ndim=1] cell_idx,
                             np.ndarray[f64, ndim=1] cell_signs):
    """Build Bk in CSC. int64 variant."""
    cdef Py_ssize_t n_cells = cell_ptr.shape[0] - 1
    return cell_ptr.copy(), cell_idx.copy(), cell_signs.copy(), n_cells


def build_Bk_from_cells(n_lower, cell_ptr, cell_idx, cell_signs):
    """Auto-dispatch by dtype."""
    if cell_ptr.dtype == np.int64:
        return build_Bk_from_cells_i64(n_lower, cell_ptr, cell_idx, cell_signs)
    return build_Bk_from_cells_i32(n_lower, cell_ptr, cell_idx, cell_signs)


def verify_chain_condition_Bk_i32(Py_ssize_t n_rows_prev,
                                   np.ndarray[i32, ndim=1] Bkm1_rp,
                                   np.ndarray[i32, ndim=1] Bkm1_ci,
                                   np.ndarray[f64, ndim=1] Bkm1_v,
                                   np.ndarray[i32, ndim=1] Bk_cp,
                                   np.ndarray[i32, ndim=1] Bk_ri,
                                   np.ndarray[f64, ndim=1] Bk_v,
                                   double tol=1e-10):
    """Verify Bk-1 * Bk = 0 via sparse matvec. Returns (ok, max_err)."""
    cdef i32[::1] rp = Bkm1_rp, ci = Bkm1_ci, cp = Bk_cp, ri = Bk_ri
    cdef f64[::1] vp = Bkm1_v, vk = Bk_v
    cdef Py_ssize_t nc = cp.shape[0] - 1, nr = rp.shape[0] - 1
    cdef Py_ssize_t nl = 0, j, k, k2, row
    cdef double max_err = 0.0, ev
    for j in range(nc):
        for k in range(cp[j], cp[j + 1]):
            if ri[k] >= nl: nl = ri[k] + 1
    cdef np.ndarray[f64, ndim=1] vec = np.zeros(nl, dtype=np.float64)
    cdef f64[::1] vv = vec
    for j in range(nc):
        for k in range(nl): vv[k] = 0.0
        for k in range(cp[j], cp[j + 1]):
            vv[ri[k]] = vk[k]
        for row in range(nr):
            ev = 0.0
            for k2 in range(rp[row], rp[row + 1]):
                ev += vp[k2] * vv[ci[k2]]
            if fabs(ev) > max_err: max_err = fabs(ev)
    return max_err < tol, max_err


def verify_chain_condition_Bk_i64(Py_ssize_t n_rows_prev,
                                   np.ndarray[i64, ndim=1] Bkm1_rp,
                                   np.ndarray[i64, ndim=1] Bkm1_ci,
                                   np.ndarray[f64, ndim=1] Bkm1_v,
                                   np.ndarray[i64, ndim=1] Bk_cp,
                                   np.ndarray[i64, ndim=1] Bk_ri,
                                   np.ndarray[f64, ndim=1] Bk_v,
                                   double tol=1e-10):
    """Verify Bk-1 * Bk = 0. int64 variant."""
    cdef i64[::1] rp = Bkm1_rp, ci = Bkm1_ci, cp = Bk_cp, ri = Bk_ri
    cdef f64[::1] vp = Bkm1_v, vk = Bk_v
    cdef Py_ssize_t nc = cp.shape[0] - 1, nr = rp.shape[0] - 1
    cdef Py_ssize_t nl = 0, j, k, k2, row
    cdef double max_err = 0.0, ev
    for j in range(nc):
        for k in range(<Py_ssize_t>cp[j], <Py_ssize_t>cp[j + 1]):
            if <Py_ssize_t>ri[k] >= nl: nl = <Py_ssize_t>ri[k] + 1
    cdef np.ndarray[f64, ndim=1] vec = np.zeros(nl, dtype=np.float64)
    cdef f64[::1] vv = vec
    for j in range(nc):
        for k in range(nl): vv[k] = 0.0
        for k in range(<Py_ssize_t>cp[j], <Py_ssize_t>cp[j + 1]):
            vv[<Py_ssize_t>ri[k]] = vk[k]
        for row in range(nr):
            ev = 0.0
            for k2 in range(<Py_ssize_t>rp[row], <Py_ssize_t>rp[row + 1]):
                ev += vp[k2] * vv[<Py_ssize_t>ci[k2]]
            if fabs(ev) > max_err: max_err = fabs(ev)
    return max_err < tol, max_err


def verify_chain_condition_Bk(n_rows_prev, Bkm1_rp, Bkm1_ci, Bkm1_v,
                               Bk_cp, Bk_ri, Bk_v, tol=1e-10):
    """Auto-dispatch by dtype."""
    if Bkm1_rp.dtype == np.int64:
        return verify_chain_condition_Bk_i64(n_rows_prev, Bkm1_rp, Bkm1_ci, Bkm1_v,
                                              Bk_cp, Bk_ri, Bk_v, tol)
    return verify_chain_condition_Bk_i32(n_rows_prev, Bkm1_rp, Bkm1_ci, Bkm1_v,
                                          Bk_cp, Bk_ri, Bk_v, tol)


# Coboundary queries

def coboundary_vertex_i32(i32 v, np.ndarray[i32, ndim=1] v2e_ptr,
                          np.ndarray[i32, ndim=1] v2e_idx):
    """Return edge indices incident to vertex v."""
    cdef i32[::1] vp = v2e_ptr, vi = v2e_idx
    cdef Py_ssize_t s = vp[v], e = vp[v + 1], n = e - s, k
    cdef np.ndarray[i32, ndim=1] r = np.empty(n, dtype=np.int32)
    cdef i32[::1] rv = r
    for k in range(n): rv[k] = vi[s + k]
    return r


def coboundary_vertex_i64(i64 v, np.ndarray[i64, ndim=1] v2e_ptr,
                          np.ndarray[i64, ndim=1] v2e_idx):
    """Edges incident to vertex v. int64 variant."""
    cdef i64[::1] vp = v2e_ptr, vi = v2e_idx
    cdef Py_ssize_t s = <Py_ssize_t>vp[v], e = <Py_ssize_t>vp[v + 1]
    cdef Py_ssize_t n = e - s, k
    cdef np.ndarray[i64, ndim=1] r = np.empty(n, dtype=np.int64)
    cdef i64[::1] rv = r
    for k in range(n): rv[k] = vi[s + k]
    return r


def coboundary_edge_i32(i32 e, np.ndarray[i32, ndim=1] e2f_ptr,
                        np.ndarray[i32, ndim=1] e2f_idx):
    """Return face indices incident to edge e."""
    cdef i32[::1] ep = e2f_ptr, ei = e2f_idx
    cdef Py_ssize_t s = ep[e], en = ep[e + 1], n = en - s, k
    cdef np.ndarray[i32, ndim=1] r = np.empty(n, dtype=np.int32)
    cdef i32[::1] rv = r
    for k in range(n): rv[k] = ei[s + k]
    return r


def coboundary_edge_i64(i64 e, np.ndarray[i64, ndim=1] e2f_ptr,
                        np.ndarray[i64, ndim=1] e2f_idx):
    """Faces incident to edge e. int64 variant."""
    cdef i64[::1] ep = e2f_ptr, ei = e2f_idx
    cdef Py_ssize_t s = <Py_ssize_t>ep[<Py_ssize_t>e]
    cdef Py_ssize_t en = <Py_ssize_t>ep[<Py_ssize_t>e + 1]
    cdef Py_ssize_t n = en - s, k
    cdef np.ndarray[i64, ndim=1] r = np.empty(n, dtype=np.int64)
    cdef i64[::1] rv = r
    for k in range(n): rv[k] = ei[s + k]
    return r


# Convenience dispatchers

def build_1rex(nV, nE, sources, targets):
    """Build 1-rex from edge arrays. Returns dict of all derived structure."""
    nV_d, degree, in_deg, out_deg = derive_vertex_set(nE, sources, targets)
    if nV_d > nV: nV = nV_d
    etypes = classify_edges_standard(nE, sources, targets)
    v2e_ptr, v2e_idx = build_vertex_to_edge_csr(nV, nE, sources, targets)
    return {
        "nV": nV, "nE": nE,
        "sources": sources, "targets": targets, "edge_types": etypes,
        "degree": degree, "in_deg": in_deg, "out_deg": out_deg,
        "v2e_ptr": v2e_ptr, "v2e_idx": v2e_idx,
    }


def hyperslice(cell_dim, cell_idx, **kw):
    """Hyperslice at any dimension. Auto-dispatches by dtype."""
    if cell_dim == 0:
        p = kw["v2e_ptr"]
        if p.dtype == np.int64:
            return hyperslice_vertex_i64(cell_idx, p, kw["v2e_idx"],
                                          kw["sources"], kw["targets"])
        return hyperslice_vertex_i32(cell_idx, p, kw["v2e_idx"],
                                      kw["sources"], kw["targets"])
    elif cell_dim == 1:
        s = kw["sources"]
        if s.dtype == np.int64:
            return hyperslice_edge_i64(cell_idx, s, kw["targets"],
                                        kw["e2f_ptr"], kw["e2f_idx"],
                                        kw["v2e_ptr"], kw["v2e_idx"])
        return hyperslice_edge_i32(cell_idx, s, kw["targets"],
                                    kw["e2f_ptr"], kw["e2f_idx"],
                                    kw["v2e_ptr"], kw["v2e_idx"])
    elif cell_dim == 2:
        c = kw["B2_col_ptr"]
        if c.dtype == np.int64:
            return hyperslice_face_i64(cell_idx, kw["nF"], c, kw["B2_row_idx"],
                                        kw["e2f_ptr"], kw["e2f_idx"])
        return hyperslice_face_i32(cell_idx, kw["nF"], c, kw["B2_row_idx"],
                                    kw["e2f_ptr"], kw["e2f_idx"])
    raise ValueError("Unsupported cell dimension: %d" % cell_dim)
