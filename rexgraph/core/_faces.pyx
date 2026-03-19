# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._faces - Face classification, extraction, metrics, and selection.

Classifies faces into proper (2+ unique boundary vertices) and
self-loop (1 vertex), filters B_2 to produce B2_hodge for exact
Hodge decomposition, extracts face descriptors, and computes
structural metrics.

Face selection methods:
    typed_face_selection - B1 decomposed by edge type; same-type
        triangles become faces, cross-type become voids.
    context_face_selection - algebraic context selection via
        E = C^T |B1| > 0; face included iff some context covers
        all three boundary edges.
    void_type_composition - decompose void triangles by cross-type
        edge composition.

Self-loop faces arise from edges like v->v. Their B_2 column has
nonzero entries but B_1 B_2 != 0 for those columns because the
boundary of a self-loop is v - v = 0 in the vertex chain group
yet nonzero in the edge chain. Filtering them out gives B2_hodge
where B_1 B_2 = 0 holds exactly.

Metrics computed:
    vertex face count - distinct faces incident to each vertex
    vertex/edge contribution - 1 / face_vertex_count, summed
    boundary asymmetry - |fc(src) - fc(tgt)| / max(fc(src), fc(tgt))
    face concentration - CV of vertex face counts within each face
    asym-rho correlation - Pearson(asymmetry, rho)

Vertex deduplication within faces uses a generation counter
(last_seen[v] = generation) to avoid per-face allocation.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,
    get_EPSILON_NORM,
)

from libc.math cimport fabs, sqrt

np.import_array()


# Face classification

@cython.boundscheck(False)
@cython.wraparound(False)
def classify_faces(B2, edge_src, edge_tgt):
    """Classify faces as proper (2+ unique vertices) or self-loop.

    A face is a self-loop when all its boundary edges connect to the
    same single vertex. These faces violate B_1 B_2 = 0 and must be
    excluded from the Hodge tier.

    Parameters
    ----------
    B2 : DualCSR, shape (nE, nF).
    edge_src, edge_tgt : int array [nE]

    Returns
    -------
    dict
        proper_mask : bool[nF]
            True for faces with 2+ unique boundary vertices.
        self_loop_mask : bool[nF]
            True for single-vertex self-loop faces.
        n_proper : int
        n_self_loop : int
        proper_indices : int[]
        self_loop_indices : int[]
    """
    cdef Py_ssize_t nF = B2.ncol

    col_ptr = np.asarray(B2.col_ptr)
    row_idx = np.asarray(B2.row_idx)
    src = np.asarray(edge_src)
    tgt = np.asarray(edge_tgt)

    cdef np.ndarray[np.uint8_t, ndim=1] proper = np.zeros(nF, dtype=np.uint8)
    cdef np.uint8_t[::1] pm = proper

    cdef Py_ssize_t f, k, e, start, end
    cdef Py_ssize_t first_v
    cdef bint multi_vertex

    for f in range(nF):
        start = int(col_ptr[f])
        end = int(col_ptr[f + 1])

        if start == end:
            pm[f] = 0
            continue

        # Get first vertex from first edge
        e = int(row_idx[start])
        first_v = int(src[e])
        multi_vertex = (int(tgt[e]) != first_v)

        if not multi_vertex:
            for k in range(start + 1, end):
                e = int(row_idx[k])
                if int(src[e]) != first_v or int(tgt[e]) != first_v:
                    multi_vertex = True
                    break

        pm[f] = 1 if multi_vertex else 0

    proper_bool = proper.astype(bool)
    self_loop_bool = ~proper_bool
    proper_idx = np.where(proper_bool)[0]
    self_loop_idx = np.where(self_loop_bool)[0]

    return {
        'proper_mask': proper_bool,
        'self_loop_mask': self_loop_bool,
        'n_proper': int(proper_idx.shape[0]),
        'n_self_loop': int(self_loop_idx.shape[0]),
        'proper_indices': proper_idx,
        'self_loop_indices': self_loop_idx,
    }


def filter_b2_hodge(B2_dense, proper_mask):
    """Filter B_2 columns to proper faces only (B2_hodge).

    The returned matrix satisfies B_1 B_2 = 0 when the original B_2
    only violates this for self-loop face columns.

    Parameters
    ----------
    B2_dense : ndarray[nE, nF], float64
    proper_mask : bool[nF]
        From classify_faces.

    Returns
    -------
    ndarray[nE, nF_hodge], float64
    """
    return np.ascontiguousarray(B2_dense[:, proper_mask], dtype=np.float64)


# Vertex face count

@cython.boundscheck(False)
@cython.wraparound(False)
def vertex_face_count_i32(B2,
                          np.ndarray[i32, ndim=1] edge_src,
                          np.ndarray[i32, ndim=1] edge_tgt,
                          Py_ssize_t nV):
    """Count distinct faces incident to each vertex.

    Parameters
    ----------
    B2 : DualCSR, shape (nE, nF).
    edge_src, edge_tgt : i32[nE]
    nV : int

    Returns
    -------
    i32[nV]
    """
    cdef Py_ssize_t nF = B2.ncol
    cdef np.ndarray[i32, ndim=1] fc = np.zeros(nV, dtype=np.int32)
    cdef np.ndarray[i64, ndim=1] last_seen = np.full(nV, -1, dtype=np.int64)

    cdef i32[::1] cp = B2.col_ptr
    cdef i32[::1] ri = B2.row_idx
    cdef i32[::1] es = edge_src, et = edge_tgt
    cdef i32[::1] fcv = fc
    cdef i64[::1] ls = last_seen

    cdef Py_ssize_t f, k
    cdef i32 e, s, t
    cdef Py_ssize_t fi

    for f in range(nF):
        fi = f
        for k in range(cp[f], cp[f + 1]):
            e = ri[k]
            s = es[e]; t = et[e]
            if ls[s] != fi:
                ls[s] = fi
                fcv[s] += 1
            if ls[t] != fi:
                ls[t] = fi
                fcv[t] += 1

    return fc


@cython.boundscheck(False)
@cython.wraparound(False)
def vertex_face_count_i64(B2,
                          np.ndarray[i64, ndim=1] edge_src,
                          np.ndarray[i64, ndim=1] edge_tgt,
                          Py_ssize_t nV):
    """Count faces per vertex. int64 variant."""
    cdef Py_ssize_t nF = B2.ncol
    cdef np.ndarray[i32, ndim=1] fc = np.zeros(nV, dtype=np.int32)
    cdef np.ndarray[i64, ndim=1] last_seen = np.full(nV, -1, dtype=np.int64)

    cdef i64[::1] cp = B2.col_ptr
    cdef i64[::1] ri = B2.row_idx
    cdef i64[::1] es = edge_src, et = edge_tgt
    cdef i32[::1] fcv = fc
    cdef i64[::1] ls = last_seen

    cdef Py_ssize_t f, k
    cdef i64 e
    cdef Py_ssize_t s, t
    cdef Py_ssize_t fi

    for f in range(nF):
        fi = f
        for k in range(cp[f], cp[f + 1]):
            e = ri[k]
            s = <Py_ssize_t>es[e]; t = <Py_ssize_t>et[e]
            if ls[s] != fi:
                ls[s] = fi
                fcv[s] += 1
            if ls[t] != fi:
                ls[t] = fi
                fcv[t] += 1

    return fc


def vertex_face_count(B2, edge_src, edge_tgt, Py_ssize_t nV):
    """Dispatch vertex_face_count by index type."""
    if B2.idx_bits == 64:
        return vertex_face_count_i64(B2, edge_src, edge_tgt, nV)
    return vertex_face_count_i32(B2, edge_src, edge_tgt, nV)


# Face extraction

def extract_faces(B2, edge_src, edge_tgt, vertex_names, edge_names,
                  face_class=None):
    """Per-face descriptors from B_2.

    Each face dict has: id, boundary ({edge_name: sign}),
    vertices (sorted names), size (boundary edge count),
    is_self_loop (True if only 1 unique vertex).

    Parameters
    ----------
    B2 : DualCSR, shape (nE, nF).
    edge_src, edge_tgt : int array [nE]
    vertex_names : list[str]
    edge_names : list[str]
    face_class : dict or None
        Output of classify_faces. If provided, is_self_loop is
        read directly. If None, computed per face.

    Returns
    -------
    list[dict]
    """
    cdef Py_ssize_t nF = B2.ncol
    cdef Py_ssize_t f, k

    col_ptr = B2.col_ptr
    row_idx = B2.row_idx
    vals_csc = B2.vals_csc

    self_loop_mask = None
    if face_class is not None:
        self_loop_mask = face_class['self_loop_mask']

    faces = []
    for f in range(nF):
        bnd = {}
        vset = set()
        start = col_ptr[f]
        end = col_ptr[f + 1]
        for k in range(start, end):
            e = int(row_idx[k])
            sign_val = float(vals_csc[k])
            bnd[edge_names[e]] = sign_val
            vset.add(int(edge_src[e]))
            vset.add(int(edge_tgt[e]))

        is_sl = len(vset) < 2
        if self_loop_mask is not None:
            is_sl = bool(self_loop_mask[f])

        faces.append({
            'id': f'f{f + 1}',
            'boundary': bnd,
            'vertices': sorted(vertex_names[v] for v in vset),
            'size': end - start,
            'is_self_loop': is_sl,
        })

    return faces


# Pearson correlation

cdef inline double _pearson_corr(f64[::1] x, f64[::1] y,
                                 Py_ssize_t n) noexcept nogil:
    """Single-pass Pearson correlation. Returns 0.0 if either signal
    has zero variance or n < 2."""
    if n < 2:
        return 0.0

    cdef double sx = 0.0, sy = 0.0
    cdef double sx2 = 0.0, sy2 = 0.0, sxy = 0.0
    cdef double xi, yi
    cdef Py_ssize_t i
    cdef double nn = <double>n
    cdef double denom_x, denom_y, denom

    for i in range(n):
        xi = x[i]; yi = y[i]
        sx += xi;  sy += yi
        sx2 += xi * xi
        sy2 += yi * yi
        sxy += xi * yi

    denom_x = nn * sx2 - sx * sx
    denom_y = nn * sy2 - sy * sy

    if denom_x < 1e-30 or denom_y < 1e-30:
        return 0.0

    denom = sqrt(denom_x * denom_y)
    return (nn * sxy - sx * sy) / denom


# Face metrics, i32 variant

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_face_metrics_i32(B2,
                             np.ndarray[i32, ndim=1] edge_src,
                             np.ndarray[i32, ndim=1] edge_tgt,
                             Py_ssize_t nV,
                             Py_ssize_t nE,
                             Py_ssize_t nF,
                             np.ndarray[i32, ndim=1] vfc,
                             np.ndarray[f64, ndim=1] rho):
    """Face structure metrics in six phases. O(nnz(B_2) + nE).

    Phase 5 uses Welford's single-pass algorithm for face
    concentration (CV), avoiding the two-pass mean-then-variance
    approach that iterates B_2 twice per face.

    Parameters
    ----------
    B2 : DualCSR, shape (nE, nF).
    edge_src, edge_tgt : i32[nE]
    nV, nE, nF : int
    vfc : i32[nV]
        Precomputed vertex face counts.
    rho : f64[nE]
        Per-edge harmonic resistance ratio.

    Returns
    -------
    dict
    """
    cdef np.ndarray[f64, ndim=1] v_avg_c = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] v_tot_c = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] v_avg_sz = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] e_avg_c = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] e_tot_c = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] e_avg_sz = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] e_asym = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f_conc = np.zeros(nF, dtype=np.float64)

    cdef f64[::1] vac = v_avg_c, vtc = v_tot_c, vas = v_avg_sz
    cdef f64[::1] eac = e_avg_c, etc_ = e_tot_c, eas = e_avg_sz
    cdef f64[::1] ea = e_asym, fc = f_conc
    cdef i32[::1] es = edge_src, et = edge_tgt
    cdef i32[::1] vfcv = vfc
    cdef f64[::1] rhov = rho

    cdef i32[::1] rp = B2.row_ptr
    cdef i32[::1] ci = B2.col_idx
    cdef i32[::1] cp = B2.col_ptr
    cdef i32[::1] ri = B2.row_idx

    cdef np.ndarray[i32, ndim=1] face_sz_arr = np.empty(nF, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] face_nv_arr = np.empty(nF, dtype=np.int32)
    cdef np.ndarray[i64, ndim=1] last_seen_arr = np.full(nV, -1, dtype=np.int64)
    cdef i32[::1] face_sz = face_sz_arr, face_nv = face_nv_arr
    cdef i64[::1] ls = last_seen_arr

    cdef Py_ssize_t f, e, k, s, t, efc
    cdef Py_ssize_t fi
    cdef double sum_inv_sz, sum_sz, inv_nv
    cdef double mx, diff
    cdef double mean, var, delta
    cdef i32 nv_count

    # Phase 1: face sizes and face vertex counts
    for f in range(nF):
        face_sz[f] = cp[f + 1] - cp[f]
        fi = f
        face_nv[f] = 0
        for k in range(cp[f], cp[f + 1]):
            e = ri[k]
            s = es[e]; t = et[e]
            if ls[s] != fi:
                ls[s] = fi
                face_nv[f] += 1
            if ls[t] != fi:
                ls[t] = fi
                face_nv[f] += 1

    # Phase 2: per-edge contribution metrics
    for e in range(nE):
        efc = rp[e + 1] - rp[e]
        if efc == 0:
            continue
        sum_inv_sz = 0.0
        sum_sz = 0.0
        for k in range(rp[e], rp[e + 1]):
            f = ci[k]
            sum_inv_sz += 1.0 / <double>face_sz[f]
            sum_sz += <double>face_sz[f]
        etc_[e] = sum_inv_sz
        eac[e] = sum_inv_sz / <double>efc
        eas[e] = sum_sz / <double>efc

    # Phase 3: boundary asymmetry per edge
    for e in range(nE):
        s = es[e]; t = et[e]
        diff = <double>(vfcv[s] - vfcv[t])
        if diff < 0:
            diff = -diff
        mx = <double>vfcv[s]
        if <double>vfcv[t] > mx:
            mx = <double>vfcv[t]
        if mx > 0.0:
            ea[e] = diff / mx

    # Phase 4: per-vertex contribution metrics
    cdef Py_ssize_t gen_offset = nF
    for f in range(nF):
        fi = f + gen_offset
        inv_nv = 1.0 / <double>face_nv[f] if face_nv[f] > 0 else 0.0
        for k in range(cp[f], cp[f + 1]):
            e = ri[k]
            s = es[e]; t = et[e]
            if ls[s] != fi:
                ls[s] = fi
                vtc[s] += inv_nv
                vas[s] += <double>face_sz[f]
            if ls[t] != fi:
                ls[t] = fi
                vtc[t] += inv_nv
                vas[t] += <double>face_sz[f]

    for s in range(nV):
        if vfcv[s] > 0:
            vac[s] = vtc[s] / <double>vfcv[s]
            vas[s] = vas[s] / <double>vfcv[s]

    # Phase 5: face concentration via Welford single-pass
    # Old code iterated B_2 twice per face (mean pass, variance pass).
    # Welford computes both in one pass with one generation counter.
    cdef Py_ssize_t gen_offset2 = 2 * nF
    for f in range(nF):
        if face_nv[f] < 2:
            fc[f] = 0.0
            continue

        fi = f + gen_offset2
        mean = 0.0
        var = 0.0
        nv_count = 0

        for k in range(cp[f], cp[f + 1]):
            e = ri[k]
            s = es[e]; t = et[e]
            if ls[s] != fi:
                ls[s] = fi
                nv_count += 1
                delta = <double>vfcv[s] - mean
                mean += delta / <double>nv_count
                var += delta * (<double>vfcv[s] - mean)
            if ls[t] != fi:
                ls[t] = fi
                nv_count += 1
                delta = <double>vfcv[t] - mean
                mean += delta / <double>nv_count
                var += delta * (<double>vfcv[t] - mean)

        if nv_count > 0 and mean > get_EPSILON_NORM():
            var = var / <double>nv_count
            fc[f] = sqrt(var) / mean
        else:
            fc[f] = 0.0

    # Phase 6: Pearson correlation between bnd_asym and rho
    cdef double corr = _pearson_corr(ea, rhov, nE)

    cdef double v_tc_sum = 0.0, e_tc_sum = 0.0
    for s in range(nV):
        v_tc_sum += vtc[s]
    for e in range(nE):
        e_tc_sum += etc_[e]

    return {
        'v_avg_contrib':   v_avg_c,
        'v_total_contrib': v_tot_c,
        'v_avg_face_size': v_avg_sz,
        'e_avg_contrib':   e_avg_c,
        'e_total_contrib': e_tot_c,
        'e_avg_face_size': e_avg_sz,
        'e_bnd_asym':      e_asym,
        'f_concentration': f_conc,
        'v_tc_sum':        v_tc_sum,
        'e_tc_sum':        e_tc_sum,
        'asym_rho_corr':   corr,
    }


# Face metrics, i64 variant

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_face_metrics_i64(B2,
                             np.ndarray[i64, ndim=1] edge_src,
                             np.ndarray[i64, ndim=1] edge_tgt,
                             Py_ssize_t nV,
                             Py_ssize_t nE,
                             Py_ssize_t nF,
                             np.ndarray[i32, ndim=1] vfc,
                             np.ndarray[f64, ndim=1] rho):
    """Face structure metrics. int64 index variant."""
    cdef np.ndarray[f64, ndim=1] v_avg_c = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] v_tot_c = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] v_avg_sz = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] e_avg_c = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] e_tot_c = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] e_avg_sz = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] e_asym = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] f_conc = np.zeros(nF, dtype=np.float64)

    cdef f64[::1] vac = v_avg_c, vtc = v_tot_c, vas = v_avg_sz
    cdef f64[::1] eac = e_avg_c, etc_ = e_tot_c, eas = e_avg_sz
    cdef f64[::1] ea = e_asym, fc = f_conc
    cdef i64[::1] es = edge_src, et = edge_tgt
    cdef i32[::1] vfcv = vfc
    cdef f64[::1] rhov = rho

    cdef i64[::1] rp = B2.row_ptr
    cdef i64[::1] ci = B2.col_idx
    cdef i64[::1] cp = B2.col_ptr
    cdef i64[::1] ri = B2.row_idx

    cdef np.ndarray[i32, ndim=1] face_sz_arr = np.empty(nF, dtype=np.int32)
    cdef np.ndarray[i32, ndim=1] face_nv_arr = np.empty(nF, dtype=np.int32)
    cdef np.ndarray[i64, ndim=1] last_seen_arr = np.full(nV, -1, dtype=np.int64)
    cdef i32[::1] face_sz = face_sz_arr, face_nv = face_nv_arr
    cdef i64[::1] ls = last_seen_arr

    cdef Py_ssize_t f, k, efc
    cdef i64 e_idx
    cdef Py_ssize_t s, t, e
    cdef Py_ssize_t fi
    cdef double sum_inv_sz, sum_sz, inv_nv
    cdef double mx, diff
    cdef double mean, var, delta
    cdef i32 nv_count

    # Phase 1
    for f in range(nF):
        face_sz[f] = <i32>(cp[f + 1] - cp[f])
        fi = f
        face_nv[f] = 0
        for k in range(cp[f], cp[f + 1]):
            e_idx = ri[k]
            s = <Py_ssize_t>es[e_idx]; t = <Py_ssize_t>et[e_idx]
            if ls[s] != fi:
                ls[s] = fi
                face_nv[f] += 1
            if ls[t] != fi:
                ls[t] = fi
                face_nv[f] += 1

    # Phase 2
    for e in range(nE):
        efc = <Py_ssize_t>(rp[e + 1] - rp[e])
        if efc == 0:
            continue
        sum_inv_sz = 0.0
        sum_sz = 0.0
        for k in range(rp[e], rp[e + 1]):
            f = <Py_ssize_t>ci[k]
            sum_inv_sz += 1.0 / <double>face_sz[f]
            sum_sz += <double>face_sz[f]
        etc_[e] = sum_inv_sz
        eac[e] = sum_inv_sz / <double>efc
        eas[e] = sum_sz / <double>efc

    # Phase 3
    for e in range(nE):
        s = <Py_ssize_t>es[e]; t = <Py_ssize_t>et[e]
        diff = <double>(vfcv[s] - vfcv[t])
        if diff < 0:
            diff = -diff
        mx = <double>vfcv[s]
        if <double>vfcv[t] > mx:
            mx = <double>vfcv[t]
        if mx > 0.0:
            ea[e] = diff / mx

    # Phase 4
    cdef Py_ssize_t gen_offset = nF
    for f in range(nF):
        fi = f + gen_offset
        inv_nv = 1.0 / <double>face_nv[f] if face_nv[f] > 0 else 0.0
        for k in range(cp[f], cp[f + 1]):
            e_idx = ri[k]
            s = <Py_ssize_t>es[e_idx]; t = <Py_ssize_t>et[e_idx]
            if ls[s] != fi:
                ls[s] = fi
                vtc[s] += inv_nv
                vas[s] += <double>face_sz[f]
            if ls[t] != fi:
                ls[t] = fi
                vtc[t] += inv_nv
                vas[t] += <double>face_sz[f]

    for s in range(nV):
        if vfcv[s] > 0:
            vac[s] = vtc[s] / <double>vfcv[s]
            vas[s] = vas[s] / <double>vfcv[s]

    # Phase 5: Welford single-pass
    cdef Py_ssize_t gen_offset2 = 2 * nF
    for f in range(nF):
        if face_nv[f] < 2:
            fc[f] = 0.0
            continue

        fi = f + gen_offset2
        mean = 0.0
        var = 0.0
        nv_count = 0

        for k in range(cp[f], cp[f + 1]):
            e_idx = ri[k]
            s = <Py_ssize_t>es[e_idx]; t = <Py_ssize_t>et[e_idx]
            if ls[s] != fi:
                ls[s] = fi
                nv_count += 1
                delta = <double>vfcv[s] - mean
                mean += delta / <double>nv_count
                var += delta * (<double>vfcv[s] - mean)
            if ls[t] != fi:
                ls[t] = fi
                nv_count += 1
                delta = <double>vfcv[t] - mean
                mean += delta / <double>nv_count
                var += delta * (<double>vfcv[t] - mean)

        if nv_count > 0 and mean > get_EPSILON_NORM():
            var = var / <double>nv_count
            fc[f] = sqrt(var) / mean
        else:
            fc[f] = 0.0

    # Phase 6
    cdef double corr = _pearson_corr(ea, rhov, nE)

    cdef double v_tc_sum = 0.0, e_tc_sum = 0.0
    for s in range(nV):
        v_tc_sum += vtc[s]
    for e in range(nE):
        e_tc_sum += etc_[e]

    return {
        'v_avg_contrib':   v_avg_c,
        'v_total_contrib': v_tot_c,
        'v_avg_face_size': v_avg_sz,
        'e_avg_contrib':   e_avg_c,
        'e_total_contrib': e_tot_c,
        'e_avg_face_size': e_avg_sz,
        'e_bnd_asym':      e_asym,
        'f_concentration': f_conc,
        'v_tc_sum':        v_tc_sum,
        'e_tc_sum':        e_tc_sum,
        'asym_rho_corr':   corr,
    }


# Combined face builder

def build_face_data(B2, edge_src, edge_tgt, Py_ssize_t nV,
                    vertex_names, edge_names,
                    np.ndarray[f64, ndim=1] rho):
    """Face classification, extraction, vertex counts, and metrics.

    Runs classify_faces first to identify self-loop faces, then
    passes the classification to extract_faces so is_self_loop is
    set on each face descriptor.

    Parameters
    ----------
    B2 : DualCSR, shape (nE, nF).
    edge_src, edge_tgt : int array [nE]
    nV : int
    vertex_names : list[str]
    edge_names : list[str]
    rho : f64[nE]
        Per-edge harmonic resistance ratio.

    Returns
    -------
    dict
        faces : list[dict]
            Per-face descriptors (id, boundary, vertices, size,
            is_self_loop).
        face_class : dict
            Output of classify_faces (proper_mask, self_loop_mask,
            n_proper, n_self_loop, proper_indices, self_loop_indices).
        vertex_face_count : i32[nV]
        metrics : dict
    """
    cdef Py_ssize_t nE = B2.nrow
    cdef Py_ssize_t nF = B2.ncol

    if nF == 0:
        return {
            'faces': [],
            'face_class': {
                'proper_mask': np.empty(0, dtype=bool),
                'self_loop_mask': np.empty(0, dtype=bool),
                'n_proper': 0,
                'n_self_loop': 0,
                'proper_indices': np.empty(0, dtype=np.intp),
                'self_loop_indices': np.empty(0, dtype=np.intp),
            },
            'vertex_face_count': np.zeros(nV, dtype=np.int32),
            'metrics': {
                'v_avg_contrib':   np.zeros(nV, dtype=np.float64),
                'v_total_contrib': np.zeros(nV, dtype=np.float64),
                'v_avg_face_size': np.zeros(nV, dtype=np.float64),
                'e_avg_contrib':   np.zeros(nE, dtype=np.float64),
                'e_total_contrib': np.zeros(nE, dtype=np.float64),
                'e_avg_face_size': np.zeros(nE, dtype=np.float64),
                'e_bnd_asym':      np.zeros(nE, dtype=np.float64),
                'f_concentration': np.empty(0, dtype=np.float64),
                'v_tc_sum':        0.0,
                'e_tc_sum':        0.0,
                'asym_rho_corr':   0.0,
            },
        }

    fc = classify_faces(B2, edge_src, edge_tgt)
    vfc = vertex_face_count(B2, edge_src, edge_tgt, nV)
    faces = extract_faces(B2, edge_src, edge_tgt, vertex_names,
                          edge_names, face_class=fc)

    if B2.idx_bits == 64:
        metrics = compute_face_metrics_i64(
            B2, edge_src, edge_tgt, nV, nE, nF, vfc, rho)
    else:
        metrics = compute_face_metrics_i32(
            B2, edge_src, edge_tgt, nV, nE, nF, vfc, rho)

    return {
        'faces': faces,
        'face_class': fc,
        'vertex_face_count': vfc,
        'metrics': metrics,
    }


# Typed boundary decomposition


@cython.boundscheck(False)
@cython.wraparound(False)
def typed_face_selection(np.ndarray[i32, ndim=1] edge_types,
                          np.ndarray[i32, ndim=1] adj_ptr,
                          np.ndarray[i32, ndim=1] adj_idx,
                          np.ndarray[i32, ndim=1] adj_edge,
                          Py_ssize_t nV, Py_ssize_t nE,
                          int n_types):
    """Typed boundary decomposition face selection.

    B_1 is decomposed by edge type: B_1 = sum_t B_{1,t}. A triangle
    (e_uv, e_uw, e_vw) is realized as a face if and only if all three
    boundary edges share the same type. Cross-type triangles become
    voids. No external parameters are needed.

    Triangle enumeration uses adjacency intersection: for each vertex u,
    for each pair of neighbors (v, w) where v < w and (v, w) are
    adjacent, the triple (u, v, w) forms a triangle. Each triangle is
    found exactly once when u = min(u, v, w).

    Parameters
    ----------
    edge_types : i32[nE]
        Per-edge type label (0, 1, ..., n_types-1).
    adj_ptr : i32[nV+1]
        Symmetric adjacency CSR row pointer.
    adj_idx : i32[nnz]
        Symmetric adjacency neighbor indices.
    adj_edge : i32[nnz]
        Edge index for each adjacency entry.
    nV, nE : int
    n_types : int
        Number of distinct edge types.

    Returns
    -------
    dict
        realized_edges : i32[nF_realized * 3]
            Flat array of edge indices for realized faces (groups of 3).
        realized_signs : f64[nF_realized * 3]
            Boundary signs for realized faces.
        nF_realized : int
        void_edges : i32[nF_void * 3]
            Flat array of edge indices for void triangles.
        void_signs : f64[nF_void * 3]
            Boundary signs for void triangles.
        nF_void : int
        face_types : i32[nF_realized]
            Type label for each realized face.
        n_triangles : int
            Total triangles enumerated.
    """
    cdef i32[::1] et = edge_types
    cdef i32[::1] ap = adj_ptr, ai = adj_idx, ae = adj_edge

    # Phase 1: enumerate all triangles, partition into realized vs void
    # For each u, iterate neighbors v > u; for each such v, intersect
    # N(u) ∩ N(v) for w > v. This finds each triangle exactly once.
    realized_e_list = []
    realized_s_list = []
    realized_t_list = []
    void_e_list = []
    void_s_list = []

    cdef Py_ssize_t u, v, w
    cdef Py_ssize_t j_v, j_w, lo_v, hi_v, lo_w, hi_w
    cdef i32 e_uv, e_uw, e_vw
    cdef i32 t_uv, t_uw, t_vw
    cdef Py_ssize_t p_u, p_w

    for u in range(nV):
        lo_v = ap[u]
        hi_v = ap[u + 1]
        for j_v in range(lo_v, hi_v):
            v = ai[j_v]
            if v <= u:
                continue
            e_uv = ae[j_v]

            # Intersect N(u) and N(v) for w > v
            lo_w = ap[v]
            hi_w = ap[v + 1]
            p_u = lo_v      # pointer into N(u), scanning for w > v
            p_w = lo_w      # pointer into N(v)

            # Advance p_u past entries <= v
            while p_u < hi_v and ai[p_u] <= v:
                p_u += 1
            # Advance p_w past entries <= v
            while p_w < hi_w and ai[p_w] <= v:
                p_w += 1

            # Merge-intersect N(u)[p_u:] and N(v)[p_w:] (both sorted, w > v)
            while p_u < hi_v and p_w < hi_w:
                if ai[p_u] < ai[p_w]:
                    p_u += 1
                elif ai[p_u] > ai[p_w]:
                    p_w += 1
                else:
                    # w = ai[p_u] = ai[p_w], triangle (u, v, w) found
                    w = ai[p_u]
                    e_uw = ae[p_u]
                    e_vw = ae[p_w]

                    t_uv = et[e_uv]
                    t_uw = et[e_uw]
                    t_vw = et[e_vw]

                    if t_uv == t_uw and t_uw == t_vw:
                        # All same type: realized face
                        realized_e_list.append(e_uv)
                        realized_e_list.append(e_uw)
                        realized_e_list.append(e_vw)
                        realized_s_list.append(1.0)
                        realized_s_list.append(-1.0)
                        realized_s_list.append(1.0)
                        realized_t_list.append(t_uv)
                    else:
                        # Cross-type: void
                        void_e_list.append(e_uv)
                        void_e_list.append(e_uw)
                        void_e_list.append(e_vw)
                        void_s_list.append(1.0)
                        void_s_list.append(-1.0)
                        void_s_list.append(1.0)

                    p_u += 1
                    p_w += 1

    cdef Py_ssize_t nF_r = len(realized_e_list) // 3
    cdef Py_ssize_t nF_v = len(void_e_list) // 3

    return {
        'realized_edges': np.array(realized_e_list, dtype=np.int32),
        'realized_signs': np.array(realized_s_list, dtype=np.float64),
        'nF_realized': int(nF_r),
        'void_edges': np.array(void_e_list, dtype=np.int32),
        'void_signs': np.array(void_s_list, dtype=np.float64),
        'nF_void': int(nF_v),
        'face_types': np.array(realized_t_list, dtype=np.int32),
        'n_triangles': int(nF_r + nF_v),
    }


# Algebraic context face selection


@cython.boundscheck(False)
@cython.wraparound(False)
def context_face_selection(np.ndarray[f64, ndim=2] B1,
                            np.ndarray[np.uint8_t, ndim=2] context_matrix,
                            np.ndarray[i32, ndim=1] adj_ptr,
                            np.ndarray[i32, ndim=1] adj_idx,
                            np.ndarray[i32, ndim=1] adj_edge,
                            Py_ssize_t nV, Py_ssize_t nE):
    """Algebraic context face selection via E = C^T |B1| > 0.

    Given a binary context matrix C (n_contexts x nV) and the boundary
    operator B1, the entry matrix E = C^T |B1| determines which edges
    are visible to each context. A triangle (e1, e2, e3) is included
    as a face if and only if some context d has
    E[d, e1] * E[d, e2] * E[d, e3] > 0.

    This is a single matrix multiply with no thresholds.

    Also computes per-context face realization counts: how many
    triangles each individual context covers.

    Parameters
    ----------
    B1 : f64[nV, nE]
        Signed incidence matrix.
    context_matrix : uint8[n_contexts, nV]
        Binary context incidence. C[d, v] = 1 if context d covers
        vertex v.
    adj_ptr : i32[nV+1]
        Symmetric adjacency CSR row pointer.
    adj_idx : i32[nnz]
        Symmetric adjacency neighbor indices.
    adj_edge : i32[nnz]
        Edge index for each adjacency entry.
    nV, nE : int

    Returns
    -------
    dict
        cycle_edges : i32[nF * 3]
            Flat edge indices for realized triangles.
        cycle_signs : f64[nF * 3]
            Boundary signs.
        cycle_lengths : i32[nF]
            Always 3 for triangles.
        nF : int
            Number of realized faces.
        n_triangles : int
            Total triangles enumerated.
        per_context_face_count : i32[n_contexts]
            Faces realized per context.
        per_context_void_fraction : f64[n_contexts]
            1 - face_count / n_triangles per context.
        void_edges : i32[nF_void * 3]
            Flat edge indices for void triangles.
        nF_void : int
    """
    cdef Py_ssize_t n_ctx = context_matrix.shape[0]
    cdef np.uint8_t[:, ::1] cm = context_matrix

    # E[d, e] = sum of C[d, v] over boundary vertices v of edge e.
    # For standard 2-endpoint edges, E[d, e] = C[d, src(e)] + C[d, tgt(e)].
    # Threshold E[d, e] > 0: edge e is visible to context d if at least
    # one boundary vertex belongs to that context.

    cdef np.ndarray[f64, ndim=2] abs_B1 = np.abs(B1)
    # C is uint8, cast to f64 for matmul
    cdef np.ndarray[f64, ndim=2] C_f = np.ascontiguousarray(
        context_matrix, dtype=np.float64)
    # E = C @ |B1|, shape (n_ctx, nE)
    cdef np.ndarray[f64, ndim=2] E = C_f @ abs_B1
    # Threshold: E[d, e] > 0 means edge e is visible to context d
    cdef np.ndarray[np.uint8_t, ndim=2] E_vis = (E > 0.0).astype(np.uint8)
    cdef np.uint8_t[:, ::1] ev = E_vis

    cdef i32[::1] ap = adj_ptr, ai = adj_idx, ae = adj_edge

    # Enumerate all triangles
    # For each triangle, check if any context covers all 3 edges.
    # Also track per-context counts.
    face_e_list = []
    face_s_list = []
    void_e_list = []
    void_s_list = []

    # Collect all triangles first so we can count per-context
    all_tri_edges = []  # list of (e1, e2, e3) tuples

    cdef Py_ssize_t u, v, w
    cdef Py_ssize_t j_v, lo_v, hi_v, lo_w, hi_w
    cdef i32 e_uv, e_uw, e_vw
    cdef Py_ssize_t p_u, p_w

    for u in range(nV):
        lo_v = ap[u]
        hi_v = ap[u + 1]
        for j_v in range(lo_v, hi_v):
            v = ai[j_v]
            if v <= u:
                continue
            e_uv = ae[j_v]

            lo_w = ap[v]
            hi_w = ap[v + 1]
            p_u = lo_v
            p_w = lo_w

            while p_u < hi_v and ai[p_u] <= v:
                p_u += 1
            while p_w < hi_w and ai[p_w] <= v:
                p_w += 1

            while p_u < hi_v and p_w < hi_w:
                if ai[p_u] < ai[p_w]:
                    p_u += 1
                elif ai[p_u] > ai[p_w]:
                    p_w += 1
                else:
                    w = ai[p_u]
                    e_uw = ae[p_u]
                    e_vw = ae[p_w]
                    all_tri_edges.append((e_uv, e_uw, e_vw))
                    p_u += 1
                    p_w += 1

    cdef Py_ssize_t n_tri = len(all_tri_edges)

    # Phase 2: for each triangle, check global coverage (any context
    # covers all 3 edges). Also build per-context face counts.
    cdef np.ndarray[i32, ndim=1] per_ctx = np.zeros(n_ctx, dtype=np.int32)
    cdef i32[::1] pcv = per_ctx
    cdef Py_ssize_t d, ti
    cdef i32 e1, e2, e3
    cdef bint any_covers

    for ti in range(n_tri):
        e1 = all_tri_edges[ti][0]
        e2 = all_tri_edges[ti][1]
        e3 = all_tri_edges[ti][2]

        any_covers = False
        for d in range(n_ctx):
            if ev[d, e1] and ev[d, e2] and ev[d, e3]:
                pcv[d] += 1
                if not any_covers:
                    any_covers = True

        if any_covers:
            face_e_list.append(e1)
            face_e_list.append(e2)
            face_e_list.append(e3)
            face_s_list.append(1.0)
            face_s_list.append(-1.0)
            face_s_list.append(1.0)
        else:
            void_e_list.append(e1)
            void_e_list.append(e2)
            void_e_list.append(e3)
            void_s_list.append(1.0)
            void_s_list.append(-1.0)
            void_s_list.append(1.0)

    cdef Py_ssize_t nF = len(face_e_list) // 3
    cdef Py_ssize_t nF_v = len(void_e_list) // 3

    # Per-context void fractions
    cdef np.ndarray[f64, ndim=1] per_ctx_vf = np.ones(n_ctx, dtype=np.float64)
    cdef f64[::1] pvf = per_ctx_vf
    if n_tri > 0:
        for d in range(n_ctx):
            pvf[d] = 1.0 - <f64>pcv[d] / <f64>n_tri

    # Build cycle_lengths (all 3s)
    cdef np.ndarray[i32, ndim=1] clens = np.full(nF, 3, dtype=np.int32)

    return {
        'cycle_edges': np.array(face_e_list, dtype=np.int32),
        'cycle_signs': np.array(face_s_list, dtype=np.float64),
        'cycle_lengths': clens,
        'nF': int(nF),
        'n_triangles': int(n_tri),
        'per_context_face_count': per_ctx,
        'per_context_void_fraction': per_ctx_vf,
        'void_edges': np.array(void_e_list, dtype=np.int32),
        'nF_void': int(nF_v),
    }


# Void type composition


def void_type_composition(np.ndarray[i32, ndim=1] void_edges,
                           np.ndarray[i32, ndim=1] edge_types,
                           Py_ssize_t nF_void,
                           int n_types):
    """Decompose void triangles by cross-type edge composition.

    Each void triangle has edges from 2 or 3 different types. This
    function counts how many voids have each type combination.

    Parameters
    ----------
    void_edges : i32[nF_void * 3]
        Flat edge indices from typed_face_selection or
        context_face_selection.
    edge_types : i32[nE]
        Per-edge type label.
    nF_void : int
        Number of void triangles.
    n_types : int
        Number of distinct edge types.

    Returns
    -------
    dict
        type_pairs : list of tuple
            Sorted type sets for each void (e.g., (0, 1), (0, 1, 2)).
        pair_counts : dict
            Mapping from type set (as tuple) to count.
        pair_fractions : dict
            Mapping from type set to fraction of all voids.
        n_cross_type_2 : int
            Voids with exactly 2 distinct edge types.
        n_cross_type_3 : int
            Voids with 3 distinct edge types.
    """
    cdef i32[::1] ve = void_edges, et = edge_types
    cdef Py_ssize_t f, idx
    cdef i32 t0, t1, t2

    pair_list = []
    pair_counts = {}
    cdef int n2 = 0, n3 = 0

    for f in range(nF_void):
        idx = f * 3
        t0 = et[ve[idx]]
        t1 = et[ve[idx + 1]]
        t2 = et[ve[idx + 2]]

        types_set = tuple(sorted(set((int(t0), int(t1), int(t2)))))
        pair_list.append(types_set)

        if types_set in pair_counts:
            pair_counts[types_set] += 1
        else:
            pair_counts[types_set] = 1

        if len(types_set) == 2:
            n2 += 1
        elif len(types_set) >= 3:
            n3 += 1

    pair_fractions = {}
    if nF_void > 0:
        for k, v in pair_counts.items():
            pair_fractions[k] = float(v) / float(nF_void)

    return {
        'type_pairs': pair_list,
        'pair_counts': pair_counts,
        'pair_fractions': pair_fractions,
        'n_cross_type_2': n2,
        'n_cross_type_3': n3,
    }
