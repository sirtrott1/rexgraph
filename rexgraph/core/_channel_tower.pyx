# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""rexgraph.core._channel_tower: four channel diagonals in O(nnz) at every arity.

For boundary coefficients c_e[v], unweighted vertex mass M[v] gives
C[e] = sum_v |c_e[v]| (M[v] - |c_e[v]|). Frustration uses twice the
opposite sign mass, weighted by relation metric magnitudes.

T and the G channel use squared boundary coefficients and squared relation
metrics. C excludes the metric. Canonical branching columns have head -1 and
shares 1/(k-1); witnesses (+1) contribute to positive mass. Declared columns
use their supplied coefficients.

transpose_incidence groups occurrences by vertex for disjoint parallel writes.
Pass a retained transpose through transposed to reuse it. The kernel evaluates
rational identities in float64.
"""

from __future__ import annotations

import numpy as np

cimport numpy as np
from cython.parallel cimport prange
from libc.stdint cimport int32_t, int64_t
from libc.math cimport fabs

np.import_array()


cdef inline void _vertex_mass(const int32_t* bp, const int32_t* ow,
                             const np.uint8_t* ih, const double* wv,
                             int64_t lo, int64_t hi,
                             const double* coef, const int64_t* src,
                             double* out) noexcept nogil:
    """Write negative and positive masses at one vertex, weighted and unweighted.

    `coef` holds boundary entries in relation incidence order; `src` maps transposed
    occurrences to their original slots. When `coef` is NULL, entries are derived
    from arity and head flags. Supplied coefficients split mass by their sign.
    """
    cdef double nw = 0.0, pw = 0.0, nu = 0.0, pu = 0.0, mg, c, we
    cdef int64_t q
    cdef Py_ssize_t f, kf
    for q in range(lo, hi):
        f = ow[q]
        we = fabs(wv[f])
        if coef != NULL:
            c = coef[src[q]]
            mg = c if c >= 0 else -c
            if c < 0:
                nw += we * mg; nu += mg
            else:
                pw += we * mg; pu += mg
            continue
        kf = bp[f + 1] - bp[f]
        if kf == 1:
            pw += we; pu += 1.0                  # the witness is (+1)
        elif ih[q]:
            nw += we; nu += 1.0                  # the head, magnitude 1
        else:
            mg = 1.0 / (kf - 1)
            pw += we * mg; pu += mg
    out[0] = nw; out[1] = pw; out[2] = nu; out[3] = pu


cdef inline void _bucket_offsets(int64_t* h, Py_ssize_t v, Py_ssize_t nV,
                                int nthr, int64_t start) noexcept nogil:
    """Turn one vertex bucket's per thread counts into write cursors, in place."""
    cdef int64_t run = start, c
    cdef int ti
    for ti in range(nthr):
        c = h[ti * nV + v]
        h[ti * nV + v] = run
        run += c


def transpose_incidence(np.ndarray boundary_ptr not None,
                        np.ndarray boundary_idx not None,
                        Py_ssize_t nV,
                        int threads=1,
                        bint positions=False):
    """Transpose relation incidence into vertex ordered CSR.

    Return pointers, relation owners and head flags. positions=True also returns
    original incidence slots for coefficient lookup. Stable counting and filling
    retain occurrence order within each vertex across thread counts.

    Threads are capped by nnz // nV so their histograms do not exceed the
    incidence size. A retained transpose can serve repeated channel readings.
    """
    cdef const int32_t[::1] bp = np.ascontiguousarray(boundary_ptr, dtype=np.int32)
    cdef const int32_t[::1] bi = np.ascontiguousarray(boundary_idx, dtype=np.int32)
    cdef Py_ssize_t nE = bp.shape[0] - 1
    cdef Py_ssize_t nnz = bp[nE] if nE >= 0 else 0
    cdef np.ndarray[int64_t, ndim=1] vptr = np.zeros(nV + 1, dtype=np.int64)
    cdef np.ndarray[int32_t, ndim=1] owner = np.zeros(nnz, dtype=np.int32)
    cdef np.ndarray[np.uint8_t, ndim=1] is_head = np.zeros(nnz, dtype=np.uint8)
    cdef np.ndarray[int64_t, ndim=1] source = np.zeros(nnz if positions else 0, dtype=np.int64)
    cdef int64_t[::1] vp = vptr
    cdef int32_t[::1] ow = owner
    cdef np.uint8_t[::1] ih = is_head
    cdef int64_t[::1] srcv = source
    cdef Py_ssize_t e, p, s, t, v
    cdef int64_t at, cnt
    cdef np.ndarray[int64_t, ndim=1] cursor
    cdef int64_t[::1] cur
    cdef int nthr = threads if threads > 0 else 1
    cdef int ti
    cdef np.ndarray[int64_t, ndim=2] hist
    cdef int64_t[:, ::1] hv
    cdef np.ndarray[np.int64_t, ndim=1] bounds
    cdef int64_t[::1] bd

    if nV > 0 and nthr > 1:
        cnt = nnz // nV                       # the scratch may not exceed the data
        if cnt < nthr:
            nthr = <int>cnt if cnt > 1 else 1

    if nthr < 2:
        with nogil:
            for p in range(nnz):
                vp[bi[p] + 1] += 1
            for p in range(nV):
                vp[p + 1] += vp[p]
        cursor = vptr[:nV].copy()
        cur = cursor
        with nogil:
            for e in range(nE):
                s = bp[e]; t = bp[e + 1]
                for p in range(s, t):
                    at = cur[bi[p]]
                    ow[at] = <int32_t>e
                    ih[at] = 1 if p == s else 0
                    if positions:
                        srcv[at] = p
                    cur[bi[p]] = at + 1
        if positions:
            return vptr, owner, is_head, source
        return vptr, owner, is_head

    # contiguous RELATION ranges, so the entry ranges are contiguous and ordered
    bounds = np.linspace(0, nE, nthr + 1).astype(np.int64)
    bd = bounds
    hist = np.zeros((nthr, nV), dtype=np.int64)
    hv = hist
    with nogil:
        for ti in prange(nthr, num_threads=nthr, schedule='static'):
            for e in range(bd[ti], bd[ti + 1]):
                for p in range(bp[e], bp[e + 1]):
                    hv[ti, bi[p]] += 1

        # totals per vertex, then the exclusive prefix over threads INSIDE each bucket,
        # written back over the histogram so it becomes each thread's write cursor
        for v in range(nV):
            cnt = 0
            for ti in range(nthr):
                cnt += hv[ti, v]
            vp[v + 1] = vp[v] + cnt
        for v in prange(nV, num_threads=nthr, schedule='static'):
            _bucket_offsets(&hv[0, 0], v, nV, nthr, vp[v])

        for ti in prange(nthr, num_threads=nthr, schedule='static'):
            for e in range(bd[ti], bd[ti + 1]):
                s = bp[e]; t = bp[e + 1]
                for p in range(s, t):
                    at = hv[ti, bi[p]]
                    ow[at] = <int32_t>e
                    ih[at] = 1 if p == s else 0
                    if positions:
                        srcv[at] = p
                    hv[ti, bi[p]] = at + 1
    if positions:
        return vptr, owner, is_head, source
    return vptr, owner, is_head


def channel_diagonals_any_arity(np.ndarray boundary_ptr not None,
                                np.ndarray boundary_idx not None,
                                Py_ssize_t nV,
                                np.ndarray w_E=None,
                                int threads=1,
                                tuple transposed=None,
                                np.ndarray coefficients=None):
    """The four diagonals (T, G, F, C) for a complex of any arity, in O(nnz).

    `boundary_ptr`/`boundary_idx` are the CSC support of B1: relation e spans
    ``boundary_idx[boundary_ptr[e]:boundary_ptr[e+1]]``, its first entry the head.
    `w_E` is the per relation weight (signed or zero allowed), or None when unweighted.
    These diagonals read its magnitude without modifying the supplied array. The
    incidence must name distinct participants within each relation.

    `threads` sets the parallel width; 1 keeps the serial path. `transposed` accepts a
    previously built `transpose_incidence` result, since the incidence does not change
    between readings of the same complex.

    `coefficients` holds the boundary entry of each incidence for a declared head or
    share. Without it, columns are derived from arity and head flags. Canonical
    weighted quadrance uses `we*we*(1.0 + share)` for arity greater than one;
    declared quadrance sums squared supplied coefficients.

    Returns (T, G, F, C) as float64 arrays of length nE.
    """
    cdef const int32_t[::1] bp = np.ascontiguousarray(boundary_ptr, dtype=np.int32)
    cdef const int32_t[::1] bi = np.ascontiguousarray(boundary_idx, dtype=np.int32)
    cdef Py_ssize_t nE = bp.shape[0] - 1
    cdef np.ndarray[double, ndim=1] w = (np.ones(nE, dtype=np.float64) if w_E is None
                                      else np.ascontiguousarray(w_E, dtype=np.float64))
    if w.shape[0] != nE or not np.all(np.isfinite(w)):
        raise ValueError("one finite weight per relation is required")
    cdef const double[::1] wv = w

    cdef np.ndarray[double, ndim=1] T = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] G = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] F = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] C = np.zeros(nE, dtype=np.float64)
    cdef double[::1] Tv = T, Gv = G, Fv = F, Cv = C

    # the mass at each vertex, split by SIGN because F reads the opposite one, and
    # kept twice because C is unweighted where F is not
    cdef np.ndarray[double, ndim=2] mass = np.empty((nV, 4), dtype=np.float64)
    cdef double[:, ::1] mv = mass

    cdef Py_ssize_t e, p, s, t, k, v
    cdef double share, mag, we, a, m

    cdef bint declared = coefficients is not None
    if transposed is None or (declared and len(transposed) < 4):
        transposed = transpose_incidence(boundary_ptr, boundary_idx, nV, threads,
                                         positions=declared)
    cdef const int64_t[::1] vp = np.ascontiguousarray(transposed[0], dtype=np.int64)
    cdef const int32_t[::1] ow = np.ascontiguousarray(transposed[1], dtype=np.int32)
    cdef const np.uint8_t[::1] ih = np.ascontiguousarray(transposed[2], dtype=np.uint8)
    cdef np.ndarray[double, ndim=1] cf = (
        np.ascontiguousarray(coefficients, dtype=np.float64) if declared
        else np.zeros(0, dtype=np.float64))
    cdef np.ndarray[int64_t, ndim=1] sr = (
        np.ascontiguousarray(transposed[3], dtype=np.int64) if declared
        else np.zeros(0, dtype=np.int64))
    cdef const double[::1] cfv = cf
    cdef const int64_t[::1] srv = sr
    cdef const double* coef = &cfv[0] if declared else NULL
    cdef const int64_t* src = &srv[0] if declared else NULL
    if declared and cf.shape[0] != bi.shape[0]:
        raise ValueError("one coefficient per incidence is required")
    cdef int nthr = threads if threads > 0 else 1

    # pass 1: over VERTICES, so each thread owns what it writes
    with nogil:
        for v in prange(nV, num_threads=nthr, schedule='static'):
            _vertex_mass(&bp[0], &ow[0], &ih[0], &wv[0], vp[v], vp[v + 1],
                         coef, src, &mv[v, 0])
    with nogil:

        # pass 2: the readings, one per relation
        for e in prange(nE, num_threads=nthr, schedule='static'):
            s = bp[e]; t = bp[e + 1]; k = t - s
            if k == 0:
                continue
            we = fabs(wv[e])
            if coef != NULL:
                # The same four readings over the column as declared. Every case below
                # is this loop specialised: T is the weighted quadrance, C the weighted
                # unweighted line graph degree, and F twice the mass this entry disagrees with in
                # sign at its own vertex. A witness carries (+1) and no head, which is
                # why the split is by the SIGN of the entry and not by slot zero.
                for p in range(s, t):
                    v = bi[p]
                    m = coef[p]
                    a = m if m >= 0 else -m
                    Tv[e] += we * we * m * m
                    Cv[e] += a * (mv[v, 2] + mv[v, 3] - a)
                    if m < 0:
                        Fv[e] += we * a * mv[v, 1]
                    else:
                        Fv[e] += we * a * mv[v, 0]
                Gv[e] = Tv[e]
                Fv[e] *= 2.0
                continue
            if k == 1:
                Tv[e] = we * we
                Gv[e] = Tv[e]
                v = bi[s]
                Cv[e] = 1.0 * (mv[v, 2] + mv[v, 3] - 1.0)      # unweighted
                Fv[e] = 2.0 * we * mv[v, 0]                    # a witness is positive
                continue
            share = 1.0 / (k - 1)
            Tv[e] = we * we * (1.0 + share)
            Gv[e] = Tv[e]
            # the head: magnitude 1, negative
            v = bi[s]
            Cv[e] += 1.0 * (mv[v, 2] + mv[v, 3] - 1.0)
            Fv[e] += we * mv[v, 1]
            # the shared entries: magnitude 1/(k-1), positive
            mag = we * share
            for p in range(s + 1, t):
                v = bi[p]
                Cv[e] += share * (mv[v, 2] + mv[v, 3] - share)
                Fv[e] += mag * mv[v, 0]
            Fv[e] *= 2.0
    return T, G, F, C
