# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._channel_tower: the four channel diagonals at ANY arity, in O(nnz).

`sparse_character.channel_diagonals` already reads these in closed form, and says
exactly where it stops: the derivation is exact "for a SIGNED PAIRWISE UNWEIGHTED
complex and only there", because a branching column carries -1 and 1/(k-1), so an
off-diagonal T-G entry is not |s_e s_j - 1| and F is not a disagreement count. So
`closed_form_applies` refuses anything non-binary and the caller assembles instead,
which is the case a relational complex is built for.

What the disagreement count was standing in for is a MAGNITUDE, and accumulating the
magnitude works at every arity. Both off-diagonal channels are sums over pairs that
share a vertex, and a pair contributes only there, so the sum reorders onto the vertex
and the pairs never have to be formed:

    M[v] = SUM over relations f incident to v of |c_f[v]|
    C[e] = SUM over v in supp(e) of |c_e[v]| * (M[v] - |c_e[v]|)
    F[e] = 2 * SUM over v in supp(e) of |c_e[v]| * (opposite-sign mass at v)

M is one pass over the incidence and the readings are a second, so the whole tower is
O(nnz) rather than O(nE^2). F needs the mass split by SIGN rather than one total, since
T-G vanishes where two relations agree at a shared vertex and doubles where they do
not, so the accumulator is kept as (negative mass, positive mass) per vertex.

THE COLUMN, AND THE WITNESS. At arity k >= 2 the head is -1 and the other k-1 entries
share 1/(k-1), which is what makes the column sum to zero. A WITNESS is k = 1 and does
NOT follow the head rule: the construction emits (+1), so its entry joins the POSITIVE
mass. Reading it as a head puts it in the wrong accumulator and only F moves, which is
the same failure the exact tower had before.

T and G need no accumulator at all. Squaring kills the sign, so both diagonals are
1 + 1/(k-1) at arity k >= 2 and 1 at a witness, and k is the support size.

THE TRANSPOSE IS THE COST, AND IT IS REUSABLE. Measured on 12M nonzeros: the tower
itself is 28 ms at 12 threads and building the transpose is 190 ms, so 87% of a cold
call is the transpose. It is a counting sort with scattered writes and that is simply
what it costs; a vectorised argsort form was measured at 1938 ms, ten times SLOWER.
The incidence does not change between readings of the same complex, so `transposed`
is an argument: build it once with `transpose_incidence` and hand it back. Nothing here
caches it, because the complex carries __slots__ and an identity-keyed cache would
invalidate on the wrong thing.

PARALLELISM NEEDS THE INCIDENCE TRANSPOSED. Accumulating straight into the vertex
arrays has every relation writing wherever its support points, so threads collide and
the fix would be an atomic per nonzero. Transposing first, one counting pass and one
fill, makes the accumulation a loop OVER VERTICES instead, where each thread owns its
own vertices and writes nothing another reads. The second pass is over relations and
reads only. Measured single threaded the tower runs at ~116M nnz/s against a memory
system that does far more, so it is latency on scattered access that is being paid,
which is what the transpose removes.

WEIGHTING IS NOT UNIFORM ACROSS THE CHANNELS, and following the tower matters more
than being consistent. T and G scale by w_e^2 and F by w_e w_f, because G is T's
unsigned twin and has to carry the same per-relation metric or diag(T) != diag(G) at
any w != 1 and the identity F is defined by breaks. C stays UNWEIGHTED: co-participation
is a topological fact about which relations meet, not a geometric one. So the vertex
mass is kept twice, weighted for F and unweighted for C.
"""

from __future__ import annotations

import numpy as np

cimport numpy as np
from cython.parallel cimport prange
from libc.stdint cimport int32_t, int64_t

np.import_array()


cdef inline void _vertex_mass(const int32_t* bp, const int32_t* ow,
                             const np.uint8_t* ih, const double* wv,
                             int64_t lo, int64_t hi,
                             double* out) noexcept nogil:
    """The four masses at one vertex. Kept in a helper so prange sees an assignment
    and not an accumulator it would infer as a reduction."""
    cdef double nw = 0.0, pw = 0.0, nu = 0.0, pu = 0.0, mg
    cdef int64_t q
    cdef Py_ssize_t f, kf
    for q in range(lo, hi):
        f = ow[q]
        kf = bp[f + 1] - bp[f]
        if kf == 1:
            pw += wv[f]; pu += 1.0                # the witness is (+1)
        elif ih[q]:
            nw += wv[f]; nu += 1.0                # the head, magnitude 1
        else:
            mg = 1.0 / (kf - 1)
            pw += wv[f] * mg; pu += mg
    out[0] = nw; out[1] = pw; out[2] = nu; out[3] = pu


cdef inline void _bucket_offsets(int64_t* h, Py_ssize_t v, Py_ssize_t nV,
                                int nthr, int64_t start) noexcept nogil:
    """Turn one bucket's per-thread counts into per-thread write cursors, in place.

    In a helper for the same reason `_vertex_mass` is: prange sees an assignment here
    rather than `run += c`, which it would otherwise infer as a reduction over the
    parallel index and refuse to let the body read back."""
    cdef int64_t run = start, c
    cdef int ti
    for ti in range(nthr):
        c = h[ti * nV + v]
        h[ti * nV + v] = run
        run += c


def transpose_incidence(np.ndarray boundary_ptr not None,
                        np.ndarray boundary_idx not None,
                        Py_ssize_t nV,
                        int threads=1):
    """Vertex -> the entries that touch it, as CSR over nnz. One counting pass, one
    fill. `owner` is the relation each entry belongs to and `is_head` whether it is
    that relation's distinguished entry.

    This is a counting sort, and it dominates a cold call: 135.7 ms of a 162.4 ms
    read at 10.5M nonzeros, where the accumulation it feeds is only ~36. It is not
    slow code (measured against a numpy argsort route at 1836 ms and scipy at 236),
    it is a serial scatter, so `threads` splits it.

    The split is STABLE, which is not decoration: the accumulation sums float
    magnitudes per vertex, so reordering a bucket changes the last bits. Each thread
    takes a contiguous range of RELATIONS, hence a contiguous range of entries, so
    thread t's entries all precede thread t+1's inside every bucket and the result is
    byte-identical to the serial fill.

    Threads are capped so the per-thread histogram never exceeds the array it is
    permuting: `nthr <= nnz // nV`. That is a comparison between two sizes the caller
    already has and not a memory budget someone picked, and it matters because the
    histogram is `nthr x nV` while the data is `nnz`.
    """
    cdef int32_t[::1] bp = np.ascontiguousarray(boundary_ptr, dtype=np.int32)
    cdef int32_t[::1] bi = np.ascontiguousarray(boundary_idx, dtype=np.int32)
    cdef Py_ssize_t nE = bp.shape[0] - 1
    cdef Py_ssize_t nnz = bp[nE] if nE >= 0 else 0
    cdef np.ndarray[int64_t, ndim=1] vptr = np.zeros(nV + 1, dtype=np.int64)
    cdef np.ndarray[int32_t, ndim=1] owner = np.zeros(nnz, dtype=np.int32)
    cdef np.ndarray[np.uint8_t, ndim=1] is_head = np.zeros(nnz, dtype=np.uint8)
    cdef int64_t[::1] vp = vptr
    cdef int32_t[::1] ow = owner
    cdef np.uint8_t[::1] ih = is_head
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
                    cur[bi[p]] = at + 1
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
                    hv[ti, bi[p]] = at + 1
    return vptr, owner, is_head


def channel_diagonals_any_arity(np.ndarray boundary_ptr not None,
                                np.ndarray boundary_idx not None,
                                Py_ssize_t nV,
                                np.ndarray w_E=None,
                                int threads=1,
                                tuple transposed=None):
    """The four diagonals (T, G, F, C) for a complex of any arity, in O(nnz).

    `boundary_ptr`/`boundary_idx` are the CSC support of B1: relation e spans
    ``boundary_idx[boundary_ptr[e]:boundary_ptr[e+1]]``, its first entry the head.
    `w_E` is the per-relation weight, or None for the unweighted tower.

    `threads` sets the parallel width; 1 keeps the serial path. `transposed` accepts a
    previously built `transpose_incidence` result, since the incidence does not change
    between readings of the same complex.

    Returns (T, G, F, C) as float64 arrays of length nE.
    """
    cdef int32_t[::1] bp = np.ascontiguousarray(boundary_ptr, dtype=np.int32)
    cdef int32_t[::1] bi = np.ascontiguousarray(boundary_idx, dtype=np.int32)
    cdef Py_ssize_t nE = bp.shape[0] - 1
    cdef np.ndarray[double, ndim=1] w = (np.ones(nE, dtype=np.float64) if w_E is None
                                         else np.ascontiguousarray(w_E, dtype=np.float64))
    cdef double[::1] wv = w

    cdef np.ndarray[double, ndim=1] T = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] G = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] F = np.zeros(nE, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] C = np.zeros(nE, dtype=np.float64)
    cdef double[::1] Tv = T, Gv = G, Fv = F, Cv = C

    # the mass at each vertex, split by SIGN because F reads the opposite one, and
    # kept twice because C is unweighted where F is not
    cdef np.ndarray[double, ndim=1] negw = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] posw = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] negu = np.zeros(nV, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] posu = np.zeros(nV, dtype=np.float64)
    cdef double[::1] negwv = negw, poswv = posw, neguv = negu, posuv = posu

    cdef Py_ssize_t e, p, s, t, k, v
    cdef double share, mag, we, a, m

    if transposed is None:
        transposed = transpose_incidence(boundary_ptr, boundary_idx, nV, threads)
    cdef int64_t[::1] vp = np.ascontiguousarray(transposed[0], dtype=np.int64)
    cdef int32_t[::1] ow = np.ascontiguousarray(transposed[1], dtype=np.int32)
    cdef np.uint8_t[::1] ih = np.ascontiguousarray(transposed[2], dtype=np.uint8)
    cdef int nthr = threads if threads > 0 else 1

    # pass 1: over VERTICES, so each thread owns what it writes
    cdef np.ndarray[double, ndim=2] mass = np.empty((nV, 4), dtype=np.float64)
    cdef double[:, ::1] mv = mass
    with nogil:
        for v in prange(nV, num_threads=nthr, schedule='static'):
            _vertex_mass(&bp[0], &ow[0], &ih[0], &wv[0], vp[v], vp[v + 1], &mv[v, 0])
    negw[:] = mass[:, 0]; posw[:] = mass[:, 1]
    negu[:] = mass[:, 2]; posu[:] = mass[:, 3]
    with nogil:

        # pass 2: the readings, one per relation
        for e in prange(nE, num_threads=nthr, schedule='static'):
            s = bp[e]; t = bp[e + 1]; k = t - s
            if k == 0:
                continue
            we = wv[e]
            if k == 1:
                Tv[e] = we * we
                Gv[e] = Tv[e]
                v = bi[s]
                Cv[e] = 1.0 * (neguv[v] + posuv[v] - 1.0)      # unweighted
                Fv[e] = 2.0 * we * negwv[v]                    # a witness is positive
                continue
            share = 1.0 / (k - 1)
            Tv[e] = we * we * (1.0 + share)
            Gv[e] = Tv[e]
            # the head: magnitude 1, negative
            v = bi[s]
            Cv[e] += 1.0 * (neguv[v] + posuv[v] - 1.0)
            Fv[e] += we * poswv[v]
            # the shared entries: magnitude 1/(k-1), positive
            mag = we * share
            for p in range(s + 1, t):
                v = bi[p]
                Cv[e] += share * (neguv[v] + posuv[v] - share)
                Fv[e] += mag * negwv[v]
            Fv[e] *= 2.0
    return T, G, F, C
