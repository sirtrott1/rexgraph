# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._ternary: a {-1, 0, +1} operator carried as two bitplanes.

A ternary entry is two bits of information and nothing more, so storing it as a float
spends 32 bits to carry 2. On an operator large enough to leave cache that is not a
storage question but a bandwidth one: the same product moves 16x fewer bytes, and both
a CPU and a GPU run out of bandwidth long before they run out of instructions.

The two planes are what the trichotomy already names elsewhere in the library:

    presence   bit set where the entry is nonzero      the EXISTENCE of the incidence
    sign       bit set where that entry is negative    its ORIENTATION

Share is deliberately absent. It is a function of the row's arity, which is the
popcount of the presence plane, so a boundary column's 1/(k-1) is derived rather than
stored, the same way boundary_ptr/boundary_idx derive it.

Against a +-1 vector the product needs no multiplier. Writing that vector as one
bitplane X, an entry agrees where sign == X and disagrees where they differ, so per 64
entries a row contributes

    popcount(P & ~(S ^ X)) - popcount(P & (S ^ X))

which is two ANDs, an XOR and two popcounts for 64 multiply-accumulates. That path is
exact: the result is a count difference, an integer, with no rounding anywhere.

Against a general float vector the planes still earn their keep by carrying the
support, so the loop visits set bits only and never touches a zero. That path is
ordinary float arithmetic and is exact only as far as float addition is.
"""

from __future__ import annotations

cimport cython
import numpy as np

cimport numpy as np
from cython.parallel cimport prange
from libc.stdint cimport int8_t, int64_t, uint64_t

np.import_array()

cdef extern from *:
    """
    /* Bit counting, resolved at RUNTIME rather than at build time.
     *
     * The build cannot assume the instruction: a portable wheel is compiled for a
     * baseline target (conda-forge hands us -march=nocona, a 2004 part), and there
     * __builtin_popcountll becomes a CALL to libgcc's software __popcountdi2, once
     * per 64-bit word. That is what this kernel was doing.
     *
     * So the ISA is chosen when the process starts, not when the wheel is built:
     * AVX-512 VPOPCNTDQ counts eight words per instruction, plain POPCNT counts one,
     * and the portable fallback keeps a machine with neither working. Nothing here
     * is a compile-time flag, so the same binary is correct everywhere and fast
     * where the silicon allows.
     */
    #include <stdint.h>
    #include <stddef.h>

    static int64_t _dis_generic(const uint64_t* p, const uint64_t* s,
                                const uint64_t* x, size_t nw) {
        int64_t a = 0;
        for (size_t w = 0; w < nw; ++w) a += __builtin_popcountll(p[w] & (s[w] ^ x[w]));
        return a;
    }

    #if defined(__x86_64__) || defined(__i386__)
    #include <immintrin.h>

    __attribute__((target("popcnt")))
    static int64_t _dis_popcnt(const uint64_t* p, const uint64_t* s,
                               const uint64_t* x, size_t nw) {
        int64_t a = 0;
        for (size_t w = 0; w < nw; ++w) a += __builtin_popcountll(p[w] & (s[w] ^ x[w]));
        return a;
    }

    __attribute__((target("avx512f,avx512vpopcntdq,popcnt")))
    static int64_t _dis_avx512(const uint64_t* p, const uint64_t* s,
                               const uint64_t* x, size_t nw) {
        __m512i acc = _mm512_setzero_si512();
        size_t w = 0;
        for (; w + 8 <= nw; w += 8) {
            __m512i pv = _mm512_loadu_si512((const void*)(p + w));
            __m512i sv = _mm512_loadu_si512((const void*)(s + w));
            __m512i xv = _mm512_loadu_si512((const void*)(x + w));
            __m512i d  = _mm512_and_si512(pv, _mm512_xor_si512(sv, xv));
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(d));
        }
        int64_t a = _mm512_reduce_add_epi64(acc);
        for (; w < nw; ++w) a += __builtin_popcountll(p[w] & (s[w] ^ x[w]));
        return a;
    }
    #endif

    typedef int64_t (*_dis_fn)(const uint64_t*, const uint64_t*, const uint64_t*, size_t);
    static _dis_fn _DIS = 0;
    static const char* _DIS_NAME = "generic";

    static void _dis_resolve(void) {
        _DIS = _dis_generic; _DIS_NAME = "generic";
    #if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx512vpopcntdq")) {
            _DIS = _dis_avx512; _DIS_NAME = "avx512-vpopcntdq";
        } else if (__builtin_cpu_supports("popcnt")) {
            _DIS = _dis_popcnt; _DIS_NAME = "popcnt";
        }
    #endif
    }

    static inline int64_t _disagree(const uint64_t* p, const uint64_t* s,
                                    const uint64_t* x, size_t nw) {
        return _DIS(p, s, x, nw);
    }

    static inline int _ctz(uint64_t b) { return __builtin_ctzll(b); }
    static inline int _pc1(uint64_t b) { return __builtin_popcountll(b); }

    /* The float path, against a general vector.
     *
     * The obvious loop walks set bits and gathers v[base+b] one at a time. That reads
     * only the support, which sounds like the efficient choice and is not: it is
     * scalar, every bit is an unpredictable branch, and it reached 3.6 GB/s against a
     * memory system that does 116.
     *
     * The bits of a word address SIXTY-FOUR CONSECUTIVE entries of v, so nothing needs
     * gathering. Eight doubles load contiguously, the presence byte becomes a mask,
     * and the sign byte splits it into one masked add and one masked subtract. The
     * whole support of a word is covered by eight of those with no branch on any
     * individual bit.
     *
     * v must be padded to nw*64 doubles so the last word can load a full vector; the
     * caller does that, and the padding is zero so it contributes nothing.
     *
     * Summation order differs from the scalar loop, so the two agree to float rounding
     * rather than bit for bit. Float addition is not associative and no arrangement of
     * it is canonical.
     */
    static double _rowf_generic(const uint64_t* p, const uint64_t* s,
                                const double* v, size_t nw) {
        double acc = 0.0;
        for (size_t w = 0; w < nw; ++w) {
            uint64_t pres = p[w];
            if (!pres) continue;
            uint64_t sgn = s[w];
            const double* base = v + (w << 6);
            while (pres) {
                uint64_t bit = pres & (~pres + 1);
                int b = __builtin_ctzll(bit);
                acc += ((sgn >> b) & 1) ? -base[b] : base[b];
                pres ^= bit;
            }
        }
        return acc;
    }

    #if defined(__x86_64__) || defined(__i386__)
    __attribute__((target("avx512f")))
    static double _rowf_avx512(const uint64_t* p, const uint64_t* s,
                               const double* v, size_t nw) {
        __m512d acc = _mm512_setzero_pd();
        for (size_t w = 0; w < nw; ++w) {
            uint64_t pres = p[w];
            if (!pres) continue;
            uint64_t sgn = s[w];
            const double* base = v + (w << 6);
            for (int g = 0; g < 8; ++g) {
                __mmask8 mp = (__mmask8)((pres >> (g * 8)) & 0xFF);
                if (!mp) continue;
                __mmask8 ms = (__mmask8)((sgn >> (g * 8)) & 0xFF);
                __m512d vv = _mm512_loadu_pd(base + g * 8);
                acc = _mm512_mask_add_pd(acc, (__mmask8)(mp & ~ms), acc, vv);
                acc = _mm512_mask_sub_pd(acc, (__mmask8)(mp & ms), acc, vv);
            }
        }
        return _mm512_reduce_add_pd(acc);
    }
    #endif

    /* Row blocking. The loop above re-reads the WHOLE of v for every row: 4.3 GB of
     * cache traffic against 134 MB of planes on a 8192x65536 operator, so v and not
     * the operator is what the machine is actually moving. Four rows share one load of
     * v, which cuts that traffic fourfold and costs four accumulator registers.
     */
    #if defined(__x86_64__) || defined(__i386__)
    __attribute__((target("avx512f")))
    static void _blockf_avx512(const uint64_t* P, const uint64_t* S, size_t stride,
                               int nrow, const double* v, size_t nw, double* out) {
        __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd();
        __m512d a2 = _mm512_setzero_pd(), a3 = _mm512_setzero_pd();
        const uint64_t *p0 = P, *p1 = P + stride, *p2 = P + 2*stride, *p3 = P + 3*stride;
        const uint64_t *s0 = S, *s1 = S + stride, *s2 = S + 2*stride, *s3 = S + 3*stride;
        for (size_t w = 0; w < nw; ++w) {
            uint64_t r0 = p0[w], r1 = nrow > 1 ? p1[w] : 0;
            uint64_t r2 = nrow > 2 ? p2[w] : 0, r3 = nrow > 3 ? p3[w] : 0;
            if (!(r0 | r1 | r2 | r3)) continue;
            uint64_t g0 = s0[w], g1 = nrow > 1 ? s1[w] : 0;
            uint64_t g2 = nrow > 2 ? s2[w] : 0, g3 = nrow > 3 ? s3[w] : 0;
            const double* base = v + (w << 6);
            for (int g = 0; g < 8; ++g) {
                __mmask8 m0 = (__mmask8)((r0 >> (g*8)) & 0xFF);
                __mmask8 m1 = (__mmask8)((r1 >> (g*8)) & 0xFF);
                __mmask8 m2 = (__mmask8)((r2 >> (g*8)) & 0xFF);
                __mmask8 m3 = (__mmask8)((r3 >> (g*8)) & 0xFF);
                if (!(m0 | m1 | m2 | m3)) continue;
                __m512d vv = _mm512_loadu_pd(base + g*8);      /* loaded ONCE for four */
                __mmask8 k;
                k = (__mmask8)((g0 >> (g*8)) & 0xFF);
                a0 = _mm512_mask_add_pd(a0, (__mmask8)(m0 & ~k), a0, vv);
                a0 = _mm512_mask_sub_pd(a0, (__mmask8)(m0 &  k), a0, vv);
                k = (__mmask8)((g1 >> (g*8)) & 0xFF);
                a1 = _mm512_mask_add_pd(a1, (__mmask8)(m1 & ~k), a1, vv);
                a1 = _mm512_mask_sub_pd(a1, (__mmask8)(m1 &  k), a1, vv);
                k = (__mmask8)((g2 >> (g*8)) & 0xFF);
                a2 = _mm512_mask_add_pd(a2, (__mmask8)(m2 & ~k), a2, vv);
                a2 = _mm512_mask_sub_pd(a2, (__mmask8)(m2 &  k), a2, vv);
                k = (__mmask8)((g3 >> (g*8)) & 0xFF);
                a3 = _mm512_mask_add_pd(a3, (__mmask8)(m3 & ~k), a3, vv);
                a3 = _mm512_mask_sub_pd(a3, (__mmask8)(m3 &  k), a3, vv);
            }
        }
        out[0] = _mm512_reduce_add_pd(a0);
        if (nrow > 1) out[1] = _mm512_reduce_add_pd(a1);
        if (nrow > 2) out[2] = _mm512_reduce_add_pd(a2);
        if (nrow > 3) out[3] = _mm512_reduce_add_pd(a3);
    }
    #endif

    static void _blockf_generic(const uint64_t* P, const uint64_t* S, size_t stride,
                                int nrow, const double* v, size_t nw, double* out) {
        for (int r = 0; r < nrow; ++r)
            out[r] = _rowf_generic(P + (size_t)r*stride, S + (size_t)r*stride, v, nw);
    }

    typedef void (*_blockf_fn)(const uint64_t*, const uint64_t*, size_t, int,
                               const double*, size_t, double*);
    static _blockf_fn _BLOCKF = 0;

    static void _blockf_resolve(void) {
        _BLOCKF = _blockf_generic;
    #if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx512f")) _BLOCKF = _blockf_avx512;
    #endif
    }

    static inline void _blockf(const uint64_t* P, const uint64_t* S, size_t stride,
                               int nrow, const double* v, size_t nw, double* out) {
        _BLOCKF(P, S, stride, nrow, v, nw, out);
    }

    typedef double (*_rowf_fn)(const uint64_t*, const uint64_t*, const double*, size_t);
    static _rowf_fn _ROWF = 0;
    static const char* _ROWF_NAME = "generic";

    static void _rowf_resolve(void) {
        _ROWF = _rowf_generic; _ROWF_NAME = "generic";
    #if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx512f")) {
            _ROWF = _rowf_avx512; _ROWF_NAME = "avx512-masked";
        }
    #endif
    }

    static inline double _rowf(const uint64_t* p, const uint64_t* s,
                               const double* v, size_t nw) {
        return _ROWF(p, s, v, nw);
    }
    """
    int64_t _disagree(const uint64_t*, const uint64_t*, const uint64_t*, size_t) noexcept nogil
    void _dis_resolve() noexcept nogil
    int _ctz(uint64_t) noexcept nogil
    int _pc1(uint64_t) noexcept nogil
    double _rowf(const uint64_t*, const uint64_t*, const double*, size_t) noexcept nogil
    void _rowf_resolve() noexcept nogil
    void _blockf_resolve() noexcept nogil
    void _blockf(const uint64_t*, const uint64_t*, size_t, int,
                 const double*, size_t, double*) noexcept nogil
    const char* _ROWF_NAME
    const char* _DIS_NAME


_dis_resolve()
_rowf_resolve()
_blockf_resolve()


def bitcount_path() -> str:
    """Which bit-counting implementation this process resolved to."""
    return (<bytes>_DIS_NAME).decode()


def float_path() -> str:
    """Which float-product implementation this process resolved to."""
    return (<bytes>_ROWF_NAME).decode()



cdef inline int64_t _row_pm1(const uint64_t* p, const uint64_t* s,
                             const uint64_t* x, Py_ssize_t nw,
                             int64_t k) noexcept nogil:
    """One row against a packed +-1 vector.

    Agreements less disagreements, but counted once rather than twice: the two
    popcounts sum to the row's arity, so

        agree - disagree = k - 2*disagree

    and k is already known. That halves both the ANDs and the popcounts, and the
    same identity is what the device lane uses.
    """
    return k - 2 * _disagree(p, s, x, nw)


def pack(np.ndarray arr not None):
    """Pack a 2-D {-1,0,1} array into (presence, sign) uint64 bitplanes, row major.

    Returns (P, S, ncols). Entry (i, j) lives in bit j % 64 of word j // 64 of row i.
    Bits past ncols stay clear, which is what lets the +-1 path skip a tail mask.
    """
    cdef np.ndarray[int8_t, ndim=2] a = np.ascontiguousarray(arr, dtype=np.int8)
    cdef Py_ssize_t nr = a.shape[0], nc = a.shape[1]
    cdef Py_ssize_t nw = (nc + 63) // 64
    cdef np.ndarray[uint64_t, ndim=2] P = np.zeros((nr, nw), dtype=np.uint64)
    cdef np.ndarray[uint64_t, ndim=2] S = np.zeros((nr, nw), dtype=np.uint64)
    cdef Py_ssize_t i, j
    cdef int8_t v
    for i in range(nr):
        for j in range(nc):
            v = a[i, j]
            if v != 0:
                P[i, j >> 6] |= (<uint64_t>1) << (j & 63)
                if v < 0:
                    S[i, j >> 6] |= (<uint64_t>1) << (j & 63)
    return P, S, nc


def unpack(np.ndarray P not None, np.ndarray S not None, Py_ssize_t ncols):
    """Rebuild the dense {-1,0,1} array. The inverse of pack, for verification."""
    cdef np.ndarray[uint64_t, ndim=2] p = np.ascontiguousarray(P)
    cdef np.ndarray[uint64_t, ndim=2] s = np.ascontiguousarray(S)
    cdef Py_ssize_t nr = p.shape[0], i, j
    cdef np.ndarray[int8_t, ndim=2] out = np.zeros((nr, ncols), dtype=np.int8)
    for i in range(nr):
        for j in range(ncols):
            if (p[i, j >> 6] >> (j & 63)) & 1:
                out[i, j] = -1 if ((s[i, j >> 6] >> (j & 63)) & 1) else 1
    return out


def pack_vector(np.ndarray vec not None):
    """Pack a +-1 vector into one sign bitplane, the form the popcount path wants."""
    cdef np.ndarray[int8_t, ndim=1] v = np.ascontiguousarray(vec, dtype=np.int8)
    cdef Py_ssize_t n = v.shape[0], nw = (n + 63) // 64, j
    cdef np.ndarray[uint64_t, ndim=1] X = np.zeros(nw, dtype=np.uint64)
    for j in range(n):
        if v[j] < 0:
            X[j >> 6] |= (<uint64_t>1) << (j & 63)
    return X


def matvec_pm1(np.ndarray P not None, np.ndarray S not None,
               np.ndarray X not None, np.ndarray K=None, int threads=0):
    """Exact integer product of a ternary operator with a packed +-1 vector.

    `K` is the per-row arity. Pass it when it is already known, which it usually is:
    it is a property of the operator and not of the vector, so recomputing it per
    matvec is work the caller has already done once.
    """
    cdef uint64_t[:, ::1] pv = np.ascontiguousarray(P)
    cdef uint64_t[:, ::1] sv = np.ascontiguousarray(S)
    cdef uint64_t[::1] xv = np.ascontiguousarray(X)
    cdef Py_ssize_t nr = pv.shape[0], nw = pv.shape[1], i
    cdef np.ndarray[int64_t, ndim=1] karr = arity(P) if K is None else \
        np.ascontiguousarray(K, dtype=np.int64)
    cdef int64_t[::1] kv = karr
    cdef np.ndarray[int64_t, ndim=1] out = np.empty(nr, dtype=np.int64)
    cdef int64_t[::1] ov = out
    cdef int nthr = threads if threads > 0 else 1
    with nogil:
        for i in prange(nr, num_threads=nthr, schedule='static'):
            ov[i] = _row_pm1(&pv[i, 0], &sv[i, 0], &xv[0], nw, kv[i])
    return out


def matvec_f64(np.ndarray P not None, np.ndarray S not None,
               np.ndarray vec not None, int threads=0):
    """Product with a general float vector.

    `vec` is padded to a whole number of words so the vectorised path can load eight
    doubles at the end of the last one. The padding is zero and contributes nothing.
    """
    cdef uint64_t[:, ::1] pv = np.ascontiguousarray(P)
    cdef uint64_t[:, ::1] sv = np.ascontiguousarray(S)
    cdef Py_ssize_t nr = pv.shape[0], nw = pv.shape[1], i
    cdef np.ndarray[double, ndim=1] padded = np.zeros(nw * 64, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] src = np.ascontiguousarray(vec, dtype=np.float64)
    padded[:src.shape[0]] = src
    cdef double[::1] vv = padded
    cdef np.ndarray[double, ndim=1] out = np.empty(nr, dtype=np.float64)
    cdef double[::1] ov = out
    cdef int nthr = threads if threads > 0 else 1
    cdef Py_ssize_t nblk = (nr + 3) // 4
    cdef Py_ssize_t r0, cnt
    with nogil:
        for i in prange(nblk, num_threads=nthr, schedule='static'):
            r0 = i * 4
            cnt = 4 if r0 + 4 <= nr else nr - r0
            _blockf(&pv[r0, 0], &sv[r0, 0], nw, <int>cnt, &vv[0], nw, &ov[r0])
    return out


def arity(np.ndarray P not None):
    """Row popcount: each relation's arity, which is what its share derives from."""
    cdef uint64_t[:, ::1] pv = np.ascontiguousarray(P)
    cdef Py_ssize_t nr = pv.shape[0], nw = pv.shape[1], i, w
    cdef np.ndarray[int64_t, ndim=1] out = np.zeros(nr, dtype=np.int64)
    cdef int64_t[::1] ov = out
    cdef int64_t acc
    for i in range(nr):
        acc = 0
        for w in range(nw):
            acc += _pc1(pv[i, w])
        ov[i] = acc
    return out
