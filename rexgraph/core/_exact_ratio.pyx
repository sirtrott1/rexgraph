# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._exact_ratio: rational readings over factored denominators.

A reading assembled from a boundary operator is a sum of rationals whose denominators
come from two axes: an incidence count per vertex, and an arity per cell. Their common
multiple grows with the term count and has no fixed width. Dividing by each axis
separately leaves the widest intermediate at one small factor against a numerator, so
128 bits suffices at any term count.

    item value   ( SUM over seeds v of a[i,v]/deg[v] ) / den[i]

`mode` selects what is taken from that per item, and `group` optionally sums the items
into a coarser index:

    SUM       the value as it stands, for a reading with no orientation to cancel
    ABS       its magnitude, for a signed reading read at the item
    COVERAGE  the unsigned total less the magnitude of the signed one, which is what a
              zero-sum column leaves behind when its support is seeded evenly

Accumulation is in fixed point and exact. The division by `den` truncates, and what it
discards is carried as a sticky bit, so the single rounding that produces the double
goes the right way. Under `group` the truncation is per item and accumulates, so
correct rounding there rests on `frac_bits`, which `frac_bits_for` sets from the item
count.
"""

from __future__ import annotations

import numpy as np

cimport cython
cimport numpy as np
from libc.math cimport ldexp
from libc.stdint cimport int64_t, uint64_t
from libc.stdlib cimport calloc, free

np.import_array()

cdef extern from *:
    ctypedef unsigned long long u128 "unsigned __int128"
    ctypedef long long i128 "__int128"


cdef extern from *:
    """
    static inline int _rex_clz64(unsigned long long x) { return __builtin_clzll(x); }
    """
    int _rex_clz64(uint64_t x) nogil


cdef enum Mode:
    _SUM = 0
    _ABS = 1
    _COVERAGE = 2

#: what to take from each item's value
SUM = <int>_SUM
ABS = <int>_ABS
COVERAGE = <int>_COVERAGE


cdef inline int _bits(u128 v) noexcept nogil:
    """Position of the highest set bit."""
    cdef uint64_t hi = <uint64_t>(v >> 64)
    cdef uint64_t lo = <uint64_t>v
    if hi:
        return 128 - _rex_clz64(hi)
    if lo:
        return 64 - _rex_clz64(lo)
    return 0


cdef inline double _round(u128 q, int shift, bint sticky) noexcept nogil:
    """`q * 2**shift` as the nearest double, ties to even.

    Keeps 54 bits, 53 for the mantissa and one to round on. Anything below them folds
    into `sticky`, which separates a tie from a value whose leading bits only resemble
    one.
    """
    cdef int b
    cdef int drop
    cdef uint64_t mant
    if q == 0:
        return 0.0
    b = _bits(q)
    if b > 54:
        drop = b - 54
        if (q & (((<u128>1) << drop) - 1)) != 0:
            sticky = True
        q >>= drop
        shift += drop
        b = 54
    mant = <uint64_t>q
    if b == 54:
        if (mant & 1) and (sticky or (mant & 2)):
            mant += 2
        mant >>= 1
        shift += 1
    return ldexp(<double>mant, shift)


cdef inline i128 _divide(i128 value, int64_t d, bint *sticky) noexcept nogil:
    """`value/d`, recording in `sticky` whether anything was discarded."""
    cdef i128 dd
    cdef i128 q
    if d <= 1:
        return value
    dd = <i128>d
    q = value / dd
    if q * dd != value:
        sticky[0] = True
    return q


def axis_ratio(const int64_t[::1] item,
               const int64_t[::1] carried,
               const int64_t[::1] seed,
               const int64_t[::1] deg,
               const int64_t[::1] den,
               Py_ssize_t n_items,
               int frac_bits,
               const int64_t[::1] group=None,
               Py_ssize_t n_groups=0,
               int mode=_SUM):
    """The reading per item, or per group when `group` maps items into one.

    `carried[k]` lands on item `item[k]` under seed `seed[k]` and may be negative: a
    boundary entry at position 0 carries the opposite sign to the arguments, and
    `COVERAGE` is the reading that measures exactly that disagreement.
    """
    cdef Py_ssize_t m = item.shape[0]
    cdef Py_ssize_t s = deg.shape[0]
    cdef Py_ssize_t i, v, out_n
    cdef int64_t it, g
    cdef bint grouped = group is not None
    cdef i128 *signed_a = <i128 *>calloc(n_items * s, sizeof(i128))
    cdef i128 *unsigned_a = NULL
    cdef i128 *acc = NULL
    cdef i128 total, mag, term, scaled
    cdef bint sticky, any_sticky
    cdef np.ndarray[np.float64_t, ndim=1] out
    cdef double[::1] o

    out_n = n_groups if grouped else n_items
    out = np.zeros(max(out_n, 0), dtype=np.float64)
    o = out
    if signed_a == NULL:
        raise MemoryError("axis accumulator")
    if mode == _COVERAGE:
        unsigned_a = <i128 *>calloc(n_items * s, sizeof(i128))
        if unsigned_a == NULL:
            free(signed_a)
            raise MemoryError("axis accumulator")
    if grouped:
        acc = <i128 *>calloc(max(out_n, 1), sizeof(i128))
        if acc == NULL:
            free(signed_a)
            if unsigned_a != NULL:
                free(unsigned_a)
            raise MemoryError("group accumulator")
    any_sticky = False
    try:
        with nogil:
            for i in range(m):
                it = item[i]
                if it < 0 or it >= n_items:
                    continue
                signed_a[it * s + seed[i]] += <i128>carried[i]
                if mode == _COVERAGE:
                    unsigned_a[it * s + seed[i]] += <i128>(
                        carried[i] if carried[i] >= 0 else -carried[i])
            for i in range(n_items):
                total = 0
                mag = 0
                sticky = False
                for v in range(s):
                    term = signed_a[i * s + v]
                    if term != 0:
                        scaled = term << frac_bits
                        total += _divide(scaled, deg[v], &sticky)
                    if mode == _COVERAGE:
                        term = unsigned_a[i * s + v]
                        if term != 0:
                            scaled = term << frac_bits
                            mag += _divide(scaled, deg[v], &sticky)
                if total == 0 and mag == 0:
                    continue
                total = _divide(total, den[i], &sticky)
                if mode == _ABS:
                    if total < 0:
                        total = -total
                elif mode == _COVERAGE:
                    mag = _divide(mag, den[i], &sticky)
                    if total < 0:
                        total = -total
                    total = mag - total
                if grouped:
                    g = group[i]
                    if 0 <= g < out_n:
                        acc[g] += total
                        if sticky:
                            any_sticky = True
                else:
                    if total > 0:
                        o[i] = _round(<u128>total, -frac_bits, sticky)
                    elif total < 0:
                        o[i] = -_round(<u128>(-total), -frac_bits, sticky)
            if grouped:
                for i in range(out_n):
                    if acc[i] > 0:
                        o[i] = _round(<u128>acc[i], -frac_bits, any_sticky)
                    elif acc[i] < 0:
                        o[i] = -_round(<u128>(-acc[i]), -frac_bits, any_sticky)
    finally:
        free(signed_a)
        if unsigned_a != NULL:
            free(unsigned_a)
        if acc != NULL:
            free(acc)
    return out


def frac_bits_for(Py_ssize_t widest_carried, Py_ssize_t n_seeds, Py_ssize_t n_items=1):
    """How far to scale each term so the accumulation stays inside 128 bits.

    A term is at most `carried << frac_bits`, there are `n_seeds` of them per item, and
    with a group they sum over `n_items`. What is left over those three is what can be
    spent on fractional bits.
    """
    cdef int used = 1                      # one bit of headroom for the sign
    cdef Py_ssize_t w = widest_carried if widest_carried > 0 else 1
    cdef Py_ssize_t c = n_seeds if n_seeds > 0 else 1
    cdef Py_ssize_t g = n_items if n_items > 0 else 1
    while w:
        w >>= 1
        used += 1
    while c:
        c >>= 1
        used += 1
    while g:
        g >>= 1
        used += 1
    return max(126 - used, 0)
