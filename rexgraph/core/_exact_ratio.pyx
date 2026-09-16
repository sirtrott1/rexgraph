# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Sparse rational readings over factored denominators.

Supplied coordinates accumulate as checked integer ratios per item. Only rows
whose intermediate integers exceed the machine bounds use Fraction arithmetic.
Results round once in numerical mode. No item by seed grid, fixed precision
cancellation threshold or dense matrix is allocated.
"""
from fractions import Fraction
from numbers import Integral

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, INT64_MAX, INT64_MIN
from libc.stdlib cimport calloc, free
from ._common cimport compute_parallel_buffer_memory

np.import_array()

SUM = 0
ABS = 1
COVERAGE = 2


cdef inline int64_t _gcd(int64_t a, int64_t b) noexcept nogil:
    cdef int64_t r
    while b:
        r = a % b
        a, b = b, r
    return a


cdef inline bint _multiply(int64_t a, int64_t b, int64_t *out) noexcept nogil:
    # b is positive. C division truncates toward zero, including INT64_MIN / b.
    if b == 1 or a == 0:
        out[0] = a
        return True
    if (a > 0 and a > INT64_MAX / b) or (a < 0 and a < INT64_MIN / b):
        return False
    out[0] = a * b
    return True


cdef inline bint _add(int64_t a, int64_t b, int64_t *out) noexcept nogil:
    if (b > 0 and a > INT64_MAX - b) or (b < 0 and a < INT64_MIN - b):
        return False
    out[0] = a + b
    return True


cdef inline bint _accumulate(int64_t *row, int64_t n, int64_t d,
                             bint coverage) noexcept nogil:
    """Commit an exact ratio addition only after all intermediate bounds pass."""
    cdef int64_t left, right, common, divisor, a, b, total, mass = 0
    if row[2] == 0:
        return False
    if n == 0:
        return True
    if coverage and n == INT64_MIN:
        return False
    if row[0] == 0 and row[1] == 0:
        row[0], row[1], row[2] = n, (-n if n < 0 else n) if coverage else 0, d
        return True
    if row[2] == d:
        common, left, right = d, 1, 1
    else:
        divisor = _gcd(row[2], d)
        left, right = d / divisor, row[2] / divisor
        if not _multiply(row[2], left, &common):
            return False
    if not _multiply(row[0], left, &a) or not _multiply(n, right, &b):
        return False
    if not _add(a, b, &total):
        return False
    if coverage:
        if not _multiply(row[1], left, &a):
            return False
        if not _multiply(-n if n < 0 else n, right, &b):
            return False
        if not _add(a, b, &mass):
            return False
    row[0], row[1], row[2] = total, mass, common
    return True


cdef void _promote_add(int64_t *row, dict wide, Py_ssize_t k,
                      object n, object d, bint coverage) except *:
    """Promote only the row that cannot complete a bounded addition."""
    if row[2]:
        wide[k] = (Fraction(int(row[0]), int(row[2])),
                   Fraction(int(row[1]), int(row[2])) if coverage else 0)
        row[2] = 0
    value, mass = wide[k]
    term = Fraction(n, d)
    wide[k] = (value + term, mass + abs(term) if coverage else 0)


cdef inline double _float_ratio(int64_t n, int64_t d) except? -1:
    # Exact operands make hardware division one correctly rounded operation.
    # Beyond 53 bits Python integer true division supplies the correct rounding.
    cdef int64_t exact_limit = (<int64_t>1) << 53
    if -exact_limit <= n <= exact_limit and d <= exact_limit:
        return <double>n / <double>d
    return (<object>n) / (<object>d)


def axis_ratio(const int64_t[::1] item,
               const int64_t[::1] carried,
               const int64_t[::1] seed,
               const int64_t[::1] deg,
               const int64_t[::1] den,
               Py_ssize_t n_items,
               int frac_bits,
               const int64_t[::1] group=None,
               Py_ssize_t n_groups=0,
               int mode=0, *, exact=False):
    """Read sum, absolute sum or coverage per item or declared group.

    Coverage is sum(abs(carried)/deg) minus abs(sum(carried/deg)), then
    divided by the item denominator. Callers supplying boundary entries
    must coalesce their primary column before taking its absolute values.
    Group -1 omits an item. Invalid coordinates and denominators are refused.

    frac_bits remains an accepted compatibility hint in [0,126]. It does
    not limit exact arithmetic. exact=True returns Fractions; otherwise
    each finished rational is converted once to its nearest float.
    """
    cdef Py_ssize_t i, m = item.shape[0], out_n
    cdef int64_t it, v, g, n, d, common = 1, magnitude = 0, bound = 0
    cdef bint coverage = mode == COVERAGE
    cdef bint absolute = mode == ABS
    cdef bint retain_exact
    cdef object value, mass, out
    cdef dict wide = {}, grouped = {}
    cdef int64_t *rows = NULL
    cdef int64_t *groups = NULL
    cdef double[::1] numerical
    if n_items < 0 or n_groups < 0:
        raise ValueError("axis sizes must be nonnegative")
    if carried.shape[0] != m or seed.shape[0] != m or den.shape[0] != n_items:
        raise ValueError("axis arrays have incompatible lengths")
    if mode not in (SUM, ABS, COVERAGE) or not 0 <= frac_bits <= 126:
        raise ValueError("unknown ratio mode or invalid precision hint")
    if not isinstance(exact, (bool, np.bool_)):
        raise TypeError("exact must be boolean")
    retain_exact = exact
    if group is not None and group.shape[0] != n_items:
        raise ValueError("group must have one coordinate per item")
    for i in range(deg.shape[0]):
        if deg[i] <= 0:
            raise ValueError("seed denominators must be positive")
        if common and not _multiply(common, deg[i] / _gcd(common, deg[i]), &common):
            common = 0
    for i in range(n_items):
        if den[i] <= 0:
            raise ValueError("item denominators must be positive")
        if group is not None and (group[i] < -1 or group[i] >= n_groups):
            raise ValueError("group coordinate is outside its declared axis")
    for i in range(m):
        it, v = item[i], seed[i]
        if it < 0 or it >= n_items or v < 0 or v >= deg.shape[0]:
            raise ValueError("ratio coordinate is outside its declared axis")
        if common:
            if carried[i] == INT64_MIN:
                common = 0
            else:
                magnitude = -carried[i] if carried[i] < 0 else carried[i]
                if not _add(bound, magnitude, &bound):
                    common = 0
    if common and bound > INT64_MAX / common:
        common = 0

    if (compute_parallel_buffer_memory(1, n_items, 3*sizeof(int64_t)) < 0
            or (group is not None and compute_parallel_buffer_memory(1, n_groups, 3*sizeof(int64_t)) < 0)):
        raise MemoryError("ratio accumulator size exceeds the addressable range")
    rows = <int64_t*>calloc(n_items, 3*sizeof(int64_t))
    if rows == NULL and n_items:
        raise MemoryError("cannot allocate ratio items")
    try:
        if common:
            # sum(abs(carried)) * common bounds every partial numerator and mass.
            # Every deg divides common. There is no truncation in this factoring.
            for i in range(n_items):
                rows[3*i+2] = common
            for i in range(m):
                it, v = item[i], seed[i]
                n = carried[i] * (common / deg[v])
                rows[3*it] += n
                if coverage:
                    rows[3*it+1] += -n if n < 0 else n
        else:
            for i in range(n_items):
                rows[3*i+2] = 1
            for i in range(m):
                it, v = item[i], seed[i]
                if not _accumulate(&rows[3*it], carried[i], deg[v], coverage):
                    _promote_add(&rows[3*it], wide, it, int(carried[i]), int(deg[v]), coverage)

        out_n = n_items if group is None else n_groups
        out = np.full(out_n, Fraction(0), dtype=object) if retain_exact else np.zeros(out_n, dtype=np.float64)
        if not retain_exact:
            numerical = out
        if group is not None:
            groups = <int64_t*>calloc(n_groups, 3*sizeof(int64_t))
            if groups == NULL and n_groups:
                raise MemoryError("cannot allocate ratio groups")
            for i in range(n_groups):
                groups[3*i+2] = 1
        for it in range(n_items):
            g = it if group is None else group[it]
            if g < 0:
                continue
            value = None
            if rows[3*it+2] == 0:
                value, mass = wide[it]
                value = (mass - abs(value) if coverage else abs(value) if absolute else value) / int(den[it])
            else:
                n = rows[3*it]
                if coverage:
                    n = rows[3*it+1] - (-n if n < 0 else n)
                elif absolute and n < 0:
                    if n == INT64_MIN:
                        value = Fraction(-int(n), int(rows[3*it+2]) * int(den[it]))
                    else:
                        n = -n
                if n == 0:
                    continue
                if value is None and not _multiply(rows[3*it+2], den[it], &d):
                    value = Fraction(int(n), int(rows[3*it+2]) * int(den[it]))
            if value is not None:
                if group is None:
                    if retain_exact:
                        out[g] = value
                    else:
                        numerical[g] = float(value)
                else:
                    _promote_add(&groups[3*g], grouped, g, value.numerator, value.denominator, False)
            elif group is not None:
                if not _accumulate(&groups[3*g], n, d, False):
                    _promote_add(&groups[3*g], grouped, g, int(n), int(d), False)
            elif n:
                if retain_exact:
                    out[g] = Fraction(int(n), int(d))
                else:
                    numerical[g] = _float_ratio(n, d)
        if group is not None:
            for g in range(n_groups):
                if groups[3*g+2] == 0:
                    value = grouped[g][0]
                    if retain_exact:
                        out[g] = value
                    else:
                        numerical[g] = float(value)
                elif groups[3*g]:
                    n, d = groups[3*g], groups[3*g+2]
                    if retain_exact:
                        out[g] = Fraction(int(n), int(d))
                    else:
                        numerical[g] = _float_ratio(n, d)
        return out
    finally:
        free(groups)
        free(rows)


def frac_bits_for(widest_carried, n_seeds, n_items=1):
    """Legacy precision hint; axis_ratio now carries exact rationals regardless."""
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0
           for v in (widest_carried, n_seeds, n_items)):
        raise ValueError("precision dimensions must be nonnegative integers")
    return max(125 - sum(max(int(v), 1).bit_length() for v in
                         (widest_carried, n_seeds, n_items)), 0)
