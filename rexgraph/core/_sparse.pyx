# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._sparse - Sparse matrix storage and operations
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f32, f64, idx_t,

    MAX_INT32_NNZ,

    can_allocate_dense_f64,
    get_max_dense_allocation_bytes,

    get_EPSILON_NORM,
)

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.math cimport fabs

from cython.parallel cimport prange

np.import_array()

# Sorting constants
cdef Py_ssize_t _ISORT_CUTOFF = 16
cdef enum:
    _QS_STACK = 128


# Sorting
# Iterative quicksort with median-of-3 pivot and insertion sort fallback.

# Swap helpers
cdef inline void _sw32(i32* a, i32* b) noexcept nogil:
    cdef i32 t = a[0]
    a[0] = b[0]
    b[0] = t
cdef inline void _sw64(i64* a, i64* b) noexcept nogil:
    cdef i64 t = a[0]
    a[0] = b[0]
    b[0] = t
cdef inline void _swf32(f32* a, f32* b) noexcept nogil:
    cdef f32 t = a[0]
    a[0] = b[0]
    b[0] = t
cdef inline void _swf64(f64* a, f64* b) noexcept nogil:
    cdef f64 t = a[0]
    a[0] = b[0]
    b[0] = t

# Insertion sort: key-only
cdef inline void _isort_i32(i32* a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef i32 key
    for i in range(1, n):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j+1] = a[j]
            j -= 1
        a[j+1] = key

cdef inline void _isort_i64(i64* a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef i64 key
    for i in range(1, n):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j+1] = a[j]
            j -= 1
        a[j+1] = key

# Insertion sort: paired key-value
cdef inline void _isort_kv_i32_f64(i32* k, f64* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef i32 kk
    cdef f64 vv
    for i in range(1, n):
        kk = k[i]
        vv = v[i]
        j = i - 1
        while j >= 0 and k[j] > kk:
            k[j+1] = k[j]
            v[j+1] = v[j]
            j -= 1
        k[j+1] = kk
        v[j+1] = vv

cdef inline void _isort_kv_i64_f64(i64* k, f64* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef i64 kk
    cdef f64 vv
    for i in range(1, n):
        kk = k[i]
        vv = v[i]
        j = i - 1
        while j >= 0 and k[j] > kk:
            k[j+1] = k[j]
            v[j+1] = v[j]
            j -= 1
        k[j+1] = kk
        v[j+1] = vv

cdef inline void _isort_kv_i32_f32(i32* k, f32* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef i32 kk
    cdef f32 vv
    for i in range(1, n):
        kk = k[i]
        vv = v[i]
        j = i - 1
        while j >= 0 and k[j] > kk:
            k[j+1] = k[j]
            v[j+1] = v[j]
            j -= 1
        k[j+1] = kk
        v[j+1] = vv

cdef inline void _isort_kv_i64_f32(i64* k, f32* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef i64 kk
    cdef f32 vv
    for i in range(1, n):
        kk = k[i]
        vv = v[i]
        j = i - 1
        while j >= 0 and k[j] > kk:
            k[j+1] = k[j]
            v[j+1] = v[j]
            j -= 1
        k[j+1] = kk
        v[j+1] = vv

# Iterative quicksort: key-only
cdef void _qsort_i32(i32* a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t stk[_QS_STACK]
    cdef Py_ssize_t sp=0, lo, hi, mid, i, j, sz
    cdef i32 piv
    if n <= _ISORT_CUTOFF:
        _isort_i32(a, n)
        return
    stk[sp]=0
    stk[sp+1]=n-1
    sp+=2
    while sp > 0:
        sp-=2
        lo=stk[sp]
        hi=stk[sp+1]
        sz=hi-lo+1
        if sz <= _ISORT_CUTOFF:
            _isort_i32(a+lo, sz)
            continue
        mid = lo + (hi-lo)//2
        if a[lo]>a[mid]: _sw32(&a[lo],&a[mid])
        if a[lo]>a[hi]:  _sw32(&a[lo],&a[hi])
        if a[mid]>a[hi]: _sw32(&a[mid],&a[hi])
        piv=a[mid]
        i=lo
        j=hi
        while True:
            while a[i]<piv: i+=1
            while a[j]>piv: j-=1
            if i>=j: break
            _sw32(&a[i],&a[j])
            i+=1
            j-=1
        if j-lo > hi-j-1:
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
        else:
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2

cdef void _qsort_i64(i64* a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t stk[_QS_STACK]
    cdef Py_ssize_t sp=0, lo, hi, mid, i, j, sz
    cdef i64 piv
    if n <= _ISORT_CUTOFF:
        _isort_i64(a, n)
        return
    stk[sp]=0
    stk[sp+1]=n-1
    sp+=2
    while sp > 0:
        sp-=2
        lo=stk[sp]
        hi=stk[sp+1]
        sz=hi-lo+1
        if sz <= _ISORT_CUTOFF:
            _isort_i64(a+lo, sz)
            continue
        mid = lo + (hi-lo)//2
        if a[lo]>a[mid]: _sw64(&a[lo],&a[mid])
        if a[lo]>a[hi]:  _sw64(&a[lo],&a[hi])
        if a[mid]>a[hi]: _sw64(&a[mid],&a[hi])
        piv=a[mid]
        i=lo
        j=hi
        while True:
            while a[i]<piv: i+=1
            while a[j]>piv: j-=1
            if i>=j: break
            _sw64(&a[i],&a[j])
            i+=1
            j-=1
        if j-lo > hi-j-1:
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
        else:
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2

# Iterative quicksort: paired key-value
cdef void _qsort_kv_i32_f64(i32* k, f64* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t stk[_QS_STACK]
    cdef Py_ssize_t sp=0, lo, hi, mid, i, j, sz
    cdef i32 piv
    if n <= _ISORT_CUTOFF:
        _isort_kv_i32_f64(k, v, n)
        return
    stk[sp]=0
    stk[sp+1]=n-1
    sp+=2
    while sp > 0:
        sp-=2
        lo=stk[sp]
        hi=stk[sp+1]
        sz=hi-lo+1
        if sz <= _ISORT_CUTOFF:
            _isort_kv_i32_f64(k+lo, v+lo, sz)
            continue
        mid = lo + (hi-lo)//2
        if k[lo]>k[mid]:
            _sw32(&k[lo],&k[mid])
            _swf64(&v[lo],&v[mid])
        if k[lo]>k[hi]:
            _sw32(&k[lo],&k[hi])
            _swf64(&v[lo],&v[hi])
        if k[mid]>k[hi]:
            _sw32(&k[mid],&k[hi])
            _swf64(&v[mid],&v[hi])
        piv=k[mid]
        i=lo
        j=hi
        while True:
            while k[i]<piv: i+=1
            while k[j]>piv: j-=1
            if i>=j: break
            _sw32(&k[i],&k[j])
            _swf64(&v[i],&v[j])
            i+=1
            j-=1
        if j-lo > hi-j-1:
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
        else:
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2

cdef void _qsort_kv_i64_f64(i64* k, f64* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t stk[_QS_STACK]
    cdef Py_ssize_t sp=0, lo, hi, mid, i, j, sz
    cdef i64 piv
    if n <= _ISORT_CUTOFF:
        _isort_kv_i64_f64(k, v, n)
        return
    stk[sp]=0
    stk[sp+1]=n-1
    sp+=2
    while sp > 0:
        sp-=2
        lo=stk[sp]
        hi=stk[sp+1]
        sz=hi-lo+1
        if sz <= _ISORT_CUTOFF:
            _isort_kv_i64_f64(k+lo, v+lo, sz)
            continue
        mid = lo + (hi-lo)//2
        if k[lo]>k[mid]:
            _sw64(&k[lo],&k[mid])
            _swf64(&v[lo],&v[mid])
        if k[lo]>k[hi]:
            _sw64(&k[lo],&k[hi])
            _swf64(&v[lo],&v[hi])
        if k[mid]>k[hi]:
            _sw64(&k[mid],&k[hi])
            _swf64(&v[mid],&v[hi])
        piv=k[mid]
        i=lo
        j=hi
        while True:
            while k[i]<piv: i+=1
            while k[j]>piv: j-=1
            if i>=j: break
            _sw64(&k[i],&k[j])
            _swf64(&v[i],&v[j])
            i+=1
            j-=1
        if j-lo > hi-j-1:
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
        else:
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2

cdef void _qsort_kv_i32_f32(i32* k, f32* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t stk[_QS_STACK]
    cdef Py_ssize_t sp=0, lo, hi, mid, i, j, sz
    cdef i32 piv
    if n <= _ISORT_CUTOFF:
        _isort_kv_i32_f32(k, v, n)
        return
    stk[sp]=0
    stk[sp+1]=n-1
    sp+=2
    while sp > 0:
        sp-=2
        lo=stk[sp]
        hi=stk[sp+1]
        sz=hi-lo+1
        if sz <= _ISORT_CUTOFF:
            _isort_kv_i32_f32(k+lo, v+lo, sz)
            continue
        mid = lo + (hi-lo)//2
        if k[lo]>k[mid]:
            _sw32(&k[lo],&k[mid])
            _swf32(&v[lo],&v[mid])
        if k[lo]>k[hi]:
            _sw32(&k[lo],&k[hi])
            _swf32(&v[lo],&v[hi])
        if k[mid]>k[hi]:
            _sw32(&k[mid],&k[hi])
            _swf32(&v[mid],&v[hi])
        piv=k[mid]
        i=lo
        j=hi
        while True:
            while k[i]<piv: i+=1
            while k[j]>piv: j-=1
            if i>=j: break
            _sw32(&k[i],&k[j])
            _swf32(&v[i],&v[j])
            i+=1
            j-=1
        if j-lo > hi-j-1:
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
        else:
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2

cdef void _qsort_kv_i64_f32(i64* k, f32* v, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t stk[_QS_STACK]
    cdef Py_ssize_t sp=0, lo, hi, mid, i, j, sz
    cdef i64 piv
    if n <= _ISORT_CUTOFF:
        _isort_kv_i64_f32(k, v, n)
        return
    stk[sp]=0
    stk[sp+1]=n-1
    sp+=2
    while sp > 0:
        sp-=2
        lo=stk[sp]
        hi=stk[sp+1]
        sz=hi-lo+1
        if sz <= _ISORT_CUTOFF:
            _isort_kv_i64_f32(k+lo, v+lo, sz)
            continue
        mid = lo + (hi-lo)//2
        if k[lo]>k[mid]:
            _sw64(&k[lo],&k[mid])
            _swf32(&v[lo],&v[mid])
        if k[lo]>k[hi]:
            _sw64(&k[lo],&k[hi])
            _swf32(&v[lo],&v[hi])
        if k[mid]>k[hi]:
            _sw64(&k[mid],&k[hi])
            _swf32(&v[mid],&v[hi])
        piv=k[mid]
        i=lo
        j=hi
        while True:
            while k[i]<piv: i+=1
            while k[j]>piv: j-=1
            if i>=j: break
            _sw64(&k[i],&k[j])
            _swf32(&v[i],&v[j])
            i+=1
            j-=1
        if j-lo > hi-j-1:
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
        else:
            if j+1<hi:
                stk[sp]=j+1
                stk[sp+1]=hi
                sp+=2
            if lo<j:
                stk[sp]=lo
                stk[sp+1]=j
                sp+=2


# Classes

cdef class CSRMatrix:
    """Compressed Sparse Row matrix backed by contiguous numpy arrays."""
    cdef public:
        object _row_ptr_arr
        object _col_idx_arr
        object _vals_arr
        Py_ssize_t nrow, ncol, nnz
        int idx_bits, val_bits

    def __init__(self, row_ptr, col_idx, vals, Py_ssize_t nrow, Py_ssize_t ncol):
        self._row_ptr_arr = np.ascontiguousarray(row_ptr)
        self._col_idx_arr = np.ascontiguousarray(col_idx)
        self._vals_arr = np.ascontiguousarray(vals)
        self.nrow = nrow
        self.ncol = ncol
        self.nnz = len(vals)
        self.idx_bits = 32 if self._row_ptr_arr.dtype == np.int32 else 64
        self.val_bits = 32 if self._vals_arr.dtype == np.float32 else 64

    @property
    def row_ptr(self): return self._row_ptr_arr
    @property
    def col_idx(self): return self._col_idx_arr
    @property
    def vals(self): return self._vals_arr

    def __repr__(self):
        return f"CSRMatrix({self.nrow}x{self.ncol}, nnz={self.nnz}, idx{self.idx_bits}, val{self.val_bits})"


cdef class DualCSR:
    """Dual CSR/CSC storage, built in a single pass through COO data."""
    cdef public:
        CSRMatrix csr
        object _col_ptr_arr, _row_idx_arr, _vals_csc_arr

    def __init__(self, CSRMatrix csr, col_ptr, row_idx, vals_csc):
        self.csr = csr
        self._col_ptr_arr = np.ascontiguousarray(col_ptr)
        self._row_idx_arr = np.ascontiguousarray(row_idx)
        self._vals_csc_arr = np.ascontiguousarray(vals_csc)

    @property
    def nrow(self): return self.csr.nrow
    @property
    def ncol(self): return self.csr.ncol
    @property
    def nnz(self): return self.csr.nnz
    @property
    def row_ptr(self): return self.csr._row_ptr_arr
    @property
    def col_idx(self): return self.csr._col_idx_arr
    @property
    def vals(self): return self.csr._vals_arr
    @property
    def col_ptr(self): return self._col_ptr_arr
    @property
    def row_idx(self): return self._row_idx_arr
    @property
    def vals_csc(self): return self._vals_csc_arr
    @property
    def idx_bits(self): return self.csr.idx_bits
    @property
    def val_bits(self): return self.csr.val_bits

    def __repr__(self):
        return f"DualCSR({self.csr.nrow}x{self.csr.ncol}, nnz={self.csr.nnz}, idx{self.csr.idx_bits}, val{self.csr.val_bits})"


# Type selection and allocation

def select_idx_bits(Py_ssize_t max_dim):
    return 64 if max_dim >= MAX_INT32_NNZ else 32

def select_val_bits(Py_ssize_t nnz, bint force_64=False):
    return 64 if (force_64 or nnz < 10_000_000) else 32

def aligned_empty_idx(Py_ssize_t n, bint use_64=False):
    return np.empty(n, dtype=np.int64 if use_64 else np.int32)

def aligned_empty_val(Py_ssize_t n, bint use_64=True):
    return np.empty(n, dtype=np.float64 if use_64 else np.float32)

def aligned_zeros_idx(Py_ssize_t n, bint use_64=False):
    return np.zeros(n, dtype=np.int64 if use_64 else np.int32)

def aligned_zeros_val(Py_ssize_t n, bint use_64=True):
    return np.zeros(n, dtype=np.float64 if use_64 else np.float32)


# COO to CSR construction

@cython.boundscheck(False)
@cython.wraparound(False)
def csr_from_coo_i32_f64(np.ndarray[i32, ndim=1] rows, np.ndarray[i32, ndim=1] cols,
                          np.ndarray[f64, ndim=1] vals, Py_ssize_t nrow, Py_ssize_t ncol):
    """Build CSR from COO. i32+f64. Duplicates summed. Rows sorted by col."""
    cdef Py_ssize_t nnz_in = rows.shape[0], i, k, dest
    cdef i32[::1] r_mv=rows, c_mv=cols
    cdef f64[::1] v_mv=vals
    cdef np.ndarray[i32, ndim=1] rc = np.zeros(nrow, dtype=np.int32)
    cdef i32[::1] rc_mv = rc
    for k in range(nnz_in): rc_mv[r_mv[k]] += 1
    cdef np.ndarray[i32, ndim=1] rp = np.empty(nrow+1, dtype=np.int32)
    cdef i32[::1] rp_mv = rp
    rp_mv[0] = 0
    for i in range(nrow): rp_mv[i+1] = rp_mv[i] + rc_mv[i]
    cdef Py_ssize_t nnz_out = rp_mv[nrow]
    cdef np.ndarray[i32, ndim=1] ci = np.empty(nnz_out, dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] cv = np.zeros(nnz_out, dtype=np.float64)
    cdef i32[::1] ci_mv=ci
    cdef f64[::1] cv_mv=cv
    cdef np.ndarray[i32, ndim=1] cur = rp[:nrow].copy()
    cdef i32[::1] cur_mv = cur
    for k in range(nnz_in):
        i = r_mv[k]
        dest = cur_mv[i]
        ci_mv[dest] = c_mv[k]
        cv_mv[dest] += v_mv[k]
        cur_mv[i] += 1
    cdef Py_ssize_t s, e, n_in
    with nogil:
        for i in range(nrow):
            s=rp_mv[i]
            e=rp_mv[i+1]
            n_in=e-s
            if n_in > 1: _qsort_kv_i32_f64(&ci_mv[s], &cv_mv[s], n_in)
    return CSRMatrix(rp, ci, cv, nrow, ncol)

@cython.boundscheck(False)
@cython.wraparound(False)
def csr_from_coo_i64_f64(np.ndarray[i64, ndim=1] rows, np.ndarray[i64, ndim=1] cols,
                          np.ndarray[f64, ndim=1] vals, Py_ssize_t nrow, Py_ssize_t ncol):
    cdef Py_ssize_t nnz_in = rows.shape[0], i, k, dest
    cdef i64[::1] r_mv=rows, c_mv=cols
    cdef f64[::1] v_mv=vals
    cdef np.ndarray[i64, ndim=1] rc = np.zeros(nrow, dtype=np.int64)
    cdef i64[::1] rc_mv = rc
    for k in range(nnz_in): rc_mv[r_mv[k]] += 1
    cdef np.ndarray[i64, ndim=1] rp = np.empty(nrow+1, dtype=np.int64)
    cdef i64[::1] rp_mv = rp
    rp_mv[0] = 0
    for i in range(nrow): rp_mv[i+1] = rp_mv[i] + rc_mv[i]
    cdef Py_ssize_t nnz_out = rp_mv[nrow]
    cdef np.ndarray[i64, ndim=1] ci = np.empty(nnz_out, dtype=np.int64)
    cdef np.ndarray[f64, ndim=1] cv = np.zeros(nnz_out, dtype=np.float64)
    cdef i64[::1] ci_mv=ci
    cdef f64[::1] cv_mv=cv
    cdef np.ndarray[i64, ndim=1] cur = rp[:nrow].copy()
    cdef i64[::1] cur_mv = cur
    for k in range(nnz_in):
        i = r_mv[k]
        dest = cur_mv[i]
        ci_mv[dest] = c_mv[k]
        cv_mv[dest] += v_mv[k]
        cur_mv[i] += 1
    cdef Py_ssize_t s, e, n_in
    with nogil:
        for i in range(nrow):
            s=rp_mv[i]
            e=rp_mv[i+1]
            n_in=e-s
            if n_in > 1: _qsort_kv_i64_f64(&ci_mv[s], &cv_mv[s], n_in)
    return CSRMatrix(rp, ci, cv, nrow, ncol)

@cython.boundscheck(False)
@cython.wraparound(False)
def csr_from_coo_i32_f32(np.ndarray[i32, ndim=1] rows, np.ndarray[i32, ndim=1] cols,
                          np.ndarray[f32, ndim=1] vals, Py_ssize_t nrow, Py_ssize_t ncol):
    cdef Py_ssize_t nnz_in = rows.shape[0], i, k, dest
    cdef i32[::1] r_mv=rows, c_mv=cols
    cdef f32[::1] v_mv=vals
    cdef np.ndarray[i32, ndim=1] rc = np.zeros(nrow, dtype=np.int32)
    cdef i32[::1] rc_mv = rc
    for k in range(nnz_in): rc_mv[r_mv[k]] += 1
    cdef np.ndarray[i32, ndim=1] rp = np.empty(nrow+1, dtype=np.int32)
    cdef i32[::1] rp_mv = rp
    rp_mv[0] = 0
    for i in range(nrow): rp_mv[i+1] = rp_mv[i] + rc_mv[i]
    cdef Py_ssize_t nnz_out = rp_mv[nrow]
    cdef np.ndarray[i32, ndim=1] ci = np.empty(nnz_out, dtype=np.int32)
    cdef np.ndarray[f32, ndim=1] cv = np.zeros(nnz_out, dtype=np.float32)
    cdef i32[::1] ci_mv=ci
    cdef f32[::1] cv_mv=cv
    cdef np.ndarray[i32, ndim=1] cur = rp[:nrow].copy()
    cdef i32[::1] cur_mv = cur
    for k in range(nnz_in):
        i = r_mv[k]
        dest = cur_mv[i]
        ci_mv[dest] = c_mv[k]
        cv_mv[dest] += v_mv[k]
        cur_mv[i] += 1
    cdef Py_ssize_t s, e, n_in
    with nogil:
        for i in range(nrow):
            s=rp_mv[i]
            e=rp_mv[i+1]
            n_in=e-s
            if n_in > 1: _qsort_kv_i32_f32(&ci_mv[s], &cv_mv[s], n_in)
    return CSRMatrix(rp, ci, cv, nrow, ncol)

@cython.boundscheck(False)
@cython.wraparound(False)
def csr_from_coo_i64_f32(np.ndarray[i64, ndim=1] rows, np.ndarray[i64, ndim=1] cols,
                          np.ndarray[f32, ndim=1] vals, Py_ssize_t nrow, Py_ssize_t ncol):
    cdef Py_ssize_t nnz_in = rows.shape[0], i, k, dest
    cdef i64[::1] r_mv=rows, c_mv=cols
    cdef f32[::1] v_mv=vals
    cdef np.ndarray[i64, ndim=1] rc = np.zeros(nrow, dtype=np.int64)
    cdef i64[::1] rc_mv = rc
    for k in range(nnz_in): rc_mv[r_mv[k]] += 1
    cdef np.ndarray[i64, ndim=1] rp = np.empty(nrow+1, dtype=np.int64)
    cdef i64[::1] rp_mv = rp
    rp_mv[0] = 0
    for i in range(nrow): rp_mv[i+1] = rp_mv[i] + rc_mv[i]
    cdef Py_ssize_t nnz_out = rp_mv[nrow]
    cdef np.ndarray[i64, ndim=1] ci = np.empty(nnz_out, dtype=np.int64)
    cdef np.ndarray[f32, ndim=1] cv = np.zeros(nnz_out, dtype=np.float32)
    cdef i64[::1] ci_mv=ci
    cdef f32[::1] cv_mv=cv
    cdef np.ndarray[i64, ndim=1] cur = rp[:nrow].copy()
    cdef i64[::1] cur_mv = cur
    for k in range(nnz_in):
        i = r_mv[k]
        dest = cur_mv[i]
        ci_mv[dest] = c_mv[k]
        cv_mv[dest] += v_mv[k]
        cur_mv[i] += 1
    cdef Py_ssize_t s, e, n_in
    with nogil:
        for i in range(nrow):
            s=rp_mv[i]
            e=rp_mv[i+1]
            n_in=e-s
            if n_in > 1: _qsort_kv_i64_f32(&ci_mv[s], &cv_mv[s], n_in)
    return CSRMatrix(rp, ci, cv, nrow, ncol)

def csr_from_coo(rows, cols, vals, Py_ssize_t nrow, Py_ssize_t ncol):
    """Build CSR from COO. Auto-selects i32/i64 x f32/f64."""
    if not isinstance(rows, np.ndarray): rows = np.asarray(rows)
    if not isinstance(cols, np.ndarray): cols = np.asarray(cols)
    if not isinstance(vals, np.ndarray): vals = np.asarray(vals)
    cdef bint i64 = (rows.dtype == np.int64 or cols.dtype == np.int64 or max(nrow,ncol) >= (1<<31)-1)
    cdef bint f32 = (vals.dtype == np.float32)
    if i64:
        r = rows.astype(np.int64, copy=False)
        c = cols.astype(np.int64, copy=False)
        return csr_from_coo_i64_f32(r, c, vals.astype(np.float32, copy=False), nrow, ncol) if f32 else csr_from_coo_i64_f64(r, c, vals.astype(np.float64, copy=False), nrow, ncol)
    r = rows.astype(np.int32, copy=False)
    c = cols.astype(np.int32, copy=False)
    return csr_from_coo_i32_f32(r, c, vals.astype(np.float32, copy=False), nrow, ncol) if f32 else csr_from_coo_i32_f64(r, c, vals.astype(np.float64, copy=False), nrow, ncol)


# Dual CSR construction (CSR + CSC in one pass)

@cython.boundscheck(False)
@cython.wraparound(False)
def dual_from_coo_i32_f64(np.ndarray[i32, ndim=1] rows, np.ndarray[i32, ndim=1] cols,
                           np.ndarray[f64, ndim=1] vals, Py_ssize_t nrow, Py_ssize_t ncol):
    """Build DualCSR (CSR+CSC) in one pass. i32+f64."""
    cdef Py_ssize_t nnz_in=rows.shape[0], i, k
    cdef i32[::1] r_mv=rows, c_mv=cols
    cdef f64[::1] v_mv=vals
    cdef np.ndarray[i32, ndim=1] rc=np.zeros(nrow,dtype=np.int32), cc=np.zeros(ncol,dtype=np.int32)
    cdef i32[::1] rc_mv=rc, cc_mv=cc
    for k in range(nnz_in):
        rc_mv[r_mv[k]]+=1
        cc_mv[c_mv[k]]+=1
    cdef np.ndarray[i32, ndim=1] rp=np.empty(nrow+1,dtype=np.int32), cp=np.empty(ncol+1,dtype=np.int32)
    cdef i32[::1] rp_mv=rp, cp_mv=cp
    cdef i32 cs=0
    for i in range(nrow):
        rp_mv[i]=cs
        cs+=rc_mv[i]
    rp_mv[nrow]=cs
    cdef Py_ssize_t nnz_out=cs
    cs=0
    for i in range(ncol):
        cp_mv[i]=cs
        cs+=cc_mv[i]
    cp_mv[ncol]=cs
    cdef np.ndarray[i32, ndim=1] ci_a=np.empty(nnz_out,dtype=np.int32), ri_a=np.empty(nnz_out,dtype=np.int32)
    cdef np.ndarray[f64, ndim=1] cv_a=np.zeros(nnz_out,dtype=np.float64), vc_a=np.zeros(nnz_out,dtype=np.float64)
    cdef i32[::1] ci_mv=ci_a, ri_mv=ri_a
    cdef f64[::1] cv_mv=cv_a, vc_mv=vc_a
    rc_mv[:]=0
    cc_mv[:]=0
    cdef i32 r, c
    cdef f64 v
    cdef Py_ssize_t dr, dc
    for k in range(nnz_in):
        r=r_mv[k]
        c=c_mv[k]
        v=v_mv[k]
        dr=rp_mv[r]+rc_mv[r]
        ci_mv[dr]=c
        cv_mv[dr]+=v
        rc_mv[r]+=1
        dc=cp_mv[c]+cc_mv[c]
        ri_mv[dc]=r
        vc_mv[dc]+=v
        cc_mv[c]+=1
    cdef Py_ssize_t s, e, n_in
    with nogil:
        for i in range(nrow):
            s=rp_mv[i]
            n_in=rp_mv[i+1]-s
            if n_in>1: _qsort_kv_i32_f64(&ci_mv[s],&cv_mv[s],n_in)
        for i in range(ncol):
            s=cp_mv[i]
            n_in=cp_mv[i+1]-s
            if n_in>1: _qsort_kv_i32_f64(&ri_mv[s],&vc_mv[s],n_in)
    return DualCSR(CSRMatrix(rp,ci_a,cv_a,nrow,ncol), cp, ri_a, vc_a)

@cython.boundscheck(False)
@cython.wraparound(False)
def dual_from_coo_i64_f64(np.ndarray[i64, ndim=1] rows, np.ndarray[i64, ndim=1] cols,
                           np.ndarray[f64, ndim=1] vals, Py_ssize_t nrow, Py_ssize_t ncol):
    """Build DualCSR single-pass. i64+f64."""
    cdef Py_ssize_t nnz_in=rows.shape[0], i, k
    cdef i64[::1] r_mv=rows, c_mv=cols
    cdef f64[::1] v_mv=vals
    cdef np.ndarray[i64, ndim=1] rc=np.zeros(nrow,dtype=np.int64), cc=np.zeros(ncol,dtype=np.int64)
    cdef i64[::1] rc_mv=rc, cc_mv=cc
    for k in range(nnz_in):
        rc_mv[r_mv[k]]+=1
        cc_mv[c_mv[k]]+=1
    cdef np.ndarray[i64, ndim=1] rp=np.empty(nrow+1,dtype=np.int64), cp=np.empty(ncol+1,dtype=np.int64)
    cdef i64[::1] rp_mv=rp, cp_mv=cp
    cdef i64 cs=0
    for i in range(nrow):
        rp_mv[i]=cs
        cs+=rc_mv[i]
    rp_mv[nrow]=cs
    cdef Py_ssize_t nnz_out=cs
    cs=0
    for i in range(ncol):
        cp_mv[i]=cs
        cs+=cc_mv[i]
    cp_mv[ncol]=cs
    cdef np.ndarray[i64, ndim=1] ci_a=np.empty(nnz_out,dtype=np.int64), ri_a=np.empty(nnz_out,dtype=np.int64)
    cdef np.ndarray[f64, ndim=1] cv_a=np.zeros(nnz_out,dtype=np.float64), vc_a=np.zeros(nnz_out,dtype=np.float64)
    cdef i64[::1] ci_mv=ci_a, ri_mv=ri_a
    cdef f64[::1] cv_mv=cv_a, vc_mv=vc_a
    rc_mv[:]=0
    cc_mv[:]=0
    cdef i64 r, c
    cdef f64 v
    cdef Py_ssize_t dr, dc
    for k in range(nnz_in):
        r=r_mv[k]
        c=c_mv[k]
        v=v_mv[k]
        dr=rp_mv[r]+rc_mv[r]
        ci_mv[dr]=c
        cv_mv[dr]+=v
        rc_mv[r]+=1
        dc=cp_mv[c]+cc_mv[c]
        ri_mv[dc]=r
        vc_mv[dc]+=v
        cc_mv[c]+=1
    cdef Py_ssize_t s, n_in
    with nogil:
        for i in range(nrow):
            s=rp_mv[i]
            n_in=rp_mv[i+1]-s
            if n_in>1: _qsort_kv_i64_f64(&ci_mv[s],&cv_mv[s],n_in)
        for i in range(ncol):
            s=cp_mv[i]
            n_in=cp_mv[i+1]-s
            if n_in>1: _qsort_kv_i64_f64(&ri_mv[s],&vc_mv[s],n_in)
    return DualCSR(CSRMatrix(rp,ci_a,cv_a,nrow,ncol), cp, ri_a, vc_a)


@cython.boundscheck(False)
@cython.wraparound(False)
def dual_from_coo_i32_f32(np.ndarray[i32, ndim=1] rows, np.ndarray[i32, ndim=1] cols,
                           np.ndarray[f32, ndim=1] vals, Py_ssize_t nrow, Py_ssize_t ncol):
    cdef Py_ssize_t nnz_in=rows.shape[0], i, k
    cdef i32[::1] r_mv=rows, c_mv=cols
    cdef f32[::1] v_mv=vals
    cdef np.ndarray[i32, ndim=1] rc=np.zeros(nrow,dtype=np.int32), cc=np.zeros(ncol,dtype=np.int32)
    cdef i32[::1] rc_mv=rc, cc_mv=cc
    for k in range(nnz_in):
        rc_mv[r_mv[k]]+=1
        cc_mv[c_mv[k]]+=1
    cdef np.ndarray[i32, ndim=1] rp=np.empty(nrow+1,dtype=np.int32), cp=np.empty(ncol+1,dtype=np.int32)
    cdef i32[::1] rp_mv=rp, cp_mv=cp
    cdef i32 cs=0
    for i in range(nrow):
        rp_mv[i]=cs
        cs+=rc_mv[i]
    rp_mv[nrow]=cs
    cdef Py_ssize_t nnz_out=cs
    cs=0
    for i in range(ncol):
        cp_mv[i]=cs
        cs+=cc_mv[i]
    cp_mv[ncol]=cs
    cdef np.ndarray[i32, ndim=1] ci_a=np.empty(nnz_out,dtype=np.int32), ri_a=np.empty(nnz_out,dtype=np.int32)
    cdef np.ndarray[f32, ndim=1] cv_a=np.zeros(nnz_out,dtype=np.float32), vc_a=np.zeros(nnz_out,dtype=np.float32)
    cdef i32[::1] ci_mv=ci_a, ri_mv=ri_a
    cdef f32[::1] cv_mv=cv_a, vc_mv=vc_a
    rc_mv[:]=0
    cc_mv[:]=0
    cdef i32 r, c
    cdef f32 v
    cdef Py_ssize_t dr, dc
    for k in range(nnz_in):
        r=r_mv[k]
        c=c_mv[k]
        v=v_mv[k]
        dr=rp_mv[r]+rc_mv[r]
        ci_mv[dr]=c
        cv_mv[dr]+=v
        rc_mv[r]+=1
        dc=cp_mv[c]+cc_mv[c]
        ri_mv[dc]=r
        vc_mv[dc]+=v
        cc_mv[c]+=1
    cdef Py_ssize_t s, n_in
    with nogil:
        for i in range(nrow):
            s=rp_mv[i]
            n_in=rp_mv[i+1]-s
            if n_in>1: _qsort_kv_i32_f32(&ci_mv[s],&cv_mv[s],n_in)
        for i in range(ncol):
            s=cp_mv[i]
            n_in=cp_mv[i+1]-s
            if n_in>1: _qsort_kv_i32_f32(&ri_mv[s],&vc_mv[s],n_in)
    return DualCSR(CSRMatrix(rp,ci_a,cv_a,nrow,ncol), cp, ri_a, vc_a)

@cython.boundscheck(False)
@cython.wraparound(False)
def dual_from_coo_i64_f32(np.ndarray[i64, ndim=1] rows, np.ndarray[i64, ndim=1] cols,
                           np.ndarray[f32, ndim=1] vals, Py_ssize_t nrow, Py_ssize_t ncol):
    cdef Py_ssize_t nnz_in=rows.shape[0], i, k
    cdef i64[::1] r_mv=rows, c_mv=cols
    cdef f32[::1] v_mv=vals
    cdef np.ndarray[i64, ndim=1] rc=np.zeros(nrow,dtype=np.int64), cc=np.zeros(ncol,dtype=np.int64)
    cdef i64[::1] rc_mv=rc, cc_mv=cc
    for k in range(nnz_in):
        rc_mv[r_mv[k]]+=1
        cc_mv[c_mv[k]]+=1
    cdef np.ndarray[i64, ndim=1] rp=np.empty(nrow+1,dtype=np.int64), cp=np.empty(ncol+1,dtype=np.int64)
    cdef i64[::1] rp_mv=rp, cp_mv=cp
    cdef i64 cs=0
    for i in range(nrow):
        rp_mv[i]=cs
        cs+=rc_mv[i]
    rp_mv[nrow]=cs
    cdef Py_ssize_t nnz_out=cs
    cs=0
    for i in range(ncol):
        cp_mv[i]=cs
        cs+=cc_mv[i]
    cp_mv[ncol]=cs
    cdef np.ndarray[i64, ndim=1] ci_a=np.empty(nnz_out,dtype=np.int64), ri_a=np.empty(nnz_out,dtype=np.int64)
    cdef np.ndarray[f32, ndim=1] cv_a=np.zeros(nnz_out,dtype=np.float32), vc_a=np.zeros(nnz_out,dtype=np.float32)
    cdef i64[::1] ci_mv=ci_a, ri_mv=ri_a
    cdef f32[::1] cv_mv=cv_a, vc_mv=vc_a
    rc_mv[:]=0
    cc_mv[:]=0
    cdef i64 r, c
    cdef f32 v
    cdef Py_ssize_t dr, dc
    for k in range(nnz_in):
        r=r_mv[k]
        c=c_mv[k]
        v=v_mv[k]
        dr=rp_mv[r]+rc_mv[r]
        ci_mv[dr]=c
        cv_mv[dr]+=v
        rc_mv[r]+=1
        dc=cp_mv[c]+cc_mv[c]
        ri_mv[dc]=r
        vc_mv[dc]+=v
        cc_mv[c]+=1
    cdef Py_ssize_t s, n_in
    with nogil:
        for i in range(nrow):
            s=rp_mv[i]
            n_in=rp_mv[i+1]-s
            if n_in>1: _qsort_kv_i64_f32(&ci_mv[s],&cv_mv[s],n_in)
        for i in range(ncol):
            s=cp_mv[i]
            n_in=cp_mv[i+1]-s
            if n_in>1: _qsort_kv_i64_f32(&ri_mv[s],&vc_mv[s],n_in)
    return DualCSR(CSRMatrix(rp,ci_a,cv_a,nrow,ncol), cp, ri_a, vc_a)

def dual_from_coo(rows, cols, vals, Py_ssize_t nrow, Py_ssize_t ncol):
    """Build DualCSR (CSR+CSC) single-pass. Auto-typed."""
    if not isinstance(rows, np.ndarray): rows = np.asarray(rows)
    if not isinstance(cols, np.ndarray): cols = np.asarray(cols)
    if not isinstance(vals, np.ndarray): vals = np.asarray(vals)
    cdef bint i64 = (rows.dtype == np.int64 or cols.dtype == np.int64 or max(nrow,ncol) >= (1<<31)-1)
    cdef bint f32 = (vals.dtype == np.float32)
    if i64:
        r=rows.astype(np.int64,copy=False)
        c=cols.astype(np.int64,copy=False)
        return dual_from_coo_i64_f32(r,c,vals.astype(np.float32,copy=False),nrow,ncol) if f32 else dual_from_coo_i64_f64(r,c,vals.astype(np.float64,copy=False),nrow,ncol)
    r=rows.astype(np.int32,copy=False)
    c=cols.astype(np.int32,copy=False)
    return dual_from_coo_i32_f32(r,c,vals.astype(np.float32,copy=False),nrow,ncol) if f32 else dual_from_coo_i32_f64(r,c,vals.astype(np.float64,copy=False),nrow,ncol)

def dual_from_csr(CSRMatrix csr):
    """Add CSC storage to existing CSRMatrix."""
    cdef Py_ssize_t nrow=csr.nrow, nnz=csr.nnz, i, k
    cdef i32[::1] rp_v, r_mv
    cdef i64[::1] rp_v64, r_mv64
    if csr.idx_bits == 32:
        rows = np.empty(nnz, dtype=np.int32)
        rp_v = csr._row_ptr_arr
        r_mv = rows
        k = 0
        for i in range(nrow):
            for _ in range(rp_v[i], rp_v[i+1]):
                r_mv[k] = <i32>i
                k += 1
    else:
        rows = np.empty(nnz, dtype=np.int64)
        rp_v64 = csr._row_ptr_arr
        r_mv64 = rows
        k = 0
        for i in range(nrow):
            for _ in range(rp_v64[i], rp_v64[i+1]):
                r_mv64[k] = <i64>i
                k += 1
    return dual_from_coo(rows, csr._col_idx_arr, csr._vals_arr, nrow, csr.ncol)


# Matvec and rmatvec

@cython.boundscheck(False)
@cython.wraparound(False)
def matvec_i32_f64(CSRMatrix A, np.ndarray[f64, ndim=1] x):
    cdef Py_ssize_t nrow=A.nrow
    cdef np.ndarray[f64, ndim=1] y = np.zeros(nrow, dtype=np.float64)
    cdef i32[::1] rp=A._row_ptr_arr, ci=A._col_idx_arr
    cdef f64[::1] av=A._vals_arr, yv=y, xv=x
    cdef Py_ssize_t i, k
    cdef f64 acc
    with nogil:
        for i in range(nrow):
            acc=0.0
            for k in range(rp[i], rp[i+1]): acc=acc+av[k]*xv[ci[k]]
            yv[i]=acc
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def matvec_i64_f64(CSRMatrix A, np.ndarray[f64, ndim=1] x):
    cdef Py_ssize_t nrow=A.nrow
    cdef np.ndarray[f64, ndim=1] y = np.zeros(nrow, dtype=np.float64)
    cdef i64[::1] rp=A._row_ptr_arr, ci=A._col_idx_arr
    cdef f64[::1] av=A._vals_arr, yv=y, xv=x
    cdef Py_ssize_t i, k
    cdef f64 acc
    with nogil:
        for i in range(nrow):
            acc=0.0
            for k in range(rp[i], rp[i+1]): acc=acc+av[k]*xv[ci[k]]
            yv[i]=acc
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def matvec_i32_f32(CSRMatrix A, np.ndarray[f32, ndim=1] x):
    cdef Py_ssize_t nrow=A.nrow
    cdef np.ndarray[f32, ndim=1] y = np.zeros(nrow, dtype=np.float32)
    cdef i32[::1] rp=A._row_ptr_arr, ci=A._col_idx_arr
    cdef f32[::1] av=A._vals_arr, yv=y, xv=x
    cdef Py_ssize_t i, k
    cdef f32 acc
    with nogil:
        for i in range(nrow):
            acc=0.0
            for k in range(rp[i], rp[i+1]): acc=acc+av[k]*xv[ci[k]]
            yv[i]=acc
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def matvec_i64_f32(CSRMatrix A, np.ndarray[f32, ndim=1] x):
    cdef Py_ssize_t nrow=A.nrow
    cdef np.ndarray[f32, ndim=1] y = np.zeros(nrow, dtype=np.float32)
    cdef i64[::1] rp=A._row_ptr_arr, ci=A._col_idx_arr
    cdef f32[::1] av=A._vals_arr, yv=y, xv=x
    cdef Py_ssize_t i, k
    cdef f32 acc
    with nogil:
        for i in range(nrow):
            acc=0.0
            for k in range(rp[i], rp[i+1]): acc=acc+av[k]*xv[ci[k]]
            yv[i]=acc
    return y

def matvec(A, x):
    cdef CSRMatrix csr
    if isinstance(A, DualCSR): csr = (<DualCSR>A).csr
    elif isinstance(A, CSRMatrix): csr = A
    else: raise TypeError(f"Expected CSRMatrix or DualCSR, got {type(A)}")
    if not isinstance(x, np.ndarray): x = np.asarray(x)
    if csr.idx_bits==64:
        return matvec_i64_f32(csr, x.astype(np.float32,copy=False)) if csr.val_bits==32 else matvec_i64_f64(csr, x.astype(np.float64,copy=False))
    return matvec_i32_f32(csr, x.astype(np.float32,copy=False)) if csr.val_bits==32 else matvec_i32_f64(csr, x.astype(np.float64,copy=False))


@cython.boundscheck(False)
@cython.wraparound(False)
def rmatvec_i32_f64(DualCSR A, np.ndarray[f64, ndim=1] x):
    cdef Py_ssize_t ncol=A.csr.ncol
    cdef np.ndarray[f64, ndim=1] y = np.zeros(ncol, dtype=np.float64)
    cdef i32[::1] cp=A._col_ptr_arr, ri=A._row_idx_arr
    cdef f64[::1] cv=A._vals_csc_arr, yv=y, xv=x
    cdef Py_ssize_t j, k
    cdef f64 acc
    with nogil:
        for j in range(ncol):
            acc=0.0
            for k in range(cp[j], cp[j+1]): acc=acc+cv[k]*xv[ri[k]]
            yv[j]=acc
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rmatvec_i64_f64(DualCSR A, np.ndarray[f64, ndim=1] x):
    cdef Py_ssize_t ncol=A.csr.ncol
    cdef np.ndarray[f64, ndim=1] y = np.zeros(ncol, dtype=np.float64)
    cdef i64[::1] cp=A._col_ptr_arr, ri=A._row_idx_arr
    cdef f64[::1] cv=A._vals_csc_arr, yv=y, xv=x
    cdef Py_ssize_t j, k
    cdef f64 acc
    with nogil:
        for j in range(ncol):
            acc=0.0
            for k in range(cp[j], cp[j+1]): acc=acc+cv[k]*xv[ri[k]]
            yv[j]=acc
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rmatvec_i32_f32(DualCSR A, np.ndarray[f32, ndim=1] x):
    cdef Py_ssize_t ncol=A.csr.ncol
    cdef np.ndarray[f32, ndim=1] y = np.zeros(ncol, dtype=np.float32)
    cdef i32[::1] cp=A._col_ptr_arr, ri=A._row_idx_arr
    cdef f32[::1] cv=A._vals_csc_arr, yv=y, xv=x
    cdef Py_ssize_t j, k
    cdef f32 acc
    with nogil:
        for j in range(ncol):
            acc=0.0
            for k in range(cp[j], cp[j+1]): acc=acc+cv[k]*xv[ri[k]]
            yv[j]=acc
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def rmatvec_i64_f32(DualCSR A, np.ndarray[f32, ndim=1] x):
    cdef Py_ssize_t ncol=A.csr.ncol
    cdef np.ndarray[f32, ndim=1] y = np.zeros(ncol, dtype=np.float32)
    cdef i64[::1] cp=A._col_ptr_arr, ri=A._row_idx_arr
    cdef f32[::1] cv=A._vals_csc_arr, yv=y, xv=x
    cdef Py_ssize_t j, k
    cdef f32 acc
    with nogil:
        for j in range(ncol):
            acc=0.0
            for k in range(cp[j], cp[j+1]): acc=acc+cv[k]*xv[ri[k]]
            yv[j]=acc
    return y

def rmatvec(A, x):
    if not isinstance(A, DualCSR): raise TypeError(f"rmatvec requires DualCSR, got {type(A)}")
    cdef DualCSR d = A
    if not isinstance(x, np.ndarray): x = np.asarray(x)
    if d.csr.idx_bits==64:
        return rmatvec_i64_f32(d, x.astype(np.float32,copy=False)) if d.csr.val_bits==32 else rmatvec_i64_f64(d, x.astype(np.float64,copy=False))
    return rmatvec_i32_f32(d, x.astype(np.float32,copy=False)) if d.csr.val_bits==32 else rmatvec_i32_f64(d, x.astype(np.float64,copy=False))


# Gram products, diagonal, access, conversion, scipy, memory

@cython.boundscheck(False)
@cython.wraparound(False)
def spmm_AtA_dense_f64(DualCSR A):
    """Compute A^T A as a dense f64 matrix.

    Raises MemoryError if the output exceeds max_dense_allocation.
    """
    cdef Py_ssize_t nrow=A.csr.nrow, ncol=A.csr.ncol
    if not can_allocate_dense_f64(ncol, ncol):
        raise MemoryError(
            f"spmm_AtA_dense_f64: {ncol}x{ncol} output requires "
            f"{8.0 * ncol * ncol / (1024**3):.2f} GB, exceeds limit of "
            f"{get_max_dense_allocation_bytes() / (1024**3):.2f} GB. "
            f"Use configure_memory(max_dense_allocation=...) to increase."
        )
    cdef np.ndarray[f64, ndim=2] out = np.zeros((ncol,ncol), dtype=np.float64)
    cdef f64[:, ::1] o = out
    cdef f64[::1] av = A.csr._vals_arr
    cdef Py_ssize_t i, k1, k2
    cdef f64 v1
    cdef i32[::1] rp, ci
    cdef i64[::1] rp64, ci64
    if A.csr.idx_bits == 32:
        rp = A.csr._row_ptr_arr
        ci = A.csr._col_idx_arr
        with nogil:
            for i in range(nrow):
                for k1 in range(rp[i], rp[i+1]):
                    v1 = av[k1]
                    for k2 in range(rp[i], rp[i+1]):
                        o[ci[k1], ci[k2]] += v1 * av[k2]
    else:
        rp64 = A.csr._row_ptr_arr
        ci64 = A.csr._col_idx_arr
        with nogil:
            for i in range(nrow):
                for k1 in range(rp64[i], rp64[i+1]):
                    v1 = av[k1]
                    for k2 in range(rp64[i], rp64[i+1]):
                        o[ci64[k1], ci64[k2]] += v1 * av[k2]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def spmm_AAt_dense_f64(DualCSR A):
    """Compute A A^T as a dense f64 matrix.

    Raises MemoryError if the output exceeds max_dense_allocation.
    """
    cdef Py_ssize_t nrow=A.csr.nrow, ncol=A.csr.ncol
    if not can_allocate_dense_f64(nrow, nrow):
        raise MemoryError(
            f"spmm_AAt_dense_f64: {nrow}x{nrow} output requires "
            f"{8.0 * nrow * nrow / (1024**3):.2f} GB, exceeds limit of "
            f"{get_max_dense_allocation_bytes() / (1024**3):.2f} GB. "
            f"Use configure_memory(max_dense_allocation=...) to increase."
        )
    cdef np.ndarray[f64, ndim=2] out = np.zeros((nrow,nrow), dtype=np.float64)
    cdef f64[:, ::1] o = out
    cdef f64[::1] cv = A._vals_csc_arr
    cdef Py_ssize_t j, k1, k2
    cdef f64 v1
    cdef i32[::1] cp, ri
    cdef i64[::1] cp64, ri64
    if A.csr.idx_bits == 32:
        cp = A._col_ptr_arr
        ri = A._row_idx_arr
        with nogil:
            for j in range(ncol):
                for k1 in range(cp[j], cp[j+1]):
                    v1 = cv[k1]
                    for k2 in range(cp[j], cp[j+1]):
                        o[ri[k1], ri[k2]] += v1 * cv[k2]
    else:
        cp64 = A._col_ptr_arr
        ri64 = A._row_idx_arr
        with nogil:
            for j in range(ncol):
                for k1 in range(cp64[j], cp64[j+1]):
                    v1 = cv[k1]
                    for k2 in range(cp64[j], cp64[j+1]):
                        o[ri64[k1], ri64[k2]] += v1 * cv[k2]
    return out

def diag(A):
    """Extract diagonal of CSRMatrix or DualCSR."""
    cdef CSRMatrix csr
    if isinstance(A, DualCSR): csr = (<DualCSR>A).csr
    elif isinstance(A, CSRMatrix): csr = A
    else: raise TypeError(f"Expected CSRMatrix or DualCSR, got {type(A)}")
    cdef Py_ssize_t n = min(csr.nrow, csr.ncol)
    cdef np.ndarray[f64, ndim=1] d = np.zeros(n, dtype=np.float64)
    cdef f64[::1] dv = d, av = csr._vals_arr
    cdef Py_ssize_t i, k
    cdef i32[::1] rp, ci
    cdef i64[::1] rp64, ci64
    if csr.idx_bits == 32:
        rp = csr._row_ptr_arr
        ci = csr._col_idx_arr
        for i in range(n):
            for k in range(rp[i], rp[i+1]):
                if ci[k] == <i32>i:
                    dv[i] = av[k]
                    break
    else:
        rp64 = csr._row_ptr_arr
        ci64 = csr._col_idx_arr
        for i in range(n):
            for k in range(rp64[i], rp64[i+1]):
                if ci64[k] == <i64>i:
                    dv[i] = av[k]
                    break
    return d

def row_entries(DualCSR A, Py_ssize_t row):
    cdef Py_ssize_t s = A.csr._row_ptr_arr[row], e = A.csr._row_ptr_arr[row+1]
    return A.csr._col_idx_arr[s:e], A.csr._vals_arr[s:e]

def col_entries(DualCSR A, Py_ssize_t col):
    cdef Py_ssize_t s = A._col_ptr_arr[col], e = A._col_ptr_arr[col+1]
    return A._row_idx_arr[s:e], A._vals_csc_arr[s:e]

def row_nnz(CSRMatrix A, Py_ssize_t row):
    return int(A._row_ptr_arr[row+1] - A._row_ptr_arr[row])

def col_nnz(DualCSR A, Py_ssize_t col):
    return int(A._col_ptr_arr[col+1] - A._col_ptr_arr[col])

def to_dense_f64(A):
    """Convert CSR or DualCSR to dense f64 matrix.

    Raises MemoryError if the output exceeds max_dense_allocation.
    """
    cdef CSRMatrix csr
    if isinstance(A, DualCSR): csr = (<DualCSR>A).csr
    elif isinstance(A, CSRMatrix): csr = A
    else: raise TypeError(f"Expected CSRMatrix or DualCSR, got {type(A)}")
    if not can_allocate_dense_f64(csr.nrow, csr.ncol):
        raise MemoryError(
            f"to_dense_f64: {csr.nrow}x{csr.ncol} matrix requires "
            f"{8.0 * csr.nrow * csr.ncol / (1024**3):.2f} GB, exceeds limit. "
            f"Use configure_memory(max_dense_allocation=...) to increase."
        )
    cdef np.ndarray[f64, ndim=2] D = np.zeros((csr.nrow, csr.ncol), dtype=np.float64)
    cdef f64[:, ::1] Dv = D
    cdef f64[::1] av = csr._vals_arr
    cdef Py_ssize_t i, k
    cdef i32[::1] rp, ci
    cdef i64[::1] rp64, ci64
    if csr.idx_bits == 32:
        rp = csr._row_ptr_arr
        ci = csr._col_idx_arr
        for i in range(csr.nrow):
            for k in range(rp[i], rp[i+1]): Dv[i, ci[k]] = av[k]
    else:
        rp64 = csr._row_ptr_arr
        ci64 = csr._col_idx_arr
        for i in range(csr.nrow):
            for k in range(rp64[i], rp64[i+1]): Dv[i, ci64[k]] = av[k]
    return D

def from_dense_f64(np.ndarray[f64, ndim=2] D, double tol=-1.0):
    """Convert dense f64 matrix to DualCSR. Drops entries with |v| <= tol.

    Default tol is the library epsilon (~1e-10).
    """
    if tol < 0.0:
        tol = get_EPSILON_NORM()
    cdef Py_ssize_t nr=D.shape[0], nc=D.shape[1], i, j, nnz=0
    for i in range(nr):
        for j in range(nc):
            if fabs(D[i,j]) > tol: nnz += 1
    cdef bint u64 = max(nr,nc) >= (1<<31)-1
    rows = np.empty(nnz, dtype=np.int64 if u64 else np.int32)
    cols = np.empty(nnz, dtype=np.int64 if u64 else np.int32)
    vals = np.empty(nnz, dtype=np.float64)
    cdef Py_ssize_t k=0
    for i in range(nr):
        for j in range(nc):
            if fabs(D[i,j]) > tol:
                rows[k]=i
                cols[k]=j
                vals[k]=D[i,j]
                k+=1
    return dual_from_coo(rows, cols, vals, nr, nc)

def to_scipy_csr(A):
    try: from scipy.sparse import csr_matrix
    except ImportError: raise ImportError("scipy required")
    cdef CSRMatrix csr
    if isinstance(A, DualCSR): csr = (<DualCSR>A).csr
    elif isinstance(A, CSRMatrix): csr = A
    else: raise TypeError(f"Expected CSRMatrix or DualCSR, got {type(A)}")
    return csr_matrix((csr._vals_arr, csr._col_idx_arr, csr._row_ptr_arr),
                      shape=(csr.nrow, csr.ncol), copy=False)

def from_scipy_csr(sp_matrix):
    sp_matrix = sp_matrix.tocsr()
    sp_matrix.sort_indices()
    csr = CSRMatrix(np.asarray(sp_matrix.indptr), np.asarray(sp_matrix.indices),
                    np.asarray(sp_matrix.data), sp_matrix.shape[0], sp_matrix.shape[1])
    return dual_from_csr(csr)

def memory_bytes(A):
    cdef CSRMatrix csr
    cdef Py_ssize_t total = 0
    if isinstance(A, DualCSR):
        csr = (<DualCSR>A).csr
        total += (<DualCSR>A)._col_ptr_arr.nbytes + (<DualCSR>A)._row_idx_arr.nbytes + (<DualCSR>A)._vals_csc_arr.nbytes
    elif isinstance(A, CSRMatrix): csr = A
    else: raise TypeError(f"Expected CSRMatrix or DualCSR, got {type(A)}")
    total += csr._row_ptr_arr.nbytes + csr._col_idx_arr.nbytes + csr._vals_arr.nbytes
    return total

def memory_report(A):
    b = memory_bytes(A)
    if b < 1024: return f"{b} B"
    elif b < 1048576: return f"{b/1024:.1f} KB"
    elif b < 1073741824: return f"{b/1048576:.1f} MB"
    return f"{b/1073741824:.2f} GB"

def validate_csr(A, str name="CSR"):
    """Validate CSR structure integrity."""
    cdef CSRMatrix csr
    if isinstance(A, DualCSR): csr = (<DualCSR>A).csr
    elif isinstance(A, CSRMatrix): csr = A
    else: raise TypeError(f"Expected CSRMatrix or DualCSR")
    rp = csr._row_ptr_arr
    ci = csr._col_idx_arr
    av = csr._vals_arr
    from rexgraph.core._common import validate_csr_arrays as _validate
    try:
        _validate(rp, ci, av, name=name)
    except Exception as e:
        return False, str(e)
    if rp[csr.nrow] != csr.nnz:
        return False, f"{name}: row_ptr[-1] ({rp[csr.nrow]}) != nnz ({csr.nnz})"
    if csr.nnz > 0:
        if ci.min() < 0: return False, f"{name}: negative column index"
        if ci.max() >= csr.ncol: return False, f"{name}: column index >= ncol"
    return True, ""
