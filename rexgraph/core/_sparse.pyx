# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._sparse: Sparse matrix storage and operations
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

    UnionFind,
    uf_init, uf_free, uf_union, uf_find, uf_component_count,
    sorted_jaccard_exact_i32,
    UnionFind64,
    uf64_init, uf64_free, uf64_union, uf64_find,
    ERR_SUCCESS,
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
# Iterative quicksort with median of 3 pivot and insertion sort fallback.

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

# Insertion sort: key only
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

# Insertion sort: paired key value
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

# Iterative quicksort: key only
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

# Iterative quicksort: paired key value
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

    @property
    def shape(self): return (self.nrow, self.ncol)

    @property
    def data(self): return self._vals_arr

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
    """Build CSR from COO. Auto selects i32/i64 x f32/f64."""
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
    """Build DualCSR single pass. i64+f64."""
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
    """Build DualCSR (CSR+CSC) single pass. Auto typed."""
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
    cdef const i32[::1] rp=A._row_ptr_arr, ci=A._col_idx_arr
    cdef const f64[::1] av=A._vals_arr, xv=x
    cdef f64[::1] yv=y
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
    cdef const i64[::1] rp=A._row_ptr_arr, ci=A._col_idx_arr
    cdef const f64[::1] av=A._vals_arr, xv=x
    cdef f64[::1] yv=y
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
    cdef const i32[::1] rp=A._row_ptr_arr, ci=A._col_idx_arr
    cdef const f32[::1] av=A._vals_arr, xv=x
    cdef f32[::1] yv=y
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
    cdef const i64[::1] rp=A._row_ptr_arr, ci=A._col_idx_arr
    cdef const f32[::1] av=A._vals_arr, xv=x
    cdef f32[::1] yv=y
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
def spmm_i32_f64(CSRMatrix A, np.ndarray[f64, ndim=2] X):
    cdef Py_ssize_t nrow=A.nrow, ncol=X.shape[1]
    cdef np.ndarray[f64, ndim=2] Y = np.zeros((nrow, ncol), dtype=np.float64)
    cdef const i32[::1] rp=A._row_ptr_arr, ci=A._col_idx_arr
    cdef const f64[::1] av=A._vals_arr
    cdef const f64[:, ::1] xv=X
    cdef f64[:, ::1] yv=Y
    cdef Py_ssize_t i, k, c, j
    cdef f64 a
    with nogil:
        for i in range(nrow):
            for k in range(rp[i], rp[i+1]):
                a=av[k]; j=ci[k]
                for c in range(ncol): yv[i, c]=yv[i, c]+a*xv[j, c]
    return Y


@cython.boundscheck(False)
@cython.wraparound(False)
def spmm_i64_f64(CSRMatrix A, np.ndarray[f64, ndim=2] X):
    cdef Py_ssize_t nrow=A.nrow, ncol=X.shape[1]
    cdef np.ndarray[f64, ndim=2] Y = np.zeros((nrow, ncol), dtype=np.float64)
    cdef const i64[::1] rp=A._row_ptr_arr, ci=A._col_idx_arr
    cdef const f64[::1] av=A._vals_arr
    cdef const f64[:, ::1] xv=X
    cdef f64[:, ::1] yv=Y
    cdef Py_ssize_t i, k, c, j
    cdef f64 a
    with nogil:
        for i in range(nrow):
            for k in range(rp[i], rp[i+1]):
                a=av[k]; j=ci[k]
                for c in range(ncol): yv[i, c]=yv[i, c]+a*xv[j, c]
    return Y


def spmm(A, X):
    """A @ X for a dense block X, one pass over the nonzeros.

    The block primitive an incidence factored channel needs: it applies to a whole
    block without assembling the operator.
    """
    cdef CSRMatrix csr
    if isinstance(A, DualCSR): csr = (<DualCSR>A).csr
    elif isinstance(A, CSRMatrix): csr = A
    else: raise TypeError(f"Expected CSRMatrix or DualCSR, got {type(A)}")
    if not isinstance(X, np.ndarray): X = np.asarray(X)
    if X.ndim != 2: raise ValueError("spmm requires a 2-D block; use matvec for a vector")
    X = np.ascontiguousarray(X, dtype=np.float64)
    if csr.val_bits == 32:
        raise TypeError("spmm carries f64 blocks; convert the carrier or use matvec")
    return spmm_i64_f64(csr, X) if csr.idx_bits == 64 else spmm_i32_f64(csr, X)


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
    cdef const i32[::1] cp=A._col_ptr_arr, ri=A._row_idx_arr
    cdef const f64[::1] cv=A._vals_csc_arr, xv=x
    cdef f64[::1] yv=y
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
    cdef const i64[::1] cp=A._col_ptr_arr, ri=A._row_idx_arr
    cdef const f64[::1] cv=A._vals_csc_arr, xv=x
    cdef f64[::1] yv=y
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
    cdef const i32[::1] cp=A._col_ptr_arr, ri=A._row_idx_arr
    cdef const f32[::1] cv=A._vals_csc_arr, xv=x
    cdef f32[::1] yv=y
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
    cdef const i64[::1] cp=A._col_ptr_arr, ri=A._row_idx_arr
    cdef const f32[::1] cv=A._vals_csc_arr, xv=x
    cdef f32[::1] yv=y
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
def rspmm_i32_f64(DualCSR A, np.ndarray[f64, ndim=2] X):
    cdef Py_ssize_t ncol=A.csr.ncol, w=X.shape[1]
    cdef np.ndarray[f64, ndim=2] Y = np.zeros((ncol, w), dtype=np.float64)
    cdef const i32[::1] cp=A._col_ptr_arr, ri=A._row_idx_arr
    cdef const f64[::1] cv=A._vals_csc_arr
    cdef const f64[:, ::1] xv=X
    cdef f64[:, ::1] yv=Y
    cdef Py_ssize_t j, k, c, i
    cdef f64 a
    with nogil:
        for j in range(ncol):
            for k in range(cp[j], cp[j+1]):
                a=cv[k]; i=ri[k]
                for c in range(w): yv[j, c]=yv[j, c]+a*xv[i, c]
    return Y


@cython.boundscheck(False)
@cython.wraparound(False)
def rspmm_i64_f64(DualCSR A, np.ndarray[f64, ndim=2] X):
    cdef Py_ssize_t ncol=A.csr.ncol, w=X.shape[1]
    cdef np.ndarray[f64, ndim=2] Y = np.zeros((ncol, w), dtype=np.float64)
    cdef const i64[::1] cp=A._col_ptr_arr, ri=A._row_idx_arr
    cdef const f64[::1] cv=A._vals_csc_arr
    cdef const f64[:, ::1] xv=X
    cdef f64[:, ::1] yv=Y
    cdef Py_ssize_t j, k, c, i
    cdef f64 a
    with nogil:
        for j in range(ncol):
            for k in range(cp[j], cp[j+1]):
                a=cv[k]; i=ri[k]
                for c in range(w): yv[j, c]=yv[j, c]+a*xv[i, c]
    return Y


def rspmm(A, X):
    """A^T @ X for a dense block, walking the dual's CSC arrays.

    The transpose half of `spmm`. Together they apply an incidence factored channel
    to a whole block in two passes over the nonzeros, with no assembled operator.
    """
    if not isinstance(A, DualCSR):
        raise TypeError(f"rspmm needs a DualCSR for the transpose direction, got {type(A)}")
    if not isinstance(X, np.ndarray): X = np.asarray(X)
    if X.ndim != 2: raise ValueError("rspmm requires a 2-D block; use rmatvec for a vector")
    X = np.ascontiguousarray(X, dtype=np.float64)
    cdef DualCSR d = <DualCSR>A
    if d.csr.val_bits == 32:
        raise TypeError("rspmm carries f64 blocks; convert the carrier or use rmatvec")
    return rspmm_i64_f64(d, X) if d.csr.idx_bits == 64 else rspmm_i32_f64(d, X)


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
    # ACCUMULATE, do not assign. A boundary column may carry two entries at the same
    # (row, col): a self loop stores -1 and +1 at its single vertex, and its boundary
    # is their sum, zero. Assigning let the second entry overwrite the first, so the
    # dense form showed a spurious +1 (a witness column) where the sparse operator the
    # kernels read has a cancelling pair. Summing duplicates is also what scipy's own
    # coo -> dense does. Structures without duplicates are unaffected: += equals = there.
    if csr.idx_bits == 32:
        rp = csr._row_ptr_arr
        ci = csr._col_idx_arr
        for i in range(csr.nrow):
            for k in range(rp[i], rp[i+1]): Dv[i, ci[k]] += av[k]
    else:
        rp64 = csr._row_ptr_arr
        ci64 = csr._col_idx_arr
        for i in range(csr.nrow):
            for k in range(rp64[i], rp64[i+1]): Dv[i, ci64[k]] += av[k]
    return D

def from_dense_f64(np.ndarray[f64, ndim=2] D, double tol=-1.0):
    """Convert a dense f64 matrix to DualCSR. Drops entries with |v| <= tol.

    **The default drops exact zeros only.** Support is structural: the nonzero pattern
    of a boundary tensor is its arity, its degree and its local parity, so dropping a
    small coefficient deletes a relation's participation rather than denoising storage.

    A caller that wants a threshold passes one, as the binary incidence conversion in
    `_boundary` does with `tol=0.5`.
    """
    if tol < 0.0:
        tol = 0.0
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

def canonical_dual(DualCSR A):
    """Coalesce sorted duplicate addresses and remove numerical zeros."""
    cdef Py_ssize_t i, p, j
    cdef double total
    cdef const i64[::1] rp = np.ascontiguousarray(A.row_ptr, dtype=np.int64)
    cdef const i64[::1] ci = np.ascontiguousarray(A.col_idx, dtype=np.int64)
    cdef const f64[::1] av = np.ascontiguousarray(A.vals, dtype=np.float64)
    rows, cols, values = [], [], []
    for i in range(A.nrow):
        p = rp[i]
        while p < rp[i + 1]:
            j, total = ci[p], 0.0
            while p < rp[i + 1] and ci[p] == j:
                total += av[p]
                p += 1
            if total != 0:
                rows.append(i)
                cols.append(j)
                values.append(total)
    if len(values) == A.nnz:
        return A
    return dual_from_coo(np.asarray(rows, dtype=np.int64),
                         np.asarray(cols, dtype=np.int64),
                         np.asarray(values, dtype=np.float64), A.nrow, A.ncol)


def csr_product(DualCSR A, DualCSR B):
    """Sparse row accumulation for A B, retaining no dense matrix workspace."""
    if A.ncol != B.nrow:
        raise ValueError("sparse product axes do not match")
    cdef Py_ssize_t i, p, q, k, j
    cdef double value
    cdef dict row
    cdef const i64[::1] arp = np.ascontiguousarray(A.row_ptr, dtype=np.int64)
    cdef const i64[::1] aci = np.ascontiguousarray(A.col_idx, dtype=np.int64)
    cdef const f64[::1] av = np.ascontiguousarray(A.vals, dtype=np.float64)
    cdef const i64[::1] brp = np.ascontiguousarray(B.row_ptr, dtype=np.int64)
    cdef const i64[::1] bci = np.ascontiguousarray(B.col_idx, dtype=np.int64)
    cdef const f64[::1] bv = np.ascontiguousarray(B.vals, dtype=np.float64)
    rows, cols, values = [], [], []
    for i in range(A.nrow):
        row = {}
        for p in range(arp[i], arp[i + 1]):
            k = aci[p]
            for q in range(brp[k], brp[k + 1]):
                j = bci[q]
                value = row.get(j, 0.0)
                row[j] = value + av[p] * bv[q]
        for j in sorted(row):
            value = row[j]
            if value != 0:
                rows.append(i)
                cols.append(j)
                values.append(value)
    return dual_from_coo(np.asarray(rows, dtype=np.int64),
                         np.asarray(cols, dtype=np.int64),
                         np.asarray(values, dtype=np.float64), A.nrow, B.ncol)


def csr_hadamard_rows(DualCSR A, DualCSR B):
    """Row sums of A elementwise B for canonical sparse rows."""
    if A.nrow != B.nrow or A.ncol != B.ncol:
        raise ValueError("sparse contraction shapes do not match")
    cdef Py_ssize_t i, p, q
    cdef const i64[::1] arp = np.ascontiguousarray(A.row_ptr, dtype=np.int64)
    cdef const i64[::1] aci = np.ascontiguousarray(A.col_idx, dtype=np.int64)
    cdef const f64[::1] av = np.ascontiguousarray(A.vals, dtype=np.float64)
    cdef const i64[::1] brp = np.ascontiguousarray(B.row_ptr, dtype=np.int64)
    cdef const i64[::1] bci = np.ascontiguousarray(B.col_idx, dtype=np.int64)
    cdef const f64[::1] bv = np.ascontiguousarray(B.vals, dtype=np.float64)
    cdef np.ndarray[f64, ndim=1] out = np.zeros(A.nrow, dtype=np.float64)
    for i in range(A.nrow):
        p, q = arp[i], brp[i]
        while p < arp[i + 1] and q < brp[i + 1]:
            if aci[p] == bci[q]:
                out[i] += av[p] * bv[q]
                p += 1
                q += 1
            elif aci[p] < bci[q]:
                p += 1
            else:
                q += 1
    return out


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



def connected_components(indptr, indices, Py_ssize_t n=-1):
    """Component label per row of a CSR pattern, and the component count.

    Union find over the stored pattern, using `UnionFind` from `_common.pxd`.

    Labels are renumbered in first appearance order, so they index densely from zero.
    Only the pattern is read: for a graph Laplacian these components are its kernel
    directions.

    Returns (labels int64[n], count).
    """
    cdef np.ndarray[i64, ndim=1] ptr = np.ascontiguousarray(indptr, dtype=np.int64)
    cdef np.ndarray[i64, ndim=1] idx = np.ascontiguousarray(indices, dtype=np.int64)
    if n < 0:
        n = ptr.shape[0] - 1
    cdef np.ndarray[i64, ndim=1] labels = np.empty(n, dtype=np.int64)
    if n == 0:
        return labels, 0

    cdef const i64[::1] pv = ptr
    cdef const i64[::1] iv = idx
    cdef i64[::1] lv = labels
    cdef Py_ssize_t row, k
    cdef i64 col
    cdef UnionFind uf
    cdef UnionFind64 uf64
    # `UnionFind` stores i32 parents, so past INT32_MAX rows the cast would wrap and
    # merge unrelated components. `_common.pxd` carries the i64 variant for that.
    cdef bint wide = n > MAX_INT32_NNZ
    if wide:
        if uf64_init(&uf64, n) != ERR_SUCCESS:
            raise MemoryError("union find allocation failed")
        try:
            with nogil:
                for row in range(n):
                    for k in range(pv[row], pv[row + 1]):
                        col = iv[k]
                        if col != row and col >= 0 and col < n:
                            uf64_union(&uf64, <i64>row, col)
                for row in range(n):
                    lv[row] = uf64_find(&uf64, <i64>row)
            count = int(uf64.n_components)
        finally:
            uf64_free(&uf64)
    else:
        if uf_init(&uf, n) != ERR_SUCCESS:
            raise MemoryError("union find allocation failed")
        try:
            with nogil:
                for row in range(n):
                    for k in range(pv[row], pv[row + 1]):
                        col = iv[k]
                        if col != row and col >= 0 and col < n:
                            uf_union(&uf, <i32>row, <i32>col)
                for row in range(n):
                    lv[row] = <i64>uf_find(&uf, <i32>row)
            count = int(uf_component_count(&uf))
        finally:
            uf_free(&uf)

    # Renumber roots to dense `0..count-1` in first appearance order.
    remap = {}
    cdef Py_ssize_t i
    for i in range(n):
        root = labels[i]
        slot = remap.get(root)
        if slot is None:
            slot = len(remap)
            remap[root] = slot
        labels[i] = slot
    return labels, count


def support_jaccard(a, b, *, exact=True):
    """Jaccard between two supports, exactly by default.

    `|A and B| / |A or B|` is a ratio of two counts, so it is a rational number and this
    returns it as one. `exact=False` gives the float reading. Both run the single merge
    pass of `sorted_jaccard_exact_i32` in `_common.pxd`; only the last step differs.

    The inputs are supports, so they are treated as sets: sorted here, and repeated
    coordinates counted once.
    """
    from fractions import Fraction

    cdef np.ndarray[i32, ndim=1] av = np.sort(
        np.ascontiguousarray(a, dtype=np.int32))
    cdef np.ndarray[i32, ndim=1] bv = np.sort(
        np.ascontiguousarray(b, dtype=np.int32))
    cdef idx_t inter = 0, uni = 0
    cdef const i32* ap = NULL
    cdef const i32* bp = NULL
    if av.shape[0]:
        ap = &av[0]
    if bv.shape[0]:
        bp = &bv[0]
    sorted_jaccard_exact_i32(ap, av.shape[0], bp, bv.shape[0], &inter, &uni)
    if uni == 0:
        return Fraction(0) if exact else 0.0
    return Fraction(int(inter), int(uni)) if exact else float(inter) / float(uni)
