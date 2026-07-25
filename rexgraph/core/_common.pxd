# rexgraph/core/_common.pxd
# cython: language_level=3
"""
Shared declarations for the rexgraph Cython layer.

Every .pyx module in rexgraph.core/ cimports from this file
to get consistent types, error codes, memory limits, and inline
utilities.

Export rules:
    Anonymous cdef enum values are exportable and work in nogil contexts.
    cdef inline functions are exportable.
    ctypedef aliases are exportable.

Usage:

    from rexgraph.core._common cimport (
        idx_t, i32, i64, f32, f64,
        ERR_SUCCESS, ERR_MEMORY,
        get_max_dense_allocation_bytes,
        should_parallelize,
        get_num_threads,
    )
"""

from libc.stdlib cimport malloc, calloc, realloc, free
from libc.string cimport memset, memcpy, memmove
from libc.stdint cimport (
    int8_t, int16_t, int32_t, int64_t,
    uint8_t, uint16_t, uint32_t, uint64_t,
)
from libc.math cimport sqrt, fabs, log, log2, exp, pow, floor, ceil, isnan, isinf

cimport cython


# OpenMP detection via C preprocessor
#
# _OPENMP is defined by compliant C compilers when OpenMP is enabled.
# We expose it as a C-level constant; the compiler eliminates dead branches.

cdef extern from *:
    """
    #ifdef _OPENMP
      #define REXGRAPH_HAS_OPENMP 1
      #include <omp.h>
    #else
      #define REXGRAPH_HAS_OPENMP 0
      static int omp_get_max_threads(void) { return 1; }
      static int omp_get_num_threads(void) { return 1; }
      static int omp_get_thread_num(void) { return 0; }
      static void omp_set_num_threads(int n) { (void)n; }
    #endif
    """
    const bint REXGRAPH_HAS_OPENMP
    int omp_get_max_threads() nogil
    int omp_get_num_threads() nogil
    int omp_get_thread_num() nogil
    void omp_set_num_threads(int num_threads) nogil

# prange compiles without OpenMP (runs serially); unconditional import
# avoids deprecated IF blocks while preserving the symbol for typed loops.
from cython.parallel cimport prange


# Error codes

cdef enum:
    ERR_SUCCESS        = 0
    ERR_MEMORY         = -1
    ERR_INVALID_ARG    = -2
    ERR_OUT_OF_BOUNDS  = -3
    ERR_OVERFLOW       = -4
    ERR_SHAPE_MISMATCH = -5
    ERR_NOT_CONVERGED  = -6
    ERR_SINGULAR       = -7
    ERR_MEMORY_LIMIT   = -8
    ERR_CANCELLED      = -9


# Integer limits

cdef enum:
    ABSOLUTE_MIN_PARALLEL_THRESHOLD = 1000
    ABSOLUTE_MAX_THREADS = 1024
    MAX_INT32_NNZ = 2147483647

cdef enum:
    INT32_MAX = 2147483647
    INT32_MIN = -2147483648
    UINT32_MAX = 4294967295

cdef enum:
    HASH_EMPTY_SLOT = -1
    HASH_DELETED_SLOT = -2


# Large constants (inline getters)

cdef inline int64_t get_INT64_MAX() noexcept nogil:
    return 9223372036854775807

cdef inline int64_t get_INT64_MIN() noexcept nogil:
    return -9223372036854775807 - 1

cdef inline uint64_t get_UINT64_MAX() noexcept nogil:
    return 18446744073709551615ULL

cdef inline int64_t get_ABSOLUTE_MAX_PARALLEL_BUFFER_BYTES() noexcept nogil:
    return 68719476736  # 64 GB

cdef inline uint64_t get_FNV_OFFSET_64() noexcept nogil:
    return 14695981039346656037ULL

cdef inline uint64_t get_FNV_PRIME_64() noexcept nogil:
    return 1099511628211ULL

cdef inline double get_EPSILON_NORM() noexcept nogil:
    return 1e-10

cdef inline double get_EPSILON_DIV() noexcept nogil:
    return 1e-12

cdef inline double get_EPSILON_EQ() noexcept nogil:
    return 1e-12

cdef inline double get_PI() noexcept nogil:
    return 3.141592653589793


# Type aliases

ctypedef int32_t  idx32_t
ctypedef int64_t  idx64_t
ctypedef Py_ssize_t idx_t

ctypedef int32_t  i32
ctypedef int64_t  i64
ctypedef float    f32
ctypedef double   f64


# Runtime configuration accessors (implemented in _common.pyx)
#
# Memory limits
#

cdef int64_t get_max_parallel_buffer_bytes() noexcept nogil
cdef int64_t get_max_total_allocation_bytes() noexcept nogil
cdef int64_t get_max_dense_allocation_bytes() noexcept nogil

#
# Parallelization thresholds
#

cdef idx_t get_min_parallel_threshold_simple() noexcept nogil
cdef idx_t get_min_parallel_threshold_transpose() noexcept nogil
cdef idx_t get_min_parallel_threshold_reduction() noexcept nogil

#
# Thread configuration
#

cdef int get_max_threads_limit() noexcept nogil
cdef int get_reserved_threads() noexcept nogil

#
# Algorithm selection (Laplacians / eigensolvers)
#

cdef Py_ssize_t get_eigen_dense_limit() noexcept nogil
cdef Py_ssize_t get_default_k() noexcept nogil
cdef double     get_fill_ratio_dense_threshold() noexcept nogil


# Thread helpers

cdef inline int get_num_threads(int requested) noexcept nogil:
    """Return the effective thread count to use.

    When OpenMP is available, applies the configured limits
    (max_threads_limit, reserved_threads) and returns the lesser
    of the requested and available counts.  Without OpenMP, returns 1.
    """
    if not REXGRAPH_HAS_OPENMP:
        return 1

    cdef int available = omp_get_max_threads()
    cdef int limit = get_max_threads_limit()
    cdef int reserved = get_reserved_threads()

    if available <= 0:
        available = 1
    if limit > 0 and available > limit:
        available = limit
    if reserved > 0 and available > reserved:
        available = available - reserved
    if available < 1:
        available = 1

    if requested <= 0:
        return available
    if requested < available:
        return requested
    return available


cdef inline int get_thread_id() noexcept nogil:
    """Return current thread ID (0 if no OpenMP)."""
    if REXGRAPH_HAS_OPENMP:
        return omp_get_thread_num()
    return 0


# Parallelization decision helpers

cdef inline bint should_parallelize(idx_t work_size, idx_t threshold) noexcept nogil:
    """True if OpenMP is available and work_size justifies parallelization."""
    if not REXGRAPH_HAS_OPENMP:
        return False
    if threshold < ABSOLUTE_MIN_PARALLEL_THRESHOLD:
        threshold = ABSOLUTE_MIN_PARALLEL_THRESHOLD
    return work_size >= threshold


cdef inline bint should_parallelize_with_memory(
    idx_t work_size,
    idx_t threshold,
    int64_t memory_required
) noexcept nogil:
    """Parallelize only if work_size >= threshold AND memory fits."""
    if not REXGRAPH_HAS_OPENMP:
        return False
    if threshold < ABSOLUTE_MIN_PARALLEL_THRESHOLD:
        threshold = ABSOLUTE_MIN_PARALLEL_THRESHOLD
    if work_size < threshold:
        return False
    if memory_required > get_max_parallel_buffer_bytes():
        return False
    return True


# Dense allocation decision helpers

cdef inline bint can_allocate_dense_f64(Py_ssize_t nrows, Py_ssize_t ncols) noexcept nogil:
    """True if an nrows x ncols float64 matrix fits within max_dense_allocation_bytes."""
    cdef int64_t mem = <int64_t>nrows * <int64_t>ncols * 8
    return mem <= get_max_dense_allocation_bytes()


cdef inline bint should_use_dense_eigen(Py_ssize_t n) noexcept nogil:
    """True if dense eigh (LAPACK dsyevd) should be used for an n x n matrix.

    Based on matrix dimension vs eigen_dense_limit.
    """
    return n <= get_eigen_dense_limit()


cdef inline bint should_use_dense_matmul(Py_ssize_t n_out) noexcept nogil:
    """True if matmul should produce a dense n_out x n_out output.

    Requires both (a) output fits in max_dense_allocation_bytes AND
    (b) dimension is within the eigen dense limit (no point building
    dense if the eigensolver will use sparse anyway).
    """
    if n_out > get_eigen_dense_limit():
        return False
    return can_allocate_dense_f64(n_out, n_out)


# Memory helpers

cdef inline void safe_free(void* ptr) noexcept nogil:
    """Free wrapper - free(NULL) is a no-op per C standard."""
    if ptr != NULL:
        free(ptr)


cdef inline int64_t compute_parallel_buffer_memory(
    int threads, idx_t elements, idx_t element_size
) noexcept nogil:
    """Compute memory for threads x elements x element_size.

    Returns -1 on overflow.
    """
    cdef int64_t t = <int64_t>threads
    cdef int64_t e = <int64_t>elements
    cdef int64_t s = <int64_t>element_size
    cdef int64_t INT64_MAX_VAL = get_INT64_MAX()

    if t <= 0 or e <= 0 or s <= 0:
        return 0
    if t > INT64_MAX_VAL // e:
        return -1
    cdef int64_t te = t * e
    if te > INT64_MAX_VAL // s:
        return -1
    return te * s


# Numeric helpers

cdef inline bint is_near_zero(double x, double eps) noexcept nogil:
    """Return True if |x| <= eps."""
    return fabs(x) <= eps


cdef inline double clamp(double x, double lo, double hi) noexcept nogil:
    """Clamp x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


cdef inline float clamp_f32(float x, float lo, float hi) noexcept nogil:
    """Clamp float x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


cdef inline int64_t clamp_i64(int64_t x, int64_t lo, int64_t hi) noexcept nogil:
    """Clamp int64 x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


cdef inline double sanitize_float64(double x) noexcept nogil:
    """Replace NaN/Inf with 0.0."""
    if isnan(x) or isinf(x):
        return 0.0
    return x


cdef inline float sanitize_float32(float x) noexcept nogil:
    """Replace NaN/Inf with 0.0f."""
    if isnan(x) or isinf(x):
        return 0.0
    return x


cdef inline double log_clamp_min(double x, double min_x) noexcept nogil:
    """Compute log(max(x, min_x))."""
    if x < min_x:
        x = min_x
    return log(x)


cdef inline double sqrt_clamp_min(double x) noexcept nogil:
    """Compute sqrt(max(x, 0))."""
    if x < 0.0:
        x = 0.0
    return sqrt(x)


# Bit operations

cdef inline int popcount_u64(uint64_t x) noexcept nogil:
    """Population count for uint64 - portable implementation."""
    cdef int count = 0
    while x:
        count += 1
        x &= x - 1
    return count


cdef inline idx_t next_power_of_two(idx_t x) noexcept nogil:
    """Smallest power of two >= x, minimum 1."""
    cdef idx_t p
    if x <= 1:
        return 1
    p = 1
    while p < x:
        p <<= 1
    return p


cdef inline bint is_power_of_two(idx_t x) noexcept nogil:
    """True iff x is a power of two."""
    return x > 0 and (x & (x - 1)) == 0


# Hash operations

cdef inline uint64_t fnv1a_hash_u64(uint64_t x) noexcept nogil:
    """FNV-1a hash for a uint64 (bytewise)."""
    cdef uint64_t h = get_FNV_OFFSET_64()
    cdef uint64_t prime = get_FNV_PRIME_64()
    cdef int i
    cdef uint8_t* bytes = <uint8_t*>&x
    for i in range(8):
        h ^= <uint64_t>bytes[i]
        h *= prime
    return h


cdef inline uint64_t mix64(uint64_t x) noexcept nogil:
    """SplitMix64 finalizer."""
    x ^= x >> 30
    x *= <uint64_t>0xBF58476D1CE4E5B9
    x ^= x >> 27
    x *= <uint64_t>0x94D049BB133111EB
    x ^= x >> 31
    return x


# CSR utilities

cdef inline bint csr_needs_int64_indptr(int64_t nnz) noexcept nogil:
    """Return True if nnz exceeds int32 capacity."""
    return nnz > MAX_INT32_NNZ


cdef inline int64_t csr_row_length_i32(const int32_t* indptr, idx_t row) noexcept nogil:
    """Get row length from int32 CSR indptr."""
    return <int64_t>(indptr[row + 1] - indptr[row])


cdef inline int64_t csr_row_length_i64(const int64_t* indptr, idx_t row) noexcept nogil:
    """Get row length from int64 CSR indptr."""
    return indptr[row + 1] - indptr[row]


# Union-find (disjoint set union)

cdef struct UnionFind:
    i32* parent
    i32* rnk
    idx_t n
    idx_t n_components


cdef inline int uf_init(UnionFind* uf, idx_t n) noexcept nogil:
    """Initialize union-find for n elements.

    Returns ERR_SUCCESS or ERR_MEMORY.
    """
    cdef idx_t i
    uf.n = n
    uf.n_components = n
    uf.parent = <i32*>malloc(n * sizeof(i32))
    uf.rnk = <i32*>calloc(n, sizeof(i32))

    if uf.parent == NULL or uf.rnk == NULL:
        if uf.parent != NULL:
            free(uf.parent)
        if uf.rnk != NULL:
            free(uf.rnk)
        uf.parent = NULL
        uf.rnk = NULL
        return ERR_MEMORY

    for i in range(n):
        uf.parent[i] = <i32>i
    return ERR_SUCCESS


cdef inline void uf_free(UnionFind* uf) noexcept nogil:
    """Free union-find memory."""
    if uf.parent != NULL:
        free(uf.parent)
        uf.parent = NULL
    if uf.rnk != NULL:
        free(uf.rnk)
        uf.rnk = NULL
    uf.n = 0
    uf.n_components = 0


cdef inline i32 uf_find(UnionFind* uf, i32 x) noexcept nogil:
    """Find root with path compression."""
    cdef i32 root = x
    cdef i32 tmp

    while uf.parent[root] != root:
        root = uf.parent[root]
    while uf.parent[x] != root:
        tmp = uf.parent[x]
        uf.parent[x] = root
        x = tmp
    return root


cdef inline bint uf_union(UnionFind* uf, i32 x, i32 y) noexcept nogil:
    """Union two sets. Returns True if they were previously separate."""
    cdef i32 rx = uf_find(uf, x)
    cdef i32 ry = uf_find(uf, y)

    if rx == ry:
        return False

    if uf.rnk[rx] < uf.rnk[ry]:
        uf.parent[rx] = ry
    elif uf.rnk[rx] > uf.rnk[ry]:
        uf.parent[ry] = rx
    else:
        uf.parent[ry] = rx
        uf.rnk[rx] += 1

    uf.n_components -= 1
    return True


cdef inline idx_t uf_component_count(UnionFind* uf) noexcept nogil:
    """Return number of connected components."""
    return uf.n_components


cdef inline bint uf_connected(UnionFind* uf, i32 x, i32 y) noexcept nogil:
    """Check if x and y are in the same component."""
    return uf_find(uf, x) == uf_find(uf, y)


# int64 variant for large graphs (> 2B nodes)

cdef struct UnionFind64:
    i64* parent
    i64* rnk
    idx_t n
    idx_t n_components


cdef inline int uf64_init(UnionFind64* uf, idx_t n) noexcept nogil:
    """Initialize int64 union-find for n elements."""
    cdef idx_t i
    uf.n = n
    uf.n_components = n
    uf.parent = <i64*>malloc(n * sizeof(i64))
    uf.rnk = <i64*>calloc(n, sizeof(i64))

    if uf.parent == NULL or uf.rnk == NULL:
        if uf.parent != NULL:
            free(uf.parent)
        if uf.rnk != NULL:
            free(uf.rnk)
        uf.parent = NULL
        uf.rnk = NULL
        return ERR_MEMORY

    for i in range(n):
        uf.parent[i] = <i64>i
    return ERR_SUCCESS


cdef inline void uf64_free(UnionFind64* uf) noexcept nogil:
    """Free int64 union-find memory."""
    if uf.parent != NULL:
        free(uf.parent)
        uf.parent = NULL
    if uf.rnk != NULL:
        free(uf.rnk)
        uf.rnk = NULL
    uf.n = 0
    uf.n_components = 0


cdef inline i64 uf64_find(UnionFind64* uf, i64 x) noexcept nogil:
    """Find root with path compression (int64)."""
    cdef i64 root = x
    cdef i64 tmp

    while uf.parent[root] != root:
        root = uf.parent[root]
    while uf.parent[x] != root:
        tmp = uf.parent[x]
        uf.parent[x] = root
        x = tmp
    return root


cdef inline bint uf64_union(UnionFind64* uf, i64 x, i64 y) noexcept nogil:
    """Union two sets (int64). Returns True if previously separate."""
    cdef i64 rx = uf64_find(uf, x)
    cdef i64 ry = uf64_find(uf, y)

    if rx == ry:
        return False

    if uf.rnk[rx] < uf.rnk[ry]:
        uf.parent[rx] = ry
    elif uf.rnk[rx] > uf.rnk[ry]:
        uf.parent[ry] = rx
    else:
        uf.parent[ry] = rx
        uf.rnk[rx] += 1

    uf.n_components -= 1
    return True


# Sorted array set operations

cdef inline idx_t sorted_intersection_count_i32(
    const i32* a, idx_t len_a,
    const i32* b, idx_t len_b
) noexcept nogil:
    """Count elements in intersection of two sorted int32 arrays."""
    cdef idx_t i = 0, j = 0
    cdef idx_t count = 0

    while i < len_a and j < len_b:
        if a[i] < b[j]:
            i += 1
        elif a[i] > b[j]:
            j += 1
        else:
            count += 1
            i += 1
            j += 1
    return count


cdef inline idx_t sorted_union_count_i32(
    const i32* a, idx_t len_a,
    const i32* b, idx_t len_b
) noexcept nogil:
    """Count elements in union of two sorted int32 arrays."""
    cdef idx_t i = 0, j = 0
    cdef idx_t count = 0

    while i < len_a and j < len_b:
        if a[i] < b[j]:
            i += 1
        elif a[i] > b[j]:
            j += 1
        else:
            i += 1
            j += 1
        count += 1

    count += (len_a - i) + (len_b - j)
    return count


cdef inline double sorted_jaccard_i32(
    const i32* a, idx_t len_a,
    const i32* b, idx_t len_b
) noexcept nogil:
    """Compute Jaccard similarity between two sorted int32 arrays."""
    cdef idx_t intersection = 0
    cdef idx_t union_size = 0
    cdef idx_t i = 0, j = 0

    while i < len_a and j < len_b:
        if a[i] < b[j]:
            i += 1
        elif a[i] > b[j]:
            j += 1
        else:
            intersection += 1
            i += 1
            j += 1
        union_size += 1

    union_size += (len_a - i) + (len_b - j)

    if union_size == 0:
        return 0.0
    return <double>intersection / <double>union_size


cdef inline idx_t sorted_intersection_write_i32(
    const i32* a, idx_t len_a,
    const i32* b, idx_t len_b,
    i32* out, idx_t max_out
) noexcept nogil:
    """Write intersection of two sorted arrays to out.

    Returns count written.
    """
    cdef idx_t i = 0, j = 0
    cdef idx_t count = 0

    while i < len_a and j < len_b and count < max_out:
        if a[i] < b[j]:
            i += 1
        elif a[i] > b[j]:
            j += 1
        else:
            out[count] = a[i]
            count += 1
            i += 1
            j += 1
    return count


# Binary search utilities

cdef inline idx_t binary_search_i32(const i32* arr, idx_t n, i32 target) noexcept nogil:
    """Binary search in sorted array. Returns index or -1 if not found."""
    cdef idx_t lo = 0
    cdef idx_t hi = n
    cdef idx_t mid

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] < target:
            lo = mid + 1
        elif arr[mid] > target:
            hi = mid
        else:
            return mid
    return -1


cdef inline bint binary_search_contains_i32(const i32* arr, idx_t n, i32 target) noexcept nogil:
    """Check if sorted array contains target."""
    return binary_search_i32(arr, n, target) >= 0


cdef inline idx_t lower_bound_i32(const i32* arr, idx_t n, i32 target) noexcept nogil:
    """Find first index where arr[i] >= target."""
    cdef idx_t lo = 0
    cdef idx_t hi = n
    cdef idx_t mid

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


cdef inline idx_t upper_bound_i32(const i32* arr, idx_t n, i32 target) noexcept nogil:
    """Find first index where arr[i] > target."""
    cdef idx_t lo = 0
    cdef idx_t hi = n
    cdef idx_t mid

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo
