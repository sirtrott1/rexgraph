# rexgraph/core/_common.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
Shared runtime utilities for the rexgraph Cython layer.

Provides:
    Runtime configuration for memory limits and parallelization thresholds.
    Algorithm selection parameters (dense/sparse crossover, eigensolver).
    System memory detection (psutil, sysconf, or 8 GB fallback).
    Python-accessible feature detection and diagnostics.
    Error handling and validation helpers.
    Memory estimation.

Configuration is system-aware by default and runtime-reconfigurable:

    from rexgraph.core._common import configure_memory, configure_algorithms
    configure_memory(max_dense_allocation=8_000_000_000)
    configure_algorithms(eigen_dense_limit=5000, default_k=30)

Or via environment variables (applied at import time):

    REXGRAPH_MAX_DENSE_GB=4
    REXGRAPH_EIGEN_DENSE_LIMIT=3000
    REXGRAPH_DEFAULT_K=20
"""

from __future__ import annotations

cimport cython
import os
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport fabs
from libc.stdint cimport int32_t, int64_t, uint64_t

# DO NOT cimport from rexgraph.core._common - it's THIS module!
# The .pxd file declarations are automatically available.

np.import_array()


# Runtime configuration state
#
# Memory limits
#

cdef int64_t _max_parallel_buffer_bytes = -1      # per-operation parallel scratch
cdef int64_t _max_total_allocation_bytes = -1     # global allocation ceiling
cdef int64_t _max_dense_allocation_bytes = -1     # single dense matrix ceiling
cdef double _parallel_buffer_fraction = 0.25
cdef double _total_allocation_fraction = 0.75
cdef double _dense_allocation_fraction = 0.25     # 25% of RAM for one dense matrix

#
# Parallelization thresholds
#

cdef idx_t _min_parallel_threshold_simple = 50000
cdef idx_t _min_parallel_threshold_transpose = 500000
cdef idx_t _min_parallel_threshold_reduction = 100000

#
# Thread management
#

cdef int _max_threads_limit = -1
cdef int _reserved_threads = 0

#
# Algorithm selection (Laplacians / eigensolvers)
#

cdef Py_ssize_t _eigen_dense_limit = 2000
cdef Py_ssize_t _default_k = 20
cdef double _fill_ratio_dense_threshold = 0.3

#
# Auto-detection state
#

cdef bint _auto_detected = False


# System memory detection

cdef int64_t _get_system_memory_bytes():
    """Get total system memory in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().total
    except Exception:
        pass

    try:
        pages = os.sysconf('SC_PHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (ValueError, OSError, AttributeError):
        pass

    return 8 * 1024 * 1024 * 1024  # 8 GB fallback


cdef int64_t _get_available_memory_bytes():
    """Get available memory in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        pass
    return _get_system_memory_bytes() // 2


cdef void _auto_detect_limits():
    """Auto-detect memory limits based on system resources.

    Called once at module import and again after any configure_*()
    call that resets limits to auto-detect (value = -1).
    """
    global _max_parallel_buffer_bytes, _max_total_allocation_bytes
    global _max_dense_allocation_bytes, _auto_detected

    if _auto_detected:
        return

    cdef int64_t total_mem = _get_system_memory_bytes()
    cdef int64_t abs_max = get_ABSOLUTE_MAX_PARALLEL_BUFFER_BYTES()

    if _max_parallel_buffer_bytes < 0:
        _max_parallel_buffer_bytes = <int64_t>(total_mem * _parallel_buffer_fraction)
        if _max_parallel_buffer_bytes > abs_max:
            _max_parallel_buffer_bytes = abs_max

    if _max_total_allocation_bytes < 0:
        _max_total_allocation_bytes = <int64_t>(total_mem * _total_allocation_fraction)

    if _max_dense_allocation_bytes < 0:
        # 25% of total RAM, clamped to [100 MB, 4 GB]
        _max_dense_allocation_bytes = <int64_t>(total_mem * _dense_allocation_fraction)
        if _max_dense_allocation_bytes < 100 * 1024 * 1024:
            _max_dense_allocation_bytes = 100 * 1024 * 1024       # floor: 100 MB
        if _max_dense_allocation_bytes > 4 * 1024 * 1024 * 1024:
            _max_dense_allocation_bytes = 4 * <int64_t>(1024 * 1024 * 1024)  # ceil: 4 GB

    _auto_detected = True


# Runtime configuration accessors

cdef int64_t get_max_parallel_buffer_bytes() noexcept nogil:
    if _max_parallel_buffer_bytes < 0:
        return 8 * 1024 * 1024 * 1024       # 8 GB fallback before init
    return _max_parallel_buffer_bytes


cdef int64_t get_max_total_allocation_bytes() noexcept nogil:
    if _max_total_allocation_bytes < 0:
        return 32 * <int64_t>(1024 * 1024 * 1024)
    return _max_total_allocation_bytes


cdef int64_t get_max_dense_allocation_bytes() noexcept nogil:
    if _max_dense_allocation_bytes < 0:
        return 2 * <int64_t>(1024 * 1024 * 1024)   # 2 GB fallback
    return _max_dense_allocation_bytes


cdef idx_t get_min_parallel_threshold_simple() noexcept nogil:
    if _min_parallel_threshold_simple < ABSOLUTE_MIN_PARALLEL_THRESHOLD:
        return ABSOLUTE_MIN_PARALLEL_THRESHOLD
    return _min_parallel_threshold_simple


cdef idx_t get_min_parallel_threshold_transpose() noexcept nogil:
    if _min_parallel_threshold_transpose < ABSOLUTE_MIN_PARALLEL_THRESHOLD:
        return ABSOLUTE_MIN_PARALLEL_THRESHOLD
    return _min_parallel_threshold_transpose


cdef idx_t get_min_parallel_threshold_reduction() noexcept nogil:
    if _min_parallel_threshold_reduction < ABSOLUTE_MIN_PARALLEL_THRESHOLD:
        return ABSOLUTE_MIN_PARALLEL_THRESHOLD
    return _min_parallel_threshold_reduction


cdef int get_max_threads_limit() noexcept nogil:
    return _max_threads_limit


cdef int get_reserved_threads() noexcept nogil:
    return _reserved_threads


cdef Py_ssize_t get_eigen_dense_limit() noexcept nogil:
    return _eigen_dense_limit


cdef Py_ssize_t get_default_k() noexcept nogil:
    return _default_k


cdef double get_fill_ratio_dense_threshold() noexcept nogil:
    return _fill_ratio_dense_threshold


# Python-accessible configuration functions

def configure_memory(
    *,
    max_parallel_buffer: int = None,
    max_total_allocation: int = None,
    max_dense_allocation: int = None,
    parallel_buffer_fraction: float = None,
    total_allocation_fraction: float = None,
    dense_allocation_fraction: float = None,
):
    """Configure memory limits for the core Cython layer.

    Parameters
    ----------
    max_parallel_buffer : int, optional
        Maximum bytes for per-operation parallel scratch buffers.
    max_total_allocation : int, optional
        Global allocation ceiling in bytes.
    max_dense_allocation : int, optional
        Maximum bytes for a single dense matrix.  Controls whether
        Laplacian construction uses dense spmm kernels or sparse scipy.
    parallel_buffer_fraction : float, optional
        Fraction of system RAM for parallel buffers (default 0.25).
    total_allocation_fraction : float, optional
        Fraction of system RAM for total allocation (default 0.75).
    dense_allocation_fraction : float, optional
        Fraction of system RAM for dense matrix allocation (default 0.25).
    """
    global _max_parallel_buffer_bytes, _max_total_allocation_bytes
    global _max_dense_allocation_bytes
    global _parallel_buffer_fraction, _total_allocation_fraction
    global _dense_allocation_fraction, _auto_detected

    cdef int64_t abs_max = get_ABSOLUTE_MAX_PARALLEL_BUFFER_BYTES()

    if parallel_buffer_fraction is not None:
        if not 0.0 < parallel_buffer_fraction <= 1.0:
            raise ValueError("parallel_buffer_fraction must be in (0, 1]")
        _parallel_buffer_fraction = parallel_buffer_fraction

    if total_allocation_fraction is not None:
        if not 0.0 < total_allocation_fraction <= 1.0:
            raise ValueError("total_allocation_fraction must be in (0, 1]")
        _total_allocation_fraction = total_allocation_fraction

    if dense_allocation_fraction is not None:
        if not 0.0 < dense_allocation_fraction <= 1.0:
            raise ValueError("dense_allocation_fraction must be in (0, 1]")
        _dense_allocation_fraction = dense_allocation_fraction

    if max_parallel_buffer is not None:
        if max_parallel_buffer < 0:
            raise ValueError("max_parallel_buffer must be non-negative")
        _max_parallel_buffer_bytes = min(max_parallel_buffer, abs_max)
    else:
        _max_parallel_buffer_bytes = -1   # re-detect

    if max_total_allocation is not None:
        if max_total_allocation < 0:
            raise ValueError("max_total_allocation must be non-negative")
        _max_total_allocation_bytes = max_total_allocation
    else:
        _max_total_allocation_bytes = -1  # re-detect

    if max_dense_allocation is not None:
        if max_dense_allocation < 0:
            raise ValueError("max_dense_allocation must be non-negative")
        _max_dense_allocation_bytes = max_dense_allocation
    else:
        _max_dense_allocation_bytes = -1  # re-detect

    _auto_detected = False
    _auto_detect_limits()


def configure_parallelization(
    *,
    min_simple: int = None,
    min_transpose: int = None,
    min_reduction: int = None,
):
    """Configure parallelization thresholds.

    Parameters
    ----------
    min_simple : int, optional
        Minimum work items for simple row-parallel loops (default 50000).
    min_transpose : int, optional
        Minimum work items for CSR transpose (default 500000).
    min_reduction : int, optional
        Minimum work items for reduction operations (default 100000).
    """
    global _min_parallel_threshold_simple
    global _min_parallel_threshold_transpose
    global _min_parallel_threshold_reduction

    if min_simple is not None:
        _min_parallel_threshold_simple = max(min_simple, ABSOLUTE_MIN_PARALLEL_THRESHOLD)

    if min_transpose is not None:
        _min_parallel_threshold_transpose = max(min_transpose, ABSOLUTE_MIN_PARALLEL_THRESHOLD)

    if min_reduction is not None:
        _min_parallel_threshold_reduction = max(min_reduction, ABSOLUTE_MIN_PARALLEL_THRESHOLD)


def configure_threads(*, max_threads: int = None, reserved_threads: int = None):
    """Configure thread management.

    Parameters
    ----------
    max_threads : int, optional
        Maximum threads to use. -1 = no limit (default).
    reserved_threads : int, optional
        Threads to reserve for the caller (default 0).
    """
    global _max_threads_limit, _reserved_threads

    if max_threads is not None:
        if max_threads < 0:
            _max_threads_limit = -1
        else:
            _max_threads_limit = min(max_threads, ABSOLUTE_MAX_THREADS)

    if reserved_threads is not None:
        if reserved_threads < 0:
            raise ValueError("reserved_threads must be non-negative")
        _reserved_threads = reserved_threads


def configure_algorithms(
    *,
    eigen_dense_limit: int = None,
    default_k: int = None,
    fill_ratio_dense_threshold: float = None,
):
    """Configure algorithm selection for Laplacians and eigensolvers.

    Parameters
    ----------
    eigen_dense_limit : int, optional
        Maximum matrix dimension for dense eigh (LAPACK dsyevd).
        Above this, sparse eigsh (ARPACK) is used.
        Default: 2000 (eigh takes ~0.4s at n=2000, ~20s at n=5000).
    default_k : int, optional
        Number of eigenvalues to compute in sparse mode (default 20).
    fill_ratio_dense_threshold : float, optional
        L0 fill ratio above which dense matmul is preferred (default 0.3).
    """
    global _eigen_dense_limit, _default_k, _fill_ratio_dense_threshold

    if eigen_dense_limit is not None:
        if eigen_dense_limit < 1:
            raise ValueError("eigen_dense_limit must be >= 1")
        _eigen_dense_limit = eigen_dense_limit

    if default_k is not None:
        if default_k < 1:
            raise ValueError("default_k must be >= 1")
        _default_k = default_k

    if fill_ratio_dense_threshold is not None:
        if not 0.0 < fill_ratio_dense_threshold <= 1.0:
            raise ValueError("fill_ratio_dense_threshold must be in (0, 1]")
        _fill_ratio_dense_threshold = fill_ratio_dense_threshold


def configure_from_environment():
    """Configure all settings from environment variables.

    Recognized variables:

        REXGRAPH_MAX_PARALLEL_BUFFER_GB   (float, GB)
        REXGRAPH_MAX_TOTAL_ALLOCATION_GB  (float, GB)
        REXGRAPH_MAX_DENSE_GB             (float, GB)
        REXGRAPH_PARALLEL_BUFFER_FRACTION (float, 0-1)
        REXGRAPH_MIN_PARALLEL_SIMPLE      (int)
        REXGRAPH_MIN_PARALLEL_TRANSPOSE   (int)
        REXGRAPH_MAX_THREADS              (int)
        REXGRAPH_RESERVED_THREADS         (int)
        REXGRAPH_EIGEN_DENSE_LIMIT        (int)
        REXGRAPH_DEFAULT_K                (int)
        REXGRAPH_FILL_RATIO_THRESHOLD     (float, 0-1)

    Returns True if any environment variables were applied.
    """
    env_map = {
        'REXGRAPH_MAX_PARALLEL_BUFFER_GB': ('mem', 'max_parallel_buffer', lambda x: int(float(x) * 1024**3)),
        'REXGRAPH_MAX_TOTAL_ALLOCATION_GB': ('mem', 'max_total_allocation', lambda x: int(float(x) * 1024**3)),
        'REXGRAPH_MAX_DENSE_GB': ('mem', 'max_dense_allocation', lambda x: int(float(x) * 1024**3)),
        'REXGRAPH_PARALLEL_BUFFER_FRACTION': ('mem', 'parallel_buffer_fraction', float),
        'REXGRAPH_MIN_PARALLEL_SIMPLE': ('par', 'min_simple', int),
        'REXGRAPH_MIN_PARALLEL_TRANSPOSE': ('par', 'min_transpose', int),
        'REXGRAPH_MAX_THREADS': ('thread', 'max_threads', int),
        'REXGRAPH_RESERVED_THREADS': ('thread', 'reserved_threads', int),
        'REXGRAPH_EIGEN_DENSE_LIMIT': ('algo', 'eigen_dense_limit', int),
        'REXGRAPH_DEFAULT_K': ('algo', 'default_k', int),
        'REXGRAPH_FILL_RATIO_THRESHOLD': ('algo', 'fill_ratio_dense_threshold', float),
    }

    groups = {'mem': {}, 'par': {}, 'thread': {}, 'algo': {}}

    for env_key, (group, param_name, converter) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                groups[group][param_name] = converter(val)
            except Exception:
                pass

    if groups['mem']:
        configure_memory(**groups['mem'])
    if groups['par']:
        configure_parallelization(**groups['par'])
    if groups['thread']:
        configure_threads(**groups['thread'])
    if groups['algo']:
        configure_algorithms(**groups['algo'])

    return any(groups[g] for g in groups)


def get_configuration():
    """Get current configuration as a dictionary.

    Returns all runtime settings including system-detected defaults.
    """
    _auto_detect_limits()

    return {
        # Memory
        "max_parallel_buffer_bytes": int(_max_parallel_buffer_bytes),
        "max_parallel_buffer_gb": _max_parallel_buffer_bytes / (1024**3),
        "max_total_allocation_bytes": int(_max_total_allocation_bytes),
        "max_total_allocation_gb": _max_total_allocation_bytes / (1024**3),
        "max_dense_allocation_bytes": int(_max_dense_allocation_bytes),
        "max_dense_allocation_gb": _max_dense_allocation_bytes / (1024**3),
        # Parallelization
        "min_parallel_threshold_simple": int(_min_parallel_threshold_simple),
        "min_parallel_threshold_transpose": int(_min_parallel_threshold_transpose),
        "min_parallel_threshold_reduction": int(_min_parallel_threshold_reduction),
        # Threads
        "max_threads_limit": int(_max_threads_limit),
        "reserved_threads": int(_reserved_threads),
        # Algorithm selection
        "eigen_dense_limit": int(_eigen_dense_limit),
        "default_k": int(_default_k),
        "fill_ratio_dense_threshold": float(_fill_ratio_dense_threshold),
        # System
        "system_memory_bytes": int(_get_system_memory_bytes()),
        "system_memory_gb": _get_system_memory_bytes() / (1024**3),
        "available_memory_bytes": int(_get_available_memory_bytes()),
        "available_memory_gb": _get_available_memory_bytes() / (1024**3),
    }


def get_parallelization_config():
    """Get parallelization configuration."""
    return {
        'min_simple': int(_min_parallel_threshold_simple),
        'min_transpose': int(_min_parallel_threshold_transpose),
        'min_reduction': int(_min_parallel_threshold_reduction),
        'max_threads': int(_max_threads_limit),
        'openmp_enabled': get_openmp_enabled(),
    }


def get_algorithm_config():
    """Get algorithm selection configuration."""
    return {
        'eigen_dense_limit': int(_eigen_dense_limit),
        'default_k': int(_default_k),
        'fill_ratio_dense_threshold': float(_fill_ratio_dense_threshold),
        'max_dense_allocation_bytes': int(_max_dense_allocation_bytes),
        'max_dense_allocation_gb': _max_dense_allocation_bytes / (1024**3),
    }


# Feature detection

def get_openmp_enabled() -> bool:
    """Check if OpenMP was enabled at compile time."""
    return bool(REXGRAPH_HAS_OPENMP)


def get_debug_enabled() -> bool:
    """Check if debug mode was enabled at compile time.

    Debug mode is detected via the REXGRAPH_DEBUG environment variable
    at import time (since C preprocessor macros are fixed at compile time
    and we no longer use Cython DEF for this).
    """
    return os.environ.get('REXGRAPH_DEBUG', '0') not in ('0', '', 'false', 'False')


def get_max_threads() -> int:
    """Get maximum available threads."""
    if REXGRAPH_HAS_OPENMP:
        return omp_get_max_threads()
    return 1


def get_effective_threads(int requested=0) -> int:
    """Get effective thread count considering limits."""
    if not REXGRAPH_HAS_OPENMP:
        return 1

    cdef int available = omp_get_max_threads()
    cdef int limit = _max_threads_limit
    cdef int reserved = _reserved_threads

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
    return min(requested, available)


def get_build_info() -> dict:
    """Get build configuration information."""
    return {
        'openmp_enabled': get_openmp_enabled(),
        'debug_enabled': get_debug_enabled(),
        'max_threads': get_max_threads(),
        'compiled': True,
    }


# Error handling

class CoreError(Exception):
    """Base exception for core Cython layer errors."""
    pass


class CoreMemoryError(CoreError, MemoryError):
    """Memory allocation failed."""
    pass


class CoreMemoryLimitError(CoreMemoryError):
    """Memory limit exceeded."""
    pass


class CoreValueError(CoreError, ValueError):
    """Invalid value."""
    pass


class CoreOverflowError(CoreError, OverflowError):
    """Integer overflow."""
    pass


def raise_on_error(int code, str context=""):
    """Raise appropriate exception for error code."""
    if code == ERR_SUCCESS:
        return

    msg = f"{context}: " if context else ""

    if code == ERR_MEMORY:
        raise CoreMemoryError(f"{msg}Memory allocation failed")
    elif code == ERR_INVALID_ARG:
        raise CoreValueError(f"{msg}Invalid argument")
    elif code == ERR_OUT_OF_BOUNDS:
        raise CoreValueError(f"{msg}Index out of bounds")
    elif code == ERR_OVERFLOW:
        raise CoreOverflowError(f"{msg}Integer overflow")
    elif code == ERR_SHAPE_MISMATCH:
        raise CoreValueError(f"{msg}Shape mismatch")
    elif code == ERR_NOT_CONVERGED:
        raise CoreError(f"{msg}Algorithm did not converge")
    elif code == ERR_SINGULAR:
        raise CoreValueError(f"{msg}Singular or degenerate case")
    elif code == ERR_MEMORY_LIMIT:
        raise CoreMemoryLimitError(f"{msg}Memory limit exceeded")
    elif code == ERR_CANCELLED:
        raise CoreError(f"{msg}Operation cancelled")
    else:
        raise CoreError(f"{msg}Unknown error code: {code}")


def check_error(int code, str context=""):
    """Check error code and raise if non-zero."""
    raise_on_error(code, context)


# Validation

def validate_csr_arrays(indptr, indices, data=None, str name="CSR"):
    """Validate CSR array shapes and dtypes."""
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)

    if indptr.ndim != 1:
        raise CoreValueError(f"{name}: indptr must be 1D")
    if indices.ndim != 1:
        raise CoreValueError(f"{name}: indices must be 1D")
    if len(indptr) < 1:
        raise CoreValueError(f"{name}: indptr must have at least 1 element")
    if indptr[0] != 0:
        raise CoreValueError(f"{name}: indptr[0] must be 0")
    if len(indices) != indptr[-1]:
        raise CoreValueError(f"{name}: indices length must equal indptr[-1]")

    if data is not None:
        data = np.asarray(data)
        if data.ndim != 1:
            raise CoreValueError(f"{name}: data must be 1D")
        if len(data) != len(indices):
            raise CoreValueError(f"{name}: data length must equal indices length")

    return True


def validate_array_size(arr, str name, Py_ssize_t min_size=0, Py_ssize_t max_size=-1):
    """Validate array size."""
    arr = np.asarray(arr)
    if arr.size < min_size:
        raise CoreValueError(f"{name}: size {arr.size} < minimum {min_size}")
    if max_size >= 0 and arr.size > max_size:
        raise CoreValueError(f"{name}: size {arr.size} > maximum {max_size}")
    return True


def check_parallel_memory(str operation_name, int threads, Py_ssize_t elements, Py_ssize_t element_size):
    """Check if parallel operation fits in memory limits."""
    cdef int64_t mem_required = <int64_t>threads * <int64_t>elements * <int64_t>element_size
    cdef int64_t limit = get_max_parallel_buffer_bytes()

    if mem_required > limit:
        raise CoreMemoryLimitError(
            f"{operation_name} requires {mem_required / (1024**3):.2f} GB, "
            f"but limit is {limit / (1024**3):.2f} GB"
        )
    return True


def check_dense_allocation(str operation_name, Py_ssize_t nrows, Py_ssize_t ncols):
    """Check if a dense matrix allocation fits in memory limits.

    Raises CoreMemoryLimitError if nrows x ncols x 8 bytes exceeds
    the max_dense_allocation limit.
    """
    cdef int64_t mem_required = <int64_t>nrows * <int64_t>ncols * 8
    cdef int64_t limit = get_max_dense_allocation_bytes()

    if mem_required > limit:
        raise CoreMemoryLimitError(
            f"{operation_name} requires {mem_required / (1024**3):.2f} GB "
            f"for a {nrows}x{ncols} dense matrix, "
            f"but limit is {limit / (1024**3):.2f} GB. "
            f"Use configure_memory(max_dense_allocation=...) to increase."
        )
    return True


def suggest_threads_for_memory(Py_ssize_t elements, Py_ssize_t element_size, int min_threads=1):
    """Suggest maximum threads that fit within memory limits."""
    cdef int64_t limit = get_max_parallel_buffer_bytes()
    cdef int64_t per_thread = <int64_t>elements * <int64_t>element_size

    if per_thread <= 0:
        return get_max_threads()

    cdef int64_t max_threads = limit // per_thread
    cdef int available = get_max_threads()

    if max_threads < min_threads:
        return min_threads
    if max_threads > available:
        return available
    return <int>max_threads


# Memory estimation

def estimate_memory_usage(int64_t n_nodes, int64_t n_edges, int avg_edge_size=2, bint include_transpose=True):
    """Estimate memory usage for hypergraph CSR structures."""
    if n_nodes < 0 or n_edges < 0 or avg_edge_size < 0:
        raise ValueError("All sizes must be non-negative")

    cdef int64_t nnz = n_edges * <int64_t>avg_edge_size

    estimates = {
        "edge_ptr": int(n_edges + 1) * 4,
        "edge_idx": int(nnz) * 4,
    }

    if include_transpose:
        estimates["node_ptr"] = int(n_nodes + 1) * 8
        estimates["node_idx"] = int(nnz) * 8

    estimates["total"] = sum(estimates.values())
    estimates["total_gb"] = estimates["total"] / (1024.0 ** 3)
    estimates["nnz_estimate"] = int(nnz)

    return estimates


def estimate_dense_matrix_bytes(Py_ssize_t n, Py_ssize_t m=-1):
    """Estimate bytes for a dense float64 matrix.

    Parameters
    ----------
    n : int
        Number of rows. If m is not given, assumes square (n x n).
    m : int, optional
        Number of columns.

    Returns
    -------
    dict with bytes, gb, fits_in_limit.
    """
    if m < 0:
        m = n
    cdef int64_t mem = <int64_t>n * <int64_t>m * 8
    cdef int64_t limit = get_max_dense_allocation_bytes()
    return {
        "bytes": int(mem),
        "gb": mem / (1024.0 ** 3),
        "limit_bytes": int(limit),
        "limit_gb": limit / (1024.0 ** 3),
        "fits_in_limit": mem <= limit,
    }


# Diagnostics

def test_parallel_execution(int n_iterations=1_000_000):
    """Test OpenMP parallel execution."""
    import time

    cdef Py_ssize_t n = <Py_ssize_t>n_iterations
    cdef Py_ssize_t i
    cdef int threads = get_max_threads()
    cdef double total = 0.0

    start = time.perf_counter()

    if REXGRAPH_HAS_OPENMP:
        with nogil:
            for i in prange(n, schedule="static", num_threads=threads):
                total += 1.0
    else:
        for i in range(n):
            total += 1.0

    elapsed = time.perf_counter() - start

    return {
        "openmp_enabled": bool(get_openmp_enabled()),
        "threads_used": int(threads),
        "iterations": int(n_iterations),
        "elapsed_seconds": float(elapsed),
        "result_correct": bool(fabs(total - <double>n_iterations) <= 1e-6),
    }


# Module initialization

_env_applied = configure_from_environment()
_auto_detect_limits()


# Module exports

__all__ = [
    # Configuration
    "configure_memory",
    "configure_parallelization",
    "configure_threads",
    "configure_algorithms",
    "configure_from_environment",
    "get_configuration",
    "get_parallelization_config",
    "get_algorithm_config",
    # Feature detection
    "get_openmp_enabled",
    "get_debug_enabled",
    "get_max_threads",
    "get_effective_threads",
    "get_build_info",
    # Error handling
    "CoreError",
    "CoreMemoryError",
    "CoreMemoryLimitError",
    "CoreValueError",
    "CoreOverflowError",
    "raise_on_error",
    "check_error",
    # Validation
    "validate_csr_arrays",
    "validate_array_size",
    "check_parallel_memory",
    "check_dense_allocation",
    "suggest_threads_for_memory",
    # Memory estimation
    "estimate_memory_usage",
    "estimate_dense_matrix_bytes",
    # Diagnostics
    "test_parallel_execution",
]
