"""
Tests for rexgraph.core._common — shared infrastructure layer.

Verifies:
    - Runtime configuration (memory, parallelization, threads, algorithms)
    - Environment variable configuration
    - Feature detection (OpenMP, build info)
    - Error code → exception mapping
    - CSR validation
    - Memory estimation
    - Dense allocation decision logic
    - Parallel execution correctness
    - Configuration persistence and reset
"""
import os
import numpy as np
import pytest

from rexgraph.core._common import (
    # Configuration
    configure_memory,
    configure_parallelization,
    configure_threads,
    configure_algorithms,
    configure_from_environment,
    get_configuration,
    get_parallelization_config,
    get_algorithm_config,
    # Feature detection
    get_openmp_enabled,
    get_debug_enabled,
    get_max_threads,
    get_effective_threads,
    get_build_info,
    # Error handling
    CoreError,
    CoreMemoryError,
    CoreMemoryLimitError,
    CoreValueError,
    CoreOverflowError,
    raise_on_error,
    check_error,
    # Validation
    validate_csr_arrays,
    validate_array_size,
    check_parallel_memory,
    check_dense_allocation,
    suggest_threads_for_memory,
    # Memory estimation
    estimate_memory_usage,
    estimate_dense_matrix_bytes,
    # Diagnostics
    test_parallel_execution,
)


# Configuration

class TestConfiguration:
    """Runtime configuration detection, mutation, and persistence."""

    def test_get_configuration_returns_dict(self):
        cfg = get_configuration()
        assert isinstance(cfg, dict)
        assert "max_dense_allocation_bytes" in cfg
        assert "system_memory_bytes" in cfg

    def test_system_memory_positive(self):
        cfg = get_configuration()
        assert cfg["system_memory_bytes"] > 0
        assert cfg["system_memory_gb"] > 0

    def test_available_memory_positive(self):
        cfg = get_configuration()
        assert cfg["available_memory_bytes"] > 0
        assert cfg["available_memory_gb"] > 0

    def test_default_limits_are_sane(self):
        """Memory limits should be positive and within system RAM."""
        cfg = get_configuration()
        sys_mem = cfg["system_memory_bytes"]
        assert 0 < cfg["max_parallel_buffer_bytes"] <= sys_mem
        assert 0 < cfg["max_total_allocation_bytes"] <= sys_mem
        assert 0 < cfg["max_dense_allocation_bytes"] <= sys_mem

    def test_dense_limit_floor(self):
        """Dense allocation should be at least 100 MB."""
        cfg = get_configuration()
        assert cfg["max_dense_allocation_bytes"] >= 100 * 1024 * 1024

    def test_dense_limit_ceiling(self):
        """Dense allocation should be at most 4 GB."""
        cfg = get_configuration()
        assert cfg["max_dense_allocation_bytes"] <= 4 * 1024 * 1024 * 1024

    def test_configure_memory_explicit(self):
        """Setting explicit memory limits persists."""
        old = get_configuration()
        try:
            configure_memory(max_dense_allocation=500_000_000)
            cfg = get_configuration()
            assert cfg["max_dense_allocation_bytes"] == 500_000_000
        finally:
            # Reset to auto-detect
            configure_memory()

    def test_configure_memory_invalid_fraction(self):
        with pytest.raises(ValueError):
            configure_memory(parallel_buffer_fraction=1.5)
        with pytest.raises(ValueError):
            configure_memory(dense_allocation_fraction=-0.1)

    def test_configure_memory_negative_bytes(self):
        with pytest.raises(ValueError):
            configure_memory(max_dense_allocation=-1)

    def test_configure_algorithms_persists(self):
        old = get_algorithm_config()
        try:
            configure_algorithms(eigen_dense_limit=999, default_k=10)
            cfg = get_algorithm_config()
            assert cfg["eigen_dense_limit"] == 999
            assert cfg["default_k"] == 10
        finally:
            configure_algorithms(
                eigen_dense_limit=old["eigen_dense_limit"],
                default_k=old["default_k"],
            )

    def test_configure_algorithms_invalid(self):
        with pytest.raises(ValueError):
            configure_algorithms(eigen_dense_limit=0)
        with pytest.raises(ValueError):
            configure_algorithms(default_k=-5)
        with pytest.raises(ValueError):
            configure_algorithms(fill_ratio_dense_threshold=2.0)


class TestParallelizationConfig:
    """Parallelization threshold configuration."""

    def test_get_parallelization_config(self):
        cfg = get_parallelization_config()
        assert "min_simple" in cfg
        assert "min_transpose" in cfg
        assert "openmp_enabled" in cfg

    def test_thresholds_have_minimum(self):
        """All thresholds should be >= 1000 (ABSOLUTE_MIN_PARALLEL_THRESHOLD)."""
        cfg = get_parallelization_config()
        assert cfg["min_simple"] >= 1000
        assert cfg["min_transpose"] >= 1000
        assert cfg["min_reduction"] >= 1000

    def test_configure_parallelization(self):
        old = get_parallelization_config()
        try:
            configure_parallelization(min_simple=200_000)
            cfg = get_parallelization_config()
            assert cfg["min_simple"] == 200_000
        finally:
            configure_parallelization(min_simple=old["min_simple"])

    def test_threshold_floor_enforced(self):
        """Setting a threshold below 1000 should clamp to 1000."""
        old = get_parallelization_config()
        try:
            configure_parallelization(min_simple=100)
            cfg = get_parallelization_config()
            assert cfg["min_simple"] >= 1000
        finally:
            configure_parallelization(min_simple=old["min_simple"])


class TestThreadConfig:
    """Thread management configuration."""

    def test_get_max_threads_positive(self):
        assert get_max_threads() >= 1

    def test_get_effective_threads_bounded(self):
        eff = get_effective_threads(0)
        assert 1 <= eff <= get_max_threads()

    def test_get_effective_threads_requested(self):
        """Requesting more threads than available returns available."""
        eff = get_effective_threads(99999)
        assert eff <= get_max_threads()

    def test_configure_threads_limit(self):
        try:
            configure_threads(max_threads=2)
            eff = get_effective_threads(0)
            assert eff <= 2
        finally:
            configure_threads(max_threads=-1)

    def test_configure_threads_reserved(self):
        try:
            configure_threads(reserved_threads=1)
            eff = get_effective_threads(0)
            # With 1 reserved, effective should be max_threads - 1 (minimum 1)
            assert eff >= 1
        finally:
            configure_threads(reserved_threads=0)

    def test_configure_threads_invalid_reserved(self):
        with pytest.raises(ValueError):
            configure_threads(reserved_threads=-1)


# Feature Detection

class TestFeatureDetection:
    """Compile-time and runtime feature detection."""

    def test_openmp_is_bool(self):
        assert isinstance(get_openmp_enabled(), bool)

    def test_debug_is_bool(self):
        assert isinstance(get_debug_enabled(), bool)

    def test_build_info_complete(self):
        info = get_build_info()
        assert "openmp_enabled" in info
        assert "max_threads" in info
        assert info["compiled"] is True

    def test_build_info_threads_consistent(self):
        info = get_build_info()
        assert info["max_threads"] == get_max_threads()


# Error Handling

class TestErrorHandling:
    """Error code → exception mapping."""

    def test_success_no_raise(self):
        raise_on_error(0)  # ERR_SUCCESS
        check_error(0)

    def test_memory_error(self):
        with pytest.raises(CoreMemoryError):
            raise_on_error(-1, "test")

    def test_invalid_arg_error(self):
        with pytest.raises(CoreValueError):
            raise_on_error(-2, "test")

    def test_out_of_bounds_error(self):
        with pytest.raises(CoreValueError):
            raise_on_error(-3, "test")

    def test_overflow_error(self):
        with pytest.raises(CoreOverflowError):
            raise_on_error(-4, "test")

    def test_shape_mismatch_error(self):
        with pytest.raises(CoreValueError):
            raise_on_error(-5, "test")

    def test_not_converged_error(self):
        with pytest.raises(CoreError):
            raise_on_error(-6, "test")

    def test_singular_error(self):
        with pytest.raises(CoreValueError):
            raise_on_error(-7, "test")

    def test_memory_limit_error(self):
        with pytest.raises(CoreMemoryLimitError):
            raise_on_error(-8, "test")

    def test_cancelled_error(self):
        with pytest.raises(CoreError):
            raise_on_error(-9, "test")

    def test_unknown_error(self):
        with pytest.raises(CoreError):
            raise_on_error(-99, "test")

    def test_error_hierarchy(self):
        """CoreMemoryLimitError is catchable as CoreMemoryError and MemoryError."""
        assert issubclass(CoreMemoryLimitError, CoreMemoryError)
        assert issubclass(CoreMemoryError, MemoryError)
        assert issubclass(CoreValueError, ValueError)
        assert issubclass(CoreOverflowError, OverflowError)

    def test_error_message_includes_context(self):
        try:
            raise_on_error(-1, "build_RL")
        except CoreMemoryError as e:
            assert "build_RL" in str(e)


# Validation

class TestValidation:
    """Input validation for CSR arrays and sizes."""

    def test_valid_csr(self):
        indptr = np.array([0, 2, 3, 5], dtype=np.int32)
        indices = np.array([1, 2, 0, 0, 1], dtype=np.int32)
        assert validate_csr_arrays(indptr, indices) is True

    def test_valid_csr_with_data(self):
        indptr = np.array([0, 2, 3], dtype=np.int32)
        indices = np.array([0, 1, 2], dtype=np.int32)
        data = np.array([1.0, -1.0, 0.5], dtype=np.float64)
        assert validate_csr_arrays(indptr, indices, data) is True

    def test_csr_indptr_not_starting_zero(self):
        indptr = np.array([1, 2, 3], dtype=np.int32)
        indices = np.array([0, 1], dtype=np.int32)
        with pytest.raises(CoreValueError):
            validate_csr_arrays(indptr, indices)

    def test_csr_indices_length_mismatch(self):
        indptr = np.array([0, 2, 3], dtype=np.int32)
        indices = np.array([0, 1], dtype=np.int32)  # should be 3 elements
        with pytest.raises(CoreValueError):
            validate_csr_arrays(indptr, indices)

    def test_csr_data_length_mismatch(self):
        indptr = np.array([0, 2], dtype=np.int32)
        indices = np.array([0, 1], dtype=np.int32)
        data = np.array([1.0], dtype=np.float64)  # too short
        with pytest.raises(CoreValueError):
            validate_csr_arrays(indptr, indices, data)

    def test_csr_2d_indptr_rejected(self):
        indptr = np.array([[0, 1], [2, 3]], dtype=np.int32)
        indices = np.array([0, 1, 2], dtype=np.int32)
        with pytest.raises(CoreValueError):
            validate_csr_arrays(indptr, indices)

    def test_validate_array_size(self):
        arr = np.zeros(10)
        assert validate_array_size(arr, "test", min_size=5) is True

    def test_validate_array_size_too_small(self):
        arr = np.zeros(3)
        with pytest.raises(CoreValueError):
            validate_array_size(arr, "test", min_size=5)

    def test_validate_array_size_too_large(self):
        arr = np.zeros(100)
        with pytest.raises(CoreValueError):
            validate_array_size(arr, "test", max_size=50)


# Memory Estimation

class TestMemoryEstimation:
    """Memory usage estimation for capacity planning."""

    def test_estimate_memory_basic(self):
        est = estimate_memory_usage(100, 500)
        assert est["total"] > 0
        assert est["total_gb"] > 0
        assert "edge_ptr" in est
        assert "edge_idx" in est

    def test_estimate_memory_includes_transpose(self):
        with_t = estimate_memory_usage(100, 500, include_transpose=True)
        without_t = estimate_memory_usage(100, 500, include_transpose=False)
        assert with_t["total"] > without_t["total"]
        assert "node_ptr" in with_t
        assert "node_ptr" not in without_t

    def test_estimate_memory_scales(self):
        small = estimate_memory_usage(100, 500)
        large = estimate_memory_usage(1000, 5000)
        assert large["total"] > small["total"]

    def test_estimate_memory_negative_raises(self):
        with pytest.raises(ValueError):
            estimate_memory_usage(-1, 100)

    def test_estimate_dense_matrix(self):
        est = estimate_dense_matrix_bytes(1000)
        assert est["bytes"] == 1000 * 1000 * 8
        assert est["gb"] == est["bytes"] / (1024 ** 3)
        assert "fits_in_limit" in est

    def test_estimate_dense_rectangular(self):
        est = estimate_dense_matrix_bytes(100, 200)
        assert est["bytes"] == 100 * 200 * 8

    def test_estimate_dense_small_fits(self):
        est = estimate_dense_matrix_bytes(10)
        assert est["fits_in_limit"] is True

    def test_estimate_dense_huge_does_not_fit(self):
        est = estimate_dense_matrix_bytes(100_000)
        # 100K x 100K x 8 = 80 GB — should not fit
        assert est["fits_in_limit"] is False


# Dense Allocation Checks

class TestDenseAllocation:
    """Dense allocation limit enforcement."""

    def test_check_small_passes(self):
        assert check_dense_allocation("test", 100, 100) is True

    def test_check_huge_raises(self):
        with pytest.raises(CoreMemoryLimitError):
            check_dense_allocation("test", 100_000, 100_000)

    def test_check_message_includes_context(self):
        try:
            check_dense_allocation("build_L1", 100_000, 100_000)
        except CoreMemoryLimitError as e:
            assert "build_L1" in str(e)
            assert "configure_memory" in str(e)  # helpful hint


# Parallel Memory Checks

class TestParallelMemory:
    """Parallel scratch buffer limit enforcement."""

    def test_small_passes(self):
        assert check_parallel_memory("test", 4, 1000, 8) is True

    def test_suggest_threads_positive(self):
        n = suggest_threads_for_memory(10000, 8)
        assert n >= 1

    def test_suggest_threads_bounded(self):
        n = suggest_threads_for_memory(10000, 8)
        assert n <= get_max_threads()


# Parallel Execution

class TestParallelExecution:
    """OpenMP parallel execution correctness."""

    def test_parallel_result_correct(self):
        """Parallel sum of 1.0 over N iterations should equal N."""
        result = test_parallel_execution(100_000)
        assert result["result_correct"] is True

    def test_parallel_threads_reported(self):
        result = test_parallel_execution(100_000)
        assert result["threads_used"] >= 1
        assert result["threads_used"] == get_max_threads()

    def test_parallel_openmp_consistent(self):
        result = test_parallel_execution(100_000)
        assert result["openmp_enabled"] == get_openmp_enabled()


# Environment Variable Configuration

class TestEnvironmentConfig:
    """Configuration via environment variables."""

    def test_configure_from_environment_returns_bool(self):
        result = configure_from_environment()
        assert isinstance(result, bool)

    def test_env_eigen_dense_limit(self):
        old = get_algorithm_config()
        try:
            os.environ["REXGRAPH_EIGEN_DENSE_LIMIT"] = "1234"
            configure_from_environment()
            cfg = get_algorithm_config()
            assert cfg["eigen_dense_limit"] == 1234
        finally:
            os.environ.pop("REXGRAPH_EIGEN_DENSE_LIMIT", None)
            configure_algorithms(eigen_dense_limit=old["eigen_dense_limit"])

    def test_env_invalid_value_ignored(self):
        """Invalid environment values should be silently ignored."""
        old = get_algorithm_config()
        try:
            os.environ["REXGRAPH_EIGEN_DENSE_LIMIT"] = "not_a_number"
            configure_from_environment()
            # Should not crash, limit should remain unchanged
            cfg = get_algorithm_config()
            assert cfg["eigen_dense_limit"] == old["eigen_dense_limit"]
        finally:
            os.environ.pop("REXGRAPH_EIGEN_DENSE_LIMIT", None)


# Algorithm Config Accessors

class TestAlgorithmConfig:
    """Algorithm selection configuration."""

    def test_get_algorithm_config_complete(self):
        cfg = get_algorithm_config()
        assert "eigen_dense_limit" in cfg
        assert "default_k" in cfg
        assert "fill_ratio_dense_threshold" in cfg
        assert "max_dense_allocation_bytes" in cfg

    def test_eigen_dense_limit_positive(self):
        cfg = get_algorithm_config()
        assert cfg["eigen_dense_limit"] >= 1

    def test_default_k_positive(self):
        cfg = get_algorithm_config()
        assert cfg["default_k"] >= 1

    def test_fill_ratio_in_range(self):
        cfg = get_algorithm_config()
        assert 0.0 < cfg["fill_ratio_dense_threshold"] <= 1.0
