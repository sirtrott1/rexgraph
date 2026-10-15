"""Vulkan sparse numerical contracts and optional hardware execution."""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph import vulkan_compute as V


@pytest.fixture
def cpu_products(monkeypatch):
    class Resident:
        def __init__(self, matrix, **kwargs):
            from rexgraph.native_sparse import as_native
            self.matrix = as_native(matrix)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def apply(self, values):
            return self.matrix.apply(values)
    monkeypatch.setattr(V, "SparseOperator", Resident)


@pytest.mark.parametrize("shape,batch", [((0, 0), 0), ((0, 3), 2), ((3, 0), 2), ((3, 3), 0)])
def test_empty_products_preserve_shape_without_a_runtime(shape, batch, monkeypatch):
    monkeypatch.setattr(V, "SparseOperator", lambda *a, **k: pytest.fail("empty product initialized a device"))
    result = V.sparse_mm(np.zeros(shape), np.zeros((shape[1], batch)))
    assert result.shape == (shape[0], batch) and result.dtype == np.float64
    assert np.array_equal(result, np.zeros_like(result))


def test_invalid_input_is_rejected_before_device_access(monkeypatch):
    monkeypatch.setattr(V, "SparseOperator", lambda *a, **k: pytest.fail("invalid product initialized a device"))
    for values in (np.ones(4), np.ones((3, 2, 1))):
        with pytest.raises(ValueError, match="matching vector"):
            V.sparse_mm(np.eye(3), values)
    with pytest.raises(TypeError, match="real values"):
        V.sparse_mm(np.eye(3), np.ones(3, dtype=complex))
    with pytest.raises(TypeError, match="real coefficients"):
        V.sparse_mm(np.eye(3, dtype=complex), np.ones(3))
    with pytest.raises(FloatingPointError, match="numerical range"):
        V.sparse_mm(np.array([[np.nan]]), np.ones(1))


@pytest.mark.parametrize("n,columns,order", [(1, 1, 1), (3, 1, 32), (11, 4, 48)])
def test_polynomial_application_matches_dense_exponential(cpu_products, n, columns, order):
    scipy_linalg = pytest.importorskip("scipy.linalg")
    from rexgraph.scale_propagator import _cheb_basis, _cheb_coeffs
    rng = np.random.default_rng(17)
    a = rng.normal(size=(n, n))
    matrix = a.T @ a + np.eye(n)
    bound = np.max(np.abs(matrix).sum(axis=1)) * 1.001
    values = rng.normal(size=(n, columns))
    nodes, basis = _cheb_basis(bound, order)
    coefficients = _cheb_coeffs(np.exp(-0.2 * nodes), basis, order)
    got = V.matfunc_apply(matrix, values, coefficients, bound)
    if order == 1:
        np.testing.assert_array_equal(got, coefficients[0] * values)
    else:
        np.testing.assert_allclose(got, scipy_linalg.expm(-0.2 * matrix) @ values, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("vector", [True, False])
def test_solver_matches_dense_solution_and_retains_zero_columns(cpu_products, vector):
    matrix = np.array([[4., -1., 0.], [-1., 4., -1.], [0., -1., 3.]])
    rhs = np.array([[1., 0.], [2., 0.], [3., 0.]])
    if vector:
        rhs = rhs[:, 0]
    result = V.block_cg(matrix, rhs, 1.0 / matrix.diagonal())
    assert result.shape == rhs.shape
    np.testing.assert_allclose(result, np.linalg.solve(matrix, rhs), rtol=1e-10, atol=1e-12)


def test_inverse_diagonal_handles_partial_column_tiles(cpu_products):
    matrix = np.diag(np.arange(1, 8, dtype=np.float64)) + np.ones((7, 7)) * 0.1
    result = V.greens_diagonal(matrix, chunk=3)
    np.testing.assert_allclose(result, np.diag(np.linalg.inv(matrix)), rtol=1e-10, atol=1e-12)


def test_explicit_vulkan_routes_to_vulkan_without_torch_or_cpu_fallback(monkeypatch):
    from rexgraph import scale_propagator as spg
    def unavailable(*args, **kwargs):
        raise RuntimeError("selected Vulkan device unavailable")
    monkeypatch.setattr(V, "matfunc_apply", unavailable)
    with pytest.raises(RuntimeError, match="selected Vulkan"):
        spg.heat_apply(np.eye(3), np.ones(3), 0.2, backend="vulkan")
    np.testing.assert_allclose(spg.heat_apply(np.eye(3), np.ones(3), 0.2, backend="cpu"), np.exp(-0.2))


def test_automatic_vulkan_preserves_cpu_fallback_on_device_failure(monkeypatch):
    from rexgraph import compute, scale_propagator as spg
    monkeypatch.setattr(compute, "get_default_backend", lambda: "vulkan")
    monkeypatch.setattr(V, "available", lambda: True)
    def unavailable(*args, **kwargs):
        raise RuntimeError("Vulkan device lost")
    monkeypatch.setattr(V, "matfunc_apply", unavailable)
    np.testing.assert_allclose(spg.heat_apply(np.eye(3), np.ones(3), 0.2), np.exp(-0.2))


def test_device_loss_still_releases_allocations_and_closes_operator():
    from contextlib import ExitStack
    from threading import RLock
    from types import SimpleNamespace
    resident = object.__new__(V.SparseOperator)
    resident._lock = RLock()
    resident._closed = False
    resident.device = object()
    def lost(device):
        raise RuntimeError("device lost")
    resident.vk = SimpleNamespace(vkDeviceWaitIdle=lost)
    released = []
    resident._vectors = ExitStack()
    resident._vectors.callback(released.append, "vectors")
    resident._resources = ExitStack()
    resident._resources.callback(released.append, "matrix")
    with pytest.raises(RuntimeError, match="device lost"):
        resident.close()
    assert released == ["vectors", "matrix"] and resident._closed


@pytest.fixture
def vulkan_device():
    if not V.available():
        pytest.skip("a Vulkan hardware GPU with float64 compute support is required")


@pytest.mark.parametrize("rows,columns,batch", [(1, 1, 1), (4, 7, 3), (63, 65, 1), (65, 131, 5)])
def test_actual_vulkan_products_and_resident_resize_match_cpu(vulkan_device, rows, columns, batch):
    rng = np.random.default_rng(rows)
    matrix = rng.normal(size=(rows, columns))
    matrix[rng.random(matrix.shape) < 0.6] = 0
    with V.SparseOperator(matrix) as resident:
        for shape in ((columns,), (columns, batch), (columns, 2), (columns, batch)):
            values = rng.normal(size=shape)
            np.testing.assert_allclose(resident.apply(values), matrix @ values, rtol=1e-12, atol=1e-12)
        resident.close()
        resident.close()
        with pytest.raises(RuntimeError, match="closed"):
            resident.apply(np.ones(columns))


def test_actual_vulkan_propagation_and_solve_match_cpu(vulkan_device):
    report = V.qualify(65)
    assert report["ok"], report


def test_actual_vulkan_allocation_failure_leaves_operator_reusable(vulkan_device, monkeypatch):
    with V.SparseOperator(np.eye(5)) as resident:
        original = resident._buffer
        def fail(size, resources):
            raise MemoryError("allocation unavailable")
        monkeypatch.setattr(resident, "_buffer", fail)
        with pytest.raises(MemoryError):
            resident.apply(np.ones(5))
        monkeypatch.setattr(resident, "_buffer", original)
        np.testing.assert_array_equal(resident.apply(np.ones(5)), np.ones(5))
