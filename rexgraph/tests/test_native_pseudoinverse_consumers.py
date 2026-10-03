"""Numerical pseudoinverse consumers, residual contracts and native storage."""
from concurrent.futures import ThreadPoolExecutor
import builtins

import numpy as np
import pytest

from rexgraph.core._hodge import least_squares
from rexgraph.native_sparse import as_native, native_diagonal
from rexgraph.scale_propagator import _greens_diagonal_lsqr
from rexgraph.sparse_character import pinv_quadratic_form
from rexgraph.sparse_interfacing import pinv_bilinear_form


def _operator(kind):
    if kind == "empty":
        return np.empty((0, 0))
    if kind == "zero":
        return np.zeros((4, 4))
    if kind == "diagonal":
        return np.diag([0., 0., 0.25, 2.])
    boundary = np.array([[-1., 0], [0.5, -1], [0.5, 0.5], [0, 0.5], [0, 0]])
    return boundary @ boundary.T


@pytest.mark.parametrize("kind", ["empty", "zero", "diagonal", "branching"])
@pytest.mark.parametrize("carrier", ["dense", "native", "dual", "csr", "scipy"])
def test_pseudoinverse_consumers_match_dense_reference(kind, carrier):
    a = _operator(kind)
    expected = np.linalg.pinv(a)
    matrix = a
    if carrier != "dense":
        matrix = as_native(a)
        if carrier == "dual":
            matrix = matrix.dual
        elif carrier == "csr":
            matrix = matrix.dual.csr
        elif carrier == "scipy":
            sparse = pytest.importorskip("scipy.sparse")
            matrix = sparse.csr_matrix(a)
    v = np.arange(1., a.shape[0] + 1)
    u = v[::-1].copy()
    np.testing.assert_allclose(pinv_quadratic_form(matrix, v), v @ expected @ v,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pinv_bilinear_form(matrix, u, v), u @ expected @ v,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_greens_diagonal_lsqr(matrix), np.diag(expected),
                               rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("scale", [1e-200, 1., 1e200])
def test_forms_retain_scaling_and_readonly_inputs(scale):
    a = _operator("branching")
    expected = np.linalg.pinv(a)
    matrix = as_native(a * scale)
    for values in (matrix.dual.row_ptr, matrix.dual.col_idx, matrix.data,
                   matrix.dual.col_ptr, matrix.dual.row_idx, matrix.dual.vals_csc):
        values.setflags(write=False)
    v = np.arange(1., 11.)[::2]
    u = v[::-1]
    original_u, original_v, original_a = u.copy(), v.copy(), matrix.data.copy()
    u.setflags(write=False)
    v.setflags(write=False)
    np.testing.assert_allclose(pinv_quadratic_form(matrix, v) * scale, v @ expected @ v,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pinv_bilinear_form(matrix, u, v) * scale, u @ expected @ v,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(u, original_u)
    np.testing.assert_array_equal(v, original_v)
    np.testing.assert_array_equal(matrix.data, original_a)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("value_dtype", [np.float32, np.float64])
def test_readonly_coo_storage_and_column_rebuild(index_dtype, value_dtype):
    from rexgraph.core import _sparse
    rows = np.array([0, 0, 1], dtype=index_dtype)
    columns = np.array([0, 1, 1], dtype=index_dtype)
    values = np.array([2., -1., 3.], dtype=value_dtype)
    for array in (rows, columns, values):
        array.setflags(write=False)
    csr = _sparse.csr_from_coo(rows, columns, values, 2, 2)
    dual = _sparse.dual_from_coo(rows, columns, values, 2, 2)
    expected = np.array([[2., -1.], [0., 3.]])
    for matrix in (csr, dual.csr):
        for array in (matrix.row_ptr, matrix.col_idx, matrix.vals):
            array.setflags(write=False)
        rebuilt = _sparse.dual_from_csr(matrix)
        np.testing.assert_allclose(_sparse.matvec(rebuilt, np.ones(2)), expected @ np.ones(2))
        np.testing.assert_allclose(_sparse.rmatvec(rebuilt, np.ones(2)), expected.T @ np.ones(2))
        np.testing.assert_allclose(least_squares(rebuilt, [1., 2.]), np.linalg.solve(expected, [1., 2.]))


@pytest.mark.parametrize("entries", [[(0, 0, 2.), (1, 1, 3.)],
    [(0, 0, 2.), (0, 0, -2.), (1, 1, 3.)],
    [(0, 0, 0.), (0, 1, 2.), (1, 1, 3.)],
    [(0, 0, 2.), (0, 0, 1.), (1, 1, 3.)]])
def test_canonical_sparse_cleanup_keeps_sums_and_removes_zero_addresses(entries):
    from rexgraph.core import _sparse
    rows, columns, values = zip(*entries, strict=True)
    matrix = _sparse.dual_from_coo(np.array(rows, np.int32), np.array(columns, np.int32),
                                  np.array(values), 2, 2)
    expected = np.zeros((2, 2))
    for row, column, value in entries:
        expected[row, column] += value
    cleaned = _sparse.canonical_dual(matrix)
    assert cleaned.nnz == np.count_nonzero(expected)
    np.testing.assert_allclose(_sparse.matvec(cleaned, np.ones(2)), expected @ np.ones(2))
    np.testing.assert_allclose(_sparse.rmatvec(cleaned, np.ones(2)), expected.T @ np.ones(2))


def test_native_consumers_work_without_scipy_or_dense_decompositions(monkeypatch):
    a = _operator("branching")
    expected = np.linalg.pinv(a)
    matrix = as_native(a)
    importer = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("SciPy forbidden")
        return importer(name, *args, **kwargs)

    def no_dense(*args, **kwargs):
        raise AssertionError("dense decomposition reached")

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    for name in ("svd", "eigh", "eigvalsh", "pinv", "lstsq"):
        monkeypatch.setattr(np.linalg, name, no_dense)
    v = np.arange(1., 6.)
    np.testing.assert_allclose(pinv_quadratic_form(matrix, v), v @ expected @ v)
    np.testing.assert_allclose(pinv_bilinear_form(matrix, v[::-1], v), v[::-1] @ expected @ v)
    np.testing.assert_allclose(_greens_diagonal_lsqr(matrix), np.diag(expected), atol=1e-10)


@pytest.mark.parametrize("bilinear", [False, True])
def test_forms_refuse_exhausted_solves(bilinear):
    matrix = native_diagonal([1., 2., 3.])
    args = (matrix, np.ones(3), np.ones(3)) if bilinear else (matrix, np.ones(3))
    call = pinv_bilinear_form if bilinear else pinv_quadratic_form
    with pytest.raises(ArithmeticError, match="converge"):
        call(*args, iter_lim=1)


@pytest.mark.parametrize("atol,btol", [(0., 1e-10), (1e-10, 0.), (0., 0.), (1e-8, 1e-13)])
def test_separate_residual_thresholds_and_zero_override(atol, btol):
    matrix = native_diagonal([2.])
    x, info = least_squares(matrix, [6.], atol=atol, btol=btol, return_info=True)
    np.testing.assert_array_equal(x, [3.])
    assert info["atol"] == atol and info["btol"] == btol
    assert all(c["relative_residual"] <= btol or c["normal_residual"] <= atol
               for c in info["columns"])
    assert pinv_quadratic_form(matrix, [6.], atol=atol, btol=btol) == 18.
    assert pinv_bilinear_form(matrix, [4.], [6.], atol=atol, btol=btol) == 12.


@pytest.mark.parametrize("threshold", [-1., 1., np.inf, np.nan, True, 1j, [1e-6]])
@pytest.mark.parametrize("name", ["atol", "btol"])
def test_refuse_invalid_residual_overrides(name, threshold):
    with pytest.raises(ValueError, match="tolerances"):
        least_squares(native_diagonal([1.]), [1.], **{name: threshold})


@pytest.mark.parametrize("values", [[True, False], [1., np.inf], [1j, 0], ["1", "2"]])
def test_forms_refuse_invalid_signal_coefficients(values):
    with pytest.raises((TypeError, FloatingPointError)):
        pinv_quadratic_form(np.eye(2), values)
    with pytest.raises((TypeError, FloatingPointError)):
        pinv_bilinear_form(np.eye(2), values, [1., 1.])


def test_native_lsqr_refuses_nonfinite_coefficients_before_iteration():
    matrix = native_diagonal([1., 2.]).with_data([np.inf, 2.])
    with pytest.raises(FloatingPointError, match="coefficients"):
        least_squares(matrix, [1., 1.])


@pytest.mark.parametrize("scale", [1e-300, 1., 1e300])
def test_native_norm_retains_extreme_finite_magnitudes(scale):
    from rexgraph.core._hodge import _stable_norm
    values = np.array([3., 0., 4., 0.]) * scale
    vector = values[::2]
    vector.setflags(write=False)
    assert np.isclose(_stable_norm(vector) / scale, 5., rtol=1e-15)
    assert _stable_norm(np.empty(0)) == 0.


def test_forms_refuse_shapes_and_unrepresentable_results():
    with pytest.raises(ValueError, match="square"):
        pinv_quadratic_form(np.ones((2, 3)), [1., 1.])
    with pytest.raises(ValueError, match="matching"):
        pinv_bilinear_form(np.eye(2), [1.], [1., 1.])
    with pytest.raises(ValueError, match="square"):
        _greens_diagonal_lsqr(np.ones((2, 3)))
    with pytest.raises(FloatingPointError, match="float64"):
        pinv_quadratic_form(np.eye(1), [1e200])
    with pytest.raises(FloatingPointError, match="float64"):
        pinv_bilinear_form(np.eye(1), [1e200], [1e200])


def test_native_lsqr_concurrent_blocks_preserve_minimum_norm():
    rng = np.random.default_rng(901)
    a = rng.normal(size=(12, 3)) @ rng.normal(size=(3, 8))
    matrix = as_native(a)
    blocks = [rng.normal(size=(12, 2)) for _ in range(8)]
    inverse = np.linalg.pinv(a)
    with ThreadPoolExecutor(max_workers=4) as pool:
        observed = list(pool.map(lambda b: least_squares(matrix, b), blocks))
    for got, rhs in zip(observed, blocks, strict=True):
        np.testing.assert_allclose(got, inverse @ rhs, atol=1e-10)


def test_green_diagonal_propagates_solver_failure(monkeypatch):
    from rexgraph.core import _hodge

    def fail(*args, **kwargs):
        raise ArithmeticError("native LSQR did not converge")

    monkeypatch.setattr(_hodge, "least_squares", fail)
    with pytest.raises(ArithmeticError, match="converge"):
        _greens_diagonal_lsqr(native_diagonal([1., 2.]))
