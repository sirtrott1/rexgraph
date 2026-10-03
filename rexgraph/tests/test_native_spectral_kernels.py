"""Native decompositions, sparse spectra and their public consumers."""
from concurrent.futures import ThreadPoolExecutor
import builtins

import numpy as np
import pytest

from rexgraph.core import _channel_tower, _l_gb, _linalg, _wave
from rexgraph.native_sparse import as_native, native_coo, native_diagonal


@pytest.mark.parametrize("shape", [(7, 3), (3, 7), (4, 4), (0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_svd_retains_input_and_vector_shapes(shape, complex_input, full):
    rng = np.random.default_rng(17)
    source = rng.normal(size=(shape[0] * 2, shape[1] * 2))
    if complex_input:
        source = source + 1j * rng.normal(size=source.shape)
    a = source[::2, ::2]
    original = a.copy()
    a.setflags(write=False)
    u, s, vh = _linalg.svd(a, full_matrices=full)
    rank = min(shape)
    assert u.shape == (shape[0], shape[0] if full else rank)
    assert vh.shape == (shape[1] if full else rank, shape[1])
    np.testing.assert_allclose((u[:, :rank] * s) @ vh[:rank], a, atol=1e-12)
    np.testing.assert_allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-12)
    np.testing.assert_allclose(_linalg.svd(a, compute_uv=False), s, atol=1e-12)
    np.testing.assert_array_equal(a, original)


@pytest.mark.parametrize("complex_input", [False, True])
def test_eigensolvers_retained_strided_input(complex_input):
    rng = np.random.default_rng(19)
    a = rng.normal(size=(18, 18))
    if complex_input:
        a = a + 1j * rng.normal(size=a.shape)
    a = (a + a.conj().T)[::2, ::2]
    original = a.copy()
    a.setflags(write=False)
    values = _linalg.eigvalsh(a)
    np.testing.assert_allclose(values, np.linalg.eigvalsh(a), atol=1e-12)
    values, vectors = _linalg.eigh(a, clip_negative_roundoff=False)
    np.testing.assert_allclose((vectors * values) @ vectors.conj().T, a, atol=1e-12)
    np.testing.assert_array_equal(a, original)


@pytest.mark.parametrize("shape", [(2, 5), (5, 2), (4, 4), (0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize("columns", [None, 0, 3])
def test_least_squares_rectangular_minimum_norm_and_blocks(shape, columns):
    rng = np.random.default_rng(23)
    a = rng.normal(size=shape)
    if min(shape) > 1:
        a[:, -1] = a[:, 0]
    b = rng.normal(size=(shape[0],) if columns is None else (shape[0], columns))
    original_a, original_b = a.copy(), b.copy()
    a.setflags(write=False)
    b.setflags(write=False)
    x, rank = _linalg.lstsq(a, b)
    expected, _, expected_rank, _ = np.linalg.lstsq(a, b, rcond=None)
    np.testing.assert_allclose(x, expected, atol=1e-11)
    assert rank == expected_rank
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(b, original_b)


@pytest.mark.parametrize("columns", [None, 0, 3])
@pytest.mark.parametrize("rcond,rank", [(None, 1), (-1.0, 2), (0.0, 2), (1.0, 2), (1e-12, 1)])
def test_least_squares_rank_cutoff_is_consistent_for_vectors_and_blocks(columns, rcond, rank):
    small = np.finfo(np.float64).eps * 0.75
    a = np.diag([1.0, small])
    b = np.ones(2) if columns is None else np.ones((2, columns))
    x, observed_rank = _linalg.lstsq(a, b, rcond=rcond)
    expected = b.copy()
    expected[1] = b[1] / small if rank == 2 else 0.0
    assert observed_rank == rank
    np.testing.assert_allclose(x, expected, atol=1e-12)


@pytest.mark.parametrize("call", [_linalg.eigh, _linalg.eigvalsh, _linalg.svd, _linalg.largest_eigenpair])
def test_dense_decompositions_refuse_nonfinite_input(call):
    with pytest.raises(ValueError, match="finite"):
        call(np.array([[np.nan]]))
    with pytest.raises(TypeError, match="numerical"):
        call(np.array([[1]], dtype=object))


def test_least_squares_refuses_invalid_rhs_and_tolerance():
    for b in (np.ones(3), np.array([np.inf, 0.0]), np.ones((2, 1, 1)), np.ones(2, complex)):
        with pytest.raises(ValueError):
            _linalg.lstsq(np.eye(2), b)
    with pytest.raises(ValueError):
        _linalg.lstsq(np.eye(2), np.ones(2), rcond=np.nan)


def test_lapack_workspaces_are_independent_across_threads():
    rng = np.random.default_rng(31)
    matrices = [rng.normal(size=(80, 80)) for _ in range(12)]
    matrices = [a + a.T for a in matrices]
    expected = [np.linalg.eigvalsh(a) for a in matrices]
    with ThreadPoolExecutor(max_workers=4) as pool:
        observed = list(pool.map(_linalg.eigvalsh, matrices))
    for got, want in zip(observed, expected, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-11)


@pytest.mark.parametrize("n", [0, 1, 2, 8, 25])
def test_selected_eigenpair_matches_dense_and_has_small_residual(n):
    rng = np.random.default_rng(37)
    a = rng.normal(size=(n, n))
    a = a + a.T
    value, vector = _linalg.largest_eigenpair(a)
    assert vector.shape == (n,)
    if n:
        assert np.isclose(value, np.linalg.eigvalsh(a)[-1], atol=1e-12)
        np.testing.assert_allclose(a @ vector, value * vector, atol=1e-12)
        assert np.isclose(vector @ vector, 1.0)
    else:
        assert value == 0.0


def _forbid_numpy_decompositions(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("NumPy decomposition reached")
    for name in ("eigh", "eigvalsh", "svd", "lstsq"):
        monkeypatch.setattr(np.linalg, name, refused)


def test_wave_observables_use_native_values_only(monkeypatch):
    psi = np.array([1, 0, 0, 1j], dtype=np.complex128) / np.sqrt(2)
    rho = np.eye(4, dtype=np.complex128) / 4
    _forbid_numpy_decompositions(monkeypatch)
    assert np.isclose(_wave.entanglement_entropy(psi, 2, 2), 1.0)
    assert np.isclose(_wave.von_neumann_entropy(rho), 2.0)
    u, s, vh = _wave.schmidt_decomposition(psi, 2, 2)
    np.testing.assert_allclose((u * s) @ vh, psi.reshape(2, 2), atol=1e-12)


def test_sparse_scalar_and_two_cell_blocks_need_no_lapack_or_scipy(monkeypatch):
    n = 6000
    rows = np.arange(n)
    matrix = native_coo(np.concatenate((rows, rows)),
                        np.concatenate((rows, rows ^ 1)),
                        np.concatenate((np.full(n, 3.0), np.ones(n))), (n, n))
    importer = builtins.__import__
    def no_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("SciPy forbidden")
        return importer(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_scipy)
    def no_dense(*args, **kwargs):
        raise AssertionError("dense spectrum reached")
    monkeypatch.setattr(_linalg, "eigvalsh", no_dense)
    _forbid_numpy_decompositions(monkeypatch)
    values = _linalg.symmetric_sparse_spectrum(matrix)
    np.testing.assert_array_equal(values, np.repeat([2.0, 4.0], n // 2))
    diagonal = native_diagonal(np.arange(1000, dtype=float))
    np.testing.assert_array_equal(_linalg.symmetric_sparse_spectrum(diagonal), np.arange(1000))


def test_sparse_spectrum_preserves_zeros_and_symmetrizes_components():
    a = np.array([[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 3, 1], [0, 0, 1, 3]], float)
    expected = np.linalg.eigvalsh(0.5 * (a + a.T))
    np.testing.assert_allclose(_linalg.symmetric_sparse_spectrum(as_native(a)), expected)
    np.testing.assert_array_equal(_linalg.symmetric_sparse_spectrum(native_diagonal(np.zeros(4))), np.zeros(4))


@pytest.mark.parametrize("coefficient", [np.nan, np.inf, -np.inf])
def test_sparse_spectrum_refuses_nonfinite_native_coefficients(coefficient):
    matrix = native_diagonal(np.ones(2)).with_data([coefficient, 1.0])
    with pytest.raises(ValueError, match="finite"):
        _linalg.symmetric_sparse_spectrum(matrix)


def test_many_components_do_not_hide_the_positive_gap(monkeypatch):
    from rexgraph.sparse_character import _smallest_pos_small_kernel
    n = 1200
    rows = np.arange(n)
    matrix = native_coo(np.concatenate((rows, rows)), np.concatenate((rows, rows ^ 1)),
                        np.concatenate((np.ones(n), -np.ones(n))), (n, n))
    _forbid_numpy_decompositions(monkeypatch)
    assert np.isclose(_smallest_pos_small_kernel(matrix), 2.0)


def test_large_kernel_gap_uses_native_minimum_norm_actions(monkeypatch):
    from rexgraph.sparse_character import _smallest_positive_eig
    matrix = native_diagonal(np.concatenate((np.zeros(518), [0.25, 1.0])))
    importer = builtins.__import__
    def no_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("SciPy forbidden")
        return importer(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_scipy)
    _forbid_numpy_decompositions(monkeypatch)
    assert np.isclose(_smallest_positive_eig(matrix, 520), 0.25, atol=1e-8)


def test_sparse_and_dense_grade_spectra_and_towers_agree(monkeypatch):
    b1 = np.array([[-1, 0, 1], [1, -1, 0], [0, 1, -1]], dtype=float)
    b2 = np.ones((3, 1))
    dense = _l_gb.l_gb_tower([b1, b2])
    sparse = _l_gb.l_gb_tower([as_native(b1), as_native(b2)])
    for got, want in zip(sparse, dense, strict=True):
        for key in ("L_gb", "top_eig", "bot_eig", "spread", "frob", "localization"):
            np.testing.assert_allclose(got[key], want[key], atol=1e-12)
    for grade in (0, 1, 2):
        np.testing.assert_allclose(_l_gb.dirac_spectrum_at_grade(as_native(b1), as_native(b2), grade),
                                   _l_gb.dirac_spectrum_at_grade(b1, b2, grade), atol=1e-12)
    _forbid_numpy_decompositions(monkeypatch)
    _l_gb.l_gb_tower([as_native(b1), as_native(b2)])


def test_channel_mass_direct_output_retains_readonly_signed_weights():
    ptr = np.array([0, 3, 5, 6], dtype=np.int32)
    idx = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)
    weights = np.array([-2.0, 0.0, 3.0])
    positive = _channel_tower.channel_diagonals_any_arity(ptr, idx, 7, abs(weights))
    for a in (ptr, idx, weights):
        a.setflags(write=False)
    observed = _channel_tower.channel_diagonals_any_arity(ptr, idx, 7, weights, threads=2)
    for got, want in zip(observed, positive, strict=True):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(weights, [-2.0, 0.0, 3.0])


@pytest.mark.parametrize("name", ["gemm_nn", "gemm_nt", "gemm_tn"])
def test_native_matrix_products_retain_strided_layout_and_empty_axes(name):
    a = np.arange(24, dtype=float).reshape(4, 6)[:, ::2]
    b = np.arange(30, dtype=float).reshape(6, 5)[::2]
    if name == "gemm_nt":
        b = b.T
    elif name == "gemm_tn":
        a = a.T
    expected = (a.T if name == "gemm_tn" else a) @ (b.T if name == "gemm_nt" else b)
    np.testing.assert_allclose(getattr(_linalg, name)(a, b), expected)
    np.testing.assert_array_equal(getattr(_linalg, name)(np.empty((0, 0)), np.empty((0, 0))), np.empty((0, 0)))


@pytest.mark.parametrize("shape", [(8, 3), (3, 8), (4, 4), (0, 4), (4, 0)])
def test_native_qr_basis_matches_reduced_projection(shape, monkeypatch):
    rng = np.random.default_rng(43)
    a = rng.normal(size=shape)
    expected, _ = np.linalg.qr(a)
    def no_numpy(*args, **kwargs):
        raise AssertionError("NumPy QR reached")
    monkeypatch.setattr(np.linalg, "qr", no_numpy)
    q = _linalg.qr_basis(a)
    assert q.shape == (shape[0], min(shape))
    np.testing.assert_allclose(q @ q.T, expected @ expected.T, atol=1e-12)
    np.testing.assert_allclose(q.T @ q, np.eye(min(shape)), atol=1e-12)
    np.testing.assert_allclose(q @ q.T @ a, a, atol=1e-12)


@pytest.mark.parametrize("columns", [None, 0, 3])
def test_native_solve_retains_input_and_refuses_singular_systems(columns):
    a = np.array([[1., 2.], [3., 5.]])
    b = np.arange(2 if columns is None else 2 * columns, dtype=float)
    if columns is not None:
        b = b.reshape(2, columns)
    original_a, original_b = a.copy(), b.copy()
    a.setflags(write=False)
    b.setflags(write=False)
    np.testing.assert_allclose(_linalg.solve(a, b), np.linalg.solve(a, b), atol=1e-12)
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(b, original_b)
    with pytest.raises(np.linalg.LinAlgError):
        _linalg.solve(np.ones((2, 2)), np.ones(2))
