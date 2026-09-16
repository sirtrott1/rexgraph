"""Native numerical actions compared with small dense reference oracles."""
import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.core._hodge import least_squares
from rexgraph.fiedler import kernel_from_boundary, minimum_norm_gram_solve
from rexgraph.green import vertex_green
from rexgraph.harmonic_sparse import harmonic_basis, harmonic_coordinates
from rexgraph.native_sparse import as_native, native_coo


@pytest.mark.parametrize("shape,rank", [((9, 4), 4), ((4, 9), 4), ((8, 7), 3),
                                      ((0, 3), 0), ((4, 0), 0), ((3, 3), 0)])
@pytest.mark.parametrize("transpose", [False, True])
def test_native_lsqr_is_the_minimum_norm_solution(shape, rank, transpose):
    rng = np.random.default_rng(812)
    a = rng.normal(size=(shape[0], rank)) @ rng.normal(size=(rank, shape[1]))
    matrix = a.T if transpose else a
    b = rng.normal(size=(matrix.shape[0], 3))
    expected = np.linalg.pinv(matrix) @ b
    actual, info = least_squares(as_native(a), b, transpose=transpose, return_info=True)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
    assert info["kernel"] == "native-lsqr"
    assert all(c["relative_residual"] <= info["tol"] or c["normal_residual"] <= info["tol"]
               for c in info["columns"])


@pytest.mark.parametrize("scale", [1e-200, 1., 1e200])
def test_native_lsqr_preserves_common_coefficient_scaling(scale):
    a = np.array([[1., 0], [1., -1], [0., 1.]])
    b = np.array([2., 1., 3.])
    np.testing.assert_allclose(least_squares(as_native(a * scale), b * scale),
                               np.linalg.lstsq(a, b, rcond=None)[0], atol=1e-11)


def test_native_lsqr_refuses_exhausted_or_invalid_solves():
    a = as_native(np.diag([1., 2., 3.]))
    with pytest.raises(ArithmeticError, match="converge"):
        least_squares(a, np.ones(3), maxiter=1)
    for tol in (0, -1, 1, np.nan, True):
        with pytest.raises(ValueError, match="tolerance"):
            least_squares(a, np.ones(3), tol=tol)
    for maxiter in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="maxiter"):
            least_squares(a, np.ones(3), maxiter=maxiter)
    with pytest.raises((ValueError, FloatingPointError)):
        least_squares(a, [1., np.inf, 0])
    with pytest.raises(TypeError, match="real"):
        least_squares(a, np.ones(3, complex))


def test_two_factor_green_keeps_every_kernel_direction():
    a = np.array([[-1., 0], [0.5, -1], [0.5, 0.5], [0, 0.5], [0, 0]])
    rhs = np.arange(15.).reshape(5, 3)
    expected = np.linalg.pinv(a @ a.T) @ rhs
    actual = minimum_norm_gram_solve(as_native(a), rhs)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(minimum_norm_gram_solve(as_native(a), np.ones(5)), 0, atol=1e-12)


def test_witness_kernel_is_structural_even_for_small_coefficients():
    a = as_native(np.array([[-1., 0], [1., 0], [0, 1e-30], [0, 0]]))
    kernel, count = kernel_from_boundary(a, native=True)
    assert count == 2
    np.testing.assert_allclose(a.T.product(kernel).data, 0, atol=0)
    assert kernel.shape == (4, 2)


def test_native_sparse_products_coalesce_duplicates_and_cancellation():
    a = native_coo([0, 0, 1, 1], [0, 0, 1, 2], [2, -1, 3, -2], (3, 4))
    b = as_native(np.arange(8.).reshape(4, 2))
    expected = np.array([[1, 0, 0, 0], [0, 3, -2, 0], [0, 0, 0, 0]])
    np.testing.assert_allclose(a.product(b).apply(np.eye(2)), expected @ np.arange(8.).reshape(4, 2))
    np.testing.assert_allclose(a.row_inner(a), np.sum(expected**2, axis=1))
    np.testing.assert_allclose(a.add(a, -1).apply(np.ones(4)), 0, atol=0)


def test_winding_does_not_round_or_overflow_integer_signals():
    from rexgraph.harmonic_sparse import harmonic_winding
    frame = as_native(np.array([[1.], [1.], [-1.]]))
    huge = 2**90 + 1
    actual = harmonic_winding(frame, np.array([huge, huge, -huge], dtype=object))
    assert actual.tolist() == [3 * huge]
    assert harmonic_winding(frame, [2**53 + 1, -2**53, 0]).tolist() == [1]


@pytest.mark.parametrize("rows,values", [([0.5], [1.]), ([True], [1.]),
    ([0], [1j]), ([0], [np.inf]), ([2**64 - 1], [1.])])
def test_native_coo_refuses_invalid_addresses_and_coefficients(rows, values):
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        native_coo(rows, [0], values, (1, 1))


@pytest.mark.parametrize("rex", [RexGraph.from_graph([0, 1, 2], [1, 2, 0]),
    RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]]),
    RexGraph.from_hypergraph([0, 3, 5, 7], [0, 1, 2, 0, 1, 1, 2])])
def test_native_frames_preserve_the_existing_coordinates(rex):
    reference = harmonic_basis(rex)
    native = harmonic_basis(rex, native=True)
    np.testing.assert_array_equal(native.apply(np.eye(native.shape[1])), reference.toarray())
    flow = np.arange(rex.nE, dtype=float)
    np.testing.assert_allclose(harmonic_coordinates(native, flow),
                               np.linalg.pinv(reference.toarray()) @ flow, atol=1e-10)
    expected = np.linalg.pinv(np.asarray(rex.B1) @ np.asarray(rex.B1).T)
    np.testing.assert_allclose(vertex_green(rex).solve(np.eye(rex.nV)), expected, atol=1e-10)


@pytest.mark.parametrize("kind", ["empty", "cycle", "filled", "branching", "loop"])
def test_native_malaugh_moments_match_the_boundary_oracle(kind):
    from rexgraph.core._sparse import to_scipy_csr
    from rexgraph.scale_propagator import malaugh_quantities

    if kind == "empty":
        rex = RexGraph(boundary_ptr=[0], boundary_idx=[])
    elif kind == "filled":
        rex = RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])
    elif kind == "branching":
        rex = RexGraph.from_hypergraph([0, 3, 5, 7], [0, 1, 2, 0, 1, 1, 2])
    elif kind == "loop":
        rex = RexGraph.from_hypergraph([0, 2, 4], [0, 0, 0, 1])
    else:
        rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    rex._ensure_clean()
    lower = to_scipy_csr(rex._B1_dual).toarray()
    upper = (to_scipy_csr(rex._B2_hodge_dual).toarray()
             if rex.nF_hodge else np.zeros((rex.nE, 0)))
    t, u = lower.T @ lower, upper @ upper.T
    tr_t, tr_u = np.trace(t), np.trace(u)
    tr_t2, tr_u2 = np.sum(t * t), np.sum(u * u)
    concentration_t = tr_t2 / tr_t**2 if tr_t else np.nan
    concentration_u = tr_u2 / tr_u**2 if tr_u else np.nan
    expected = {
        "c2_H": concentration_t / concentration_u,
        "c2_H_inv": concentration_u / concentration_t,
        "c2": concentration_u / concentration_t,
        "c2_E": tr_u2 / tr_t2 if tr_t2 else np.nan,
        "H_T": -np.log(concentration_t),
        "H_S": -np.log(concentration_u),
    }
    for weights in (None, np.arange(1., rex.nE + 1)):
        rex._w_E = weights
        actual = malaugh_quantities(rex)
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name], value, atol=1e-12, equal_nan=True)
