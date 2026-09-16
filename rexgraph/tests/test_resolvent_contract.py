"""The shifted inverse is not a pseudoinverse; solver success is measured."""
import numpy as np
import pytest

from rexgraph import sparse_character
from rexgraph.graph import RexGraph
from rexgraph.green import GreenOperator
from rexgraph.linear_operator import RexOperator, hodge_operator


def fixture():
    return RexGraph.from_simplicial(np.array([0, 1, 0], np.int32),
        np.array([1, 2, 2], np.int32), np.array([[0, 1, 2]], np.int32))


@pytest.mark.parametrize("grade", [0, 1, 2])
@pytest.mark.parametrize("alpha", [0, .2, 3])
@pytest.mark.parametrize("columns", [None, 0, 3])
def test_resolvent_matches_small_inverse_oracle(grade, alpha, columns):
    rex = fixture()
    op = hodge_operator(rex, grade)
    n = op.shape[0]
    shape = (n,) if columns is None else (n, columns)
    rhs = np.arange(np.prod(shape), dtype=float).reshape(shape) - 2
    matrix = np.eye(n) + alpha * op.as_scipy().toarray()
    expected = np.linalg.solve(matrix, rhs)
    green = GreenOperator.resolvent(op, alpha)
    actual, info = green.solve_with_info(rhs)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual.shape == shape
    assert info["status"] == "observed"
    assert len(info["relative_residuals"]) == (1 if columns is None else columns)
    assert all(r <= 1e-10 for r in info["relative_residuals"])
    assert dict(green.parameters)["alpha"] == alpha


def test_alpha_zero_does_not_evaluate_operator_or_solver(monkeypatch):
    def forbidden(*args, **kw):
        pytest.fail("zero-alpha inverse is identity without any operator action")
    rex = fixture()
    op = RexOperator("zero-test", (3, 3), 1, 1, forbidden, source=rex, symmetric=True, psd=True)
    monkeypatch.setattr(sparse_character, "_block_cg", forbidden)
    rhs = np.array([1., 2., 3.])
    actual, info = GreenOperator.resolvent(op, 0).solve_with_info(rhs)
    np.testing.assert_array_equal(actual, rhs)
    assert actual is not rhs
    assert info["kernel"] == "identity"


@pytest.mark.parametrize("options", [{"alpha": -1}, {"alpha": True}, {"alpha": float("inf")},
    {"tol": 0}, {"tol": 1}, {"tol": float("nan")}, {"tol": True},
    {"maxiter": 0}, {"maxiter": 1.5}, {"maxiter": True}])
def test_invalid_solver_controls_refuse(options):
    with pytest.raises((ValueError, TypeError)):
        GreenOperator.resolvent(hodge_operator(fixture(), 1), **options)


@pytest.mark.parametrize("magnitude", [1., 1e200, 1e-200])
def test_false_success_does_not_pass_overflow_or_underflow_residuals(monkeypatch, magnitude):
    op = hodge_operator(fixture(), 1)
    monkeypatch.setattr(sparse_character, "_block_cg", lambda a, b, d, **kw: (np.zeros_like(b), {}))
    with pytest.raises(RuntimeError, match="measured residual tolerance"):
        GreenOperator.resolvent(op).solve(np.full(3, magnitude))


def test_solver_nonconvergence_is_not_returned_as_a_field(monkeypatch):
    def nonconvergent(*args, **kwargs):
        raise ArithmeticError("iteration limit")
    monkeypatch.setattr(sparse_character, "_block_cg", nonconvergent)
    with pytest.raises(RuntimeError, match="did not converge"):
        GreenOperator.resolvent(hodge_operator(fixture(), 1)).solve(np.ones(3))


@pytest.mark.parametrize("rhs", [np.full(3, 1j), np.full(3, np.nan), np.full(3, np.inf)])
def test_invalid_rhs_is_not_silently_cast(rhs):
    with pytest.raises((ValueError, TypeError)):
        GreenOperator.resolvent(hodge_operator(fixture(), 1)).solve(rhs)


def test_present_empty_grade_returns_empty_without_cg(monkeypatch):
    rex = RexGraph.from_graph([], [])
    monkeypatch.setattr(sparse_character, "_block_cg", lambda *a, **kw: pytest.fail("empty solve ran CG"))
    value, info = GreenOperator.resolvent(hodge_operator(rex, 1)).solve_with_info(np.empty(0))
    assert value.shape == (0,) and info["kernel"] == "empty-zero"
