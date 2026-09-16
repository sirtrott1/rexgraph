"""Metric Green actions: independent small inverses and sparse only execution."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph import sparse_character
from rexgraph.cells import cell_count
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.green import GreenOperator
from rexgraph.linear_operator import RexOperator
from rexgraph.weighted_hodge import weighted_hodge


def make(case="branching"):
    if case == "branching":
        return RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2])
    if case == "witness":
        return RexGraph.from_hypergraph([0, 1, 3], [0, 0, 1])
    if case == "parallel":
        return RexGraph.from_graph([0, 0, 1], [1, 1, 2])
    if case == "face":
        return RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])
    return RexGraph.from_cells(solid_octahedron_3rex())


def hand_operator():
    r = make()
    return weighted_hodge(r, 1, metric=DiagonalMetric(r, 1, (2, 3)),
                          lower_metric=DiagonalMetric(r, 0, (1, 2, 3, 4)))


@pytest.mark.parametrize("case", ["branching", "witness", "parallel", "face", "grade3"])
@pytest.mark.parametrize("sector", ["down", "up", "sum"])
@pytest.mark.parametrize("columns", [None, 0, 3])
def test_all_grades_match_independent_coordinate_oracle(case, sector, columns):
    r = make(case)
    boundaries = r.graded_boundaries()
    metrics = [DiagonalMetric(r, k, tuple(Q(i+2, i+1) for i in range(cell_count(r, k))))
               for k in range(len(boundaries)+1)]
    for k, metric in enumerate(metrics):
        m = np.asarray(metric.weights, float)
        n = len(m)
        matrix = np.zeros((n, n))
        kw = {"metric": metric}
        if k and sector != "up":
            b = boundaries[k-1].toarray()
            matrix += (b.T * np.asarray(metrics[k-1].weights, float)) @ b / m[:, None]
            kw["lower_metric"] = metrics[k-1]
        if k < len(boundaries) and sector != "down":
            b = boundaries[k].toarray()
            matrix += (b / np.asarray(metrics[k+1].weights, float)) @ b.T * m
            kw["upper_metric"] = metrics[k+1]
        shape = (n,) if columns is None else (n, columns)
        rhs = np.arange(int(np.prod(shape)), dtype=float).reshape(shape)-2
        expected = np.linalg.solve(np.eye(n)+.4*matrix, rhs)
        green = GreenOperator.resolvent(weighted_hodge(r, k, sector=sector, **kw), .4)
        out, info = green.solve_with_info(rhs)
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=2e-10)
        assert out.shape == shape
        assert green.metric is metric
        assert info["metric_digest"] == metric.coefficient_digest
        assert info["residual_norm"] == "euclidean-original-system"
        assert all(v <= 1e-10 for v in info["relative_residuals"])


def test_hand_inverse_and_green_metric_contractions_are_not_euclidean():
    op = hand_operator()
    green = GreenOperator.resolvent(op)
    rhs = np.array([3, 2])
    np.testing.assert_allclose(green.solve(rhs), [93/44, 41/22])
    assert green.quadrance(rhs) == pytest.approx(525/22)
    expected = np.array([[21/22, 15/22], [15/22, 39/22]])
    np.testing.assert_allclose(green.gram(np.eye(2)), expected)
    np.testing.assert_allclose(green.quadrance(np.eye(2)), np.diag(expected))
    assert green.spread(np.eye(2))[0, 1] == pytest.approx(1-225/(21*39))
    inverse = green.solve(np.eye(2))
    assert not np.allclose(inverse, inverse.T)
    np.testing.assert_allclose(np.array([2, 3])[:, None]*inverse, expected)


def test_cg_receives_the_symmetric_metric_system_and_inverse_metric_preconditioner(monkeypatch):
    op = hand_operator()
    original = sparse_character._block_cg
    def checked(a, b, dinv, **kw):
        matrix = np.column_stack([a(e) for e in np.eye(2)])
        np.testing.assert_allclose(a(np.eye(2)), matrix)
        np.testing.assert_allclose(matrix, [[52/27, -10/9], [-10/9, 7/3]])
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(dinv, [1.5, 1.])
        np.testing.assert_allclose(b, [[2/3], [2/3]])
        assert kw["tol"] == pytest.approx(1e-10*2/3)
        return original(a, b, dinv, **kw)
    monkeypatch.setattr(sparse_character, "_block_cg", checked)
    np.testing.assert_allclose(GreenOperator.resolvent(op).solve(np.array([3., 2.])), [93/44, 41/22])


@pytest.mark.parametrize("scale", [Q(1, 10**100), Q(10**100)])
def test_common_metric_rescaling_preserves_operator_and_inverse(scale):
    op = hand_operator()
    r = op.source
    scaled = weighted_hodge(r, 1,
        metric=DiagonalMetric(r, 1, tuple(scale*w for w in op.grade_metric.weights)),
        lower_metric=DiagonalMetric(r, 0, tuple(scale*w for w in op.lower_metric.weights)))
    np.testing.assert_allclose(GreenOperator.resolvent(scaled).solve(np.array([3., 2.])), [93/44, 41/22])


def test_weighted_kernel_is_preserved_not_euclidean_constant_or_deflated():
    r = make("parallel")
    m = DiagonalMetric(r, 0, (2, 3, 5))
    op = weighted_hodge(r, 0, metric=m)
    kernel = 1/np.array(m.weights, float)
    np.testing.assert_allclose(op.apply(kernel), 0, atol=1e-15)
    assert not np.allclose(op.apply(np.ones(3)), 0)
    np.testing.assert_allclose(GreenOperator.resolvent(op, 100).solve(kernel), kernel)


def test_changed_source_population_is_refused_even_at_alpha_zero(monkeypatch):
    op = hand_operator()
    green = GreenOperator.resolvent(op, 0)
    monkeypatch.setattr("rexgraph.graded_metric.cell_count", lambda *a, **kw: 3)
    with pytest.raises(ValueError, match="population changed"):
        green.solve(np.ones(2))


@pytest.mark.parametrize("scale", [1., 1e200, 1e-200])
def test_large_and_tiny_rhs_are_solved_without_cg_norm_overflow(scale):
    out = GreenOperator.resolvent(hand_operator()).solve(np.array([3., 2.])*scale)
    np.testing.assert_allclose(out/scale, [93/44, 41/22])


@pytest.mark.parametrize("scale", [1., 1e200, 1e-200])
def test_false_success_checked_in_original_not_metric_weighted_equation(monkeypatch, scale):
    monkeypatch.setattr(sparse_character, "_block_cg", lambda a, b, d, **kw: (np.zeros_like(b), {}))
    with pytest.raises(RuntimeError, match="measured residual"):
        GreenOperator.resolvent(hand_operator()).solve(np.array([3., 2.])*scale)


def test_wrong_small_metric_coordinate_is_not_hidden_by_weighted_residual(monkeypatch):
    r = make()
    op = weighted_hodge(r, 1, sector="up", metric=DiagonalMetric(r, 1, (Q(1, 10**12), 1)))
    # L=0, so x=b. A weighted residual alone would hide error in coordinate 0.
    monkeypatch.setattr(sparse_character, "_block_cg", lambda a, b, d, **kw: (np.array([[0.], [1.]]), {}))
    with pytest.raises(RuntimeError, match="measured residual"):
        GreenOperator.resolvent(op).solve(np.ones(2))


def test_solver_observations_are_per_invocation_and_nonconvergence_refuses(monkeypatch):
    green = GreenOperator.resolvent(hand_operator())
    _, first = green.solve_with_info(np.ones(2))
    _, second = green.solve_with_info(np.zeros(2))
    assert first["iterations"][0] > 0 and second["iterations"] == [0]
    def nonconvergent(*a, **kw):
        raise ArithmeticError("iteration limit")
    monkeypatch.setattr(sparse_character, "_block_cg", nonconvergent)
    with pytest.raises(RuntimeError, match="did not converge"):
        green.solve(np.ones(2))


def test_indefinite_difference_is_not_mistaken_for_psd_even_at_alpha_zero():
    op = weighted_hodge(make("face"), 1, sector="difference")
    with pytest.raises(ValueError, match="positive semidefinite"):
        GreenOperator.resolvent(op, 0)


@pytest.mark.parametrize("values", [np.ones(3), np.ones((2, 1, 1)), np.full(2, 1j),
    np.array([True, False]), np.array(["1", "2"]), np.full(2, np.nan), np.full(2, np.inf)])
def test_invalid_inputs_fail_even_for_identity(values):
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        GreenOperator.resolvent(hand_operator(), 0).solve(values)


@pytest.mark.parametrize("weights", [(Q(10**400), 1), (Q(1, 10**400), 1), (1e-200, 1e200)])
def test_unrepresentable_numeric_metric_refuses_without_rationalizing(weights):
    r = make()
    op = weighted_hodge(r, 1, metric=DiagonalMetric(r, 1, weights))
    with pytest.raises(FloatingPointError):
        GreenOperator.resolvent(op).solve(np.ones(2))


def test_identity_does_not_evaluate_factors_or_cg(monkeypatch):
    def forbidden(*args, **kw):
        pytest.fail("identity evaluated an operator or solver")
    op = replace(hand_operator(), matvec=forbidden)
    monkeypatch.setattr(sparse_character, "_block_cg", forbidden)
    rhs = np.array([Q(3), Q(2)])
    out, info = GreenOperator.resolvent(op, 0).solve_with_info(rhs)
    np.testing.assert_array_equal(out, [3., 2.])
    assert out is not rhs and info["kernel"] == "identity"


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0)])
def test_empty_spaces_and_blocks_do_not_run_cg(shape, monkeypatch):
    r = RexGraph.from_graph([], []) if shape[0] == 0 else make()
    monkeypatch.setattr(sparse_character, "_block_cg", lambda *a, **kw: pytest.fail("empty CG"))
    out, info = GreenOperator.resolvent(weighted_hodge(r, 1)).solve_with_info(np.empty(shape))
    assert out.shape == shape and info["kernel"] == "empty-zero"


def test_large_native_path_never_materializes_matrix(monkeypatch):
    n = 4096
    r = RexGraph.from_graph(np.arange(n), np.arange(1, n+1))
    op = weighted_hodge(r, 1, metric=DiagonalMetric(r, 1, tuple(2+i%4 for i in range(n))))
    def forbidden(*a, **kw):
        pytest.fail("metric resolvent materialized a matrix or eigenbasis")
    for name in ("eigh", "eigvalsh", "inv", "solve"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    monkeypatch.setattr(RexOperator, "as_scipy", forbidden)
    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    x = np.linspace(-1, 1, n)
    out = GreenOperator.resolvent(op, .5).solve(x)
    np.testing.assert_allclose(out+.5*op.apply(out), x, rtol=1e-8, atol=1e-9)
