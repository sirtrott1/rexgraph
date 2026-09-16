"""Paper participation transition, exact actions and compiled solver parity."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.markov import MarkovView, pagerank
from rexgraph.markov_oracle import PairwiseMarkovOracle, pairwise_pagerank_oracle


def fixture():
    return RexGraph.from_cells([5, [[0, 1, 2], [2, 3], [1], [3, 3]]],
                               w_E=[Q(2), Q(3), Q(5), Q(7)])


def oracle():
    # U=abs(B1) W, not support counts. The repeated endpoint column cancels.
    u = np.array([[2, 0, 0, 0], [1, 0, 5, 0], [1, 3, 0, 0],
                  [0, 3, 0, 0], [0, 0, 0, 0]], float)
    a = u/np.where(u.sum(axis=0), u.sum(axis=0), 1)
    b = u/np.where(u.sum(axis=1), u.sum(axis=1), 1)[:, None]
    p = a@b.T
    p[:, 4] = 1/5
    return p


@pytest.mark.parametrize("shape", [(5,), (5, 0), (5, 3)])
def test_tensor_action_and_transpose_match_independent_oracle(shape):
    view = MarkovView(fixture())
    x = np.arange(np.prod(shape)).reshape(shape)
    p = oracle()
    for transpose in (False, True):
        action = view.transpose_apply if transpose else view.apply
        expected = (p.T if transpose else p)@x
        exact = action(x.astype(object), exact=True)
        assert all(isinstance(v, Q) for v in exact.flat)
        np.testing.assert_allclose(np.asarray(exact, float), expected)
        np.testing.assert_allclose(action(x), expected)
        np.testing.assert_allclose(action(x*(1+2j)), expected*(1+2j))
    np.testing.assert_array_equal(view.apply(x.astype(object), exact=True).sum(axis=0), x.sum(axis=0))
    np.testing.assert_array_equal(view.transpose_apply(np.ones(5, object), exact=True), np.ones(5, object))


@pytest.mark.parametrize("damping", [0, Q(1, 2), Q(17, 20), .99])
def test_tensor_fixed_point_and_measured_bound(damping):
    p = oracle()
    seed = np.array([2., 0, 0, 0, 1])
    expected = np.linalg.solve(np.eye(5)-float(damping)*p, (1-float(damping))*seed/3)
    result, info = pagerank(MarkovView(fixture()), damping,
                           seed, maxiter=10000, report=True)
    assert info["kernel"] == "native-tensor-pagerank"
    assert info["converged"] and info["error_bound_l1"] <= 1e-10
    assert np.linalg.norm(result-expected, 1) <= info["error_bound_l1"]+1e-13
    assert result.sum() == pytest.approx(1)


@pytest.mark.parametrize("n", [0, 1, 4])
def test_tensor_empty_incidence_and_dangling(n):
    r = RexGraph.from_cells([n, []])
    view = MarkovView(r)
    x = np.ones(n, object)
    np.testing.assert_array_equal(view.apply(x, exact=True), x)
    np.testing.assert_array_equal(view.transpose_apply(x, exact=True), x)
    np.testing.assert_allclose(pagerank(view), np.full(n, 1/n) if n else np.empty(0))


@pytest.mark.parametrize("arity", [2, 3, 4, 17, 257])
def test_full_arity_shares_and_no_pair_expansion(arity, monkeypatch):
    from rexgraph.native_sparse import NativeSparse
    r = RexGraph.from_cells([arity, [list(range(arity))]])
    def refuse(*a, **kw):
        pytest.fail("pairwise or materialized transition reached")
    monkeypatch.setattr(RexGraph, "_require_pairwise_c1", refuse)
    monkeypatch.setattr(RexGraph, "_ensure_src_tgt", refuse)
    monkeypatch.setattr(NativeSparse, "product", refuse)
    monkeypatch.setattr(NativeSparse, "as_scipy", refuse)
    expected = np.array([Q(1, 2)]+[Q(1, 2*(arity-1))]*(arity-1), object)
    x = np.array([Q(1)]+[Q(0)]*(arity-1), object)
    view = MarkovView(r)
    np.testing.assert_array_equal(view.apply(x, exact=True), expected)
    np.testing.assert_allclose(view.apply(x), np.asarray(expected, float))
    assert not hasattr(view, "adjacency") and view.matrix is None and view.matrix_factory is None
    assert pagerank(view).sum() == pytest.approx(1)


def test_tensor_and_pairwise_are_distinct_even_at_arity_two():
    r = RexGraph.from_cells([3, [[0, 1], [1, 2]]])
    x = np.array([Q(1), Q(0), Q(0)], object)
    pair = PairwiseMarkovOracle(r).apply(x, exact=True)
    tensor = MarkovView(r).apply(x, exact=True)
    np.testing.assert_array_equal(tensor, (x+pair)/2)
    assert not np.array_equal(tensor, pair)


@pytest.mark.parametrize("weight", [Q(0), Q(1, 10**400), Q(10**400)])
def test_exact_normalization_precedes_numeric_conversion(weight):
    r = RexGraph.from_cells([3, [[0, 1, 2]]], w_E=[weight])
    view = MarkovView(r)
    x = np.array([1, 0, 0], object)
    expected = [Q(1, 2), Q(1, 4), Q(1, 4)] if weight else [Q(1, 3)]*3
    np.testing.assert_array_equal(view.apply(x, exact=True), expected)
    np.testing.assert_allclose(view.apply(x), np.asarray(expected, float))


def test_paper_pairwise_exact_field_is_a_numerical_regression_oracle():
    r = RexGraph.from_cells([5, [[i, i+1] for i in range(4)]])
    expected = [Q(438721, 1512560), Q(12461, 37814), Q(289, 1480),
                Q(4913, 37814), Q(83521, 1512560)]
    result = pairwise_pagerank_oracle(PairwiseMarkovOracle(r), Q(17, 20), [1, 0, 0, 0, 0], tol=1e-13)
    np.testing.assert_allclose(result, np.asarray(expected, float), atol=1e-13)


def test_tensor_rejects_negative_metrics_and_unconverged_results():
    with pytest.raises(ValueError, match="nonnegative"):
        MarkovView(RexGraph.from_cells([3, [[0, 1, 2]]], w_E=[-1]))
    with pytest.raises(RuntimeError, match="bound"):
        pagerank(MarkovView(fixture()), maxiter=1, tol=1e-14)


@pytest.mark.parametrize("seed", [[0]*5, [-1, 1, 1, 1, 1], [True]*5, [1j]*5, [1, 2]])
def test_tensor_seed_refusals(seed):
    with pytest.raises((TypeError, ValueError)):
        pagerank(MarkovView(fixture()), seed=seed)


def test_tensor_probability_underflow_keeps_the_exact_action():
    r = RexGraph.from_cells([3, [[0, 1], [0, 2]]], w_E=[Q(1), Q(1, 10**400)])
    view = MarkovView(r)
    x = np.array([1, 0, 0], object)
    exact = view.apply(x, exact=True)
    assert exact[2] > 0 and sum(exact) == 1
    with pytest.raises(FloatingPointError, match="underflow"):
        view.apply(x)


def test_tensor_captured_source_state_is_checked():
    r = fixture()
    view = MarkovView(r)
    pagerank(view)
    r.set_cell_attrs([0, 1, 2, 3], w_E=[1, 1, 1, 1])
    for action in (lambda: view.apply(np.ones(5)), lambda: pagerank(view)):
        with pytest.raises(ValueError, match="state changed"):
            action()
