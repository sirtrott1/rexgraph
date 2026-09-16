"""Native compatibility actions and independent small PageRank oracles."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.core import _standard
from rexgraph.io.catalog import object_digest
from rexgraph.markov_oracle import PairwiseMarkovOracle, pairwise_pagerank_oracle


def fixture():
    return RexGraph.from_cells([4, [[0, 1], [1, 2], [0, 1], [2, 2]], []],
                               w_E=[Q(1, 3), Q(2), Q(3, 7), Q(4)])


def oracle(ptr, idx, weights, n):
    a = np.zeros((n, n))
    for u in range(n):
        for k in range(ptr[u], ptr[u+1]):
            a[idx[k], u] += weights[k]
    for u in range(n):
        mass = a[:, u].sum()
        a[:, u] = a[:, u]/mass if mass else np.full(n, 1/n)
    return a


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("steps", [1, 2, 3, 4])
def test_result_buffer_matches_each_explicit_iteration(dtype, steps):
    ptr, idx = np.array([0, 1, 3, 4], dtype), np.array([1, 0, 2, 1], dtype)
    weights = np.array([1., 1., 2., 2.])
    p = oracle(ptr, idx, weights, 3)
    expected = np.full(3, 1/3)
    for _ in range(steps):
        expected = .85*(p@expected)+.15/3
    result, info = _standard.pagerank(ptr, idx, weights, 3, 2, max_iter=steps, tol=1e-14, report=True)
    np.testing.assert_allclose(result, expected, atol=1e-15)
    assert info["iterations"] == steps
    assert info["residual_l1"] == pytest.approx(np.abs(.85*p@result+.15/3-result).sum())


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("weight", [0., 1e-320, 1e-250, 1., 1e250])
def test_dangling_mass_and_small_weights_have_no_threshold(dtype, weight):
    ptr, idx = np.array([0, 1, 2, 2], dtype), np.array([1, 0], dtype)
    weights = np.array([weight, weight])
    seed = np.array([1., 0., 0.])
    result, info = _standard.pagerank(ptr, idx, weights, 3, 1, seed=seed, max_iter=1000, report=True)
    p = oracle(ptr, idx, weights, 3)
    expected = np.linalg.solve(np.eye(3)-.85*p, .15*seed)
    np.testing.assert_allclose(result, expected, atol=1e-8)
    assert abs(result.sum()-1) < 1e-14 and info["converged"]
    assert np.abs(result-expected).sum() <= info["error_bound_l1"]+1e-14


@pytest.mark.parametrize("shape", [(4,), (4, 0), (4, 2)])
def test_exact_transition_transpose_and_quadrature(shape):
    r = fixture()
    view = PairwiseMarkovOracle(r)
    x = np.arange(np.prod(shape)).reshape(shape).astype(object)
    y = view.apply(x, exact=True)
    np.testing.assert_array_equal(y.sum(axis=0), x.sum(axis=0))
    ones = np.full(shape, Q(1), object)
    np.testing.assert_array_equal(view.transpose_apply(ones, exact=True), ones)
    assert sum((a*b for a, b in zip(ones.flat, y.flat, strict=True)), Q(0)) == sum(
        (a*b for a, b in zip(view.transpose_apply(ones, exact=True).flat, x.flat, strict=True)), Q(0))
    p = oracle(*view.adjacency(), 4)
    np.testing.assert_allclose(view.apply(np.asarray(x, float)), p@np.asarray(x, float))
    np.testing.assert_allclose(view.transpose_apply(np.asarray(x, float)), p.T@np.asarray(x, float))
    np.testing.assert_allclose(np.asarray(y, float), p@np.asarray(x, float))


def test_kernel_reuse_and_no_matrix_or_spectral_export(monkeypatch):
    from rexgraph.linear_operator import RexOperator
    view = PairwiseMarkovOracle(fixture())
    def refuse(*a, **kw):
        pytest.fail("dense, SciPy or spectral export reached")
    for name in ("as_scipy", "as_native"):
        monkeypatch.setattr(RexOperator, name, refuse)
    for name in ("solve", "eig", "eigh", "inv"):
        monkeypatch.setattr(np.linalg, name, refuse)
    result, info = pairwise_pagerank_oracle(view, report=True)
    assert info["kernel"] == "native-pagerank" and result.sum() == pytest.approx(1)
    assert info["error_bound_l1"] <= 1e-10


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_readonly_buffers_and_directed_scatter(dtype):
    ptr, idx = np.array([0, 1, 2, 2], dtype), np.array([1, 2], dtype)
    weights = np.ones(2)
    for a in (ptr, idx, weights):
        a.setflags(write=False)
    result, info = _standard.pagerank(ptr, idx, weights, 3, 2, report=True)
    p = oracle(ptr, idx, weights, 3)
    np.testing.assert_allclose(result, np.linalg.solve(np.eye(3)-.85*p, np.full(3, .15/3)), atol=1e-8)
    assert info["converged"]


def test_degree_overflow_uses_scaled_normalization():
    ptr, idx = np.array([0, 2, 2, 2], np.int32), np.array([1, 2], np.int32)
    probabilities, dangling = _standard.markov_weights(ptr, idx, np.array([1e308, 1e308]), 3)
    np.testing.assert_array_equal(probabilities, [.5, .5])
    np.testing.assert_array_equal(dangling, [0, 1, 1])


def test_complex_observables_and_exact_rational_weights():
    view = PairwiseMarkovOracle(fixture())
    x = np.array([1+2j, 2-3j, -2j, 4+1j])
    p = oracle(*view.adjacency(), 4)
    np.testing.assert_allclose(view.apply(x), p@x)
    np.testing.assert_allclose(view.transpose_apply(x), p.T@x)
    expected = Q(1, 3)/(Q(1, 3)+Q(3, 7)+Q(2)) + Q(3, 7)/(Q(1, 3)+Q(3, 7)+Q(2))
    assert view.apply(np.array([0, 1, 0, 0], object), exact=True)[0] == expected


@pytest.mark.parametrize("weight", [-1, Q(-1, 3)])
def test_signed_metric_does_not_become_a_stochastic_weight(weight):
    with pytest.raises(ValueError, match="nonnegative"):
        PairwiseMarkovOracle(RexGraph.from_cells([2, [[0, 1]]], w_E=[weight]))


def test_unrepresentable_metric_retains_exact_action_without_becoming_dangling():
    view = PairwiseMarkovOracle(RexGraph.from_cells([2, [[0, 1]]], w_E=[Q(1, 10**400)]))
    np.testing.assert_array_equal(view.apply(np.array([1, 0], object), exact=True), [0, 1])
    with pytest.raises(FloatingPointError, match="underflows"):
        pairwise_pagerank_oracle(view)


@pytest.mark.parametrize("cells", [[[0, 1, 2]], [[0]]])
def test_primary_branching_and_witnesses_are_not_erased(cells):
    with pytest.raises(ValueError, match="pairwise"):
        PairwiseMarkovOracle(RexGraph.from_cells([3, cells]))


def test_source_tower_and_state_are_retained():
    face = [(0, 1), (1, 1), (2, -1)]
    diff = [(0, 1), (1, -1)]
    r = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]], [face, face], [diff, diff], [diff]])
    digest = object_digest(r)
    view = PairwiseMarkovOracle(r)
    pairwise_pagerank_oracle(view)
    assert object_digest(r) == digest and view.source is r
    r.set_cell_attrs([0, 1, 2], w_E=[2, 3, 4])
    with pytest.raises(ValueError, match="state changed"):
        view.apply(np.ones(3))
    with pytest.raises(ValueError, match="state changed"):
        pairwise_pagerank_oracle(view)


def test_exhaustion_is_not_reported_as_a_converged_field():
    with pytest.raises(RuntimeError, match="error bound"):
        pairwise_pagerank_oracle(PairwiseMarkovOracle(fixture()), maxiter=1)


@pytest.mark.parametrize("n", [0, 1, 3])
def test_empty_and_entirely_isolated_sources(n):
    view = PairwiseMarkovOracle(RexGraph.from_cells([n, []]))
    expected = np.full(n, 1/n) if n else np.empty(0)
    np.testing.assert_allclose(pairwise_pagerank_oracle(view), expected)
    np.testing.assert_array_equal(view.apply(np.ones(n, object), exact=True), np.ones(n, object))


@pytest.mark.parametrize("change", ["negative", "nan", "pointer", "index", "dtype", "damping", "tol", "maxiter", "seed"])
def test_unsafe_kernel_inputs_are_refused_before_unchecked_loops(change):
    ptr, idx, w = np.array([0, 1, 2], np.int32), np.array([1, 0], np.int32), np.ones(2)
    kw = {}
    if change == "negative": w[0] = -1
    if change == "nan": w[0] = np.nan
    if change == "pointer": ptr[1] = 9
    if change == "index": idx[0] = 2
    if change == "dtype": idx = idx.astype(float)
    if change == "damping": kw["damping"] = 1
    if change == "tol": kw["tol"] = 0
    if change == "maxiter": kw["max_iter"] = True
    if change == "seed": kw["seed"] = [0, 0]
    with pytest.raises((ValueError, TypeError)):
        _standard.pagerank(ptr, idx, w, 2, 1, **kw)
