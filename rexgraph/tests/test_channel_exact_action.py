"""Full rational channels use sparse incidence, with dense oracles only here."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.channel_operator import channel_operator
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import RexOperator, metric_adjoint
from rexgraph.operator_bracket import operator_bracket


def make(cells, weights=None, **kw):
    return RexGraph(boundary_ptr=np.array([0, *np.cumsum([len(c) for c in cells])], np.int32),
                    boundary_idx=np.array([v for c in cells for v in c], np.int32),
                    w_E=None if weights is None else np.array(weights, dtype=object), **kw)


def reference(cells, weights, count=False):
    """Independent small pairwise entry oracle, not a production algorithm."""
    n = len(cells)
    b = [{v: Q(1) if len(c) == 1 else Q(-1) if j == 0 else Q(1, len(c)-1)
          for j, v in enumerate(c)} for c in cells]
    t, g, c = [np.full((n, n), Q(0), dtype=object) for _ in range(3)]
    for i, a in enumerate(b):
        for j, d in enumerate(b):
            common = a.keys() & d.keys()
            t[i, j] = weights[i]*weights[j]*sum((a[v]*d[v] for v in common), Q(0))
            g[i, j] = weights[i]*weights[j]*sum((abs(a[v]*d[v]) for v in common), Q(0))
            if i != j:
                c[i, j] = -sum((Q(1) if count else abs(a[v]*d[v]) for v in common), Q(0))
    f = t-g
    for i in range(n):
        f[i, i] = sum((abs(f[i, j]) for j in range(n) if i != j), Q(0))
        c[i, i] = -sum(c[i], Q(0))
    return dict(zip("TGFC", (t, g, f, c), strict=True))


CASES = [
    ([(0, 1), (1, 2), (2, 0)], [Q(2, 3), Q(-3), Q(0)]),
    ([(0,), (0, 1), (1, 0, 2), (2, 0, 1, 3)], [Q(-2), Q(3), Q(1, 5), Q(4, 7)]),
    ([(0, 1, 2, 3, 4), (0, 1, 2, 3, 4), (6,)], [Q(1), Q(2), Q(3)]),
]


@pytest.mark.parametrize("cells,weights", CASES)
@pytest.mark.parametrize("name", list("TGFC"))
@pytest.mark.parametrize("count", [False, True])
@pytest.mark.parametrize("width", [None, 0, 2])
def test_full_channel_and_transpose_against_independent_rational_oracle(cells, weights, name, count, width):
    r = make(cells, weights, c_channel="count" if count else "share")
    shape = (len(cells),) if width is None else (len(cells), width)
    x = np.asarray([Q(i-2, 7) for i in range(int(np.prod(shape)))], object).reshape(shape)
    oracle = reference(cells, weights, count)[name]
    op = channel_operator(r, name)
    assert op.exact_matvec is not None and op.exact_transpose_matvec is not None
    y = op.apply(x, exact=True)
    np.testing.assert_array_equal(y, oracle@x)
    np.testing.assert_array_equal(op.transpose_apply(x, exact=True), oracle.T@x)
    np.testing.assert_array_equal(op.diagonal(exact=True), np.diag(oracle))
    np.testing.assert_allclose(op.apply(np.asarray(x, float)), np.asarray(y, float), atol=1e-12)
    assert all(isinstance(v, Q) for v in y.flat)


@pytest.mark.parametrize("name", list("TGFC"))
def test_exact_path_never_calls_numerical_factories_or_materializes_hub(name, monkeypatch):
    n = 4096
    r = RexGraph.from_graph(np.zeros(n, int), np.arange(1, n+1))
    def forbidden(*args, **kw):
        pytest.fail("exact channel went through numerical or assembled matrices")
    monkeypatch.setattr("rexgraph.channel_operator.build_factored_operator", forbidden)
    monkeypatch.setattr("rexgraph.channel_operator.build_sparse_channels", forbidden)
    monkeypatch.setattr(RexOperator, "as_scipy", forbidden)
    monkeypatch.setattr(RexGraph, "B1", property(forbidden))
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    op = channel_operator(r, name)
    result = op.apply(np.ones(n, int), exact=True)
    np.testing.assert_array_equal(result, np.full(n, Q(n+1 if name in "TG" else 0), dtype=object))


@pytest.mark.parametrize("weight", [Q(1, 10**400), Q(10**400)])
@pytest.mark.parametrize("name", list("TGFC"))
def test_exact_weights_and_results_need_not_fit_float64(weight, name, monkeypatch):
    cells, weights = [(0, 1), (1, 2)], [weight, -weight]
    r = make(cells, weights)
    def forbidden(*args, **kw):
        pytest.fail("exact action read the float relation metric")
    monkeypatch.setattr(RexGraph, "edge_metric", property(forbidden))
    x = np.array([1, 2])
    op = channel_operator(r, name)
    np.testing.assert_array_equal(op.apply(x, exact=True), reference(cells, weights)[name]@x)


def test_float_weight_means_its_stored_binary_value_not_a_guessed_decimal():
    r = make([(0, 1)], [0.1])
    y = channel_operator(r, "T").apply(np.array([1]), exact=True)[0]
    assert y == 2*Q(0.1)**2 and y != Q(1, 50)


@pytest.mark.parametrize("name", list("TFC"))
def test_normalized_G_selection_does_not_disable_other_exact_channels(name):
    cells, weights = [(0, 1, 2), (1, 0, 2)], [Q(2), Q(3)]
    r = make(cells, weights, g_channel="normalized")
    x = np.array([1, -2])
    op = channel_operator(r, name)
    np.testing.assert_array_equal(op.apply(x, exact=True), reference(cells, weights)[name]@x)


def test_normalized_G_keeps_its_exact_diagonal_without_false_rational_full_action():
    r = make([(0, 1, 2), (1, 0, 2)], [2, 3], g_channel="normalized")
    op = channel_operator(r, "G")
    # Off diagonal is -5/sqrt(126), irrational; the diagonal remains rational.
    assert op.diagonal(exact=True).tolist() == [Q(5, 9), Q(5, 14)]
    assert op.exact_matvec is None and op.exact_transpose_matvec is None
    with pytest.raises(TypeError, match="certified exact"):
        op.apply(np.array([1, 0]), exact=True)
    with pytest.raises(TypeError, match="exact"):
        metric_adjoint(op).apply(np.array([1, 0]), exact=True)


def test_signed_weights_conjugate_F_and_do_not_change_C():
    cells = [(0, 1), (1, 2), (2, 0)]
    positive = make(cells, [1, 2, 3])
    signed = make(cells, [1, -2, 3])
    sign, x = np.array([1, -1, 1]), np.array([2, -3, 5])
    for name in "TGF":
        np.testing.assert_array_equal(channel_operator(signed, name).apply(x, exact=True),
            sign*channel_operator(positive, name).apply(sign*x, exact=True))
    np.testing.assert_array_equal(channel_operator(signed, "C").apply(x, exact=True),
                                 channel_operator(positive, "C").apply(x, exact=True))


def test_brackets_and_adjoint_receive_the_real_exact_hooks():
    cells, weights = CASES[1]
    r = make(cells, weights)
    a, b = channel_operator(r, "T"), channel_operator(r, "F")
    oracle = reference(cells, weights)
    x = np.arange(4)
    k = operator_bracket(a, b)
    matrix = oracle["T"]@oracle["F"]-oracle["F"]@oracle["T"]
    np.testing.assert_array_equal(k.apply(x, exact=True), matrix@x)
    np.testing.assert_array_equal(metric_adjoint(k).apply(x, exact=True), matrix.T@x)
    assert any(k.apply(x, exact=True)) and not k.psd


@pytest.mark.parametrize("name", list("TGFC"))
@pytest.mark.parametrize("width", [None, 0, 2])
def test_empty_cells_and_blocks(name, width):
    r = RexGraph.from_graph([], [])
    shape = (0,) if width is None else (0, width)
    x = np.empty(shape, int)
    y = channel_operator(r, name).apply(x, exact=True)
    assert y.shape == shape and y.dtype == object


@pytest.mark.parametrize("change", ["population", "selection", "vertex_metric"])
def test_reused_exact_handle_refuses_changed_source_contract(change):
    r = make([(0, 1), (1, 2)])
    op = channel_operator(r, "F")
    op.apply(np.ones(2, int), exact=True)
    if change == "population":
        r.add_edges([2], [3])
    elif change == "selection":
        r._c_channel = "count"
    else:
        r.w_V = np.array([1, 2, 1])
    with pytest.raises(ValueError):
        op.apply(np.ones(2, int), exact=True)


@pytest.mark.parametrize("values", [[1.0, 2.0], [True, False], [1j, 2j], ["1", "2"]])
def test_exact_inputs_are_not_reconstructed(values):
    with pytest.raises(TypeError, match="integers or Fractions"):
        channel_operator(make([(0, 1), (1, 2)]), "T").apply(np.array(values), exact=True)
