"""Typed metrics and explicit Green variants, with independent small oracles."""
import json
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import diagonal_metric
from rexgraph.graph import RexGraph
from rexgraph.green import GreenOperator
from rexgraph.linear_operator import RexOperator

from rcql import Executor, call, param, parse, query, source


def make():
    return RexGraph.from_graph([0, 1], [1, 2])


def co(rex, values, grade=1, **kw):
    return Cochain(grade, np.asarray(values), source=rex, **kw)


def run(rex, *expressions, explain=False, **params):
    return Executor(sources={"r": rex}, params=params).execute(
        query(source("r"), *expressions, explain=explain))


def test_explicit_metric_changes_contractions_without_changing_default():
    rex = make()
    a, b, weights = co(rex, [1, 1]), co(rex, [-1, 1]), co(rex, [1, 4])
    metric = call("METRIC", 1, param("w"))
    result = run(rex, call("MOMENT", param("a"), param("b"), metric, True),
                 call("QUADRANCE", param("a"), True, metric),
                 call("SPREAD", param("a"), param("b"), True, metric),
                 call("QUADRANCE", param("a"), True),
                 call("MOMENT", param("a"), param("b"), None, True), a=a, b=b, w=weights)
    assert result.values == (Q(3), Q(5), Q(16, 25), Q(2), Q(0))
    assert all(x.value == "rational" for x in result.exactness)
    descriptor = result.provenance[0]["result_type"]["metric"]
    assert descriptor["positive_definite"] is True
    assert descriptor["coefficient_domain"] == "rational"
    assert descriptor["shape"] == [2, 2]
    assert len(descriptor["coefficient_digest"]) == 64
    assert "weights" not in json.dumps(descriptor)


def test_textual_api_and_signed_moment():
    rex = make()
    ex = Executor(sources={"r": rex}, params={"a": co(rex, [1, 2]),
        "b": co(rex, [-3, -4]), "w": co(rex, [Q(1, 2), Q(2, 3)])})
    result = ex.execute(parse('FROM $r RETURN MOMENT($a, $b, METRIC(1, $w), true)'))
    assert result.values[0] == Q(-41, 6)


def test_complex_metric_moment_and_spread_are_hermitian():
    rex = make()
    a, b, m = co(rex, [1j, 1]), co(rex, [1, 1j]), diagonal_metric(rex, 1, co(rex, [2, 3]))
    result = run(rex, call("MOMENT", param("a"), param("b"), param("m")),
                 call("QUADRANCE", param("a"), False, param("m")),
                 call("SPREAD", param("a"), param("b"), False, param("m")), a=a, b=b, m=m)
    assert result.values == (1j, 5., 24/25)
    assert result.provenance[0]["result_type"]["domain"] == "complex"


def test_identity_metric_and_zero_spread_remain_distinct_from_green():
    rex = make()
    result = run(rex, call("MOMENT", call("ZERO", 1), call("ZERO", 1), call("METRIC", 1), True),
                 call("SPREAD", call("ZERO", 1), call("ZERO", 1), True, call("METRIC", 1)))
    assert result.values == (Q(0), None)


def test_block_moment_is_a_frobenius_contraction_not_a_gram_matrix():
    rex = make()
    a, b = co(rex, [[1, 2], [3, 4]]), co(rex, [[-3, 1], [-2, 0]])
    metric = diagonal_metric(rex, 1, co(rex, [Q(1, 3), Q(5, 2)]))
    result = run(rex, call("MOMENT", param("a"), param("b"), param("m"), True),
                 call("QUADRANCE", param("a"), True, param("m")), a=a, b=b, m=metric)
    assert result.values == (Q(-46, 3), Q(385, 6))
    with pytest.raises(ValueError, match="one-dimensional"):
        run(rex, call("SPREAD", param("a"), param("b"), False, param("m")), a=a, b=b, m=metric)


@pytest.mark.parametrize("weights", [[0, 1], [-1, 1], [float("nan"), 1], [1j, 1], [True, False], [[1], [2]]])
@pytest.mark.parametrize("explain", [False, True])
def test_bad_known_metric_weights_fail_before_adapters(weights, explain, monkeypatch):
    import rcql.executor
    rex = make()
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *args: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, call("METRIC", 1, param("w")), explain=explain, w=co(rex, weights))


@pytest.mark.parametrize("mismatch", ["basis", "grade", "source", "float-exact", "variance", "shape"])
def test_metric_alignment_and_exactness_refuse_before_adapters(mismatch, monkeypatch):
    import rcql.executor
    rex = make()
    a, b = co(rex, [1, 2]), co(rex, [3, 4])
    m = diagonal_metric(rex, 1)
    if mismatch == "basis":
        m = diagonal_metric(rex, 1, co(rex, [1, 2], cell_keys=("b", "a")))
    elif mismatch == "grade":
        m = diagonal_metric(rex, 0)
    elif mismatch == "source":
        m = diagonal_metric(make(), 1)
    elif mismatch == "float-exact":
        m = diagonal_metric(rex, 1, co(rex, [1., 2.]))
    elif mismatch == "variance":
        b = Chain(1, b.values, source=rex)
    elif mismatch == "shape":
        b = co(rex, [[3], [4]])
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *args: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, call("MOMENT", param("a"), param("b"), param("m"), True), a=a, b=b, m=m)


def test_computed_metric_positivity_is_deferred_not_claimed_as_proven():
    rex = make()
    expr = call("METRIC", 1, call("ZERO", 1))
    result = run(rex, expr, explain=True)
    encoded = json.dumps(result.values, default=str)
    assert "positive_metric" in encoded and "deferred" in encoded
    with pytest.raises(ValueError, match="strictly positive"):
        run(rex, expr)


@pytest.mark.parametrize("grade", [0, 1, 2])
@pytest.mark.parametrize("columns", [None, 0, 3])
def test_green_solve_and_apply_use_the_same_matrix_free_resolvent(grade, columns, monkeypatch):
    rex = RexGraph.from_simplicial(np.array([0, 1, 0], np.int32),
        np.array([1, 2, 2], np.int32), np.array([[0, 1, 2]], np.int32))
    action = call("HODGE_OPERATOR", grade)
    matrix = run(rex, action).values[0].as_scipy().toarray()
    n = len(matrix)
    shape = (n,) if columns is None else (n, columns)
    rhs = co(rex, np.arange(np.prod(shape), dtype=float).reshape(shape)-2, grade)
    expected = np.linalg.solve(np.eye(n)+.3*matrix, rhs.values)
    def forbidden(*args, **kw):
        pytest.fail("resolvent assembled a matrix or diagonalized")
    monkeypatch.setattr(RexOperator, "as_scipy", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    green = call("RESOLVENT", action, .3, 1e-10, 100)
    result = run(rex, call("GREEN_SOLVE", green, param("b")),
                 call("APPLY", green, param("b")), b=rhs)
    for value in result.values:
        np.testing.assert_allclose(value.values, expected, atol=1e-12)
    desc = next(n for n in result.native_plan["nodes"] if n.get("operator") == "RESOLVENT")["result"]["operator"]
    assert desc["kernel_policy"] == "inverse-I-plus-alpha-L"
    assert dict(desc["parameters"])["alpha"] == .3
    event = next(e for e in result.execution if e["operator"] == "GREEN_SOLVE")
    assert event["methods"][0]["method"] == "green-solve"
    assert all(r <= 1e-10 for r in event["methods"][0]["relative_residuals"])


def test_pseudoinverse_and_resolvent_treat_kernel_differently():
    rex = make()
    result = run(rex, call("GREEN_SOLVE", call("GREEN"), param("b")),
        call("GREEN_SOLVE", call("RESOLVENT", call("HODGE_OPERATOR", 0)), param("b")),
        b=co(rex, np.ones(3), 0))
    np.testing.assert_allclose(result.values[0].values, 0, atol=1e-12)
    np.testing.assert_allclose(result.values[1].values, 1, atol=1e-12)


@pytest.mark.parametrize("text", [
    "RESOLVENT(BOUNDARY(1))", "RESOLVENT(GREEN())", "RESOLVENT(HODGE_OPERATOR(1), -1)",
    "RESOLVENT(HODGE_OPERATOR(1), true)", "RESOLVENT(HODGE_OPERATOR(1), 1, 0)",
    "RESOLVENT(HODGE_OPERATOR(1), 1, 0.1, 0)", "GREEN_SOLVE(HODGE_OPERATOR(1), ZERO(1))",
    "GREEN_SOLVE(GREEN(), ZERO(1))", "METRIC(-1)", "METRIC(1, ZERO(0))",
])
def test_invalid_action_contracts_fail_before_adapters(text, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *args: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError, SyntaxError)):
        Executor(sources={"r": make()}).execute(parse("EXPLAIN FROM $r RETURN " + text))


def test_complex_apply_refuses_at_type_boundary(monkeypatch):
    import rcql.executor
    rex = make()
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *args: pytest.fail("adapter ran"))
    with pytest.raises(TypeError, match="real"):
        run(rex, call("GREEN_SOLVE", call("RESOLVENT", call("HODGE_OPERATOR", 1)), param("b")),
            b=co(rex, [1j, 1]))


def test_external_resolvent_preserves_constructor_parameters():
    rex = make()
    op = run(rex, call("HODGE_OPERATOR", 1)).values[0]
    green = GreenOperator.resolvent(op, .7, tol=1e-8, maxiter=42)
    result = run(rex, call("GREEN_SOLVE", param("g"), call("ZERO", 1)), g=green)
    encoded = json.dumps(result.native_plan)
    assert '"alpha", 0.7' in encoded
    assert '"maxiter", 42' in encoded


@pytest.mark.parametrize("name", list("TGFC"))
@pytest.mark.parametrize("g_channel", ["raw", "normalized"])
def test_branching_weighted_channel_resolvents_match_reference(name, g_channel, monkeypatch):
    rex = RexGraph(boundary_ptr=np.array([0, 3, 5], np.int32),
        boundary_idx=np.array([0, 1, 2, 1, 0], np.int32), w_E=np.array([2., 3.]), g_channel=g_channel)
    action = call("CHANNEL", name)
    matrix = run(rex, action).values[0].as_scipy().toarray()
    rhs = co(rex, [[1., -1.], [2., 3.]])
    expected = np.linalg.solve(np.eye(2) + .25*matrix, rhs.values)
    monkeypatch.setattr(RexOperator, "as_scipy", lambda *a: pytest.fail("assembled matrix"))
    result = run(rex, call("GREEN_SOLVE", call("RESOLVENT", action, .25), param("b")), b=rhs)
    np.testing.assert_allclose(result.values[0].values, expected, atol=1e-12)


def test_explicit_metric_respects_declared_noncanonical_basis():
    rex = make()
    a = co(rex, [1, 2], cell_keys=("b", "a"))
    weights = co(rex, [3, 4], cell_keys=("b", "a"))
    result = run(rex, call("QUADRANCE", param("a"), True, call("METRIC", 1, param("w"))), a=a, w=weights)
    assert result.values[0] == Q(19)
    assert result.provenance[0]["result_type"]["metric"]["basis"]["ordering"] == ["b", "a"]


@pytest.mark.parametrize("scale", [1e100, 1e-100])
def test_metric_spread_avoids_squaring_quadrance_range(scale):
    rex = make()
    a, b = co(rex, np.array([1., 1.])*scale), co(rex, np.array([-1., 1.])*scale)
    result = run(rex, call("SPREAD", param("a"), param("b"), False, call("METRIC", 1, param("w"))),
        a=a, b=b, w=co(rex, [1, 4]))
    assert result.values[0] == pytest.approx(16/25)


@pytest.mark.parametrize("expression", [call("METRIC", -1), call("RESOLVENT", call("HODGE_OPERATOR", 1), -1)])
def test_negative_builder_controls_reach_validation_without_parser(expression, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *args: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(make(), expression, explain=True)
