"""Typed exact adjugate application and full grade homotopy certification."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.chain_map import ChainHomotopy, CoordinateComplex, GradedMap
from rexgraph.cochain import Cochain
from rexgraph.graph import RexGraph

from rcql import Executor, call, param, parse, query, source


def run(rex, name, *args, explain=False):
    return Executor(sources={"r": rex}).execute(replace(query(source("r"), call(name, *args)), explain=explain)).values[0]


def interval():
    rex = RexGraph.from_cells([2, [[0, 1]]])
    c = CoordinateComplex.from_rex(rex)
    f = GradedMap(c, c, ((), ()))
    g = GradedMap(c, c, (((0, 0, -1), (1, 0, 1)), ((0, 0, -1),)))
    return rex, f, g, (((0, 0, 1),), ())


def test_adjugate_nested_and_literal_actions_preserve_exact_coefficients():
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2], [2, 0]]])
    expression = call("ADJUGATE", call("CHANNEL", "T"))
    literal = run(rex, "ADJUGATE", call("CHANNEL", "T"))
    x = Cochain(1, np.array([Q(1, 3), Q(1, 5), 2**100], object), source=rex)
    result = run(rex, "APPLY", expression, x, True)
    np.testing.assert_array_equal(result.values, literal.apply(x.values, exact=True))
    np.testing.assert_array_equal(run(rex, "APPLY", literal, x, True).values, result.values)
    plan = run(rex, "APPLY", expression, x, True, explain=True)["returns"][0]["result"]
    assert plan["exactness"] == "rational"
    assert dict(run(rex, "ADJUGATE", call("CHANNEL", "T"), explain=True)["returns"][0]["result"]["operator"]["parameters"])["coefficient_actions"] == 6


@pytest.mark.parametrize("explain", [False, True])
def test_adjugate_refuses_uncertified_and_rectangular_actions(explain):
    rex, _, _, _ = interval()
    for expression in (call("HODGE_OPERATOR", 1), call("BOUNDARY", 1)):
        with pytest.raises(TypeError):
            run(rex, "ADJUGATE", expression, explain=explain)


def test_adjugate_explain_does_not_evaluate_any_primal(monkeypatch):
    from rexgraph.linear_operator import RexOperator
    rex, _, _, _ = interval()
    monkeypatch.setattr(RexOperator, "apply", lambda *a, **k: pytest.fail("EXPLAIN applied primal"))
    run(rex, "ADJUGATE", call("CHANNEL", "T"), explain=True)


def test_homotopy_direct_nested_and_member_contracts():
    rex, f, g, h = interval()
    result = run(rex, "HOMOTOPY", f, g, h)
    assert isinstance(result, ChainHomotopy) and result.residuals == (0, 0)
    assert run(rex, "HOMOTOPY", call("CHAIN_MAP", f), call("CHAIN_MAP", g), h).residuals == (0, 0)
    parsed = parse("FROM $r LET h = HOMOTOPY($f, $g, $w) RETURN h.residuals, h.shapes, h.coefficient_digest")
    output = Executor(sources={"r": rex}, params={"f": f, "g": g, "w": h}).execute(parsed)
    assert output.values == ((Q(0), Q(0)), ((1, 2), (0, 1)), result.coefficient_digest)


@pytest.mark.parametrize("explain", [False, True])
def test_homotopy_wrong_sign_or_foreign_source_is_rejected_before_adapters(monkeypatch, explain):
    import rcql.executor
    rex, f, g, h = interval()
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("invalid proof reached adapter"))
    for left, right, witness in ((g, f, h), (f, interval()[2], h), (f, g, ((), ()))):
        with pytest.raises((TypeError, ValueError)):
            run(rex, "HOMOTOPY", left, right, witness, explain=explain)


def test_homotopy_explain_verifies_the_exact_equation_without_adapters(monkeypatch):
    import rcql.executor
    rex, f, g, h = interval()
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("EXPLAIN ran adapter"))
    info = run(rex, "HOMOTOPY", f, g, h, explain=True)["returns"][0]
    assert info["result"]["kind"] == "ChainHomotopy"
    assert any(p["name"] == "homotopy_certificate" and p["status"] == "verified" for p in info["predicates"])


def test_homotopy_abstract_descriptors_defer_the_proof():
    from rcql import SourcePolicy, bind
    from rcql.planning import _carrier_literal
    rex, f, g, h = interval()
    binding = bind("r", rex, SourcePolicy.allow("*"))
    types = [_carrier_literal(binding, x) for x in (f, g)]
    result = Executor(sources={"r": rex}, params=dict(zip(("f", "g"), types, strict=True))).execute(
        query(source("r"), call("HOMOTOPY", param("f"), param("g"), h), explain=True))
    info = result.values[0]["returns"][0]
    assert any(p["name"] == "homotopy_certificate" and p["status"] == "deferred" for p in info["predicates"])
