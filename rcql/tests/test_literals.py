"""Text literals preserve their arithmetic, content and plan provenance."""
from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest
from rexgraph.graph import RexGraph

from rcql import Exactness, Executor, Literal, call, parse, query, source


@pytest.mark.parametrize("text,value", [
    ("17/20", Fraction(17, 20)), ("-17 / 20", Fraction(-17, 20)),
    ("17/-20", Fraction(-17, 20)), ("-17/-20", Fraction(17, 20)),
    ("+2/4", Fraction(1, 2)), ("0/9", Fraction(0)),
    ("12/3", Fraction(4)), ("9007199254740993/9007199254740992",
                           Fraction(9007199254740993, 9007199254740992)),
    ("-17", -17), ("+23", 23), ("-0.25", -0.25), ("1e-10", 1e-10),
    ("-2.5E+3", -2500.0), ("NONE", None), ("none", None),
])
def test_literal_values_and_builder_parity(text, value):
    parsed = parse(f" \n FROM $r RETURN {text} \t\n")
    assert parsed == query(source("r"), value)
    assert type(parsed.returns[0].value) is type(value)


@pytest.mark.parametrize("value", [
    "frustration Δ ↔ géométrie", "骨格 🐝", 'a "quote" and \\ path',
    "line\nbreak\tand tab", "", "nul\0control", "FROM $r RETURN BETTI(1)",
])
@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_unicode_and_json_escapes_round_trip(value, ensure_ascii):
    expr = parse(f"FROM $r RETURN {json.dumps(value, ensure_ascii=ensure_ascii)}").returns[0]
    assert expr == Literal(value)


@pytest.mark.parametrize("value", [
    "1/0", "1/-0", "1.0/2", "1/2.0", "1e2/3", "1/2/3", "1/", "1/$n",
    "1e9999", "-1e9999", "1+2", "1-2", "1e", "--1", "1.",
    r'"bad\q"', '"unterminated', '"unescaped\nnewline"',
])
def test_invalid_literals_are_syntax_errors(value):
    with pytest.raises(SyntaxError):
        parse(f"FROM $r RETURN {value}")


def test_exact_rational_result_and_json_plan_never_round():
    rex = RexGraph.from_graph([0], [1])
    text = "FROM $r RETURN 9007199254740993/9007199254740992, 4/2, 0.5, NONE"
    result = Executor(sources={"r": rex}).execute(parse(text))
    assert result.values == (Fraction(9007199254740993, 9007199254740992), Fraction(2), 0.5, None)
    assert result.exactness == (Exactness.RATIONAL, Exactness.RATIONAL,
                                Exactness.APPROXIMATE, Exactness.STRUCTURAL)
    assert result.native_plan["nodes"][0]["value"] == {
        "rational": {"numerator": 9007199254740993, "denominator": 9007199254740992}}
    assert result.plan[:2] == ("9007199254740993/9007199254740992", "2/1")
    assert json.loads(json.dumps(result.native_plan, allow_nan=False)) == result.native_plan
    explanation = Executor(sources={"r": rex}).execute(parse("EXPLAIN " + text))
    assert explanation.values[0]["returns"][0]["literal"] == result.native_plan["nodes"][0]["value"]


def test_rational_numeric_arguments_keep_approximate_operator_contract():
    rex = RexGraph.from_graph([0], [1])
    executor = Executor(sources={"r": rex})
    text = "FROM $r RETURN HODGE_OPERATOR(1, 17/20), RESOLVENT(CHANNEL(\"F\"), 1/2, 1/10000000000), METRIC(1, NONE)"
    result = executor.execute(parse(text))
    expected = executor.execute(query(source("r"), call("HODGE_OPERATOR", 1, Fraction(17, 20))))
    np.testing.assert_array_equal(result.values[0].apply(np.ones(1)), expected.values[0].apply(np.ones(1)))
    # Constructing a handle is structural; its declared action is numerical.
    hodge = next(n for n in result.native_plan["nodes"] if n.get("operator") == "HODGE_OPERATOR")
    assert hodge["result"]["exactness"] == "structural"
    assert hodge["result"]["operator"]["arithmetic"] == "approximate"
    json.dumps(result.native_plan, allow_nan=False)
    json.dumps(executor.execute(parse("EXPLAIN " + text)).values[0], allow_nan=False)


def test_integral_rational_does_not_silently_become_a_grade():
    rex = RexGraph.from_graph([0], [1])
    with pytest.raises(TypeError):
        Executor(sources={"r": rex}).execute(parse("FROM $r RETURN BETTI(2/2)"))
