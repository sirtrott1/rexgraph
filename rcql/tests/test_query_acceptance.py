"""Static query refusals and exact scalar reductions."""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from rcql import Executor, Exactness, call, param, parse, query
from rcql.ast import Literal, MatchBinding, Query
from rexgraph.graph import RexGraph


@pytest.fixture
def executor():
    return Executor(sources={"r": RexGraph.from_graph(sources=[0, 1], targets=[1, 2])})


@pytest.mark.parametrize("expression,expected,exactness", [
    ("COUNT(CELLS(1))", 2, Exactness.INTEGER),
    ("COUNT([])", 0, Exactness.INTEGER),
    ("COUNT([NONE, true, 4])", 3, Exactness.INTEGER),
    ("SUM([])", 0, Exactness.INTEGER),
    ("SUM([1152921504606846976, 1])", 2**60 + 1, Exactness.INTEGER),
    ("SUM([1/3, 2/7])", Fraction(13, 21), Exactness.RATIONAL),
    ("SUM([1e16, 1.0, -1e16])", 1.0, Exactness.APPROXIMATE),
    ("MEAN([1, 2])", Fraction(3, 2), Exactness.RATIONAL),
    ("MEAN([1/3, 2/7])", Fraction(13, 42), Exactness.RATIONAL),
    ("MEAN([])", None, Exactness.STRUCTURAL),
    ("MEAN([1.0, 2.0])", 1.5, Exactness.APPROXIMATE),
    ("SUM([ARITY(CELL(1, 0)), ARITY(CELL(1, 1))])", 4, Exactness.INTEGER),
])
def test_scalar_reduction_contract(executor, expression, expected, exactness):
    request = parse("FROM $r RETURN " + expression)
    result = executor.execute(request)
    assert result.values == (expected,)
    assert result.exactness == (exactness,)
    explained = executor.execute(replace(request, explain=True)).values[0]
    assert explained["returns"][0]["result"]["exactness"] == exactness.value
    assert not executor.execute(replace(request, explain=True)).execution


def test_numpy_integer_reduction_does_not_overflow(executor):
    for name, expected in (("SUM", 2**64), ("MEAN", Fraction(2**63))):
        result = executor.execute(query(param("r"), call(name, [np.uint64(2**63), np.uint64(2**63)])))
        assert result.values == (expected,)


@pytest.mark.parametrize("expression", [
    'SUM([true])', 'MEAN([NONE])', 'SUM([[1]])', 'MEAN(CELLS(1))',
    'COUNT("text")', 'SUM([INDICATOR(CELL(1, 0))])',
    'INDICATOR(CELL(1, 0)) > 0', 'DESCRIBE() = 0', 'true < 2',
    'NONE < 1', '"text" > 1', '1 < 2 < 3',
])
def test_invalid_expression_is_refused_before_adapters(executor, monkeypatch, expression):
    monkeypatch.setattr("rcql.executor.get_operator", lambda *a: pytest.fail("adapter ran"))
    for explain in (False, True):
        with pytest.raises((TypeError, ValueError)):
            executor.execute(replace(parse("FROM $r RETURN " + expression), explain=explain))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_readings_are_refused(executor, value):
    executor.params["bad"] = value
    for text in ("SUM([$bad])", "$bad = $bad"):
        with pytest.raises(ValueError, match="finite"):
            executor.execute(parse("FROM $r RETURN " + text))


@pytest.mark.parametrize("clause", [
    {"where": Literal(True)}, {"order": ((Literal(1), False),)},
    {"limit": 0}, {"offset": 1}, {"limit": True}, {"offset": 0.5},
    {"offset": None}, {"order": ((Literal(1), "DESC"),)},
])
def test_typed_query_cannot_silently_drop_row_clauses(clause):
    with pytest.raises((ValueError, TypeError)):
        Query(param("r"), (Literal(1),), **clause)


def test_match_requires_an_expression():
    with pytest.raises(TypeError):
        MatchBinding("e", [1, 2])


def test_sort_refusal_precedes_store_reader(monkeypatch, executor):
    monkeypatch.setattr("rcql.executor.get_operator", lambda *a: pytest.fail("adapter ran"))
    request = parse("FROM $r MATCH e IN CELLS(1) RETURN e ORDER BY e")
    for explain in (False, True):
        with pytest.raises(TypeError, match="scalar"):
            executor.execute(replace(request, explain=explain))


def test_nullable_catalog_order_is_explicit(executor):
    from rcql.comparison import order_key
    assert sorted([2, None, 1], key=order_key) == [None, 1, 2]
    assert sorted([2, None, 1], key=order_key, reverse=True) == [2, 1, None]
