"""Sigma queries name their weighting family and expose the numeric contract."""
from dataclasses import replace

import numpy as np
import pytest
from rexgraph.cochain import Cochain
from rexgraph.graph import RexGraph

from rcql import Executor, call, parse, query, source


@pytest.fixture
def rex():
    return RexGraph.from_graph(np.array([0, 1, 2, 0, 1]), np.array([1, 2, 3, 2, 3]))


FAMILY = ([1, 2, 4, 3], [-2, -1, 0, 1], list("TGFC"), "share")


def run(rex, name, *args, explain=False):
    return Executor(sources={"r": rex}).execute(replace(query(source("r"), call(name, *args)), explain=explain)).values[0]


@pytest.mark.parametrize("name,args", [
    ("SIGMA_OPERATOR", (0.6, *FAMILY)), ("CRITICAL_COMMUTATOR", (0.6, *FAMILY)),
    ("CRITICAL_RATE", FAMILY), ("CRITICAL_RATE", (*FAMILY, "slope")),
])
def test_literal_nested_and_explain_actions_agree(rex, name, args):
    value = Cochain(1, np.arange(rex.nE), source=rex)
    literal = run(rex, name, *args)
    nested = run(rex, "APPLY", call(name, *args), value)
    direct = run(rex, "APPLY", literal, value)
    np.testing.assert_allclose(nested.values, literal.apply(value.values))
    np.testing.assert_allclose(direct.values, nested.values)
    plan = run(rex, name, *args, explain=True)["returns"][0]["result"]
    assert plan["operator"]["coefficient_domain"] == "real"
    assert not plan["operator"]["exact_action"]
    parameters = dict(plan["operator"]["parameters"])
    assert parameters["family"] == "explicit-exponential-vertex"
    assert parameters["g_channel"] == "raw" and parameters["c_channel"] == "share"
    assert parameters["derivative"] == "analytic"
    for explain in (False, True):
        with pytest.raises(TypeError, match="exact"):
            run(rex, "APPLY", call(name, *args), value, True, explain=explain)


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("args", [
    (0.5, [1]*3, [0]*3, ["T"], "share"),
    (0.5, [1, 1, 0, 1], [0]*4, ["T"], "share"),
    (0.5, [1]*4, [0]*4, ["T", "T"], "share"),
    (0.5, [1]*4, [0]*4, [], "share"),
    (0.5, [1]*4, [0]*4, ["T"], "inferred"),
    (0.5, [1]*4, [0]*4, ["T"], "share", 2),
    (True, [1]*4, [0]*4, ["T"], "share"),
    (float("nan"), [1]*4, [0]*4, ["T"], "share"),
])
def test_declarations_are_validated_before_execution(rex, args, explain):
    with pytest.raises((ValueError, TypeError)):
        run(rex, "SIGMA_OPERATOR", *args, explain=explain)


def test_explain_never_evaluates_sigma_actions(rex, monkeypatch):
    from rexgraph.native_sparse import NativeSparse
    def fail(*args, **kw):
        pytest.fail("EXPLAIN applied a numerical action")
    monkeypatch.setattr(NativeSparse, "apply", fail)
    monkeypatch.setattr(NativeSparse, "transpose_apply", fail)
    run(rex, "SIGMA_OPERATOR", 0.5, *FAMILY, explain=True)
    run(rex, "CRITICAL_RATE", *FAMILY, explain=True)


def test_text_query_requires_and_reports_the_family(rex):
    result = Executor(sources={"r": rex}).execute(
        parse('FROM REX("r") RETURN SIGMA_OPERATOR(0.5, [1,2,4,3], [-2,-1,0,1], ["T","G","F","C"], "share")'))
    assert result.values[0].shape == (5, 5)
    with pytest.raises(TypeError):
        run(rex, "SIGMA_OPERATOR", 0.5)


def test_critical_queries_on_a_reopened_rcdb_store_match_the_source(rex, tmp_path):
    from rcdb import open_store
    from rexgraph.sigma_operator import sigma_operator
    uri = f"rex://{tmp_path / 'critical'}"
    store = open_store(uri)
    try:
        store.put("family", rex, analytics=False)
    finally:
        store.close()
    store = open_store(uri)
    try:
        expression = parse('FROM RCDB_GET($db,"family") LET x=INDICATOR(CELL(1,0)) '
            'RETURN APPLY(SIGMA_OPERATOR(0.6,[1,2,4,3],[-2,-1,0,1],["T","G","F","C"],"share"),x), '
            'APPLY(CRITICAL_COMMUTATOR(0.5,[1,2,4,3],[-2,-1,0,1],["T","G","F","C"],"share"),x), '
            'APPLY(ADJUGATE(CHANNEL("T")),x,true)')
        result = Executor(sources={"db": store}).execute(expression)
        np.testing.assert_allclose(result.values[0].values,
            sigma_operator(rex, 0.6, *FAMILY).apply(np.array([1, 0, 0, 0, 0])))
        assert not np.any(result.values[1].values)
        from fractions import Fraction
        assert all(isinstance(v, Fraction) for v in result.values[2].values)
        assert store.read_record("family").record.version == 1
    finally:
        store.close()
