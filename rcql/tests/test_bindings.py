"""LET is a typed single evaluation DAG binding, not textual substitution."""
from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from rexgraph import Chain, Cochain
from rexgraph.chain_map import CoordinateComplex, GradedMap
from rexgraph.graph import RexGraph

from rcql import (
    BoundSource,
    Executor,
    LetBinding,
    Reference,
    SourcePolicy,
    bind,
    call,
    let,
    parse,
    plan_query,
    query,
    ref,
    source,
)
from rcql.operators import _REGISTRY
from rcql.signatures import _CATALOGUE


@pytest.fixture
def rex():
    return RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])


def run(rex, text, **params):
    return Executor(sources={"r": rex}, params=params).execute(parse(text))


def test_parser_builder_and_sequential_binding_parity(rex):
    text = ('FROM $r LET x = INDICATOR(CELL(1, 0)) LET f = CHANNEL("F") '
            'LET y = APPLY(f, x) RETURN y, x, QUADRANCE(x, TRUE)')
    built = query(source("r"), ref("y"), ref("x"), call("QUADRANCE", ref("x"), True),
                  bindings=(let("x", call("INDICATOR", call("CELL", 1, 0))),
                            let("f", call("CHANNEL", "F")),
                            let("y", call("APPLY", ref("f"), ref("x")))))
    assert parse(text) == built
    assert isinstance(built.bindings[0], LetBinding)
    assert isinstance(built.returns[0], Reference)
    result = Executor(sources={"r": rex}).execute(built)
    np.testing.assert_array_equal(result.values[1].values, [1, 0, 0])
    assert result.values[2] == Fraction(1)
    assert [b["name"] for b in result.native_plan["bindings"]] == ["x", "f", "y"]
    assert result.provenance[0]["reference"] == "y"
    assert result.provenance[0]["execution"]["operator"] == "APPLY"
    assert result.provenance[0]["result_type"]["basis"]["grade"] == 1
    assert json.loads(json.dumps(result.native_plan, allow_nan=False)) == result.native_plan


def test_locals_and_external_parameters_are_distinct(rex):
    result = run(rex, "FROM $r LET x = 17/20 LET y = x RETURN x, y, $x", x=13)
    assert result.values == (Fraction(17, 20), Fraction(17, 20), 13)
    assert result.native_plan["outputs"][0] == result.native_plan["outputs"][1]
    assert result.native_plan["outputs"][0] != result.native_plan["outputs"][2]


def test_operator_calls_are_not_shadowed_by_bare_local_names(rex):
    result = run(rex, "FROM $r LET BETTI = 17 RETURN BETTI, BETTI(0)")
    assert result.values == (17, 1)


@pytest.mark.parametrize("name", ["", "a-b", "$a", "1a", "骨格", "TRUE", "none", "LET", "RETURN"])
def test_invalid_local_names_have_the_same_builder_refusal(name):
    with pytest.raises(ValueError):
        ref(name)
    with pytest.raises(ValueError):
        let(name, 1)


@pytest.mark.parametrize("text", [
    "FROM $r LET x = 1 LET x = 2 RETURN x",
    "FROM $r LET $x = 1 RETURN $x", "FROM $r LET x 1 RETURN x",
    "FROM $r LET TRUE = 1 RETURN TRUE", "FROM $r LET x = RETURN x",
])
def test_invalid_binding_syntax(text):
    with pytest.raises(SyntaxError):
        parse(text)


@pytest.mark.parametrize("bindings", [
    (let("x", ref("x")),), (let("x", ref("y")), let("y", 1)),
    (let("x", 1), let("x", 2)),
])
def test_duplicate_self_and_forward_bindings_refused_before_execution(rex, bindings, monkeypatch):
    def forbidden(*args):
        pytest.fail("an adapter executed before the binding error")
    monkeypatch.setattr("rcql.executor.get_operator", forbidden)
    built = query(source("r"), call("BETTI", 0), bindings=bindings)
    with pytest.raises(ValueError):
        Executor(sources={"r": rex}).execute(built)


def test_self_reference_in_text_is_not_an_accidental_nullary_call(rex):
    with pytest.raises(ValueError, match="unbound local"):
        run(rex, "FROM $r LET x = x RETURN x")


def test_unknown_and_case_mismatched_references_are_not_external_parameters(rex):
    for name in ("missing", "X"):
        built = query(source("r"), ref(name), bindings=(let("x", 1),))
        with pytest.raises(ValueError, match="unbound local"):
            Executor(sources={"r": rex}, params={name: 7}).execute(built)


def test_all_bindings_are_checked_before_any_adapter_including_unused_ones(rex, monkeypatch):
    def forbidden(*args):
        pytest.fail("an adapter ran before complete static checking")
    monkeypatch.setattr("rcql.executor.get_operator", forbidden)
    with pytest.raises(ValueError, match="grade"):
        run(rex, "FROM $r LET valid = BETTI(0) LET unused = BETTI(99) RETURN valid")
    with pytest.raises(PermissionError):
        run(BoundSource(rex, SourcePolicy.allow("identity")),
            "FROM $r LET unused = BETTI(0) RETURN 1")


def test_explain_does_not_run_binding_adapters_and_preserves_effects(rex, monkeypatch):
    def forbidden(*args):
        pytest.fail("EXPLAIN executed an adapter")
    monkeypatch.setattr("rcql.executor.get_operator", forbidden)
    result = run(rex, 'EXPLAIN FROM $r LET x = INDICATOR(CELL(1, 0)) RETURN QUADRANCE(x, TRUE)')
    explanation = result.values[0]
    assert explanation["bindings"][0]["name"] == "x"
    assert explanation["returns"][0]["arguments"][0]["reference"] == "x"
    assert explanation["effects"] == ["read"]
    assert len(result.native_plan["bindings"]) == 1


def test_observable_binding_evaluates_once_in_order_even_if_unused(rex, monkeypatch):
    fn = _REGISTRY["CELL"].fn
    seen = []
    def observed(source, grade, index):
        seen.append(index)
        return fn(source, grade, index)
    monkeypatch.setitem(_REGISTRY, "CELL", replace(_REGISTRY["CELL"], fn=observed))
    monkeypatch.setitem(_CATALOGUE, "CELL", replace(_CATALOGUE["CELL"], memoizable=False))
    result = run(rex, 'FROM $r LET a = CELL(1, 0) LET unused = CELL(1, 1) '
                 'LET b = a RETURN a, b, INDICATOR(a), INDICATOR(b), CELL(1, 0)')
    assert seen == [0, 1, 0]
    assert result.values[0] is result.values[1]
    assert result.values[2] is result.values[3]
    assert result.native_plan["outputs"][0] != result.native_plan["outputs"][-1]
    assert sum(n["operator"] == "INDICATOR" for n in result.execution) == 1
    run(rex, 'FROM $r LET a = CELL(1, 0) RETURN a')
    assert seen == [0, 1, 0, 0]  # never a cross query cache


def test_bound_known_metrics_receive_the_same_static_coefficient_validation(rex):
    bad = Cochain(1, np.array([1, -1, 1]), source=rex)
    with pytest.raises(ValueError, match="positive"):
        run(rex, 'EXPLAIN FROM $r LET w = $w LET again = w RETURN METRIC(1, again)', w=bad)


def test_bound_chain_map_keeps_its_static_verification(rex):
    c = CoordinateComplex.from_rex(rex)
    declaration = GradedMap(c, c, tuple(tuple((i, i, 1) for i in range(n)) for n in c.sizes))
    result = run(rex, 'EXPLAIN FROM $r LET p = $p LET q = p RETURN CHAIN_MAP(q)', p=declaration)
    predicates = result.values[0]["returns"][0]["predicates"]
    assert any(p["status"] == "verified" and "chain" in p["name"] for p in predicates)


def test_optimizer_does_not_drop_bindings_when_rewriting_an_independent_return(rex):
    face = Chain(2, np.array([1]), source=rex)
    result = run(rex, 'FROM $r LET f = 17/20 RETURN BOUNDARY(1, BOUNDARY(2, $c)), f', c=face)
    assert result.rewrites
    assert result.values[1] == Fraction(17, 20)
    assert result.native_plan["bindings"][0]["name"] == "f"


def test_optimizer_uses_exact_input_proofs_through_aliases_in_binding_definitions(rex):
    face = Chain(2, np.array([1]), source=rex)
    result = run(rex, 'FROM $r LET c = $c LET alias = c '
                 'LET z = BOUNDARY(1, BOUNDARY(2, alias)) RETURN z', c=face)
    assert len(result.rewrites) == 1
    np.testing.assert_array_equal(result.values[0].values, np.zeros(3, dtype=int))
    assert [b["name"] for b in result.native_plan["bindings"]] == ["c", "alias", "z"]
    assert not any(n.get("operator") == "BOUNDARY" for n in result.native_plan["nodes"])


def test_binding_reference_keeps_source_and_basis_checks(rex):
    other = RexGraph.from_graph([0, 1, 0], [1, 2, 2])
    foreign = Cochain(1, np.ones(3, dtype=int), source=other)
    with pytest.raises((ValueError, TypeError), match="source|bound"):
        run(rex, 'FROM $r LET x = $x RETURN QUADRANCE(x, TRUE)', x=foreign)


def test_many_reference_aliases_lower_linearly_without_recursive_expansion(rex):
    count = 1500
    bindings = (let("v0", 17),) + tuple(let(f"v{i}", ref(f"v{i-1}")) for i in range(1, count))
    built = query(source("r"), ref(f"v{count-1}"), bindings=bindings)
    planned = plan_query(bind("r", rex, SourcePolicy.allow("*")), built)
    assert planned.effects == frozenset()
    assert len(planned.dag().nodes) == 1
    assert len(planned.explain()["bindings"]) == count
    result = Executor(sources={"r": rex}).execute(built)
    assert result.values == (17,)


def test_query_rejects_untyped_binding_entries(rex):
    with pytest.raises(TypeError, match="LetBinding"):
        Executor(sources={"r": rex}).execute(query(source("r"), 1, bindings=(("x", 1),)))
