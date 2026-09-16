"""Native DAG execution, structural certificates, and honest method provenance."""
from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from rexgraph import Chain, Cochain
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.green import GreenOperator, vertex_green
from rexgraph.linear_operator import RexOperator, hodge_operator

from rcql import (
    BoundSource,
    Call,
    Executor,
    SourcePolicy,
    bind,
    call,
    param,
    parse,
    plan_query,
    query,
    source,
)
from rcql.operators import _REGISTRY


@pytest.fixture
def rex():
    return RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])


def _execute(rex, *expressions, explain=False, **params):
    return Executor(sources={"r": rex}, params=params).execute(
        replace(query(source("r"), *expressions), explain=explain),
    )


def test_dag_is_stable_serializable_and_is_used_by_execution(rex, monkeypatch):
    expression = call("HODGE", call("INDICATOR", call("CELL", 1, 0)))
    original_eval = Executor._eval

    def no_tree(self, expr, *args, **kwargs):
        if isinstance(expr, Call):
            pytest.fail("read expression was executed from the syntax tree")
        return original_eval(self, expr, *args, **kwargs)
    monkeypatch.setattr(Executor, "_eval", no_tree)
    result = _execute(rex, expression, expression)
    plan = result.native_plan
    assert plan["schema"] == "rcql.native-plan"
    assert plan["version"] == 1
    assert plan["outputs"][0] == plan["outputs"][1]
    assert result.values[0] is result.values[1]
    assert sum(n.get("operator") == "HODGE" for n in plan["nodes"]) == 1
    assert sum(n["operator"] == "HODGE" for n in result.execution) == 1
    seen = set()
    for node in plan["nodes"]:
        assert set(node["inputs"]) <= seen
        seen.add(node["id"])
    assert _execute(rex, expression, expression).native_plan == plan
    assert json.loads(json.dumps(plan, allow_nan=False)) == plan
    assert result.provenance[0]["node"] == plan["outputs"][0]
    assert result.provenance[0]["execution"]["method_status"] == "unreported"


def test_nonreusable_child_prevents_parent_common_subexpression_elimination(rex, monkeypatch):
    from rcql.signatures import _CATALOGUE
    monkeypatch.setitem(_CATALOGUE, "CELL", replace(_CATALOGUE["CELL"], memoizable=False))
    fn = _REGISTRY["CELL"].fn
    calls = []
    def observed(*args):
        calls.append(args)
        return fn(*args)
    monkeypatch.setitem(_REGISTRY, "CELL", replace(_REGISTRY["CELL"], fn=observed))
    expression = call("INDICATOR", call("CELL", 1, 0))
    result = _execute(rex, expression, expression)
    assert len(calls) == 2
    assert result.native_plan["outputs"][0] != result.native_plan["outputs"][1]
    assert sum(n["operator"] == "INDICATOR" for n in result.execution) == 2


def test_operator_descriptor_has_two_spaces_and_an_empty_upper_sector(rex):
    explained = _execute(rex, call("BOUNDARY", 2), call("COBOUNDARY", 2),
                         call("HODGE_OPERATOR", 1, 0.25), call("GREEN"), explain=True)
    returns = explained.values[0]["returns"]
    boundary, upper, hodge, green = [row["result"]["operator"] for row in returns]
    assert (boundary["domain"]["grade"], boundary["codomain"]["grade"]) == (2, 1)
    assert boundary["shape"] == (3, 1)
    assert upper["shape"] == (0, 1)
    assert upper["codomain"]["grade"] == 3
    assert any(p["name"] == "empty_codomain" and p["status"] == "verified" for p in returns[1]["predicates"])
    assert hodge["metric"] == "euclidean-cell"
    assert hodge["psd"] is True
    assert hodge["parameters"] == (("alpha", 0.25),)
    assert green["kernel_policy"] == "moore-penrose"
    assert json.dumps(explained.native_plan)


def test_standalone_inference_uses_the_same_descriptor_and_predicates(rex):
    from rcql import infer
    binding = bind("r", rex, SourcePolicy.allow("read"))
    action = infer(binding, "HODGE_OPERATOR", (1,)).result
    seed = infer(binding, "ZERO", (1,)).result
    applied = infer(binding, "APPLY", (action, seed))
    assert applied.result.shape.dims == (3,)
    assert applied.result.basis == action.operator.codomain
    assert any(p.name == "operator_spaces" and p.status == "verified" for p in applied.predicates)


def test_apply_uses_spaces_not_display_names_and_retains_block_shape(rex):
    action = replace(hodge_operator(rex, 1), name="not a recognized display name")
    field = Cochain(1, np.ones((3, 2)), source=rex)
    result = _execute(rex, call("APPLY", param("action"), param("field")), action=action, field=field)
    assert result.values[0].values.shape == (3, 2)
    assert result.provenance[0]["result_type"]["shape"] == [3, 2]
    np.testing.assert_allclose(result.values[0].values, action.apply(field.values))


def test_apply_refuses_operator_with_invented_codomain_population(rex):
    action = RexOperator("wrong", (1, 3), 1, 1, lambda values: values[:1], source=rex)
    with pytest.raises(ValueError, match="operator space"):
        _execute(rex, call("APPLY", param("op"), call("ZERO", 1)), op=action)


@pytest.mark.parametrize("text", [
    "HODGE_OPERATOR(1, -1)", "HODGE_OPERATOR(99)", "BETTI(-1)", "RANK(0)",
    "ZERO(99)", "CELL(1, 99)", "CELLS(1, (99))", "SHOW_OPERATORS(-1)", "SHOW_OPERATORS(1001)",
])
@pytest.mark.parametrize("explain", [False, True])
def test_bad_structural_requests_fail_before_any_adapter(rex, monkeypatch, text, explain):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *args: pytest.fail("adapter was resolved"))
    with pytest.raises((ValueError, TypeError, SyntaxError)):
        Executor(sources={"r": rex}).execute(parse(("EXPLAIN " if explain else "") + "FROM $r RETURN " + text))


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -float("inf")])
def test_hodge_refuses_nonfinite_coupling_during_explain(rex, alpha):
    with pytest.raises(ValueError, match="finite"):
        _execute(rex, call("HODGE_OPERATOR", 1, alpha), explain=True)


def test_temporal_predicates_do_not_reconstruct_snapshots(monkeypatch):
    timeline = TemporalRex([])
    timeline.append_snapshot(RexGraph.from_graph([0], [1]))
    timeline.append_snapshot(RexGraph.from_graph([0, 1], [1, 2]))
    monkeypatch.setattr(timeline, "reconstruct_at", lambda *args: pytest.fail("snapshot was reconstructed"))
    good = _execute(timeline, call("SIGNAL_SOURCE", call("TEMPORAL_DELTA", 1)), explain=True)
    assert good.values[0]["returns"][0]["result"]["temporal"]["version"] == 1
    for expr in (call("TEMPORAL_DELTA", 0), call("TEMPORAL_DELTA", 2),
                 call("RELATION_SIGNAL", call("TEMPORAL_DELTA", 1), "bogus")):
        with pytest.raises(ValueError):
            _execute(timeline, expr, explain=True)


def test_exact_normalized_g_predicates_retain_rational_math_without_channel_assembly(monkeypatch):
    rex = RexGraph.from_hypergraph([0, 4, 6], [0, 1, 2, 3, 1, 0], g_channel="normalized")
    from rexgraph.linear_operator import RexOperator
    monkeypatch.setattr(RexOperator, "as_scipy", lambda *args: pytest.fail("operator assembled"))
    explained = _execute(rex, call("CHARACTER", True), explain=True)
    predicates = explained.values[0]["returns"][0]["predicates"]
    assert any(p["name"] == "normalized_G_diagonal" and p["status"] == "verified" for p in predicates)
    assert _execute(rex, call("CHARACTER", True)).exactness[0].value == "rational"
    bad = RexGraph(sources=np.array([0, 1]), targets=np.array([1, 2]),
                   w_E=np.array([1.0, -1.0]), g_channel="normalized")
    with pytest.raises(ValueError, match="nonnegative"):
        _execute(bad, call("CHARACTER", True), explain=True)


def test_rewrite_provenance_carries_the_checked_exact_identity(rex):
    face = Chain(2, np.array([Fraction(2, 3)], dtype=object), source=rex)
    expr = call("BOUNDARY", 1, call("BOUNDARY", 2, param("face")))
    result = _execute(rex, expr, face=face)
    assert result.rewrites
    trace = result.native_plan["rewrites"][0]
    assert any(p["name"] == "chain_condition" and p["status"] == "verified" for p in trace["predicates"])
    assert result.provenance[0]["logical_operator"] == "BOUNDARY"
    assert result.execution[-1]["operator"] == "ZERO"
    assert result.provenance[0]["execution"]["methods"][0]["method"] == "integer-zero"


def test_unverified_chain_is_not_promoted_to_a_proof(rex, monkeypatch):
    import rexgraph.graded_boundary
    monkeypatch.setattr(rexgraph.graded_boundary, "_exact_chain_residual", lambda *args: None)
    face = Chain(2, np.array([1]), source=rex)
    expr = call("BOUNDARY", 1, call("BOUNDARY", 2, param("face")))
    result = _execute(rex, expr, face=face)
    assert not result.rewrites
    node = result.native_plan["nodes"][-1]
    assert any(p["name"] == "chain_condition" and p["status"] == "unknown" for p in node["predicates"])
    assert result.execution[-1]["methods"][0]["method"] == "exact-incidence-action"


def test_native_binding_does_not_evaluate_betti_to_classify_source(rex, monkeypatch):
    monkeypatch.setattr(RexGraph, "betti", property(lambda self: pytest.fail("betti was evaluated")))
    binding = bind("r", rex, SourcePolicy.allow("read"))
    assert binding.ref.state_digest
    plan = plan_query(binding, query(source("r"), call("HODGE_OPERATOR", 1)))
    assert plan.dag().explain()["source"]["state"]["state_digest"] == binding.ref.state_digest


def test_input_values_are_not_serialized_as_provenance(rex):
    field = Cochain(1, np.array([104729.0, 104729.0, 104729.0]), source=rex)
    result = _execute(rex, call("QUADRANCE", param("field")), field=field)
    encoded = json.dumps({"plan": result.native_plan, "provenance": result.provenance}, allow_nan=False)
    assert "104729" not in encoded
    assert "RexGraph(" not in encoded
    assert result.provenance[0]["source_state"]["state_digest"]


def test_rewrite_trace_does_not_dump_literal_chain_coefficients(rex):
    face = Chain(2, np.array([104729], dtype=np.int64), source=rex)
    result = _execute(rex, call("BOUNDARY", 1, call("BOUNDARY", 2, face)))
    assert result.rewrites
    assert "104729" not in json.dumps(result.native_plan)


def test_fresh_binding_records_changed_source_state():
    rex = RexGraph.from_graph([0], [1])
    first = _execute(rex, call("ZERO", 1)).provenance[0]["source_state"]["state_digest"]
    rex.add_edges([1], [2])
    second = _execute(rex, call("ZERO", 1)).provenance[0]["source_state"]["state_digest"]
    assert first != second


def test_method_observers_do_not_mix_concurrent_calls():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from rcql.execution_trace import capture_methods, record_method
    barrier = Barrier(2)
    def collect(label):
        with capture_methods() as observations:
            barrier.wait(timeout=5)
            record_method(label)
        return observations
    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(collect, "left")
        right = pool.submit(collect, "right")
        assert left.result() == [{"method": "left"}]
        assert right.result() == [{"method": "right"}]


def test_green_reports_cg_and_branching_lsqr_from_actual_invocations(rex):
    pairwise = _execute(rex, call("APPLY", call("GREEN"), call("INDICATOR", call("CELL", 0, 0))))
    event = pairwise.execution[-1]["methods"][0]
    assert event["kernel"] == "deflated-block-cg"
    assert event["fallback"] is False
    branching = RexGraph.from_hypergraph([0, 3], [0, 1, 2])
    result = _execute(branching, call("APPLY", call("GREEN"), call("ZERO", 0)))
    assert result.execution[-1]["methods"][0]["kernel"] == "native-factor-lsqr"
    assert result.native_plan["nodes"][-1]["physical"]["status"] == "deferred"


def test_green_fallback_and_custom_solver_are_reported_without_invented_certificates(rex, monkeypatch):
    import rexgraph.sparse_character
    monkeypatch.setattr(rexgraph.sparse_character, "_block_cg", lambda op, values, *args, **kw: np.zeros_like(values))
    seed = np.array([1.0, -1.0, 0.0])
    action = vertex_green(rex)
    result, info = action.solve_with_info(seed)
    assert info["kernel"] == "native-factor-lsqr" and info["fallback"]
    np.testing.assert_allclose(action.operator.apply(result), seed, atol=1e-9)
    custom = GreenOperator(action.operator, lambda values: values)
    returned, info = custom.solve_with_info(seed)
    assert returned is seed
    assert info == {"kernel": None, "status": "unreported"}


def test_resolvent_and_empty_green_report_their_actual_method(rex):
    action = GreenOperator.resolvent(hodge_operator(rex, 1))
    values, info = action.solve_with_info(np.ones(3))
    assert info["kernel"] == "native-block-cg"
    np.testing.assert_allclose(action.solve(np.ones(3)), values)
    empty = RexGraph.from_graph([], [])
    _, info = vertex_green(empty).solve_with_info(np.empty(0))
    assert info["kernel"] == "empty-zero"


def test_deny_refusal_does_not_resolve_an_adapter(rex, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *args: pytest.fail("adapter was resolved"))
    with pytest.raises(PermissionError, match="read"):
        Executor(sources={"r": BoundSource(rex, SourcePolicy.allow())}).execute(
            query(source("r"), call("HODGE_OPERATOR", 1)),
        )


def test_empty_upper_exact_action_matches_its_descriptor(rex):
    field = Cochain(2, np.array([Fraction(2, 3)], dtype=object), source=rex)
    result = _execute(rex, call("COBOUNDARY", 2, param("field")), field=field)
    assert result.values[0].values.shape == (0,)
    assert result.exactness[0].value == "rational"
    assert result.provenance[0]["result_type"]["exactness"] == "rational"


@pytest.mark.parametrize("values,expected", [
    ([1, 2, 3], "integer"), ([1, Fraction(2, 3), 3], "rational"),
])
def test_literal_object_coefficient_contracts_agree_before_and_after_execution(rex, values, expected):
    field = Cochain(1, np.array(values, dtype=object), source=rex)
    result = _execute(rex, param("field"), field=field)
    assert result.exactness[0].value == expected
    assert result.provenance[0]["result_type"]["exactness"] == expected
