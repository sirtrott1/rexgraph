"""Whole RCQL phrases type-check before their numerical adapters are resolved."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rcql import (
    BasisRef,
    Domain,
    Exactness,
    Executor,
    RCType,
    SourcePolicy,
    TemporalRef,
    ValueKind,
    Variance,
    at,
    at_time,
    bind,
    call,
    param,
    parse,
    plan_query,
    query,
    source,
)


@pytest.fixture
def rex():
    from rexgraph.graph import RexGraph

    return RexGraph.from_graph(sources=[0, 1, 2], targets=[1, 2, 0])


def test_whole_phrase_composes_primary_cell_contracts_without_execution(rex):
    binding = bind("complex", rex, SourcePolicy.allow("*"))
    phrase = query(
        source("complex"),
        call("SHARE", call("COMPOSITE", call("CELL", 1, 0))),
    )

    plan = plan_query(binding, phrase)
    result = plan.returns[0].result
    assert (result.kind, result.grade, result.domain, result.exactness) == (
        ValueKind.CHAIN, 0, Domain.RATIONAL, Exactness.RATIONAL,
    )
    assert plan.returns[0].children[0].result.kind is ValueKind.COMPOSITE_BINARY

    explained = plan.explain()
    assert explained["source"] == "complex"
    assert explained["returns"][0]["operator"] == "SHARE"
    assert explained["returns"][0]["arguments"][0]["operator"] == "COMPOSITE"
    assert explained["returns"][0]["arguments"][0]["arguments"][0]["operator"] == "CELL"
    assert explained["returns"][0]["result"]["basis"] == {
        "source_id": "complex", "grade": 0, "ordering": "canonical",
    }


def test_temporal_phrase_preserves_the_delta_time_through_nested_calls():
    from rexgraph.graph import RexGraph, TemporalRex

    timeline = TemporalRex([])
    timeline.append_snapshot(RexGraph.from_graph(sources=[0], targets=[1]), at=1.0)
    timeline.append_snapshot(RexGraph.from_graph(sources=[0, 1], targets=[1, 2]), at=4.0)
    binding = bind("timeline", timeline, SourcePolicy.allow("*"))
    phrase = query(
        source("timeline"),
        call("SIGNAL_HODGE", call("TEMPORAL_DELTA", 1)),
    )

    plan = plan_query(binding, phrase)
    result = plan.returns[0].result
    assert (result.kind, result.grade, result.temporal.version, result.exactness) == (
        ValueKind.HODGE_SPLIT, 1, 1, Exactness.APPROXIMATE,
    )
    assert plan.explain()["returns"][0]["arguments"][0]["result"]["temporal"] == {
        "version": 1, "as_of": None, "valid_at": None,
    }

    generic = plan_query(binding, query(
        source("timeline"),
        call("HODGE", call("RELATION_SIGNAL", call("TEMPORAL_DELTA", 1))),
    ))
    assert generic.returns[0].result.kind is ValueKind.HODGE_SPLIT
    assert generic.returns[0].result.temporal.version == 1


def test_hodge_and_green_phrases_declare_numerical_actions_but_not_their_inputs(rex):
    binding = bind("complex", rex, SourcePolicy.allow("*"))
    phrase = query(
        source("complex"),
        call("HODGE", call("ZERO", 1)),
        call("APPLY", call("GREEN"), call("ZERO", 0)),
        call("QUADRANCE", call("SHARE", call("COMPOSITE", call("CELL", 1, 0))), True),
    )

    plan = plan_query(binding, phrase)
    hodge, green_field, quadrance = (item.result for item in plan.returns)
    assert (hodge.kind, hodge.grade, hodge.exactness) == (
        ValueKind.HODGE_SPLIT, 1, Exactness.APPROXIMATE,
    )
    assert (green_field.kind, green_field.grade, green_field.exactness) == (
        ValueKind.FIELD, 0, Exactness.APPROXIMATE,
    )
    assert (quadrance.kind, quadrance.domain, quadrance.exactness) == (
        ValueKind.EXACT_RATIONAL, Domain.RATIONAL, Exactness.RATIONAL,
    )


def test_phrase_refuses_a_bad_inner_call_before_any_outer_adapter_runs(rex):
    binding = bind("complex", rex, SourcePolicy.allow("*"))
    phrase = query(source("complex"), call("COMPOSITE", call("CELL", 0, 0)))

    with pytest.raises(TypeError, match="grade=1"):
        plan_query(binding, phrase)


def test_phrase_refuses_two_unaligned_temporal_c1_fields():
    from rexgraph.graph import RexGraph

    rex = RexGraph.from_graph(sources=[0, 1], targets=[1, 2])
    binding = bind("complex", rex, SourcePolicy.allow("*"))
    left = RCType(
        "C1", grade=1, kind=ValueKind.COCHAIN, variance=Variance.COCHAIN,
        domain=Domain.REAL, exactness=Exactness.APPROXIMATE, source=binding.ref,
        basis=BasisRef("complex", 1), temporal=TemporalRef(version=1),
    )
    right = left.with_(temporal=TemporalRef(version=2))
    for phrase in (
        query(source("complex"), call("SPREAD", param("left"), param("right"))),
        query(source("complex"), call("ACCUMULATE", param("left"), param("right"))),
    ):
        with pytest.raises(TypeError, match="matching grade, variance, basis, source, and temporal state"):
            plan_query(binding, phrase, parameters={"left": left, "right": right})


def test_explain_runs_the_whole_static_phrase_plan_without_running_an_adapter():
    class ExplodingRex:
        nV = 3
        betti = (1, 0)

        def relation_supports(self):  # pragma: no cover - reaching this is the failure
            raise AssertionError("EXPLAIN ran the composite adapter")

    phrase = query(source("exploding"), call("COMPOSITE", call("CELL", 1, 0)), explain=True)
    result = Executor(sources={"exploding": ExplodingRex()}).execute(phrase)

    assert result.exactness == (Exactness.STRUCTURAL,)
    assert result.values[0]["returns"][0]["operator"] == "COMPOSITE"
    assert result.values[0]["returns"][0]["arguments"][0]["result"]["kind"] == "Cell"


def test_normal_execution_preflights_the_same_invalid_whole_phrase():
    class ExplodingRex:
        nV = 3
        betti = (1, 0)

        def relation_supports(self):  # pragma: no cover - reaching this is the failure
            raise AssertionError("execution reached the composite adapter")

    phrase = query(source("exploding"), call("COMPOSITE", call("CELL", 0, 0)))

    with pytest.raises(TypeError, match="grade=1"):
        Executor(sources={"exploding": ExplodingRex()}).execute(phrase)


def test_planning_keeps_a_native_metric_cochain_as_a_source_bound_carrier(rex):
    from rexgraph.cochain import Cochain

    metric = Cochain(1, np.array([2, 3, 5]), source=rex)
    binding = bind("complex", rex, SourcePolicy.allow("*"))
    phrase = query(source("complex"), call("METRIC_CURVATURE", metric))

    planned = plan_query(binding, phrase)
    assert planned.returns[0].result.kind is ValueKind.METRIC_CURVATURE
    assert planned.returns[0].result.exactness is Exactness.RATIONAL
    literal = planned.explain()["returns"][0]["arguments"][0]["literal"]
    assert literal["type"]["kind"] == "Cochain"
    assert literal["type"]["source"] == "complex"


def test_functionals_and_actions_accept_direct_cell_boundary_carriers(rex):
    executor = Executor(sources={"complex": rex})

    quadrance = executor.execute(query(
        source("complex"),
        call("QUADRANCE", call("BOUNDARY", call("CELL", 1, 0)), True),
    ))
    hodge = executor.execute(query(
        source("complex"),
        call("HODGE", call("COBOUNDARY", call("CELL", 0, 0))),
    ))

    # The direct C1 boundary retains its exact rational coefficients rather than
    # becoming a projected adjacency row before the functional reads it.
    assert quadrance.values == (Fraction(2),)
    assert quadrance.exactness == (Exactness.RATIONAL,)
    assert set(hodge.values[0]) == {"gradient", "curl", "harmonic"}


def test_direct_kary_boundary_keeps_one_relation_column_for_exact_functionals():
    from rexgraph.graph import RexGraph

    complex_ = RexGraph.from_hypergraph(
        np.array([0, 3]), np.array([0, 1, 2]),
    )
    result = Executor(sources={"complex": complex_}).execute(query(
        source("complex"),
        call("ARITY", call("COMPOSITE", call("CELL", 1, 0))),
        call("QUADRANCE", call("BOUNDARY", call("CELL", 1, 0)), True),
    ))

    # One arity-three C1 relation has boundary (-1, 1/2, 1/2), so its
    # quadrance is 3/2. No star or clique columns were introduced to ask it.
    assert result.values == (3, Fraction(3, 2))
    assert result.exactness == (Exactness.INTEGER, Exactness.RATIONAL)

    curvature = Executor(sources={"complex": complex_}).execute(query(
        source("complex"), call("METRIC_CURVATURE", call("INDICATOR", call("CELL", 1, 0))),
    ))
    # A single uniform C1 metric has zero strain; the important contract is
    # that the result is exact rational topology rather than a traversal score.
    assert curvature.values[0].total == Fraction(0)
    assert curvature.exactness == (Exactness.RATIONAL,)


def test_chain_condition_rewrite_returns_an_exact_zero_chain():
    from rexgraph.cochain import Chain
    from rexgraph.graph import RexGraph

    triangle = RexGraph.from_simplicial(
        np.array([0, 1, 0]), np.array([1, 2, 2]), np.array([[0, 1, 2]]),
    )
    face = Chain(2, np.ones(triangle.nF, dtype=np.int64), source=triangle)
    result = Executor(sources={"triangle": triangle}).execute(query(
        source("triangle"), call("BOUNDARY", 1, call("BOUNDARY", 2, face)),
    ))

    assert result.values[0].grade == 0
    assert result.values[0].values.tolist() == [0, 0, 0]
    assert result.exactness == (Exactness.INTEGER,)
    assert [rewrite.reason for rewrite in result.rewrites] == [
        "consecutive boundaries compose to zero",
    ]


def test_at_binds_one_temporal_snapshot_for_direct_cell_and_boundary_queries():
    from rexgraph.graph import RexGraph, TemporalRex

    timeline = TemporalRex([])
    timeline.append_snapshot(RexGraph.from_graph(sources=[0], targets=[1]), at=1.0)
    timeline.append_snapshot(RexGraph.from_graph(sources=[0, 1], targets=[1, 2]), at=4.0)
    executor = Executor(sources={"timeline": timeline})

    direct = executor.execute(query(
        at(source("timeline"), 1), call("BOUNDARY", call("CELL", 1, 1)),
    ))
    seeded = executor.execute(query(
        at(source("timeline"), 1),
        call("APPLY", call("GREEN"), call("INDICATOR", call("CELL", 0, 0))),
    ))
    timed = executor.execute(query(
        at_time(source("timeline"), 4.0), call("BOUNDARY", call("CELL", 1, 1)),
    ))
    held = executor.execute(query(
        at_time(source("timeline"), 3.5), call("BOUNDARY", call("CELL", 1, 0)),
    ))
    explained = executor.execute(parse(
        'EXPLAIN FROM AT(REX("timeline"), 1) RETURN BOUNDARY(CELL(1, 1))'
    ))
    timed_explained = executor.execute(parse(
        'EXPLAIN FROM AT_TIME(REX("timeline"), 4.0) RETURN BOUNDARY(CELL(1, 1))'
    ))

    assert direct.values[0].cell.source.nE == 2
    assert seeded.values[0].grade == 0
    assert timed.values[0].cell.source.nE == 2
    assert held.values[0].cell.source.nE == 1
    plan = explained.values[0]
    assert plan["source"] == "timeline"
    assert plan["returns"][0]["result"]["temporal"] == {
        "version": 1, "as_of": None, "valid_at": None,
    }
    assert timed_explained.values[0]["returns"][0]["result"]["temporal"] == {
        "version": None, "as_of": 4.0, "valid_at": None,
    }
