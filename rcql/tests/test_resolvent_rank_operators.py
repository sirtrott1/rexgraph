"""RESOLVENT_RANK reads the core resolvent rank; the imported names state their implementation."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.coordinate_map import CoordinateMetric
from rexgraph.graph import RexGraph
from rexgraph.native_field import NativeFieldCalculus
from rexgraph.resolvent_ranking import resolvent_rank

from rcql import Executor, call, parse, query, source


def _path():
    return RexGraph(sources=np.array([0, 1]), targets=np.array([1, 2]), w_E=[Q(1), Q(1, 2)])


def _k4_one_face():
    rex = RexGraph(sources=np.array([0, 0, 0, 1, 1, 2], np.int32), targets=np.array([1, 2, 3, 2, 3, 3], np.int32))
    rex.add_faces([[0, 1, 3]])
    rex._ensure_clean()
    return rex


def _declared(rex):
    base = NativeFieldCalculus.from_rex(rex)
    spaces = base.complex.spaces
    return NativeFieldCalculus(base.complex, (
        CoordinateMetric.diagonal(spaces[0], [Q(1), Q(2), Q(3), Q(4)]),
        CoordinateMetric.diagonal(spaces[1], [Q(k + 1, 2) for k in range(6)]),
        CoordinateMetric.diagonal(spaces[2], [Q(5, 3)])))


def test_text_reproduces_the_weighted_pagerank_of_the_paper():
    out = Executor(sources={"r": _path()}).execute(parse(
        'FROM $r RETURN RESOLVENT_RANK(0, INDICATOR(CELL(0, 0)), 1/2, "walk"), RESOLVENT_RANK(0)'))
    assert list(out.values[0].values) == [Q(5, 9), Q(1, 3), Q(1, 9)]
    assert out.values[0].grade == 0
    assert list(out.values[1].values) == list(resolvent_rank(_path(), 0))
    assert [e.value for e in out.exactness] == ["rational", "rational"]


def test_a_field_calculus_supplies_metrics_at_every_grade():
    rex = _k4_one_face()
    calculus = _declared(rex)
    out = Executor(sources={"r": rex}, params={"c": calculus}).execute(parse(
        'FROM $r RETURN RESOLVENT_RANK(1, calculus=$c), EFFECTIVE_MODES(1, calculus=$c), '
        'EFFECTIVE_MODES(1, "completed", "energy", $c), HARMONIC_LOG(2, calculus=$c)'))
    from rexgraph.harmonic_modes import grade_traces
    assert list(out.values[0].values) == list(resolvent_rank(calculus, 1))
    assert out.values[1] == grade_traces(calculus, 1).effective_modes()
    assert out.values[2] == grade_traces(calculus, 1).effective_modes("completed", "energy")
    assert out.values[1] != grade_traces(rex, 1).effective_modes()
    assert out.values[3] == pytest.approx(grade_traces(calculus, 2).harmonic_log())


def test_metric_values_read_from_fields_replace_the_policy():
    from rexgraph.cochain import Cochain
    rex = _path()
    weights = Cochain(0, np.array([Q(1), Q(1, 3), Q(1, 2)], dtype=object), source=rex)
    out = Executor(sources={"r": rex}, params={"w": weights}).execute(parse(
        'FROM $r RETURN RESOLVENT_RANK(0, damping=1/2, metrics=[METRIC(0, $w)])'))
    assert list(out.values[0].values) == list(resolvent_rank(rex, 0, None, Q(1, 2), "walk"))


@pytest.mark.parametrize("name", ["MARKOV_VIEW", "PAGERANK", "PAGERANK_EXACT"])
def test_a_retired_name_says_what_replaced_it(name):
    with pytest.raises(KeyError, match="was renamed"):
        Executor(sources={"r": _path()}).execute(parse(f"FROM $r RETURN {name}()"))


def test_explain_is_typed_and_does_not_solve(monkeypatch):
    import rcql.executor
    import rexgraph.resolvent_ranking
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("EXPLAIN ran adapter"))
    monkeypatch.setattr(rexgraph.resolvent_ranking, "resolvent_rank", lambda *a, **k: pytest.fail("EXPLAIN solved"))
    plan = Executor(sources={"r": _k4_one_face()}).execute(
        replace(query(source("r"), call("RESOLVENT_RANK", 1)), explain=True))
    declared = plan.values[0]["returns"][0]
    assert declared["result"]["exactness"] == "rational"
    statuses = {p["name"]: p["status"] for p in declared["predicates"]}
    assert statuses["chain_condition"] == "verified" and statuses["resolvent_rank"] == "deferred"


@pytest.mark.parametrize("args", [(9,), (True,), (0, None, Q(1)), (0, None, Q(1, 2), "median"),
                                  (0, None, 0.85)])
@pytest.mark.parametrize("explain", [False, True])
def test_bad_arguments_are_refused_before_any_adapter(args, explain, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("invalid input ran adapter"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": _k4_one_face()}).execute(
            replace(query(source("r"), call("RESOLVENT_RANK", *args)), explain=explain))


@pytest.mark.parametrize("text", ['RESOLVENT_RANK(1, calculus=$c)', 'EFFECTIVE_MODES(1, calculus=$c)'])
def test_a_calculus_of_another_source_is_refused(text, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("foreign calculus ran adapter"))
    with pytest.raises(ValueError, match="another selected source"):
        Executor(sources={"r": _k4_one_face()}, params={"c": _declared(_k4_one_face())}).execute(
            parse(f"FROM $r RETURN {text}"))


def test_a_seed_at_another_grade_is_refused():
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": _k4_one_face()}).execute(parse(
            'FROM $r RETURN RESOLVENT_RANK(1, INDICATOR(CELL(0, 0)))'))
