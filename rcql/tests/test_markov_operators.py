"""Markov compatibility planning, native actions and reopened store state."""
from contextlib import closing
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rcql import Executor, parse
from rexgraph import RexGraph
from rexgraph.cochain import Cochain
from rexgraph.markov import ParticipationWalk, pagerank_iteration


def fixture():
    return RexGraph.from_cells([4, [[0, 1], [1, 2]]], w_E=[Q(2), Q(3)])


def test_query_restart_actions_and_transpose_match_core():
    r = fixture()
    x = Cochain(0, np.array([Q(1), Q(0), Q(0), Q(0)], object), source=r)
    out = Executor(sources={"r": r}, params={"x": x}).execute(parse(
        'FROM $r LET p=PARTICIPATION_WALK() RETURN APPLY(p,$x,true),'
        'APPLY(ADJOINT(p),$x,true),PAGERANK_ITERATION(p,seed=$x),PAGERANK_ITERATION(p)'))
    view = ParticipationWalk(r)
    np.testing.assert_array_equal(out.values[0].values, view.apply(x.values, exact=True))
    np.testing.assert_array_equal(out.values[1].values, view.transpose_apply(x.values, exact=True))
    np.testing.assert_allclose(out.values[2].values, pagerank_iteration(view, seed=x.values))
    np.testing.assert_allclose(out.values[3].values, pagerank_iteration(view))


def test_indicator_restart_uses_canonical_c0():
    out = Executor(sources={"r": fixture()}).execute(parse(
        'FROM $r RETURN PAGERANK_ITERATION(PARTICIPATION_WALK(),seed=INDICATOR(CELL(0,0)))'))
    assert out.values[0].values.sum() == pytest.approx(1)


@pytest.mark.parametrize("expression", [
    'PARTICIPATION_WALK(1)', 'PARTICIPATION_WALK(true)', 'PARTICIPATION_WALK(policy="clique")',
    'PAGERANK_ITERATION(CHANNEL("T"))', 'PAGERANK_ITERATION(PARTICIPATION_WALK(),damping=1)',
    'PAGERANK_ITERATION(PARTICIPATION_WALK(),damping=true)', 'PAGERANK_ITERATION(PARTICIPATION_WALK(),tol=0)',
    'PAGERANK_ITERATION(PARTICIPATION_WALK(),maxiter=0)',
    'PAGERANK_ITERATION(PARTICIPATION_WALK(),seed=INDICATOR(CELL(1,0)))'])
@pytest.mark.parametrize("explain", [False, True])
def test_contract_refusals_precede_adapters(expression, explain, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((TypeError, ValueError, SyntaxError)):
        Executor(sources={"r": fixture()}).execute(replace(parse(f'FROM $r RETURN {expression}'), explain=explain))


def test_explain_never_builds_a_view_or_runs_a_solver(monkeypatch):
    import rexgraph.markov
    def refuse(*a, **kw):
        pytest.fail("execution reached from EXPLAIN")
    monkeypatch.setattr(rexgraph.markov.ParticipationWalk, "__init__", refuse)
    monkeypatch.setattr(rexgraph.markov, "pagerank", refuse)
    result = Executor(sources={"r": fixture()}).execute(parse(
        'EXPLAIN FROM $r RETURN PAGERANK_ITERATION(PARTICIPATION_WALK())'))
    assert not result.execution


@pytest.mark.parametrize("cells", [[[0, 1, 2]], [[0]]])
def test_explain_accepts_primary_branching_without_a_mode(cells):
    r = RexGraph.from_cells([3, cells])
    result = Executor(sources={"r": r}).execute(parse('EXPLAIN FROM $r RETURN PARTICIPATION_WALK()'))
    assert not result.execution


def test_native_view_parameter_and_foreign_source_refusal():
    r = fixture()
    view = ParticipationWalk(r)
    result = Executor(sources={"r": r}, params={"p": view}).execute(parse('FROM $r RETURN PAGERANK_ITERATION($p)'))
    np.testing.assert_allclose(result.values[0].values, pagerank_iteration(view))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": fixture()}, params={"p": view}).execute(parse('FROM $r RETURN PAGERANK_ITERATION($p)'))


def test_store_reopening_rebuilds_the_same_view_and_scores(tmp_path):
    rcdb = pytest.importorskip("rcdb")
    r = fixture()
    path = f"rex://{tmp_path / 'store'}"
    with closing(rcdb.open_store(path)) as store:
        store.put("r", r)
    with closing(rcdb.open_store(path)) as store:
        out = Executor(sources={"db": store}).execute(parse(
            'FROM RCDB_GET($db,"r") RETURN PAGERANK_ITERATION(PARTICIPATION_WALK()),'
            'APPLY(PARTICIPATION_WALK(),INDICATOR(CELL(0,0)),true)'))
    np.testing.assert_allclose(out.values[0].values, pagerank_iteration(ParticipationWalk(r)))
    assert sum(out.values[1].values) == 1


@pytest.mark.parametrize("stored", [False, True])
def test_tensor_query_preserves_policy_and_exact_action(tmp_path, stored):
    r = RexGraph.from_cells([4, [[0, 1, 2], [2, 3]]], w_E=[Q(2), Q(3)])
    query = (' RETURN APPLY(PARTICIPATION_WALK(),INDICATOR(CELL(0,0)),true),'
             'PAGERANK_ITERATION(PARTICIPATION_WALK()),PAGERANK_ITERATION($p)')
    view = ParticipationWalk(r)
    if stored:
        rcdb = pytest.importorskip("rcdb")
        uri = f"rex://{tmp_path / 'tensor'}"
        with closing(rcdb.open_store(uri)) as store:
            store.put("r", r)
        with closing(rcdb.open_store(uri)) as store:
            # Bind the parameter to the reopened object as required by identity.
            reopened = store.get("r")
            out = Executor(sources={"r": reopened}, params={"p": ParticipationWalk(reopened)}).execute(
                parse('FROM $r'+query))
            fresh = Executor(sources={"db": store}).execute(parse(
                'FROM RCDB_GET($db,"r") RETURN PAGERANK_ITERATION(PARTICIPATION_WALK())'))
            np.testing.assert_allclose(fresh.values[0].values, pagerank_iteration(view))
    else:
        out = Executor(sources={"r": r}, params={"p": view}).execute(parse('FROM $r'+query))
    np.testing.assert_array_equal(out.values[0].values, view.apply(np.array([1, 0, 0, 0], object), exact=True))
    for value in out.values[1:]:
        np.testing.assert_allclose(value.values, pagerank_iteration(view))


def test_explain_tensor_accepts_branching_without_execution(monkeypatch):
    import rexgraph.markov
    r = RexGraph.from_cells([3, [[0, 1, 2]]])
    monkeypatch.setattr(rexgraph.markov.ParticipationWalk, "__init__", lambda *a, **kw: pytest.fail("view built"))
    out = Executor(sources={"r": r}).execute(parse(
        'EXPLAIN FROM $r RETURN PAGERANK_ITERATION(PARTICIPATION_WALK())'))
    assert not out.execution


@pytest.mark.parametrize("arguments", ["", "grade=0"])
def test_markov_physical_plan_and_runtime_are_natively_tensor(arguments):
    query = f'FROM $r RETURN PAGERANK_ITERATION(PARTICIPATION_WALK({arguments}))'
    for explain in (False, True):
        out = Executor(sources={"r": fixture()}).execute(replace(parse(query), explain=explain))
        nodes = [n for n in out.native_plan["nodes"] if n.get("operator") == "PARTICIPATION_WALK"]
        assert len(nodes) == 1 and nodes[0]["physical"]["method"] == "core-participation-walk"
        if not explain:
            methods = [m["method"] for event in out.execution for m in event["methods"]]
            assert "native-participation-walk" in methods
            assert "native-tensor-pagerank" in methods


@pytest.mark.parametrize("policy", ["pairwise", "tensor"])
def test_removed_mode_flags_are_not_silently_reinterpreted(policy):
    for explain in (False, True):
        with pytest.raises((TypeError, ValueError, SyntaxError)):
            Executor(sources={"r": fixture()}).execute(replace(parse(
                f'FROM $r RETURN PARTICIPATION_WALK(policy="{policy}")'), explain=explain))
    with pytest.raises(TypeError):
        ParticipationWalk(fixture(), policy=policy)


def test_endpoint_oracle_cannot_enter_native_pagerank():
    from rexgraph.markov_oracle import PairwiseMarkovOracle
    r = fixture()
    oracle = PairwiseMarkovOracle(r)
    with pytest.raises(TypeError):
        pagerank_iteration(oracle)
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": r}, params={"p": oracle}).execute(parse('FROM $r RETURN PAGERANK_ITERATION($p)'))
