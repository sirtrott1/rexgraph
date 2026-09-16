"""Typed group words, inverses and persistent native source reads."""
from dataclasses import replace
from fractions import Fraction as Q
from contextlib import closing

import numpy as np
import pytest

from rcql import Executor, parse
from rexgraph import RexGraph
from rexgraph.cochain import Cochain


def fixture():
    return RexGraph.from_cells([4, [[2, 0, 1, 3], [0, 2]]], g_channel="raw")


GROUP = 'RESOLVENT_GROUP([CHANNEL("T"),CHANNEL("G")],[1/2,2/3],[1,-2,1])'


@pytest.mark.parametrize("exact", [True, False])
def test_native_group_action_inverse_and_members(exact):
    r = fixture()
    x = Cochain(1, np.array([Q(2, 3), Q(-1, 7)], object), source=r)
    out = Executor(sources={"r": r}, params={"x": x, "e": exact}).execute(parse(
        f'FROM $r LET g={GROUP} RETURN APPLY(g.inverse,APPLY(g,$x,$e),$e),g.word,g.scales,'
        'APPLY(g.identity,$x,$e),APPLY(ADJOINT(g),$x,$e)'))
    if exact:
        np.testing.assert_array_equal(out.values[0].values, x.values)
    else:
        np.testing.assert_allclose(out.values[0].values, np.asarray(x.values, float), atol=1e-9)
    assert out.values[1] == (1, -2, 1) and out.values[2] == (Q(1, 2), Q(2, 3))
    assert "native-resolvent-group" in str(out.execution)


@pytest.mark.parametrize("expression", [
    'RESOLVENT_GROUP([],[])', 'RESOLVENT_GROUP([CHANNEL("T")],[-1])',
    'RESOLVENT_GROUP([CHANNEL("T")],[0.5])', 'RESOLVENT_GROUP([CHANNEL("T")],[true])',
    'RESOLVENT_GROUP([CHANNEL("T")],[1],[0])', 'RESOLVENT_GROUP([CHANNEL("T")],[1],[2])',
    'RESOLVENT_GROUP([CHANNEL("T")],[1],[true])', 'RESOLVENT_GROUP([CHANNEL("T")],[1],tol=0.0)',
    'RESOLVENT_GROUP([CHANNEL("T")],[1],maxiter=0)', 'RESOLVENT_GROUP([CELL(1,0)],[1])',
    'RESOLVENT_GROUP([COMMUTATOR(CHANNEL("T"),CHANNEL("G"))],[1])', f'{GROUP}.operators'])
@pytest.mark.parametrize("explain", [False, True])
def test_invalid_contracts_refused_before_adapter(expression, explain, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": fixture()}).execute(replace(parse(f'FROM $r RETURN {expression}'), explain=explain))


def test_explain_no_construction_or_solve(monkeypatch):
    import rexgraph.rational_operator as core
    monkeypatch.setattr(core.ResolventGroup, "__init__", lambda *a, **kw: pytest.fail("group ran"))
    out = Executor(sources={"r": fixture()}).execute(parse(f'EXPLAIN FROM $r RETURN {GROUP}.inverse'))
    assert not out.execution


def test_foreign_parameter_and_existing_group_binding():
    from rexgraph.channel_operator import channel_operator
    from rexgraph.rational_operator import ResolventGroup
    r = fixture()
    foreign = channel_operator(fixture(), "T")
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": r}, params={"op": foreign}).execute(parse(
            'FROM $r RETURN RESOLVENT_GROUP([$op],[1])'))
    group = ResolventGroup([channel_operator(r, "T")], [1], [1])
    out = Executor(sources={"r": r}, params={"g": group}).execute(parse(
        'FROM $r RETURN $g.inverse.word'))
    assert out.values == ((-1,),)


def test_rcdb_reopen_group_does_not_publish(tmp_path):
    import rcdb
    from rexgraph.io.catalog import object_digest
    r = fixture()
    path = f"rex://{tmp_path / 'db'}"
    with closing(rcdb.open_store(path)) as db:
        db.put("r", r)
    with closing(rcdb.open_store(path)) as db:
        out = Executor(sources={"db": db}).execute(parse(
            f'FROM RCDB_GET($db,"r") LET g={GROUP} LET x=INDICATOR(CELL(1,0)) '
            'RETURN APPLY(g.inverse,APPLY(g,x,true),true),STATE_HASH()'))
        np.testing.assert_array_equal(out.values[0].values, [Q(1), Q(0)])
        assert out.values[1] == object_digest(r) == object_digest(db.get("r"))
