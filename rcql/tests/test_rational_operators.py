"""Rational action planning, source ownership and native store reopening."""
from dataclasses import replace
from contextlib import closing
from fractions import Fraction as Q

import numpy as np
import pytest

from rcql import Executor, parse
from rexgraph import RexGraph
from rexgraph.channel_operator import channel_operator
from rexgraph.cochain import Cochain
from rexgraph.operator_bracket import operator_bracket
from rexgraph.rational_operator import cayley, complex_structure, rational_rotation


def fixture():
    return RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2], g_channel="raw")


G = 'COMMUTATOR(CHANNEL("T"),CHANNEL("G"))'


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("name", ["CAYLEY", "COMPLEX_STRUCTURE", "RATIONAL_ROTATION"])
def test_query_actions_match_core(name, exact):
    r = fixture()
    g = operator_bracket(channel_operator(r, "T"), channel_operator(r, "G"))
    choices = {"CAYLEY": (f'CAYLEY({G},1/2)', cayley(g, Q(1, 2))),
               "COMPLEX_STRUCTURE": (f'COMPLEX_STRUCTURE({G})', complex_structure(g)),
               "RATIONAL_ROTATION": (f'RATIONAL_ROTATION(COMPLEX_STRUCTURE({G}),3,4,5)',
                                     rational_rotation(complex_structure(g), 3, 4, 5))}
    expression, action = choices[name]
    x = Cochain(1, np.array([Q(2, 3), Q(5, 7)], object), source=r)
    out = Executor(sources={"r": r}, params={"x": x, "exact": exact}).execute(parse(
        f'FROM $r LET a={expression} RETURN APPLY(a,$x,$exact),APPLY(ADJOINT(a),$x,$exact)'))
    np.testing.assert_array_equal(out.values[0].values, action.apply(x.values, exact=exact))
    np.testing.assert_array_equal(out.values[1].values, action.transpose_apply(x.values, exact=exact))


@pytest.mark.parametrize("expression", [
    'CAYLEY(CHANNEL("T"),1)', f'CAYLEY({G},0.5)', f'CAYLEY({G},true)',
    f'CAYLEY({G},1,tol=0.0)', f'CAYLEY({G},1,maxiter=0)',
    f'COMPLEX_STRUCTURE({G},scale=0)', f'COMPLEX_STRUCTURE({G},scale=0.5)',
    f'RATIONAL_ROTATION({G},3,4,5)', f'RATIONAL_ROTATION(COMPLEX_STRUCTURE({G}),1,1,1)',
    f'RATIONAL_ROTATION(COMPLEX_STRUCTURE({G}),3,4,-5)'])
@pytest.mark.parametrize("explain", [False, True])
def test_bad_contracts_fail_before_adapters(expression, explain, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"r": fixture()}).execute(replace(parse(f'FROM $r RETURN {expression}'), explain=explain))


def test_explain_runs_no_generator_or_solver(monkeypatch):
    import rexgraph.rational_operator as core
    def refuse(*a, **kw):
        pytest.fail("Core action ran during EXPLAIN")
    for name in ("cayley", "complex_structure", "rational_rotation"):
        monkeypatch.setattr(core, name, refuse)
    out = Executor(sources={"r": fixture()}).execute(parse(
        f'EXPLAIN FROM $r RETURN CAYLEY({G},1/2),RATIONAL_ROTATION(COMPLEX_STRUCTURE({G}),3,4,5)'))
    assert not out.execution


def test_literal_structure_preserves_descriptor_and_refuses_foreign_state():
    r = fixture()
    j = complex_structure(operator_bracket(channel_operator(r, "T"), channel_operator(r, "G")))
    x = Cochain(1, np.array([1, 2], object), source=r)
    out = Executor(sources={"r": r}, params={"j": j, "x": x}).execute(parse(
        'FROM $r RETURN APPLY(RATIONAL_ROTATION($j,3,4,5),$x,true)'))
    np.testing.assert_array_equal(out.values[0].values, rational_rotation(j, 3, 4, 5).apply(x.values, exact=True))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"other": fixture()}, params={"j": j}).execute(parse(
            'FROM $other RETURN RATIONAL_ROTATION($j,3,4,5)'))


def test_rcdb_reopening_recomputes_native_certified_actions(tmp_path):
    rcdb = pytest.importorskip("rcdb")
    path = f"rex://{tmp_path / 'store'}"
    r = fixture()
    with closing(rcdb.open_store(path)) as db:
        db.put("r", r)
    query = parse(f'FROM RCDB_GET($db,"r") LET x=INDICATOR(CELL(1,0)) '
                  f'RETURN APPLY(CAYLEY({G},1/2),x,true),'
                  f'APPLY(RATIONAL_ROTATION(COMPLEX_STRUCTURE({G}),3,4,5),x,true)')
    with closing(rcdb.open_store(path)) as db:
        out = Executor(sources={"db": db}).execute(query)
        assert all(all(isinstance(v, Q) for v in value.values) for value in out.values)
        assert all(sum(v*v for v in value.values) == 1 for value in out.values)


def test_complex_coefficients_match_the_declared_query_domain():
    r = fixture()
    x = Cochain(1, np.array([1+2j, 3-4j]), source=r)
    out = Executor(sources={"r": r}, params={"x": x}).execute(parse(
        f'FROM $r LET c=CAYLEY({G},1/2) RETURN APPLY(ADJOINT(c),APPLY(c,$x))'))
    np.testing.assert_allclose(out.values[0].values, x.values)
