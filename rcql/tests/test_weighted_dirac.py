"""Exact graded calculus, refusal before execution, and native plan contracts."""
import json
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import RexOperator
from rexgraph.weighted_dirac import GradedChain, weighted_dirac

from rcql import Executor, call, param, parse, query, source


def make():
    return RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2])


def run(rex, text, **params):
    return Executor(sources={"r": rex}, params=params).execute(parse(text))


def test_exact_hand_query_and_full_descriptor_provenance():
    rex = make()
    result = run(rex, '''FROM $r LET m=METRIC(1,$w) LET v=METRIC(0,$v)
        LET d=DIRAC(metrics=[m,v]) LET a=ANTI_DIRAC([m,v])
        LET x=GRADED_CHAIN(components=[$x0,$x1])
        LET dx=APPLY(d,x,exact=true)
        RETURN dx AS dx, GRADE_COMPONENT(APPLY(a,x,true),1) AS ax1,
          GRADE_COMPONENT(APPLY(d,dx,true),1) AS ddx1,
          MOMENT($x1,GRADE_COMPONENT(APPLY(d,dx,true),1),m,true) AS energy''',
        w=Cochain(1, np.array([2, 3]), source=rex), v=Cochain(0, np.array([1, 2, 3, 4]), source=rex),
        x0=Chain(0, np.array([1, 2, 3, 4]), source=rex), x1=Chain(1, np.array([3, 2]), source=rex))
    assert result.named_values["dx"].component(0).values.tolist() == [Q(-1), Q(1), Q(-1), Q(1)]
    assert result.named_values["dx"].component(1).values.tolist() == [Q(-1), Q(8, 3)]
    assert result.named_values["ax1"].values.tolist() == [Q(1), Q(-8, 3)]
    assert result.named_values["ddx1"].values.tolist() == [Q(7, 3), Q(-2, 3)]
    assert result.named_values["energy"] == 10
    assert [e.value for e in result.exactness] == ["rational"]*4
    node = next(n for n in result.native_plan["nodes"] if n.get("operator") == "DIRAC")
    desc = node["result"]["graded_operator"]
    assert [b["grade"] for b in desc["bases"]] == [0, 1]
    assert desc["sizes"] == [4, 2] and desc["exact_action"]
    assert desc["metric_self_adjoint"] and not desc["psd"] and not desc["symmetric"]
    assert desc["grade_metrics"][1]["coefficient_digest"] == DiagonalMetric(rex, 1, (2, 3)).coefficient_digest
    assert node["physical"]["method"] == "factored-weighted-dirac-handle"
    assert any(e["methods"][0]["method"] == "rational-weighted-dirac-action"
               for e in result.execution if e["operator"] == "APPLY")
    assert len([n for n in result.native_plan["nodes"] if n.get("operator") == "DIRAC"]) == 1
    json.dumps(result.native_plan, allow_nan=False)


@pytest.mark.parametrize("anti", [False, True])
@pytest.mark.parametrize("columns", [None, 0, 3])
@pytest.mark.parametrize("exact", [False, True])
def test_text_builder_external_and_core_agree(anti, columns, exact):
    rex = make()
    shape = (2,) if columns is None else (2, columns)
    x = Chain(1, np.ones(shape, int), source=rex)
    m = DiagonalMetric(rex, 1, (2, 3))
    name = "ANTI_DIRAC" if anti else "DIRAC"
    text = f'FROM $r RETURN APPLY({name}(metrics=[$m]),GRADED_CHAIN([$x]),{str(exact).lower()})'
    built = query(source("r"), call("APPLY", call(name, metrics=[param("m")]), call("GRADED_CHAIN", [param("x")]), exact))
    assert built == parse(text)
    state, action = GradedChain(rex, [x]), weighted_dirac(rex, metrics=[m], anti=anti)
    result = run(rex, text, x=x, m=m)
    ext = run(rex, f'FROM $r RETURN APPLY($d,$x,{str(exact).lower()})', x=state, d=action)
    via_list_parameter = run(rex, f'FROM $r RETURN APPLY({name}($metrics),GRADED_CHAIN($components),{str(exact).lower()})',
                             components=[x], metrics=[m])
    for out in (result.values[0], ext.values[0], via_list_parameter.values[0]):
        for c, expected in zip(out.components, action.apply(state, exact=exact).components, strict=True):
            np.testing.assert_array_equal(c.values, expected.values)
        assert out.sizes == (4, 2)
        assert out.component(0).values.shape == (4, *shape[1:])


@pytest.mark.parametrize("bad", ["variance", "basis", "shape", "source", "duplicate", "block",
                                  "metric-source", "metric-basis", "metric-duplicate", "metric-grade",
                                  "metric-kind", "metric-exact", "carrier-exact", "bool", "text"])
@pytest.mark.parametrize("explain", [False, True])
def test_bad_input_before_any_adapter(bad, explain, monkeypatch):
    rex = make()
    x = Chain(1, np.array([1, 2]), source=rex)
    m = DiagonalMetric(rex, 1, (2, 3))
    components, metrics = [x], [m]
    if bad == "variance": components = [Cochain(1, x.values, source=rex)]
    if bad == "basis": components = [Chain(1, x.values, ("a", "b"), rex)]
    if bad == "shape": components = [x.with_values(np.ones(3, int))]
    if bad == "source": components = [Chain(1, x.values, source=make())]
    if bad == "duplicate": components = [x, x]
    if bad == "block": components += [Chain(0, np.ones((4, 1), int), source=rex)]
    if bad == "metric-source": metrics = [DiagonalMetric(make(), 1, (2, 3))]
    if bad == "metric-basis": metrics = [DiagonalMetric(rex, 1, (2, 3), ("a", "b"))]
    if bad == "metric-duplicate": metrics = [m, m]
    if bad == "metric-grade": metrics = [DiagonalMetric(rex, 2, ())]
    if bad == "metric-kind": metrics = [x]
    if bad == "metric-exact": metrics = [DiagonalMetric(rex, 1, (2., 3.))]
    if bad == "carrier-exact": components = [x.with_values(np.ones(2))]
    if bad == "bool": components = [x.with_values(np.ones(2, bool))]
    if bad == "text": components = [x.with_values(np.array(["a", "b"]))]
    monkeypatch.setattr("rcql.executor.get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, ("EXPLAIN " if explain else "") +
            'FROM $r RETURN APPLY(DIRAC($metrics),GRADED_CHAIN($components),true)', metrics=metrics, components=components)


@pytest.mark.parametrize("expr", [
    'DIRAC(grades=[0,1])', 'ANTI_DIRAC(types=["x"])', 'DIRAC(1)',
    'APPLY(DIRAC(),ZERO(1,"chain"))', 'GRADED_CHAIN([ZERO(1)])',
    'GRADE_COMPONENT(GRADED_CHAIN([]),-1)', 'GRADE_COMPONENT(GRADED_CHAIN([]),2)',
    'GRADE_COMPONENT(GRADED_CHAIN([]),true)', 'APPLY(HODGE_SUM(1),GRADED_CHAIN([]))',
    'RESOLVENT(DIRAC())', 'ADJOINT(DIRAC())', 'MOMENT(GRADED_CHAIN([]),GRADED_CHAIN([]))',
    'GRADE(DIRAC())', 'GRADE(GRADED_CHAIN([]))',
])
def test_unsupported_or_wrong_variance_contracts_refused(expr, monkeypatch):
    monkeypatch.setattr("rcql.executor.get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError, SyntaxError)):
        run(make(), "EXPLAIN FROM $r RETURN " + expr)


def test_full_higher_tower_and_signed_square():
    rex = RexGraph.from_cells(solid_octahedron_3rex())
    x = Chain(3, np.ones(1, int), source=rex)
    result = run(rex, '''FROM $r LET x=GRADED_CHAIN([$x]) LET d=DIRAC() LET a=ANTI_DIRAC()
        RETURN GRADE_COMPONENT(APPLY(d,APPLY(d,x,true),true),3),
          GRADE_COMPONENT(APPLY(a,APPLY(a,x,true),true),3), APPLY(HODGE_SUM(3),$x,true)''', x=x)
    np.testing.assert_array_equal(result.values[0].values, result.values[2].values)
    np.testing.assert_array_equal(result.values[1].values, -result.values[2].values)


def test_exact_domain_unsupported_higher_refused_before_adapters(monkeypatch):
    from rexgraph.native_sparse import as_native
    rex = RexGraph.from_cells(solid_octahedron_3rex())
    upper = as_native(rex._graded_duals[0])
    rex._graded_duals = [upper.with_data(upper.data * .5).dual]
    run(rex, 'FROM $r RETURN APPLY(DIRAC(),GRADED_CHAIN([]))')
    monkeypatch.setattr("rcql.executor.get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises(TypeError, match="certified exact"):
        run(rex, 'FROM $r RETURN APPLY(DIRAC(),GRADED_CHAIN([]),true)')


def test_complex_and_no_materialization(monkeypatch):
    rex = make()
    x = Chain(1, np.array([1j, 2]), source=rex)
    monkeypatch.setattr(RexOperator, "as_scipy", lambda *a: pytest.fail("matrix assembled"))
    result = run(rex, 'FROM $r RETURN GRADE_COMPONENT(APPLY(DIRAC(),GRADED_CHAIN([$x])),0)', x=x)
    np.testing.assert_allclose(result.values[0].values, [-2+1j/3, 1j/3, 2-1j, 1j/3])
    assert result.provenance[0]["result_type"]["domain"] == "complex"


def test_computed_metric_positivity_remains_deferred():
    rex = make()
    text = 'FROM $r RETURN DIRAC([METRIC(1,ZERO(1))])'
    result = run(rex, "EXPLAIN " + text)
    encoded = json.dumps(result.values)
    assert 'deferred' in encoded and 'metric_self_adjoint' in encoded
    with pytest.raises(ValueError, match="positive"):
        run(rex, text)


def test_empty_tower_and_exact_omitted_grade_zeros():
    rex = RexGraph(sources=np.array([], int), targets=np.array([], int))
    result = run(rex, 'FROM $r RETURN APPLY(ANTI_DIRAC(),GRADED_CHAIN([]),true)')
    assert result.values[0].sizes == (0, 0) and result.values[0].exact


@pytest.mark.parametrize("bad", ["shape", "basis", "time", "block"])
def test_explicit_graded_type_declarations_are_checked(bad, monkeypatch):
    from rcql import SourcePolicy
    from rcql.binding import bind
    from rcql.planning import _carrier_literal
    from rcql.types import BasisRef, TemporalRef
    rex = make()
    binding = bind("r", rex, SourcePolicy.allow("*"))
    state = _carrier_literal(binding, GradedChain(rex))
    if bad == "shape": state = state.with_(member_shapes=())
    if bad == "basis": state = state.with_(graded_bases=(BasisRef("r", 1), BasisRef("r", 0)))
    if bad == "time": state = state.with_(temporal=TemporalRef(version=99))
    if bad == "block": state = state.with_(member_shapes=((4, 2), (2, 1)))
    monkeypatch.setattr("rcql.executor.get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, 'EXPLAIN FROM $r RETURN GRADE_COMPONENT($state,0)', state=state)
