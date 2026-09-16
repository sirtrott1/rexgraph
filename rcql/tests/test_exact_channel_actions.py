"""The exact channel action is the same operation, source and carrier as numeric APPLY."""
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.channel_operator import channel_operator
from rexgraph.cochain import Chain, Cochain, Field
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.operator_bracket import operator_bracket

from rcql import Executor, call, param, parse, query, source


def make(normalized=False):
    return RexGraph(boundary_ptr=np.array([0, 4, 6], np.int32),
                    boundary_idx=np.array([0, 1, 2, 3, 1, 0], np.int32),
                    w_E=np.array([Q(2), Q(3) if normalized else Q(-3)], object),
                    g_channel="normalized" if normalized else "raw")


def run(r, text, **params):
    return Executor(sources={"r": r}, params=params).execute(parse(text))


MATRICES = {
    "T": np.array([[Q(16, 3), Q(8)], [Q(8), Q(18)]], object),
    "G": np.array([[Q(16, 3), Q(-8)], [Q(-8), Q(18)]], object),
    "F": np.array([[Q(16), Q(16)], [Q(16), Q(16)]], object),
    "C": np.array([[Q(4, 3), Q(-4, 3)], [Q(-4, 3), Q(4, 3)]], object),
}


@pytest.mark.parametrize("name", list("TGFC"))
@pytest.mark.parametrize("width", [None, 0, 2])
def test_text_builder_external_handle_and_exact_field_chaining(name, width):
    r = make()
    shape = (2,) if width is None else (2, width)
    x = Cochain(1, np.asarray([Q(i-2, 3) for i in range(int(np.prod(shape)))], object).reshape(shape), source=r)
    text = f'FROM $r LET c=CHANNEL("{name}") LET y=APPLY(c,$x,exact=true) RETURN y, APPLY($c,y,true), c'
    built = query(source("r"), call("APPLY", call("CHANNEL", name), param("x"), exact=True))
    assert built == parse(f'FROM $r RETURN APPLY(CHANNEL("{name}"),$x,exact=true)')
    result = run(r, text, x=x, c=channel_operator(r, name))
    assert all(isinstance(v, Field) for v in result.values[:2])
    np.testing.assert_array_equal(result.values[0].values, MATRICES[name]@x.values)
    np.testing.assert_array_equal(result.values[1].values, MATRICES[name]@MATRICES[name]@x.values)
    assert [e.value for e in result.exactness] == ["rational", "rational", "structural"]
    desc = result.provenance[2]["result_type"]["operator"]
    assert desc["exact_action"] and desc["exact_transpose"] and desc["coefficient_domain"] == "rational"
    assert result.provenance[0]["result_type"]["kind"] == "Field"
    methods = [e["methods"][0]["method"] for e in result.execution if e["operator"] == "APPLY"]
    assert methods == ["rational-channel-action"]*2
    nodes = [n for n in result.native_plan["nodes"] if n.get("operator") == "APPLY"]
    assert all(n["physical"]["method"] == "rational-channel-action" for n in nodes)


@pytest.mark.parametrize("name", list("TGFC"))
def test_numeric_defaults_still_return_approximate_fields(name):
    r = make()
    x = Cochain(1, np.array([1, 2]), source=r)
    result = run(r, f'FROM $r RETURN APPLY(CHANNEL("{name}"),$x)', x=x)
    assert isinstance(result.values[0], Field) and result.exactness[0].value == "approximate"
    np.testing.assert_allclose(result.values[0].values, np.asarray(MATRICES[name]@x.values, float))


def test_exact_bracket_and_metric_adjoint_share_the_channel_hooks():
    r = make()
    x = Cochain(1, np.array([Q(3, 2), Q(-2, 5)]), source=r)
    m = DiagonalMetric(r, 1, (2, 3))
    result = run(r, '''FROM $r LET t=CHANNEL("T") LET f=CHANNEL("F")
        LET k=COMMUTATOR(t,f) RETURN APPLY(k,$x,true), APPLY(ADJOINT(t,$m,$m),$x,true),
        APPLY($k,$x,true), MOMENT($x,APPLY(f,$x,true),none,true)''',
        x=x, m=m, k=operator_bracket(channel_operator(r, "T"), channel_operator(r, "F")))
    t, f = MATRICES["T"], MATRICES["F"]
    np.testing.assert_array_equal(result.values[0].values, (t@f-f@t)@x.values)
    np.testing.assert_array_equal(result.values[2].values, result.values[0].values)
    np.testing.assert_array_equal(result.values[1].values, (t@(np.array([2, 3])*x.values))/np.array([2, 3]))
    assert result.values[3] == Q(484, 25)
    assert all(e.value == "rational" for e in result.exactness)


def test_explain_does_not_apply_or_build_any_exact_or_numeric_channel(monkeypatch):
    r = make()
    def forbidden(*a, **kw):
        pytest.fail("EXPLAIN built or executed a channel action")
    monkeypatch.setattr("rexgraph.channel_operator._exact_channel_action", forbidden)
    monkeypatch.setattr("rexgraph.channel_operator.build_factored_operator", forbidden)
    monkeypatch.setattr("rcql.executor.get_operator", forbidden)
    result = run(r, 'EXPLAIN FROM $r RETURN APPLY(CHANNEL("F"),ZERO(1),true)')
    assert "rational-channel-action" in str(result.native_plan)


@pytest.mark.parametrize("bad", ["normalized", "float", "complex", "chain", "foreign", "grade", "axis", "basis"])
@pytest.mark.parametrize("explain", [False, True])
def test_exact_domain_refusals_are_preexecution(bad, explain, monkeypatch):
    r = make(bad == "normalized")
    x = Cochain(1, np.ones(2, int), source=r)
    if bad == "float":
        x = x.with_values(np.ones(2, float))
    elif bad == "complex":
        x = x.with_values(np.ones(2, complex))
    elif bad == "chain":
        x = Chain(1, x.values, source=r)
    elif bad == "foreign":
        x = Cochain(1, x.values, source=make())
    elif bad == "grade":
        x = Cochain(0, np.ones(4, int), source=r)
    elif bad == "axis":
        x = x.with_values(np.ones(3, int))
    elif bad == "basis":
        x = Cochain(1, x.values, cell_keys=("b", "a"), source=r)
    monkeypatch.setattr("rcql.executor.get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(r, ("EXPLAIN " if explain else "")+'FROM $r RETURN APPLY(CHANNEL("G"),$x,true)', x=x)


def test_normalized_diagonal_and_exact_F_coexist():
    r = make(True)
    x = Cochain(1, np.array([1, 2]), source=r)
    result = run(r, 'FROM $r RETURN APPLY(CHANNEL("F"),$x,true), SCALE_MOMENT(CHANNEL("G"),1,true,true)', x=x)
    assert result.values[0].values.tolist() == [Q(-16), Q(16)]
    assert all(e.value == "rational" for e in result.exactness)


@pytest.mark.parametrize("weight", [Q(1, 10**400), Q(10**400)])
def test_exact_action_does_not_cross_the_float_metric_boundary(weight):
    r = RexGraph.from_graph([0], [1], w_E=np.array([weight], object))
    x = Cochain(1, np.array([1]), source=r)
    result = run(r, 'FROM $r RETURN APPLY(CHANNEL("T"),$x,true)', x=x)
    assert result.values[0].values[0] == 2*weight**2


@pytest.mark.parametrize("name", list("TGFC"))
def test_empty_exact_fields_preserve_block_axes(name):
    r = RexGraph.from_graph([], [])
    x = Cochain(1, np.empty((0, 2), int), source=r)
    result = run(r, f'FROM $r RETURN APPLY(CHANNEL("{name}"),$x,true)', x=x)
    assert result.values[0].values.shape == (0, 2) and result.exactness[0].value == "rational"
