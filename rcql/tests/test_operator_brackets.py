"""Brackets preserve operand contracts through text, parameters and native plans."""
import json
from fractions import Fraction as Q

import numpy as np
import pytest
import scipy.sparse as sp
from rexgraph.channel_operator import channel_operator
from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import RexOperator
from rexgraph.operator_bracket import operator_bracket
from rexgraph.weighted_dirac import GradedChain, weighted_dirac
from rexgraph.weighted_hodge import weighted_hodge

from rcql import Executor, call, param, parse, query, source


def make():
    return RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2])


def run(r, text, **params):
    return Executor(sources={"r": r}, params=params).execute(parse(text))


def test_exact_dirac_brackets_retain_full_tower_and_metric_provenance():
    r = make()
    m = DiagonalMetric(r, 1, (2, 3))
    x = GradedChain(r, [Chain(1, np.array([1, 2]), source=r)])
    result = run(r, '''FROM $r LET d=DIRAC([$m]) LET a=ANTI_DIRAC([$m])
        LET k=COMMUTATOR(left=d,right=a) LET z=ANTICOMMUTATOR(d,a)
        RETURN APPLY(k,$x,true), APPLY(z,$x,true), k, z''', m=m, x=x)
    h = weighted_hodge(r, 1, sector="difference", metric=m)
    np.testing.assert_array_equal(result.values[0].component(1).values, 2*h.apply(x.component(1).values, exact=True))
    assert not any(any(c.values.flat) for c in result.values[1].components)
    assert [e.value for e in result.exactness] == ["rational", "rational", "structural", "structural"]
    desc = result.provenance[2]["result_type"]["graded_operator"]
    assert desc["bracket_kind"] == "commutator" and desc["anti"] is None
    assert desc["metric_self_adjoint"] and not desc["psd"]
    assert [o["construction"] for o in desc["operands"]] == ["weighted-dirac"]*2
    assert m.coefficient_digest in json.dumps(result.native_plan, allow_nan=False)
    events = [e for e in result.execution if e["operator"] == "APPLY"]
    assert all(e["methods"][0]["method"] == "rational-operator-bracket-action" for e in events)
    node = next(n for n in result.native_plan["nodes"] if n.get("operator") == "COMMUTATOR")
    assert node["physical"]["method"] == "factored-operator-bracket-handle"


@pytest.mark.parametrize("name", ["COMMUTATOR", "ANTICOMMUTATOR"])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("columns", [None, 0, 3])
def test_text_builder_and_external_composed_handles_agree(name, graded, columns):
    r = make()
    shape = (2,) if columns is None else (2, columns)
    seed = Chain(1, np.arange(int(np.prod(shape))).reshape(shape)+1, source=r)
    x = GradedChain(r, [seed]) if graded else seed
    a, b = (weighted_dirac(r), weighted_dirac(r, anti=True)) if graded else (
        weighted_hodge(r, 1), weighted_hodge(r, 1, sector="difference"))
    k = operator_bracket(a, b, anti=name == "ANTICOMMUTATOR")
    text = f'FROM $r RETURN APPLY({name}(left=$a,right=$b),$x,true), APPLY($k,$x,true)'
    built = query(source("r"), call("APPLY", call(name, left=param("a"), right=param("b")), param("x"), True),
                  call("APPLY", param("k"), param("x"), True))
    assert built == parse(text)
    result = run(r, text, a=a, b=b, k=k, x=x)
    left, right = result.values
    if graded:
        for u, v in zip(left.components, right.components, strict=True):
            np.testing.assert_array_equal(u.values, v.values)
    else:
        np.testing.assert_array_equal(left.values, right.values)


def test_channel_bracket_exposes_exact_capability_but_not_green_psd_certificate():
    r = make()
    x = Cochain(1, np.array([1j, 2]), source=r)
    result = run(r, 'FROM $r LET k=COMMUTATOR(CHANNEL("T"),CHANNEL("G")) RETURN APPLY(k,$x), k', x=x)
    a, b = channel_operator(r, "T").as_scipy(), channel_operator(r, "G").as_scipy()
    np.testing.assert_allclose(result.values[0].values, (a@b-b@a)@x.values)
    desc = result.provenance[1]["result_type"]["operator"]
    assert desc["euclidean_skew_adjoint"] and desc["exact_action"] and not desc["psd"]
    assert result.provenance[0]["result_type"]["domain"] == "complex"
    assert [o["construction"] for o in desc["operands"]] == ["channel", "channel"]
    exact = run(r, 'FROM $r RETURN APPLY(COMMUTATOR(CHANNEL("T"),CHANNEL("G")),ZERO(1),true)')
    assert exact.exactness[0].value == "rational" and not any(exact.values[0].values)
    with pytest.raises(TypeError, match="PSD"):
        run(r, 'FROM $r RETURN RESOLVENT(ANTICOMMUTATOR(CHANNEL("T"),CHANNEL("G")))')


def test_nested_brackets_and_adjoint_keep_every_operand_descriptor():
    r = make()
    m = DiagonalMetric(r, 1, (2, 3))
    result = run(r, '''FROM $r LET a=HODGE_DOWN(1,$m) LET b=HODGE_SUM(1)
        LET k=COMMUTATOR(a,ANTICOMMUTATOR(a,b))
        RETURN APPLY(ADJOINT(k),$x,true), k''', m=m, x=Chain(1, np.array([1, 2]), source=r))
    k = result.values[1]
    np.testing.assert_array_equal(result.values[0].values, k.transpose_apply(np.array([1, 2]), exact=True))
    assert m.coefficient_digest in json.dumps(result.native_plan)


def test_different_graded_metrics_remain_composable_without_common_form_claim():
    r = make()
    m = DiagonalMetric(r, 1, (2, 3))
    result = run(r, 'FROM $r RETURN COMMUTATOR(DIRAC(),DIRAC([$m]))', m=m)
    desc = result.provenance[0]["result_type"]["graded_operator"]
    assert desc["grade_metrics"] == [] and desc["metric_self_adjoint"] is None and desc["metric_skew_adjoint"] is None
    assert m.coefficient_digest in json.dumps(desc)


def test_known_complex_matrix_and_its_adjoint_do_not_claim_real_outputs():
    r = make()
    a = sp.csr_matrix(np.array([[1j, 2], [3, 0]]))
    b = sp.csr_matrix(np.array([[1, 0], [0, 2]]))
    op = RexOperator("complex", (2, 2), 1, 1, lambda x: a@x, matrix=a, source=r, variance="chain")
    bp = RexOperator("real", (2, 2), 1, 1, lambda x: b@x, matrix=b, source=r, variance="chain")
    x = Chain(1, np.array([1, 2]), source=r)
    result = run(r, 'FROM $r LET k=ANTICOMMUTATOR($a,$b) RETURN APPLY(k,$x), APPLY(ADJOINT(k),$x)', a=op, b=bp, x=x)
    k = a@b+b@a
    np.testing.assert_allclose(result.values[0].values, k@x.values)
    np.testing.assert_allclose(result.values[1].values, k.conjugate().T@x.values)
    assert all(p["result_type"]["domain"] == "complex" for p in result.provenance)


@pytest.mark.parametrize("bad", ["source", "grade", "variance", "axes", "basis", "carrier-source", "mixed", "green"])
@pytest.mark.parametrize("explain", [False, True])
def test_invalid_spaces_fail_before_adapters(bad, explain, monkeypatch):
    r = make()
    a, b = weighted_hodge(r, 1), weighted_hodge(r, 1)
    x = Chain(1, np.ones(2, int), source=r)
    if bad == "source":
        b = weighted_hodge(make(), 1)
    elif bad == "grade":
        b = weighted_hodge(r, 0)
    elif bad == "variance":
        b = channel_operator(r, "T")
    elif bad == "axes":
        x = Chain(1, np.ones(3, int), source=r)
    elif bad == "basis":
        x = Chain(1, x.values, ("b", "a"), r)
    elif bad == "carrier-source":
        x = Chain(1, x.values, source=make())
    elif bad == "mixed":
        b = weighted_dirac(r)
    elif bad == "green":
        from rexgraph.green import GreenOperator
        b = GreenOperator.resolvent(b)
    monkeypatch.setattr("rcql.executor.get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(r, ("EXPLAIN " if explain else "")+'FROM $r RETURN APPLY(COMMUTATOR($a,$b),$x)', a=a, b=b, x=x)


def test_stale_channel_inside_external_bracket_is_found_before_execution(monkeypatch):
    r = make()
    k = operator_bracket(channel_operator(r, "T"), channel_operator(r, "G"))
    r._g_channel = "normalized"  # Simulate in place source drift, not a supported setter.
    monkeypatch.setattr("rcql.executor.get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises(ValueError, match="selection changed"):
        run(r, 'EXPLAIN FROM $r RETURN APPLY($k,ZERO(1))', k=k)


def test_no_exactness_is_inferred_from_accidental_numerical_zero():
    r = make()
    result = run(r, 'FROM $r LET a=HODGE_SUM(1,$m) RETURN APPLY(COMMUTATOR(a,a),$x)',
                 m=DiagonalMetric(r, 1, (2., 3.)), x=Chain(1, np.array([Q(1), Q(2)]), source=r))
    assert not any(result.values[0].values)
    assert result.exactness[0].value == "approximate"


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("shape", [(0,), (0, 2)])
def test_explicit_empty_upper_bracket_keeps_its_zero_axis(exact, shape):
    r = make()
    grade = len(r.graded_boundaries())+1
    op = RexOperator("empty", (0, 0), grade, grade, lambda x: x.copy(), source=r,
                     exact_matvec=lambda x: x.copy())
    x = Cochain(grade, np.empty(shape, int), source=r)
    result = run(r, 'FROM $r RETURN APPLY(COMMUTATOR($a,$a),$x,$exact)', a=op, x=x, exact=exact)
    assert result.values[0].grade == grade and result.values[0].values.shape == shape
    assert result.exactness[0].value == ("rational" if exact else "approximate")
