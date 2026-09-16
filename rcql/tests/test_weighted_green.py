"""Metric Green contracts survive planning, native execution and external handles."""
import json
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.green import GreenOperator
from rexgraph.weighted_hodge import weighted_hodge

from rcql import Executor, parse


def make():
    return RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2])


def run(r, text, **params):
    return Executor(sources={"r": r}, params=params).execute(parse(text))


def test_hand_solution_chain_metric_provenance_and_external_parity():
    r = make()
    x = Chain(1, np.array([3, 2]), source=r)
    m, lower = DiagonalMetric(r, 1, (2, 3)), DiagonalMetric(r, 0, (1, 2, 3, 4))
    g = GreenOperator.resolvent(weighted_hodge(r, 1, metric=m, lower_metric=lower))
    result = run(r, '''FROM $r LET h=HODGE_SUM(1,$m,$lower) LET g=RESOLVENT(h)
        LET y=GREEN_SOLVE(g,$x) RETURN y, APPLY(g,$x), GREEN_SOLVE($g,$x),
        MOMENT($x,y,$m), g''', x=x, m=m, lower=lower, g=g)
    for v in result.values[:3]:
        assert isinstance(v, Chain) and v.source is r
        np.testing.assert_allclose(v.values, [93/44, 41/22])
    assert result.values[3] == pytest.approx(525/22)
    assert [e.value for e in result.exactness] == ["approximate"]*4 + ["structural"]
    desc = result.provenance[4]["result_type"]["operator"]
    assert desc["metric_self_adjoint"] and desc["metric_psd"]
    assert not desc["symmetric"] and not desc["psd"]
    assert not desc["exact_action"] and not desc["exact_transpose"] and not desc["transpose_available"]
    assert desc["primal_operator"]["exact_action"] is True
    assert desc["construction"] == "metric-resolvent"
    encoded = json.dumps(result.native_plan, allow_nan=False)
    assert lower.coefficient_digest in encoded and m.coefficient_digest in encoded
    methods = [e["methods"][0] for e in result.execution if e["operator"] == "GREEN_SOLVE"]
    assert methods and all(e["kernel"] == "native-metric-block-cg" for e in methods)
    assert all(e["residual_norm"] == "euclidean-original-system" for e in methods)


@pytest.mark.parametrize("sector", ["DOWN", "UP", "SUM"])
@pytest.mark.parametrize("columns", [None, 0, 3])
def test_text_and_external_block_handles_agree(sector, columns):
    r = make()
    shape = (2,) if columns is None else (2, columns)
    x = Chain(1, np.arange(int(np.prod(shape))).reshape(shape), source=r)
    m = DiagonalMetric(r, 1, (Q(2, 3), Q(4, 5)))
    h = weighted_hodge(r, 1, sector=sector.lower(), metric=m)
    g = GreenOperator.resolvent(h, Q(1, 3), tol=1e-9, maxiter=42)
    result = run(r, f'''FROM $r RETURN GREEN_SOLVE(RESOLVENT(HODGE_{sector}(1,$m),
        alpha=1/3,tol=1/1000000000,maxiter=42),$x), APPLY($g,$x)''', x=x, m=m, g=g)
    for v in result.values:
        np.testing.assert_allclose(v.values, g.solve(x.values))
        assert v.values.shape == shape
    assert '"maxiter", 42' in json.dumps(result.native_plan)


@pytest.mark.parametrize("bad", ["variance", "source", "grade", "basis", "shape", "complex", "exact"])
@pytest.mark.parametrize("explain", [False, True])
def test_invalid_carriers_are_refused_before_adapter(bad, explain, monkeypatch):
    r = make()
    x = Chain(1, np.ones(2, int), source=r)
    if bad == "variance":
        x = Cochain(1, x.values, source=r)
    elif bad == "source":
        x = Chain(1, x.values, source=make())
    elif bad == "grade":
        x = Chain(0, np.ones(4, int), source=r)
    elif bad == "basis":
        x = Chain(1, x.values, ("b", "a"), r)
    elif bad == "shape":
        x = Chain(1, np.ones(3, int), source=r)
    elif bad == "complex":
        x = x.with_values(np.array([1j, 2]))
    text = "FROM $r RETURN APPLY(RESOLVENT(HODGE_SUM(1)),$x" + (",true)" if bad == "exact" else ")")
    monkeypatch.setattr("rcql.executor.get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((ValueError, TypeError)):
        run(r, ("EXPLAIN " if explain else "")+text, x=x)


def test_computed_metric_proof_is_deferred_and_invalid_value_refused():
    text = "FROM $r RETURN RESOLVENT(HODGE_SUM(1, METRIC(1,ZERO(1))))"
    explained = run(make(), "EXPLAIN "+text)
    encoded = json.dumps(explained.values)
    assert "resolvent_metric_psd" in encoded and "deferred" in encoded
    with pytest.raises(ValueError, match="positive"):
        run(make(), text)


@pytest.mark.parametrize("expression", ["RESOLVENT(HODGE_DIFFERENCE(1))", "RESOLVENT(DIRAC())",
    "ADJOINT(RESOLVENT(HODGE_SUM(1)))", "GREEN_SOLVE(RESOLVENT(HODGE_OPERATOR(1)),$x)"])
def test_no_indefinite_dirac_or_transpose_capability_is_invented(expression, monkeypatch):
    r = make()
    monkeypatch.setattr("rcql.executor.get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((ValueError, TypeError)):
        run(r, "EXPLAIN FROM $r RETURN "+expression, x=Chain(1, np.ones(2, int), source=r))
