"""Rectangular type coordinates retain explicit cross endpoints in native plans."""
import json
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import diagonal_metric
from rexgraph.graph import RexGraph
from rexgraph.type_accession import AccessionFamily, CoordinateSpace, CrossMetric, TypeAccession

from rcql import BoundSource, Executor, SourcePolicy, bind, call, param, parse, query, source
from rcql.planning import _carrier_literal
from rcql.types import ShapeRef, TemporalRef


def make():
    rex = RexGraph.from_graph([0, 1], [1, 2])
    a = TypeAccession(rex, 1, "summary", ((0, 0, 1), (0, 1, 2)),
                      coordinates=CoordinateSpace("summary-space", ("total",)))
    b = TypeAccession(rex, 1, "detail", ((0, 0, 1), (1, 1, -1), (2, 0, 1), (2, 1, 1)),
                      coordinates=CoordinateSpace("detail-space", ("first", "second", "joint")))
    metric = CrossMetric(a, b, ((0, 0, 2), (0, 1, -3), (0, 2, Q(1, 2))))
    return rex, a, b, metric


def co(rex, values):
    return Cochain(1, np.asarray(values), source=rex)


def run(rex, *exprs, explain=False, **params):
    return Executor(sources={"r": rex}, params=params).execute(query(source("r"), *exprs, explain=explain))


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("block", [False, True])
def test_nested_rectangular_accessions_and_cross_block_keep_full_provenance(exact, block):
    rex, a, b, metric = make()
    x = np.array([[1, 2], [2, 1]]) if block else np.array([1, 2])
    family = call("ACCESS_TYPES", param("x"), param("f"), exact)
    pair = call("CO_RELATE", call("ACCESS", param("x"), param("a"), exact),
                call("ACCESS", param("x"), param("b"), exact), param("m"), exact)
    result = run(rex, family, pair, x=co(rex, x), a=a, b=b, f=AccessionFamily((a, b)), m=metric)
    expected = np.sum((np.array([[1, 2]]) @ x) * (
        np.array([[Q(2), Q(-3), Q(1, 2)]]) @ (np.array([[1, 0], [0, -1], [1, 1]]) @ x)))
    assert result.values[1] == expected == (Q(163, 2) if block else Q(95, 2))
    assert result.exactness[1].value == ("rational" if exact else "approximate")
    family_desc = result.provenance[0]["result_type"]
    assert family_desc["shape"] == ([2, None, 2] if block else [2, None])
    assert family_desc["member_shapes"] == ([[1, 2], [3, 2]] if block else [[1], [3]])
    assert family_desc["accessions"][0]["coordinates"] == {"name": "summary-space", "keys": ["total"]}
    assert family_desc["accessions"][1]["shape"] == [3, 2]
    pair_desc = result.provenance[1]["result_type"]
    assert pair_desc["metric"] is None  # not a positive cell metric
    cross = pair_desc["cross_metric"]
    assert cross["shape"] == [1, 3] and cross["construction"] == "sparse-cross-pairing"
    assert cross["coefficient_digest"] == metric.coefficient_digest
    assert cross["left"]["coordinates"]["name"] == "summary-space"
    assert cross["right"]["coordinates"]["name"] == "detail-space"
    assert "entries" not in json.dumps(cross)
    method = "rational-sparse-cross-contraction" if exact else "csr-sesquilinear-cross-contraction"
    node = next(n for n in result.native_plan["nodes"] if n.get("operator") == "CO_RELATE")
    assert node["physical"]["method"] == method
    assert any(p["name"] == "cross_metric_positivity" and p["status"] == "not-asserted" for p in node["predicates"])
    assert result.execution[-1]["methods"][0]["method"] == method


def test_text_query_supplies_explicit_cross_metric_not_ambient_identity():
    rex, a, b, metric = make()
    result = Executor(sources={"r": rex}, params={"x": co(rex, [1, 2]), "a": a, "b": b, "m": metric}).execute(
        parse("FROM $r RETURN CO_RELATE(ACCESS($x,$a,true), ACCESS($x,$b,true), $m, true)"))
    assert result.values == (Q(95, 2),)


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("mismatch", ["space-name", "space-order", "source", "grade", "basis", "variance", "block",
    "float-block", "float-field", "omitted", "ambient", "reversed", "time"])
def test_cross_mismatches_refuse_before_any_adapter(explain, mismatch, monkeypatch):
    import rcql.executor
    rex, a, b, metric = make()
    u, v = a.apply(co(rex, [1, 2]), exact=True), b.apply(co(rex, [1, 2]), exact=True)
    if mismatch == "space-name":
        other = replace(a, coordinates=CoordinateSpace("wrong", a.coordinates.keys))
        u = other.apply(co(rex, [1, 2]), exact=True)
    elif mismatch == "space-order":
        other = replace(b, coordinates=CoordinateSpace(b.coordinates.name, b.coordinates.keys[::-1]))
        v = other.apply(co(rex, [1, 2]), exact=True)
    elif mismatch == "source":
        metric = make()[3]
    elif mismatch == "grade":
        metric = CrossMetric(replace(a, grade=0), replace(b, grade=0), metric.entries)
    elif mismatch == "basis":
        metric = CrossMetric(replace(a, cell_keys=("b", "a")), replace(b, cell_keys=("b", "a")), metric.entries)
    elif mismatch == "variance":
        u = a.apply(Chain(1, np.array([1, 2]), source=rex), exact=True)
    elif mismatch == "block":
        u = a.apply(co(rex, [[1], [2]]), exact=True)
    elif mismatch == "float-block":
        metric = CrossMetric(a, b, ((0, 0, 0.),))
    elif mismatch == "float-field":
        u = a.apply(co(rex, [1, 2]))
    elif mismatch == "omitted":
        metric = None
    elif mismatch == "ambient":
        metric = diagonal_metric(rex, 1)
    elif mismatch == "reversed":
        metric = metric.transpose()
    else:
        binding = bind("r", rex, SourcePolicy.allow("*"))
        metric = replace(_carrier_literal(binding, metric), temporal=TemporalRef(version=2))
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, call("CO_RELATE", param("u"), param("v"), param("m"), True), explain=explain, u=u, v=v, m=metric)


def test_named_same_length_coordinates_do_not_acquire_cell_identity(monkeypatch):
    import rcql.executor
    rex, a, _, _ = make()
    a = replace(a, coordinates=CoordinateSpace("not-cells", ("a", "b")))
    u = a.apply(co(rex, [1, 2]), exact=True)
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises(TypeError, match="explicit CrossMetric"):
        run(rex, call("CO_RELATE", param("u"), param("u"), None, True), u=u)
    for expr in (call("BOUNDARY", 1, param("u")), call("APPLY", call("HODGE_OPERATOR", 1), param("u")),
                 call("ACCESS", param("u"), param("a"))):
        with pytest.raises(TypeError):
            run(rex, expr, u=u, a=a)


@pytest.mark.parametrize("external", [False, True])
def test_rectangular_tensor_requires_coherent_family_metric_before_adapters(external, monkeypatch):
    import rcql.executor
    rex, a, b, _ = make()
    family = AccessionFamily((a, b))
    expr = param("f") if external else call("ACCESS_TYPES", param("x"), param("f"), True)
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises(TypeError, match="coherent family form"):
        run(rex, call("MOMENT_TENSOR", expr, None, True), explain=True,
            x=co(rex, [1, 2]), f=family.apply(co(rex, [1, 2]), exact=True) if external else family)


@pytest.mark.parametrize("variance", ["chain", "cochain"])
def test_external_heterogeneous_families_preserve_variance_and_member_axes(variance):
    rex, a, b, _ = make()
    x = co(rex, [1, 2]) if variance == "cochain" else Chain(1, np.array([1, 2]), source=rex)
    family = AccessionFamily((a, b)).apply(x, exact=True)
    result = run(rex, param("f"), f=family)
    desc = result.provenance[0]["result_type"]
    assert desc["variance"] == variance
    assert desc["member_shapes"] == [[1], [3]]
    assert desc["shape"] == [2, None]
    assert result.values[0].names == ("summary", "detail")


def test_cross_block_reuse_is_query_local_and_inputs_remain_distinct(monkeypatch):
    rex, a, b, metric = make()
    u, v = a.apply(co(rex, [1, 2]), exact=True), b.apply(co(rex, [1, 2]), exact=True)
    observed, original = [], CrossMetric.moment
    def measured(self, *args, **kw):
        observed.append(self.coefficient_digest)
        return original(self, *args, **kw)
    monkeypatch.setattr(CrossMetric, "moment", measured)
    expr = call("CO_RELATE", param("u"), param("v"), param("m"), True)
    other = call("CO_RELATE", param("u"), param("v"), param("other"), True)
    result = run(rex, expr, expr, other, u=u, v=v, m=metric, other=CrossMetric(a, b, ((0, 0, -1),)))
    assert result.values == (Q(95, 2), Q(95, 2), Q(-5))
    assert len(observed) == 2 and observed[0] != observed[1]


def test_complex_cross_pairing_and_negative_self_pairing_keep_their_meaning():
    rex, a, b, metric = make()
    u, v = a.apply(co(rex, [1j, 1])), b.apply(co(rex, [1, 2j]))
    result = run(rex, call("CO_RELATE", param("u"), param("v"), param("m")),
                 call("CO_RELATE", param("v"), param("u"), param("mt")), u=u, v=v, m=metric, mt=metric.transpose())
    assert result.values == (12+11.5j, 12-11.5j)
    assert all(p["result_type"]["domain"] == "complex" for p in result.provenance)
    u = a.apply(co(rex, [1, 2]), exact=True)
    result = run(rex, call("CO_RELATE", param("u"), param("u"), param("m"), True),
                 u=u, m=CrossMetric(a, a, ((0, 0, -1),)))
    assert result.values[0] == Q(-25)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("empty", ["coordinates", "block", "source"])
def test_empty_rectangular_pipeline_retains_structural_axes_and_arithmetic(exact, empty):
    rex, a, b, metric = make()
    x = co(rex, [1, 2])
    if empty == "coordinates":
        a = replace(a, coordinates=CoordinateSpace("empty", ()), entries=())
        metric = CrossMetric(a, b, ())
    elif empty == "block":
        x = co(rex, np.empty((2, 0), dtype=int))
    else:
        rex = RexGraph.from_graph([], [])
        a, b = replace(a, source=rex, entries=()), replace(b, source=rex, entries=())
        metric, x = CrossMetric(a, b, ()), co(rex, np.empty(0, dtype=int))
    result = run(rex, call("CO_RELATE", call("ACCESS", param("x"), param("a"), exact),
        call("ACCESS", param("x"), param("b"), exact), param("m"), exact), x=x, a=a, b=b, m=metric)
    assert result.values[0] == 0
    assert result.exactness[0].value == ("rational" if exact else "approximate")
    assert result.provenance[0]["result_type"]["cross_metric"]["shape"] == list(metric.shape)


@pytest.mark.parametrize("defect", ["population", "block-shape", "member-shape"])
def test_stale_or_forged_structural_axes_refuse_before_execution(defect, monkeypatch):
    import rcql.executor
    rex, a, b, metric = make()
    family = AccessionFamily((a, b)).apply(co(rex, [1, 2]), exact=True)
    u, v = family.views
    expr = call("CO_RELATE", param("u"), param("v"), param("m"), True)
    if defect == "population":
        rex.add_edges(np.array([2], np.int32), np.array([0], np.int32))
    else:
        binding = bind("r", rex, SourcePolicy.allow("*"))
        if defect == "block-shape":
            metric = replace(_carrier_literal(binding, metric), shape=ShapeRef((3, 1)))
        else:
            # Ambient tensor construction is supported, so this must reach the
            # member axis checker rather than the coordinate tensor refusal.
            aa = replace(a, coordinates=None)
            bb = replace(b, coordinates=None, entries=((1, 1, 1),))
            family = AccessionFamily((aa, bb)).apply(co(rex, [1, 2]), exact=True)
            family = replace(_carrier_literal(binding, family), member_shapes=((3,), (1,)))
            # Reject the malformed input before the consumer can run.
            expr = call("MOMENT_TENSOR", param("f"), None, True)
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, expr, explain=True, u=u, v=v, m=metric, f=family)


def test_cross_metric_does_not_bypass_read_capabilities(monkeypatch):
    import rcql.executor
    rex, a, b, metric = make()
    u, v = a.apply(co(rex, [1, 2]), exact=True), b.apply(co(rex, [1, 2]), exact=True)
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises(PermissionError):
        Executor(sources={"r": BoundSource(rex, SourcePolicy.allow())}, params={"u": u, "v": v, "m": metric}).execute(
            query(source("r"), call("CO_RELATE", param("u"), param("v"), param("m"), True)))
