"""Coherent rectangular family forms retain factors and checked native contracts."""
import json
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.type_accession import (
    AccessionFamily,
    CoordinateSpace,
    CrossMetric,
    FamilyMetric,
    TypeAccession,
    TypedFamily,
    TypeRealization,
)

from rcql import BoundSource, Executor, SourcePolicy, bind, call, param, parse, query, source
from rcql.planning import _carrier_literal
from rcql.types import ShapeRef, TemporalRef


def make():
    rex = RexGraph.from_graph([0, 1], [1, 2])
    a = TypeAccession(rex, 1, "summary", ((0, 0, 1), (0, 1, 2)),
                      coordinates=CoordinateSpace("summary-space", ("total",)))
    b = TypeAccession(rex, 1, "detail", ((0, 0, 1), (1, 1, -1), (2, 0, 1), (2, 1, 1)),
                      coordinates=CoordinateSpace("detail-space", ("x", "y", "z")))
    ea = TypeRealization(a, ((0, 0, 1), (1, 0, -1)))
    eb = TypeRealization(b, ((0, 0, 1), (0, 2, Q(1, 2)), (1, 1, 2), (1, 2, -1)))
    return rex, AccessionFamily((a, b)), FamilyMetric((ea, eb), DiagonalMetric(rex, 1, (2, 3)))


def co(rex, values):
    return Cochain(1, np.asarray(values), source=rex)


def run(rex, *exprs, explain=False, **params):
    return Executor(sources={"r": rex}, params=params).execute(query(source("r"), *exprs, explain=explain))


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("block", [False, True])
def test_nested_rectangular_tensor_preserves_factors_and_type_axes(exact, block):
    rex, maps, metric = make()
    x = np.array([[1, 2], [2, 1]]) if block else np.array([1, 2])
    a = np.array([[1], [-1]]) @ np.array([[1, 2]]) @ x
    b = np.array([[1, 0, Q(1, 2)], [0, 2, -1]]) @ np.array([[1, 0], [0, -1], [1, 1]]) @ x
    expected = [[np.sum(u*(np.diag([2, 3]) @ v)) for v in (a, b)] for u in (a, b)]
    expr = call("MOMENT_TENSOR", call("ACCESS_TYPES", param("x"), param("maps"), exact), param("m"), exact)
    result = run(rex, expr, x=co(rex, x), maps=maps, m=metric)
    assert result.values[0].values.tolist() == expected
    assert result.values[0].names == ("summary", "detail")
    assert result.exactness[0].value == ("rational" if exact else "approximate")
    desc = result.provenance[0]["result_type"]
    assert desc["shape"] == [2, 2] and desc["metric"] is None and desc["cross_metric"] is None
    form = desc["family_metric"]
    assert form["shape"] == [4, 4] and form["metric"]["shape"] == [2, 2]
    assert form["psd"] is True and form["positive_definite"] is None
    assert form["coefficient_digest"] == metric.coefficient_digest
    assert [e["shape"] for e in form["realizations"]] == [[2, 1], [2, 3]]
    assert all(e["injective"] is None and e["chain_preserving"] is None for e in form["realizations"])
    assert form["realizations"][1]["accession"]["coordinates"]["keys"] == ["x", "y", "z"]
    assert "entries" not in json.dumps(form)
    method = "rational-factored-family-contraction" if exact else "csr-hermitian-family-contraction"
    node = next(n for n in result.native_plan["nodes"] if n.get("operator") == "MOMENT_TENSOR")
    assert node["physical"]["method"] == method
    assert any(p["name"] == "family_metric_factorization" and p["status"] == "verified" for p in node["predicates"])
    assert any(p["name"] == "realization_injectivity" and p["status"] == "not-asserted" for p in node["predicates"])
    executed = result.execution[-1]["methods"][0]
    assert executed["method"] == method and executed["realization_actions"] == 2


def test_text_query_and_external_form_handle_are_executable():
    rex, maps, metric = make()
    result = Executor(sources={"r": rex}, params={"x": co(rex, [1, 2]), "maps": maps, "m": metric}).execute(
        parse("FROM $r RETURN MOMENT_TENSOR(ACCESS_TYPES($x,$maps,true),$m,true), $m"))
    assert result.values[0].values.tolist() == [[125, 130], [130, Q(319, 2)]]
    assert result.values[1] is metric and result.exactness[1].value == "structural"
    assert result.provenance[1]["result_type"]["kind"] == "FamilyMetric"


@pytest.mark.parametrize("selection", ["reverse", "subset", "new-measurement"])
def test_external_families_use_explicit_type_lookup_not_position(selection):
    rex, maps, metric = make()
    family = maps.apply(co(rex, [1, 2]), exact=True)
    if selection == "reverse":
        family = TypedFamily(family.views[::-1])
        expected = [[Q(319, 2), 130], [130, 125]]
    elif selection == "subset":
        family = TypedFamily((family.views[1],))
        expected = [[Q(319, 2)]]
    else:
        a = replace(maps.accessions[0], entries=((0, 0, -1),))
        family = TypedFamily((a.apply(co(rex, [1, 2]), exact=True), family.views[1]))
        expected = [[5, -26], [-26, Q(319, 2)]]
    result = run(rex, call("MOMENT_TENSOR", param("f"), param("m"), True), f=family, m=metric)
    assert result.values[0].values.tolist() == expected
    assert [a["name"] for a in result.provenance[0]["result_type"]["accessions"]] == list(family.names)


def test_family_co_relate_is_the_same_signed_hermitian_entry_as_tensor():
    rex, maps, metric = make()
    u = maps.accessions[0].apply(co(rex, [1j, 1]))
    v = maps.accessions[1].apply(co(rex, [1, 2j]))
    result = run(rex, call("CO_RELATE", param("u"), param("v"), param("m")),
                 call("CO_RELATE", param("v"), param("u"), param("m")),
                 call("MOMENT_TENSOR", param("f"), param("m")), u=u, v=v, f=TypedFamily((u, v)), m=metric)
    assert result.values[:2] == (32+34j, 32-34j)
    assert result.values[2].values.tolist() == [[25, 32+34j], [32-34j, 117.5]]
    assert all(p["result_type"]["domain"] == "complex" for p in result.provenance)
    assert result.provenance[0]["result_type"]["cross_metric"] is None


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("mismatch", ["foreign", "grade", "basis", "name", "coordinates", "order", "float-factor", "float-base", "float-field", "time"])
def test_family_mismatches_fail_before_any_adapter(explain, mismatch, monkeypatch):
    import rcql.executor
    rex, maps, metric = make()
    family = maps.apply(co(rex, [1, 2]), exact=True)
    if mismatch == "foreign":
        metric = make()[2]
    elif mismatch in {"grade", "basis"}:
        changes = {"grade": 0} if mismatch == "grade" else {"cell_keys": ("b", "a")}
        factors = tuple(TypeRealization(replace(e.accession, **changes), e.entries) for e in metric.realizations)
        metric = FamilyMetric(factors, DiagonalMetric(rex, 0, (1, 2, 3)) if mismatch == "grade" else DiagonalMetric(rex, 1, (2, 3), ("b", "a")))
    elif mismatch in {"name", "coordinates", "order"}:
        a, b = maps.accessions
        if mismatch == "name":
            a = replace(a, name="missing")
        elif mismatch == "coordinates":
            a = replace(a, coordinates=CoordinateSpace("other", a.coordinates.keys))
        else:
            b = replace(b, coordinates=CoordinateSpace(b.coordinates.name, b.coordinates.keys[::-1]))
        family = AccessionFamily((a, b)).apply(co(rex, [1, 2]), exact=True)
    elif mismatch == "float-factor":
        e = TypeRealization(maps.accessions[0], ((0, 0, 1.), (0, 0, -1.)))
        metric = FamilyMetric((e, metric.realizations[1]), metric.metric)
    elif mismatch == "float-base":
        metric = FamilyMetric(metric.realizations, DiagonalMetric(rex, 1, (2., 3.)))
    elif mismatch == "float-field":
        family = maps.apply(co(rex, [1, 2]))
    else:
        binding = bind("r", rex, SourcePolicy.allow("*"))
        metric = replace(_carrier_literal(binding, metric), temporal=TemporalRef(version=2))
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, call("MOMENT_TENSOR", param("f"), param("m"), True), explain=explain, f=family, m=metric)


@pytest.mark.parametrize("mismatch", ["variance", "block", "type", "coordinates"])
def test_family_cross_pairing_checks_each_endpoint_before_adapters(mismatch, monkeypatch):
    import rcql.executor
    rex, maps, metric = make()
    u, v = maps.apply(co(rex, [1, 2]), exact=True).views
    if mismatch == "variance":
        u = maps.accessions[0].apply(Chain(1, np.array([1, 2]), source=rex), exact=True)
    elif mismatch == "block":
        u = maps.accessions[0].apply(co(rex, [[1], [2]]), exact=True)
    elif mismatch == "type":
        u = replace(maps.accessions[0], name="other").apply(co(rex, [1, 2]), exact=True)
    else:
        a = replace(maps.accessions[0], coordinates=CoordinateSpace("other", ("x",)))
        u = a.apply(co(rex, [1, 2]), exact=True)
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, call("CO_RELATE", param("u"), param("v"), param("m"), True), explain=True, u=u, v=v, m=metric)


@pytest.mark.parametrize("defect", ["factor-shape", "base-shape", "form-shape", "duplicate-type", "positivity", "population"])
def test_structural_family_predicates_do_not_execute_realizations(defect, monkeypatch):
    import rcql.executor
    rex, maps, metric = make()
    family = maps.apply(co(rex, [1, 2]), exact=True)
    if defect == "population":
        rex.add_edges(np.array([2], np.int32), np.array([0], np.int32))
    else:
        binding = bind("r", rex, SourcePolicy.allow("*"))
        metric = _carrier_literal(binding, metric)
        desc = metric.family_metric
        if defect == "factor-shape":
            desc = replace(desc, realizations=(replace(desc.realizations[0], shape=(1, 2)), desc.realizations[1]))
        elif defect == "base-shape":
            desc = replace(desc, metric=replace(desc.metric, shape=(1, 1)))
        elif defect == "form-shape":
            metric = replace(metric, shape=ShapeRef((2, 2)))
        elif defect == "duplicate-type":
            desc = replace(desc, realizations=(desc.realizations[0], desc.realizations[0]))
        else:
            desc = replace(desc, metric=replace(desc.metric, positive_definite=None))
        metric = replace(metric, family_metric=desc)
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    monkeypatch.setattr(TypeRealization, "apply", lambda *a, **kw: pytest.fail("realization ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, call("MOMENT_TENSOR", param("f"), param("m"), True), explain=True, f=family, m=metric)


def test_explain_is_nonexecuting_and_repeated_tensor_reuses_realization_actions(monkeypatch):
    rex, maps, metric = make()
    expr = call("MOMENT_TENSOR", call("ACCESS_TYPES", param("x"), param("maps"), True), param("m"), True)
    actions, original = [], TypeRealization.apply
    def measured(self, *args, **kw):
        actions.append(self.accession.name)
        return original(self, *args, **kw)
    monkeypatch.setattr(TypeRealization, "apply", measured)
    run(rex, expr, explain=True, x=co(rex, [1, 2]), maps=maps, m=metric)
    assert actions == []
    result = run(rex, expr, expr, x=co(rex, [1, 2]), maps=maps, m=metric)
    assert actions == ["summary", "detail"] and result.values[0] is result.values[1]


def test_family_metric_is_not_an_independent_cross_block_or_ordinary_cell_metric(monkeypatch):
    import rcql.executor
    rex, maps, metric = make()
    family = maps.apply(co(rex, [1, 2]), exact=True)
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises(TypeError):
        run(rex, call("MOMENT_TENSOR", param("f"), param("m"), True), f=family, m=CrossMetric(*maps.accessions, ()))
    with pytest.raises(TypeError):
        run(rex, call("MOMENT", param("x"), param("x"), param("m"), True), x=co(rex, [1, 2]), m=metric)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("empty", ["source", "block", "coordinates"])
def test_empty_factored_family_pipeline_retains_type_axes_and_exact_zero(exact, empty):
    rex, maps, metric = make()
    x = co(rex, [1, 2])
    if empty == "source":
        rex = RexGraph.from_graph([], [])
        maps = AccessionFamily(tuple(replace(a, source=rex, entries=()) for a in maps.accessions))
        metric = FamilyMetric(tuple(TypeRealization(a, ()) for a in maps.accessions), DiagonalMetric(rex, 1, ()))
        x = co(rex, np.empty(0, dtype=int))
    elif empty == "block":
        x = co(rex, np.empty((2, 0), dtype=int))
    else:
        maps = AccessionFamily(tuple(replace(a, coordinates=CoordinateSpace(a.name, ()), entries=()) for a in maps.accessions))
        metric = FamilyMetric(tuple(TypeRealization(a, ()) for a in maps.accessions), metric.metric)
    result = run(rex, call("MOMENT_TENSOR", call("ACCESS_TYPES", param("x"), param("maps"), exact), param("m"), exact),
                 x=x, maps=maps, m=metric)
    assert result.values[0].values.tolist() == [[0, 0], [0, 0]]
    assert result.exactness[0].value == ("rational" if exact else "approximate")
    assert result.provenance[0]["result_type"]["shape"] == [2, 2]


def test_family_metric_obeys_read_policy_before_adapters(monkeypatch):
    import rcql.executor
    rex, maps, metric = make()
    family = maps.apply(co(rex, [1, 2]), exact=True)
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises(PermissionError):
        Executor(sources={"r": BoundSource(rex, SourcePolicy.allow())}, params={"f": family, "m": metric}).execute(
            query(source("r"), call("MOMENT_TENSOR", param("f"), param("m"), True)))
