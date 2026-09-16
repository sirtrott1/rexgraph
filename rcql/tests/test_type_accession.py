"""Type and cell axes stay distinct through exact/native query execution."""
import json
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
import scipy.sparse as sp
from rexgraph.cochain import Chain, Cochain, Field
from rexgraph.graded_metric import diagonal_metric
from rexgraph.graph import RexGraph
from rexgraph.type_accession import AccessionFamily, TypeAccession, TypedFamily, co_relate

from rcql import BoundSource, Executor, SourcePolicy, call, param, parse, query, source
from rcql.planning import _carrier_literal
from rcql.types import TemporalRef


def make():
    return RexGraph.from_graph([0, 1], [1, 2])


def co(rex, values, **kw):
    return Cochain(1, np.asarray(values), source=rex, **kw)


def maps(rex, **kw):
    return AccessionFamily((TypeAccession(rex, 1, "mixed", ((0, 0, 1), (0, 1, 1), (1, 1, 1)), **kw),
                            TypeAccession(rex, 1, "shared", ((0, 0, 1), (1, 0, -1)), **kw)))


def run(rex, *exprs, explain=False, **params):
    return Executor(sources={"r": rex}, params=params).execute(query(source("r"), *exprs, explain=explain))


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("block", [False, True])
def test_type_family_tensor_matches_independent_common_metric_oracle(exact, block, monkeypatch):
    rex = make()
    p, q = np.array([[1, 1], [0, 1]]), np.array([[1, 0], [-1, 0]])
    x = np.array([[1, 2], [2, -1]]) if block else np.array([1, 2])
    fields = [p @ x, q @ x]
    m = np.diag([2, 3])
    expected = [[np.sum(u * (m @ v)) for v in fields] for u in fields]
    family = maps(rex)
    def forbidden(*a, **kw):
        pytest.fail("type contraction materialized ambient dense/eigen matrix")
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    tensor = call("MOMENT_TENSOR", call("ACCESS_TYPES", param("x"), param("maps"), exact),
                  call("METRIC", 1, param("w")), exact)
    result = run(rex, tensor, x=co(rex, x), maps=family, w=co(rex, [2, 3]))
    assert result.values[0].values.tolist() == expected
    assert result.values[0].names == ("mixed", "shared")
    assert result.exactness[0].value == ("rational" if exact else "approximate")
    desc = result.provenance[0]["result_type"]
    assert desc["kind"] == "TypedMomentTensor"
    assert desc["shape"] == [2, 2]
    assert [a["name"] for a in desc["accessions"]] == ["mixed", "shared"]
    assert all(a["chain_preserving"] is None for a in desc["accessions"])
    access_node = next(n for n in result.native_plan["nodes"] if n.get("operator") == "ACCESS_TYPES")
    assert access_node["result"]["shape"] == ([2, 2, 2] if block else [2, 2])
    assert access_node["physical"]["method"] == ("rational-sparse-accession" if exact else "csr-accession")
    methods = result.execution[-1]["methods"]
    assert methods[0]["output_axes"] == "type-by-type"


def test_text_query_uses_declared_map_parameters_and_signed_cross_moment():
    rex = make()
    family = maps(rex)
    executor = Executor(sources={"r": rex}, params={"x": co(rex, [1, 2]),
        "p": family.accessions[0], "q": family.accessions[1], "w": co(rex, [1, 4])})
    result = executor.execute(parse(
        "FROM $r RETURN CO_RELATE(ACCESS($x, $p, true), ACCESS($x, $q, true), METRIC(1, $w), true)"))
    assert result.values[0] == Q(-5)
    assert result.provenance[0]["result_type"]["kind"] == "ExactRational"


def test_external_views_and_families_retain_arithmetic_and_type_order():
    rex = make()
    family = maps(rex).apply(co(rex, [1, 2]), exact=True)
    result = run(rex, call("CO_RELATE", param("p"), param("q"), None, True),
        call("MOMENT_TENSOR", param("family"), None, True), p=family["mixed"], q=family["shared"], family=family)
    assert result.values[0] == 1
    assert result.values[1].values.tolist() == [[13, 1], [1, 2]]
    assert [e.value for e in result.exactness] == ["rational", "rational"]
    desc = result.provenance[0]["result_type"]["accessions"]
    assert desc[0]["coefficient_digest"] != desc[1]["coefficient_digest"]
    assert "entries" not in json.dumps(desc)


def test_complex_cross_moment_is_not_squared_or_abs_value():
    rex = make()
    a = TypeAccession(rex, 1, "a", ((0, 0, 1), (1, 1, 1)))
    b = TypeAccession(rex, 1, "b", a.entries)
    u, v = a.apply(co(rex, [1j, 1])), b.apply(co(rex, [1, 1j]))
    m = diagonal_metric(rex, 1, co(rex, [2, 3]))
    result = run(rex, call("CO_RELATE", param("u"), param("v"), param("m")),
        call("MOMENT_TENSOR", param("family"), param("m")), u=u, v=v, family=TypedFamily((u, v)), m=m)
    assert result.values[0] == 1j
    assert result.values[1].values.tolist() == [[5, 1j], [-1j, 5]]
    assert result.provenance[0]["result_type"]["domain"] == "complex"
    assert result.provenance[1]["result_type"]["domain"] == "complex"


@pytest.mark.parametrize("carrier", ["chain", "cochain", "field"])
def test_access_preserves_variance_without_claiming_chain_preservation(carrier):
    rex = make()
    value = co(rex, [1, 2])
    if carrier == "chain":
        value = Chain(1, value.values, source=rex)
    elif carrier == "field":
        value = Field(value, "external")
    a = maps(rex).accessions[0]
    result = run(rex, call("ACCESS", param("x"), param("p"), True), x=value, p=a)
    desc = result.provenance[0]["result_type"]
    assert desc["variance"] == ("chain" if carrier == "chain" else "cochain")
    assert result.values[0].values.tolist() == [3, 2]
    facts = result.native_plan["nodes"][-1]["predicates"]
    assert next(p for p in facts if p["name"] == "accession_chain_preservation")["status"] == "not-asserted"


def test_metric_and_accessions_preserve_noncanonical_basis():
    rex = make()
    keys = ("second", "first")
    result = run(rex, call("MOMENT_TENSOR", call("ACCESS_TYPES", param("x"), param("maps"), True),
        call("METRIC", 1, param("w")), True), x=co(rex, [1, 2], cell_keys=keys),
        maps=maps(rex, cell_keys=keys), w=co(rex, [2, 3], cell_keys=keys))
    assert result.values[0].values.tolist() == [[30, 0], [0, 5]]
    assert result.provenance[0]["result_type"]["basis"]["ordering"] == list(keys)


@pytest.mark.parametrize("mismatch", ["source", "grade", "basis", "float-map", "float-value", "shape", "name-only"])
@pytest.mark.parametrize("explain", [False, True])
def test_bad_access_inputs_refuse_before_adapters(mismatch, explain, monkeypatch):
    import rcql.executor
    rex = make()
    value, a = co(rex, [1, 2]), maps(rex).accessions[0]
    if mismatch == "source":
        a = maps(make()).accessions[0]
    elif mismatch == "grade":
        a = TypeAccession(rex, 0, "a", ((0, 0, 1),))
    elif mismatch == "basis":
        a = maps(rex, cell_keys=("b", "a")).accessions[0]
    elif mismatch == "float-map":
        a = TypeAccession(rex, 1, "a", ((0, 0, 1.),))
    elif mismatch == "float-value":
        value = co(rex, [1., 2.])
    elif mismatch == "shape":
        value = co(rex, [1, 2, 3])
    else:
        a = "mechanical"
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, call("ACCESS", param("x"), param("p"), True), explain=explain, x=value, p=a)


@pytest.mark.parametrize("mismatch", ["basis", "metric-source", "variance", "shape", "float-exact", "time"])
def test_cross_type_contraction_alignment_is_checked_before_adapters(mismatch, monkeypatch):
    import rcql.executor
    from rcql import bind
    rex = make()
    a, b = maps(rex).accessions
    u, v = a.apply(co(rex, [1, 2]), exact=True), b.apply(co(rex, [1, 2]), exact=True)
    metric = diagonal_metric(rex, 1)
    if mismatch == "basis":
        c = maps(rex, cell_keys=("b", "a")).accessions[1]
        v = c.apply(co(rex, [1, 2], cell_keys=("b", "a")), exact=True)
    elif mismatch == "metric-source":
        metric = diagonal_metric(make(), 1)
    elif mismatch == "variance":
        v = b.apply(Chain(1, np.array([1, 2]), source=rex), exact=True)
    elif mismatch == "shape":
        v = b.apply(co(rex, [[1], [2]]), exact=True)
    elif mismatch == "float-exact":
        v = b.apply(co(rex, [1, 2]))
    else:
        binding = bind("r", rex, SourcePolicy.allow("*"))
        u = replace(_carrier_literal(binding, u), temporal=TemporalRef(version=1))
        v = replace(_carrier_literal(binding, v), temporal=TemporalRef(version=2))
        assert u.with_(temporal=None).same_space(v.with_(temporal=None))
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises((TypeError, ValueError)):
        run(rex, call("CO_RELATE", param("u"), param("v"), param("m"), True), u=u, v=v, m=metric)


def test_accessions_do_not_bypass_read_policy(monkeypatch):
    import rcql.executor
    rex = make()
    a = maps(rex).accessions[0]
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises(PermissionError):
        Executor(sources={"r": BoundSource(rex, SourcePolicy.allow())}, params={"p": a}).execute(
            query(source("r"), call("ACCESS", call("ZERO", 1), param("p"))))


def test_repeated_family_access_is_reused_with_distinct_map_inputs(monkeypatch):
    rex = make()
    observed = []
    original = TypeAccession.apply
    def measured(self, *args, **kw):
        observed.append(self.name)
        return original(self, *args, **kw)
    monkeypatch.setattr(TypeAccession, "apply", measured)
    expr = call("ACCESS_TYPES", param("x"), param("maps"), True)
    result = run(rex, expr, expr, call("MOMENT_TENSOR", expr, None, True), x=co(rex, [1, 2]), maps=maps(rex))
    assert observed == ["mixed", "shared"]
    assert result.values[0] is result.values[1]


def test_type_views_are_not_accepted_as_an_unlabelled_cell_cochain():
    rex = make()
    view = maps(rex).accessions[0].apply(co(rex, [1, 2]), exact=True)
    for expr in (call("BOUNDARY", 1, param("v")), call("APPLY", call("HODGE_OPERATOR", 1), param("v")),
                 call("ACCESS", param("v"), param("p"))):
        with pytest.raises(TypeError):
            run(rex, expr, v=view, p=maps(rex).accessions[0])


def test_exact_cross_reading_agrees_with_direct_core():
    rex = make()
    family = maps(rex).apply(co(rex, [Q(2, 3), Q(5, 7)]), exact=True)
    result = run(rex, call("CO_RELATE", param("a"), param("b"), None, True),
                 a=family["mixed"], b=family["shared"])
    assert result.values[0] == co_relate(family["mixed"], family["shared"], exact=True) == Q(4, 9)


@pytest.mark.parametrize("grade", [0, 1, 2])
@pytest.mark.parametrize("exact", [False, True])
def test_all_present_grades_have_a_separate_type_axis(grade, exact):
    from rexgraph.cells import cell_count
    rex = RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])
    n = cell_count(rex, grade)
    a = TypeAccession(rex, grade, "weighted", tuple((i, i, Q(i+1, 3)) for i in range(n)))
    x = Cochain(grade, np.ones(n, dtype=int), source=rex)
    result = run(rex, call("MOMENT_TENSOR", call("ACCESS_TYPES", param("x"), param("family"), exact), None, exact),
                 x=x, family=AccessionFamily((a,)))
    expected = sum((Q(i+1, 3)**2 for i in range(n)), Q(0))
    assert result.values[0].values.shape == (1, 1)
    assert result.values[0].values[0, 0] == (expected if exact else pytest.approx(float(expected)))
    assert result.provenance[0]["result_type"]["grade"] == grade


@pytest.mark.parametrize("exact", [False, True])
def test_empty_grade_and_empty_block_keep_named_type_quadrances(exact):
    rex = RexGraph.from_graph([], [])
    family = AccessionFamily((TypeAccession(rex, 1, "first", ()), TypeAccession(rex, 1, "second", ())))
    result = run(rex, call("MOMENT_TENSOR", call("ACCESS_TYPES", call("ZERO", 1), param("f"), exact), None, exact), f=family)
    assert result.values[0].values.tolist() == [[0, 0], [0, 0]]
    assert result.exactness[0].value == ("rational" if exact else "approximate")
    rex = make()
    result = run(rex, call("MOMENT_TENSOR", call("ACCESS_TYPES", param("x"), param("f"), exact), None, exact),
                 x=co(rex, np.empty((2, 0), dtype=int)), f=maps(rex))
    assert result.values[0].values.tolist() == [[0, 0], [0, 0]]


def test_population_mutation_is_rejected_during_explain_before_any_action(monkeypatch):
    import rcql.executor
    rex = make()
    family = maps(rex)
    rex.add_edges(np.array([2], np.int32), np.array([0], np.int32))
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter ran"))
    with pytest.raises(ValueError, match="accession axes"):
        run(rex, call("ACCESS_TYPES", call("ZERO", 1), param("f")), explain=True, f=family)


def test_zero_float_map_retains_approximate_source_contract():
    rex = make()
    a = TypeAccession(rex, 1, "cancelled", ((0, 0, 1.), (0, 0, -1.)))
    result = run(rex, call("ACCESS", call("ZERO", 1), param("p")), p=a)
    assert result.exactness[0].value == "approximate"
    assert result.provenance[0]["result_type"]["accessions"][0]["coefficient_domain"] == "real"
    with pytest.raises(TypeError, match="integer/rational"):
        run(rex, call("ACCESS", call("ZERO", 1), param("p"), True), explain=True, p=a)


def test_external_mixed_integer_fraction_family_has_rational_contract():
    from rexgraph.type_accession import TypeView
    rex = make()
    a, b = maps(rex).accessions
    family = TypedFamily((TypeView(a, co(rex, [1, 2])), TypeView(b, co(rex, [Q(1, 2), Q(1, 3)]))))
    result = run(rex, call("MOMENT_TENSOR", param("f"), None, True), f=family)
    assert result.exactness[0].value == "rational"
    assert result.values[0].values.tolist() == [[Q(5), Q(7, 6)], [Q(7, 6), Q(13, 36)]]
