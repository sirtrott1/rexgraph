"""Sparse action composition, exact certificates and finite variation identities."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.cochain import Chain, Cochain
from rexgraph.graph import RexGraph
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.linear_operator import RexOperator, boundary_operator
from rexgraph.weighted_hodge import weighted_hodge

from rcql import Executor, call, query, source
from rcql.calculus_operators import ADAPTERS, chain, dependence, strain


@pytest.fixture
def rex():
    return RexGraph.from_simplicial(np.array([0, 1, 0]), np.array([1, 2, 2]), np.array([[0, 1, 2]]))


def run(rex, name, *args, explain=False):
    return Executor(sources={"r": rex}).execute(replace(query(source("r"), call(name, *args)), explain=explain)).values[0]


def test_composition_is_ordered_exact_and_factored(rex, monkeypatch):
    b = boundary_operator(rex, 1)
    from rexgraph.linear_operator import metric_adjoint
    bt = metric_adjoint(b)
    chain_action = run(rex, "CHAIN", [b, bt])
    monkeypatch.setattr(RexOperator, "as_native", lambda *a: pytest.fail("composed matrix requested"))
    x = Chain(1, np.array([Q(1, 3), 2**100, -2], dtype=object), source=rex)
    actual = run(rex, "TRANSFER", chain_action, x)
    expected = bt.apply(b.apply(x.values, exact=True), exact=True)
    np.testing.assert_array_equal(actual.values, expected)
    np.testing.assert_array_equal(run(rex, "APPLY", chain_action, x, True).values, expected)
    np.testing.assert_array_equal(chain_action.transpose_apply(x.values, exact=True), expected)
    assert run(rex, "TRANSFER", chain_action, x, x) == sum(a*b for a, b in zip(x.values, expected, strict=True))
    info = run(rex, "TRANSFER", chain_action, x, explain=True)["returns"][0]["result"]
    assert info["grade"] == 1 and info["variance"] == "chain" and info["exactness"] == "rational"


@pytest.mark.parametrize("explain", [False, True])
def test_composition_rejects_space_and_variance_confusion(rex, explain):
    foreign = RexGraph(sources=np.array([0]), targets=np.array([1]))
    b = boundary_operator(rex, 1)
    for factors in ([], [b, b], [b, boundary_operator(foreign, 1)]):
        with pytest.raises((ValueError, TypeError)):
            run(rex, "CHAIN", factors, explain=explain)
    action = chain(rex, [b])
    for x in (Cochain(1, np.array([1, 2, 3]), source=rex), Chain(1, np.array([1, 2, 3]), source=rex, cell_keys=(2, 1, 0))):
        with pytest.raises((ValueError, TypeError)):
            run(rex, "TRANSFER", action, x, explain=explain)


def test_rectangular_transfer_preserves_grade_and_blocks(rex):
    x = Chain(1, np.array([[1, 2], [3, 4], [5, 6]]), source=rex)
    action = chain(rex, [boundary_operator(rex, 1)])
    result = run(rex, "TRANSFER", action, x)
    assert isinstance(result, Chain) and result.grade == 0 and result.values.shape == (3, 2)
    np.testing.assert_array_equal(result.values, action.apply(x.values, exact=True))


def test_dependence_returns_exact_family_coordinates(rex):
    x = Cochain(1, np.array([Q(1, 3), 2**100, 0], dtype=object), source=rex)
    y = Cochain(1, 3*x.values, source=rex)
    zero = Cochain(1, np.zeros(3, dtype=int), source=rex)
    result = run(rex, "DEPENDENCE", [x, y, zero])
    assert result["rank"] == 1 and result["nullity"] == 2 and result["dependent"]
    for dependency in result["kernel"]:
        assert all(sum(coefficient * [x, y, zero][index].values[row]
                       for index, coefficient in dependency) == 0 for row in range(3))
    assert dependence(rex, []) == {"rank": 0, "nullity": 0, "kernel": (),
                                   "coordinates": "ordered-input-family", "dependent": False}


def test_strain_uses_the_full_boundary_product(rex):
    weight = DiagonalMetric(rex, 1, (1, 2, 4))
    action = run(rex, "STRAIN", 1, weight)
    b1, b2 = boundary_operator(rex, 1), boundary_operator(rex, 2)
    x = Chain(2, np.array([Q(2, 3)], dtype=object), source=rex)
    expected = b1.apply(np.array(weight.weights, dtype=object) * b2.apply(x.values, exact=True), exact=True)
    actual = run(rex, "APPLY", action, x, True)
    assert actual.grade == 0 and action.shape == (3, 1)
    np.testing.assert_array_equal(actual.values, expected)
    assert np.any(expected != 0)
    identity = strain(rex, 1, DiagonalMetric(rex, 1, (1, 1, 1)))
    assert all(v == 0 for v in identity.apply(x.values, exact=True))
    # Homogeneity and the chain law give B W B = B (W-I) B.
    unweighted = b1.apply(b2.apply(x.values, exact=True), exact=True)
    np.testing.assert_array_equal(expected, expected - unweighted)


def test_strain_missing_upper_grade_is_an_exact_rectangular_zero():
    rex = RexGraph.from_hypergraph(np.array([0, 4]), np.array([0, 1, 2, 3]))
    action = run(rex, "STRAIN", 1, DiagonalMetric(rex, 1, (1,)))
    x = Chain(2, np.empty(0, dtype=object), source=rex)
    result = run(rex, "APPLY", action, x, True)
    assert action.shape == (4, 0) and result.grade == 0
    assert result.values.tolist() == [Q(0)]*4


@pytest.mark.parametrize("explain", [False, True])
def test_strain_refuses_wrong_grade_or_named_basis(rex, explain):
    for grade, weight in ((0, DiagonalMetric(rex, 0, (1, 1, 1))),
                          (2, DiagonalMetric(rex, 1, (1, 1, 1))),
                          (1, DiagonalMetric(rex, 1, (1, 1, 1), (2, 1, 0)))):
        with pytest.raises((ValueError, TypeError)):
            run(rex, "STRAIN", grade, weight, explain=explain)


def test_metric_homotopy_is_exact_positive_and_preserves_endpoints(rex):
    a = DiagonalMetric(rex, 1, (Q(1, 3), 2, 2**100))
    b = DiagonalMetric(rex, 1, (3, 4, 1))
    for parameter in (0, 1, Q(2, 3)):
        result = run(rex, "METRIC_HOMOTOPY", a, b, parameter)
        assert result.exact and result.weights == tuple((1-parameter)*x + parameter*y for x, y in zip(a.weights, b.weights, strict=True))
        field = Chain(1, np.array([1, 2, 3]), source=rex)
        expression = call("MOMENT", field, field, call("METRIC_HOMOTOPY", a, b, parameter), True)
        actual = Executor(sources={"r": rex}).execute(query(source("r"), expression)).values[0]
        assert actual == sum(w*x*x for w, x in zip(result.weights, field.values, strict=True))


@pytest.mark.parametrize("parameter", [-1, 2, True, float("nan"), float("inf")])
@pytest.mark.parametrize("explain", [False, True])
def test_metric_homotopy_refuses_nonconvex_parameters(rex, parameter, explain):
    metric = DiagonalMetric(rex, 1, (1, 1, 1))
    with pytest.raises((ValueError, TypeError)):
        run(rex, "METRIC_HOMOTOPY", metric, metric, parameter, explain=explain)


@pytest.mark.parametrize("weighted", [False, True])
def test_action_and_variation_exact_finite_identity(rex, weighted):
    metric = DiagonalMetric(rex, 1, (Q(1, 3), 2, 3)) if weighted else None
    operator = weighted_hodge(rex, 1, metric=metric)
    x = Chain(1, np.array([Q(1, 7), 3, -2], dtype=object), source=rex)
    direction = Chain(1, np.array([2, -1, 3]), source=rex)
    forcing = Chain(1, np.array([1, 2, 3]), source=rex)
    h = Q(2, 7)
    changed = Chain(1, x.values + h*direction.values, source=rex)
    before = run(rex, "ACTION", x, operator, forcing)
    after = run(rex, "ACTION", changed, operator, forcing)
    derivative = run(rex, "VARIATION", operator, x, direction, forcing)
    remainder = run(rex, "ACTION", direction, operator)
    assert after - before == h*derivative + h*h*remainder
    residual = run(rex, "VARIATION", operator, x, None, forcing)
    np.testing.assert_array_equal(residual.values, operator.apply(x.values, exact=True) - forcing.values)
    if not weighted:
        assert run(rex, "DIFFERENTIAL", x, direction) == 2*run(rex, "VARIATION", operator, x, direction)


@pytest.mark.parametrize("explain", [False, True])
def test_actions_reject_nonsymmetric_and_nonreal_inputs(rex, explain):
    value = Chain(1, np.ones(3, dtype=int), source=rex)
    with pytest.raises((ValueError, TypeError)):
        run(rex, "ACTION", value, boundary_operator(rex, 1), explain=explain)
    with pytest.raises((ValueError, TypeError)):
        run(rex, "ACTION", Chain(1, np.ones(3, dtype=complex), source=rex),
            weighted_hodge(rex, 1), None, False, explain=explain)


def test_all_calculus_adapter_names_are_registered():
    from rcql import catalogued
    from rcql.arguments import EXPRESSION_ARGUMENTS
    from rcql.calculus_contracts import ARGUMENTS
    assert set(ADAPTERS) == set(ARGUMENTS)
    assert set(ADAPTERS) <= catalogued()
    assert all(EXPRESSION_ARGUMENTS[k] == v for k, v in ARGUMENTS.items())


def test_cross_metric_has_ordered_rectangular_coordinates(rex):
    from rexgraph.type_accession import CoordinateSpace, TypeAccession
    a = TypeAccession(rex, 1, "a", ((0, 0, 1),), coordinates=CoordinateSpace("a", ("u",)))
    b = TypeAccession(rex, 1, "b", ((0, 1, 1), (1, 2, 1)), coordinates=CoordinateSpace("b", ("v", "w")))
    block = run(rex, "CROSS_METRIC", a, b, [(0, 0, Q(2, 3)), (0, 1, -3)])
    assert block.shape == (1, 2) and block.exact
    field = Cochain(1, np.array([2, 3, 4]), source=rex)
    reading = call("CO_RELATE", call("ACCESS", field, a, True), call("ACCESS", field, b, True),
                   call("CROSS_METRIC", a, b, [(0, 0, Q(2, 3)), (0, 1, -3)]), True)
    assert Executor(sources={"r": rex}).execute(query(source("r"), reading)).values[0] == -20
    for explain in (False, True):
        with pytest.raises((ValueError, TypeError)):
            run(rex, "CROSS_METRIC", b, a, [(0, 1, 1)], explain=explain)


def test_quadratic_variation_rejects_nonfinite_outputs(rex):
    operator = RexOperator("invalid", (3, 3), 1, 1, lambda x: np.full_like(x, np.inf),
                           source=rex, symmetric=True)
    with pytest.raises((ValueError, FloatingPointError)):
        run(rex, "VARIATION", operator, Cochain(1, np.ones(3), source=rex), None, None, False)


def test_dependence_record_members_are_typed(rex):
    from rcql import parse
    result = Executor(sources={"r": rex}).execute(parse(
        'FROM $r LET x=INDICATOR(CELL(1,0)) LET d=DEPENDENCE([x,x]) '
        'RETURN d.rank, d.nullity, d.dependent, d.kernel'))
    assert result.values[:3] == (1, 1, True)
    assert result.values[3] == (((0, Q(-1)), (1, Q(1))),)
