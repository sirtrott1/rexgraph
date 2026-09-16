"""Independent rectangular oracles, ordered cross forms and sparse contracts."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.cochain import Chain, Cochain, Field
from rexgraph.graded_metric import diagonal_metric
from rexgraph.graph import RexGraph
from rexgraph.type_accession import (
    AccessionFamily,
    CoordinateField,
    CoordinateSpace,
    CrossMetric,
    TypeAccession,
    TypedFamily,
    TypeView,
    co_relate,
    moment_tensor,
)


def make():
    rex = RexGraph.from_graph([0, 1], [1, 2])
    a = TypeAccession(rex, 1, "a", ((0, 0, 1), (0, 1, 2)),
                      coordinates=CoordinateSpace("summary", ("total",)))
    b = TypeAccession(rex, 1, "b", ((0, 0, 1), (1, 1, -1), (2, 0, 1), (2, 1, 1)),
                      coordinates=CoordinateSpace("detail", ("first", "second", "joint")))
    metric = CrossMetric(a, b, ((0, 0, 2), (0, 1, -3), (0, 2, Q(1, 2))))
    return rex, a, b, metric


def co(rex, values):
    return Cochain(1, np.asarray(values), source=rex)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("block", [False, True])
def test_rectangular_action_and_cross_pairing_match_independent_oracle(exact, block):
    rex, a, b, metric = make()
    x = np.array([[1, 2], [2, 1]]) if block else np.array([1, 2])
    p, q = np.array([[1, 2]]), np.array([[1, 0], [0, -1], [1, 1]])
    m = np.array([[Q(2), Q(-3), Q(1, 2)]], dtype=object)
    family = AccessionFamily((a, b)).apply(co(rex, x), exact=exact)
    u, v = family["a"], family["b"]
    assert a.shape == (1, 2) and b.shape == (3, 2) and metric.shape == (1, 3)
    assert isinstance(u.carrier, CoordinateField) and not isinstance(u.carrier, (Chain, Cochain))
    np.testing.assert_array_equal(u.values, p @ x)
    np.testing.assert_array_equal(v.values, q @ x)
    expected = np.sum((p @ x)*(m @ (q @ x)))
    assert co_relate(u, v, metric, exact=exact) == expected
    assert co_relate(v, u, metric.transpose(), exact=exact) == expected
    if exact:
        assert isinstance(co_relate(u, v, metric, exact=True), Q)
    with pytest.raises(ValueError, match="endpoint"):
        co_relate(v, u, metric, exact=exact)


def test_complex_fields_use_conjugate_left_and_explicit_transpose():
    rex, a, b, metric = make()
    u = a.apply(co(rex, [1j, 1]))
    v = b.apply(co(rex, [1, 2j]))
    # (2-i) * (5/2 + 7i) = 12 + 23i/2, not a norm.
    assert co_relate(u, v, metric) == 12 + 11.5j
    assert co_relate(v, u, metric.transpose()) == 12 - 11.5j


@pytest.mark.parametrize("kind", ["chain", "cochain", "field"])
def test_coordinate_carrier_retains_input_variance(kind):
    rex, a, _, _ = make()
    x = co(rex, [1, 2])
    if kind == "chain":
        x = Chain(1, x.values, source=rex)
    if kind == "field":
        x = Field(x, "original")
    view = a.apply(x, exact=True)
    assert view.variance == ("chain" if kind == "chain" else "cochain")
    assert view.values.tolist() == [Q(5)]
    assert not view.values.flags.writeable
    with pytest.raises(TypeError, match="Chain or Cochain"):
        a.apply(view.carrier)


@pytest.mark.parametrize("metric_kind", ["omitted", "ambient"])
def test_coordinate_axes_are_never_implicitly_identified_even_at_equal_length(metric_kind):
    rex, _, _, _ = make()
    a = TypeAccession(rex, 1, "a", ((0, 0, 1), (1, 1, 1)),
                      coordinates=CoordinateSpace("not-cells", ("x", "y")))
    u = a.apply(co(rex, [1, 2]), exact=True)
    metric = None if metric_kind == "omitted" else diagonal_metric(rex, 1)
    with pytest.raises(TypeError, match="explicit CrossMetric"):
        co_relate(u, u, metric, exact=True)
    with pytest.raises(TypeError, match="coherent family form"):
        moment_tensor(TypedFamily((u,)), metric, exact=True)


def test_cross_endpoints_match_declared_spaces_not_map_names_or_equal_lengths():
    rex, a, b, metric = make()
    u, v = a.apply(co(rex, [1, 2]), exact=True), b.apply(co(rex, [1, 2]), exact=True)
    other_map = replace(a, name="different map", entries=((0, 0, -1),))
    other_view = other_map.apply(co(rex, [1, 2]), exact=True)
    assert co_relate(other_view, v, metric, exact=True) == Q(-19, 2)
    for space in (CoordinateSpace("other", a.coordinates.keys),
                  CoordinateSpace(a.coordinates.name, ("other-key",))):
        wrong = replace(a, coordinates=space).apply(co(rex, [1, 2]), exact=True)
        with pytest.raises(ValueError, match="endpoint"):
            co_relate(wrong, v, metric, exact=True)
    with pytest.raises(ValueError, match="declared source"):
        TypeView(a, replace(u.carrier, space=CoordinateSpace("wrong", ("total",))))


@pytest.mark.parametrize("mismatch", ["source", "grade", "basis", "variance", "block"])
def test_cross_space_and_carrier_mismatches_refuse(mismatch):
    rex, a, b, metric = make()
    u, v = a.apply(co(rex, [1, 2]), exact=True), b.apply(co(rex, [1, 2]), exact=True)
    if mismatch == "source":
        other, oa, _, _ = make()
        u = oa.apply(co(other, [1, 2]), exact=True)
    elif mismatch == "grade":
        oa = replace(a, grade=0)
        u = oa.apply(Cochain(0, np.array([1, 2, 3]), source=rex), exact=True)
    elif mismatch == "basis":
        oa = replace(a, cell_keys=("second", "first"))
        u = oa.apply(Cochain(1, np.array([1, 2]), source=rex, cell_keys=oa.cell_keys), exact=True)
    elif mismatch == "variance":
        u = a.apply(Chain(1, np.array([1, 2]), source=rex), exact=True)
    else:
        u = a.apply(co(rex, [[1], [2]]), exact=True)
    with pytest.raises(ValueError):
        co_relate(u, v, metric, exact=True)


@pytest.mark.parametrize("entries", [((1, 0, 1),), ((0, 3, 1),), ((False, 0, 1),),
    ((0, .5, 1),), ((0, 0, True),), ((0, 0, 1j),), ((0, 0, float("nan")),),
    ((0, 0, float("inf")),), ((0, 0, 1e308), (0, 0, 1e308))])
def test_invalid_cross_coefficients_and_axes_refuse(entries):
    _, a, b, _ = make()
    with pytest.raises((TypeError, ValueError, OverflowError)):
        CrossMetric(a, b, entries)


@pytest.mark.parametrize("space", [("", ()), ("s", ("a", "a")), ("s", (1,)), ("s", ("",)), ("s", "abc")])
def test_coordinate_space_requires_stable_named_order(space):
    with pytest.raises((TypeError, ValueError)):
        CoordinateSpace(*space)


def test_cross_coefficients_are_copied_coalesced_hashed_and_can_be_negative():
    rex, a, _, _ = make()
    entries = [[0, 0, Q(-1, 3)], [0, 0, Q(-1, 6)]]
    metric = CrossMetric(a, a, entries)
    entries[0][2] = 99
    assert metric.entries == ((0, 0, Q(-1, 2)),)
    assert metric.coefficient_digest == CrossMetric(a, a, ((0, 0, Q(-1, 2)),)).coefficient_digest
    u = a.apply(co(rex, [1, 2]), exact=True)
    assert co_relate(u, u, metric, exact=True) == Q(-25, 2)  # not a quadrance


def test_nonsymmetric_square_cross_block_is_not_silently_symmetrized():
    rex, _, _, _ = make()
    a = TypeAccession(rex, 1, "a", ((0, 0, 1), (1, 1, 1)),
                      coordinates=CoordinateSpace("s", ("x", "y")))
    u, v = a.apply(co(rex, [1, 0]), exact=True), a.apply(co(rex, [0, 1]), exact=True)
    metric = CrossMetric(a, a, ((0, 1, 2), (1, 0, -3)))
    assert co_relate(u, v, metric, exact=True) == 2
    assert co_relate(v, u, metric, exact=True) == -3
    assert co_relate(v, u, metric.transpose(), exact=True) == 2


@pytest.mark.parametrize("weight", [Q(10**400), Q(1, 10**400)])
def test_extreme_exact_cross_weights_never_cast_through_float(weight):
    rex, a, b, _ = make()
    metric = CrossMetric(a, b, ((0, 0, weight),))
    u, v = a.apply(co(rex, [1, 2]), exact=True), b.apply(co(rex, [1, 2]), exact=True)
    assert co_relate(u, v, metric, exact=True) == 5*weight
    with pytest.raises(FloatingPointError, match="representable"):
        co_relate(u, v, metric)


def test_zero_float_cross_block_and_transpose_remain_approximate():
    rex, a, b, _ = make()
    metric = CrossMetric(a, b, ((0, 0, 1.), (0, 0, -1.)))
    u, v = a.apply(co(rex, [1, 2]), exact=True), b.apply(co(rex, [1, 2]), exact=True)
    assert not metric.exact and not metric.transpose().exact
    assert co_relate(u, v, metric) == 0.
    assert metric.transpose().transpose().coefficient_digest == metric.coefficient_digest
    with pytest.raises(TypeError, match="integer/rational cross metric"):
        co_relate(u, v, metric, exact=True)
    zero = CrossMetric(a, b, ())
    with pytest.raises(TypeError, match="integers or Fractions"):
        co_relate(u, b.apply(co(rex, [1, 2])), zero, exact=True)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("empty", ["coordinate", "block", "source"])
def test_empty_outputs_and_blocks_preserve_all_declared_axes(exact, empty):
    rex, a, b, metric = make()
    x = co(rex, [1, 2])
    if empty == "coordinate":
        a = replace(a, entries=(), coordinates=CoordinateSpace("empty", ()))
        metric = CrossMetric(a, b, ())
    elif empty == "block":
        x = co(rex, np.empty((2, 0), dtype=int))
    else:
        rex = RexGraph.from_graph([], [])
        a, b = replace(a, source=rex, entries=()), replace(b, source=rex, entries=())
        metric, x = CrossMetric(a, b, ()), co(rex, np.empty(0, dtype=int))
    u, v = AccessionFamily((a, b)).apply(x, exact=exact).views
    result = co_relate(u, v, metric, exact=exact)
    assert result == 0 and isinstance(result, Q if exact else float)
    assert u.values.shape == (a.shape[0], *x.values.shape[1:])
    assert v.values.shape == (b.shape[0], *x.values.shape[1:])


def test_population_mutation_invalidates_rectangular_maps_views_and_cross_forms():
    rex, a, b, metric = make()
    u, v = a.apply(co(rex, [1, 2]), exact=True), b.apply(co(rex, [1, 2]), exact=True)
    rex.add_edges(np.array([2], np.int32), np.array([0], np.int32))
    with pytest.raises(ValueError, match="population changed"):
        co_relate(u, v, metric, exact=True)


def test_large_rectangular_pipeline_forbids_dense_operator_materialization(monkeypatch):
    n = 4096
    rex = RexGraph.from_graph(np.zeros(n, dtype=int), np.arange(1, n+1))
    a = TypeAccession(rex, 1, "reduce", tuple((i % 2, i, Q(1, 3)) for i in range(n)),
                      coordinates=CoordinateSpace("parity", ("even", "odd")))
    b = TypeAccession(rex, 1, "cells", ((0, 0, 1), (n-1, n-1, 1)))
    metric = CrossMetric(a, b, ((0, 0, 2), (1, n-1, -1)))
    def forbidden(*args, **kw):
        pytest.fail("rectangular pipeline requested dense/eigen materialization")
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(np, "diag", forbidden)
    family = AccessionFamily((a, b)).apply(co(rex, np.ones(n, dtype=int)), exact=True)
    assert co_relate(*family.views, metric, exact=True) == Q(n, 6)
    numerical = AccessionFamily((a, b)).apply(co(rex, np.ones(n, dtype=int)))
    assert co_relate(*numerical.views, metric) == pytest.approx(n/6)
