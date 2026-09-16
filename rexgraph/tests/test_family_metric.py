"""Coherent rectangular Gram forms, exact oracles and sparse realization actions."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.type_accession import (
    AccessionFamily,
    CoordinateField,
    CoordinateSpace,
    CrossMetric,
    FamilyMetric,
    TypeAccession,
    TypedFamily,
    TypedMomentTensor,
    TypeRealization,
    TypeView,
    co_relate,
    moment_tensor,
)


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


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("block", [False, True])
def test_realized_gram_matches_independent_dense_and_cross_block_oracles(exact, block):
    rex, maps, metric = make()
    x = np.array([[1, 2], [2, 1]]) if block else np.array([1, 2])
    views = maps.apply(co(rex, x), exact=exact)
    p, q = np.array([[1, 2]]), np.array([[1, 0], [0, -1], [1, 1]])
    ea, eb, m = np.array([[1], [-1]]), np.array([[1, 0, Q(1, 2)], [0, 2, -1]]), np.diag([2, 3])
    fields = [ea @ (p @ x), eb @ (q @ x)]
    expected = [[np.sum(u * (m @ v)) for v in fields] for u in fields]
    tensor = moment_tensor(views, metric, exact=exact)
    assert tensor.values.tolist() == expected
    assert tensor.names == ("summary", "detail") and tensor.metric is metric
    assert tensor.family is views
    assert metric.shape == (4, 4) and tensor.values.shape == (2, 2)
    if not block:
        assert expected == [[125, 130], [130, Q(319, 2)]]
    cross = ea.T @ m @ eb
    explicit = CrossMetric(*maps.accessions, tuple((i, j, cross[i, j])
        for i in range(cross.shape[0]) for j in range(cross.shape[1]) if cross[i, j]))
    assert co_relate(*views.views, metric, exact=exact) == expected[0][1]
    assert co_relate(*views.views, explicit, exact=exact) == expected[0][1]
    if exact:
        assert all(isinstance(v, Q) for v in tensor.values.flat)


def test_noninjective_realization_has_exact_zero_quadrance_for_nonzero_view():
    rex, maps, metric = make()
    b = maps.accessions[1]
    kernel = TypeView(b, CoordinateField(rex, 1, b.coordinates, np.array([-1, 1, 2]), "cochain"))
    assert any(kernel.values) and not np.any(metric.realization_for(kernel).apply(kernel, exact=True).values)
    assert co_relate(kernel, kernel, metric, exact=True) == Q(0)
    assert moment_tensor(TypedFamily((kernel,)), metric, exact=True).values.tolist() == [[0]]


def test_complex_gram_is_hermitian_and_cross_sign_is_not_erased():
    rex, maps, metric = make()
    u = maps.accessions[0].apply(co(rex, [1j, 1]))
    v = maps.accessions[1].apply(co(rex, [1, 2j]))
    # z_u=(2+i,-2-i); z_v=(3/2+i,-1-6i).
    tensor = moment_tensor(TypedFamily((u, v)), metric)
    assert tensor.values.tolist() == [[25, 32+34j], [32-34j, 117.5]]
    reversed_metric = FamilyMetric((metric.realizations[0],
        replace(metric.realizations[1], entries=tuple((i, j, -w) for i, j, w in metric.realizations[1].entries))), metric.metric)
    views = maps.apply(co(rex, [1, 2]), exact=True)
    assert moment_tensor(views, reversed_metric, exact=True).values.tolist() == [[125, -130], [-130, Q(319, 2)]]


def test_subfamilies_reordering_and_new_measurements_in_same_coordinates():
    rex, maps, metric = make()
    views = maps.apply(co(rex, [1, 2]), exact=True)
    assert moment_tensor(TypedFamily(views.views[::-1]), metric, exact=True).values.tolist() == [[Q(319, 2), 130], [130, 125]]
    assert moment_tensor(TypedFamily((views.views[1],)), metric, exact=True).values.tolist() == [[Q(319, 2)]]
    other = replace(maps.accessions[0], entries=((0, 0, -1),)).apply(co(rex, [1, 2]), exact=True)
    assert co_relate(other, views.views[1], metric, exact=True) == -26
    wrong_name = replace(maps.accessions[0], name="unregistered").apply(co(rex, [1, 2]), exact=True)
    with pytest.raises(ValueError, match="no realization"):
        co_relate(wrong_name, views.views[1], metric, exact=True)


@pytest.mark.parametrize("mismatch", ["source", "grade", "basis", "coordinates", "variance", "block"])
def test_realizations_and_family_pairings_enforce_all_spaces(mismatch):
    rex, maps, metric = make()
    u, v = maps.apply(co(rex, [1, 2]), exact=True).views
    if mismatch == "source":
        other, omaps, _ = make()
        u = omaps.accessions[0].apply(co(other, [1, 2]), exact=True)
    elif mismatch == "grade":
        a = replace(maps.accessions[0], grade=0)
        u = a.apply(Cochain(0, np.array([1, 2, 3]), source=rex), exact=True)
    elif mismatch == "basis":
        a = replace(maps.accessions[0], cell_keys=("b", "a"))
        u = a.apply(Cochain(1, np.array([1, 2]), source=rex, cell_keys=a.cell_keys), exact=True)
    elif mismatch == "coordinates":
        a = replace(maps.accessions[0], coordinates=CoordinateSpace("other", ("total",)))
        u = a.apply(co(rex, [1, 2]), exact=True)
    elif mismatch == "variance":
        u = maps.accessions[0].apply(Chain(1, np.array([1, 2]), source=rex), exact=True)
    else:
        u = maps.accessions[0].apply(co(rex, [[1], [2]]), exact=True)
    with pytest.raises(ValueError):
        co_relate(u, v, metric, exact=True)


@pytest.mark.parametrize("defect", ["empty", "duplicate", "foreign", "wrong-basis", "not-diagonal", "not-realization"])
def test_invalid_family_declarations_refuse(defect):
    rex, maps, metric = make()
    factors, base = metric.realizations, metric.metric
    if defect == "empty":
        factors = ()
    elif defect == "duplicate":
        factors = (factors[0], factors[0])
    elif defect == "foreign":
        base = make()[2].metric
    elif defect == "wrong-basis":
        base = DiagonalMetric(rex, 1, (2, 3), ("b", "a"))
    elif defect == "not-diagonal":
        base = CrossMetric(*maps.accessions, ())
    else:
        factors = (maps.accessions[0],)
    with pytest.raises((TypeError, ValueError)):
        FamilyMetric(factors, base)


@pytest.mark.parametrize("entries", [((2, 0, 1),), ((0, 1, 1),), ((False, 0, 1),), ((0, .5, 1),),
    ((0, 0, True),), ((0, 0, 1j),), ((0, 0, float("nan")),), ((0, 0, 1e308), (0, 0, 1e308))])
def test_realization_axes_and_coefficients_are_validated(entries):
    _, maps, _ = make()
    with pytest.raises((TypeError, ValueError, OverflowError)):
        TypeRealization(maps.accessions[0], entries)


def test_factors_are_copied_coalesced_and_digest_tracks_metric_and_order():
    rex, maps, metric = make()
    entries = [[0, 0, Q(1, 3)], [0, 0, Q(1, 6)]]
    e = TypeRealization(maps.accessions[0], entries)
    entries[0][2] = 99
    assert e.entries == ((0, 0, Q(1, 2)),)
    assert e.coefficient_digest == TypeRealization(e.accession, ((0, 0, Q(1, 2)),)).coefficient_digest
    assert metric.coefficient_digest != FamilyMetric(metric.realizations[::-1], metric.metric).coefficient_digest
    assert metric.coefficient_digest != FamilyMetric(metric.realizations, DiagonalMetric(rex, 1, (2, 4))).coefficient_digest


@pytest.mark.parametrize("weight", [Q(10**400), Q(1, 10**400)])
def test_extreme_primary_rational_factors_do_not_pass_through_double(weight):
    rex, maps, metric = make()
    e = TypeRealization(maps.accessions[0], ((0, 0, weight),))
    form = FamilyMetric((e,), metric.metric)
    u = maps.accessions[0].apply(co(rex, [1, 2]), exact=True)
    assert moment_tensor(TypedFamily((u,)), form, exact=True).values[0, 0] == 50*weight**2
    with pytest.raises(FloatingPointError, match="representable"):
        moment_tensor(TypedFamily((u,)), form)


@pytest.mark.parametrize("floating", ["factor", "base", "field", "unused-factor"])
def test_exactness_requires_the_supplied_form_and_fields_to_be_exact(floating):
    rex, maps, metric = make()
    family = maps.apply(co(rex, [1, 2]), exact=True)
    if floating in {"factor", "unused-factor"}:
        e = TypeRealization(maps.accessions[1], ((0, 0, 1.), (0, 0, -1.)))
        assert not e.exact and not e.entries
        metric = FamilyMetric((metric.realizations[0], e), metric.metric)
        if floating == "unused-factor":
            family = TypedFamily((family.views[0],))
    elif floating == "base":
        metric = FamilyMetric(metric.realizations, DiagonalMetric(rex, 1, (2., 3.)))
    else:
        family = maps.apply(co(rex, [1, 2]))
    with pytest.raises(TypeError, match="integer|Fraction"):
        moment_tensor(family, metric, exact=True)
    assert moment_tensor(family, metric).values.dtype.kind == "f"


@pytest.mark.parametrize("grade", [0, 1, 2])
@pytest.mark.parametrize("variance", ["chain", "cochain"])
def test_realization_returns_declared_ambient_carrier_at_every_present_grade(grade, variance):
    from rexgraph.cells import cell_count
    rex = RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])
    n = cell_count(rex, grade)
    a = TypeAccession(rex, grade, "a", ((0, n-1, 2),), coordinates=CoordinateSpace("single", ("x",)))
    e = TypeRealization(a, ((n-1, 0, Q(1, 2)),))
    cls = Chain if variance == "chain" else Cochain
    view = a.apply(cls(grade, np.ones(n, dtype=int), source=rex), exact=True)
    realized = e.apply(view, exact=True)
    assert isinstance(realized, cls) and realized.grade == grade and realized.source is rex
    assert realized.values.tolist() == [Q(0)]*(n-1)+[Q(1)]


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("empty", ["block", "coordinate", "source"])
def test_zero_and_empty_realizations_preserve_family_axes(exact, empty):
    rex, maps, metric = make()
    x = co(rex, [1, 2])
    if empty == "block":
        x = co(rex, np.empty((2, 0), dtype=int))
    elif empty == "coordinate":
        maps = AccessionFamily(tuple(replace(a, entries=(), coordinates=CoordinateSpace(a.name, ())) for a in maps.accessions))
        metric = FamilyMetric(tuple(TypeRealization(a, ()) for a in maps.accessions), metric.metric)
    else:
        rex = RexGraph.from_graph([], [])
        maps = AccessionFamily(tuple(replace(a, source=rex, entries=()) for a in maps.accessions))
        metric = FamilyMetric(tuple(TypeRealization(a, ()) for a in maps.accessions), DiagonalMetric(rex, 1, ()))
        x = co(rex, np.empty(0, dtype=int))
    result = moment_tensor(maps.apply(x, exact=exact), metric, exact=exact)
    assert result.values.tolist() == [[0, 0], [0, 0]]
    assert result.values.dtype.kind == ("O" if exact else "f")


def test_direct_tensor_checks_family_without_reapplying_realizations(monkeypatch):
    rex, maps, metric = make()
    family = maps.apply(co(rex, [1, 2]), exact=True)
    def forbidden(*args, **kw):
        pytest.fail("tensor validation applied a realization")
    monkeypatch.setattr(TypeRealization, "apply", forbidden)
    values = np.array([[125, 130], [130, Q(319, 2)]])
    tensor = TypedMomentTensor(values, family, metric)
    values[0, 0] = 99
    assert tensor.values[0, 0] == 125 and not tensor.values.flags.writeable
    rex.add_edges(np.array([2], np.int32), np.array([0], np.int32))
    with pytest.raises(ValueError, match="population changed"):
        TypedMomentTensor(values, family, metric)


def test_large_factored_family_never_builds_cross_blocks_or_dense_gram(monkeypatch):
    n = 4096
    rex = RexGraph.from_graph(np.zeros(n, dtype=int), np.arange(1, n+1))
    a = TypeAccession(rex, 1, "a", ((0, 0, 1),), coordinates=CoordinateSpace("one", ("x",)))
    b = TypeAccession(rex, 1, "b", ((0, 1, 1), (1, 2, 1)), coordinates=CoordinateSpace("two", ("y", "z")))
    ea = TypeRealization(a, tuple((i, 0, Q(1, 3)) for i in range(n)))
    eb = TypeRealization(b, ((0, 0, -1), (n-1, 1, 2)))
    metric = FamilyMetric((ea, eb), DiagonalMetric(rex, 1, (1,)*n))
    family = AccessionFamily((a, b)).apply(co(rex, np.ones(n, dtype=int)), exact=True)
    def forbidden(*args, **kw):
        pytest.fail("family contraction requested dense/eigen/cross-block materialization")
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(CrossMetric, "__post_init__", forbidden)
    actions, original = [], TypeRealization.apply
    def measured(self, *args, **kw):
        actions.append(self.accession.name)
        return original(self, *args, **kw)
    monkeypatch.setattr(TypeRealization, "apply", measured)
    assert moment_tensor(family, metric, exact=True).values.tolist() == [[Q(n, 9), Q(1, 3)], [Q(1, 3), 5]]
    assert actions == ["a", "b"]
    np.testing.assert_allclose(moment_tensor(family, metric).values, [[n/9, 1/3], [1/3, 5]])
    assert actions == ["a", "b", "a", "b"]
