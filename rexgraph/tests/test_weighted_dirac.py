"""Independent graded/metric identities and coordinate oracles for D and A."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.cochain import Chain, Cochain
from rexgraph.dirac_propagator import dirac_from_rex
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import RexOperator, _boundaries
from rexgraph.weighted_dirac import GradedChain, weighted_dirac
from rexgraph.weighted_hodge import weighted_hodge


def make(kind="branch"):
    if kind == "branch":
        return RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2])
    if kind == "witness":
        return RexGraph.from_hypergraph([0, 1, 3], [0, 0, 1])
    if kind == "parallel":
        return RexGraph.from_hypergraph([0, 2, 4, 6], [0, 1, 0, 1, 1, 2])
    if kind == "solid":
        return RexGraph.from_cells(solid_octahedron_3rex())
    return RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])


def stacked(state):
    return np.concatenate([c.values for c in state.components], axis=0)


def states(rex, columns=None):
    d = weighted_dirac(rex)
    shape = () if columns is None else (columns,)
    return GradedChain(rex, [Chain(k, np.arange(n * (columns if columns is not None else 1)).reshape((n, *shape)) + k+1,
                                   source=rex) for k, n in enumerate(d.sizes)])


@pytest.mark.parametrize("kind", ["branch", "witness", "parallel", "triangle", "solid"])
@pytest.mark.parametrize("columns", [None, 0, 2])
def test_exact_dirac_hodge_identities(kind, columns):
    rex = make(kind)
    x = states(rex, columns)
    metrics = [DiagonalMetric(rex, k, tuple(Q(i+2, k+1) for i in range(n))) for k, n in enumerate(x.sizes)]
    d, a = weighted_dirac(rex, metrics=metrics), weighted_dirac(rex, metrics=metrics, anti=True)
    dx, ax = d.apply(x, exact=True), a.apply(x, exact=True)
    dd, aa = d.apply(dx, exact=True), a.apply(ax, exact=True)
    da, ad = d.apply(ax, exact=True), a.apply(dx, exact=True)
    for k, c in enumerate(x.components):
        kwargs = {"metric": metrics[k]}
        if k:
            kwargs["lower_metric"] = metrics[k-1]
        if k+1 < len(metrics):
            kwargs["upper_metric"] = metrics[k+1]
        h = weighted_hodge(rex, k, **kwargs).apply(c.values, exact=True)
        difference = weighted_hodge(rex, k, sector="difference", **kwargs).apply(c.values, exact=True)
        np.testing.assert_array_equal(dd.component(k).values, h)
        np.testing.assert_array_equal(aa.component(k).values, -h)
        np.testing.assert_array_equal(da.component(k).values + ad.component(k).values, np.zeros_like(h))
        np.testing.assert_array_equal((da.component(k).values - ad.component(k).values)/2, difference)
    # All identities are exact Fractions, including branching shares and every carried grade.
    assert d.exact and a.exact and dx.exact and ax.exact


@pytest.mark.parametrize("anti", [False, True])
@pytest.mark.parametrize("kind", ["branch", "witness", "parallel", "triangle", "solid"])
def test_numerical_dense_oracle_and_hermitian_transpose(kind, anti):
    rex = make(kind)
    x = states(rex, 2)
    metrics = [DiagonalMetric(rex, k, tuple(range(2, n+2))) for k, n in enumerate(x.sizes)]
    d = weighted_dirac(rex, metrics=metrics, anti=anti)
    off = np.cumsum([0, *x.sizes])
    oracle = np.zeros(d.shape)
    for k, b in enumerate(_boundaries(rex), 1):
        b = b.as_scipy()  # Explicit dense oracle; production uses DualCSR actions.
        lower, upper = slice(off[k-1], off[k]), slice(off[k], off[k+1])
        oracle[lower, upper] = b.toarray()
        oracle[upper, lower] = (-1 if anti else 1) * np.diag(1/np.asarray(metrics[k].weights, float)) @ b.toarray().T @ np.diag(np.asarray(metrics[k-1].weights, float))
    x = GradedChain(rex, [c.with_values(np.asarray(c.values, complex)*(1+2j)) for c in x.components])
    np.testing.assert_allclose(stacked(d.apply(x)), oracle @ stacked(x), atol=1e-12)
    np.testing.assert_allclose(stacked(d.transpose_apply(x)), oracle.T @ stacked(x), atol=1e-12)
    y = d.apply(x)
    metric = np.concatenate([np.asarray(m.weights, float) for m in metrics])[:, None]
    lhs = np.vdot(stacked(x), metric * stacked(d.apply(y)))
    rhs = np.vdot(stacked(d.apply(x)), metric * stacked(y))
    np.testing.assert_allclose(lhs, (-1 if anti else 1)*rhs)
    assert d.metric_skew_adjoint is anti and d.metric_self_adjoint is not anti


def test_independent_hand_branching_and_metric_values():
    rex = make()
    x = GradedChain(rex, [Chain(0, np.array([1, 2, 3, 4]), source=rex),
                          Chain(1, np.array([3, 2]), source=rex)])
    metrics = [DiagonalMetric(rex, 0, (1, 2, 3, 4)), DiagonalMetric(rex, 1, (2, 3))]
    d = weighted_dirac(rex, metrics=metrics)
    y = d.apply(x, exact=True)
    assert y.component(0).values.tolist() == [Q(-1), Q(1), Q(-1), Q(1)]
    assert y.component(1).values.tolist() == [Q(-1), Q(8, 3)]
    assert d.apply(y, exact=True).component(1).values.tolist() == [Q(7, 3), Q(-2, 3)]
    a = weighted_dirac(rex, metrics=metrics, anti=True).apply(x, exact=True)
    assert a.component(1).values.tolist() == [Q(1), Q(-8, 3)]


def test_identity_numeric_agrees_with_existing_sparse_dirac():
    rex = make("solid")
    x = states(rex, 3)
    np.testing.assert_allclose(stacked(weighted_dirac(rex).apply(x)),
                               dirac_from_rex(rex).matvec(stacked(x)))


def test_exact_metric_adjointness_and_coordinate_transpose():
    rex = make()
    x = states(rex)
    y = GradedChain(rex, [c.with_values(c.values+2) for c in x.components])
    metrics = [DiagonalMetric(rex, 0, (1, 2, 3, 4)), DiagonalMetric(rex, 1, (2, 3))]
    def pairing(u, v):
        return sum(m.moment(a, b, exact=True) for m, a, b in zip(metrics, u.components, v.components, strict=True))
    for anti in (False, True):
        d = weighted_dirac(rex, metrics=metrics, anti=anti)
        assert pairing(x, d.apply(y, exact=True)) == (-1 if anti else 1)*pairing(d.apply(x, exact=True), y)
        assert sum(stacked(x)*stacked(d.apply(y, exact=True))) == sum(stacked(d.transpose_apply(x, exact=True))*stacked(y))


@pytest.mark.parametrize("bad", ["variance", "source", "basis", "shape", "block", "duplicate", "grade", "bool", "nan", "text"])
def test_invalid_seeds_refused(bad):
    rex = make()
    c = Chain(1, np.array([1, 2]), source=rex)
    if bad == "variance": c = Cochain(1, c.values, source=rex)
    if bad == "source": c = Chain(1, c.values, source=make())
    if bad == "basis": c = Chain(1, c.values, ("a", "b"), rex)
    if bad == "shape": c = c.with_values(np.ones(3, int))
    if bad == "block": c = c.with_values(np.ones((2, 1, 1), int))
    if bad == "grade": c = Chain(2, np.empty(0, int), source=rex)
    if bad == "bool": c = c.with_values(np.ones(2, bool))
    if bad == "nan": c = c.with_values(np.array([1, np.nan]))
    if bad == "text": c = c.with_values(np.array(["1", "2"]))
    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        GradedChain(rex, [c, c] if bad == "duplicate" else [c])


@pytest.mark.parametrize("bad", ["source", "basis", "duplicate", "grade", "kind"])
def test_invalid_metrics_refused(bad):
    rex = make()
    m = DiagonalMetric(rex, 1, (2, 3))
    if bad == "source": m = DiagonalMetric(make(), 1, (2, 3))
    if bad == "basis": m = DiagonalMetric(rex, 1, (2, 3), ("a", "b"))
    if bad == "grade": m = DiagonalMetric(rex, 2, ())
    if bad == "kind": m = (2, 3)
    with pytest.raises((TypeError, ValueError)):
        weighted_dirac(rex, metrics=[m, m] if bad == "duplicate" else [m])


def test_float_metrics_and_carriers_do_not_acquire_exactness():
    rex = make()
    x = states(rex)
    d = weighted_dirac(rex, metrics=[DiagonalMetric(rex, 1, (2., 3.))])
    assert not d.exact
    with pytest.raises(TypeError, match="certified exact"):
        d.apply(x, exact=True)
    x = GradedChain(rex, [x.component(1).with_values(np.ones(2))])
    with pytest.raises(TypeError, match="integers or Fractions"):
        weighted_dirac(rex).apply(x, exact=True)
    assert not d.apply(x).exact


def test_extreme_rational_metrics_never_pass_through_float():
    rex = make()
    x = states(rex)
    d = weighted_dirac(rex, metrics=[DiagonalMetric(rex, 1, (Q(10**400), Q(1, 10**400)))])
    assert d.apply(x, exact=True).exact
    with pytest.raises(FloatingPointError):
        d.apply(x)


def test_zero_tower_and_sparse_hub_no_materialization(monkeypatch):
    rex = RexGraph(sources=np.array([], int), targets=np.array([], int))
    x = GradedChain(rex)
    d = weighted_dirac(rex, metrics=[DiagonalMetric(rex, 0, ())])
    assert d.exact and d.apply(x, exact=True).sizes == (0, 0)
    hub = RexGraph(sources=np.zeros(4096, int), targets=np.arange(1, 4097))
    x = GradedChain(hub, [Chain(1, np.ones(4096, int), source=hub)])
    def forbidden(*a, **k): pytest.fail("matrix/eigenbasis materialization")
    monkeypatch.setattr(RexOperator, "as_scipy", forbidden)
    for name in ("diag",): monkeypatch.setattr(np, name, forbidden)
    for name in ("eig", "eigh", "inv"): monkeypatch.setattr(np.linalg, name, forbidden)
    y = weighted_dirac(hub).apply(x, exact=True)
    assert y.component(0).values[0] == -4096
    assert not any(y.component(1).values)


def test_seed_copy_and_missing_grade_shapes():
    rex = make()
    original = np.ones((2, 0), int)
    x = GradedChain(rex, [Chain(1, original, source=rex)])
    assert [c.values.shape for c in x.components] == [(4, 0), (2, 0)]
    assert x.exact and not x.component(1).values.flags.writeable
    with pytest.raises(TypeError): x.component(True)
    with pytest.raises(ValueError): x.component(-1)
    with pytest.raises(ValueError): x.component(2)


def test_noninteger_higher_boundary_numeric_only_and_nonfinite_refusal():
    from rexgraph.native_sparse import as_native
    rex = make("solid")
    upper = as_native(rex._graded_duals[0])
    rex._graded_duals = [upper.with_data(upper.data * .5).dual]
    d = weighted_dirac(rex)
    assert not d.exact
    assert not d.apply(states(rex)).exact
    with pytest.raises(TypeError, match="certified exact"): d.apply(states(rex), exact=True)
    upper = as_native(rex._graded_duals[0])
    data = upper.data.copy()
    data[0] = np.nan
    rex._graded_duals = [upper.with_data(data).dual]
    with pytest.raises(ValueError, match="finite"): weighted_dirac(rex)
