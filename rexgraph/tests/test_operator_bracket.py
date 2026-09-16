"""Ordinary brackets are ordered factored actions, not products or guessed zeros."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.cells import cell_count
from rexgraph.channel_operator import channel_operator
from rexgraph.cochain import Chain
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.green import GreenOperator
from rexgraph.linear_operator import RexOperator, metric_adjoint
from rexgraph.operator_bracket import operator_bracket
from rexgraph.weighted_dirac import GradedChain, weighted_dirac
from rexgraph.weighted_hodge import weighted_hodge


def make(case="branching"):
    if case == "branching":
        return RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2])
    if case == "witness":
        return RexGraph.from_hypergraph([0, 1, 3], [0, 0, 1])
    if case == "face":
        return RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])
    return RexGraph.from_cells(solid_octahedron_3rex())


def declared(r, coefficients, **kw):
    q = np.array([[Q(v) for v in row] for row in coefficients], object)
    a = np.asarray(q, float)
    return RexOperator("declared", (2, 2), 1, 1, lambda x: a@x, source=r,
        variance="chain", transpose_matvec=lambda x: a.T@x,
        exact_matvec=lambda x: q@x, exact_transpose_matvec=lambda x: q.T@x, **kw)


@pytest.mark.parametrize("anti", [False, True])
@pytest.mark.parametrize("columns", [None, 0, 3])
def test_hand_order_exact_transpose_and_nested_adjoint(anti, columns):
    r = make()
    a = declared(r, [[1, 2], [0, 3]])
    b = declared(r, [[0, 1], [-1, 2]])
    k = operator_bracket(a, b, anti=anti)
    expected = np.array([[-2, 8], [-4, 10]] if anti else [[-2, 2], [-2, 2]], object)
    shape = (2,) if columns is None else (2, columns)
    x = np.arange(int(np.prod(shape))).reshape(shape)+2
    np.testing.assert_array_equal(k.apply(x, exact=True), expected@x)
    np.testing.assert_array_equal(k.transpose_apply(x, exact=True), expected.T@x)
    np.testing.assert_allclose(k.apply(x), np.asarray(expected@x, float))
    np.testing.assert_array_equal(metric_adjoint(k).apply(x, exact=True), expected.T@x)
    assert not k.symmetric and not k.psd and k.metric_self_adjoint is None
    with pytest.raises(TypeError, match="no sparse matrix"):
        k.as_scipy()


def test_exact_nested_jacobi_identity_and_antisymmetry():
    r = make()
    a, b, c = (declared(r, v) for v in ([[1, 2], [0, 3]], [[0, 1], [-1, 2]], [[2, -3], [4, 1]]))
    x = np.array([Q(2, 3), Q(-1, 7)], object)
    def bracket(u, v):
        return operator_bracket(u, v)
    pieces = [bracket(a, bracket(b, c)), bracket(b, bracket(c, a)), bracket(c, bracket(a, b))]
    assert not any(sum(k.apply(x, exact=True) for k in pieces))
    np.testing.assert_array_equal(bracket(a, b).apply(x, exact=True), -bracket(b, a).apply(x, exact=True))


@pytest.mark.parametrize("anti", [False, True])
def test_two_psd_operands_do_not_imply_psd_bracket(anti):
    r = make()
    a = declared(r, [[1, 0], [0, 0]], symmetric=True, psd=True)
    b = declared(r, [[1, 2], [2, 4]], symmetric=True, psd=True)
    k = operator_bracket(a, b, anti=anti)
    assert k.symmetric is anti and not k.psd
    assert k.euclidean_skew_adjoint is (None if anti else True)
    if anti:
        x = np.array([1, -1])
        assert x@k.apply(x, exact=True) == -2
    with pytest.raises(ValueError, match="positive semidefinite"):
        GreenOperator.resolvent(k)


@pytest.mark.parametrize("name", ["G", "F", "C"])
@pytest.mark.parametrize("selection", ["raw", "normalized"])
def test_channel_brackets_match_small_oracle_and_declared_exact_actions(name, selection):
    r = RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2], g_channel=selection)
    a, b = channel_operator(r, "T"), channel_operator(r, name)
    aa, bb = a.as_scipy().toarray(), b.as_scipy().toarray()
    k = operator_bracket(a, b)
    x = np.array([[1j, 2], [2, -1j]])
    np.testing.assert_allclose(k.apply(x), (aa@bb-bb@aa)@x, atol=1e-12)
    np.testing.assert_allclose(k.transpose_apply(x), (aa@bb-bb@aa).T@x, atol=1e-12)
    assert k.euclidean_skew_adjoint is True
    if name == "G" and selection == "normalized":
        with pytest.raises(TypeError, match="certified exact"):
            k.apply(np.ones(2, int), exact=True)
    else:
        exact = k.apply(np.ones(2, int), exact=True)
        assert all(isinstance(v, Q) for v in exact)
        np.testing.assert_allclose(np.asarray(exact, float), (aa@bb-bb@aa)@np.ones(2), atol=1e-12)


@pytest.mark.parametrize("case", ["branching", "witness", "face", "grade3"])
@pytest.mark.parametrize("columns", [None, 0, 2])
def test_full_tower_dirac_brackets_equal_the_exact_hodge_identities(case, columns):
    r = make(case)
    sizes = [cell_count(r, k) for k in range(len(r.graded_boundaries())+1)]
    ms = [DiagonalMetric(r, k, tuple(Q(i+2, i+1) for i in range(n))) for k, n in enumerate(sizes)]
    d, a = weighted_dirac(r, metrics=ms), weighted_dirac(r, metrics=ms, anti=True)
    k, zero = operator_bracket(d, a), operator_bracket(d, a, anti=True)
    components = []
    for grade, n in enumerate(sizes):
        shape = (n,) if columns is None else (n, columns)
        components.append(Chain(grade, np.arange(int(np.prod(shape))).reshape(shape)+1, source=r))
    x = GradedChain(r, components)
    out = k.apply(x, exact=True)
    zeros = zero.apply(x, exact=True)
    for grade, _n in enumerate(sizes):
        diff = weighted_hodge(r, grade, sector="difference", metric=ms[grade],
            lower_metric=ms[grade-1] if grade else None, upper_metric=ms[grade+1] if grade+1 < len(sizes) else None)
        np.testing.assert_array_equal(out.component(grade).values, 2*diff.apply(x.component(grade).values, exact=True))
        assert not any(zeros.component(grade).values.flat)
    assert k.metric_self_adjoint and zero.metric_skew_adjoint


@pytest.mark.parametrize("anti", [False, True])
def test_graded_transpose_uses_reverse_product_order(anti):
    r = make()
    ms = [DiagonalMetric(r, 1, (2, 3))]
    d, a = weighted_dirac(r, metrics=ms), weighted_dirac(r, metrics=ms, anti=True)
    k = operator_bracket(d, a, anti=anti)
    n = sum(d.sizes)
    eye = np.eye(n, dtype=int)
    offsets = np.cumsum((0, *d.sizes))
    basis = GradedChain(r, [Chain(i, eye[offsets[i]:offsets[i+1]], source=r) for i in range(len(d.sizes))])
    def flat(state):
        return np.concatenate([c.values for c in state.components], axis=0)
    matrix = flat(k.apply(basis, exact=True))
    np.testing.assert_array_equal(flat(k.transpose_apply(basis, exact=True)), matrix.T)
    numeric = k.apply(GradedChain(r, [c.with_values(c.values.astype(complex)*1j) for c in basis.components]))
    np.testing.assert_allclose(flat(numeric), np.asarray(matrix, complex)*1j)


def test_different_metrics_are_composable_but_do_not_share_an_adjoint_certificate():
    r = make()
    a, b = weighted_dirac(r), weighted_dirac(r, metrics=[DiagonalMetric(r, 1, (2, 3))])
    k = operator_bracket(a, b)
    assert k.grade_metrics == () and k.metric_self_adjoint is None and k.metric_skew_adjoint is None
    x = GradedChain(r, [Chain(1, np.array([1, 2]), source=r)])
    assert any(any(c.values.flat) for c in k.apply(x, exact=True).components)
    h = operator_bracket(weighted_hodge(r, 1), weighted_hodge(r, 1, metric=DiagonalMetric(r, 1, (2, 3))))
    assert h.grade_metrics == () and h.metric_skew_adjoint is None and not h.symmetric


def test_broken_chain_law_is_not_optimized_to_a_false_zero():
    from rexgraph.native_sparse import as_native
    r = make("grade3")
    upper = as_native(r._graded_duals[0])
    data = upper.data.copy()
    data[0] += 1
    r._graded_duals = [upper.with_data(data).dual]
    d, a = weighted_dirac(r), weighted_dirac(r, anti=True)
    x = GradedChain(r, [Chain(3, np.ones(1, int), source=r)])
    out = operator_bracket(d, a, anti=True).apply(x, exact=True)
    assert any(any(c.values.flat) for c in out.components)


@pytest.mark.parametrize("bad", ["foreign", "variance", "axes", "grade", "mixed", "green", "anti"])
def test_invalid_operands_refuse(bad):
    r = make()
    a, b = weighted_hodge(r, 1), weighted_hodge(r, 1)
    if bad == "foreign":
        b = weighted_hodge(make(), 1)
    elif bad == "variance":
        b = replace(b, variance="cochain")
    elif bad == "axes":
        b = replace(b, shape=(3, 3))
    elif bad == "grade":
        b = replace(b, domain_grade=0, codomain_grade=0)
    elif bad == "mixed":
        b = weighted_dirac(r)
    elif bad == "green":
        b = GreenOperator.resolvent(b)
    with pytest.raises((ValueError, TypeError)):
        operator_bracket(a, b, anti=1 if bad == "anti" else False)


def test_missing_exact_and_transpose_are_not_manufactured():
    r = make()
    a = RexOperator("opaque", (2, 2), 1, 1, lambda x: x, source=r)
    k = operator_bracket(a, a)
    assert not k.has_transpose and k.exact_matvec is None
    with pytest.raises(TypeError, match="transpose"):
        k.transpose_apply(np.ones(2))
    with pytest.raises(TypeError, match="exact"):
        k.apply(np.ones(2, int), exact=True)


def test_large_bracket_never_assembles_products_or_eigenbases(monkeypatch):
    n = 4096
    r = RexGraph.from_graph(np.zeros(n, int), np.arange(1, n+1))
    a, b = channel_operator(r, "T"), channel_operator(r, "F")
    def forbidden(*a, **kw):
        pytest.fail("bracket materialized a product matrix or eigenbasis")
    monkeypatch.setattr(RexOperator, "as_scipy", forbidden)
    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    x = np.arange(n, dtype=float)
    y = operator_bracket(a, b).apply(x)
    assert y.shape == x.shape and np.all(np.isfinite(y))


@pytest.mark.parametrize("graded", [False, True])
def test_empty_axes_are_preserved(graded):
    r = RexGraph.from_graph([], [])
    op = weighted_dirac(r) if graded else weighted_hodge(r, 1)
    x = GradedChain(r, [Chain(1, np.empty((0, 2), int), source=r)]) if graded else np.empty((0, 2), int)
    out = operator_bracket(op, op).apply(x, exact=True)
    if graded:
        assert all(c.values.shape == (0, 2) for c in out.components)
    else:
        assert out.shape == (0, 2)


def test_explicit_empty_upper_endomorphism_does_not_invent_higher_grades():
    r = make()
    grade = len(r.graded_boundaries())+1
    op = RexOperator("empty", (0, 0), grade, grade, lambda x: x.copy(), source=r,
                     exact_matvec=lambda x: x.copy(), exact_transpose_matvec=lambda x: x.copy())
    x = np.empty((0, 2), int)
    assert operator_bracket(op, op).apply(x, exact=True).shape == x.shape
    beyond = replace(op, domain_grade=grade+1, codomain_grade=grade+1)
    with pytest.raises(ValueError, match="not present"):
        operator_bracket(beyond, beyond)
