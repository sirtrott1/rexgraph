"""Exact operator identities with dense matrices confined to test oracles."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.linear_operator import RexOperator
from rexgraph.operator_bracket import operator_bracket
from rexgraph.rational_operator import cayley, complex_structure, rational_rotation


def generator(matrix, grade=1, source=None):
    a = np.asarray(matrix, dtype=object)
    n = len(a)
    r = source if source is not None else RexGraph.from_cells([n+1, [[i, i+1] for i in range(n)]])
    diagonal = np.diag(np.arange(n, dtype=object))
    other = np.full((n, n), Q(0), dtype=object)
    for i in range(n):
        for j in range(n):
            if i != j:
                other[i, j] = Q(a[i, j])/(i-j)
    def wrap(q):
        return RexOperator("declared symmetric", (n, n), grade, grade, lambda x: np.asarray(q, float)@x,
            source=r, symmetric=True, exact_matvec=lambda x: q@x, exact_transpose_matvec=lambda x: q@x)
    return operator_bracket(wrap(diagonal), wrap(other))


@pytest.mark.parametrize("parameter", [Q(0), Q(1, 2), Q(-3, 7), Q(2)])
@pytest.mark.parametrize("shape", [(3,), (3, 0), (3, 2)])
def test_cayley_exact_inverse_and_quadrance(parameter, shape):
    g = generator([[0, 1, 2], [-1, 0, 3], [-2, -3, 0]])
    x = np.arange(np.prod(shape)).reshape(shape).astype(object)
    r = cayley(g, parameter)
    y = r.apply(x, exact=True)
    np.testing.assert_array_equal(y-parameter*g.apply(y, exact=True), x+parameter*g.apply(x, exact=True))
    np.testing.assert_array_equal(r.transpose_apply(y, exact=True), x)
    assert sum((Q(v)**2 for v in y.flat), Q(0)) == sum((Q(v)**2 for v in x.flat), Q(0))
    numeric = r.apply(np.asarray(x, float))
    np.testing.assert_allclose(numeric, np.asarray(y, float), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("triple", [(3, 4, 5), (5, -12, 13), (-1, 0, 1), (1, 0, 1)])
def test_partial_structure_and_rotation_fix_the_real_kernel(triple):
    g = generator([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
    j = complex_structure(g)
    assert dict(j.parameters)["scale"] == 2
    x = np.array([Q(2, 3), Q(-1, 7), Q(5)], dtype=object)
    jx = j.apply(x, exact=True)
    np.testing.assert_array_equal(j.apply(j.apply(jx, exact=True), exact=True), -jx)
    r = rational_rotation(j, *triple)
    y = r.apply(x, exact=True)
    assert y[-1] == x[-1]
    assert sum(v*v for v in y) == sum(v*v for v in x)
    np.testing.assert_array_equal(r.transpose_apply(y, exact=True), x)
    np.testing.assert_allclose(r.apply(np.asarray(x, float)), np.asarray(y, float))


def test_cayley_and_pythagorean_conventions_agree():
    j = complex_structure(generator([[0, 1], [-1, 0]]))
    x = np.array([1, 2], dtype=object)
    np.testing.assert_array_equal(cayley(j, Q(1, 2)).apply(x, exact=True),
                                  rational_rotation(j, 3, 4, 5).apply(x, exact=True))


@pytest.mark.parametrize("matrix, message", [
    ([[0, 1, 1], [-1, 0, 0], [-1, 0, 0]], "leaves Q"),
    ([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 2], [0, 0, -2, 0]], "single positive")])
def test_nonrational_or_multiple_frequencies_are_not_spectral_fallbacks(matrix, message):
    with pytest.raises(ValueError, match=message):
        complex_structure(generator(matrix))


@pytest.mark.parametrize("scale", [0, -1, Q(1, 2), 1.0, True])
def test_invalid_or_incorrect_scale_is_refused(scale):
    with pytest.raises((TypeError, ValueError)):
        complex_structure(generator([[0, 2], [-2, 0]]), scale)


@pytest.mark.parametrize("triple", [(1, 1, 1), (3, 4, -5), (3.0, 4, 5), (True, 0, 1)])
def test_invalid_rotation_coefficients_are_refused(triple):
    with pytest.raises((TypeError, ValueError)):
        rational_rotation(complex_structure(generator([[0, 1], [-1, 0]])), *triple)


def test_no_matrix_materialization_or_spectral_read(monkeypatch):
    g = generator([[0, 2], [-2, 0]])
    def refuse(*a, **kw):
        pytest.fail("matrix or spectral path reached")
    monkeypatch.setattr(RexOperator, "as_native", refuse)
    monkeypatch.setattr(RexOperator, "as_scipy", refuse)
    for name in ("svd", "eigh", "eig", "inv", "solve"):
        monkeypatch.setattr(np.linalg, name, refuse)
    j = complex_structure(g)
    for r in (cayley(g, Q(1, 3)), rational_rotation(j, 3, 4, 5)):
        assert r.apply(np.array([1, 2], object), exact=True).shape == (2,)


def test_same_population_source_mutation_invalidates_certificates():
    g = generator([[0, 1], [-1, 0]])
    j = complex_structure(g)
    g.source.set_cell_attrs([0, 1], w_E=[2, 3])
    with pytest.raises(ValueError, match="state changed"):
        j.apply(np.array([1, 2], object), exact=True)
    with pytest.raises(ValueError, match="state changed"):
        cayley(j, 0)


def test_zero_parameter_never_runs_generator_and_large_Q_is_not_cast(monkeypatch):
    g = generator([[0, 1], [-1, 0]])
    zero = cayley(g, 0)
    def refuse(*a, **kw):
        pytest.fail("zero Cayley evaluated its generator")
    monkeypatch.setattr(type(g), "apply", refuse)
    np.testing.assert_array_equal(zero.apply(np.array([1, 2], object), exact=True), [1, 2])
    cayley(g, Q(10**400))


def test_zero_generator_complex_structure_and_rotation_are_explicit():
    j = complex_structure(generator([[0, 0], [0, 0]]))
    x = np.array([1, 2], object)
    np.testing.assert_array_equal(j.apply(x, exact=True), [0, 0])
    np.testing.assert_array_equal(rational_rotation(j, 3, 4, 5).apply(x, exact=True), x)


@pytest.mark.parametrize("grade", range(6))
def test_selected_grade_preserves_the_complete_source_tower(grade):
    from rexgraph.cells import cell_count
    from rexgraph.io.catalog import object_digest
    face = [(0, 1), (1, 1), (2, -1)]
    difference = [(0, 1), (1, -1)]
    r = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]],
                            [face, face], [difference, difference], [difference]])
    digest = object_digest(r)
    n = cell_count(r, grade, allow_empty_upper=True)
    a = np.full((n, n), Q(0), object)
    if n > 1:
        a[0, 1], a[1, 0] = Q(2), Q(-2)
    g = generator(a, grade, r)
    x = np.arange(n).astype(object)
    for action in (cayley(g, Q(1, 2)), rational_rotation(complex_structure(g), 3, 4, 5)):
        y = action.apply(x, exact=True)
        assert action.domain_grade == grade and action.source is r
        np.testing.assert_array_equal(action.transpose_apply(y, exact=True), x)
        np.testing.assert_allclose(action.apply(np.asarray(x, float)), np.asarray(y, float))
    assert object_digest(r) == digest


def test_numerical_exhaustion_is_not_an_exact_solve_limit():
    g = generator([[0, 1, 2], [-1, 0, 3], [-2, -3, 0]])
    action = cayley(g, 1, maxiter=1)
    x = np.array([1, 2, 3], object)
    y = action.apply(x, exact=True)
    np.testing.assert_array_equal(y-g.apply(y, exact=True), x+g.apply(x, exact=True))
    with pytest.raises(RuntimeError):
        action.apply(np.asarray(x, float))


def test_original_equation_residual_is_checked(monkeypatch):
    from rexgraph.green import GreenOperator
    action = cayley(generator([[0, 1], [-1, 0]]), 1)
    monkeypatch.setattr(GreenOperator, "solve", lambda self, x: np.zeros_like(x))
    with pytest.raises(RuntimeError, match="original equation"):
        action.apply(np.array([1, 2], float))


@pytest.mark.parametrize("block", [False, True])
def test_complex_coefficients_use_the_real_action_without_loss(block):
    g = generator([[0, 2], [-2, 0]])
    x = np.array([[1+2j, 3-1j], [2-4j, -2+3j]])
    x = x if block else x[:, 0]
    for action in (cayley(g, Q(1, 3)), complex_structure(g),
                   rational_rotation(complex_structure(g), 3, 4, 5)):
        y = action.apply(x)
        expected = action.apply(x.real) + 1j*action.apply(x.imag)
        np.testing.assert_allclose(y, expected)
        np.testing.assert_allclose(action.transpose_apply(y), x)
