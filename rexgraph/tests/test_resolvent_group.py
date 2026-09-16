"""Exact group words reuse factored Core actions and positive solves."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.channel_operator import channel_operator
from rexgraph.rational_operator import ResolventGroup, resolvent_word


def fixture():
    r = RexGraph.from_cells([4, [[2, 0, 1, 3], [0, 2]]], g_channel="raw")
    return r, [channel_operator(r, c) for c in ("T", "G")]


@pytest.mark.parametrize("word", [(), (1,), (-1,), (1, 2), (2, 1), (1, -2, 1), (1, 2, -2, -1)])
@pytest.mark.parametrize("block", [False, True])
def test_exact_inverse_transpose_and_dense_reference(word, block):
    r, ops = fixture()
    group = ResolventGroup(ops, [Q(1, 2), Q(2, 3)], word)
    x = np.array([[Q(2, 3), Q(3, 5)], [Q(-1, 7), Q(4, 9)]], object)
    if not block:
        x = x[:, 0]
    y = group.apply(x, exact=True)
    np.testing.assert_array_equal(group.inverse.apply(y, exact=True), x)
    np.testing.assert_array_equal(group.identity.apply(x, exact=True), x)
    eye = np.array([[Q(1), Q(0)], [Q(0), Q(1)]], object)
    matrix = group.apply(eye, exact=True)
    np.testing.assert_array_equal(group.transpose_apply(x, exact=True), matrix.T @ x)
    expected = np.eye(2)
    for letter in word:
        i = abs(letter)-1
        forward = np.eye(2) + float(group.scales[i])*np.asarray(ops[i].apply(eye, exact=True), float)
        expected = expected @ (np.linalg.inv(forward) if letter > 0 else forward)
    np.testing.assert_allclose(np.asarray(y, float), expected @ np.asarray(x, float), atol=1e-12)
    np.testing.assert_allclose(group.apply(np.asarray(x, float)), np.asarray(y, float), atol=1e-10)
    assert group.matrix is None and group.matrix_factory is None


def test_words_do_not_assume_commutation_or_single_parameter_closure():
    _, ops = fixture()
    group = ResolventGroup(ops, [1, 1])
    x = np.array([Q(1), Q(0)], object)
    assert any(group.element([1, 2]).apply(x, exact=True) != group.element([2, 1]).apply(x, exact=True))
    product = ResolventGroup([ops[0], ops[0]], [1, 2], [1, 2])
    summed = ResolventGroup([ops[0]], [3], [1])
    assert any(product.apply(x, exact=True) != summed.apply(x, exact=True))
    np.testing.assert_array_equal(product.apply(x, exact=True), product.element([2, 1]).apply(x, exact=True))


def test_zero_parameter_cancellation_and_large_exact_coefficients(monkeypatch):
    _, ops = fixture()
    from rexgraph.green import GreenOperator
    group = ResolventGroup(ops, [0, 10**400], [1, 2, -2])
    assert group.word == ()
    x = np.array([Q(1), Q(2)], object)
    np.testing.assert_array_equal(group.apply(x, exact=True), x)
    exact = group.element([2])
    monkeypatch.setattr(GreenOperator, "solve", lambda *a: pytest.fail("numerical solve ran"))
    np.testing.assert_array_equal(exact.inverse.apply(exact.apply(x, exact=True), exact=True), x)


@pytest.mark.parametrize("parameters,word", [([], []), ([-1], []), ([True], []), ([0.5], []),
    ([1], [0]), ([1], [2]), ([1], [True]), ([1], [1.0]), ([1], "1")])
def test_invalid_word_contract(parameters, word):
    with pytest.raises((TypeError, ValueError)):
        resolvent_word(parameters, word)


def test_source_and_generator_contracts():
    r, ops = fixture()
    other, foreign = fixture()
    with pytest.raises(ValueError, match="one source"):
        ResolventGroup([ops[0], foreign[0]], [1, 1])
    with pytest.raises(ValueError, match="parameter"):
        ResolventGroup(ops, [1])
    from rexgraph.operator_bracket import operator_bracket
    with pytest.raises(TypeError, match="PSD"):
        ResolventGroup([operator_bracket(*ops)], [1])
    group = ResolventGroup(ops, [1, 1], [1])
    r.add_faces([[0]], [[1]])
    with pytest.raises(ValueError, match="changed"):
        group.apply(np.ones(2, dtype=int), exact=True)
    with pytest.raises(ValueError, match="changed"):
        _ = group.inverse


def test_complex_numeric_and_empty_block():
    _, ops = fixture()
    group = ResolventGroup(ops, [1, 2], [1, -2])
    x = np.array([1+2j, -3+4j])
    np.testing.assert_allclose(group.inverse.apply(group.apply(x)), x, atol=1e-9)
    assert group.apply(np.empty((2, 0), object), exact=True).shape == (2, 0)


def test_same_operator_resolvent_identity_over_q():
    _, ops = fixture()
    a, b = Q(2, 3), Q(5, 7)
    group = ResolventGroup([ops[0], ops[0]], [a, b])
    x = np.array([Q(1), Q(-2)], object)
    ra, rb = group.element([1]), group.element([2])
    left = ra.apply(x, exact=True) - rb.apply(x, exact=True)
    right = (b-a)*ra.apply(ops[0].apply(rb.apply(x, exact=True), exact=True), exact=True)
    np.testing.assert_array_equal(left, right)
