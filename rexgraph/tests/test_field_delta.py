"""Exact correspondence defects compared with explicit independent matrices."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.chain_map import CoordinateComplex, GradedMap
from rexgraph.cochain import Chain, Cochain
from rexgraph.field_delta import field_delta, field_delta_moment
from rexgraph.graph import RexGraph
from rexgraph.type_accession import CoordinateSpace


def fixture():
    a = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]], [[(0, 1), (1, 1), (2, -1)]]])
    left = CoordinateComplex.from_rex(a)
    right = CoordinateComplex((CoordinateSpace("C0", ("a", "b", "c", "d")),
                               CoordinateSpace("C1", ("p", "q")), CoordinateSpace("C2", ("f",))),
                              (((0, 0, -1), (1, 0, 1), (1, 1, -1), (2, 1, 1)), ()))
    mapping = GradedMap(left, right, (((0, 0, 1), (1, 1, 1), (2, 2, 1)),
                                     ((0, 0, Q(1, 3)), (1, 1, 2)), ((0, 0, 1),)))
    return a, mapping


def dense(entries, shape):
    matrix = np.full(shape, Q(0), dtype=object)
    for i, j, v in entries:
        matrix[i, j] += v
    return matrix


@pytest.mark.parametrize("grade", [0, 1, 2])
@pytest.mark.parametrize("block", [False, True])
def test_formula_on_every_grade_with_rectangular_axes(grade, block):
    rex, j = fixture()
    a, b = j.domain, j.codomain
    x = np.arange(1, a.sizes[grade]+1, dtype=int)
    if block:
        x = np.column_stack((x, x*3))
    result = field_delta(Chain(grade, x, source=rex), j)
    jm = [dense(e, shape) for e, shape in zip(j.components, j.shapes, strict=True)]
    down = np.empty((0, *x.shape[1:]), dtype=object)
    up = down.copy()
    if grade:
        ba = dense(a.boundaries[grade-1], (a.sizes[grade-1], a.sizes[grade]))
        bb = dense(b.boundaries[grade-1], (b.sizes[grade-1], b.sizes[grade]))
        down = (bb @ jm[grade] - jm[grade-1] @ ba) @ x
    if grade+1 < len(a.sizes):
        ba = dense(a.boundaries[grade], (a.sizes[grade], a.sizes[grade+1]))
        bb = dense(b.boundaries[grade], (b.sizes[grade], b.sizes[grade+1]))
        up = (bb.T @ jm[grade] - jm[grade+1] @ ba.T) @ x
    np.testing.assert_array_equal(np.asarray(result["down"]["values"], dtype=object).reshape(down.shape), down)
    np.testing.assert_array_equal(np.asarray(result["up"]["values"], dtype=object).reshape(up.shape), up)
    qd, qu = sum(v*v for v in down.flat), sum(v*v for v in up.flat)
    assert result["moment"] == qd+qu and result["oriented_moment"] == qd-qu
    assert field_delta_moment(Chain(grade, x, source=rex), j, oriented=True) == qd-qu


def test_chain_map_does_not_imply_zero_coboundary_defect():
    rex = RexGraph.from_cells([2, [[0, 1]]])
    c = CoordinateComplex.from_rex(rex)
    j = GradedMap(c, c, (((0, 0, 1), (0, 1, 1)), ()))
    verified = j.verify()
    result = field_delta(Chain(0, np.array([1, 0]), source=rex), verified)
    assert result["down_quadrance"] == 0 and result["up_quadrance"] == 1
    assert result["moment"] == 1 and result["oriented_moment"] == -1


def test_full_upper_tower_and_large_exact_coefficients():
    face, difference = [(0, 1), (1, 1), (2, -1)], [(0, 1), (1, -1)]
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]],
                               [face, face], [difference, difference], [difference]])
    c = CoordinateComplex.from_rex(rex)
    j = GradedMap(c, c, tuple(tuple((i, i, Q(k+1, 7)) for i in range(n)) for k, n in enumerate(c.sizes)))
    large = 2**100 + 1
    result = field_delta(Chain(4, np.array([large], dtype=object), source=rex), j)
    assert result["down_quadrance"] == 2*Q(large, 7)**2
    assert result["up"]["shape"] == (0,) and result["up_quadrance"] == 0


def test_scalar_moment_does_not_construct_diagnostic_records(monkeypatch):
    import rexgraph.field_delta as core
    rex, j = fixture()
    x = Chain(1, np.ones(3, dtype=int), source=rex)
    expected = field_delta(x, j)["moment"]
    monkeypatch.setattr(core, "field_delta", lambda *a: pytest.fail("record constructed"))
    assert core.field_delta_moment(x, j) == expected


@pytest.mark.parametrize("kind", ["float", "cochain", "foreign", "keys", "bad-chain"])
def test_invalid_domains_are_not_coerced(kind):
    rex, j = fixture()
    x = Chain(1, np.ones(3, dtype=int), source=rex)
    if kind == "float":
        x = Chain(1, np.ones(3), source=rex)
    elif kind == "cochain":
        x = Cochain(1, np.ones(3, dtype=int), source=rex)
    elif kind == "foreign":
        x = Chain(1, np.ones(3, dtype=int), source=fixture()[0])
    elif kind == "keys":
        x = Chain(1, np.ones(3, dtype=int), source=rex, cell_keys=(2, 1, 0))
    else:
        target = CoordinateComplex(j.codomain.spaces, (j.codomain.boundaries[0], ((0, 0, 1),)))
        j = GradedMap(j.domain, target, j.components)
    with pytest.raises((TypeError, ValueError)):
        field_delta(x, j)
