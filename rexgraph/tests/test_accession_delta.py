"""Exact sparse accession differences against independent matrix arithmetic."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.accession_delta import accession_delta
from rexgraph.chain_map import CoordinateComplex, GradedMap
from rexgraph.graph import RexGraph
from rexgraph.type_accession import CoordinateSpace, TypeAccession


def fixture(grade=1, rectangular=False):
    left = RexGraph.from_cells([3, [[0, 1, 2], [0]], []])
    right = RexGraph.from_cells([4, [[0, 1], [1, 2], [1, 3]], []])
    a, b = CoordinateComplex.from_rex(left), CoordinateComplex.from_rex(right)
    mapping = GradedMap(a, b, (((0, 0, 1), (1, 1, 2), (3, 2, 1)),
                              ((0, 0, Q(1, 3)), (1, 1, 2), (2, 0, -1))))
    coordinates = CoordinateSpace("measurement", ("a", "b")) if rectangular else None
    def entries(n):
        rows = 2 if rectangular else n
        return tuple((i, j, Q(i + j + 1, 7)) for i in range(rows) for j in range(n))
    old = TypeAccession(left, grade, "old", entries(a.sizes[grade]), coordinates=coordinates)
    new = TypeAccession(right, grade, "new", entries(b.sizes[grade]), coordinates=coordinates)
    return old, new, mapping


def matrix(entries, shape):
    value = np.full(shape, Q(0), dtype=object)
    for i, j, x in entries:
        value[i, j] += x
    return value


@pytest.mark.parametrize("grade", [0, 1])
@pytest.mark.parametrize("rectangular", [False, True])
def test_exact_formula_rectangular_ambient_and_empty_axes(grade, rectangular):
    old, new, j = fixture(grade, rectangular)
    result = accession_delta(old, new, j)
    jm = matrix(j.components[grade], j.shapes[grade])
    a, b = matrix(old.entries, old.shape), matrix(new.entries, new.shape)
    expected = b @ jm - (a if rectangular else jm @ a)
    np.testing.assert_array_equal(matrix(result["entries"], result["shape"]), expected)
    assert result["coefficient_domain"] == "Q" and all(isinstance(v, Q) for _, _, v in result["entries"])
    assert result["implicit_zero"] == 0


def test_large_rationals_cancellation_reuses_core_sparse_composition(monkeypatch):
    import rexgraph.accession_delta as core
    calls = []
    compose = core._exact_compose_columns
    def counted(*args):
        calls.append(True)
        return compose(*args)
    monkeypatch.setattr(core, "_exact_compose_columns", counted)
    old, _, _ = fixture()
    rex = old.source
    c = CoordinateComplex.from_rex(rex)
    j = GradedMap(c, c, tuple(tuple((i, i, 1) for i in range(n)) for n in c.sizes))
    a = TypeAccession(rex, 1, "a", ((0, 0, Q(2**140 + 1, 17)),))
    b = replace(a, entries=(*a.entries, (1, 0, Q(1, 2**100))))
    assert accession_delta(a, a, j)["entries"] == ()
    assert accession_delta(a, b, j)["entries"] == ((1, 0, Q(1, 2**100)),)
    assert len(calls) == 4


def test_empty_measurement_axes_preserve_shape():
    rex = RexGraph.from_cells([3, []])
    c = CoordinateComplex.from_rex(rex)
    j = GradedMap(c, c, (((0, 0, 1), (1, 1, 1), (2, 2, 1)), ()))
    for coordinates in (None, CoordinateSpace("two readings", ("a", "b"))):
        a = TypeAccession(rex, 1, "empty", (), coordinates=coordinates)
        result = accession_delta(a, a, j)
        assert result["shape"] == ((0 if coordinates is None else 2), 0)
        assert result["entries"] == ()


@pytest.mark.parametrize("kind", ["float", "keys", "grade", "foreign", "stale", "mixed", "coordinate-order"])
def test_incompatible_measurements_are_refused(kind):
    old, new, j = fixture()
    if kind == "float":
        old = replace(old, entries=((0, 0, 0.0),))
    elif kind == "keys":
        old = replace(old, cell_keys=(0, 1))
    elif kind == "grade":
        new = fixture(0)[1]
    elif kind == "foreign":
        new = fixture()[1]
    elif kind == "stale":
        old.source.add_edges([0], [1])
    elif kind == "mixed":
        old = replace(old, coordinates=CoordinateSpace("x", ("a", "b")))
    else:
        old, new, j = fixture(rectangular=True)
        new = replace(new, coordinates=CoordinateSpace("measurement", ("b", "a")))
    with pytest.raises((TypeError, ValueError)):
        accession_delta(old, new, j)


def test_full_higher_grade_tower_and_explicit_chain_map():
    face, diff = [(0, 1), (1, 1), (2, -1)], [(0, 1), (1, -1)]
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]], [face, face], [diff, diff], [diff]])
    c = CoordinateComplex.from_rex(rex)
    j = GradedMap(c, c, tuple(tuple((i, i, 1) for i in range(n)) for n in c.sizes)).verify()
    a = TypeAccession(rex, 4, "old", ((0, 0, 2),))
    b = TypeAccession(rex, 4, "new", ((0, 0, 5),))
    assert accession_delta(a, b, j)["entries"] == ((0, 0, Q(3)),)
