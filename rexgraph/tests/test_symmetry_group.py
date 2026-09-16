"""Exact chain symmetries and ordered products without matrix inverses."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.chain_map import CoordinateComplex, GradedMap, SymmetryGroup


def identity(c):
    return GradedMap(c, c, tuple(tuple((i, i, 1) for i in range(n)) for n in c.sizes))


def path():
    r = RexGraph.from_cells([3, [[0, 1], [1, 2]]], relation_ids=[17, 91], w_E=[2, 3])
    c = CoordinateComplex.from_rex(r)
    p = GradedMap(c, c, (((2, 0, 1), (1, 1, 1), (0, 2, 1)), ((1, 0, -1), (0, 1, -1))))
    return r, c, p


def matrix(p, grade):
    a = np.zeros(p.shapes[grade], object)
    for i, j, value in p.components[grade]:
        a[i, j] = value
    return a


def test_path_reflection_preserves_boundary_not_arbitrary_metric():
    _, c, p = path()
    group = SymmetryGroup([p], [1])
    assert group.map.declaration.components == p.components
    assert group.map.commutation_residuals == (0,)
    assert group.inverse.map.declaration.components == p.components
    assert group.element([1, 1]).map.declaration.components == identity(c).components
    # The boundary certificate does not silently include a different metric.
    u = matrix(p, 1)
    metric = np.diag(np.array([2, 3], object))
    assert not np.array_equal(u.T @ metric @ u, metric)
    assert group.word == (1,) and group.generator_count == 1


def test_noncommuting_products_and_inverse_exact_rational():
    rex = RexGraph.from_cells([2, []])
    c = CoordinateComplex.from_rex(rex)
    a = GradedMap(c, c, (((0, 0, Q(3, 5)), (0, 1, Q(-4, 5)),
                          (1, 0, Q(4, 5)), (1, 1, Q(3, 5))), ()))
    b = GradedMap(c, c, (((0, 0, 1), (1, 1, -1)), ()))
    g = SymmetryGroup([a, b], [1, 2])
    assert not np.array_equal(matrix(a, 0) @ matrix(b, 0), matrix(b, 0) @ matrix(a, 0))
    np.testing.assert_array_equal(matrix(g.map.declaration, 0), matrix(a, 0) @ matrix(b, 0))
    assert g.inverse.word == (-2, -1)
    assert g.map.then(g.inverse.map).declaration.components == identity(c).components
    assert g.identity.map.declaration.components == identity(c).components


def test_arbitrary_grade_branching_witness_and_empty_components():
    face = [(0, 1), (1, 1), (2, -1)]
    d = [(0, 1), (1, -1)]
    high = RexGraph.from_cells([4, [[0, 1], [1, 2], [0, 2]], [face, face], [d, d], [d]])
    branch = RexGraph.from_cells([5, [[0, 1, 2, 3], [4], [2, 2]]])
    for rex in (high, branch, RexGraph.from_cells([0, []])):
        c = CoordinateComplex.from_rex(rex)
        p = GradedMap(c, c, tuple(tuple((i, i, -1) for i in range(n)) for n in c.sizes))
        g = SymmetryGroup([p], [1, -1, 1])
        assert g.word == (1,)
        assert g.map.declaration.components == p.components
        assert g.sizes == c.sizes


@pytest.mark.parametrize("word", [[0], [2], [True], [1.0], "1", [np.bool_(True)]])
def test_invalid_words(word):
    _, _, p = path()
    with pytest.raises((ValueError, TypeError)):
        SymmetryGroup([p], word)


def test_reject_wrong_source_spaces_nonorthogonal_and_nonchain_maps():
    rex, c, p = path()
    foreign = CoordinateComplex.from_rex(rex)
    wrong = GradedMap(c, foreign, identity(c).components)
    scale = GradedMap(c, c, tuple(tuple((i, j, 2*v) for i, j, v in e) for e in identity(c).components))
    not_chain = GradedMap(c, c, (p.components[0], identity(c).components[1]))
    for generators in ([], [1], [wrong], [p, identity(foreign)], [scale], [not_chain]):
        with pytest.raises((ValueError, TypeError)):
            SymmetryGroup(generators)


def test_invalid_original_chain_is_rejected_even_for_identity():
    r = RexGraph(sources=[0, 1], targets=[1, 2], B2_col_ptr=[0, 1], B2_row_idx=[0], B2_vals=[1])
    c = CoordinateComplex.from_rex(r)
    with pytest.raises(ValueError, match="chain law"):
        SymmetryGroup([identity(c)])


def test_chain_capture_preserves_large_upper_integers():
    from rexgraph.native_sparse import csr_carrier, empty_native
    r = RexGraph.from_cells([0, []])
    large = 2**60+1
    r._graded_duals = [empty_native((0, 2)).dual,
        csr_carrier(np.array([0, 1, 2]), np.array([0, 0]), np.array([large, -large]), (2, 1))]
    c = CoordinateComplex.from_rex(r)
    assert c.sizes == (0, 0, 0, 2, 1)
    assert c.boundaries[-1] == ((0, 0, Q(large)), (1, 0, Q(-large)))
    assert SymmetryGroup([identity(c)], [1]).map.commutation_residuals == (0, 0, 0, 0)


def test_no_product_construction_until_map(monkeypatch):
    _, c, p = path()
    monkeypatch.setattr(GradedMap, "then", lambda *a: pytest.fail("product materialized"))
    g = SymmetryGroup([p], [1])
    assert g.inverse.word == (-1,) and g.identity.word == () and g.sizes == c.sizes


def test_stale_boundary_state_refused():
    rex, _, p = path()
    g = SymmetryGroup([p], [1])
    rex.add_edges([0], [2], relation_ids=[101])
    for read in (lambda: g.map, lambda: g.inverse, lambda: g.identity, lambda: g.sizes):
        with pytest.raises(ValueError, match="changed"):
            read()
