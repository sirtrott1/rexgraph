"""Exact quotient dimensions across grades and primary arities."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.native_homology import _column_quotient, homology_split
from rexgraph.native_sparse import native_coo


@pytest.mark.parametrize("filled", [False, True])
@pytest.mark.parametrize("reversed_", [False, True])
def test_bigon_chain_and_homology_multiplicity_differ(filled, reversed_):
    other = [1, 0] if reversed_ else [0, 1]
    face = [[(0, 1), (1, 1 if reversed_ else -1)]] if filled else []
    rex = RexGraph.from_cells([2, [[0, 1], other], face])
    result = homology_split(rex, 1)
    assert (result.betti, result.simple, result.multiplicity) == (int(not filled), 0, int(not filled))
    assert result.chain_multiplicity == 1 and result.quotient_upper_rank == 0
    from rexgraph.harmonic_sparse import multiplicity_groups, simple_cycle_dimension
    # Existing explicit matrix quotient is an independent compatibility oracle.
    pytest.importorskip("scipy")
    assert simple_cycle_dimension(rex, groups=multiplicity_groups(rex)) == result.simple


def test_witnesses_branches_and_zero_columns_keep_their_meanings():
    rex = RexGraph.from_cells([4, [[0, 1, 2, 3], [0, 1, 2, 3], [2], [2], [0, 0], [0, 0]]])
    result = homology_split(rex, 1)
    assert (result.betti, result.simple, result.multiplicity, result.chain_multiplicity) == (4, 2, 2, 2)
    assert homology_split(rex, 0).simple == 2
    assert homology_split(rex, 0).multiplicity == 0
    with pytest.raises(ValueError, match="not present"):
        homology_split(rex, 2)


@pytest.mark.parametrize("filled", [False, True])
def test_duplicate_faces_and_filling_solid(filled):
    face = [(0, 1), (1, 1), (2, -1)]
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]], [face, face],
                               [[(0, 1), (1, -1)]] if filled else []])
    result = homology_split(rex, 2)
    assert result.chain_multiplicity == 1
    assert (result.simple, result.multiplicity) == (0, int(not filled))
    assert result.simple + result.multiplicity == rex.betti_tower[2]


def test_exact_equality_does_not_group_proportional_or_rounded_columns():
    columns = [{0: Q(1, 3)}, {0: Q(-1, 3)}, {0: Q(2, 3)},
               {0: Q(1, 3) + Q(1, 10**15)}, {}, {}]
    keep, projection = _column_quotient(columns)
    assert keep == [0, 2, 3, 4, 5]
    assert projection[:2] == [(0, 1), (0, -1)]


def test_full_tower_chain_failure_and_noninteger_upper_are_refused():
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]], [[(0, 1), (1, 1), (2, -1)]]])
    rex._graded_duals = [native_coo([0], [0], [1], (1, 1)).dual]
    with pytest.raises(ValueError, match="chain condition"):
        homology_split(rex, 0)
    rex._graded_duals = [native_coo([0], [0], [0.5], (1, 1)).dual]
    with pytest.raises(ValueError, match="integer higher"):
        homology_split(rex, 1)


@pytest.mark.parametrize("grade", [-1, 5, True, 1.0, "1", np.bool_(True)])
def test_invalid_grades(grade):
    rex = RexGraph.from_graph([0], [1])
    with pytest.raises((TypeError, ValueError)):
        homology_split(rex, grade)


def test_no_face_graphs_agree_with_cycle_rank_and_legacy_grouping():
    pytest.importorskip("scipy")
    from rexgraph.harmonic_sparse import multiplicity_dimension
    rng = np.random.default_rng(407)
    for _ in range(24):
        rex = RexGraph.from_cells([9, rng.integers(0, 9, (30, 2)).tolist()])
        result = homology_split(rex, 1)
        assert result.multiplicity == multiplicity_dimension(rex)
        assert result.simple + result.multiplicity == rex.betti[1]


def test_relation_metrics_do_not_define_the_quotient():
    rex = RexGraph(sources=[0, 1], targets=[1, 0], w_E=[Q(2, 3), Q(9, 7)])
    assert homology_split(rex).multiplicity == 1


def test_native_builder_matches_explicit_export_through_grade_four():
    pytest.importorskip("scipy")
    from rexgraph.graded_boundary import build_graded_boundaries, verify_chain
    from rexgraph.native_sparse import NativeSparse
    face = [(0, 1), (1, 1), (2, -1)]
    difference = [(0, 1), (1, -1)]
    cells = [3, [[0, 1], [1, 2], [0, 2]], [face, face], [difference, difference], [difference]]
    native = build_graded_boundaries(cells, native=True)
    oracle = build_graded_boundaries(cells)
    assert verify_chain(native) == (True, 0.0)
    for actual, expected in zip(native, oracle, strict=True):
        assert isinstance(actual, NativeSparse)
        np.testing.assert_array_equal(actual.as_scipy().toarray(), expected.toarray())
    rex = RexGraph.from_cells(cells)
    assert rex.betti_tower == (1, 0, 0, 0, 0)
    assert homology_split(rex, 3).multiplicity == 0


def test_native_numerical_chain_check_does_not_require_scipy():
    from rexgraph.graded_boundary import verify_chain
    lower = native_coo([0, 0], [0, 1], [0.5, -0.5], (1, 2))
    upper = native_coo([0, 1], [0, 0], [0.1, 0.1], (2, 1))
    assert verify_chain([lower, upper]) == (True, 0.0)


def test_rank_reduces_representatives_not_every_occurrence(monkeypatch):
    import rexgraph.native_homology as module
    original = module._rank_integer_columns
    calls = []

    def observed(shape, columns):
        calls.append(shape)
        return original(shape, columns)

    monkeypatch.setattr(module, "_rank_integer_columns", observed)
    rex = RexGraph.from_cells([4, [[0, 1, 2, 3]] * 1000])
    assert homology_split(rex).multiplicity == 999
    assert calls == [(4, 1)]
