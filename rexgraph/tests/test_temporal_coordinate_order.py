"""Mutation verification aligns explicit C1 identities across replay order."""
import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.io.mutation import (
    prepare_mutation, verify_mutation, apply_mutation, mutation_from_bytes,
    mutation_to_bytes, _structural_projection_matches,
)


def state(ids):
    columns = {10: [0, 1, 2], 20: [3, 4], 30: [5, 6], 90: [7, 8], 5: [9], 2: [10]}
    selected = [columns[i] for i in ids]
    return RexGraph(boundary_ptr=np.array([0, *np.cumsum([len(v) for v in selected])], dtype=np.int32),
                    boundary_idx=np.array([i for v in selected for i in v], dtype=np.int32),
                    relation_ids=np.array(ids, dtype=np.int64))


@pytest.mark.parametrize('ids', [[10,20,30,90,5,2],
    [10,90,20,30], [20,10,30], [10,30,90,5]])
def test_mutation_preserves_carried_coordinates_through_portable_state(ids):
    pytest.importorskip('safetensors')
    before, after = state([10,20,30]), state(ids)
    package = prepare_mutation(before, after, tx_time=1)
    package = mutation_from_bytes(mutation_to_bytes(package))
    assert verify_mutation(package, previous=before)
    restored = apply_mutation(package, previous=before)
    for name in ('relation_ids','_boundary_ptr','_boundary_idx'):
        np.testing.assert_array_equal(getattr(restored,name),getattr(after,name))


@pytest.mark.parametrize('ids', [[10,20,30,90,5], [20,10,30]])
def test_mutation_verification_accepts_valid_coordinate_changes(ids):
    before, after = state([10,20,30]), state(ids)
    package = prepare_mutation(before, after, tx_time=1)
    assert verify_mutation(package, previous=before)


def test_projection_rejects_different_identities_and_columns():
    original = state([10,20,30])
    changed = state([10,20,90])
    assert not _structural_projection_matches(original, changed)
    changed = state([20,10,30])
    changed._boundary_idx[0] = 0
    assert not _structural_projection_matches(original, changed)


def test_projection_transports_face_rows_and_attribution():
    def triangle(order, face_sign=1, weight=2):
        columns = [[0,1],[1,2],[0,2]]
        inverse = np.argsort(order)
        return RexGraph(boundary_ptr=np.array([0,2,4,6],dtype=np.int32),
            boundary_idx=np.array([v for i in order for v in columns[i]],dtype=np.int32),
            relation_ids=np.asarray([11,22,33])[order], w_E=np.asarray([weight,3,4])[order],
            signs=np.asarray([1,-1,1])[order], B2_col_ptr=np.array([0,3],dtype=np.int32),
            B2_row_idx=inverse.astype(np.int32), B2_vals=np.asarray([1,1,-1])*face_sign)
    original, reordered = triangle([0,1,2]), triangle([2,0,1])
    assert _structural_projection_matches(original, reordered)
    assert not _structural_projection_matches(original, triangle([2,0,1],face_sign=-1))
    assert not _structural_projection_matches(original, triangle([2,0,1],weight=5))


@pytest.mark.parametrize('reorder', [False, True])
def test_projection_does_not_round_exact_face_coefficients(reorder):
    from fractions import Fraction
    first = RexGraph(sources=[0,1,0], targets=[1,2,2], relation_ids=[1,2,3],
                     B2_col_ptr=[0,3], B2_row_idx=[0,1,2], B2_vals=[1,1,-1])
    second = RexGraph(sources=[0,0,1] if reorder else [0,1,0],
                      targets=[2,1,2] if reorder else [1,2,2],
                      relation_ids=[3,1,2] if reorder else [1,2,3],
                      B2_col_ptr=[0,3], B2_row_idx=[1,2,0] if reorder else [0,1,2],
                      B2_vals=[1,1,-1])
    exact = Fraction(2**54+1, 2**54)
    first._B2_vals = np.asarray([Fraction(1),Fraction(1),Fraction(-1)], dtype=object)
    second._B2_vals = np.asarray([exact,exact,-exact], dtype=object)
    assert not _structural_projection_matches(first, second)
