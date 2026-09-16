"""Original cell closure and exact storage, without an optional matrix backend."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.cells import Cell, CellSet
from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest
from rexgraph.io.partition_state import build_rex_partition, faces_in_support, partition_tower
from rexgraph.io.rex_state import from_state, to_state
from rexgraph.native_sparse import csr_carrier, empty_native, restrict_carrier, sparse_arrays


def tower():
    face = [(0, 1), (1, 1), (2, -1)]
    difference = [(0, 1), (1, -1)]
    return RexGraph.from_cells([4, [[0, 1], [1, 2], [0, 2]],
        [face, face], [difference, difference], [difference]], relation_ids=[91, 17, 53])


def test_complete_closure_keeps_identities_isolation_metrics_and_maps():
    rex = tower()
    rex._w_E = np.array([Q(2, 3), Q(3, 4), Q(5, 6)], object)
    rex._signs = np.array([-1., 1., -1.])
    rex._w_boundary = {(0, 1): Q(7, 3)}
    rex._g_channel, rex._c_channel = "normalized", "count"
    rex._agent_meta = {"private": "not structural"}
    before = object_digest(rex)
    result = build_rex_partition(rex, [0, 0, 0], grade_masks={4: [1]}, v_mask=[0, 0, 0, 1])
    assert result.cell_maps == ((0, 1, 2, 3), (0, 1, 2), (0, 1), (0, 1), (0,))
    assert result.rex.relation_ids.tolist() == [91, 17, 53]
    assert result.rex._w_E.tolist() == rex._w_E.tolist()
    assert result.rex._signs.tolist() == rex._signs.tolist()
    assert result.rex._w_boundary == rex._w_boundary
    assert (result.rex.g_channel, result.rex.c_channel) == ("normalized", "count")
    assert result.rex.nV == 4 and not getattr(result.rex, "_agent_meta", {})
    assert object_digest(rex) == before
    restored = from_state(to_state(result.rex))
    assert object_digest(restored) == result.state.result_state
    assert restored.relation_ids.tolist() == [91, 17, 53]
    for original, actual in zip(partition_tower(rex)[1], partition_tower(restored)[1], strict=True):
        assert original == actual
    result.rex._w_E[0] = Q(9)
    assert rex._w_E[0] == Q(2, 3)
    with pytest.raises(ValueError, match="changed"):
        result.check_state()


def test_isolated_vertices_change_selection_digest_and_survive_roundtrip():
    rex = RexGraph.from_cells([4, [[0, 1]]])
    empty = build_rex_partition(rex, [0])
    vertex = build_rex_partition(rex, [0], v_mask=[0, 0, 0, 1])
    assert vertex.state.selection_digest != empty.state.selection_digest
    assert vertex.cell_maps == ((3,), ())
    assert (vertex.rex.nV, vertex.rex.nE) == (1, 0)
    assert from_state(to_state(vertex.rex)).betti_tower == (1, 0)


@pytest.mark.parametrize("support", [[0, 1, 2, 3], [2, 2], [3]])
def test_primary_slots_survive_restriction_without_pair_expansion(support):
    rex = RexGraph(boundary_ptr=[0, len(support)], boundary_idx=support, relation_ids=[73])
    result = build_rex_partition(rex, [1])
    mapping = {old: new for new, old in enumerate(sorted(set(support)))}
    assert result.rex._boundary_idx.tolist() == [mapping[i] for i in support]
    assert result.rex._boundary_ptr.tolist() == [0, len(support)]
    assert result.rex.relation_ids.tolist() == [73]
    assert result.cell_maps == (tuple(sorted(set(support))), (0,))


@pytest.mark.parametrize("mask", [[256], [-1], [0.5], [float("nan")], [float("inf")],
                                  ["1"], [[1]], 1, [1, 0], [1j]])
def test_mask_is_validated_before_coercion(mask):
    with pytest.raises(ValueError):
        build_rex_partition(RexGraph.from_graph([0], [1]), mask)


def test_higher_integer_storage_never_rounds_or_loses_empty_grades():
    rex = RexGraph.from_cells([2, [[0, 1]]])
    large = 2**60+1
    rex._graded_duals = [empty_native((0, 2)).dual,
        csr_carrier(np.array([0, 1, 2]), np.array([0, 0]), np.array([large, -large]), (2, 1))]
    result = build_rex_partition(rex, [0], grade_masks={4: [1]})
    assert result.cell_maps == ((), (), (), (0, 1), (0,))
    assert [b.shape for b in partition_tower(result.rex)[0]] == [(0, 0), (0, 0), (0, 2), (2, 1)]
    restored = from_state(to_state(result.rex))
    data = sparse_arrays(restored._graded_duals[1])[2]
    assert data.dtype == np.dtype("int64") and data.tolist() == [large, -large]


def test_faces_are_containment_not_intersection_or_inferred_cycles():
    rex = tower()
    assert faces_in_support(CellSet(rex, 1, (0, 1, 2))).indices == (0, 1)
    assert faces_in_support(CellSet(rex, 1, (0, 1))).indices == ()
    assert faces_in_support(Cell(rex, 1, 0)).indices == ()
    bare = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]]])
    assert faces_in_support(CellSet(bare, 1, (0, 1, 2))).indices == ()
    with pytest.raises(TypeError, match="C1"):
        faces_in_support(Cell(rex, 0, 0))
    zero = RexGraph(sources=[0], targets=[0], B2_col_ptr=[0, 1], B2_row_idx=[0], B2_vals=[1])
    zero._B2_vals[0] = 0
    assert faces_in_support(CellSet(zero, 1, ())).indices == ()


def test_restriction_refuses_raw_invalid_faces_instead_of_filtering():
    rex = RexGraph(sources=[0, 1], targets=[1, 2], B2_col_ptr=[0, 1], B2_row_idx=[0], B2_vals=[1])
    for read in (lambda: build_rex_partition(rex, [1, 1]),
                 lambda: faces_in_support(CellSet(rex, 1, (0, 1)))):
        with pytest.raises(ValueError, match="chain condition"):
            read()


def test_nonintegral_upper_map_is_not_rounded_into_a_certificate():
    rex = RexGraph(sources=[0], targets=[0], B2_col_ptr=[0, 1], B2_row_idx=[0], B2_vals=[0.5])
    with pytest.raises(ValueError, match="integer higher"):
        build_rex_partition(rex, [1])


def test_native_slice_retains_duplicate_addresses_and_original_integer_values():
    matrix = csr_carrier(np.array([0, 2, 3]), np.array([1, 1, 0]),
                         np.array([2**60+1, -2**60, 7]), (2, 2))
    result = restrict_carrier(matrix, [1, 0], [1, 0])
    ptr, idx, data, shape = sparse_arrays(result)
    assert shape == (2, 2) and ptr.tolist() == [0, 1, 3]
    assert idx.tolist() == [1, 0, 0] and data.tolist() == [7, 2**60+1, -2**60]
    for rows, cols in (([0, 0], [1]), ([0.5], [1]), ([0], [-1]), ([2], [0])):
        with pytest.raises(ValueError):
            restrict_carrier(matrix, rows, cols)
