"""Primary differences retain exact shares and every union coordinate."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.tensor_diff import difference_tensor


def rex(cells, ids=None, vertices=4):
    return RexGraph.from_cells([vertices, cells], relation_ids=ids)


def test_witness_birth_loop_and_new_vertex_are_not_dropped():
    before = rex([], vertices=1)
    after = rex([[0], [0, 0], [0, 1]], vertices=2)
    d, r = difference_tensor(before, after)
    assert d.shape == (2, 3) and d.vertex_keys == (0, 1)
    assert d.relation_pairs == ((None, 0), (None, 1), (None, 2))
    assert d.entries == ((0, 0, Q(1)), (0, 2, Q(-1)), (1, 2, Q(1)))
    assert r["max_column_sum"] == 1 and r["rank"] == 2 and r["nullity"] == 1
    assert r["unmapped_relations"] == 0 and r["only_in_input"] == 3
    np.testing.assert_array_equal(d.apply(np.array([2, 3, 5])), [-3, 5])
    np.testing.assert_array_equal(d.transpose_apply(np.array([2, 5])), [2, 0, 3])
    np.testing.assert_allclose(d.apply(np.array([2, 3, 5]), exact=False), [-3, 5])


def test_identity_alignment_keeps_structural_change_on_one_column():
    a = rex([[0, 1, 2, 3], [0, 1]], [11, 37])
    b = rex([[0, 1], [1, 0, 2, 3]], [37, 11])
    d, r = difference_tensor(a, b)
    assert d.matching == "identity" and d.relation_pairs == ((0, 1), (1, 0))
    assert d.entries == ((0, 0, Q(4, 3)), (1, 0, Q(-4, 3)))
    assert r["frobenius2"] == Q(32, 9) and r["rank"] == 1
    block = np.array([[3, 6], [9, 1]])
    np.testing.assert_array_equal(d.apply(block), [[4, 8], [-4, -8], [0, 0], [0, 0]])


def test_support_matching_preserves_multiplicity_and_matches_identical_first():
    a = rex([[0, 1], [0, 1], [0], [0, 0]])
    b = rex([[1, 0], [0, 1], [0, 0], [0]])
    d, r = difference_tensor(a, b)
    assert r["agree"] == 3 and r["disagree"] == 1 and len(d.relation_pairs) == 4
    assert r["only_in_input"] == r["only_in_reference"] == 0
    witness_loop, reading = difference_tensor(rex([[0]]), rex([[0, 0]]))
    assert witness_loop.shape[1] == 2 and reading["in_both"] == 0


def test_vertex_labels_form_an_injective_complete_union():
    a, b = rex([[0, 1]], vertices=2), rex([[1, 0], [0, 2]], vertices=3)
    d, r = difference_tensor(a, b, ref_labels=["a", "b"], inp_labels=["b", "a", "c"])
    assert d.vertex_keys == ("a", "b", "c") and r["agree"] == 1 and r["only_in_input"] == 1
    np.testing.assert_array_equal(d.apply(np.ones(2, dtype=int)), [0, -1, 1])


@pytest.mark.parametrize("labels", [(None, ["a"]), (["a", "a"], ["a"]),
    ([1, 2], ["a"]), (["a"], ["a"]), ("ab", ["a"])])
def test_bad_vertex_correspondences_refused(labels):
    with pytest.raises((TypeError, ValueError)):
        difference_tensor(rex([[0, 1]], vertices=2), rex([[0]], vertices=1),
                          ref_labels=labels[0], inp_labels=labels[1])


def test_identity_demotion_is_explicit_and_captured_states_cannot_drift():
    a, b = rex([[0, 1]], [9]), rex([[0, 1]])
    with pytest.raises(ValueError, match="support"):
        difference_tensor(a, b)
    d, _ = difference_tensor(a, b, matching="support")
    a.set_cell_attrs([0], w_E=[2])
    with pytest.raises(ValueError, match="changed"):
        d.apply(np.ones(1, dtype=int))
    with pytest.raises(ValueError, match="changed"):
        _ = d.readings


def test_empty_exact_difference_has_correct_shape_and_zero_rank():
    d, r = difference_tensor(rex([], vertices=0), rex([], vertices=3))
    assert d.shape == (3, 0) and d.entries == () and r["rank"] == r["nullity"] == 0
    np.testing.assert_array_equal(d.apply(np.empty(0, dtype=int)), [0, 0, 0])
