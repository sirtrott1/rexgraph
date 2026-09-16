"""Difference results expose bounded entries, not their live endpoint objects."""
import pytest
from rexgraph.graph import RexGraph
from rexgraph.tensor_diff import difference_tensor
from system.serialize import json_value


def test_difference_result_has_a_bounded_exact_entry_preview():
    a = RexGraph.from_cells([3, [[0, 1, 2]]], relation_ids=[7])
    b = RexGraph.from_cells([3, [[1, 0, 2]]], relation_ids=[7])
    d, _ = difference_tensor(a, b)
    result = json_value(d, max_values=1)
    assert result["kind"] == "BoundaryDifference" and result["shape"] == [3, 1]
    assert result["nnz"] == 2 and result["entries_truncated"] is True
    assert result["entries"] == [[0, 0, {"numerator": 3, "denominator": 2}]]
    assert "reference" not in result and "other" not in result
    a.set_cell_attrs([0], w_E=[2])
    with pytest.raises(ValueError, match="changed"):
        json_value(d)
