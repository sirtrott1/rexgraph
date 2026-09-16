"""Relative results use bounded rational previews, not source traversal."""
import json

import pytest

from rexgraph.cells import Cell
from rexgraph.graph import RexGraph
from rexgraph.relative_quotient import relative_quotient
from system.serialize import json_value


@pytest.mark.parametrize("limit", [0, 1, 256])
def test_result_preview_is_bounded_and_does_not_compute_homology(limit, monkeypatch):
    import rexgraph.graded_boundary as core
    rex = RexGraph.from_cells([4, [[0, 1, 2, 3]]])
    rex._agent_meta = {"secret": "not a quotient coordinate"}
    quotient = relative_quotient(Cell(rex, 0, 0))
    monkeypatch.setattr(core, "_rank_integer_columns", lambda *a: pytest.fail("rank executed"))
    result = json_value(quotient, max_values=limit)
    assert result["kind"] == "RelativeQuotient" and result["sizes"] == [3, 1]
    assert result["boundaries"][0]["nnz"] == 3
    assert len(result["boundaries"][0]["entries"]) == min(limit, 3)
    assert result["boundaries"][0]["entries_truncated"] == (limit < 3)
    if limit:
        assert result["boundaries"][0]["entries"][0][2] == {"numerator": 1, "denominator": 3}
    assert "secret" not in json.dumps(result, allow_nan=False)
    assert "source" not in result and "projection" not in result
    assert "_homology" not in quotient.__dict__
    rex.add_edges([0], [1])
    with pytest.raises(ValueError, match="changed"):
        json_value(quotient)
