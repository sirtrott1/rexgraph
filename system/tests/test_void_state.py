"""Bounded void previews do not traverse the source or trigger lazy ranks."""
import json

from rexgraph.cells import CellSet
from rexgraph.graph import RexGraph
from rexgraph.void_state import void_state, VoidState
from system.serialize import json_value


def test_bounded_void_preview_does_not_compute_homology(monkeypatch):
    rex = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2], [0, 2]]])
    value = void_state(CellSet(rex, 1, range(rex.nE)))
    monkeypatch.setattr(VoidState, "homology", property(lambda self: (_ for _ in ()).throw(AssertionError("rank ran"))))
    result = json.loads(json.dumps(json_value(value, max_values=1)))
    assert result["n_voids"] == result["n_potential"] == 2 and result["strain"] == 6
    assert len(result["columns"]) == len(result["potential"]) == len(result["region"]) == 1
    assert result["columns_truncated"] and result["region_truncated"]
    assert result["homology"] == "query explicitly" and "source" not in result
