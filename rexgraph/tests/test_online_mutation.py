"""The online field cache follows the current native boundary after public edits."""
import numpy as np
import pytest
from rexgraph.graph import RexGraph
from rexgraph.flow.online import GreensCochainField


@pytest.mark.parametrize("mutation", ["append", "replace"])
def test_online_operator_refreshes_after_graph_mutation(mutation):
    graph = RexGraph(sources=np.array([0, 0], np.int32), targets=np.array([1, 2], np.int32))
    field = GreensCochainField()
    field.predict(graph, [0, 1])
    if mutation == "append":
        graph.add_edges([1], [2])
    else:
        graph.remove_edges(np.array([False, True]))
        graph.add_edges([2], [3])
    graph._ensure_clean()
    region = np.arange(graph.nE)
    target = np.arange(graph.nE, dtype=float) + 1
    result = field.correct(graph, region, target)
    fresh = GreensCochainField()
    fresh.correct(graph, region, target)
    np.testing.assert_allclose(field._phi_vec(field._keys(graph)), fresh._phi_vec(fresh._keys(graph)))
    assert result["updated"]
