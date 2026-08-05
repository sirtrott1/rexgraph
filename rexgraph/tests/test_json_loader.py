"""
Tests for rexgraph.io.json_loader: JSON graph format loaders.

No heavy dependencies. Tests use temporary JSON files.

Verifies:
    - Format auto-detection: rexgraph, edge_list, cytoscape, networkx, adjacency
    - Edge list: string vertex names mapped, correct nV/nE
    - Cytoscape: nodes/edges parsed, weights extracted
    - NetworkX: node-link parsed
    - Adjacency: matrix -> RexGraph, threshold filtering
    - Matrix CSV: square matrix loaded, threshold and sign handling
    - load_json dispatches correctly
"""
import json

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.json_loader import (
    load_adjacency_json,
    load_cytoscape_json,
    load_edge_list_json,
    load_json,
    load_matrix_csv,
    load_networkx_json,
)

# Helpers

def _write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def _write_csv_matrix(matrix, path, labels=None):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if labels:
            writer.writerow([""] + labels)
        for i, row in enumerate(matrix):
            prefix = [labels[i]] if labels else []
            writer.writerow(prefix + [str(x) for x in row])


# Edge List JSON

class TestEdgeListJSON:

    def test_basic(self, tmp_path):
        data = {"edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "A", "target": "C"},
        ]}
        path = str(tmp_path / "edges.json")
        _write_json(data, path)
        rex = load_edge_list_json(path)
        assert isinstance(rex, RexGraph)
        assert rex.nV == 3
        assert rex.nE == 3

    def test_bare_list(self, tmp_path):
        data = [
            {"source": "X", "target": "Y"},
            {"source": "Y", "target": "Z"},
        ]
        path = str(tmp_path / "edges.json")
        _write_json(data, path)
        rex = load_json(path)  # auto-detect
        assert rex.nE == 2

    def test_with_weight(self, tmp_path):
        data = {"edges": [
            {"source": "A", "target": "B", "weight": "2.5"},
            {"source": "B", "target": "C", "weight": "1.0"},
        ]}
        path = str(tmp_path / "edges.json")
        _write_json(data, path)
        rex = load_edge_list_json(path)
        assert rex.nE == 2


# Cytoscape JSON

class TestCytoscapeJSON:

    def test_dict_format(self, tmp_path):
        data = {
            "elements": {
                "nodes": [
                    {"data": {"id": "n1"}},
                    {"data": {"id": "n2"}},
                    {"data": {"id": "n3"}},
                ],
                "edges": [
                    {"data": {"source": "n1", "target": "n2"}},
                    {"data": {"source": "n2", "target": "n3"}},
                ],
            }
        }
        path = str(tmp_path / "cyto.json")
        _write_json(data, path)
        rex = load_cytoscape_json(path)
        assert rex.nV == 3
        assert rex.nE == 2

    def test_flat_format(self, tmp_path):
        data = {
            "elements": [
                {"group": "nodes", "data": {"id": "a"}},
                {"group": "nodes", "data": {"id": "b"}},
                {"group": "edges", "data": {"source": "a", "target": "b"}},
            ]
        }
        path = str(tmp_path / "cyto.json")
        _write_json(data, path)
        rex = load_cytoscape_json(path)
        assert rex.nV == 2
        assert rex.nE == 1

    def test_weighted_edges(self, tmp_path):
        data = {
            "elements": {
                "nodes": [{"data": {"id": "a"}}, {"data": {"id": "b"}}],
                "edges": [{"data": {"source": "a", "target": "b", "weight": -2.0}}],
            }
        }
        path = str(tmp_path / "cyto.json")
        _write_json(data, path)
        rex = load_cytoscape_json(path)
        assert rex.nE == 1


# NetworkX JSON

class TestNetworkXJSON:

    def test_basic(self, tmp_path):
        data = {
            "nodes": [{"id": 0}, {"id": 1}, {"id": 2}],
            "links": [
                {"source": 0, "target": 1},
                {"source": 1, "target": 2},
            ],
        }
        path = str(tmp_path / "nx.json")
        _write_json(data, path)
        rex = load_networkx_json(path)
        assert rex.nV == 3
        assert rex.nE == 2

    def test_auto_detect(self, tmp_path):
        data = {
            "nodes": [{"id": "x"}, {"id": "y"}],
            "links": [{"source": "x", "target": "y"}],
        }
        path = str(tmp_path / "nx.json")
        _write_json(data, path)
        rex = load_json(path)  # auto-detect as networkx
        assert rex.nE == 1


# Adjacency JSON

class TestAdjacencyJSON:

    def test_matrix_key(self, tmp_path):
        data = {"matrix": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]}
        path = str(tmp_path / "adj.json")
        _write_json(data, path)
        rex = load_adjacency_json(path)
        assert rex.nV == 3
        assert rex.nE == 2  # upper triangle: (0,1), (1,2)

    def test_bare_list(self, tmp_path):
        data = [[0, 1], [1, 0]]
        path = str(tmp_path / "adj.json")
        _write_json(data, path)
        rex = load_json(path)  # auto-detect
        assert rex.nV == 2
        assert rex.nE == 1

    def test_non_square_raises(self, tmp_path):
        data = {"matrix": [[0, 1, 0], [1, 0, 1]]}
        path = str(tmp_path / "adj.json")
        _write_json(data, path)
        with pytest.raises(ValueError, match="square"):
            load_adjacency_json(path)


# Matrix CSV

class TestMatrixCSV:

    def test_basic(self, tmp_path):
        path = str(tmp_path / "matrix.csv")
        _write_csv_matrix(
            [[0, 0.5, 0.3], [0.5, 0, 0.8], [0.3, 0.8, 0]],
            path, labels=["g1", "g2", "g3"])
        rex = load_matrix_csv(path)
        assert rex.nV == 3
        assert rex.nE == 3  # all 3 upper-triangle entries nonzero

    def test_threshold(self, tmp_path):
        path = str(tmp_path / "matrix.csv")
        _write_csv_matrix(
            [[0, 0.1, 0.9], [0.1, 0, 0.2], [0.9, 0.2, 0]],
            path)
        rex = load_matrix_csv(path, threshold=0.5)
        # Only (0,2) with weight 0.9 survives
        assert rex.nE == 1

    def test_empty_after_threshold(self, tmp_path):
        path = str(tmp_path / "matrix.csv")
        _write_csv_matrix([[0, 0.1], [0.1, 0]], path)
        rex = load_matrix_csv(path, threshold=1.0)
        assert rex.nE == 0


# Auto-detection

class TestAutoDetect:

    def test_rexgraph_format(self, tmp_path):
        """boundary_ptr key -> rexgraph format."""
        data = {
            "boundary_ptr": [0, 2, 4],
            "boundary_idx": [0, 1, 1, 2],
        }
        path = str(tmp_path / "rex.json")
        _write_json(data, path)
        rex = load_json(path)
        assert isinstance(rex, RexGraph)

    def test_unknown_raises(self, tmp_path):
        data = {"random_key": 42}
        path = str(tmp_path / "unknown.json")
        _write_json(data, path)
        with pytest.raises(ValueError):
            load_json(path)


# Signed-topology round-trip fidelity (Wave-0 correctness)

def _signed_directed_faced_graph():
    """A signed, directed 2-rex with a filled face (negative orientation
    entry in B2) AND a branching edge (>2 boundary vertices).

    This exercises every field that defines the signed complex:
    boundary_ptr/idx, edge signs, B2_col_ptr/row_idx/B2_vals, directed.
    """
    boundary_ptr = np.array([0, 2, 4, 6, 9], dtype=np.int32)
    # e0={0,1} e1={0,2} e2={1,2} e3={1,2,3} (branching)
    boundary_idx = np.array([0, 1, 0, 2, 1, 2, 1, 2, 3], dtype=np.int32)
    B2_col_ptr = np.array([0, 3], dtype=np.int32)
    B2_row_idx = np.array([0, 1, 2], dtype=np.int32)
    B2_vals = np.array([1.0, -1.0, 1.0], dtype=np.float64)  # negative orientation
    signs = np.array([1.0, -1.0, 1.0, 1.0], dtype=np.float64)  # negative sign
    return RexGraph(
        boundary_ptr=boundary_ptr, boundary_idx=boundary_idx,
        B2_col_ptr=B2_col_ptr, B2_row_idx=B2_row_idx, B2_vals=B2_vals,
        signs=signs, directed=True,
    )


class TestSignedTopologyRoundtrip:
    """The signed complex must survive a native-JSON round-trip EXACTLY."""

    def _assert_same(self, rex, rex2):
        assert np.array_equal(rex2._boundary_ptr, rex._boundary_ptr)
        assert np.array_equal(rex2._boundary_idx, rex._boundary_idx)
        assert np.array_equal(rex2._B2_col_ptr, rex._B2_col_ptr)
        assert np.array_equal(rex2._B2_row_idx, rex._B2_row_idx)
        # B2 orientation signs (including the negative) must survive.
        assert np.allclose(rex2._B2_vals, rex._B2_vals)
        assert np.any(rex2._B2_vals < 0)
        # Edge signs (including the negative) must survive.
        assert rex2._signs is not None
        assert np.allclose(np.asarray(rex2._signs), np.asarray(rex._signs))
        assert np.any(np.asarray(rex2._signs) < 0)
        # Directedness must survive.
        assert rex2._directed == rex._directed is True

    def test_to_json_dict_roundtrip(self):
        from rexgraph.io.json_loader import _load_rexgraph_json
        rex = _signed_directed_faced_graph()
        rex2 = _load_rexgraph_json(rex.to_json())
        self._assert_same(rex, rex2)

    def test_to_json_file_roundtrip(self, tmp_path):
        rex = _signed_directed_faced_graph()
        path = str(tmp_path / "signed.json")
        _write_json(rex.to_json(), path)
        rex2 = load_json(path)
        self._assert_same(rex, rex2)

    def test_io_save_load_json_roundtrip(self, tmp_path):
        from rexgraph import io
        rex = _signed_directed_faced_graph()
        path = str(tmp_path / "signed_io.json")
        io.save(path, rex, format="json")
        rex2 = io.load(path, format="json")
        self._assert_same(rex, rex2)

    def test_w_boundary_roundtrip(self):
        from rexgraph.io.json_loader import _load_rexgraph_json
        boundary_ptr = np.array([0, 2, 4], dtype=np.int32)
        boundary_idx = np.array([0, 1, 1, 2], dtype=np.int32)
        rex = RexGraph(
            boundary_ptr=boundary_ptr, boundary_idx=boundary_idx,
            w_boundary={(0, 0): 0.7, (1, 2): 0.3},
        )
        rex2 = _load_rexgraph_json(rex.to_json())
        assert rex2._w_boundary == rex._w_boundary


class TestAdjacencyThreshold:
    """_load_adjacency_json must honor its threshold param (like load_matrix_csv)."""

    def test_threshold_prunes_edges(self, tmp_path):
        # |w| <= threshold excluded; only the 0.9 edge survives at 0.5.
        data = {"matrix": [[0, 0.1, 0.9], [0.1, 0, 0.2], [0.9, 0.2, 0]]}
        path = str(tmp_path / "adj.json")
        _write_json(data, path)
        rex = load_json(path, threshold=0.5)
        assert rex.nE == 1

    def test_threshold_default_keeps_all(self, tmp_path):
        data = {"matrix": [[0, 0.1, 0.9], [0.1, 0, 0.2], [0.9, 0.2, 0]]}
        path = str(tmp_path / "adj.json")
        _write_json(data, path)
        rex = load_json(path)
        assert rex.nE == 3
