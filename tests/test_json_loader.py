"""
Tests for rexgraph.io.json_loader -- JSON graph format loaders.

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
import os
import tempfile

import numpy as np
import pytest

from rexgraph.io.json_loader import (
    load_json,
    load_edge_list_json,
    load_cytoscape_json,
    load_networkx_json,
    load_adjacency_json,
    load_matrix_csv,
)
from rexgraph.graph import RexGraph


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
