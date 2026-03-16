"""
IO layer tests for rexgraph.

Tests data ingest (CSV, JSON), persistent storage (bundle),
format dispatch, and round-trip fidelity.

Tier 1 (always runs): csv_loader, json_loader, bundle
Tier 2 (skip if missing): zarr, hdf5, arrow, parquet, sql
"""
import json
import os
import tempfile

import numpy as np
import pytest

from rexgraph.graph import RexGraph


# Helpers

def _make_k4():
    """K4 simplicial complex for round-trip tests."""
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32),
    )


def _make_triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


def _write_csv(path, header, rows):
    """Write a CSV file from header + row data."""
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


# CSV Loader

class TestCSVLoader:
    """Test csv_loader column classification, weight construction, and to_rex."""

    def test_basic_load(self, tmp_path):
        csv_path = str(tmp_path / "edges.csv")
        _write_csv(csv_path, ["source", "target"], [
            ["A", "B"], ["B", "C"], ["A", "C"],
        ])
        from rexgraph.io.csv_loader import load_edge_csv
        gd = load_edge_csv(csv_path)
        assert gd.nV == 3
        assert gd.nE == 3
        assert set(gd.vertices) == {"A", "B", "C"}

    def test_column_classification_type(self, tmp_path):
        csv_path = str(tmp_path / "edges.csv")
        _write_csv(csv_path,
            ["source", "target", "interaction_type"],
            [["A", "B", "binding"], ["B", "C", "phosphorylation"],
             ["A", "C", "binding"]],
        )
        from rexgraph.io.csv_loader import load_edge_csv, ColumnRole
        gd = load_edge_csv(csv_path)
        assert gd.profiles["interaction_type"].role == ColumnRole.TYPE

    def test_column_classification_polarity(self, tmp_path):
        csv_path = str(tmp_path / "edges.csv")
        _write_csv(csv_path,
            ["source", "target", "effect"],
            [["EGFR", "RAS", "stimulation"],
             ["RAS", "RAF", "activation"],
             ["P53", "MDM2", "inhibition"]],
        )
        from rexgraph.io.csv_loader import load_edge_csv, ColumnRole
        gd = load_edge_csv(csv_path)
        assert gd.profiles["effect"].role == ColumnRole.POLARITY
        assert "inhibition" in gd.negative_types

    def test_polarity_signs(self, tmp_path):
        csv_path = str(tmp_path / "edges.csv")
        _write_csv(csv_path,
            ["source", "target", "effect"],
            [["A", "B", "activation"],
             ["B", "C", "inhibition"],
             ["A", "C", "activation"]],
        )
        from rexgraph.io.csv_loader import load_edge_csv
        gd = load_edge_csv(csv_path)
        # Inhibition edge should have negative weight
        assert gd.w_E[1] < 0
        assert gd.w_E[0] > 0
        assert gd.w_E[2] > 0

    def test_numeric_weight(self, tmp_path):
        csv_path = str(tmp_path / "edges.csv")
        _write_csv(csv_path,
            ["source", "target", "score"],
            [["A", "B", "0.9"], ["B", "C", "0.5"], ["A", "C", "0.1"]],
        )
        from rexgraph.io.csv_loader import load_edge_csv, ColumnRole
        gd = load_edge_csv(csv_path)
        assert gd.profiles["score"].role == ColumnRole.NUMERIC
        assert abs(gd.w_E[0] - 0.9) < 1e-10
        assert abs(gd.w_E[1] - 0.5) < 1e-10

    def test_ordinal_scaling(self, tmp_path):
        csv_path = str(tmp_path / "edges.csv")
        _write_csv(csv_path,
            ["source", "target", "confidence"],
            [["A", "B", "high"], ["B", "C", "low"], ["A", "C", "medium"]],
        )
        from rexgraph.io.csv_loader import load_edge_csv, ColumnRole
        gd = load_edge_csv(csv_path)
        assert gd.profiles["confidence"].role == ColumnRole.ORDINAL
        # high=1.0, low=0.3, medium=0.6
        assert abs(gd.w_E[0] - 1.0) < 1e-10
        assert abs(gd.w_E[1] - 0.3) < 1e-10
        assert abs(gd.w_E[2] - 0.6) < 1e-10

    def test_evidence_delimited(self, tmp_path):
        csv_path = str(tmp_path / "edges.csv")
        _write_csv(csv_path,
            ["source", "target", "databases"],
            [["A", "B", "SIGNOR;KEGG"], ["B", "C", "BioGRID;STRING"],
             ["A", "C", "KEGG"], ["C", "D", "SIGNOR;KEGG;Reactome"]],
        )
        from rexgraph.io.csv_loader import load_edge_csv, ColumnRole
        gd = load_edge_csv(csv_path)
        assert gd.profiles["databases"].role == ColumnRole.EVIDENCE

    def test_role_override(self, tmp_path):
        csv_path = str(tmp_path / "edges.csv")
        _write_csv(csv_path,
            ["source", "target", "regulation"],
            [["A", "B", "up"], ["B", "C", "down"], ["A", "C", "up"]],
        )
        from rexgraph.io.csv_loader import load_edge_csv, ColumnRole
        # Without override, "regulation" is name-matched to polarity
        gd1 = load_edge_csv(csv_path)
        # With override, force it to type
        gd2 = load_edge_csv(csv_path, roles={"regulation": ColumnRole.TYPE})
        assert gd2.profiles["regulation"].role == ColumnRole.TYPE

    def test_to_rex(self, tmp_path):
        csv_path = str(tmp_path / "edges.csv")
        _write_csv(csv_path,
            ["source", "target", "effect"],
            [["A", "B", "activation"],
             ["B", "C", "inhibition"],
             ["A", "C", "activation"]],
        )
        from rexgraph.io.csv_loader import load_edge_csv
        gd = load_edge_csv(csv_path)
        rex = gd.to_rex()
        assert isinstance(rex, RexGraph)
        assert rex.nV == 3
        assert rex.nE == 3

    def test_to_rex_signs_wired(self, tmp_path):
        """Polarity from CSV flows into L_frustration via signs."""
        csv_path = str(tmp_path / "edges.csv")
        _write_csv(csv_path,
            ["source", "target", "effect"],
            [["A", "B", "activation"],
             ["B", "C", "inhibition"],
             ["A", "C", "activation"]],
        )
        from rexgraph.io.csv_loader import load_edge_csv
        gd = load_edge_csv(csv_path)
        rex = gd.to_rex()
        # Signs should affect the frustration Laplacian
        signs = rex._edge_signs
        assert signs[1] == -1.0  # inhibition
        assert signs[0] == 1.0

    def test_summary(self, tmp_path):
        csv_path = str(tmp_path / "edges.csv")
        _write_csv(csv_path,
            ["source", "target", "type", "score"],
            [["A", "B", "binding", "0.9"],
             ["B", "C", "phosphorylation", "0.5"]],
        )
        from rexgraph.io.csv_loader import load_edge_csv
        gd = load_edge_csv(csv_path)
        s = gd.summary()
        assert "2 vertices" in s or "3 vertices" in s
        assert "type" in s.lower()

    def test_empty_csv_raises(self, tmp_path):
        csv_path = str(tmp_path / "empty.csv")
        with open(csv_path, "w") as f:
            f.write("source,target\n")
        from rexgraph.io.csv_loader import load_edge_csv
        with pytest.raises(ValueError):
            load_edge_csv(csv_path)


# JSON Loader

class TestJSONEdgeList:
    """Test edge-list JSON loading."""

    def test_basic(self, tmp_path):
        path = str(tmp_path / "edges.json")
        _write_json(path, {"edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "A", "target": "C"},
        ]})
        from rexgraph.io.json_loader import load_json
        rex = load_json(path)
        assert rex.nV == 3
        assert rex.nE == 3

    def test_bare_list(self, tmp_path):
        """List of edge dicts without wrapper."""
        path = str(tmp_path / "edges.json")
        _write_json(path, [
            {"source": "X", "target": "Y"},
            {"source": "Y", "target": "Z"},
        ])
        from rexgraph.io.json_loader import load_json
        rex = load_json(path)
        assert rex.nV == 3
        assert rex.nE == 2

    def test_with_metadata(self, tmp_path):
        """Edge metadata gets classified and wired into signs."""
        path = str(tmp_path / "edges.json")
        _write_json(path, {"edges": [
            {"source": "EGFR", "target": "RAS", "effect": "activation"},
            {"source": "P53", "target": "MDM2", "effect": "inhibition"},
        ]})
        from rexgraph.io.json_loader import load_json
        rex = load_json(path)
        assert rex.nV == 4
        assert rex.nE == 2


class TestJSONRexGraph:
    """Test RexGraph native JSON round-trip."""

    def test_roundtrip_triangle(self, tmp_path):
        path = str(tmp_path / "tri.json")
        rex = _make_triangle()
        _write_json(path, rex.to_json())
        from rexgraph.io.json_loader import load_json
        rex2 = load_json(path)
        assert rex2.nV == rex.nV
        assert rex2.nE == rex.nE
        assert np.allclose(rex2.B1, rex.B1)

    def test_roundtrip_k4(self, tmp_path):
        path = str(tmp_path / "k4.json")
        rex = _make_k4()
        d = rex.to_json()
        # to_json doesn't store B2_vals, add it for round-trip
        d["B2_vals"] = rex._B2_vals.tolist()
        _write_json(path, d)
        from rexgraph.io.json_loader import load_json
        rex2 = load_json(path)
        assert rex2.nV == 4
        assert rex2.nE == 6

    def test_autodetect_format(self, tmp_path):
        path = str(tmp_path / "rex.json")
        rex = _make_triangle()
        _write_json(path, rex.to_json())
        from rexgraph.io.json_loader import _detect_format
        with open(path) as f:
            data = json.load(f)
        assert _detect_format(data) == "rexgraph"


class TestJSONCytoscape:
    """Test Cytoscape.js JSON loading."""

    def test_basic(self, tmp_path):
        path = str(tmp_path / "cytoscape.json")
        _write_json(path, {
            "elements": {
                "nodes": [
                    {"data": {"id": "A"}},
                    {"data": {"id": "B"}},
                    {"data": {"id": "C"}},
                ],
                "edges": [
                    {"data": {"source": "A", "target": "B", "weight": 1.0}},
                    {"data": {"source": "B", "target": "C", "weight": -0.5}},
                    {"data": {"source": "A", "target": "C", "weight": 0.8}},
                ],
            }
        })
        from rexgraph.io.json_loader import load_json
        rex = load_json(path)
        assert rex.nV == 3
        assert rex.nE == 3

    def test_negative_weight_signs(self, tmp_path):
        """Negative Cytoscape weights become signs."""
        path = str(tmp_path / "cytoscape.json")
        _write_json(path, {
            "elements": {
                "nodes": [{"data": {"id": "A"}}, {"data": {"id": "B"}}],
                "edges": [{"data": {"source": "A", "target": "B", "weight": -2.5}}],
            }
        })
        from rexgraph.io.json_loader import load_json
        rex = load_json(path)
        assert rex._signs is not None
        assert rex._signs[0] == -1.0

    def test_flat_elements(self, tmp_path):
        """Cytoscape with flat element list (group field)."""
        path = str(tmp_path / "cytoscape.json")
        _write_json(path, {
            "elements": [
                {"group": "nodes", "data": {"id": "X"}},
                {"group": "nodes", "data": {"id": "Y"}},
                {"group": "edges", "data": {"source": "X", "target": "Y"}},
            ]
        })
        from rexgraph.io.json_loader import load_json
        rex = load_json(path, format="cytoscape")
        assert rex.nV == 2
        assert rex.nE == 1


class TestJSONNetworkX:
    """Test NetworkX node-link JSON loading."""

    def test_basic(self, tmp_path):
        path = str(tmp_path / "nx.json")
        _write_json(path, {
            "nodes": [{"id": 0}, {"id": 1}, {"id": 2}],
            "links": [
                {"source": 0, "target": 1},
                {"source": 1, "target": 2},
            ],
        })
        from rexgraph.io.json_loader import load_json
        rex = load_json(path)
        assert rex.nV == 3
        assert rex.nE == 2

    def test_weighted(self, tmp_path):
        path = str(tmp_path / "nx.json")
        _write_json(path, {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "links": [
                {"source": "a", "target": "b", "weight": 3.0},
                {"source": "b", "target": "c", "weight": -1.0},
            ],
        })
        from rexgraph.io.json_loader import load_json
        rex = load_json(path)
        assert rex._w_E is not None
        assert rex._signs is not None


class TestJSONAdjacency:
    """Test adjacency matrix JSON loading."""

    def test_bare_matrix(self, tmp_path):
        path = str(tmp_path / "adj.json")
        _write_json(path, [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ])
        from rexgraph.io.json_loader import load_json
        rex = load_json(path)
        assert rex.nV == 3
        assert rex.nE == 3

    def test_dict_format(self, tmp_path):
        path = str(tmp_path / "adj.json")
        _write_json(path, {
            "matrix": [[0, 0.9, 0], [0.9, 0, 0.5], [0, 0.5, 0]],
            "labels": ["gene1", "gene2", "gene3"],
        })
        from rexgraph.io.json_loader import load_json
        rex = load_json(path)
        assert rex.nV == 3
        assert rex.nE == 2


class TestMatrixCSV:
    """Test square matrix CSV loading (correlation matrices)."""

    def test_basic(self, tmp_path):
        path = str(tmp_path / "corr.csv")
        _write_csv(path,
            ["", "g1", "g2", "g3"],
            [["g1", "1.0", "0.8", "-0.3"],
             ["g2", "0.8", "1.0", "0.5"],
             ["g3", "-0.3", "0.5", "1.0"]],
        )
        from rexgraph.io.json_loader import load_matrix_csv
        rex = load_matrix_csv(path)
        assert rex.nV == 3
        assert rex.nE == 3  # all nonzero off-diagonal

    def test_threshold(self, tmp_path):
        path = str(tmp_path / "corr.csv")
        _write_csv(path,
            ["", "g1", "g2", "g3"],
            [["g1", "1.0", "0.8", "0.1"],
             ["g2", "0.8", "1.0", "0.05"],
             ["g3", "0.1", "0.05", "1.0"]],
        )
        from rexgraph.io.json_loader import load_matrix_csv
        rex = load_matrix_csv(path, threshold=0.2)
        assert rex.nE == 1  # only g1-g2 survives

    def test_signs_from_negative_correlation(self, tmp_path):
        path = str(tmp_path / "corr.csv")
        _write_csv(path,
            ["", "A", "B", "C"],
            [["A", "1.0", "0.9", "-0.7"],
             ["B", "0.9", "1.0", "0.3"],
             ["C", "-0.7", "0.3", "1.0"]],
        )
        from rexgraph.io.json_loader import load_matrix_csv
        rex = load_matrix_csv(path)
        # Should have signs for the negative correlation
        assert rex._signs is not None
        assert np.any(rex._signs < 0)


# Bundle save/load

class TestBundle:
    """Test .rex bundle round-trip."""

    def test_roundtrip_triangle(self, tmp_path):
        path = str(tmp_path / "test.rex")
        rex = _make_triangle()
        from rexgraph.io.bundle import save_rex, load_rex
        save_rex(path, rex)
        rex2 = load_rex(path)
        assert rex2.nV == rex.nV
        assert rex2.nE == rex.nE
        assert np.allclose(rex2.B1, rex.B1)

    def test_roundtrip_k4(self, tmp_path):
        path = str(tmp_path / "k4.rex")
        rex = _make_k4()
        from rexgraph.io.bundle import save_rex, load_rex
        save_rex(path, rex)
        rex2 = load_rex(path)
        assert rex2.nV == 4
        assert rex2.nE == 6
        assert rex2.nF > 0
        assert rex2.betti == rex.betti

    def test_cache_all(self, tmp_path):
        """Saving with cache='all' stores spectral properties."""
        path = str(tmp_path / "cached.rex")
        rex = _make_k4()
        # Force spectral computation
        _ = rex.betti
        from rexgraph.io.bundle import save_rex, load_rex
        save_rex(path, rex, cache="all")
        # Verify cache directory exists
        import pathlib
        cache_dir = pathlib.Path(path) / "cache"
        assert cache_dir.exists()

    def test_manifest_json(self, tmp_path):
        path = str(tmp_path / "test.rex")
        rex = _make_triangle()
        from rexgraph.io.bundle import save_rex
        save_rex(path, rex)
        import pathlib
        manifest = json.loads((pathlib.Path(path) / "MANIFEST.json").read_text())
        assert manifest["object_type"] == "RexGraph"
        assert manifest["nV"] == 3
        assert manifest["nE"] == 3


# Format dispatch

class TestFormatDispatch:
    """Test save/load auto-detection by extension."""

    def test_rex_extension(self, tmp_path):
        path = str(tmp_path / "graph.rex")
        rex = _make_triangle()
        from rexgraph.io import save, load
        save(path, rex)
        rex2 = load(path)
        assert rex2.nV == rex.nV

    def test_json_extension(self, tmp_path):
        path = str(tmp_path / "graph.json")
        rex = _make_triangle()
        _write_json(path, rex.to_json())
        from rexgraph.io import load
        rex2 = load(path)
        assert rex2.nV == rex.nV

    def test_zarr_raises_without_dep(self, tmp_path):
        path = str(tmp_path / "graph.zarr")
        rex = _make_triangle()
        from rexgraph.io import save, HAS_ZARR
        if not HAS_ZARR:
            with pytest.raises(ImportError):
                save(path, rex)

    def test_hdf5_raises_without_dep(self, tmp_path):
        path = str(tmp_path / "graph.h5")
        rex = _make_triangle()
        from rexgraph.io import save, HAS_HDF5
        if not HAS_HDF5:
            with pytest.raises(ImportError):
                save(path, rex)


# Full pipeline integration

class TestIOPipeline:
    """End-to-end: CSV/JSON ingest -> compute -> save -> load -> verify."""

    def test_csv_to_rex_to_bundle(self, tmp_path):
        """Research pipeline: load CSV, build rex, compute RCF, save, reload."""
        csv_path = str(tmp_path / "pathway.csv")
        _write_csv(csv_path,
            ["source", "target", "effect", "score"],
            [["EGFR", "RAS", "activation", "0.95"],
             ["RAS", "RAF", "activation", "0.88"],
             ["RAF", "MEK", "activation", "0.92"],
             ["MEK", "ERK", "activation", "0.90"],
             ["P53", "MDM2", "inhibition", "0.85"],
             ["MDM2", "P53", "inhibition", "0.80"]],
        )
        from rexgraph.io.csv_loader import load_edge_csv
        from rexgraph.io.bundle import save_rex, load_rex

        gd = load_edge_csv(csv_path)
        rex = gd.to_rex()
        assert rex.nV == 7
        assert rex.nE == 6

        # Compute some properties
        _ = rex.betti
        _ = rex.RL

        # Save and reload
        bundle_path = str(tmp_path / "pathway.rex")
        save_rex(bundle_path, rex)
        rex2 = load_rex(bundle_path)
        assert rex2.nV == rex.nV
        assert rex2.nE == rex.nE

    def test_json_to_rex_to_bundle(self, tmp_path):
        """Load Cytoscape export, compute, save as bundle."""
        json_path = str(tmp_path / "network.json")
        _write_json(json_path, {
            "elements": {
                "nodes": [{"data": {"id": f"n{i}"}} for i in range(5)],
                "edges": [
                    {"data": {"source": "n0", "target": "n1", "weight": 1.0}},
                    {"data": {"source": "n1", "target": "n2", "weight": 0.8}},
                    {"data": {"source": "n2", "target": "n3", "weight": -0.5}},
                    {"data": {"source": "n3", "target": "n4", "weight": 0.9}},
                    {"data": {"source": "n0", "target": "n4", "weight": 0.7}},
                ],
            }
        })
        from rexgraph.io.json_loader import load_json
        from rexgraph.io.bundle import save_rex, load_rex

        rex = load_json(json_path)
        assert rex.nV == 5
        assert rex.nE == 5

        bundle_path = str(tmp_path / "network.rex")
        save_rex(bundle_path, rex)
        rex2 = load_rex(bundle_path)
        assert rex2.nV == 5
        assert rex2.nE == 5
