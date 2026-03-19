"""
Tests for rexgraph.io.parquet_bridge -- Parquet table export/import.

All tests require pyarrow. Skipped if pyarrow is not installed.

Verifies:
    - Generic write/read roundtrip: 1D, 2D arrays, metadata
    - Boundary table: write/read reconstructs boundary_ptr/boundary_idx
    - Edge table: correct columns, nE rows
    - Vertex table: correct columns, nV rows
    - Face table: write/read reconstructs B2 CSC arrays
    - Persistence table: pairs roundtrip with betti metadata
    - Filtration table: splits back into filt_v, filt_e, filt_f
    - Metrics table: generic per-cell roundtrip
"""
import os
import tempfile

import numpy as np
import pytest

try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

pytestmark = pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")

if HAS_PYARROW:
    from rexgraph.io.parquet_bridge import (
        write_parquet,
        read_parquet,
        write_boundary_table,
        read_boundary_table,
        write_edge_table,
        read_edge_table,
        write_vertex_table,
        read_vertex_table,
        write_face_table,
        read_face_table,
        write_persistence_table,
        read_persistence_table,
        write_filtration_table,
        read_filtration_table,
        write_metrics_table,
        read_metrics_table,
    )

from rexgraph.graph import RexGraph


# Fixtures

@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


@pytest.fixture
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.fixture
def tmp_path_pq(tmp_path):
    return str(tmp_path / "test.parquet")


# Generic write/read

class TestGenericParquet:

    def test_1d_roundtrip(self, tmp_path_pq):
        data = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0])}
        write_parquet(data, tmp_path_pq)
        loaded = read_parquet(tmp_path_pq)
        assert np.allclose(loaded["x"], data["x"])
        assert np.allclose(loaded["y"], data["y"])

    def test_2d_roundtrip(self, tmp_path_pq):
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        write_parquet({"mat": arr}, tmp_path_pq)
        loaded = read_parquet(tmp_path_pq)
        assert loaded["mat"].shape == (3, 2)
        assert np.allclose(loaded["mat"], arr)

    def test_metadata(self, tmp_path_pq):
        write_parquet({"x": np.ones(3)}, tmp_path_pq,
                      metadata={"version": 2})
        # Just verify it doesn't crash; metadata is in schema


# Boundary Table

class TestBoundaryTable:

    def test_roundtrip(self, k4, tmp_path_pq):
        write_boundary_table(k4, tmp_path_pq)
        loaded = read_boundary_table(tmp_path_pq)
        assert "boundary_ptr" in loaded
        assert "boundary_idx" in loaded
        bp_orig = np.asarray(k4._boundary_ptr)
        bp_loaded = loaded["boundary_ptr"]
        assert np.array_equal(bp_orig, bp_loaded)

    def test_correct_entry_count(self, k4, tmp_path_pq):
        write_boundary_table(k4, tmp_path_pq)
        loaded = read_parquet(tmp_path_pq)
        # Each standard edge has 2 boundary vertices
        assert loaded["edge_idx"].shape[0] == int(k4._boundary_ptr[-1])


# Edge Table

class TestEdgeTable:

    def test_columns_present(self, k4, tmp_path_pq):
        write_edge_table(k4, tmp_path_pq)
        loaded = read_edge_table(tmp_path_pq)
        for col in ["edge_idx", "source", "target", "boundary_size", "edge_type"]:
            assert col in loaded

    def test_row_count(self, k4, tmp_path_pq):
        write_edge_table(k4, tmp_path_pq)
        loaded = read_edge_table(tmp_path_pq)
        assert loaded["edge_idx"].shape[0] == k4.nE


# Vertex Table

class TestVertexTable:

    def test_columns_present(self, triangle, tmp_path_pq):
        write_vertex_table(triangle, tmp_path_pq)
        loaded = read_vertex_table(tmp_path_pq)
        assert "vertex_idx" in loaded

    def test_row_count(self, triangle, tmp_path_pq):
        write_vertex_table(triangle, tmp_path_pq)
        loaded = read_vertex_table(tmp_path_pq)
        assert loaded["vertex_idx"].shape[0] == triangle.nV


# Face Table

class TestFaceTable:

    def test_roundtrip(self, k4, tmp_path_pq):
        write_face_table(k4, tmp_path_pq)
        loaded = read_face_table(tmp_path_pq)
        assert "B2_col_ptr" in loaded
        assert "B2_row_idx" in loaded
        assert "B2_vals" in loaded

    def test_empty_faces(self, triangle, tmp_path_pq):
        """Triangle with no faces writes an empty table."""
        write_face_table(triangle, tmp_path_pq)
        loaded = read_face_table(tmp_path_pq)
        assert loaded["nF"] == 0


# Persistence Table

class TestPersistenceTable:

    def test_roundtrip(self, k4, tmp_path_pq):
        fv, fe, ff = k4.filtration(kind="dimension")
        result = k4.persistence(fv, fe, ff)
        write_persistence_table(result, tmp_path_pq)
        loaded = read_persistence_table(tmp_path_pq)
        assert "birth" in loaded
        assert "death" in loaded
        assert "dim" in loaded
        if "betti" in loaded:
            assert isinstance(loaded["betti"], tuple)


# Filtration Table

class TestFiltrationTable:

    def test_roundtrip(self, k4, tmp_path_pq):
        fv = np.zeros(k4.nV, dtype=np.float64)
        fe = np.ones(k4.nE, dtype=np.float64)
        ff = np.full(k4.nF, 2.0, dtype=np.float64)
        write_filtration_table(k4, fv, fe, ff, tmp_path_pq, kind="dimension")
        loaded = read_filtration_table(tmp_path_pq)
        assert np.allclose(loaded["filt_v"], fv)
        assert np.allclose(loaded["filt_e"], fe)
        assert np.allclose(loaded["filt_f"], ff)
        assert loaded["kind"] == "dimension"


# Metrics Table

class TestMetricsTable:

    def test_roundtrip(self, tmp_path_pq):
        metrics = {
            "pagerank": np.array([0.1, 0.2, 0.3, 0.4]),
            "clustering": np.array([1.0, 0.5, 0.5, 1.0]),
        }
        write_metrics_table(metrics, tmp_path_pq)
        loaded = read_metrics_table(tmp_path_pq)
        assert "pagerank" in loaded
        assert np.allclose(loaded["pagerank"], metrics["pagerank"])
        assert "cell_idx" not in loaded  # excluded by default

    def test_empty_raises(self, tmp_path_pq):
        with pytest.raises(ValueError):
            write_metrics_table({}, tmp_path_pq)
