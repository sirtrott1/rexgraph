"""
Tests for rexgraph.io.sql_bridge -- SQL database bridge.

Requires sqlalchemy and pandas. Skipped if either is not installed.
Uses in-memory SQLite for all tests (no file I/O).

Verifies:
    - Engine: get_engine returns a usable engine
    - Boundary table: write/read reconstructs boundary_ptr/idx
    - Edge table: correct columns, nE rows
    - Vertex table: correct columns, nV rows
    - Face table: write/read reconstructs B2 CSC arrays
    - Persistence table: pairs roundtrip with betti metadata
    - Filtration table: splits back into filt_v, filt_e, filt_f
    - Metrics table: roundtrip, empty raises
"""
import numpy as np
import pytest

try:
    import sqlalchemy
    import pandas
    HAS_SQL_DEPS = True
except ImportError:
    HAS_SQL_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_SQL_DEPS, reason="sqlalchemy/pandas not installed")

if HAS_SQL_DEPS:
    from rexgraph.io.sql_bridge import (
        get_engine,
        write_boundary_sql,
        read_boundary_sql,
        write_edge_sql,
        read_edge_sql,
        write_vertex_sql,
        read_vertex_sql,
        write_face_sql,
        read_face_sql,
        write_persistence_sql,
        read_persistence_sql,
        write_filtration_sql,
        read_filtration_sql,
        write_metrics_sql,
        read_metrics_sql,
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
def engine():
    return get_engine("sqlite:///:memory:")


# Engine

class TestEngine:

    def test_returns_engine(self):
        eng = get_engine("sqlite:///:memory:")
        assert eng is not None

    def test_shared_memory(self):
        """Same in-memory SQLite engine returned for same conn string."""
        e1 = get_engine("sqlite:///:memory:")
        e2 = get_engine("sqlite:///:memory:")
        assert e1 is e2


# Boundary Table

class TestBoundaryTable:

    def test_roundtrip(self, k4, engine):
        write_boundary_sql(k4, engine, "boundary")
        loaded = read_boundary_sql(engine, "boundary")
        assert "boundary_ptr" in loaded
        assert "boundary_idx" in loaded
        bp_orig = np.asarray(k4._boundary_ptr)
        assert np.array_equal(bp_orig, loaded["boundary_ptr"])

    def test_entry_count(self, k4, engine):
        write_boundary_sql(k4, engine, "boundary")
        loaded = read_boundary_sql(engine, "boundary")
        n_entries = int(k4._boundary_ptr[-1])
        assert loaded["boundary_idx"].shape[0] == n_entries


# Edge Table

class TestEdgeTable:

    def test_columns(self, k4, engine):
        write_edge_sql(k4, engine, "edges")
        loaded = read_edge_sql(engine, "edges")
        for col in ["edge_idx", "source", "target", "boundary_size", "edge_type"]:
            assert col in loaded

    def test_row_count(self, k4, engine):
        write_edge_sql(k4, engine, "edges")
        loaded = read_edge_sql(engine, "edges")
        assert loaded["edge_idx"].shape[0] == k4.nE


# Vertex Table

class TestVertexTable:

    def test_columns(self, triangle, engine):
        write_vertex_sql(triangle, engine, "vertices")
        loaded = read_vertex_sql(engine, "vertices")
        assert "vertex_idx" in loaded

    def test_row_count(self, triangle, engine):
        write_vertex_sql(triangle, engine, "vertices")
        loaded = read_vertex_sql(engine, "vertices")
        assert loaded["vertex_idx"].shape[0] == triangle.nV


# Face Table

class TestFaceTable:

    def test_roundtrip(self, k4, engine):
        write_face_sql(k4, engine, "faces")
        loaded = read_face_sql(engine, "faces")
        assert "B2_col_ptr" in loaded
        assert "B2_row_idx" in loaded
        assert "B2_vals" in loaded

    def test_empty_faces(self, triangle, engine):
        write_face_sql(triangle, engine, "faces")
        loaded = read_face_sql(engine, "faces")
        assert loaded["nF"] == 0


# Persistence Table

class TestPersistenceTable:

    def test_roundtrip(self, k4, engine):
        fv, fe, ff = k4.filtration(kind="dimension")
        result = k4.persistence(fv, fe, ff)
        write_persistence_sql(result, engine, "persistence")
        loaded = read_persistence_sql(engine, "persistence")
        assert "birth" in loaded
        assert "death" in loaded
        assert "dim" in loaded
        if "betti" in loaded:
            assert isinstance(loaded["betti"], tuple)


# Filtration Table

class TestFiltrationTable:

    def test_roundtrip(self, k4, engine):
        fv = np.zeros(k4.nV, dtype=np.float64)
        fe = np.ones(k4.nE, dtype=np.float64)
        ff = np.full(k4.nF, 2.0, dtype=np.float64)
        write_filtration_sql(k4, fv, fe, ff, engine, "filtration", kind="dimension")
        loaded = read_filtration_sql(engine, "filtration")
        assert np.allclose(loaded["filt_v"], fv)
        assert np.allclose(loaded["filt_e"], fe)
        assert np.allclose(loaded["filt_f"], ff)
        assert loaded["kind"] == "dimension"


# Metrics Table

class TestMetricsTable:

    def test_roundtrip(self, engine):
        metrics = {
            "pagerank": np.array([0.1, 0.2, 0.3, 0.4]),
            "clustering": np.array([1.0, 0.5, 0.5, 1.0]),
        }
        write_metrics_sql(metrics, engine, "metrics")
        loaded = read_metrics_sql(engine, "metrics")
        assert "pagerank" in loaded
        assert np.allclose(loaded["pagerank"], metrics["pagerank"])
        assert "cell_idx" not in loaded  # excluded by default

    def test_empty_raises(self, engine):
        with pytest.raises(ValueError):
            write_metrics_sql({}, engine, "metrics")
