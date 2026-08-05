"""
Tests for rexgraph.io.sql_bridge: SQL database bridge.

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
import json

import numpy as np
import pytest

try:
    import pandas
    import sqlalchemy
    HAS_SQL_DEPS = True
except ImportError:
    HAS_SQL_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_SQL_DEPS, reason="sqlalchemy/pandas not installed")

if HAS_SQL_DEPS:
    from rexgraph.io.sql_bridge import (
        get_engine,
        read_boundary_sql,
        read_character_sql,
        read_edge_sql,
        read_face_sql,
        read_filtration_sql,
        read_metrics_sql,
        read_persistence_sql,
        read_vertex_character_sql,
        read_vertex_sql,
        read_void_sql,
        reconstruct_rex_sql,
        write_boundary_sql,
        write_character_sql,
        write_edge_sql,
        write_face_sql,
        write_filtration_sql,
        write_metrics_sql,
        write_persistence_sql,
        write_vertex_character_sql,
        write_vertex_sql,
        write_void_sql,
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
def branching():
    # An arity-3 hyperedge {0,1,2} embedded as a branching edge (Definition 3.2),
    # plus two standard edges (0,3) and (1,4).
    ptr = np.array([0, 3, 5, 7], dtype=np.int32)
    idx = np.array([0, 1, 2, 0, 3, 1, 4], dtype=np.int32)
    return RexGraph.from_hypergraph(ptr, idx)


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


# Character Tables (v2 / RCF)

class TestCharacterTable:

    def test_roundtrip(self, k4, engine):
        write_character_sql(k4, engine, table="character")
        loaded = read_character_sql(engine, table="character")
        assert "edge_idx" in loaded
        assert loaded["edge_idx"].shape[0] == k4.nE
        chi = np.asarray(k4.structural_character)
        chi_cols = [c for c in loaded if c.startswith("chi_")]
        assert len(chi_cols) == chi.shape[1]


class TestVertexCharacterTable:

    def test_roundtrip(self, k4, engine):
        write_vertex_character_sql(k4, engine, table="vertex_character")
        loaded = read_vertex_character_sql(engine, table="vertex_character")
        assert "vertex_idx" in loaded
        assert "kappa" in loaded
        assert loaded["vertex_idx"].shape[0] == k4.nV
        order = np.argsort(loaded["vertex_idx"])
        assert np.allclose(loaded["kappa"][order], np.asarray(k4.coherence))


class TestVoidTable:

    def test_roundtrip(self, k4, engine):
        write_void_sql(k4, engine, table="void")
        loaded = read_void_sql(engine, table="void")
        assert "void_idx" in loaded
        assert "eta" in loaded
        assert "fills_beta" in loaded


# Branching / hyperedge topology round-trip through the edge table

class TestBranchingEdgeRoundtrip:

    def test_all_endpoints_preserved(self, branching, engine):
        # Edge 0 is an arity-3 branching edge; source/target alone (2 endpoints)
        # would silently drop vertex 2.
        assert int(np.asarray(branching.edge_types)[0]) == 2  # EdgeType.BRANCHING
        write_edge_sql(branching, engine, "edges")
        loaded = read_edge_sql(engine, "edges")
        assert "endpoints" in loaded
        eps = [json.loads(s) for s in loaded["endpoints"]]
        assert eps[0] == [0, 1, 2]
        assert eps[1] == [0, 3]
        assert eps[2] == [1, 4]
        # The full general-boundary CSR round-trips.
        flat = [v for ep in eps for v in ep]
        assert flat == [int(v) for v in np.asarray(branching._boundary_idx)]


# Dtype fidelity: SQLite widens int32 -> int64 through a generic driver-level
# fetch (and pandas' read_sql inherits that widening); Core plus the recorded
# numpy dtype must cast back exactly.

class TestDtypeFidelity:

    def test_sql_preserves_int32_dtype(self, engine):
        g = RexGraph(sources=np.array([0, 1, 2, 3], dtype=np.int32),
                     targets=np.array([1, 2, 3, 0], dtype=np.int32))
        write_edge_sql(g, engine, "e")
        out = read_edge_sql(engine, "e")
        assert out["source"].dtype == np.int32
        assert out["target"].dtype == np.int32

    def test_sql_preserves_float64_dtype(self, k4, engine):
        write_face_sql(k4, engine, "faces_dt")
        loaded = read_face_sql(engine, "faces_dt")
        assert loaded["B2_vals"].dtype == np.float64


# No pandas on the SQL read/write path at all.

class TestNoPandasImport:

    def test_sql_no_pandas_import(self):
        import sys
        sys.modules.pop("pandas", None)
        import importlib

        import rexgraph.io.sql_bridge as sb
        importlib.reload(sb)

        eng = sb.get_engine("sqlite:///:memory:")
        g = RexGraph(sources=np.array([0, 1], dtype=np.int32),
                     targets=np.array([1, 2], dtype=np.int32))
        sb.write_boundary_sql(g, eng, "b_boundary")
        sb.write_edge_sql(g, eng, "b_edge")
        r = sb.reconstruct_rex_sql(eng, boundary="b_boundary", edge="b_edge")
        assert r.nE == g.nE
        assert "pandas" not in sys.modules   # the SQL path never imports pandas


# Full reconstruct: boundary (+ optional face, + optional edge weights) -> RexGraph.
# Each write_*_sql call uses its own explicit, distinct table name (there is no
# fixed prefix convention shared across write_boundary_sql/write_face_sql/write_edge_sql).

class TestReconstructRexSql:

    def test_boundary_only(self, triangle, engine):
        write_boundary_sql(triangle, engine, "tri_boundary")
        r = reconstruct_rex_sql(engine, boundary="tri_boundary")
        assert r.nE == triangle.nE
        assert r.nV == triangle.nV
        assert np.array_equal(np.asarray(r._boundary_ptr), np.asarray(triangle._boundary_ptr))
        assert np.array_equal(np.asarray(r._boundary_idx), np.asarray(triangle._boundary_idx))

    def test_boundary_face_edge(self, k4, engine):
        w = np.arange(1, k4.nE + 1, dtype=np.float64)
        k4_w = RexGraph(boundary_ptr=np.asarray(k4._boundary_ptr),
                        boundary_idx=np.asarray(k4._boundary_idx),
                        B2_col_ptr=np.asarray(k4._B2_col_ptr),
                        B2_row_idx=np.asarray(k4._B2_row_idx),
                        B2_vals=np.asarray(k4._B2_vals),
                        w_E=w)
        write_boundary_sql(k4_w, engine, "k4_b1")
        write_face_sql(k4_w, engine, "k4_b2")
        write_edge_sql(k4_w, engine, "k4_edges")

        r = reconstruct_rex_sql(engine, boundary="k4_b1", face="k4_b2", edge="k4_edges")

        assert r.nE == k4_w.nE
        assert r.nF == k4_w.nF
        assert np.allclose(np.asarray(r.betti), np.asarray(k4_w.betti))
        assert np.allclose(np.asarray(r.w_E), w)

    def test_missing_face_and_edge_tables_are_optional(self, triangle, engine):
        write_boundary_sql(triangle, engine, "solo_boundary")
        # No face/edge tables written; passing names that do not exist must not error.
        r = reconstruct_rex_sql(engine, boundary="solo_boundary", face="nope_face", edge="nope_edge")
        assert r.nE == triangle.nE
        assert r.nF == 0
        assert r.w_E is None


def test_read_sql_batches_accepts_a_connection():
    # store.py passes engine.connect() (a Connection); the Core path must accept it like pandas did.
    import numpy as np

    from rexgraph.io.sql_bridge import get_engine, read_sql_batches, write_metrics_sql
    eng = get_engine("sqlite:///:memory:")
    write_metrics_sql({"a": np.array([1, 2, 3], np.int64), "b": np.array([0.5, 1.5, 2.5])},
                      eng, "m", cell_dim=0)
    conn = eng.connect()
    batch = next(read_sql_batches(conn, "m"))    # a Connection, not an Engine
    assert list(np.asarray(batch["a"])) == [1, 2, 3]
    conn.close()
