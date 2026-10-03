"""All store views agree on counts; typed values do not fabricate graph statistics."""
from contextlib import closing
from fractions import Fraction

import pytest

from rexgraph import RexGraph
from rcdb import FileStore, LocalStore, MemoryStore, ObjectStore, RexStore, SQLStore


def opened(kind, root):
    if kind == "memory": return MemoryStore()
    if kind == "local": return LocalStore(root)
    if kind == "file": return FileStore(root, read_only=False)
    if kind == "rex": return RexStore(root, read_only=False)
    if kind == "object":
        pytest.importorskip("fsspec")
        return ObjectStore(f"file://{root}", read_only=False)
    pytest.importorskip("sqlalchemy")
    return SQLStore(f"sqlite:///{root}.sqlite", native=False if kind == "legacy-sql" else None)


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "file", "rex", "object", "legacy-sql"])
def test_store_statistics_have_one_common_public_schema(kind, tmp_path):
    with closing(opened(kind, tmp_path / kind)) as store:
        a = RexGraph.from_graph([0], [1])
        b = RexGraph.from_graph([0, 1], [1, 2])
        store.put("r", a, analytics=False)
        store.put("r", b, analytics=False)
        result = store.stats()
        assert result["count"] == result["n_records"] == 1
        assert result["n_versions"] == 2
        assert result["total_vertices"] == 3 and result["total_edges"] == 2
        assert result["backend"] == store.backend
        store.delete("r")
        result = store.stats()
        assert result["count"] == result["n_records"] == result["total_edges"] == 0
        assert result["n_versions"] == (2 if kind in {"memory", "local", "sql"} else 0)
        assert result["mean_kappa"] is None


@pytest.mark.parametrize("kind", ["memory", "local", "sql"])
def test_generic_signatures_cannot_distort_graph_statistics(kind, tmp_path):
    with closing(opened(kind, tmp_path / kind)) as store:
        store.put("graph", RexGraph.from_graph([0], [1]), analytics=False)
        sig = {"object_type": "NativeValue", "nV": 999, "nE": 888, "kappa_mean": 1000}
        store.put_record("value", Fraction(1, 7), signature=sig)
        result = store.stats()
        assert result["count"] == result["n_records"] == 2
        assert result["n_versions"] == 2 and result["total_vertices"] == 2 and result["total_edges"] == 1
        assert result["mean_kappa"] == 0
        store.delete("graph")
        result = store.stats()
        assert result["count"] == result["n_records"] == 1 and result["n_versions"] == 2
        assert result["total_vertices"] == result["total_edges"] == 0 and result["mean_kappa"] is None
