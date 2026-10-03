"""Advertised System views must execute against the registered RCQL binding."""
from contextlib import closing
from io import StringIO

import pytest
from fastapi.testclient import TestClient

from rcql import BoundSource, DatasetSource, Executor, SourcePolicy
from rexgraph import RexGraph, RelationSpec
from rexgraph.io.catalog import FileCatalog
from rexgraph.io.declaration import DatasetDeclaration
from rexgraph.io.records import RecordField, RecordSchema
from system import register_source, remove_source
from system.inspection import available_views, source_row
from system.panels import panel_query
from system.server.app import app


STRUCTURAL = ["Overview", "Structure", "Hodge", "Character", "Flow", "State", "Queries"]


def declared():
    return DatasetSource(DatasetDeclaration("csv", RecordSchema((RecordField("a"), RecordField("b"))),
                         RelationSpec("pair", ("a", "b"))), StringIO("a,b\nx,y\n"))


@pytest.mark.parametrize("kind", ("rex", "dataset"))
def test_advertised_structural_views_are_executable_and_keep_the_source(kind):
    value = declared() if kind == "dataset" else RexGraph.from_graph([0], [1])
    source = BoundSource(value, SourcePolicy.allow("read"))
    row = source_row("live/name", source)
    assert row["panels"] == STRUCTURAL
    for name in STRUCTURAL:
        if name == "Queries": continue
        result = Executor(sources={"live/name": source}).execute(panel_query(name, "live/name", source))
        assert result.values and result.provenance
        if kind == "dataset":
            assert all(value.digest in p["source_state"]["name"] for p in result.provenance)


def test_navigation_listing_never_materializes_a_declared_dataset(monkeypatch):
    value = declared()
    monkeypatch.setattr(DatasetSource, "materialize", lambda self: pytest.fail("navigation materialized input"))
    assert source_row("data", BoundSource(value, SourcePolicy.allow("read")))["panels"] == STRUCTURAL


@pytest.mark.parametrize("name", ("Green", "Critical", "Temporal", "Models", "Agents", "unregistered"))
def test_unsupported_panels_refuse_before_materialization(name, monkeypatch):
    value = declared()
    monkeypatch.setattr(DatasetSource, "materialize", lambda self: pytest.fail("unsupported panel materialized input"))
    with pytest.raises(KeyError, match="has no query"):
        panel_query(name, "data", value)


@pytest.mark.parametrize("permissions,expected", (((), []), (("read",), ["Overview", "Queries"]),
                         (("read", "files", "file_read"), ["Overview", "Files", "Queries"]),
                         (("*",), ["Overview", "Files", "Queries"])))
def test_catalog_navigation_retains_its_existing_file_capabilities(tmp_path, permissions, expected):
    value = BoundSource(FileCatalog([tmp_path]), SourcePolicy.allow(*permissions))
    assert source_row("catalog", value)["panels"] == expected


def test_store_navigation_checks_rcql_surface_without_reading_data(monkeypatch):
    rcdb = pytest.importorskip("rcdb")
    with closing(rcdb.MemoryStore()) as store:
        def forbidden(*a, **kw): pytest.fail("navigation read store data")
        monkeypatch.setattr(store, "stats", forbidden)
        monkeypatch.setattr(store, "state_digest", forbidden)
        value = BoundSource(store, SourcePolicy.allow("read"))
        row = source_row("store", value)
        assert row["panels"] == ["Overview", "RCDB", "Queries"]
        assert row["default_query"] == 'FROM RCDB("store") RETURN RCDB_STATS()'


def test_minimal_store_without_stats_has_no_invented_statistics_view_or_query():
    class Store:
        def get(self, key): raise AssertionError("source listing called get")
        def history(self, key): raise AssertionError("source listing read history")
    value = BoundSource(Store(), SourcePolicy.allow("read"))
    row = source_row("minimal", value)
    assert row["kind"] == "RCDBStore" and row["default_query"] is None
    assert row["panels"] == ["Overview", "Queries"]
    with pytest.raises(TypeError, match="statistics capability"):
        panel_query("RCDB", "minimal", value)


@pytest.mark.parametrize("permissions", ((), ("read",)))
def test_rcdb_panel_uses_the_same_registered_capability_and_bounded_statistics(permissions):
    rcdb = pytest.importorskip("rcdb")
    with closing(rcdb.MemoryStore()) as store:
        store.put_record("r", None, tx_time=1.)
        name = "db/name"
        register_source(name, store, policy=SourcePolicy.allow(*permissions))
        try:
            with TestClient(app) as client:
                row = next(r for r in client.get("/api/sources").json()["sources"] if r["name"] == name)
                response = client.get("/api/panels/rcdb", params={"source": name})
                if not permissions:
                    assert row["panels"] == [] and response.status_code == 400
                else:
                    assert "RCDB" in row["panels"] and response.status_code == 200, response.text
                    result = response.json()
                    assert result["values"][0]["n_records"] == store.stats()["n_records"] == 1
                    assert not {"root", "path", "uri"}.intersection(result["values"][0])
                    assert result["provenance"][0]["logical_operator"] == "RCDB_STATS"
        finally:
            remove_source(name)


def test_unknown_and_denied_sources_have_explicit_navigation():
    assert available_views(object()) == ["Overview", "Queries"]
    assert source_row("hidden", BoundSource(object(), SourcePolicy.allow()))["panels"] == []
