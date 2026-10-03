"""Declared sources reach inspection, panels and editor through the same binding."""
from contextlib import closing
from io import StringIO
import json
import sqlite3
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from rcql import BoundSource, DatasetSource, Executor, SourcePolicy, parse
from rexgraph import RexGraph, RelationSpec
from rexgraph.io.declaration import DatasetDeclaration
from rexgraph.io.records import RecordField, RecordSchema
from system import register_dataset, register_source, remove_source
from system.inspection import source_row
from system.panels import panel_query
from system.server.app import app


def declaration(reader="csv", options=None):
    return DatasetDeclaration(reader, RecordSchema((RecordField("a"), RecordField("b"))),
                              RelationSpec("pair", ("a", "b")), options or {})


@pytest.fixture
def dataset_source():
    return DatasetSource(declaration(), StringIO("a,b\nx,y\n"))


@pytest.mark.parametrize("name", ["declared", "two words", "a/b", "a'quote\"\\name", "λ/edge\nnext"])
def test_source_names_roundtrip_from_editor_query_and_detail_route(name):
    source = register_dataset(name, declaration(), StringIO("a,b\nx,y\n"), policy=SourcePolicy.allow("read"))
    try:
        with TestClient(app) as client:
            row = next(r for r in client.get("/api/sources").json()["sources"] if r["name"] == name)
            assert row["accessible"] and row["kind"] == "Dataset"
            assert row["description"]["digest"] == source.digest
            parsed = parse(row["default_query"])
            result = Executor(sources={name: BoundSource(source, SourcePolicy.allow("read"))}).execute(parsed)
            assert result.values[0]["cells"] == (2, 1) and result.values[1] == 1
            detail = client.get("/api/source", params={"name": name})
            assert detail.status_code == 200, detail.text
            assert detail.json()["kind"] == "Dataset" and detail.json()["cells"] == [2, 1]
            assert detail.json()["digest"] == source.digest
            if "\n" not in name:
                assert client.get("/api/sources/" + quote(name, safe="")).json() == detail.json()
    finally:
        remove_source(name)


@pytest.mark.parametrize("panel", ["state", "structure", "hodge", "character", "flow"])
def test_structural_panels_materialize_the_declared_binding_with_provenance(panel, dataset_source):
    name = "panel-dataset"
    register_source(name, dataset_source, policy=SourcePolicy.allow("read"))
    try:
        with TestClient(app) as client:
            response = client.get(f"/api/panels/{panel}", params={"source": name})
            assert response.status_code == 200, response.text
            refs = [p["source_state"] for p in response.json()["provenance"]]
            assert refs and all(dataset_source.digest in p["name"] for p in refs)
            assert all(p["state_digest"] == dataset_source.as_record()["state_digest"] for p in refs)
    finally:
        remove_source(name)


def test_listing_uses_bounded_dataset_metadata_without_materializing(monkeypatch, dataset_source):
    monkeypatch.setattr(DatasetSource, "materialize", lambda self: pytest.fail("listing materialized data"))
    row = source_row("bounded", BoundSource(dataset_source, SourcePolicy.allow("read")))
    assert row["description"]["reader"] == {"name": "csv", "version": 1}
    assert "parameters" not in row["description"]["reader"]
    assert row["default_query"].startswith("FROM DATASET(")


def test_denied_sources_refuse_before_metadata_or_materialization(monkeypatch, dataset_source):
    monkeypatch.setattr(DatasetSource, "materialize", lambda self: pytest.fail("denied materialization"))
    monkeypatch.setattr(DatasetSource, "as_record", lambda self: pytest.fail("denied metadata"))
    name = "denied-inspection"
    register_source(name, dataset_source, policy=SourcePolicy.allow())
    try:
        with TestClient(app) as client:
            row = next(r for r in client.get("/api/sources").json()["sources"] if r["name"] == name)
            assert not row["accessible"] and row["default_query"] is None
            assert "description" not in row and "kind" not in row
            assert client.get("/api/source", params={"name": name}).status_code == 403
            assert client.get("/api/panels/structure", params={"source": name}).status_code == 400
        with pytest.raises(PermissionError):
            panel_query("structure", name, BoundSource(dataset_source, SourcePolicy.allow()))
    finally:
        remove_source(name)


def test_inspection_never_discloses_dataset_paths_queries_or_reader_options(tmp_path):
    path = tmp_path / "private-source.sqlite"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE pairs (a TEXT, b TEXT, token TEXT)")
        connection.execute("INSERT INTO pairs VALUES ('x', 'y', 'private-token')")
    declared = declaration("sqlite", {"query": "SELECT a, b FROM pairs WHERE token = ?",
                                      "parameters": ("private-token",)})
    source = DatasetSource(declared, path)
    from system.inspection import source_description
    for data in (source_row("sql-data", source), source_description(source, detail=True)):
        text = json.dumps(data)
        assert str(path) not in text and "private-token" not in text
        assert "SELECT a" not in text and "parameters" not in text


@pytest.mark.parametrize("kind", ["rex", "catalog", "store"])
def test_editor_defaults_use_the_registered_source_kind(kind, tmp_path):
    if kind == "rex":
        value = RexGraph.from_graph([0], [1]); form = "REX"
    elif kind == "catalog":
        from rexgraph.io.catalog import FileCatalog
        (tmp_path/"entry.safetensors").write_bytes(b"entry")
        value = FileCatalog([tmp_path]); form = "CATALOG"
    else:
        rcdb = pytest.importorskip("rcdb")
        value = rcdb.MemoryStore(); form = "RCDB"
    try:
        bound = BoundSource(value, SourcePolicy.allow("read", "files", "file_read"))
        row = source_row("source/name", bound)
        assert row["default_query"].startswith("FROM " + form + "(")
        assert Executor(sources={"source/name":bound}).execute(parse(row["default_query"])).values
    finally:
        if kind == "store":
            value.close()


def test_catalog_default_requires_its_existing_file_capabilities(tmp_path):
    from rexgraph.io.catalog import FileCatalog
    row = source_row("catalog", BoundSource(FileCatalog([tmp_path]), SourcePolicy.allow("read")))
    assert row["accessible"] and row["default_query"] is None


def test_catalog_query_parameter_route_retains_names_and_file_policy(tmp_path):
    from rexgraph.io.catalog import FileCatalog
    (tmp_path/"entry.safetensors").write_bytes(b"entry")
    name = "files/path\nnext"
    try:
        register_source(name, FileCatalog([tmp_path]), policy=SourcePolicy.allow("read", "files", "file_read"))
        with TestClient(app) as client:
            response = client.get("/api/catalog", params={"name": name})
            assert response.status_code == 200 and response.json()["name"] == name
            assert len(response.json()["entries"]) == 1
        register_source(name, FileCatalog([tmp_path]), policy=SourcePolicy.allow("read"))
        with TestClient(app) as client:
            response = client.get("/api/catalog", params={"name": name})
            assert response.status_code == 400 and "requires" in response.text
    finally:
        remove_source(name)


def test_panel_planning_and_execution_use_one_source_snapshot(monkeypatch, dataset_source):
    from system.state import sources
    name = "snapshot-panel"
    original = sources.snapshot
    count = 0
    register_source(name, dataset_source)
    def snapshot():
        nonlocal count
        count += 1
        if count > 1:
            pytest.fail("panel rebound sources after planning")
        return original()
    monkeypatch.setattr(sources, "snapshot", snapshot)
    try:
        with TestClient(app) as client:
            assert client.get("/api/panels/structure", params={"source": name}).status_code == 200
        assert count == 1
    finally:
        remove_source(name)


def test_unknown_source_has_metadata_and_no_invented_query():
    row = source_row("unknown", object())
    assert row["description"]["kind"] == "Unknown" and row["default_query"] is None


def test_corrupt_dataset_inspection_is_reported_instead_of_disguised_as_generic(dataset_source):
    dataset_source._payload = b"broken"
    name = "corrupt-inspection"
    register_source(name, dataset_source)
    try:
        with TestClient(app) as client:
            response = client.get("/api/source", params={"name": name})
            assert response.status_code == 400
    finally:
        remove_source(name)
