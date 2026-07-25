"""Tests for the Database Manager (connections, table browsing, import) and
schema DDL export."""

import sqlite3

import pytest

from agent import schema_complex as sc


def _make_db(path, cyclic=True):
    con = sqlite3.connect(str(path))
    if cyclic:
        con.executescript(
            "CREATE TABLE a(id INTEGER PRIMARY KEY, b_id INTEGER REFERENCES b(id));"
            "CREATE TABLE b(id INTEGER PRIMARY KEY, a_id INTEGER REFERENCES a(id));"
            "CREATE TABLE c(id INTEGER PRIMARY KEY);")
    else:
        con.executescript(
            "CREATE TABLE users(id INTEGER PRIMARY KEY);"
            "CREATE TABLE orders(id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id));")
    con.commit()
    con.close()


class TestDdlExport:
    def test_ddl_from_spec_is_cycle_safe(self):
        spec = {"tables": [
            {"name": "a", "columns": ["id", "b_id"], "primary_key": ["id"],
             "foreign_keys": [{"columns": ["b_id"], "references": "b"}]},
            {"name": "b", "columns": ["id", "a_id"], "primary_key": ["id"],
             "foreign_keys": [{"columns": ["a_id"], "references": "a"}]}]}
        ddl = sc.export_schema_ddl(sc.parse_schema_json(spec))
        assert "CREATE TABLE a" in ddl and "CREATE TABLE b" in ddl
        # the cycle-closing FK is emitted as a trailing ALTER
        assert "ALTER TABLE" in ddl

    def test_list_tables(self, tmp_path):
        dbf = tmp_path / "x.db"
        _make_db(dbf)
        rows = sc.list_tables("sqlite:///" + str(dbf))
        names = {r["table"] for r in rows}
        assert {"a", "b", "c"} <= names


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    from fastapi.testclient import TestClient
    # isolate the connections store and RCDB
    cfg = tmp_path_factory.mktemp("cfg")
    import agent.server.routes.dbmanager as DM
    from agent.secrets import FileSecretStore
    DM._SECRETS = FileSecretStore(str(cfg / "connections.json"))
    os.environ["REXGRAPH_RCDB_URI"] = f"sqlite:///{cfg / 'rcdb.sqlite'}"
    import agent.server.routes.rcdb as R
    R._STORE = None
    from agent.server.app import app
    from agent.server.auth import get_auth_manager
    get_auth_manager().disable_auth()
    with TestClient(app) as c:
        yield c
    R._STORE = None


class TestDbManagerRoutes:
    def test_connection_crud_and_masking(self, client):
        r = client.post("/api/v1/dbmanager/connections",
                        json={"name": "prod", "uri": "postgresql://u:secret@h/db"})
        assert r.status_code == 200
        assert "secret" not in r.json()["uri"] and "****" in r.json()["uri"]
        lst = client.get("/api/v1/dbmanager/connections").json()["connections"]
        assert any(c["name"] == "prod" for c in lst)
        assert all("secret" not in c["uri"] for c in lst)
        assert client.delete("/api/v1/dbmanager/connections/prod").json()["deleted"]

    def test_test_browse_import(self, client, tmp_path_factory):
        dbf = tmp_path_factory.mktemp("db") / "live.db"
        _make_db(dbf)
        uri = "sqlite:///" + str(dbf)
        client.post("/api/v1/dbmanager/connections", json={"name": "live", "uri": uri})
        # test
        assert client.post("/api/v1/dbmanager/test", json={"name": "live"}).json()["ok"]
        # browse
        tables = client.post("/api/v1/dbmanager/tables", json={"name": "live"}).json()["tables"]
        assert {t["table"] for t in tables} >= {"a", "b", "c"}
        # import -> diagnose -> RCDB
        imp = client.post("/api/v1/dbmanager/import", json={"name": "live"}).json()
        assert imp["verdict"] == "cycles_present"
        assert imp["stored_as"] == "live-schema"
        # confirm it's now an RCDB record
        q = client.post("/api/v1/db/query", json={"tags_any": ["schema"]}).json()
        assert "live-schema" in [rec["id"] for rec in q["records"]]

    def test_strain_on_connection(self, client, tmp_path_factory):
        import sqlite3
        dbf = tmp_path_factory.mktemp("cstrain") / "s.db"
        con = sqlite3.connect(str(dbf))
        con.executescript(
            "CREATE TABLE A(id INTEGER PRIMARY KEY);"
            "CREATE TABLE D(id INTEGER PRIMARY KEY);"
            "CREATE TABLE C(id INTEGER PRIMARY KEY, a INTEGER REFERENCES A(id), d INTEGER REFERENCES D(id));"
            "CREATE TABLE B(id INTEGER PRIMARY KEY, a INTEGER REFERENCES A(id), "
            "c INTEGER REFERENCES C(id), d INTEGER REFERENCES D(id));")
        con.executemany("INSERT INTO A(id) VALUES(?)", [(i,) for i in range(5)])
        con.executemany("INSERT INTO D(id) VALUES(?)", [(i,) for i in range(5)])
        con.executemany("INSERT INTO C(id,a,d) VALUES(?,0,0)", [(i,) for i in range(2)])
        con.executemany("INSERT INTO B(id,a,c,d) VALUES(?,0,0,0)", [(i,) for i in range(100)])
        con.commit()
        con.close()
        client.post("/api/v1/dbmanager/connections",
                    json={"name": "strain_db", "uri": "sqlite:///" + str(dbf)})
        r = client.post("/api/v1/dbmanager/strain", json={"name": "strain_db"}).json()
        assert r["has_geometry"] and r["total_strain"] > 0
        assert r["row_counts"]["B"] == 100

    def test_ddl_endpoint(self, client):
        spec = {"tables": [{"name": "users", "primary_key": ["id"]},
                           {"name": "orders",
                            "foreign_keys": [{"columns": ["user_id"], "references": "users"}]}]}
        ddl = client.post("/api/v1/dbmanager/ddl", json={"spec": spec}).json()["ddl"]
        assert "CREATE TABLE users" in ddl and "CREATE TABLE orders" in ddl
