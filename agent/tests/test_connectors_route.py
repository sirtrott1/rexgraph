"""Route tests for /api/v1/connectors via TestClient."""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient


def _shop_db(path):
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE customers(id INTEGER PRIMARY KEY);"
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, "
        "  customer_id INT REFERENCES customers(id));"
        "CREATE TABLE order_items(order_id INT REFERENCES orders(id), "
        "  product_id INT, PRIMARY KEY(order_id, product_id));")
    con.executemany("INSERT INTO orders(id,customer_id) VALUES(?,0)",
                    [(i,) for i in range(100)])
    con.executemany("INSERT INTO order_items(order_id,product_id) VALUES(?,?)",
                    [(i, i % 5) for i in range(400)])
    con.commit(); con.close()


@pytest.fixture
def client(tmp_path, monkeypatch):
    # isolate the app RCDB into a temp sqlite store for this test
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "rcdb.sqlite"))
    import agent.server.routes.rcdb as rcdbroute
    rcdbroute._STORE = None
    from agent.server.app import app
    # entered, so the app's lifespan runs and its shutdown disposes the SQL engines
    with TestClient(app) as client:
        yield client
    rcdbroute._STORE = None


def test_list_endpoint(client):
    r = client.get("/api/v1/connectors")
    assert r.status_code == 200
    names = {c["connector"] for c in r.json()["connectors"]}
    assert {"SQLConnector", "WarehouseConnector", "GraphConnector"} <= names


def test_read_and_validate_endpoints(client, tmp_path):
    dbf = str(tmp_path / "shop.db"); _shop_db(dbf)
    uri = "sqlite:///" + dbf
    r = client.post("/api/v1/connectors/read", json={"uri": uri, "weights": True})
    assert r.status_code == 200
    body = r.json()
    assert body["nV"] == 3 and body["nE"] == 2 and body["weighted"]
    r = client.post("/api/v1/connectors/validate", json={"uri": uri})
    assert r.status_code == 200 and r.json()["ok"] is True


def test_ingest_lands_in_rcdb(client, tmp_path):
    dbf = str(tmp_path / "shop.db"); _shop_db(dbf)
    r = client.post("/api/v1/connectors/ingest",
                    json={"uri": "sqlite:///" + dbf, "id": "shop", "tags": ["demo"]})
    assert r.status_code == 200 and r.json()["stored_as"] == "shop"
    ids = [x["id"] for x in client.get("/api/v1/db/list").json()["records"]]
    assert "shop" in ids


def test_in_memory_edges_via_scheme(client):
    r = client.post("/api/v1/connectors/read",
                    json={"scheme": "edges", "source": [["a", "b", 3.0], ["b", "c", 1.0]]})
    assert r.status_code == 200
    assert r.json()["nV"] == 3 and r.json()["nE"] == 2


def test_missing_source_is_400(client):
    r = client.post("/api/v1/connectors/read", json={})
    assert r.status_code == 400
