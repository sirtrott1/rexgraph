"""Tests for agent.connectors.service and the rexgraph-connect CLI."""

from __future__ import annotations

import sqlite3

import pytest

from agent.connectors import service as svc
from agent.cli import connect as cli


def _shop_db(path):
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE customers(id INTEGER PRIMARY KEY);"
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, "
        "  customer_id INT REFERENCES customers(id));"
        "CREATE TABLE order_items(order_id INT REFERENCES orders(id), "
        "  product_id INT, PRIMARY KEY(order_id, product_id));")
    con.executemany("INSERT INTO customers(id) VALUES(?)", [(i,) for i in range(30)])
    con.executemany("INSERT INTO orders(id,customer_id) VALUES(?,0)",
                    [(i,) for i in range(100)])
    con.executemany("INSERT INTO order_items(order_id,product_id) VALUES(?,?)",
                    [(i, i % 5) for i in range(400)])
    con.commit(); con.close()


# service layer

def test_list_connectors_groups_and_reports_capabilities():
    rows = svc.list_connectors()
    by = {r["connector"]: r for r in rows}
    assert "SQLConnector" in by and "WarehouseConnector" in by
    sql = by["SQLConnector"]
    schemes = {s["scheme"] for s in sql["schemes"]}
    assert {"sqlite", "postgresql", "mysql"} <= schemes
    assert sql["supports"]["modality"] and sql["supports"]["faces"]
    # sqlite driver is present in-env; every scheme carries a driver flag
    assert all("driver_available" in s for s in sql["schemes"])
    assert next(s for s in sql["schemes"] if s["scheme"] == "sqlite")["driver_available"]


def test_driver_status_flags_missing_with_hint():
    inmem = svc.driver_status("ontology")
    assert inmem["available"] and inmem["hint"] == ""
    wh = svc.driver_status("snowflake")
    assert wh["available"] is False and "warehouse" in wh["hint"]


def test_read_validate_ingest_roundtrip(tmp_path):
    dbf = str(tmp_path / "shop.db")
    _shop_db(dbf)
    uri = "sqlite:///" + dbf

    summ = svc.read(uri, with_weights=True)
    assert summ["nV"] == 3 and summ["nE"] == 2 and summ["weighted"]

    assert svc.validate(uri).ok

    store = "sqlite:///" + str(tmp_path / "rcdb.sqlite")
    out = svc.ingest(uri, "shop", store_uri=store, tags=["demo"])
    assert out["stored_as"] == "shop"
    # persisted structure is retrievable from a freshly opened store
    from agent.rcdb import open_store
    got = open_store(store).get("shop")
    assert got is not None and got.nV == 3 and got.nE == 2


def test_read_in_memory_shape():
    summ = svc.read("edges", source=[("x", "y", 2.0), ("y", "z", 1.0)])
    assert summ["nV"] == 3 and summ["nE"] == 2 and summ["weighted"]


# CLI

def test_cli_list_runs(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "SQLConnector" in out and "driver missing" in out


def test_cli_read_and_validate(tmp_path, capsys):
    dbf = str(tmp_path / "shop.db")
    _shop_db(dbf)
    uri = "sqlite:///" + dbf
    assert cli.main(["read", uri, "--weights"]) == 0
    assert "nV=3" in capsys.readouterr().out
    assert cli.main(["validate", uri]) == 0        # PASS -> exit 0


def test_cli_ingest(tmp_path):
    dbf = str(tmp_path / "shop.db")
    _shop_db(dbf)
    store = "sqlite:///" + str(tmp_path / "rcdb.sqlite")
    assert cli.main(["ingest", "sqlite:///" + dbf, "--store", store,
                     "--id", "shop", "--tags", "a,b"]) == 0
    from agent.rcdb import open_store
    assert open_store(store).get("shop") is not None


def test_cli_resolves_saved_connection(tmp_path, monkeypatch):
    # a saved-connection name resolves to its URI via the SecretStore
    secrets_file = tmp_path / "conns.json"
    monkeypatch.setenv("REXGRAPH_SECRETS_URI", "file://" + str(secrets_file))
    from agent.secrets import open_secret_store
    dbf = str(tmp_path / "shop.db")
    _shop_db(dbf)
    open_secret_store().put("shopdb", "sqlite:///" + dbf, "sql")
    assert cli._resolve("shopdb") == "sqlite:///" + dbf
