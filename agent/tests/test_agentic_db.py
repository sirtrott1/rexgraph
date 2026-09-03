"""agent.agentic_db: a live SQLite database bridged into the RCDB + hive."""
import sqlite3

import pytest
from agent.agentic_db import AgenticDB

from agent import agent_complex, hive, rcdb


@pytest.fixture
def db_url(tmp_path):
    path = tmp_path / "shop.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(id));
        CREATE TABLE order_items (order_id INTEGER, product_id INTEGER, qty INTEGER,
            PRIMARY KEY (order_id, product_id),
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id));
        CREATE TABLE suppliers (id INTEGER PRIMARY KEY, name TEXT);
        INSERT INTO customers VALUES (1, 'Ada', 'ada@x.io');
        INSERT INTO products VALUES (1, 'Widget', 9.99);
        INSERT INTO orders VALUES (1, 1, 9.99);
        INSERT INTO order_items VALUES (1, 1, 3);
        INSERT INTO suppliers VALUES (1, 'Acme');
    """)
    con.commit(); con.close()
    return f"sqlite:///{path}"


@pytest.fixture
def make_db():
    """Build AgenticDB instances and close every one when the test ends.

    AgenticDB owns a connection pool. Each test built one or two inline and nothing
    released them, so the pools sat until the collector reached them and surfaced as
    warnings attributed to whatever ran next. A factory keeps that ownership in one
    place instead of a try/finally in every test.
    """
    made = []

    def _make(*args, **kwargs):
        db = AgenticDB(*args, **kwargs)
        made.append(db)
        return db

    yield _make
    for db in made:
        db.close()


def test_schema_reflected_into_rcdb(db_url, make_db):
    store = rcdb.MemoryStore()
    db = make_db(db_url, store=store)
    # the schema complex is now in the RCDB memory
    assert store.get(db.schema_id) is not None
    names = {t["name"] for t in db.tables()}
    assert {"customers", "orders", "order_items", "products", "suppliers"} <= names


def test_health_is_topological(db_url, make_db):
    db = make_db(db_url)
    h = db.health()
    assert h["n_tables"] >= 5
    assert h["betti"] is not None                       # a real complex was analyzed


def test_search_builds_join_from_fk_graph(db_url, make_db):
    db = make_db(db_url)
    r = db.search("orders from customers")
    assert "JOIN customers" in r["sql"] and "orders.customer_id = customers.id" in r["sql"]
    assert r["n"] == 1 and r["rows"][0]["name"] == "Ada"


def test_search_auto_inserts_junction_table(db_url, make_db):
    db = make_db(db_url)
    r = db.search("customers and products")             # not directly related
    # the join plan must route through orders + order_items (the junction)
    assert set(r["join_tables"]) == {"customers", "orders", "order_items", "products"}
    assert r["n"] == 1


def test_search_refuses_unjoinable_reference(db_url, make_db):
    db = make_db(db_url)
    r = db.search("customers and suppliers")
    assert "error" in r and "unjoinable" in r["error"]
    assert "suppliers" in r.get("disconnected", [])


def test_extract_reads_one_table(db_url, make_db):
    db = make_db(db_url)
    r = db.extract("products", limit=10)
    assert r["n"] == 1 and r["rows"][0]["name"] == "Widget"
    assert db.extract("nope")["error"]


def test_modify_is_guarded(db_url, make_db):
    ro = make_db(db_url)                               # read-only by default
    assert ro.modify("INSERT INTO suppliers VALUES (2, 'Beta')")["ok"] is False

    rw = make_db(db_url, writable=True)
    assert rw.modify("DROP TABLE suppliers")["ok"] is False           # DDL blocked
    assert rw.modify("UPDATE x; DELETE y")["ok"] is False             # multi-statement blocked
    ok = rw.modify("INSERT INTO suppliers VALUES (2, 'Beta')")
    assert ok["ok"] is True and ok["rowcount"] == 1
    assert rw.extract("suppliers")["n"] == 2                          # the write landed


def test_attach_to_hive_and_invoke(db_url, make_db):
    hive.reset_hive(); agent_complex.reset_live()
    h = hive.get_hive()
    db = make_db(db_url)
    names = db.attach_to_hive(h, prefix="shop")
    assert "shop.search" in names and "shop.modify" not in names     # read-only: no write bee
    # an agent operates the database through the swarm
    out = h.invoke("shop.search", "orders from customers")
    assert out["n"] == 1
    # the invocation was recorded in the coordination complex
    assert h.monitor()["n_interactions"] >= 1
