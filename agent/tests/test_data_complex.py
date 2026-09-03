"""agent.data_complex: records as a relational complex - clusters, outliers, centrality."""
import sqlite3

from agent.agentic_db import AgenticDB
from agent.data_complex import analyze_rows, rows_to_complex


def test_records_cluster_by_shared_value():
    rows = [{"id": f"o{i}", "cust": c} for i, c in enumerate(["A", "A", "A", "B", "B", "C"])]
    r = analyze_rows(rows, link_on="cust", id_col="id")
    assert r["n_rows"] == 6 and r["n_clusters"] == 3
    assert set(r["clusters"][0]) == {"o0", "o1", "o2"}       # the largest cluster first
    assert "o5" in r["outliers"]                             # cust C: a single record, isolated
    assert r["central"]                                      # coherence centrality reported


def test_no_shared_values_all_outliers():
    rows = [{"id": "a", "k": 1}, {"id": "b", "k": 2}]
    r = analyze_rows(rows, link_on="k", id_col="id")
    assert r["n_clusters"] == 2 and set(r["outliers"]) == {"a", "b"}
    # Records with no shared value are still declared participants, so the complex exists
    # with two grade-zero cells and no relation. This previously asserted None, on the rule
    # that no edges meant no complex; that rule dropped every unlinked record and made the
    # complex an incomplete picture of the record set. An unlinked record is a participant
    # in nothing, which is a fact the complex can hold, and not a reason for it to be absent.
    rex, meta = rows_to_complex(rows, link_on="k")
    assert int(rex.nV) == 2 and int(rex.nE) == 0
    assert int(rex.betti[0]) == 2
    # no id_col on this call, so participants are labelled by row index
    assert meta["unattached_participants"] == ["0", "1"]


def test_agentic_db_data_complex(tmp_path):
    path = tmp_path / "shop.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER);
        INSERT INTO orders VALUES (1,10),(2,10),(3,10),(4,20),(5,30);
    """)
    con.commit(); con.close()
    # AgenticDB owns a connection pool, so the test that opens it closes it
    with AgenticDB(f"sqlite:///{path}") as db:
        r = db.data_complex("orders", link_on="customer_id", id_col="id")
        assert r["n_rows"] == 5
        assert len(r["clusters"][0]) == 3                        # customer 10's three orders
        assert {"4", "5"} <= set(r["outliers"])                 # customers 20 and 30: one order each
        assert r["source"] == "orders"
