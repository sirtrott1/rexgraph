"""agent.query_manager: query-as-complex lifecycle, schema mapping, convergence, memory."""
import pytest

from agent import query_manager as qm
from agent import schema_complex as sc
from agent import rcdb


@pytest.fixture
def model():
    return sc.parse_schema_json({"tables": [
        {"name": "customers", "columns": ["id", "name", "email"], "primary_key": ["id"]},
        {"name": "orders", "columns": ["id", "customer_id", "total"], "primary_key": ["id"],
         "foreign_keys": [{"columns": ["customer_id"], "references": "customers"}]},
        {"name": "order_items", "columns": ["order_id", "product_id", "qty"],
         "primary_key": ["order_id", "product_id"],
         "foreign_keys": [{"columns": ["order_id"], "references": "orders"},
                          {"columns": ["product_id"], "references": "products"}]},
        {"name": "products", "columns": ["id", "name", "price"], "primary_key": ["id"]},
        {"name": "suppliers", "columns": ["id", "name"], "primary_key": ["id"]},
    ]})


def test_query_becomes_a_complex():
    mgr = qm.QueryManager()
    s = mgr.open("how do receptors regulate genes and proteins")
    sig = s.current().signature
    assert sig["n_concepts"] >= 3
    assert "betti" in sig                          # it built an actual complex


def test_schema_touch_and_join_path(model):
    mgr = qm.QueryManager(schema=model)
    st = mgr.open("how do customers relate to orders and products").current()
    assert set(st.schema["touched_tables"]) == {"customers", "orders", "products"}
    assert st.schema["joinable"] is True
    assert st.schema["join_path"] and st.schema["join_path"][0] in st.schema["touched_tables"]


def test_disconnected_reference_is_flagged(model):
    mgr = qm.QueryManager(schema=model)
    st = mgr.open("show customers and suppliers together").current()
    assert st.schema["joinable"] is False           # no FK path between them
    assert "suppliers" in st.schema["disconnected_tables"]


def test_plural_singular_matching(model):
    mgr = qm.QueryManager(schema=model)
    st = mgr.open("list orders by customer and product").current()
    assert {"orders", "customers", "products"} <= set(st.schema["touched_tables"])


def test_convergence_converging_vs_drifting():
    mgr = qm.QueryManager()
    s = mgr.open("orders from customers")
    s.evolve("orders from customers by region")
    s.evolve("orders from customers by region and month")
    conv = s.convergence()
    assert conv["trend"] in ("converging", "stable")
    assert s.progressing() is True

    d = mgr.open("orders from customers")
    d.evolve("the weather is sunny with light rain")   # unrelated -> drift
    assert d.convergence()["trend"] == "drifting"
    assert d.progressing() is False


def test_memory_persist_and_recall():
    store = rcdb.MemoryStore()
    mgr = qm.QueryManager(store=store)
    s = mgr.open("how do receptors regulate genes and proteins")
    s.resolve("Receptors bind ligands which regulate gene expression.")
    # the resolved query complex is now in the memory cache
    assert store.get(s.id) is not None
    hits = mgr.recall("what regulates genes and proteins in receptors")
    assert hits and hits[0]["query"] == "how do receptors regulate genes and proteins"
    assert hits[0]["answer"].startswith("Receptors bind")


def test_no_schema_is_graceful():
    mgr = qm.QueryManager()
    st = mgr.open("anything at all here").current()
    assert st.schema == {"linked": False}
