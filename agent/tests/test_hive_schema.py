"""agent.hive_schema: the hive's structure as a versioned, cause-tagged complex."""
from agent import hive as hivemod, agent_complex, rcdb
from agent.hive_schema import HiveSchema


def _hive():
    hivemod.reset_hive(); agent_complex.reset_live()
    h = hivemod.get_hive()
    h.attach("lead", "http://x", role="queen", model="m", specialties=["coordinate"])
    return h


def test_initial_snapshot_captures_structure():
    h = _hive()
    hs = HiveSchema(h, store=rcdb.MemoryStore())
    v = hs.snapshot(cause="init")
    assert v.get("unchanged") is False
    rex, meta = hs.complex()
    assert rex is not None
    assert "hive" in meta["vertex_labels"] and "lead" in meta["vertex_labels"]


def test_schema_versions_on_structural_change():
    h = _hive()
    hs = HiveSchema(h, store=rcdb.MemoryStore())
    hs.snapshot(cause="init")
    # a new task deploys a worker -> the self-schema changes -> a new version
    h.add_worker("coder", lambda d, **k: d, capability="analyze", worker_type="analyzer:code")
    v2 = hs.snapshot(cause="new task: code generation")
    assert v2["unchanged"] is False
    # new data attaches a database -> another version
    v3 = hs.attach_resource("shop_db", "database", links=[("coder", "reads")],
                            cause="new data: attached shop_db")
    assert v3["unchanged"] is False
    # nothing changed -> no spurious version
    assert hs.snapshot(cause="noop")["unchanged"] is True

    evo = hs.evolution()
    assert [e["version"] for e in evo] == [1, 2, 3]
    assert evo[1]["cause"].startswith("new task")
    assert evo[2]["cause"].startswith("new data")
    # the structure genuinely grew
    assert evo[0]["n_nodes"] < evo[2]["n_nodes"]


def test_resource_appears_in_the_complex():
    h = _hive()
    hs = HiveSchema(h, store=rcdb.MemoryStore())
    hs.attach_resource("warehouse", "store", links=[("lead", "writes")], cause="new store")
    _, meta = hs.complex()
    assert "warehouse" in meta["vertex_labels"]


def test_detach_versions_too():
    h = _hive()
    hs = HiveSchema(h, store=rcdb.MemoryStore())
    hs.snapshot(cause="init")
    hs.attach_resource("db1", "database", cause="attach")
    v = hs.detach_resource("db1", cause="db1 decommissioned")
    assert v["unchanged"] is False
    assert "db1" not in hs.complex()[1]["vertex_labels"]
