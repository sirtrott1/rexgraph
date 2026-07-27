import numpy as np
from agent.agent.warehouse.assemble import assemble
from agent.agent.rcdb import FileStore

FIXTURE = ("src\tx1\tdst\tx2\tw\n" +
           "".join(f"A{t}\t.\tB{l}\t.\t{0.1 + 0.7*l + 3.0*t}\n"
                   for t in range(1, 5) for l in range(1, 9)))


def _fixture(tmp_path):
    p = tmp_path / "e.tsv"; p.write_text(FIXTURE); return str(p)


def test_assemble_runs_end_to_end(tmp_path):
    store = FileStore(str(tmp_path / "rcdb"))
    sweep = [{"archetype": "hgnn", "params": {"d_hid": 8, "n_layers": 1}, "seed": 0},
             {"archetype": "hgnn", "params": {"d_hid": 16, "n_layers": 1}, "seed": 1}]
    rep = assemble(_fixture(tmp_path), store=store, source="src", target="dst", weight="w",
                   n_tiers=2, sweep=sweep, steps=30)
    assert len(rep["tiers"]) >= 1
    for t in rep["tiers"]:
        assert t["best"] is not None and np.isfinite(t["best"]["metric"])
        assert t["rcdb_id"] in [r.id for r in store.list(limit=100)]     # persisted
        assert t["best"]["bee"] is not None                              # deployed as a bee


def test_assemble_survives_a_failing_config(tmp_path):
    store = FileStore(str(tmp_path / "rcdb"))
    sweep = [{"archetype": "hgnn", "params": {"d_hid": 8, "n_layers": 1}, "seed": 0},
             {"archetype": "hgnn", "params": {"d_hid": -1}, "seed": 1}]   # invalid -> training fails
    rep = assemble(_fixture(tmp_path), store=store, source="src", target="dst", weight="w",
                   n_tiers=1, sweep=sweep, steps=20)
    assert rep["tiers"][0]["best"] is not None    # the good config still wins; wave not sunk


def test_rcdb_record_carries_complex_and_types(tmp_path):
    store = FileStore(str(tmp_path / "rcdb"))
    rep = assemble(_fixture(tmp_path), store=store, source="src", target="dst", weight="w",
                   n_tiers=1,
                   sweep=[{"archetype": "hgnn", "params": {"d_hid": 8, "n_layers": 1}, "seed": 0}],
                   steps=20)
    rid = rep["tiers"][0]["rcdb_id"]
    rec = store.get_record(rid)
    assert rec is not None and rec.signature is not None and rec.signature.get("betti") is not None  # structural signature present
    assert "col_types" in rec.to_dict().get("meta", {})        # typed context travels with the record


def test_live_path_imports_no_pandas():
    # A fresh interpreter importing the warehouse live path must not drag in pandas. Running in a
    # subprocess makes this a real assertion (an in-process check would be a no-op cache hit, since
    # the modules are already imported by the tests above).
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import sys
        import agent.agent.warehouse.source
        import agent.agent.warehouse.assemble
        import agent.agent.warehouse.foundry_tasks
        assert "pandas" not in sys.modules, "warehouse live path imported pandas"
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"live-path import pulled in pandas or failed:\n{r.stdout}\n{r.stderr}"
