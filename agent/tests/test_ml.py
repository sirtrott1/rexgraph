"""The in-agent model-builder subsystem: agent.models + the /ml routes + train/ingest phases."""
import pytest

pytest.importorskip("torch")

from agent import hive_config, lifecycle, models


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    hive_config.reset_store()
    lifecycle.reset_store()
    yield
    hive_config.reset_store()
    lifecycle.reset_store()


def test_models_subpackage_and_archetypes():
    names = {a["name"] for a in models.list_archetypes()}
    assert {"mlp", "cnn", "lm", "hgnn"} <= names
    # built on rexgraph.nn, with the bridges present
    assert all(hasattr(models, x) for x in ("run", "load_bundle", "to_rcdb",
                                            "bundle_from_core", "core_to_rcdb"))


def test_run_trains_an_archetype():
    r = models.run("mlp", steps=40, device="cpu")
    assert r["archetype"] == "mlp" and r["metric"] is not None


def test_ml_routes(monkeypatch):
    monkeypatch.setenv("RCF_RATE_LIMIT", "0")
    from fastapi.testclient import TestClient
    from agent.server.app import app
    c = TestClient(app)
    arcs = c.get("/api/v1/ml/archetypes").json()["archetypes"]
    assert {a["name"] for a in arcs} >= {"mlp", "cnn", "lm", "hgnn"}
    comp = c.get("/api/v1/ml/components").json()["components"]
    assert "optimizer" in comp and "attention" in comp
    r = c.post("/api/v1/ml/run", json={"archetype": "mlp", "steps": 30}).json()
    assert r["metric"] is not None


def test_train_and_ingest_phases_registered():
    names = {p["name"] for p in lifecycle.phases()}
    assert {"train", "ingest"} <= names


def test_ingest_phase_knowledge_core_to_complex():
    triples = [["Metformin", "treats", "Diabetes"], ["Metformin", "activates", "AMPK"],
               ["AMPK", "regulates", "Glucose"], ["Insulin", "treats", "Diabetes"],
               ["Insulin", "regulates", "Glucose"], ["Diabetes", "affects", "Glucose"]]
    rl = lifecycle.run("ingest", triples=triples,
                       labels={"Metformin": "drug", "Insulin": "drug", "Diabetes": "disease",
                               "AMPK": "gene", "Glucose": "metabolite"})
    assert rl.status == "ok"
    assert rl.result["n_nodes"] == 5 and rl.result["n_classes"] >= 2


def test_train_phase_via_lifecycle():
    rl = lifecycle.run("train", archetype="mlp", steps=30, device="cpu")
    assert rl.status == "ok" and rl.result["metric"] is not None


def test_predict_infers_and_writes_through_io(tmp_path):
    """A trained checkpoint runs on new data and its predictions round-trip through rexgraph.io."""
    import rexgraph.io as rio
    ckpt = str(tmp_path / "m")
    models.run("mlp", steps=30, save_to=ckpt)
    p = models.predict(ckpt)                                  # infer on synthetic data
    assert p["archetype"] == "mlp" and p["n"] > 0
    assert p["predictions"].shape[0] == p["n"] and p["metric"] is not None
    assert models.predict(ckpt, split="test")["n"] < p["n"]  # a held-out split is smaller
    out = str(tmp_path / "preds.safetensors")                # predictions back through the IO layer
    r = models.predict(ckpt, save_to=out)
    V, labels, feat, meta = rio.load_vectors(out)
    assert r["saved_to"] == out and len(V) == p["n"]


def test_pipeline_phase_threads_the_stages():
    """The pipeline phase runs source -> complex -> train -> predict -> hive worker as one op."""
    from agent import lifecycle, hive
    hive.reset_hive()
    triples = [["a", "r", "b"], ["b", "r", "c"], ["c", "r", "a"], ["a", "s", "x"], ["b", "s", "y"]]
    rl = lifecycle.run("pipeline", triples=triples, labels={"a": 0, "b": 1, "c": 0},
                       archetype="hgnn", steps=15, predict=True, worker="kg")
    assert rl.status == "ok"
    assert rl.result["stages"] == ["trustgraph", "train", "predict", "worker"]
    assert rl.result["predict"]["n"] > 0
    assert "kg" in {b["name"] for b in hive.get_hive().status()["bees"]}   # model joined the hive
    assert "pipeline" in {p["name"] for p in lifecycle.phases()}


def test_pipeline_phase_needs_a_source():
    from agent import lifecycle
    rl = lifecycle.run("pipeline", archetype="mlp")
    assert rl.status == "ok" and "skipped" in rl.result


def test_train_hardening_schedule_accum_resume(tmp_path):
    """Training loop: lr schedule + warmup, gradient accumulation, resume, and amp no-op on CPU."""
    from agent.models.train import _lr_at
    r = models.run("mlp", steps=20, schedule="cosine", warmup=3, grad_accum=2)
    assert r["metric"] is not None
    assert _lr_at(0, 20, 1.0, "cosine", 4) < _lr_at(4, 20, 1.0, "cosine", 4)    # warmup rises
    assert _lr_at(19, 20, 1.0, "cosine", 4) < _lr_at(4, 20, 1.0, "cosine", 4)   # then decays
    ck = str(tmp_path / "m")
    models.run("mlp", steps=10, save_to=ck)
    assert models.run("mlp", steps=10, resume=ck)["metric"] is not None         # continue training
    assert models.run("mlp", steps=5, amp=True, device="cpu")["metric"] is not None  # amp no-op on cpu


def test_compute_config_flows_through_setup_and_operation(tmp_path, monkeypatch):
    """A setup's compute config round-trips, and every operation applies it (run-logged)."""
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    from agent import hive_config, lifecycle
    from agent.hive_config import HiveProfile, ComputeSpec
    from rexgraph import compute
    # schema round-trips (incl. back-compat: a profile with no compute section)
    p = HiveProfile.from_dict(HiveProfile(id="x", name="X", compute=ComputeSpec(threads=6)).to_dict())
    assert p.compute.threads == 6
    assert HiveProfile.from_dict({"id": "o", "name": "O"}).compute.backend == "auto"
    # an operation applies the active setup's compute config, run-logged
    store = hive_config.get_store()
    store.save(HiveProfile(id="capped", name="Capped", compute=ComputeSpec(threads=2)))
    store.set_active("capped")
    compute.set_threads(None)
    rl = lifecycle.run("test")
    assert any(s["msg"].startswith("compute: threads=2") for s in rl.steps)
    assert compute.get_threads() == 2
