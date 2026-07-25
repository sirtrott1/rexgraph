"""agent.foundry: LMs forge NN worker bees; device placement; the control hierarchy."""
import pytest

from agent import hive as hivemod, agent_complex
from agent import foundry as fo_mod
from agent.foundry import ModelFoundry, resolve_device, hierarchy, choose_archetype, bundle_from_rows


def test_resolve_device_explicit_and_auto(monkeypatch):
    assert resolve_device("mlp", "cpu") == "cpu"                  # explicit request wins
    assert resolve_device("cnn", "auto") == "cpu"                 # conv stays on the CPU
    import rexgraph.compute as compute
    monkeypatch.setattr(compute, "recommended_backend", lambda: "rocm")
    assert resolve_device("mlp", "auto") == "cuda"               # an iGPU/GPU is available
    monkeypatch.setattr(compute, "recommended_backend", lambda: "cpu")
    assert resolve_device("mlp", "auto") == "cpu"


def _hive():
    hivemod.reset_hive(); agent_complex.reset_live()
    return hivemod.get_hive()


def test_forge_registers_and_invokes_an_nn(tmp_path):
    pytest.importorskip("torch")
    h = _hive()
    foundry = ModelFoundry(h, store_dir=str(tmp_path))
    card = foundry.forge("net", "mlp", steps=3, device="cpu")
    assert card["archetype"] == "mlp" and card["device"] == "cpu"
    bee = h.get("net")
    assert bee is not None and bee.worker_type == "model:mlp" and bee.capability == "predict"
    # an LM drives the network it built
    out = foundry.invoke("net")
    assert out["archetype"] == "mlp" and out["n"] > 0
    assert foundry.roster()[0]["name"] == "net"


def test_hierarchy_shows_lms_over_the_networks(tmp_path):
    pytest.importorskip("torch")
    h = _hive()
    h.attach("lead", "http://x", role="queen", model="m", specialties=["coordinate"])   # an LM
    foundry = ModelFoundry(h, store_dir=str(tmp_path))
    foundry.forge_many([{"name": "n1", "archetype": "mlp", "steps": 3, "device": "cpu"},
                        {"name": "n2", "archetype": "mlp", "steps": 3, "device": "cpu"}])
    hier = hierarchy(h)
    assert any(c["name"] == "lead" for c in hier["controllers"])          # LM on top
    assert {n["name"] for n in hier["networks"]} == {"n1", "n2"}          # the NNs beneath
    assert len(foundry.roster()) == 2


def test_choose_archetype_structural():
    assert choose_archetype("classify these images of cats") == "cnn"
    assert choose_archetype("node classification on a graph") == "hgnn"
    assert choose_archetype("model this text token sequence") == "lm"
    assert choose_archetype("predict churn from tabular features") == "mlp"
    assert choose_archetype("something vague") == "mlp"          # default

    class Bundle:
        kind = "image"
    assert choose_archetype("", Bundle()) == "cnn"               # the data's kind wins


def test_forge_from_task_coder_chooses(tmp_path):
    pytest.importorskip("torch")
    h = _hive()
    foundry = ModelFoundry(h, store_dir=str(tmp_path))
    card = foundry.forge_from_task("net", "predict a target", steps=3, device="cpu",
                                   coder=lambda p: '{"archetype": "mlp", "params": {}}')
    assert card["archetype"] == "mlp" and card["chosen_by"] == "coder"
    assert h.get("net") is not None and card["task"] == "predict a target"


def test_forge_from_task_falls_back_to_heuristic(tmp_path):
    pytest.importorskip("torch")
    h = _hive()
    foundry = ModelFoundry(h, store_dir=str(tmp_path))
    # the coder returns no valid JSON -> the structural heuristic decides from the task
    card = foundry.forge_from_task("net", "classify these images", steps=3, device="cpu",
                                   coder=lambda p: "hmm, not sure")
    assert card["chosen_by"] == "heuristic" and card["archetype"] == "cnn"


def test_bundle_from_rows_shapes():
    pytest.importorskip("torch")
    rows = [{"a": 1, "b": "x", "y": "cat"}, {"a": 2, "b": "y", "y": "dog"},
            {"a": 3, "b": "x", "y": "cat"}]
    b = bundle_from_rows(rows, target="y", features=["a", "b"])
    assert b.kind == "vector"
    assert b.meta["feat_dim"] == 2 and b.meta["n_classes"] == 2   # cat/dog -> 2 classes; 'b' encoded


def test_forge_on_rows_trains_on_real_data(tmp_path):
    pytest.importorskip("torch")
    import random
    random.seed(0)
    rows = []
    for _ in range(120):
        f1, f2 = random.random(), random.random()
        rows.append({"f1": f1, "f2": f2, "label": 1 if f1 > 0.5 else 0})   # learnable pattern
    h = _hive()
    foundry = ModelFoundry(h, store_dir=str(tmp_path))
    card = foundry.forge_on_rows("net", rows, target="label", features=["f1", "f2"],
                                 steps=30, device="cpu")
    assert card["archetype"] == "mlp" and card["trained_on"] == "rows" and card["n_rows"] == 120
    assert card["metric"] is not None                            # trained on the actual data
    bee = h.get("net")
    assert bee is not None and bee.worker_type == "model:mlp"


def test_forge_falls_back_to_cpu_when_gpu_unusable(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    h = _hive()
    foundry = ModelFoundry(h, store_dir=str(tmp_path))
    # force 'auto' to pick cuda, then make the cuda training path raise once
    import rexgraph.compute as compute
    monkeypatch.setattr(compute, "recommended_backend", lambda: "rocm")
    from agent import models
    real_run = models.run
    calls = {"n": 0}

    def flaky(archetype, **kw):
        calls["n"] += 1
        if kw.get("device") == "cuda":
            raise RuntimeError("gpu not really usable")
        return real_run(archetype, **kw)

    monkeypatch.setattr(models, "run", flaky)
    card = foundry.forge("net", "mlp", steps=3, device="auto")
    assert card["device"] == "cpu" and calls["n"] == 2                    # tried gpu, degraded to cpu
    assert h.get("net") is not None
