"""
Backend bridge for the model lifecycle: pick_device rides the rexgraph.compute execution stack,
ComputeSpec.backend threads through the train/load/infer/save lifecycle, and a checkpoint saved on
one backend loads (map_location) and infers on the resolved backend.

Runs on either a CPU-only or a single-GPU host: every GPU move is guarded behind the resolved
device, so the assertions are written against invariants (GPU only when a GPU is actually usable),
not against a fixed device.
"""
import pytest

pytest.importorskip("torch")

from rexgraph import compute
from rexgraph.nn import optim

# (a) pick_device resolves through the compute stack -----------------------------------------------

def test_pick_device_auto_consistent_with_compute_stack():
    dev = optim.pick_device("auto")
    assert isinstance(dev, str) and dev.split(":")[0] in ("cpu", "cuda", "mps")
    # invariant: a non-cpu device only when the compute stack reports a usable GPU
    if dev.split(":")[0] == "cuda":
        assert compute.gpu_count() > 0
    if compute.gpu_count() == 0:
        assert dev == "cpu"
    # None and 'auto' resolve identically
    assert optim.pick_device(None) == dev
    # the resolved device is real: a tensor lives on it
    import torch
    assert str(torch.zeros(1, device=dev).device).split(":")[0] == dev.split(":")[0]


def test_pick_device_explicit_overrides_and_cpu_safety():
    assert optim.pick_device("cpu") == "cpu"            # cpu always forces CPU
    assert optim.pick_device("openmp") == "cpu"         # a CPU compute-backend name maps to cpu
    # a GPU request degrades cleanly on a host with no usable GPU
    if compute.gpu_count() == 0:
        assert optim.pick_device("cuda") == "cpu"
        assert optim.pick_device("rocm") == "cpu"
    else:
        assert optim.pick_device("cuda").split(":")[0] == "cuda"


# (b) lifecycle round-trip: build -> train -> save -> load(map_location) -> infer ------------------

def test_lifecycle_roundtrip_save_load_infer_on_resolved_backend(tmp_path):
    from agent.models import store
    from agent.models import train as T

    from agent import models

    dev = optim.pick_device("auto")
    ckpt = str(tmp_path / "ckpt")

    # build + train a couple of steps on the resolved backend
    model, cfg, bundle = models.build("mlp", seed=0)
    model = model.to(dev)
    bundle.to(dev)
    T.train_one(model, bundle, steps=3, device=dev, seed=0)

    # reference output from the in-memory trained model
    ref, _ = T.predict_on(model, bundle, bundle.kind)

    # SAVE (weights written device-agnostically) then LOAD onto the resolved device
    store.save_checkpoint(ckpt, model, "mlp", cfg, bundle=bundle)
    loaded, conf = store.load_checkpoint(ckpt, device=dev)

    # the loaded model landed on the resolved device
    assert str(next(loaded.parameters()).device).split(":")[0] == dev.split(":")[0]

    # INFER on the resolved backend and reproduce the saved model's output exactly
    bundle.to(dev)
    got, _ = T.predict_on(loaded, bundle, bundle.kind)
    assert (ref == got).all()


def test_predict_maps_checkpoint_onto_resolved_device(tmp_path):
    """The high-level predict() path resolves the device through pick_device and runs there."""
    from agent import models
    dev = optim.pick_device("auto")
    ckpt = str(tmp_path / "m")
    models.run("mlp", steps=3, device=dev, save_to=ckpt)
    p = models.predict(ckpt, device="auto")            # 'auto' rides the compute stack
    assert p["n"] > 0 and p["predictions"].shape[0] == p["n"]
    # forcing CPU always works regardless of host
    assert models.predict(ckpt, device="cpu")["n"] == p["n"]


# (c) ComputeSpec.backend threads through the lifecycle train phase --------------------------------

def test_train_phase_bridges_setup_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    from agent.hive_config import ComputeSpec, HiveProfile

    from agent import hive_config, lifecycle
    hive_config.reset_store()
    lifecycle.reset_store()
    store = hive_config.get_store()

    # a setup pinned to cpu drives training on cpu with NO explicit device param
    store.save(HiveProfile(id="cpuonly", name="C", compute=ComputeSpec(backend="cpu")))
    store.set_active("cpuonly")
    rl = lifecycle.run("train", archetype="mlp", steps=3)
    assert rl.status == "ok" and rl.result["metric"] is not None
    assert any("device=cpu" in s["msg"] for s in rl.steps)

    # an 'auto' setup resolves through the compute stack and still trains
    store.save(HiveProfile(id="autob", name="A", compute=ComputeSpec(backend="auto")))
    store.set_active("autob")
    rl2 = lifecycle.run("train", archetype="mlp", steps=3)
    assert rl2.status == "ok" and rl2.result["metric"] is not None
    resolved = optim.pick_device("auto")
    assert any(f"device={resolved}" in s["msg"] for s in rl2.steps)

    hive_config.reset_store()
    lifecycle.reset_store()
