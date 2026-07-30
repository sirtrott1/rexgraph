import functools

from agent.coordinator_adapter import work_units


def _echo(tag, i):
    return (tag, i)


def test_kind_maps_to_coordinator_type():
    tasks = [
        {"id": "a", "kind": "llm_chat", "fn": (lambda: 1)},
        {"id": "b", "kind": "monitor", "fn": (lambda: 2)},
        {"id": "c", "kind": "heat_kernel", "fn": (lambda: 3)},
        {"id": "d", "kind": "mystery", "fn": (lambda: 4)},
    ]
    u = {x["id"]: x for x in work_units(tasks)}
    assert u["a"]["type"] == "io_llm"
    assert u["b"]["type"] == "cpu_coordination"
    assert u["c"]["type"] == "gpu_kernel"
    assert u["d"]["type"] == "cpu_coordination"
    assert callable(u["a"]["fn"])


def test_end_to_end_mixed_wave_through_coordinator():
    from rexgraph.coordinator import Coordinator
    co = Coordinator()
    # cpu tasks land on the proc (process-pool) lane, so their fn must be picklable -> partials
    tasks = ([{"id": f"io{i}", "kind": "llm_chat", "fn": functools.partial(_echo, "io", i)} for i in range(6)]
             + [{"id": f"cp{i}", "kind": "monitor", "fn": functools.partial(_echo, "cp", i)} for i in range(4)])
    res = co.run_wave(work_units(tasks))
    assert res["io0"] == ("io", 0) and res["cp3"] == ("cp", 3)
    assert len(res) == 10


def test_to_type_handles_a_train_kind_without_the_repo_root_on_the_path():
    """`agent.*` is a doubled prefix: the package is agent/agent/, so it imports
    as agent.*. The doubled form resolved only when the cwd was the repo root, so a
    train: task kind raised ModuleNotFoundError for any normal consumer."""
    from agent.coordinator_adapter import _to_type

    assert _to_type("analyze") == "cpu_coordination"
    # the train: branch must classify, not raise
    assert _to_type("train:hgnn") in ("cpu_coordination", "gpu_kernel")
    assert _to_type("train:mlp") in ("cpu_coordination", "gpu_kernel")


def test_work_units_accepts_a_train_task():
    """work_units is what hive.py imports; it must classify a training task."""
    from agent.coordinator_adapter import work_units

    units = work_units([{"id": "t1", "kind": "train:hgnn", "fn": lambda: None}])
    assert len(units) == 1
    assert units[0]["type"] in ("cpu_coordination", "gpu_kernel")


def test_no_product_module_uses_the_doubled_agent_prefix():
    """Guard the whole package: a doubled prefix only works from the repo root, so it
    is a latent crash wherever it appears."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "agent" / "agent"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r"^\s*(from|import)\s+agent\.agent\b", line):
                offenders.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
    assert offenders == [], "doubled agent.agent prefix:\n" + "\n".join(offenders)
