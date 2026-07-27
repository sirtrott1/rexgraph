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
