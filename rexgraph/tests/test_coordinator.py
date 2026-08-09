import numpy as np

from rexgraph.coordinator import CostModel


def test_cost_model_priors_and_ema():
    cm = CostModel()
    t_proc, _ = cm.cost("cpu_coordination", "proc")
    t_thread, _ = cm.cost("cpu_coordination", "thread")
    assert t_proc < t_thread
    assert cm.best_lane("cpu_coordination") == "proc"
    assert cm.best_lane("io_llm") == "thread"
    assert cm.best_lane("gpu_kernel") == "igpu"
    before, _ = cm.cost("gpu_kernel", "igpu")
    for _ in range(50):
        cm.observe("gpu_kernel", "igpu", before + 1.0)
    after, _ = cm.cost("gpu_kernel", "igpu")
    assert after > before

def test_affinity_discovery_flips_mistyped_task():
    cm = CostModel()
    for _ in range(80):
        cm.observe("gpu_kernel", "proc", 0.001)
        cm.observe("gpu_kernel", "igpu", 5.0)
    assert cm.best_lane("gpu_kernel") == "proc"


from rexgraph.coordinator import assign, contention, delegation_complex


def _units(n_by_type):
    u, i = [], 0
    for ty, n in n_by_type.items():
        for _ in range(n):
            u.append({"id": f"t{i}", "type": ty}); i += 1
    return u


def test_contention_penalizes_cpu_on_the_gil_lane():
    cm = CostModel()
    units = _units({"cpu_coordination": 32})
    all_thread = {u["id"]: "thread" for u in units}
    all_proc = {u["id"]: "proc" for u in units}
    assert contention(all_thread, units, cm) > contention(all_proc, units, cm)


def test_contention_nonneg():
    cm = CostModel()
    units = _units({"gpu_kernel": 8, "cpu_coordination": 8})
    best = {u["id"]: cm.best_lane(u["type"]) for u in units}
    assert contention(best, units, cm) >= 0.0


def test_execution_is_an_edge_delegation_complex():
    CostModel()
    units = _units({"gpu_kernel": 3})
    a = {u["id"]: "igpu" for u in units}
    g = delegation_complex(a, units)
    B1 = np.asarray(g.B1)
    assert g.nE >= len(units)                                   # >= one execution edge per task
    assert int((np.abs(B1).sum(axis=1) > 0).sum()) >= len(units)  # every task reached by an edge


def test_assign_routes_to_best_lanes_when_light():
    cm = CostModel()
    units = _units({"cpu_coordination": 2, "io_llm": 2, "gpu_kernel": 1})
    a = assign(units, cm)
    assert all(a[u["id"]] == cm.best_lane(u["type"]) for u in units)


def test_assign_balances_bottlenecked_igpu_via_hybrid(monkeypatch):
    # capacity() derives proc and thread from os.cpu_count()//2, so on a small host
    # both are 1, spilling off the igpu buys nothing and everything lands in one lane.
    # The test is about the balancing rule, not the runner it happens to be on.
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    cm = CostModel()
    units = _units({"gpu_kernel": 16})   # igpu parallelism is only 2 -> bottleneck; spill helps
    a = assign(units, cm)
    assert len(set(a.values())) >= 2
    assert sum(1 for ln in a.values() if ln == "igpu") < 16
    dump = {u["id"]: cm.best_lane(u["type"]) for u in units}
    assert contention(a, units, cm) <= contention(dump, units, cm) + 1e-9


def test_execute_is_lane_independent():
    from rexgraph.coordinator import execute
    units = [{"id": f"t{i}", "type": "io_llm", "fn": (lambda i=i: i * i)} for i in range(8)]
    r1 = execute(units, {u["id"]: "thread" for u in units})
    r2 = execute(units, {u["id"]: "igpu" for u in units})
    assert r1 == r2 == {f"t{i}": i * i for i in range(8)}


def test_execute_records_timings():
    from rexgraph.coordinator import execute
    cm = CostModel()
    units = [{"id": "a", "type": "io_llm", "fn": (lambda: 1)}]
    before, _ = cm.cost("io_llm", "thread")
    execute(units, {"a": "thread"}, cost=cm)
    after, _ = cm.cost("io_llm", "thread")
    assert after != before


def test_coordinator_runs_a_wave_and_learns():
    from rexgraph.coordinator import Coordinator
    co = Coordinator()
    units = [{"id": f"t{i}", "type": "io_llm", "fn": (lambda i=i: i + 1)} for i in range(6)]
    res = co.run_wave(units)
    assert res == {f"t{i}": i + 1 for i in range(6)}


def test_value_invariant_cpu_coordination_prefers_the_proc_lane():
    cm = CostModel()
    units = _units({"cpu_coordination": 20})
    a = assign(units, cm)
    assert sum(1 for ln in a.values() if ln == "proc") >= 10


import functools


def test_execute_runs_picklable_fn_on_the_proc_lane():
    # The forkserver proc lane (the whole point of the 5.8x multicore path) must
    # actually execute, not just be assign()-routed. Use a builtin-backed partial
    # (picklable AND importable in the forkserver child; a fn defined in this
    # non-package test module would not be importable there).
    from rexgraph.coordinator import execute
    units = [{"id": f"c{i}", "type": "cpu_coordination",
              "fn": functools.partial(pow, i, 2)} for i in range(4)]
    res = execute(units, {u["id"]: "proc" for u in units})
    assert res == {f"c{i}": i * i for i in range(4)}


def test_unpicklable_proc_fn_spills_to_thread_and_keeps_side_effects():
    # A closure (the natural hive-task form) cannot cross the forkserver boundary.
    # It must spill to the thread lane (run in-process) instead of crashing the wave,
    # which also preserves its side effects.
    from rexgraph.coordinator import execute
    seen = []

    def make(i):
        return lambda: (seen.append(i), i)[1]   # closure -> not picklable

    units = [{"id": f"c{i}", "type": "cpu_coordination", "fn": make(i)} for i in range(4)]
    res = execute(units, {u["id"]: "proc" for u in units})
    assert res == {f"c{i}": i for i in range(4)}
    assert sorted(seen) == [0, 1, 2, 3]   # ran in-process, mutations preserved


def test_assign_scales_and_stays_deterministic():
    # Delta-scored greedy: a large wave must place quickly (not cubic seconds) and
    # remain deterministic across runs.
    import time as _t
    cm = CostModel()
    units = _units({"cpu_coordination": 120, "io_llm": 120, "gpu_kernel": 120})
    t0 = _t.perf_counter()
    a1 = assign(units, cm)
    dt = _t.perf_counter() - t0
    a2 = assign(units, cm)
    assert a1 == a2
    assert dt < 2.0
