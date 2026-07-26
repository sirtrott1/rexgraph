import numpy as np
from rexgraph.coordinator import CostModel, LANES, TYPES

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


from rexgraph.coordinator import contention, capacity, delegation_complex, assign


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
    cm = CostModel()
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


def test_assign_balances_bottlenecked_igpu_via_hybrid():
    cm = CostModel()
    units = _units({"gpu_kernel": 16})   # igpu parallelism is only 2 -> bottleneck; spill helps
    a = assign(units, cm)
    assert len(set(a.values())) >= 2
    assert sum(1 for ln in a.values() if ln == "igpu") < 16
    dump = {u["id"]: cm.best_lane(u["type"]) for u in units}
    assert contention(a, units, cm) <= contention(dump, units, cm) + 1e-9
