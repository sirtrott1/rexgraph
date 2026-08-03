import rexgraph.coordinator as co
from rexgraph.coordinator import CostModel, assign, capacity, contention


def _units(specs):
    # specs: list of (type, weight); ids auto-assigned
    return [{"id": f"t{i}", "type": t, "weight": w} for i, (t, w) in enumerate(specs)]


def test_weight_absent_defaults_to_one_and_matches_old_behavior():
    cm = CostModel()
    u = [{"id": "a", "type": "io_llm"}]  # no weight key
    a = assign(u, cm)
    assert a["a"] == cm.best_lane("io_llm")


def test_high_weight_task_keeps_the_fast_lane_when_a_lane_bottlenecks():
    # Many gpu_kernel tasks bottleneck the igpu (parallelism 2). A high-weight one should stay on
    # igpu (its best lane) while low-weight peers spill off.
    cm = CostModel()
    specs = [("gpu_kernel", 1.0)] * 15 + [("gpu_kernel", 50.0)]
    u = _units(specs)
    a = assign(u, cm)
    heavy = u[-1]["id"]
    assert a[heavy] == "igpu"  # the prioritized task holds the fast lane


def test_capacity_share_splits_worker_counts():
    co.reset_shares()
    co.register_hive_share("A", 2.0)
    co.register_hive_share("B", 1.0)
    capA = capacity(co.share_fraction("A"))
    capB = capacity(co.share_fraction("B"))
    assert capA["proc"] >= capB["proc"]
    assert capA["proc"] + capB["proc"] <= capacity()["proc"] + 1  # split, not doubled
    co.reset_shares()


def test_lone_hive_gets_full_capacity():
    co.reset_shares()
    co.register_hive_share("solo", 5.0)
    assert capacity(co.share_fraction("solo")) == capacity()
    co.reset_shares()


def test_contention_accepts_explicit_capacity():
    cm = CostModel()
    u = _units([("cpu_coordination", 1.0)] * 4)
    a = {x["id"]: "proc" for x in u}
    small = {"proc": 1.0, "thread": 1.0, "igpu": 1.0}
    assert contention(a, u, cm, cap=small) >= contention(a, u, cm)
