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
