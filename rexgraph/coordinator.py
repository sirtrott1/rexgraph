"""Hive Coordinator (v1): place tasks onto compute lanes by minimizing a relational-complex
contention objective. See rexgraph-internal spec 2026-07-25-hive-coordinator-design.md."""
from __future__ import annotations

LANES = ("proc", "thread", "igpu")
TYPES = ("cpu_coordination", "io_llm", "gpu_kernel")

# (time_s prior, bandwidth_demand prior) per (type, lane), seeded from the 2026-07-25 benchmark:
# cpu_coordination scales on the forkserver (proc), is GIL-flat on threads; io_llm is I/O-bound
# (cheap on threads, pointless elsewhere); gpu_kernel is cheap on the iGPU, dearer on the CPU.
_PRIORS = {
    "cpu_coordination": {"proc": (0.10, 0.6), "thread": (0.60, 0.6), "igpu": (0.50, 0.6)},
    "io_llm":           {"proc": (1.00, 0.1), "thread": (0.10, 0.1), "igpu": (1.00, 0.1)},
    "gpu_kernel":       {"proc": (0.30, 0.9), "thread": (0.30, 0.9), "igpu": (0.10, 0.9)},
}
_EMA = 0.15


class CostModel:
    """(task_type, lane) -> (expected_time_s, bandwidth_demand in [0,1]). Priors seed it; observe()
    refines the time by EMA and can flip the best lane when measurements contradict the type prior."""

    def __init__(self):
        self._t = {ty: {ln: _PRIORS[ty][ln][0] for ln in LANES} for ty in TYPES}
        self._bw = {ty: {ln: _PRIORS[ty][ln][1] for ln in LANES} for ty in TYPES}

    def cost(self, task_type: str, lane: str) -> tuple[float, float]:
        return self._t[task_type][lane], self._bw[task_type][lane]

    def observe(self, task_type: str, lane: str, time_s: float) -> None:
        cur = self._t[task_type][lane]
        self._t[task_type][lane] = (1 - _EMA) * cur + _EMA * float(time_s)

    def best_lane(self, task_type: str) -> str:
        return min(LANES, key=lambda ln: self._t[task_type][ln])
