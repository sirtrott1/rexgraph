"""Hive Coordinator (v1): place tasks onto compute lanes by minimizing a relational-complex
contention objective. See rexgraph-internal spec 2026-07-25-hive-coordinator-design.md."""
from __future__ import annotations

LANES = ("proc", "thread", "igpu")
TYPES = ("cpu_coordination", "io_llm", "gpu_kernel")

# (time_s prior, bandwidth_demand prior) per (type, lane), seeded from the 2026-07-25 benchmark:
# cpu_coordination scales on the forkserver (proc), is GIL-flat on threads; io_llm is I/O-bound
# (cheap on threads, pointless elsewhere); gpu_kernel is cheap on the iGPU, dearer on the CPU.
_PRIORS = {
    "cpu_coordination": {"proc": (0.10, 0.6), "thread": (0.80, 0.6), "igpu": (1.00, 0.6)},
    "io_llm":           {"proc": (1.00, 0.1), "thread": (0.10, 0.1), "igpu": (1.00, 0.1)},
    "gpu_kernel":       {"proc": (1.00, 0.9), "thread": (1.00, 0.9), "igpu": (0.10, 0.9)},
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


# --- Resource complex + contention sensor (edge-centric delegation complex) ---
import os
import numpy as np

BRAIN = 0
_LANE_V = {"proc": 1, "thread": 2, "igpu": 3}
_HUB = 4
_BW_LAMBDA = 0.02   # gentle weight on the CPU<->iGPU bandwidth-war term vs the primary wall-clock


def capacity() -> dict:
    """Per-lane parallelism (how many task-edges the operator runs at once). proc = physical cores
    (forkserver workers); thread = a comparable core-wide I/O pool; igpu = a small slot count
    (a bandwidth-bound single device)."""
    cores = os.cpu_count() or 8
    return {"proc": float(max(1, cores // 2)), "thread": float(max(1, cores // 2)), "igpu": 2.0}


def _lane_groups(assignment, units, cost):
    by_id = {u["id"]: u for u in units}
    time = {ln: 0.0 for ln in LANES}
    bw = {ln: 0.0 for ln in LANES}
    for tid, ln in assignment.items():
        t, b = cost.cost(by_id[tid]["type"], ln)
        time[ln] += t
        bw[ln] += b
    return time, bw


def contention(assignment: dict, units: list, cost: CostModel) -> float:
    """Nonnegative contention of a placement - the objective the actuator minimizes. Task execution
    is an EDGE from an operator (lane) to the task (see delegation_complex). Contention is the wave
    WALL-CLOCK (max over lanes of load/parallelism, since lanes run concurrently) plus a small
    CPU<->iGPU bandwidth-war term (the two drawing the shared unified bandwidth at once)."""
    time, bw = _lane_groups(assignment, units, cost)
    cap = capacity()
    wall = max((time[ln] / cap[ln] for ln in LANES), default=0.0)
    bw_war = min(bw["proc"], bw["igpu"])   # co-drawn shared bandwidth = the circulating (curl) part
    return wall + _BW_LAMBDA * bw_war


def delegation_complex(assignment: dict, units: list):
    """The edge-centric delegation complex (owner's model: an operator running a task IS an EDGE
    from the brain, via the operator lane, to the task - not a vertex label). Vertices: brain, proc,
    thread, igpu, hub, tasks. Edges: brain->lane (operator channels), lane->task (each execution,
    the datum), and proc->hub / igpu->hub (shared bandwidth). Returned as a RexGraph for monitoring /
    character analysis (bottleneck lanes = centrality, delegation deadlocks = cycles). Built on
    demand, never in the actuator hot loop."""
    from rexgraph.graph import RexGraph
    src, tgt = [], []
    lanes_used = sorted({ln for ln in assignment.values()})
    for ln in lanes_used:
        src.append(BRAIN); tgt.append(_LANE_V[ln])              # brain -> operator lane
    tid_v = {}
    nxt = _HUB + 1
    for u in units:
        tid_v[u["id"]] = nxt; nxt += 1
    for tid, ln in assignment.items():
        src.append(_LANE_V[ln]); tgt.append(tid_v[tid])          # operator -> task (the execution EDGE)
    for ln in ("proc", "igpu"):
        if ln in lanes_used:
            src.append(_LANE_V[ln]); tgt.append(_HUB)            # shared bandwidth coupling
    return RexGraph(sources=np.array(src, dtype=np.int32), targets=np.array(tgt, dtype=np.int32))


# --- Flow actuator (marginal-contention greedy) ---
def assign(units: list, cost: CostModel) -> dict:
    """Greedy marginal-contention placement: seed each task on its best lane, then repeatedly move
    the task whose relocation most reduces total contention, until no move helps. Deterministic
    (stable order), so results never depend on scheduling."""
    a = {u["id"]: cost.best_lane(u["type"]) for u in units}
    improved = True
    while improved:
        improved = False
        base = contention(a, units, cost)
        best_move = None
        best_gain = 1e-12
        for u in units:
            tid = u["id"]
            cur = a[tid]
            for ln in LANES:
                if ln == cur:
                    continue
                a[tid] = ln
                gain = base - contention(a, units, cost)
                if gain > best_gain:
                    best_gain = gain
                    best_move = (tid, ln)
                a[tid] = cur
        if best_move is not None:
            a[best_move[0]] = best_move[1]
            improved = True
    return a
