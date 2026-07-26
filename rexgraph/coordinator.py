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


def _contention_from_sums(time: dict, bw: dict, cap: dict) -> float:
    """Contention from precomputed per-lane time/bw sums (the actuator hot path)."""
    wall = max((time[ln] / cap[ln] for ln in LANES), default=0.0)
    bw_war = min(bw["proc"], bw["igpu"])   # co-drawn shared bandwidth = the circulating (curl) part
    return wall + _BW_LAMBDA * bw_war


def contention(assignment: dict, units: list, cost: CostModel) -> float:
    """Nonnegative contention of a placement - the objective the actuator minimizes. Task execution
    is an EDGE from an operator (lane) to the task (see delegation_complex). Contention is the wave
    WALL-CLOCK (max over lanes of load/parallelism, since lanes run concurrently) plus a small
    CPU<->iGPU bandwidth-war term (the two drawing the shared unified bandwidth at once)."""
    time, bw = _lane_groups(assignment, units, cost)
    return _contention_from_sums(time, bw, capacity())


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
    (stable order), so results never depend on scheduling.

    Same greedy/tie-break semantics as a full-recompute search, but each candidate move is scored
    as an O(1) DELTA against cached per-lane time/bw sums instead of re-summing every task. That
    drops the actuator from ~O(n^3) to ~O(n^2) so a large per-wave placement stays cheap."""
    by_id = {u["id"]: u for u in units}
    cap = capacity()
    a = {u["id"]: cost.best_lane(u["type"]) for u in units}
    time = {ln: 0.0 for ln in LANES}
    bw = {ln: 0.0 for ln in LANES}
    for tid, ln in a.items():
        t, b = cost.cost(by_id[tid]["type"], ln)
        time[ln] += t
        bw[ln] += b

    improved = True
    while improved:
        improved = False
        base = _contention_from_sums(time, bw, cap)
        best_move = None
        best_gain = 1e-12
        for u in units:
            tid = u["id"]
            cur = a[tid]
            tc, bc = cost.cost(u["type"], cur)
            for ln in LANES:
                if ln == cur:
                    continue
                tn, bn = cost.cost(u["type"], ln)
                time[cur] -= tc; bw[cur] -= bc; time[ln] += tn; bw[ln] += bn
                gain = base - _contention_from_sums(time, bw, cap)
                time[cur] += tc; bw[cur] += bc; time[ln] -= tn; bw[ln] -= bn
                if gain > best_gain:
                    best_gain = gain
                    best_move = (tid, ln, cur, tc, bc, tn, bn)
        if best_move is not None:
            tid, ln, cur, tc, bc, tn, bn = best_move
            a[tid] = ln
            time[cur] -= tc; bw[cur] -= bc; time[ln] += tn; bw[ln] += bn
            improved = True
    return a


# --- Dispatch seam: execute an assignment across the compute lanes ---
import time as _time
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def _run_one(u):
    t0 = _time.perf_counter()
    res = u["fn"]()
    return u["id"], res, _time.perf_counter() - t0


def _picklable(fn) -> bool:
    """Can this fn cross the forkserver boundary? A closure/lambda/bound-method over hive state
    (the natural form of a real hive task) cannot, and would otherwise crash the whole wave."""
    try:
        pickle.dumps(fn)
        return True
    except Exception:
        return False


def execute(units: list, assignment: dict, cost: "CostModel|None" = None) -> dict:
    """Execute each unit's `fn` on its assigned lane: proc -> process pool (true multicore for CPU-
    bound work), thread/igpu -> thread pool (I/O and GPU-launch are GIL-light). Results are keyed by
    id and are INDEPENDENT of lane and order. Folds per-task timing into cost.

    Two guards make this safe for real (not just test) hive tasks:
    - PICKLABILITY: a proc-lane fn that cannot be pickled (a closure/lambda/bound-method over hive
      state) is transparently spilled to the thread lane instead of crashing the forkserver pool.
    - SIDE EFFECTS: the proc lane runs the fn in a child process, so in-process mutations do NOT
      propagate back - only the return value does. The picklability spill covers the common stateful
      case (closures run in-process on the thread lane, mutations preserved); a picklable fn that
      relies on mutating shared parent state must not be routed to proc. Cost timing is recorded
      against the lane the fn ACTUALLY ran on (post-spill), so the model never learns a wrong lane."""
    by_id = {u["id"]: u for u in units}
    eff_lane = dict(assignment)  # lane each fn actually runs on (after any spill)

    proc_units = []
    for u in [by_id[t] for t, l in assignment.items() if l == "proc"]:
        if _picklable(u["fn"]):
            proc_units.append(u)
        else:
            eff_lane[u["id"]] = "thread"  # spill: run in-process, preserve side effects

    thread_units = [by_id[t] for t, l in eff_lane.items() if l in ("thread", "igpu")]
    results = {}
    timings = []

    def drain(pool_units, ex):
        for tid, res, dt in ex.map(_run_one, pool_units):
            results[tid] = res
            timings.append((by_id[tid]["type"], eff_lane[tid], dt))

    if thread_units:
        with ThreadPoolExecutor(max_workers=min(32, len(thread_units))) as ex:
            drain(thread_units, ex)
    if proc_units:
        # forkserver, not raw fork: this process has imported torch/numpy (many threads), and
        # os.fork() in a multi-threaded process warns and can deadlock. forkserver is also the
        # mechanism that gave CPU-bound coordination its 5.8x multicore scaling in benchmarks.
        import multiprocessing as _mp
        ctx = _mp.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 8, len(proc_units)),
                                 mp_context=ctx) as ex:
            drain(proc_units, ex)
    if cost is not None:
        for ty, ln, dt in timings:
            cost.observe(ty, ln, dt)
    return results


# --- Coordinator: the per-wave plan -> execute -> learn loop (cadence = per-wave in v1) ---
class Coordinator:
    """For each wave of tasks: solve the placement (assign) that minimizes contention, execute it
    across the compute lanes, and fold measured timings back into the cost model so the next wave is
    smarter. One solver invoked per wave; the static and continuous cadences reuse the same solve."""

    def __init__(self, cost: "CostModel|None" = None):
        self.cost = cost or CostModel()

    def plan(self, units: list) -> dict:
        return assign(units, self.cost)

    def run_wave(self, units: list) -> dict:
        a = self.plan(units)
        return execute(units, a, cost=self.cost)
