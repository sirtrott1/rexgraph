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
_LAMBDA_PRI = 0.5   # how hard priority weights bias placement vs the primary makespan term

_ACTIVE_SHARES: dict = {}


def register_hive_share(name: str, share: float) -> None:
    """Register a hive as active with a relative resource share (used to split lane capacity when
    several hives run at once). Idempotent; a share <= 0 is treated as 1.0."""
    _ACTIVE_SHARES[name] = float(share) if share and share > 0 else 1.0


def unregister_hive_share(name: str) -> None:
    _ACTIVE_SHARES.pop(name, None)


def reset_shares() -> None:
    _ACTIVE_SHARES.clear()


def share_fraction(name: str) -> float:
    """This hive's fraction of the total active share (1.0 if it is the only active hive)."""
    total = sum(_ACTIVE_SHARES.values())
    if total <= 0 or name not in _ACTIVE_SHARES:
        return 1.0
    return _ACTIVE_SHARES[name] / total


def capacity(share_fraction: float = 1.0) -> dict:
    """Per-lane parallelism, optionally scaled by this hive's share of the machine. proc = physical
    cores (forkserver workers); thread = a comparable core-wide I/O pool; igpu = a small slot count
    (a bandwidth-bound single device). A share below 1.0 splits proc/thread down (never below 1)."""
    cores = os.cpu_count() or 8
    base_proc = float(max(1, cores // 2))
    base_thread = float(max(1, cores // 2))
    f = share_fraction if (share_fraction and 0 < share_fraction <= 1.0) else 1.0
    return {"proc": float(max(1, int(base_proc * f))),
            "thread": float(max(1, int(base_thread * f))),
            "igpu": 2.0}


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


def _priority_penalty(assignment: dict, units: list, cost: CostModel) -> float:
    """Sum over tasks of (weight - 1) * (time on assigned lane - time on the task type's best lane).
    Weight is centered on its own neutral value (1.0) so a wave with no weights contributes zero
    penalty regardless of placement (the objective reduces exactly to the unweighted wall-clock
    term). Above-neutral weight grows the penalty as the task is pushed off its best lane, so the
    actuator prefers to spill below-neutral (low-priority) work instead."""
    by_id = {u["id"]: u for u in units}
    pen = 0.0
    for tid, ln in assignment.items():
        u = by_id[tid]
        w = float(u.get("weight", 1.0)) - 1.0
        best = cost.best_lane(u["type"])
        pen += w * (cost.cost(u["type"], ln)[0] - cost.cost(u["type"], best)[0])
    return pen


def contention(assignment: dict, units: list, cost: CostModel, cap: dict | None = None) -> float:
    """Nonnegative contention of a placement. Wave WALL-CLOCK (max over lanes of load/parallelism)
    plus a small CPU<->iGPU bandwidth-war term plus a priority penalty that keeps high-weight tasks
    on their fast lane. `cap` overrides the per-lane capacity (e.g. a hive-share-scaled capacity)."""
    time, bw = _lane_groups(assignment, units, cost)
    base = _contention_from_sums(time, bw, cap or capacity())
    return base + _LAMBDA_PRI * _priority_penalty(assignment, units, cost)


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
def assign(units: list, cost: CostModel, cap: dict | None = None) -> dict:
    """Greedy marginal-contention placement with O(1) delta-scored moves. Same greedy/tie-break as a
    full recompute. Each unit may carry a `weight` (default 1.0, centered so the neutral value
    contributes no penalty); the priority penalty is separable per task, so a move's penalty delta
    is (weight - 1)*(time_new - time_cur)."""
    by_id = {u["id"]: u for u in units}
    cap = cap or capacity()
    a = {u["id"]: cost.best_lane(u["type"]) for u in units}
    time = {ln: 0.0 for ln in LANES}
    bw = {ln: 0.0 for ln in LANES}
    for tid, ln in a.items():
        t, b = cost.cost(by_id[tid]["type"], ln)
        time[ln] += t
        bw[ln] += b
    penalty = 0.0   # seed is best_lane for all, so the penalty starts at zero

    improved = True
    while improved:
        improved = False
        base = _contention_from_sums(time, bw, cap) + _LAMBDA_PRI * penalty
        best_move = None
        best_gain = 1e-12
        for u in units:
            tid = u["id"]
            cur = a[tid]
            w = float(u.get("weight", 1.0)) - 1.0
            tc, bc = cost.cost(u["type"], cur)
            for ln in LANES:
                if ln == cur:
                    continue
                tn, bn = cost.cost(u["type"], ln)
                time[cur] -= tc; bw[cur] -= bc; time[ln] += tn; bw[ln] += bn
                dpen = w * (tn - tc)
                cand = _contention_from_sums(time, bw, cap) + _LAMBDA_PRI * (penalty + dpen)
                gain = base - cand
                time[cur] += tc; bw[cur] += bc; time[ln] -= tn; bw[ln] -= bn
                if gain > best_gain:
                    best_gain = gain
                    best_move = (tid, ln, cur, tc, bc, tn, bn, dpen)
        if best_move is not None:
            tid, ln, cur, tc, bc, tn, bn, dpen = best_move
            a[tid] = ln
            time[cur] -= tc; bw[cur] -= bc; time[ln] += tn; bw[ln] += bn
            penalty += dpen
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


def _partition_spill(units: list, assignment: dict):
    """Split units by assigned lane, spilling any unpicklable proc fn to the thread lane (it then
    runs in-process, preserving side effects). Returns (by_id, proc_units, thread_units, eff_lane)
    where eff_lane[id] is the lane the fn will ACTUALLY run on."""
    by_id = {u["id"]: u for u in units}
    eff = dict(assignment)
    proc = []
    for u in [by_id[t] for t, l in assignment.items() if l == "proc"]:
        if _picklable(u["fn"]):
            proc.append(u)
        else:
            eff[u["id"]] = "thread"
    thread = [by_id[t] for t, l in eff.items() if l in ("thread", "igpu")]
    return by_id, proc, thread, eff


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
      against the lane the fn ACTUALLY ran on (post-spill), so the model never learns a wrong lane.

    See LanePools for the managed, warm-pool path (this function creates and tears down a fresh
    pool per wave, which is the right behavior for the standalone/test path)."""
    by_id, proc_units, thread_units, eff_lane = _partition_spill(units, assignment)
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


import threading


def _pin_worker():
    """Pool initializer: pin this worker to a single core to keep its L1-L3 cache hot. Best-effort;
    a no-op where affinity control is unavailable."""
    try:
        import os as _os
        pid = _os.getpid()
        ncores = _os.cpu_count() or 1
        if hasattr(_os, "sched_setaffinity"):
            _os.sched_setaffinity(0, {pid % ncores})
    except Exception:
        pass


class LanePools:
    """Managed execution lanes with an idle-aware lifecycle: lazy (no pool until a lane is used),
    warm (a created pool is reused across waves), and reaped when idle (a daemon reaper closes a
    lane idle past its TTL and self-exits once both lanes are cold, so nothing lingers at rest)."""

    def __init__(self, hive: str = "default", *, now=_time.monotonic,
                 idle_ttl_proc: float = 30.0, idle_ttl_thread: float = 120.0,
                 affinity: bool = False, cap: "dict|None" = None, reaper_tick: float = 1.0):
        self.hive = hive
        self._now = now
        self._ttl = {"proc": idle_ttl_proc, "thread": idle_ttl_thread}
        self._affinity = affinity
        self._cap = cap
        self._reaper_tick = reaper_tick
        self._pools = {"proc": None, "thread": None}
        self._last = {"proc": 0.0, "thread": 0.0}
        self._lock = threading.RLock()
        self._reaper = None
        self.reaper_alive = False

    # --- lane pool management ---
    def _make(self, lane: str):
        if lane == "thread":
            return ThreadPoolExecutor(max_workers=32)
        import multiprocessing as _mp
        ctx = _mp.get_context("forkserver")
        workers = int((self._cap or capacity())["proc"]) if self._cap else (os.cpu_count() or 8)
        init = _pin_worker if self._affinity else None
        return ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx, initializer=init)

    def _ensure(self, lane: str):
        with self._lock:
            if self._pools[lane] is None:
                self._pools[lane] = self._make(lane)
            self._last[lane] = self._now()
            self._start_reaper_locked()
            return self._pools[lane]

    def _start_reaper_locked(self):
        if self._reaper is None or not self._reaper.is_alive():
            self.reaper_alive = True
            self._reaper = threading.Thread(target=self._reaper_loop, daemon=True,
                                            name=f"lanepools-reaper-{self.hive}")
            self._reaper.start()

    def _reap_once(self) -> bool:
        """Close any lane idle past its TTL. Returns True while any pool remains open."""
        with self._lock:
            now = self._now()
            for lane in ("proc", "thread"):
                pool = self._pools[lane]
                if pool is not None and (now - self._last[lane]) >= self._ttl[lane]:
                    pool.shutdown(wait=False)
                    self._pools[lane] = None
            any_open = any(self._pools[l] is not None for l in ("proc", "thread"))
            if not any_open:
                self.reaper_alive = False
            return any_open

    def _reaper_loop(self):
        import time as _t
        while True:
            _t.sleep(self._reaper_tick)
            if not self._reap_once():
                return

    # --- execution ---
    def run(self, units: list, assignment: dict, cost: "CostModel|None" = None) -> dict:
        by_id, proc_units, thread_units, eff_lane = _partition_spill(units, assignment)
        results = {}
        timings = []

        def drain(pool_units, ex):
            for tid, res, dt in ex.map(_run_one, pool_units):
                results[tid] = res
                timings.append((by_id[tid]["type"], eff_lane[tid], dt))

        if thread_units:
            drain(thread_units, self._ensure("thread"))
        if proc_units:
            drain(proc_units, self._ensure("proc"))
        if cost is not None:
            for ty, ln, dt in timings:
                cost.observe(ty, ln, dt)
        return results

    def status(self) -> dict:
        with self._lock:
            now = self._now()
            out = {}
            for lane in ("proc", "thread"):
                pool = self._pools[lane]
                out[lane] = {"state": "warm" if pool is not None else "cold",
                             "idle_s": round(now - self._last[lane], 2) if pool is not None else None}
            return out

    def shutdown(self) -> None:
        with self._lock:
            for lane in ("proc", "thread"):
                if self._pools[lane] is not None:
                    self._pools[lane].shutdown(wait=False)
                    self._pools[lane] = None
            self.reaper_alive = False


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
