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


import logging as _logging
_log = _logging.getLogger(__name__)


def _run_one(u):
    """Run one unit's fn, catching its exception so a single bad task cannot abort the whole wave's
    map (which would otherwise re-run every already-completed task in the serial fallback). Returns
    (id, result, seconds, error_repr) with error_repr None on success."""
    t0 = _time.perf_counter()
    try:
        res = u["fn"]()
        return u["id"], res, _time.perf_counter() - t0, None
    except Exception as ex:
        return u["id"], None, _time.perf_counter() - t0, repr(ex)


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
        # Per-task isolation: a failed task is logged and OMITTED from results (its id simply does
        # not appear), so one bad fn never aborts the wave or forces a full re-run of its peers.
        for tid, res, dt, err in ex.map(_run_one, pool_units):
            if err is not None:
                _log.warning("coordinator task '%s' failed on lane %s: %s", tid, eff_lane[tid], err)
                continue
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


def _inner_threads(workers: int, cores_budget: "int|None" = None) -> int:
    """Inner native (BLAS/OpenMP) thread budget per proc worker: budget // workers, so
    workers * inner tracks the CORE BUDGET this pool is entitled to (the same arithmetic parallel_map
    uses). `cores_budget` is the machine cores for a single coordinator, but the hive's SHARE of the
    cores when several coordinators run at once - otherwise N concurrent pools each assume all cores
    and oversubscribe (N * workers * inner threads). Never below 1."""
    budget = cores_budget if cores_budget else (os.cpu_count() or 1)
    return max(1, int(budget) // max(1, int(workers)))


_WORKER_TL = None   # holds the per-worker threadpool limiter for the worker's whole lifetime


def _proc_worker_init(inner: int, affinity: bool):
    """forkserver proc-lane worker setup. CAPS this worker's inner native (BLAS / OpenMP) thread
    pools to `inner`, so N workers each running threaded BLAS do not oversubscribe the machine
    (workers * inner tracks the core budget, the same arithmetic parallel_map uses). Without this,
    a BLAS-heavy batch runs about 10x SLOWER than serial (32 workers each spawning 32 BLAS threads).
    Optionally pins the worker to a core to keep its cache hot."""
    global _WORKER_TL
    import os as _os
    inner = max(1, int(inner))
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _os.environ[_v] = str(inner)           # for any BLAS pool not yet initialized
    try:
        import threadpoolctl
        _WORKER_TL = threadpoolctl.threadpool_limits(inner)   # runtime cap for already-loaded pools
    except Exception:
        pass
    if affinity:
        try:
            if hasattr(_os, "sched_setaffinity"):
                _os.sched_setaffinity(0, {_os.getpid() % (_os.cpu_count() or 1)})
        except Exception:
            pass


class LanePools:
    """Managed execution lanes with an idle-aware lifecycle: lazy (no pool until a lane is used),
    warm (a created pool is reused across waves), and reaped when idle (a daemon reaper closes a
    lane idle past its TTL and self-exits once both lanes are cold, so nothing lingers at rest)."""

    def __init__(self, hive: str = "default", *, now=_time.monotonic,
                 idle_ttl_proc: float = 30.0, idle_ttl_thread: float = 120.0,
                 affinity: bool = False, cap: "dict|None" = None, reaper_tick: float = 1.0,
                 cores_budget: "int|None" = None):
        self.hive = hive
        self._now = now
        self._ttl = {"proc": idle_ttl_proc, "thread": idle_ttl_thread}
        self._affinity = affinity
        self._cap = cap
        # This pool's share of machine cores (for the inner-thread budget). Defaults to all cores
        # (single coordinator); pass the hive's share when several coordinators run concurrently so
        # they do not collectively oversubscribe.
        self._cores_budget = int(cores_budget) if cores_budget else (os.cpu_count() or 8)
        self._reaper_tick = reaper_tick
        self._pools = {"proc": None, "thread": None}
        self._last = {"proc": 0.0, "thread": 0.0}
        self._active = {"proc": 0, "thread": 0}   # in-flight waves per lane (never reap a busy lane)
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
        workers = max(1, workers)
        # Cap each worker's inner BLAS/OpenMP threads so workers * inner tracks THIS pool's core
        # budget (its share of the machine when several coordinators run), never oversubscribing.
        inner = _inner_threads(workers, self._cores_budget)
        return ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                   initializer=_proc_worker_init, initargs=(inner, self._affinity))

    def _ensure(self, lane: str):
        with self._lock:
            if self._pools[lane] is None:
                self._pools[lane] = self._make(lane)
            self._last[lane] = self._now()
            self._start_reaper_locked()
            return self._pools[lane]

    def _start_reaper_locked(self):
        # Key off the lock-protected reaper_alive flag, NOT thread.is_alive(). _reap_once clears the
        # flag under this same lock at the instant it decides to exit, so a wave arriving while the
        # old thread is still winding down sees the flag False and starts a fresh reaper. That closes
        # the window where a newly created pool could be left with no reaper watching it. A brief
        # overlap of two daemon reapers is harmless: both reap idempotently under the lock and exit.
        if not self.reaper_alive:
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
                idle = pool is not None and (now - self._last[lane]) >= self._ttl[lane]
                if idle and self._active[lane] == 0:      # never reap a lane running a wave
                    pool.shutdown(wait=False)
                    self._pools[lane] = None
            any_open = any(self._pools[l] is not None for l in ("proc", "thread"))
            if not any_open:
                self.reaper_alive = False
            return any_open

    def _reaper_loop(self):
        import time as _t
        try:
            while True:
                _t.sleep(self._reaper_tick)
                if not self._reap_once():
                    return
        except Exception:
            # never strand the flag: a crashed reaper must let the next wave start a fresh one
            with self._lock:
                self.reaper_alive = False

    # --- execution ---
    def run(self, units: list, assignment: dict, cost: "CostModel|None" = None) -> dict:
        by_id, proc_units, thread_units, eff_lane = _partition_spill(units, assignment)
        results = {}
        timings = []

        def drain(pool_units, ex):
            # Per-task isolation (see execute): a failed task is logged and omitted, never aborting
            # the wave or re-running its peers.
            for tid, res, dt, err in ex.map(_run_one, pool_units):
                if err is not None:
                    _log.warning("coordinator task '%s' failed on lane %s: %s",
                                 tid, eff_lane[tid], err)
                    continue
                results[tid] = res
                timings.append((by_id[tid]["type"], eff_lane[tid], dt))

        def run_lane(lane, lane_units):
            ex = self._ensure(lane)             # create/warm the pool, stamp last, start reaper
            with self._lock:
                self._active[lane] += 1         # mark busy so the reaper cannot close it mid-wave
            try:
                drain(lane_units, ex)
            finally:
                with self._lock:
                    self._active[lane] -= 1
                    if self._pools[lane] is not None:
                        self._last[lane] = self._now()   # refresh idle clock at COMPLETION too

        if thread_units:
            run_lane("thread", thread_units)
        if proc_units:
            run_lane("proc", proc_units)
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
    """Per-wave plan -> execute -> learn loop. With a `pools` (LanePools) it dispatches through the
    managed warm lanes; without, it uses per-wave `execute`. `cap` is an optional hive-share-scaled
    capacity used by the actuator."""

    def __init__(self, cost: "CostModel|None" = None, pools: "LanePools|None" = None,
                 cap: "dict|None" = None):
        self.cost = cost or CostModel()
        self.pools = pools
        self.cap = cap

    def plan(self, units: list) -> dict:
        return assign(units, self.cost, cap=self.cap)

    def run_wave(self, units: list) -> dict:
        a = self.plan(units)
        if self.pools is not None:
            return self.pools.run(units, a, cost=self.cost)
        return execute(units, a, cost=self.cost)
