"""Hive Coordinator (v1): place tasks onto compute lanes by minimizing a relational-complex
contention objective."""
from __future__ import annotations

import math as _math

LANES = ("proc", "thread", "igpu")
TYPES = ("cpu_coordination", "io_llm", "local_llm", "gpu_kernel")

# (time_s prior, bandwidth_demand prior) per (type, lane), seeded from the 2026-07-25 benchmark:
# cpu_coordination scales on the forkserver (proc), is GIL-flat on threads; io_llm is I/O-bound
# (cheap on threads, pointless elsewhere); gpu_kernel is cheap on the iGPU, dearer on the CPU.
#
# local_llm is NOT io_llm. io_llm is a REMOTE call: the caller blocks on a socket, so it is cheap
# on a thread and draws no local bandwidth, which is what the 0.1 prior says. A model running on
# THIS box is the opposite and is the heaviest bandwidth draw in the system: measured 2026-08-23 on
# the 8060S, a 27B Q4 decodes at 12.5 tok/s x 16.5 GB = 224 GB/s against a ~256 GB/s bus, so ~87%
# of it, and attention is only ~6% of that. Calling it 0.1 tells the actuator it can run a wave of
# gpu_kernel work alongside generation for free, when in fact the two are competing for one bus.
#
# The bandwidth is drawn by the SERVER, not by the caller's lane, so it is high on all three: the
# caller's thread blocking does not make the bus quieter. Time is cheap on a thread for the same
# reason it is for io_llm: the caller is waiting, not working.
_PRIORS = {
    "cpu_coordination": {"proc": (0.10, 0.6), "thread": (0.80, 0.6), "igpu": (1.00, 0.6)},
    "io_llm":           {"proc": (1.00, 0.1), "thread": (0.10, 0.1), "igpu": (1.00, 0.1)},
    # TIME mirrors io_llm exactly and only BANDWIDTH differs. That is deliberate: the
    # agent adapter routes LLM work here partly for a CORRECTNESS reason. A spawn or a
    # live-server attach mutates hive state in place and must not run in a forkserver
    # child, where the mutation would be lost, so it relies on the thread lane being
    # strictly cheapest. Giving local_llm a cheaper proc or igpu time would silently move
    # that work off-thread and lose the mutation. The fix here is the bus accounting, not
    # the placement, so the placement is left identical by construction.
    "local_llm":        {"proc": (1.00, 0.9), "thread": (0.10, 0.9), "igpu": (1.00, 0.9)},
    "gpu_kernel":       {"proc": (1.00, 0.9), "thread": (1.00, 0.9), "igpu": (0.10, 0.9)},
}

# Concurrent requests to ONE local server SHARE their weight reads, so N generations cost far less
# than N times one. Measured on the 35B-A3B MoE, aggregate decode: 1 stream 53.4 tok/s, 2 concurrent
# 74.1, 4 concurrent 81.2, 8 concurrent 120.8.
LOCAL_LLM_BATCH_GAIN = {1: 1.00, 2: 1.39, 4: 1.52, 8: 2.26}

# Types whose cost is SHARED rather than additive: the bus draw belongs to one server serving all of
# them, and their wall-clock overlaps. Everything else stays strictly additive as before.
_SHARED_TYPES = frozenset({"local_llm"})


#### which lanes share a bus is HARDWARE, so it is declared and not hardcoded
#
# `min(proc, igpu)` baked in unified memory. On the 8060S that is right, since the iGPU and
# the CPU are on one physical bus, so a draw on either is a draw on the same resource --
# and on a discrete-GPU box it is wrong, because VRAM is its own pool and an igpu draw
# costs a CPU draw nothing.
#
# So the topology is a COMPLEX, edge-primary the way everything else here is: a bus is a
# vertex, a LANE is the relation over the buses it draws on, and two lanes meet exactly
# when they share one. A lane on a single bus is a 1-ary witness relation, and two
# witnesses on one vertex meet there, which is what makes the unified case fall out
# rather than being special-cased.
UNIFIED_MEMORY = {"mem": ("proc", "thread", "igpu")}
SPLIT_MEMORY = {"sys_mem": ("proc", "thread"), "vram": ("igpu",)}
_BUSES = UNIFIED_MEMORY


def bus_topology() -> dict:
    """The active bus -> lanes map. Unified memory by default, which is what the machine
    this was measured on has and what the previous hardcoded form assumed."""
    return dict(_BUSES)


def _validate_buses(buses: dict) -> None:
    for name, lanes in buses.items():
        bad = [ln for ln in lanes if ln not in LANES]
        if bad:
            raise ValueError(f"bus {name!r} names unknown lanes {bad}; lanes are {LANES}")


def set_bus_topology(buses: dict | None) -> None:
    """Declare the PROCESS-default topology. None restores unified memory.

    Per-machine topology belongs on a CostModel (`set_buses`); this is the convenience
    for the common case of one process describing one machine.
    """
    global _BUSES
    if buses is None:
        _BUSES = UNIFIED_MEMORY
        return
    _validate_buses(buses)
    _BUSES = {k: tuple(v) for k, v in buses.items()}


def _buses_of(cost) -> dict:
    """The topology the given cost model describes, or the process default."""
    fn = getattr(cost, "buses", None)
    return fn() if callable(fn) else bus_topology()


def bus_topology_for(unified) -> dict | None:
    """The topology implied by whether the compute GPU's memory is unified.

    True -> UNIFIED_MEMORY, False -> SPLIT_MEMORY, and None stays None: an undetermined
    answer must not become a guess, because guessing wrong silently mis-prices every
    bandwidth decision while guessing nothing only asks the caller to declare.

    Pure on purpose (it takes the answer rather than probing), so rexgraph does not
    reach into the agent layer for hardware. `agent.local_runtime.detect_gpus` is what
    produces the input.
    """
    if unified is True:
        return dict(UNIFIED_MEMORY)
    if unified is False:
        return dict(SPLIT_MEMORY)
    return None


def bus_complex(buses: dict | None = None):
    """The topology as a relational complex: buses are vertices, lanes are the relations
    over the buses they draw on.

    Returned so the structure can be READ with the rest of the machinery: `Sheaf` over
    it recovers which lanes meet, through `mediators`, from the incidence alone, rather
    than the sharing being a fact buried in an expression. The hot path below uses the
    derived lane sets, not this object: the complex is the declaration, not the inner loop.
    """
    from rexgraph.graph import RexGraph

    b = buses or bus_topology()
    bus_ids = {name: i for i, name in enumerate(sorted(b))}
    ptr, idx = [0], []
    lanes = []
    for ln in LANES:
        on = sorted(bus_ids[name] for name, ls in b.items() if ln in ls)
        if not on:
            continue
        idx.extend(on)
        ptr.append(len(idx))
        lanes.append(ln)
    if len(ptr) == 1:
        return None, []
    rex = RexGraph.from_hypergraph(np.array(ptr, np.int64), np.array(idx, np.int64))
    rex._ensure_clean()
    return rex, lanes


def _bus_draws(bw: dict, buses: dict | None = None) -> list[list[float]]:
    """Per bus, the draws of the lanes on it."""
    return [[bw[ln] for ln in lanes if ln in bw]
            for lanes in (buses or bus_topology()).values()]


def _interp_gain(table: dict, n: int) -> float:
    """Linear between measured points, FLAT above the largest: past what was measured the
    honest answer is the last thing seen, not a projection."""
    if n <= 1:
        return 1.0
    pts = sorted(table)
    if not pts:
        return 1.0
    if n >= pts[-1]:
        return table[pts[-1]]
    lo = max(k for k in pts if k <= n)
    hi = min(k for k in pts if k >= n)
    if lo == hi:
        return table[lo]
    f = (n - lo) / (hi - lo)
    return table[lo] + f * (table[hi] - table[lo])


def _enforce_gain_monotone(table: dict) -> None:
    """Clamp the curve so n/gain(n) never decreases, in place.

    The actuator's greedy requires that absorbing one more unit into a batch cannot make
    the wave cheaper; if it could, the search would pile every unit onto one lane. gain(n)
    may rise with n, but no faster than n itself, so the admissible ceiling at n is
    gain(prev) * n / prev. A learned value above that is clamped down to it.
    """
    pts = sorted(table)
    for prev, cur in zip(pts, pts[1:], strict=False):
        ceiling = table[prev] * (cur / prev)
        if table[cur] > ceiling:
            table[cur] = ceiling
        if table[cur] < table[prev]:          # and gain itself never falls
            table[cur] = table[prev]


def _batch_gain(n: int) -> float:
    """The SEEDED curve, for callers without a CostModel. A model that has been fed
    measurements should be asked instead: see CostModel.batch_gain."""
    return _interp_gain(LOCAL_LLM_BATCH_GAIN, n)


def share_key(unit: dict) -> str:
    """Which shared resource a unit draws on. Units sharing a key share weight reads.

    An explicit `share_group` is the honest answer when the caller knows it: a hive with
    three bees has three servers, and requests to different bees share nothing. Without
    one, units of a type are assumed to hit the same server, which is the single-bee case
    and the common one.
    """
    return str(unit.get("share_group") or unit.get("type"))

# And it does not stack with speculation: the same measurement gives 8 concurrent + MTP at 93.9
# tok/s against 120.8 without it, because MTP fills idle compute with drafting and batching fills
# it with other sequences. Single stream is the other way round: MTP 89.4 against 60.3. So the
# choice is a function of how many generations are in flight, not a setting to fix once.
_EMA = 0.15


class CostModel:
    """(task_type, lane) -> (expected_time_s, bandwidth_demand in [0,1]). Priors seed it; observe()
    refines the time by EMA and can flip the best lane when measurements contradict the type prior."""

    def __init__(self):
        self._t = {ty: {ln: _PRIORS[ty][ln][0] for ln in LANES} for ty in TYPES}
        self._bw = {ty: {ln: _PRIORS[ty][ln][1] for ln in LANES} for ty in TYPES}
        self._gain = dict(LOCAL_LLM_BATCH_GAIN)   # per-model, so two boxes do not collide
        self._base = None                         # measured n = 1 aggregate tok/s
        self._pending = []                        # n > 1 samples seen before any baseline
        # The bus topology is HARDWARE, so it belongs to the model that describes a
        # machine and not to the process: a hive spanning a unified-memory laptop and a
        # discrete-GPU desktop has to hold both at once. Seeded from the process default
        # so a single-machine caller never has to say anything.
        self._buses = None

    def cost(self, task_type: str, lane: str) -> tuple[float, float]:
        return self._t[task_type][lane], self._bw[task_type][lane]

    def observe(self, task_type: str, lane: str, time_s: float) -> None:
        cur = self._t[task_type][lane]
        self._t[task_type][lane] = (1 - _EMA) * cur + _EMA * float(time_s)

    def best_lane(self, task_type: str) -> str:
        return min(LANES, key=lambda ln: self._t[task_type][ln])

    #### the batch gain, learned the same way the times are
    def batch_gain(self, n: int) -> float:
        """This model's aggregate throughput multiplier for `n` concurrent units.

        Starts at the seeded table and moves as `observe_throughput` is fed. Kept on the
        model rather than in the module constant so two coordinators on different hardware
        do not overwrite each other's curve: the seed is the 35B-A3B MoE on an 8060S and
        a dense model or another box has a different one."""
        return _interp_gain(self._gain, n)

    def observe_throughput(self, n: int, tok_per_s: float) -> None:
        """Record that `n` concurrent units achieved `tok_per_s` AGGREGATE.

        n = 1 refines the baseline; n > 1 refines the gain at that point as the ratio to
        it. A gain is meaningless before a baseline exists, so early n > 1 samples are
        held and replayed once one does. Dropping them would quietly discard the first
        wave of a fresh process, which is exactly when the seed is least trustworthy.

        Every update is followed by _enforce_gain_monotone, because the actuator needs
        n/gain(n) to keep rising. One unlucky sample could otherwise make a bigger batch
        score CHEAPER than a smaller one, and the greedy would hoard units onto one lane
        forever. A learned curve must not be able to break the search.
        """
        n = int(n)
        r = float(tok_per_s)
        if n < 1 or not (r > 0.0) or not _math.isfinite(r):
            return
        if n == 1:
            self._base = r if self._base is None else (1 - _EMA) * self._base + _EMA * r
            for pend_n, pend_r in self._pending:
                self._fold_gain(pend_n, pend_r / self._base)
            self._pending.clear()
            return
        if self._base is None:
            self._pending.append((n, r))
            if len(self._pending) > 32:
                self._pending.pop(0)
            return
        self._fold_gain(n, r / self._base)

    def _fold_gain(self, n: int, gain: float) -> None:
        if not (gain > 0.0) or not _math.isfinite(gain):
            return
        cur = self._gain.get(n, _interp_gain(self._gain, n))
        self._gain[n] = (1 - _EMA) * cur + _EMA * gain
        _enforce_gain_monotone(self._gain)

    #### the machine this model describes
    def buses(self) -> dict:
        """This model's bus -> lanes map, falling back to the process default."""
        return dict(self._buses) if self._buses is not None else bus_topology()

    def set_buses(self, buses: dict | None) -> None:
        """Declare the hardware THIS model describes. None defers to the process default.

        A 7900 XTX or a 3070 has its own VRAM, so `SPLIT_MEMORY`; a Strix Halo laptop is
        unified, so `UNIFIED_MEMORY`. Getting it wrong does not crash, it silently charges
        a bandwidth war between pools that do not touch, or fails to charge one that does.
        """
        if buses is None:
            self._buses = None
            return
        _validate_buses(buses)
        self._buses = {k: tuple(v) for k, v in buses.items()}

    def gain_table(self) -> dict:
        """A copy of the learned curve, for reporting. The baseline is separate because a
        gain is a ratio and the caller usually wants to see what it is a ratio TO."""
        return {"baseline_tok_s": self._base, "gain": dict(self._gain)}


#### Resource complex + contention sensor (edge-centric delegation complex)
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


def _new_state(gain=None):
    """Lane accumulators. `solo_t`/`solo_bw` are the additive part; `grp[lane][key]` is
    `[count, time_sum, bw_once]` for a shared resource, whose bus draw is counted ONCE
    however many units draw on it and whose wall-clock is divided by the batch gain."""
    return {"solo_t": {ln: 0.0 for ln in LANES}, "solo_bw": {ln: 0.0 for ln in LANES},
            "grp": {ln: {} for ln in LANES},
            # the LEARNED curve when a model supplies one, so a coordinator that has been
            # fed measurements prices batches by what it saw and not by the seed
            "gain": gain or _batch_gain}


def _state_apply(st, unit, ty, lane, t, b, sign=1):
    if ty in _SHARED_TYPES:
        g = st["grp"][lane]
        key = share_key(unit)
        e = g.get(key)
        if e is None:
            e = g[key] = [0, 0.0, b]
        e[0] += sign
        e[1] += sign * t
        if e[0] <= 0:
            del g[key]
    else:
        st["solo_t"][lane] += sign * t
        st["solo_bw"][lane] += sign * b


def _state_sums(st):
    time = dict(st["solo_t"])
    bw = dict(st["solo_bw"])
    for ln in LANES:
        for _key, (n, tsum, bonce) in st["grp"][ln].items():
            time[ln] += tsum / st["gain"](n)
            bw[ln] += bonce                      # one server, one weight stream
    return time, bw


def _lane_groups(assignment, units, cost):
    by_id = {u["id"]: u for u in units}
    st = _new_state(getattr(cost, "batch_gain", None))
    for tid, ln in assignment.items():
        u = by_id[tid]
        t, b = cost.cost(u["type"], ln)
        _state_apply(st, u, u["type"], ln, t, b, +1)
    return _state_sums(st)


def _contention_from_sums(time: dict, bw: dict, cap: dict, buses: dict | None = None) -> float:
    """Contention from precomputed per-lane time/bw sums (the actuator hot path).

    The bandwidth term counts what is CO-DRAWN, which is the circulating part: a lane
    drawing while the others are idle has the bus to itself and is not at war with
    anyone. That was written as `min(proc, igpu)`, which says it for two lanes but
    silently excluded the third, so anything on the thread lane drew for free, and
    since free is cheap, the actuator PREFERRED that lane, which is where a blocking
    call into a local model lands. One local_llm plus one gpu_kernel scored 0.0680 on
    proc, 0.1000 on igpu and 0.0500 on thread, and io_llm at 0.1 bandwidth tied
    local_llm at 0.9 exactly, both at 0.1180.

    The generalisation is `total - max`, because for two terms that IS the minimum:
    min(a, b) = (a + b) - max(a, b). So every lane's draw is counted, the largest one
    is credited with owning the bus, and the rest are the co-drawn mass contending with
    it. Whenever the thread lane draws nothing this returns the OLD value exactly, so
    the correction is confined to the case that was wrong.

    Measured, 8060S, 2026-08-23: a 27B Q4 decodes at 12.5 tok/s x 16.5 GB = 224 GB/s
    against a ~256 GB/s bus, so a local model is ~87% of it and is the heaviest draw in
    the system. Scoring a wave that mixes generation with gpu_kernel work as free was
    not a small error.
    """
    wall = max((time[ln] / cap[ln] for ln in LANES), default=0.0)
    bw_war = 0.0
    for draws in _bus_draws(bw, buses):
        if draws:
            bw_war += sum(draws) - max(draws)     # co-drawn mass on THAT bus
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
    base = _contention_from_sums(time, bw, cap or capacity(), _buses_of(cost))
    return base + _LAMBDA_PRI * _priority_penalty(assignment, units, cost)


def delegation_complex(assignment: dict, units: list):
    """The edge-centric delegation complex (owner's model: an operator running a task IS an EDGE
    from the brain, via the operator lane, to the task, not a vertex label). Vertices: brain, proc,
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


#### Flow actuator (marginal-contention greedy)
def assign(units: list, cost: CostModel, cap: dict | None = None) -> dict:
    """Greedy marginal-contention placement with O(1) delta-scored moves. Same greedy/tie-break as a
    full recompute. Each unit may carry a `weight` (default 1.0, centered so the neutral value
    contributes no penalty); the priority penalty is separable per task, so a move's penalty delta
    is (weight - 1)*(time_new - time_cur)."""
    by_id = {u["id"]: u for u in units}
    cap = cap or capacity()
    a = {u["id"]: cost.best_lane(u["type"]) for u in units}
    # A shared type's cost is not separable per unit, since moving one changes what the
    # others on that lane cost, so the running state is per (lane, share group) rather than a
    # single float, and a move re-derives only the two groups it touched. Still O(1) per
    # candidate: _state_sums walks LANES and the few live groups, not the units.
    buses = _buses_of(cost)
    st = _new_state(getattr(cost, "batch_gain", None))
    for tid, ln in a.items():
        u = by_id[tid]
        t, b = cost.cost(u["type"], ln)
        _state_apply(st, u, u["type"], ln, t, b, +1)
    penalty = 0.0   # seed is best_lane for all, so the penalty starts at zero

    improved = True
    while improved:
        improved = False
        time, bw = _state_sums(st)
        base = _contention_from_sums(time, bw, cap, buses) + _LAMBDA_PRI * penalty
        best_move = None
        best_gain = 1e-12
        for u in units:
            tid = u["id"]
            cur = a[tid]
            ty = u["type"]
            w = float(u.get("weight", 1.0)) - 1.0
            tc, bc = cost.cost(ty, cur)
            for ln in LANES:
                if ln == cur:
                    continue
                tn, bn = cost.cost(ty, ln)
                _state_apply(st, u, ty, cur, tc, bc, -1)
                _state_apply(st, u, ty, ln, tn, bn, +1)
                t2, b2 = _state_sums(st)
                dpen = w * (tn - tc)
                cand = _contention_from_sums(t2, b2, cap, buses) + _LAMBDA_PRI * (penalty + dpen)
                gain = base - cand
                _state_apply(st, u, ty, ln, tn, bn, -1)
                _state_apply(st, u, ty, cur, tc, bc, +1)
                if gain > best_gain:
                    best_gain = gain
                    best_move = (tid, ln, cur, tc, bc, tn, bn, dpen, u, ty)
        if best_move is not None:
            tid, ln, cur, tc, bc, tn, bn, dpen, u, ty = best_move
            a[tid] = ln
            _state_apply(st, u, ty, cur, tc, bc, -1)
            _state_apply(st, u, ty, ln, tn, bn, +1)
            penalty += dpen
            improved = True
    return a


def detect_bus_topology() -> dict | None:
    """This machine's topology, or None when it cannot be determined.

    The probe lives in the agent layer (it is a fact about a host, not about the math),
    so this is a soft import: rexgraph stays usable without it and simply declines to
    guess. Any failure is a None, never an assertion.
    """
    try:
        from agent.local_runtime import detect_gpus
    except Exception:
        return None
    try:
        gpus = detect_gpus()
    except Exception:
        return None
    pick = None
    with_vram = [g for g in gpus if g.get("vram_gb")]
    if with_vram:
        pick = max(with_vram, key=lambda g: g["vram_gb"])
    elif gpus:
        pick = gpus[0]
    return bus_topology_for((pick or {}).get("unified"))


#### Dispatch seam: execute an assignment across the compute lanes
import logging as _logging
import pickle
import time as _time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

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


def execute(units: list, assignment: dict, cost: CostModel|None = None) -> dict:
    """Execute each unit's `fn` on its assigned lane: proc -> process pool (true multicore for CPU-
    bound work), thread/igpu -> thread pool (I/O and GPU-launch are GIL-light). Results are keyed by
    id and are INDEPENDENT of lane and order. Folds per-task timing into cost.

    Two guards make this safe for real (not just test) hive tasks:
    - PICKLABILITY: a proc-lane fn that cannot be pickled (a closure/lambda/bound-method over hive
      state) is transparently spilled to the thread lane instead of crashing the forkserver pool.
    - SIDE EFFECTS: the proc lane runs the fn in a child process, so in-process mutations do NOT
      propagate back, only the return value does. The picklability spill covers the common stateful
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


def _inner_threads(workers: int, cores_budget: int|None = None) -> int:
    """Inner native (BLAS/OpenMP) thread budget per proc worker: budget // workers, so
    workers * inner tracks the CORE BUDGET this pool is entitled to (the same arithmetic parallel_map
    uses). `cores_budget` is the machine cores for a single coordinator, but the hive's SHARE of the
    cores when several coordinators run at once, since otherwise N concurrent pools each assume all cores
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
                 affinity: bool = False, cap: dict|None = None, reaper_tick: float = 1.0,
                 cores_budget: int|None = None):
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

    #### lane pool management
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

    #### execution
    def run(self, units: list, assignment: dict, cost: CostModel|None = None) -> dict:
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


#### Coordinator: the per-wave plan -> execute -> learn loop (cadence = per-wave in v1)
class Coordinator:
    """Per-wave plan -> execute -> learn loop. With a `pools` (LanePools) it dispatches through the
    managed warm lanes; without, it uses per-wave `execute`. `cap` is an optional hive-share-scaled
    capacity used by the actuator."""

    def __init__(self, cost: CostModel|None = None, pools: LanePools|None = None,
                 cap: dict|None = None, buses: dict|None = None, detect: bool = True):
        self.cost = cost or CostModel()
        self.pools = pools
        self.cap = cap
        # Describe THIS machine. An explicit `buses` wins; otherwise ask the hardware,
        # and if it cannot tell, leave the model deferring to the process default rather
        # than asserting a topology. `detect=False` skips the probe for a coordinator
        # that stands for a machine other than the one it runs on. A remote bee's
        # hardware is not this box's.
        if buses is not None:
            self.cost.set_buses(buses)
        elif detect and cost is None:
            self.cost.set_buses(detect_bus_topology())

    @property
    def buses(self) -> dict:
        return self.cost.buses()

    def plan(self, units: list) -> dict:
        return assign(units, self.cost, cap=self.cap)

    def run_wave(self, units: list) -> dict:
        a = self.plan(units)
        if self.pools is not None:
            return self.pools.run(units, a, cost=self.cost)
        return execute(units, a, cost=self.cost)
