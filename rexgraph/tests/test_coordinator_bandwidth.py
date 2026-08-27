"""The contention objective counts every lane's bandwidth draw.

`min(proc, igpu)` said the right thing for two lanes, contention being what is CO-DRAWN
since a lane drawing alone owns the bus, but it excluded the third, so anything on the
thread lane drew for free. Free being cheap, the actuator PREFERRED that lane, which is
exactly where a blocking call into a local model lands, and a local model is the
heaviest draw on the box (~87% of memory bandwidth, measured).

The fix is `total - max`, which for two terms IS the minimum, so it is a generalisation
and not a replacement: with no thread draw it returns the old value bit for bit.
"""
import numpy as np
import pytest

import rexgraph.coordinator as C


def _cap():
    return C.capacity()


def _caps():
    """Per-lane capacity across a range of core counts. `capacity()` reads
    os.cpu_count(), so anything asserted against the host's own capacity is a claim
    about the machine the test happens to run on: a 3-core runner gives 1 per lane
    where a 32-core box gives 16, and the wall-clock term scales with it."""
    return [{"proc": float(max(1, c // 2)), "thread": float(max(1, c // 2)),
             "igpu": 2.0} for c in (2, 3, 4, 8, 16, 32, 64)]


def _bw_term(a, units, cm, cap):
    """The bandwidth half of the objective, isolated by zeroing the wall clock."""
    _, bw = C._lane_groups(a, units, cm)
    return C._contention_from_sums({ln: 0.0 for ln in C.LANES}, bw, cap,
                                   C._buses_of(cm))


def _llm_units(ty):
    return ([{"id": f"l{i}", "type": ty} for i in range(4)] +
            [{"id": f"g{i}", "type": "gpu_kernel"} for i in range(4)])


def _fixed(ty):
    """One assignment held constant, so swapping the type moves the COST and nothing
    else. Letting `assign` run twice compares two different problems: raising the
    thread lane's draw makes the greedy re-place the gpu_kernel units, and at 16 per
    lane it answers by moving two of them ONTO the thread lane, which lowers
    `total - max` for local_llm below io_llm's. That is the actuator doing its job,
    not the type being cheaper."""
    units = _llm_units(ty)
    return {u["id"]: ("thread" if u["id"][0] == "l" else "igpu") for u in units}, units


def _old(time, bw, cap):
    """The previous objective, for the equivalence test."""
    wall = max(time[ln] / cap[ln] for ln in C.LANES)
    return wall + C._BW_LAMBDA * min(bw["proc"], bw["igpu"])


def test_sum_minus_max_is_the_minimum_for_two_terms():
    """The identity the generalisation rests on: min(a,b) = (a+b) - max(a,b)."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        a, b = rng.uniform(0, 5, 2)
        assert abs(((a + b) - max(a, b)) - min(a, b)) < 1e-12


def test_it_is_bit_identical_when_the_thread_lane_draws_nothing():
    """The safety property. Every placement that was scored correctly before must be
    scored the same now, so the correction is confined to the case that was wrong."""
    rng = np.random.default_rng(1)
    cap = _cap()
    worst = 0.0
    for _ in range(3000):
        bw = {"proc": float(rng.uniform(0, 3)), "igpu": float(rng.uniform(0, 3)),
              "thread": 0.0}
        t = {ln: float(rng.uniform(0, 2)) for ln in C.LANES}
        worst = max(worst, abs(C._contention_from_sums(t, bw, cap) - _old(t, bw, cap)))
    assert worst < 1e-12, worst


def test_it_never_under_counts_once_the_thread_lane_draws():
    """Counting a draw that was ignored can only raise the score, never lower it."""
    rng = np.random.default_rng(2)
    cap = _cap()
    for _ in range(3000):
        bw = {ln: float(rng.uniform(0, 3)) for ln in C.LANES}
        t = {ln: float(rng.uniform(0, 2)) for ln in C.LANES}
        assert C._contention_from_sums(t, bw, cap) >= _old(t, bw, cap) - 1e-12


def test_a_lane_drawing_alone_still_pays_nothing():
    """The idea being preserved: one draw and no other is not contention. If this broke,
    the fix would have turned a co-drawn term into a plain total."""
    cap = _cap()
    t = {ln: 0.1 for ln in C.LANES}
    for lane in C.LANES:
        bw = {ln: (0.9 if ln == lane else 0.0) for ln in C.LANES}
        wall = max(t[ln] / cap[ln] for ln in C.LANES)
        assert abs(C._contention_from_sums(t, bw, cap) - wall) < 1e-12, lane


def test_the_thread_lane_no_longer_draws_for_free():
    """The defect, isolated from wall-clock. Hold TIME equal across lanes so only the
    bandwidth term can move, then put a draw on each lane against a fixed igpu draw.
    Under `min(proc, igpu)` the thread placement scored strictly less than proc because
    its bandwidth was not counted at all; now the two agree."""
    cap = _cap()
    t = {ln: 0.1 for ln in C.LANES}          # equal time: bandwidth is the only variable
    fixed = 0.9                              # something else already drawing on the igpu
    scores = {}
    for lane in C.LANES:
        bw = {ln: 0.0 for ln in C.LANES}
        bw[lane] += 0.9
        bw["igpu"] += fixed
        scores[lane] = C._contention_from_sums(t, bw, cap)
    assert abs(scores["thread"] - scores["proc"]) < 1e-12, scores
    assert scores["thread"] > min(t[ln] / cap[ln] for ln in C.LANES) + 1e-9, scores
    # and the old formula scored the thread placement at exactly the wall-clock: free
    bw_thread = {"proc": 0.0, "thread": 0.9, "igpu": fixed}
    assert abs(_old(t, bw_thread, cap)
               - max(t[ln] / cap[ln] for ln in C.LANES)) < 1e-12


def test_local_llm_is_distinguished_from_io_llm():
    """They tied at 0.1180 before, despite 0.9 bandwidth against 0.1, because the
    difference sat on the invisible lane. What the fix moved is the BANDWIDTH term,
    so that is what is read, at every capacity.

    Deliberately not the TOTAL. A local model is also a shared type, so four
    concurrent units divide their wall clock by the batch gain, and that pulls the
    total the other way: at 1 per lane io_llm totals 0.4080 against local_llm's
    0.2812, and from 2 per lane up the order reverses. Which of the two effects
    wins is a property of the host's core count, not of the cost model, so the
    total is the wrong thing to assert. The bandwidth term is above io_llm's
    everywhere."""
    cm = C.CostModel()
    for cap in _caps():
        bwt, total = {}, {}
        for ty in ("io_llm", "local_llm"):
            a, units = _fixed(ty)
            bwt[ty] = _bw_term(a, units, cm, cap)
            total[ty] = C.contention(a, units, cm, cap)
        assert bwt["local_llm"] > bwt["io_llm"] + 1e-6, (cap, bwt)
        assert abs(total["local_llm"] - total["io_llm"]) > 1e-6, (cap, total)


def test_local_llm_carries_the_measured_bandwidth():
    """A local model is ~87% of the bus, not the 0.1 that describes a remote socket."""
    cm = C.CostModel()
    for lane in C.LANES:
        assert cm.cost("local_llm", lane)[1] == pytest.approx(0.9)
        assert cm.cost("io_llm", lane)[1] == pytest.approx(0.1)
    assert "local_llm" in C.TYPES


def test_the_objective_stays_nonnegative():
    rng = np.random.default_rng(3)
    cap = _cap()
    for _ in range(2000):
        bw = {ln: float(rng.uniform(0, 4)) for ln in C.LANES}
        t = {ln: float(rng.uniform(0, 3)) for ln in C.LANES}
        assert C._contention_from_sums(t, bw, cap) >= 0.0


def test_local_llm_keeps_the_thread_lane_strictly_cheapest():
    """A correctness constraint, not a preference. agent.coordinator_adapter routes LLM
    work to a thread lane partly because a spawn or a live-server attach mutates hive
    state IN PLACE and must not run in a forkserver child, where the mutation would be
    lost. If local_llm ever made proc or igpu as cheap in TIME, that work would migrate
    off-thread and the mutation would vanish. The correction to this type is the bus
    accounting; the placement must stay put."""
    cm = C.CostModel()
    for ty in ("io_llm", "local_llm"):
        assert cm.best_lane(ty) == "thread", ty
        t = {ln: cm.cost(ty, ln)[0] for ln in C.LANES}
        assert t["thread"] < t["proc"] and t["thread"] < t["igpu"], (ty, t)
    # and the two must agree on TIME, differing only in bandwidth
    for ln in C.LANES:
        assert cm.cost("io_llm", ln)[0] == cm.cost("local_llm", ln)[0], ln


def test_local_llm_placement_matches_io_llm_but_draws_more():
    """The fix must change what a placement COSTS, not which placement is chosen.
    Held across the capacity range, since the placement is what has to be stable."""
    cm = C.CostModel()
    for cap in _caps():
        lanes, bwt = {}, {}
        for ty in ("io_llm", "local_llm"):
            units = _llm_units(ty)
            a = C.assign(units, cm, cap)
            lanes[ty] = sorted({ln for t, ln in a.items() if t[0] == "l"})
            bwt[ty] = _bw_term(*_fixed(ty), cm, cap)
        # the placement the actuator chooses for the model itself is unchanged
        assert lanes["io_llm"] == lanes["local_llm"] == ["thread"], (cap, lanes)
        # and on one held placement the local model is the heavier draw
        assert bwt["local_llm"] > bwt["io_llm"] + 1e-6, (cap, bwt)
