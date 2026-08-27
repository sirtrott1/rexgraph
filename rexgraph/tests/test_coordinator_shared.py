"""A shared resource costs less shared than it does N times over.

Concurrent requests to ONE local server read the weights once per decode step and serve
every one of them, and their wall-clock overlaps. Measured on the 35B-A3B MoE: aggregate
decode 53.4 tok/s at one stream against 120.8 at eight, so 2.26x. An objective that sums
per unit cannot say that, and will spread generations to level per-lane load when putting
them on one server is the faster placement.

Everything not in `_SHARED_TYPES` stays strictly additive, and one unit of a shared type
costs exactly what it did before, so the change is confined to the case it is about.
"""
import numpy as np
import pytest

import rexgraph.coordinator as C


def _units(n, ty, group=None, prefix="u"):
    out = []
    for i in range(n):
        u = {"id": f"{prefix}{i}", "type": ty}
        if group:
            u["share_group"] = group
        out.append(u)
    return out


def _sums(assignment, units, cost):
    return C._lane_groups(assignment, units, cost)


def test_one_shared_unit_costs_what_it_always_did():
    """Backward compatibility where it matters: a single generation is unchanged."""
    cm = C.CostModel()
    for lane in C.LANES:
        u = _units(1, "local_llm")
        t, b = _sums({"u0": lane}, u, cm)
        want_t, want_b = cm.cost("local_llm", lane)
        assert t[lane] == pytest.approx(want_t)
        assert b[lane] == pytest.approx(want_b)


def test_the_bus_draw_does_not_grow_with_the_batch():
    """One server, one weight stream. Eight concurrent generations do not read the
    weights eight times, so the bandwidth term must not multiply."""
    cm = C.CostModel()
    solo = cm.cost("local_llm", "thread")[1]
    for n in (1, 2, 4, 8, 32):
        u = _units(n, "local_llm")
        _t, b = _sums({f"u{i}": "thread" for i in range(n)}, u, cm)
        assert b["thread"] == pytest.approx(solo), n


def test_wall_clock_is_sub_additive_but_still_increasing():
    """Cheaper together, never free: n/gain(n) has to keep rising or the greedy would
    hoard units onto one lane to keep driving the cost down."""
    cm = C.CostModel()
    per = cm.cost("local_llm", "thread")[0]
    got = []
    for n in (1, 2, 4, 8, 16):
        u = _units(n, "local_llm")
        t, _b = _sums({f"u{i}": "thread" for i in range(n)}, u, cm)
        got.append(t["thread"])
        assert t["thread"] < n * per + 1e-12, n          # sub-additive
    assert all(b > a for a, b in zip(got, got[1:], strict=False)), got  # and monotone


def test_an_additive_type_is_untouched():
    cm = C.CostModel()
    for ty in ("io_llm", "gpu_kernel", "cpu_coordination"):
        per_t, per_b = cm.cost(ty, "thread")
        for n in (1, 3, 7):
            u = _units(n, ty)
            t, b = _sums({f"u{i}": "thread" for i in range(n)}, u, cm)
            assert t["thread"] == pytest.approx(n * per_t), (ty, n)
            assert b["thread"] == pytest.approx(n * per_b), (ty, n)


def test_separate_share_groups_share_nothing():
    """Three bees are three servers. Requests to different bees do not share weight
    reads, and saying otherwise would under-price a fleet badly."""
    cm = C.CostModel()
    same = _units(4, "local_llm", group="beeA")
    split = (_units(2, "local_llm", group="beeA", prefix="a") +
             _units(2, "local_llm", group="beeB", prefix="b"))
    t_same, b_same = _sums({u["id"]: "thread" for u in same}, same, cm)
    t_split, b_split = _sums({u["id"]: "thread" for u in split}, split, cm)
    assert b_split["thread"] > b_same["thread"] + 1e-12       # two weight streams
    assert t_split["thread"] > t_same["thread"] + 1e-12       # two smaller batches


def test_the_batch_gain_is_flat_above_what_was_measured():
    """8 is the largest measured point, so a wave of 40 gets the gain of 8 rather than an
    extrapolation nobody checked."""
    top = max(C.LOCAL_LLM_BATCH_GAIN)
    for n in (top, top + 1, top * 5):
        assert C._batch_gain(n) == pytest.approx(C.LOCAL_LLM_BATCH_GAIN[top])
    assert C._batch_gain(1) == 1.0
    assert C._batch_gain(0) == 1.0
    for a, b in zip(range(1, 20), range(2, 21), strict=False):        # gain never decreases
        assert C._batch_gain(b) >= C._batch_gain(a) - 1e-12


def test_the_incremental_actuator_agrees_with_a_full_recompute():
    """assign() maintains its state incrementally and its docstring promises the same
    answer as a recompute. Sub-additive costs are not separable per unit, so this is the
    invariant most at risk from the change."""
    cm = C.CostModel()
    rng = np.random.default_rng(0)
    types = ("local_llm", "io_llm", "gpu_kernel", "cpu_coordination")
    for trial in range(60):
        n = int(rng.integers(2, 10))
        units = []
        for i in range(n):
            ty = types[int(rng.integers(0, len(types)))]
            u = {"id": f"u{i}", "type": ty}
            if ty == "local_llm" and rng.random() < 0.5:
                u["share_group"] = f"bee{int(rng.integers(0, 2))}"
            units.append(u)
        a = C.assign(units, cm)
        # the state the actuator carried must match one built from scratch
        t_inc, b_inc = C._lane_groups(a, units, cm)
        st = C._new_state()
        for u in units:
            tt, bb = cm.cost(u["type"], a[u["id"]])
            C._state_apply(st, u, u["type"], a[u["id"]], tt, bb, +1)
        t_fresh, b_fresh = C._state_sums(st)
        for ln in C.LANES:
            assert t_inc[ln] == pytest.approx(t_fresh[ln]), (trial, ln)
            assert b_inc[ln] == pytest.approx(b_fresh[ln]), (trial, ln)


def test_state_apply_is_reversible():
    """The candidate loop applies a move, scores it, and undoes it. If undo leaked, the
    scores would drift as the sweep proceeded."""
    cm = C.CostModel()
    st = C._new_state()
    us = _units(3, "local_llm") + _units(2, "gpu_kernel", prefix="g")
    for u in us:
        t, b = cm.cost(u["type"], "thread")
        C._state_apply(st, u, u["type"], "thread", t, b, +1)
    before = C._state_sums(st)
    u = us[0]
    t, b = cm.cost(u["type"], "thread")
    tn, bn = cm.cost(u["type"], "igpu")
    C._state_apply(st, u, u["type"], "thread", t, b, -1)
    C._state_apply(st, u, u["type"], "igpu", tn, bn, +1)
    C._state_apply(st, u, u["type"], "igpu", tn, bn, -1)
    C._state_apply(st, u, u["type"], "thread", t, b, +1)
    after = C._state_sums(st)
    for i in (0, 1):
        for ln in C.LANES:
            assert before[i][ln] == pytest.approx(after[i][ln]), ln
    assert all(not st["grp"][ln] or all(e[0] > 0 for e in st["grp"][ln].values())
               for ln in C.LANES), "no zero-count groups left behind"


def test_batching_generations_beats_spreading_them():
    """The behaviour this exists for. Four generations on one server should not be split
    across lanes to level load when co-scheduling them is the cheaper placement."""
    cm = C.CostModel()
    units = _units(4, "local_llm", group="bee")
    together = C.contention({u["id"]: "thread" for u in units}, units, cm)
    spread = C.contention({"u0": "thread", "u1": "proc", "u2": "igpu", "u3": "thread"},
                          units, cm)
    assert together < spread, (together, spread)


#### the gain is learned, and a learned curve must not be able to break the search
def test_the_gain_table_learns_from_measured_throughput():
    """Same shape as CostModel.observe for times: feed what was measured, EMA it in."""
    cm = C.CostModel()
    seeded = cm.batch_gain(8)
    cm.observe_throughput(1, 60.3)
    for _ in range(30):
        cm.observe_throughput(8, 200.0)          # this box scales better than the seed
    assert cm.batch_gain(8) > seeded + 1e-6
    assert cm.gain_table()["baseline_tok_s"] == pytest.approx(60.3, rel=1e-6)


def test_a_gain_sample_before_any_baseline_is_replayed_not_dropped():
    """A gain is a ratio, so an n>1 sample means nothing until an n=1 one exists. Dropping
    it would discard the first wave of a fresh process, which is when the seed is least
    trustworthy."""
    cm = C.CostModel()
    before = cm.batch_gain(4)
    for _ in range(20):
        cm.observe_throughput(4, 300.0)          # no baseline yet: held
    assert cm.batch_gain(4) == pytest.approx(before)
    cm.observe_throughput(1, 60.0)               # baseline arrives, held samples fold in
    assert cm.batch_gain(4) > before + 1e-6


@pytest.mark.parametrize("bad", [6000.0, 1e9, 1e-9])
def test_no_measurement_can_make_a_bigger_batch_cheaper(bad):
    """The invariant the greedy rests on. gain may rise with n but never faster than n,
    so n/gain(n) cannot fall; if it could, absorbing a unit would lower the wave's cost
    and the actuator would hoard every unit onto one lane."""
    cm = C.CostModel()
    cm.observe_throughput(1, 60.0)
    for _ in range(50):
        for n in (2, 4, 8):
            cm.observe_throughput(n, bad)
    ratios = [n / cm.batch_gain(n) for n in range(1, 24)]
    assert all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:], strict=False)), ratios
    gains = [cm.batch_gain(n) for n in range(1, 24)]
    assert all(b >= a - 1e-9 for a, b in zip(gains, gains[1:], strict=False)), gains


@pytest.mark.parametrize("junk", [0.0, -5.0, float("nan"), float("inf")])
def test_junk_throughput_is_ignored(junk):
    cm = C.CostModel()
    cm.observe_throughput(1, 60.0)
    before = dict(cm.gain_table()["gain"])
    cm.observe_throughput(4, junk)
    cm.observe_throughput(1, junk)
    assert cm.gain_table()["gain"] == before
    assert cm.gain_table()["baseline_tok_s"] == pytest.approx(60.0)


def test_the_learned_curve_reaches_the_objective():
    """A model that has been fed measurements must price batches by what it saw, not by
    the module seed. Otherwise the learning is decorative."""
    units = _units(8, "local_llm", group="bee")
    a = {u["id"]: "thread" for u in units}
    seed = C.CostModel()
    learned = C.CostModel()
    learned.observe_throughput(1, 60.3)
    for _ in range(30):
        learned.observe_throughput(8, 200.0)
    t_seed, _ = C._lane_groups(a, units, seed)
    t_learn, _ = C._lane_groups(a, units, learned)
    assert t_learn["thread"] < t_seed["thread"] - 1e-9


def test_two_models_do_not_share_a_curve():
    """The seed is one model on one box. Two coordinators on different hardware must not
    overwrite each other, which is why the table is per-CostModel and not the module dict."""
    a, b = C.CostModel(), C.CostModel()
    a.observe_throughput(1, 60.0)
    for _ in range(30):
        a.observe_throughput(8, 240.0)
    assert a.batch_gain(8) != pytest.approx(b.batch_gain(8))
    assert b.batch_gain(8) == pytest.approx(C.LOCAL_LLM_BATCH_GAIN[8])
    assert C.LOCAL_LLM_BATCH_GAIN[8] == 2.26, "the module seed was mutated"


def test_the_incremental_state_does_not_drift():
    """assign() applies and reverts a move for every unit-lane pair it scores, so float
    residue would accumulate over a long wave and the answer would depend on how many
    candidates were examined."""
    cm = C.CostModel()
    rng = np.random.default_rng(7)
    types = ("local_llm", "io_llm", "gpu_kernel", "cpu_coordination")
    worst = 0.0
    for _ in range(25):
        n = int(rng.integers(20, 100))
        units = []
        for i in range(n):
            ty = types[int(rng.integers(0, len(types)))]
            u = {"id": f"u{i}", "type": ty}
            if ty == "local_llm" and rng.random() < 0.5:
                u["share_group"] = f"bee{int(rng.integers(0, 3))}"
            units.append(u)
        a = C.assign(units, cm)
        t_i, b_i = C._lane_groups(a, units, cm)
        st = C._new_state(cm.batch_gain)
        for u in units:
            tt, bb = cm.cost(u["type"], a[u["id"]])
            C._state_apply(st, u, u["type"], a[u["id"]], tt, bb, +1)
        t_f, b_f = C._state_sums(st)
        for ln in C.LANES:
            worst = max(worst, abs(t_i[ln] - t_f[ln]), abs(b_i[ln] - b_f[ln]))
    assert worst == 0.0, worst
