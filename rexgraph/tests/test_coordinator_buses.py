"""Which lanes share a bus is hardware, so it is declared, not hardcoded.

`min(proc, igpu)` baked in unified memory. That is right on the 8060S, where the iGPU
and the CPU are on one physical bus, and wrong on a discrete-GPU box where VRAM is its
own pool and an igpu draw costs a CPU draw nothing.

The topology is a complex, edge-primary: a bus is a vertex, a LANE is the relation over
the buses it draws on, and two lanes meet exactly when they share one. A lane on a single
bus is a 1-ary witness relation, and two witnesses on one vertex meet there, so the
unified case falls out of the incidence rather than being special-cased.
"""
import numpy as np
import pytest

import rexgraph.coordinator as C
from rexgraph.sheaf import Sheaf


@pytest.fixture(autouse=True)
def _restore():
    """The topology is process state, so every test puts it back."""
    before = C.bus_topology()
    yield
    C.set_bus_topology(before)


def _t():
    return {ln: 0.1 for ln in C.LANES}


def _old(time, bw, cap):
    """The formula before the topology was declarable: unified memory, hardcoded."""
    wall = max(time[ln] / cap[ln] for ln in C.LANES)
    draws = [bw[ln] for ln in C.LANES]
    return wall + C._BW_LAMBDA * (sum(draws) - max(draws))


def test_the_default_is_unified_memory():
    assert C.bus_topology() == {"mem": ("proc", "thread", "igpu")}


def test_unified_memory_reproduces_the_previous_formula_exactly():
    """The correction must not move the machine it was measured on."""
    C.set_bus_topology(C.UNIFIED_MEMORY)
    cap = C.capacity()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3000):
        bw = {ln: float(rng.uniform(0, 3)) for ln in C.LANES}
        t = {ln: float(rng.uniform(0, 2)) for ln in C.LANES}
        worst = max(worst, abs(C._contention_from_sums(t, bw, cap) - _old(t, bw, cap)))
    assert worst < 1e-12, worst


def test_split_memory_frees_the_igpu_from_the_cpu_lanes():
    """A discrete GPU has its own pool: a VRAM draw and a system-memory draw are not at
    war, and the unified formula would have charged for it."""
    C.set_bus_topology(C.SPLIT_MEMORY)
    cap, t = C.capacity(), _t()
    wall = max(t[ln] / cap[ln] for ln in C.LANES)
    only = {"proc": 0.9, "thread": 0.0, "igpu": 0.9}
    assert C._contention_from_sums(t, only, cap) == pytest.approx(wall)
    both = {"proc": 0.9, "thread": 0.9, "igpu": 0.0}
    assert C._contention_from_sums(t, both, cap) > wall + 1e-9


def test_a_lane_alone_on_its_bus_pays_nothing():
    """The idea the pairwise min encoded, preserved per bus rather than globally."""
    for topo in (C.UNIFIED_MEMORY, C.SPLIT_MEMORY):
        C.set_bus_topology(topo)
        cap, t = C.capacity(), _t()
        wall = max(t[ln] / cap[ln] for ln in C.LANES)
        for lane in C.LANES:
            bw = {ln: (0.9 if ln == lane else 0.0) for ln in C.LANES}
            assert C._contention_from_sums(t, bw, cap) == pytest.approx(wall), (topo, lane)


def test_a_lane_on_two_buses_contends_on_both():
    """The genuinely OVERLAPPING cover, which a partition could not express: thread draws
    from both pools, so it is at war with proc on one and igpu on the other."""
    C.set_bus_topology({"a": ("proc", "thread"), "b": ("thread", "igpu")})
    cap, t = C.capacity(), _t()
    wall = max(t[ln] / cap[ln] for ln in C.LANES)
    shared = C._contention_from_sums(t, {"proc": 0.9, "thread": 0.9, "igpu": 0.9}, cap)
    # proc<->igpu share nothing, so they alone are free
    ends = C._contention_from_sums(t, {"proc": 0.9, "thread": 0.0, "igpu": 0.9}, cap)
    assert ends == pytest.approx(wall)
    assert shared > ends + 1e-9


@pytest.mark.parametrize("topo,expect", [
    (C.UNIFIED_MEMORY, {("proc", "thread"), ("proc", "igpu"), ("thread", "igpu")}),
    (C.SPLIT_MEMORY, {("proc", "thread")}),
    ({"a": ("proc", "thread"), "b": ("thread", "igpu")},
     {("proc", "thread"), ("thread", "igpu")}),
])
def test_the_sheaf_recovers_the_sharing_from_the_incidence(topo, expect):
    """The declaration is a complex, so which lanes meet is READ from it rather than
    restated: Sheaf.mediators over the bus vertices gives exactly the sharing pairs."""
    C.set_bus_topology(topo)
    rex, lanes = C.bus_complex()
    sh = Sheaf(rex, stalk_dim=1, grade=1)
    got = {tuple(sorted((lanes[a], lanes[b]))) for a, b, _m in sh.meets()}
    assert got == {tuple(sorted(p)) for p in expect}


def test_the_bus_complex_is_edge_primary_and_arity_general():
    """Lanes are the relations and buses their boundary. On unified memory each lane is a
    1-ary witness on one vertex; on an overlapping cover a lane becomes 2-ary."""
    C.set_bus_topology(C.UNIFIED_MEMORY)
    rex, lanes = C.bus_complex()
    from rexgraph.harmonic_sparse import _b1_csc
    arity = np.diff(_b1_csc(rex).indptr)
    assert len(lanes) == 3 and int(rex.nV) == 1
    assert set(arity.tolist()) == {1}, arity
    C.set_bus_topology({"a": ("proc", "thread"), "b": ("thread", "igpu")})
    rex2, _ = C.bus_complex()
    arity2 = np.diff(_b1_csc(rex2).indptr)
    assert sorted(arity2.tolist()) == [1, 1, 2], arity2


def test_an_unknown_lane_is_refused():
    with pytest.raises(ValueError, match="unknown lanes"):
        C.set_bus_topology({"mem": ("proc", "gpu0")})
    assert C.bus_topology() == {"mem": ("proc", "thread", "igpu")}


def test_none_restores_the_default():
    C.set_bus_topology(C.SPLIT_MEMORY)
    C.set_bus_topology(None)
    assert C.bus_topology() == C.UNIFIED_MEMORY


def test_placement_responds_to_the_topology():
    """The point of declaring it: on split memory the actuator may use the igpu without
    paying a bandwidth war it is not actually in."""
    cm = C.CostModel()
    units = ([{"id": f"g{i}", "type": "gpu_kernel"} for i in range(3)] +
             [{"id": f"c{i}", "type": "cpu_coordination"} for i in range(3)])
    scores = {}
    for name, topo in (("unified", C.UNIFIED_MEMORY), ("split", C.SPLIT_MEMORY)):
        C.set_bus_topology(topo)
        scores[name] = C.contention(C.assign(units, cm), units, cm)
    assert scores["split"] <= scores["unified"] + 1e-12, scores


#### a fleet is heterogeneous, so the topology belongs to the machine
def test_two_models_hold_different_hardware_at_once():
    """The reason this is per-model: a hive spanning a unified-memory laptop and a
    discrete-GPU desktop has to describe both simultaneously, and a process-global
    topology can only be one of them."""
    laptop, desktop = C.CostModel(), C.CostModel()
    laptop.set_buses(C.UNIFIED_MEMORY)
    desktop.set_buses(C.SPLIT_MEMORY)
    units = [{"id": "g0", "type": "gpu_kernel"}, {"id": "c0", "type": "cpu_coordination"}]
    a = {"g0": "igpu", "c0": "proc"}
    assert laptop.buses() == C.UNIFIED_MEMORY
    assert desktop.buses() == C.SPLIT_MEMORY
    # on split memory the VRAM draw and the system draw are not at war
    assert C.contention(a, units, desktop) < C.contention(a, units, laptop) - 1e-9


def test_an_undeclared_model_defers_to_the_process_default():
    m = C.CostModel()
    assert m.buses() == C.bus_topology()
    C.set_bus_topology(C.SPLIT_MEMORY)
    assert m.buses() == C.SPLIT_MEMORY, "an undeclared model follows the process"


def test_a_declared_model_ignores_the_process_default():
    """Otherwise one machine reconfiguring the process would silently re-describe another."""
    m = C.CostModel()
    m.set_buses(C.UNIFIED_MEMORY)
    C.set_bus_topology(C.SPLIT_MEMORY)
    assert m.buses() == C.UNIFIED_MEMORY
    m.set_buses(None)
    assert m.buses() == C.SPLIT_MEMORY, "None goes back to deferring"


def test_set_buses_validates_lane_names():
    m = C.CostModel()
    with pytest.raises(ValueError, match="unknown lanes"):
        m.set_buses({"vram": ("igpu", "npu")})
    assert m.buses() == C.bus_topology(), "a refused declaration must not half-apply"


def test_assign_respects_the_model_topology_not_the_process_one():
    """contention() is not the only path: the actuator scores candidates itself, so it has
    to read the same topology or placement and price would disagree."""
    C.set_bus_topology(C.UNIFIED_MEMORY)
    units = ([{"id": f"g{i}", "type": "gpu_kernel"} for i in range(3)] +
             [{"id": f"c{i}", "type": "cpu_coordination"} for i in range(3)])
    split = C.CostModel()
    split.set_buses(C.SPLIT_MEMORY)
    a = C.assign(units, split)
    # the score the actuator converged on must equal a fresh evaluation under the SAME
    # topology; if assign() had used the process default these would differ
    assert C.contention(a, units, split) == pytest.approx(
        C._contention_from_sums(*C._lane_groups(a, units, split), C.capacity(),
                                C.SPLIT_MEMORY)
        + C._LAMBDA_PRI * C._priority_penalty(a, units, split))


def test_a_mixed_fleet_prices_the_same_wave_differently_per_machine():
    """Four machines, three with their own VRAM and one unified. The same wave should not
    cost the same everywhere, which is the whole point of describing them separately."""
    fleet = {}
    for name, topo in (("laptop-unified", C.UNIFIED_MEMORY),
                       ("desk-7900xtx", C.SPLIT_MEMORY),
                       ("desk-7900xt", C.SPLIT_MEMORY),
                       ("desk-3070", C.SPLIT_MEMORY)):
        m = C.CostModel()
        m.set_buses(topo)
        fleet[name] = m
    units = ([{"id": f"g{i}", "type": "gpu_kernel"} for i in range(2)] +
             [{"id": f"l{i}", "type": "local_llm"} for i in range(4)])
    scores = {n: C.contention(C.assign(units, m), units, m) for n, m in fleet.items()}
    assert len({round(v, 9) for v in scores.values()}) > 1, scores
    assert scores["desk-7900xtx"] == pytest.approx(scores["desk-7900xt"])
    assert scores["laptop-unified"] != pytest.approx(scores["desk-3070"])


#### auto-configuration from the hardware
@pytest.mark.parametrize("unified,expect", [
    (True, C.UNIFIED_MEMORY), (False, C.SPLIT_MEMORY), (None, None),
])
def test_bus_topology_for_refuses_to_guess(unified, expect):
    """None must stay None. A wrong topology silently mis-prices every bandwidth
    decision; an absent one only asks the caller to declare."""
    got = C.bus_topology_for(unified)
    assert got == (dict(expect) if expect else None)


def test_a_coordinator_describes_the_machine_it_runs_on():
    co = C.Coordinator()
    detected = C.detect_bus_topology()
    if detected is None:
        assert co.buses == C.bus_topology()      # deferring, not guessing
    else:
        assert co.buses == detected


def test_an_explicit_topology_beats_detection():
    co = C.Coordinator(buses=C.SPLIT_MEMORY)
    assert co.buses == C.SPLIT_MEMORY


def test_detect_can_be_declined_for_a_remote_machine():
    """A coordinator standing for another host must not describe itself with THIS box's
    hardware."""
    cm = C.CostModel()
    C.Coordinator(cost=cm, detect=True)
    assert cm.buses() == C.bus_topology(), "a supplied model is not overwritten"
    co2 = C.Coordinator(detect=False)
    assert co2.buses == C.bus_topology()


def test_detection_never_raises():
    """It is a probe of the world, so it fails to None rather than upward."""
    assert C.detect_bus_topology() is None or isinstance(C.detect_bus_topology(), dict)
