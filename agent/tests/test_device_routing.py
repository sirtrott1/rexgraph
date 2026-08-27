"""Splitting a batch across device-pinned bees, when the devices contend.

Measured on this laptop (Qwen2.5-Coder-7B Q4, unified memory), with the contended rates
taken under SUSTAINED mutual load rather than one generation each:

    igpu solo 46.98  contended 31.94 (68%)
    cpu  solo 21.41  contended  8.23 (38%)

so the pair makes 40.17 tok/s against 46.98 for the iGPU alone. On unified memory,
co-scheduling the CPU alongside the iGPU is a NET LOSS: the CPU bee takes more bandwidth
than it contributes. A scheduler that assumes more devices is more throughput takes that
trade every time.
"""
import pytest
from agent.device_routing import (
    POLICIES,
    DeviceRate,
    co_scheduling_pays,
    expected_makespan,
    plan_split,
)

UNIFIED = [DeviceRate("igpu", "igpu", 46.98, 31.94),
           DeviceRate("cpu", "cpu", 21.41, 8.23)]
DISCRETE = [DeviceRate("gpu", "gpu:0", 60.0, 57.0),
            DeviceRate("cpu", "cpu", 20.0, 18.0)]


def test_co_scheduling_is_refused_when_it_loses():
    """The measured unified case: together 40.17, best alone 46.98."""
    worth, agg, best = co_scheduling_pays(UNIFIED)
    assert worth is False
    assert agg == pytest.approx(40.17, abs=0.01)
    assert best == pytest.approx(46.98, abs=0.01)


def test_co_scheduling_is_taken_when_it_wins():
    worth, agg, best = co_scheduling_pays(DISCRETE)
    assert worth is True and agg > best


def test_the_contention_policy_uses_one_device_when_two_are_worse():
    """The point of pricing contention is knowing when NOT to spread. On unified memory
    the contention-aware answer is the same as the naive fast one, and that is correct
    rather than a failure to be clever."""
    split = plan_split(UNIFIED, 16, "contention")
    assert split == {"igpu": 16, "cpu": 0}
    assert split == plan_split(UNIFIED, 16, "fastest")


def test_the_contention_policy_does_spread_when_it_pays():
    split = plan_split(DISCRETE, 16, "contention")
    assert split["cpu"] > 0 and split["gpu"] > split["cpu"]
    assert sum(split.values()) == 16


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("n", [0, 1, 7, 16, 33])
def test_every_policy_places_exactly_the_work_given(policy, n):
    assert sum(plan_split(UNIFIED, n, policy).values()) == n


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="policy must be one of"):
        plan_split(UNIFIED, 4, "greedy")


def test_the_makespan_is_piecewise_because_contention_ends():
    """Charging the contended rate for the whole run predicted 116.7s for an even split
    that actually took 61.0s: the iGPU finishes early and the CPU runs the tail alone at
    more than twice its contended rate."""
    even = {"igpu": 8, "cpu": 8}
    got = expected_makespan(UNIFIED, even, 120)
    assert 55.0 < got < 70.0, got                     # measured 61.01
    flat = 8 * 120 / 8.23                             # the old, wrong, whole-run form
    assert got < flat - 20.0, (got, flat)


def test_the_makespan_matches_what_was_measured():
    """fastest 39.40s and round_robin 61.01s were measured; the model has to land near
    them or it cannot be used to choose between them."""
    assert expected_makespan(UNIFIED, {"igpu": 16, "cpu": 0}, 120) == pytest.approx(
        39.40, rel=0.10)
    assert expected_makespan(UNIFIED, {"igpu": 8, "cpu": 8}, 120) == pytest.approx(
        61.01, rel=0.10)


def test_a_lone_bee_is_charged_its_solo_rate():
    """Nothing else is running, so there is nothing to contend with."""
    solo = expected_makespan(UNIFIED, {"igpu": 10, "cpu": 0}, 120)
    assert solo == pytest.approx(10 * 120 / 46.98, rel=1e-6)


def test_a_bee_with_no_measured_contention_falls_back_to_solo():
    r = [DeviceRate("a", "cpu", 10.0), DeviceRate("b", "cpu", 5.0)]
    assert r[0].busy == 10.0
    assert plan_split(r, 6, "contention")            # does not raise on missing data


def test_no_bees_and_no_work_are_both_empty():
    assert plan_split([], 8, "contention") == {}
    assert plan_split(UNIFIED, 0, "contention") == {}
    assert expected_makespan(UNIFIED, {}, 120) == 0.0


#### the partition is the decision, and it has to be searched
from agent.device_routing import Partition, best_partition  # noqa: E402

# the measured sweep on this laptop: CPU worker thread count against both rates,
# iGPU alone sustained 44.16 tok/s
SWEEP = [Partition(2, {"igpu": 42.72, "cpu": 4.50}),
         Partition(4, {"igpu": 40.88, "cpu": 6.07}),
         Partition(8, {"igpu": 38.85, "cpu": 9.43}),
         Partition(16, {"igpu": 32.63, "cpu": 10.50})]
SOLO_BEST = 44.16


def test_the_optimum_is_interior():
    """Neither end wins. Giving the CPU worker every thread maximises the CPU and
    minimises the machine, which is the whole reason a partition has to be searched."""
    best, _ = best_partition(SWEEP, SOLO_BEST)
    assert best.knob == 8
    ends = [p for p in SWEEP if p.knob in (2, 16)]
    assert all(best.total > p.total for p in ends)


def test_more_cpu_threads_can_mean_more_cpu_and_less_machine():
    """The trap, stated as a test: 16 threads produce MORE cpu throughput than 8 and a
    lower total, because the iGPU falls further than the CPU rises."""
    at8 = next(p for p in SWEEP if p.knob == 8)
    at16 = next(p for p in SWEEP if p.knob == 16)
    assert at16.rates["cpu"] > at8.rates["cpu"]
    assert at16.rates["igpu"] < at8.rates["igpu"]
    assert at16.total < at8.total


def test_co_scheduling_pays_at_the_right_partition():
    """1.09x at eight threads. The earlier conclusion that it never pays came from
    measuring the sixteen-thread point and nothing else."""
    worth, agg, best = co_scheduling_pays(SWEEP, SOLO_BEST)
    assert worth is True
    assert agg == pytest.approx(48.28, abs=0.01)
    assert agg / best == pytest.approx(1.09, abs=0.01)


def test_one_bad_allocation_does_not_get_to_decide():
    """Handed only the worst point, the flat-rate form says no, correctly about THAT
    allocation, and wrongly about the machine. The sweep is what makes it an answer."""
    worst = [DeviceRate("igpu", "igpu", 44.16, 32.63),
             DeviceRate("cpu", "cpu", 21.41, 10.50)]
    assert co_scheduling_pays(worst)[0] is False
    assert co_scheduling_pays(SWEEP, SOLO_BEST)[0] is True


def test_a_partition_that_loses_everywhere_is_still_declined():
    """Searching must not turn into always saying yes."""
    bad = [Partition(4, {"g": 30.0, "c": 2.0}), Partition(8, {"g": 25.0, "c": 4.0})]
    _best, beats = best_partition(bad, solo_best=44.0)
    assert beats is False
    assert co_scheduling_pays(bad, 44.0)[0] is False


def test_best_partition_handles_nothing_measured():
    assert best_partition([], 10.0) == (None, False)
    assert best_partition([Partition(1, {})], 10.0) == (None, False)


def test_without_a_solo_baseline_it_reports_the_best_it_saw():
    best, beats = best_partition(SWEEP)
    assert best.knob == 8 and beats is True
