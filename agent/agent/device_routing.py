"""Split a batch of work across device-pinned bees, knowing that they contend.

A llama.cpp server binds its device at spawn and cannot be moved, so placing work
"on the CPU" means ROUTING it to a bee that was spawned there. The scheduling question
is therefore how to split N independent generations across bees whose devices share a
bus, and the answer is not obvious: on unified memory the CPU and the iGPU draw on one
pool, so running both costs each of them something.

Measured on this laptop, Qwen2.5-Coder-7B Q4:

    iGPU alone 47.29 tok/s, CPU alone 22.93
    together   iGPU 39.78 (84% of solo), CPU 17.13 (75%), aggregate 56.90

so using both is worth 1.20x over the best single device, against the 1.48x that perfect
additivity would give. The 0.28 difference is the bandwidth war, and it is exactly what
`rexgraph.coordinator` prices.

Three policies are implemented because the naive two are what anyone would write first
and it is worth being able to show what they cost:

    fastest       everything to the quickest bee. Leaves the other device idle.
    round_robin   alternate. Bounded by the SLOWEST bee, so the makespan is set by the
                  worst device rather than the best.
    contention    split in proportion to each bee's rate UNDER CONTENTION, so every bee
                  finishes at the same moment and nothing waits on a straggler.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DeviceRate", "Partition", "plan_split", "expected_makespan",
           "best_partition", "co_scheduling_pays", "POLICIES"]

POLICIES = ("fastest", "round_robin", "contention")


@dataclass
class DeviceRate:
    """One bee's measured throughput, solo and while sharing its bus.

    `contended` is the rate when the other bees on the same host are also busy. It is a
    measurement, not a derivation: how far it falls below `solo` is what the bus topology
    is about, and on a discrete GPU it should barely fall at all.
    """
    name: str
    device: str
    solo: float
    contended: float | None = None

    @property
    def busy(self) -> float:
        """The rate to plan with when every bee is working."""
        return self.contended if self.contended and self.contended > 0 else self.solo


def plan_split(rates, n_tasks: int, policy: str = "contention") -> dict:
    """How many of `n_tasks` each bee should take. Returns {bee_name: count}.

    `contention` shares work in proportion to the CONTENDED rates, which is what makes
    every bee finish together: a bee twice as fast should take twice as many, and using
    the solo rates would over-assign the device that degrades most under sharing.
    """
    rates = [r for r in rates if r.solo > 0]
    if not rates or n_tasks <= 0:
        return {}
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")

    if policy == "fastest":
        best = max(rates, key=lambda r: r.solo)
        return {r.name: (n_tasks if r is best else 0) for r in rates}

    if policy == "round_robin":
        out = {r.name: n_tasks // len(rates) for r in rates}
        for i in range(n_tasks % len(rates)):
            out[rates[i].name] += 1
        return out

    worth, _agg, _best = co_scheduling_pays(rates)
    if not worth:
        # Only correct when the rates given are the BEST partition's. Handed one bad
        # allocation this will fall back to a single device and look decisive about it,
        # which is the mistake this module was written with: search the partition first
        # (best_partition) and pass the winner's rates here.
        best = max(rates, key=lambda r: r.solo)
        return {r.name: (n_tasks if r is best else 0) for r in rates}
    total = sum(r.busy for r in rates)
    out = {r.name: int(n_tasks * r.busy / total) for r in rates}
    # hand the remainder to the fastest, which is where it costs the least
    left = n_tasks - sum(out.values())
    for r in sorted(rates, key=lambda r: -r.busy)[:max(left, 0)]:
        out[r.name] += 1
    return out


def expected_makespan(rates, split: dict, tokens_each: int) -> float:
    """Seconds until the LAST bee finishes, which is what a batch actually costs.

    PIECEWISE, because contention ends. While several bees are working they all run at
    their contended rates; the moment one finishes, the survivors speed up. Charging the
    contended rate for the whole run overestimates badly: it predicted 116.7s for an
    even split that actually took 61.0s, because the iGPU finished early and the CPU ran
    the tail alone at more than twice its contended rate.
    """
    by = {r.name: r for r in rates}
    left = {n: c * tokens_each for n, c in split.items() if c > 0}
    if not left:
        return 0.0
    t = 0.0
    while left:
        shared = len(left) > 1
        rate = {n: (by[n].busy if shared else by[n].solo) for n in left}
        step = min(left[n] / max(rate[n], 1e-9) for n in left)
        t += step
        for n in list(left):
            left[n] -= step * rate[n]
            if left[n] <= 1e-6:
                del left[n]
    return t


@dataclass
class Partition:
    """One allocation of the shared machine, and what each device measured under it.

    `knob` is whatever was varied to get here, the CPU worker's thread count on this
    machine. It is carried rather than assumed so a different machine can partition on a
    different axis without this needing to know which.
    """
    knob: int
    rates: dict          # device name -> tok/s measured UNDER this partition

    @property
    def total(self) -> float:
        return sum(self.rates.values())


def best_partition(samples, solo_best: float | None = None):
    """The allocation with the highest AGGREGATE throughput, and whether it beats solo.

    Returns (partition, beats_solo). The point is that there is an interior optimum and
    it has to be searched for: measured on this laptop, giving the CPU worker 16 threads
    produces MORE cpu throughput (10.50 tok/s against 9.43 at eight) and less total
    (43.13 against 48.28), because the iGPU falls from 38.85 to 32.63. Maximising either
    device alone minimises the machine.

    An earlier version of this file concluded from the 16-thread point alone that
    co-scheduling never pays. It pays 1.09x at the right partition. A scheduler that
    declines after one sample is measuring its own configuration, not the hardware.
    """
    live = [p for p in samples if p.rates]
    if not live:
        return None, False
    best = max(live, key=lambda p: p.total)
    if solo_best is None:
        return best, True
    return best, best.total > solo_best


def co_scheduling_pays(rates_or_samples, solo_best: float | None = None) -> tuple:
    """(worth_it, aggregate, best_solo) for the BEST partition available.

    Given Partitions it searches them; given a flat list of DeviceRate it reads the one
    allocation it was handed, which is only an answer about that allocation. Declining
    should mean "no partition wins", never "the partition I happened to try did not".
    """
    if rates_or_samples and isinstance(rates_or_samples[0], Partition):
        best, beats = best_partition(rates_or_samples, solo_best)
        agg = best.total if best else 0.0
        return (beats if solo_best is not None else True), agg, (solo_best or 0.0)
    live = [r for r in rates_or_samples if r.solo > 0]
    if len(live) < 2:
        return False, sum(r.solo for r in live), max((r.solo for r in live), default=0.0)
    agg = sum(r.busy for r in live)
    best = max(r.solo for r in live)
    return agg > best, agg, best



