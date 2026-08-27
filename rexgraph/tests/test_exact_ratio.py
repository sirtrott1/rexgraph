"""Exact rational reading over factored denominators, rendered as the nearest double."""
from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np
import pytest

_ex = pytest.importorskip("rexgraph.core._exact_ratio")


def _run(rows, carried, seed, deg, den, n, frac=None, group=None, n_groups=0,
         mode=0):
    rows = np.asarray(rows, np.int64)
    carried = np.asarray(carried, np.int64)
    if frac is None:
        frac = _ex.frac_bits_for(int(np.abs(carried).max(initial=1)) * max(len(rows), 1),
                                 len(deg), n)
    g = None if group is None else np.asarray(group, np.int64)
    return _ex.axis_ratio(rows, carried, np.asarray(seed, np.int64),
                          np.asarray(deg, np.int64), np.asarray(den, np.int64),
                          n, int(frac), g, int(n_groups), int(mode))


def _one(a, deg, den):
    return _run([0], [a], [0], [deg], [den], 1)[0]


def test_a_ratio_a_double_holds_exactly_comes_back_exactly():
    assert _one(15, 2, 1) == 7.5
    assert _one(1, 4, 1) == 0.25
    assert _one(1, 1, 1) == 1.0
    assert _one(3, 2, 2) == 0.75


def test_a_ratio_a_double_cannot_hold_comes_back_correctly_rounded():
    """One rounding is allowed and it has to go the right way. 1/10**9 was a ulp low
    before the division carried a sticky bit: its leading 54 quotient bits look like a
    tie and the remainder past them is what says it is not."""
    for d in (3, 7, 10 ** 9, 10 ** 15, 2 ** 40 - 1):
        assert _one(1, d, 1) == float(Fraction(1, d)), d
        assert _one(1, 1, d) == float(Fraction(1, d)), d


def test_the_two_denominators_are_never_multiplied_together():
    """The reason the kernel has no width bound. Forty seeds each with an 18 bit degree
    puts their common multiple past seven hundred bits; dividing the axes separately
    keeps every intermediate inside 128."""
    rng = random.Random(3)
    s = 40
    deg = [rng.getrandbits(18) + 1 for _ in range(s)]
    carried = [rng.getrandbits(24) + 1 for _ in range(s)]
    den = [rng.getrandbits(24) + 1]
    got = _run([0] * s, carried, list(range(s)), deg, den, 1)[0]
    want = sum(Fraction(carried[v], deg[v]) for v in range(s)) / den[0]
    lcm = 1
    for d in deg:
        # math.gcd, not np.gcd: np.gcd returns int64 and promotes the whole expression,
        # so the lcm wraps silently past 2**63. The face solver had the same bug.
        lcm = lcm * d // math.gcd(lcm, d)
    assert lcm.bit_length() > 400, "the common multiple really is out of range"
    assert got == float(want)


def test_contributions_accumulate_before_the_division_not_after():
    """Summing the doubles would round once per contribution. The kernel sums the
    integers per seed and rounds once, which is the whole reason it exists."""
    got = _run([0, 0, 0], [7, 11, 13], [0, 1, 0], [3, 5], [9], 1)[0]
    want = (Fraction(7 + 13, 3) + Fraction(11, 5)) / 9
    assert got == float(want)


def test_a_row_nothing_reaches_is_exactly_zero():
    out = _run([0], [5], [0], [2], [3, 3, 3], 3)
    assert out[1] == 0.0 and out[2] == 0.0
    assert out[0] == float(Fraction(5, 2 * 3))


def test_against_the_rationals_over_random_shapes():
    """The property stated directly: whatever the shape, the double returned is the
    double nearest the exact rational."""
    rng = random.Random(7)
    for _ in range(150):
        n, s, m = 12, rng.randint(1, 40), 90
        deg = [rng.getrandbits(rng.choice([4, 12, 18])) + 1 for _ in range(s)]
        den = [rng.getrandbits(24) + 1 for _ in range(n)]
        rows = [rng.randrange(n) for _ in range(m)]
        seed = [rng.randrange(s) for _ in range(m)]
        carried = [rng.getrandbits(24) + 1 for _ in range(m)]
        got = _run(rows, carried, seed, deg, den, n)
        acc = [[0] * s for _ in range(n)]
        for i in range(m):
            acc[rows[i]][seed[i]] += carried[i]
        for r in range(n):
            want = sum(Fraction(acc[r][v], deg[v]) for v in range(s)) / den[r]
            assert got[r] == float(want), (r, s)


def test_the_scaling_leaves_room_for_the_sum():
    """`frac_bits_for` is what keeps the accumulation inside 128 bits, so it has to
    shrink as either the contributions or the seed count grow."""
    wide = _ex.frac_bits_for(1, 1)
    assert wide > 100
    assert _ex.frac_bits_for(1 << 30, 1) < wide
    assert _ex.frac_bits_for(1, 1 << 10) < wide
    assert _ex.frac_bits_for(1 << 60, 1 << 60) >= 0


def test_a_signed_contribution_cancels_where_it_should():
    """A boundary entry at position 0 carries the opposite sign to the arguments, so a
    column whose support is seeded evenly sums to zero. SUM says so and ABS agrees."""
    # -2 at the head against +1 twice: the column is zero-sum at k=3
    got = _run([0, 0, 0], [-2, 1, 1], [0, 0, 0], [1], [1], 1, mode=_ex.SUM)
    assert got[0] == 0.0
    assert _run([0, 0, 0], [-2, 1, 1], [0, 0, 0], [1], [1], 1, mode=_ex.ABS)[0] == 0.0


def test_coverage_is_what_a_zero_sum_column_leaves_behind():
    """The signed reading cancels; the unsigned one does not. COVERAGE is the gap, and
    it is the whole magnitude when the cancellation is total."""
    got = _run([0, 0, 0], [-2, 1, 1], [0, 0, 0], [1], [1], 1, mode=_ex.COVERAGE)
    assert got[0] == float(Fraction(4, 1))          # |−2|+1+1 with nothing left signed


def test_grouping_sums_items_before_the_rounding():
    """A group's value is its items summed in fixed point, so the group rounds once
    rather than once per item."""
    item = [0, 1, 2]
    carried = [1, 1, 1]
    seed = [0, 0, 0]
    deg = [3]
    den = [7, 11, 13]
    group = [0, 0, 1]
    got = _run(item, carried, seed, deg, den, 3, group=group, n_groups=2)
    want0 = Fraction(1, 3 * 7) + Fraction(1, 3 * 11)
    want1 = Fraction(1, 3 * 13)
    assert got[0] == float(want0) and got[1] == float(want1)


def test_every_mode_matches_the_rationals_under_grouping():
    rng = random.Random(23)
    for _ in range(80):
        n, s, m, ng = 14, rng.randint(1, 20), 70, 4
        deg = [rng.getrandbits(rng.choice([4, 12, 18])) + 1 for _ in range(s)]
        den = [rng.getrandbits(20) + 1 for _ in range(n)]
        item = [rng.randrange(n) for _ in range(m)]
        seed = [rng.randrange(s) for _ in range(m)]
        carried = [rng.choice([1, -1]) * (rng.getrandbits(20) + 1) for _ in range(m)]
        group = [rng.randrange(ng) for _ in range(n)]
        a = [[0] * s for _ in range(n)]
        au = [[0] * s for _ in range(n)]
        for i in range(m):
            a[item[i]][seed[i]] += carried[i]
            au[item[i]][seed[i]] += abs(carried[i])
        for mode in (_ex.SUM, _ex.ABS, _ex.COVERAGE):
            got = _run(item, carried, seed, deg, den, n, group=group, n_groups=ng,
                       mode=mode, frac=_ex.frac_bits_for(
                           max(abs(c) for c in carried) * m, s, n))
            tot = [Fraction(0)] * ng
            for r in range(n):
                S = sum(Fraction(a[r][v], deg[v]) for v in range(s)) / den[r]
                M = sum(Fraction(au[r][v], deg[v]) for v in range(s)) / den[r]
                tot[group[r]] += S if mode == _ex.SUM else (
                    abs(S) if mode == _ex.ABS else M - abs(S))
            for g in range(ng):
                assert got[g] == float(tot[g]), (mode, g)
