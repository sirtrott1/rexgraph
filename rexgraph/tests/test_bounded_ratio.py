"""Checked integer accumulation agrees with Q without object work per incidence."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.core import _exact_ratio as ratio


def run(items, values, seeds, degrees, denominators, *, group=None, **kwargs):
    arrays = [np.asarray(v, dtype=np.int64) for v in (items, values, seeds, degrees, denominators)]
    return ratio.axis_ratio(*arrays, len(denominators), 0,
                            None if group is None else np.asarray(group, dtype=np.int64), **kwargs)


def oracle(items, values, seeds, degrees, denominators, mode, group, count):
    signed, mass = [Q(0)] * len(denominators), [Q(0)] * len(denominators)
    for i, c, v in zip(items, values, seeds, strict=True):
        term = Q(int(c), int(degrees[v]))
        signed[i] += term
        mass[i] += abs(term)
    out = [Q(0)] * count
    for i, (s, u) in enumerate(zip(signed, mass, strict=True)):
        value = s if mode == ratio.SUM else abs(s) if mode == ratio.ABS else u - abs(s)
        if group[i] >= 0:
            out[group[i]] += value / int(denominators[i])
    return out


@pytest.mark.parametrize("mode", [ratio.SUM, ratio.ABS, ratio.COVERAGE])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_small_denominators_use_the_same_exact_contract(mode, grouped, exact):
    rng = np.random.default_rng(176)
    for _ in range(25):
        items, values = rng.integers(17, size=90), rng.integers(-99, 100, size=90)
        seeds, degrees, den = rng.integers(11, size=90), rng.integers(1, 10, size=11), rng.integers(1, 10, size=17)
        owner = rng.integers(-1, 4, size=17) if grouped else np.arange(17)
        count = 4 if grouped else 17
        expected = oracle(items, values, seeds, degrees, den, mode, owner, count)
        got = run(items, values, seeds, degrees, den, mode=mode, exact=exact,
                  group=owner if grouped else None, n_groups=count if grouped else 0)
        assert list(got) == (expected if exact else list(map(float, expected)))


@pytest.mark.parametrize("mode", [ratio.SUM, ratio.ABS, ratio.COVERAGE])
@pytest.mark.parametrize("grouped", [False, True])
def test_bounded_numerical_path_does_not_construct_fractions(monkeypatch, mode, grouped):
    def refuse(*args, **kwargs):
        raise AssertionError("bounded numerical accumulation must stay in compiled integers")
    monkeypatch.setattr(ratio, "Fraction", refuse)
    out = run([0, 0, 1], [3, -1, 7], [0, 1, 0], [3, 7], [5, 11], mode=mode,
              group=[0, 0] if grouped else None, n_groups=1 if grouped else 0)
    expected = oracle([0, 0, 1], [3, -1, 7], [0, 1, 0], [3, 7], [5, 11], mode,
                      [0, 0] if grouped else [0, 1], 1 if grouped else 2)
    assert list(out) == list(map(float, expected))


@pytest.mark.parametrize("mode", [ratio.SUM, ratio.ABS, ratio.COVERAGE])
@pytest.mark.parametrize("exact", [False, True])
def test_overflow_promotes_only_the_affected_item(monkeypatch, mode, exact):
    constructed = []
    def track(n=0, d=None):
        result = Q(n) if d is None else Q(n, d)
        constructed.append(result)
        return result
    monkeypatch.setattr(ratio, "Fraction", track)
    # Ordinary items have two supplied entries each. Only item zero overflows.
    items = [0, 0] + [i for i in range(1, 1001) for _ in range(2)]
    values = [2**63-1, 2**63-1] + [1, -1]*1000
    seeds = [0]*len(items)
    got = run(items, values, seeds, [1], [1]*1001, mode=mode, exact=exact)
    expected = (0 if mode == ratio.COVERAGE else 2*(2**63-1))
    assert got[0] == (Q(expected) if exact else float(expected))
    assert list(got[1:]) == [2 if mode == ratio.COVERAGE else 0]*1000
    assert len(constructed) < (1010 if exact and mode == ratio.COVERAGE else 10)


@pytest.mark.parametrize("mode", [ratio.SUM, ratio.ABS, ratio.COVERAGE])
@pytest.mark.parametrize("exact", [False, True])
def test_integer_limits_products_and_group_overflow(mode, exact):
    fixtures = [
        ([0, 0], [-(2**63), -1], [0, 0], [1], [1], [0]),
        ([0, 0], [2**63-1, 2**63-1], [0, 0], [1], [1], [0]),
        ([0, 0], [2**62, -2**62], [0, 1], [2, 3], [1], [0]),
        ([0, 0], [1, -1], [0, 1], [2**62-1, 2**62], [1], [0]),
        ([0], [1], [0], [2**62], [7], [0]),
        ([0, 1], [2**63-1, 2**63-1], [0, 0], [1], [1, 1], [0, 0]),
        ([0, 1], [1, -1], [0, 0], [1], [2**62-1, 2**62], [0, 0]),
        ([0], [-(2**63)], [0], [1], [1], [0]),
    ]
    for items, values, seeds, degrees, den, owner in fixtures:
        for grouped in (False, True):
            axis = owner if grouped else list(range(len(den)))
            expected = oracle(items, values, seeds, degrees, den, mode, axis, 1 if grouped else len(den))
            got = run(items, values, seeds, degrees, den, mode=mode, exact=exact,
                      group=owner if grouped else None, n_groups=1 if grouped else 0)
            assert list(got) == (expected if exact else list(map(float, expected)))


@pytest.mark.parametrize("sign", [-1, 1])
def test_integer_true_division_preserves_rounding_past_53_bits(sign):
    rng = np.random.default_rng(732)
    for _ in range(200):
        n, d = int(rng.integers(2**53, 2**63-1)), int(rng.integers(1, 2**63-1))
        got = run([0], [sign*n], [0], [d], [1])[0]
        assert got == float(Q(sign*n, d))


@pytest.mark.parametrize("exact", [False, True])
def test_empty_axes_and_omitted_groups(exact):
    assert run([], [], [], [], [], exact=exact).size == 0
    assert list(run([0], [1], [0], [3], [1], group=[-1], n_groups=2, exact=exact)) == [0, 0]


def test_accumulator_byte_product_is_checked_before_allocation():
    with pytest.raises(MemoryError, match="addressable"):
        run([0], [1], [0], [1], [1], group=[-1], n_groups=(2**63-1)//24+1)


def test_unbounded_global_lcm_does_not_promote_bounded_items(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("local ratios fit even when the global LCM does not")
    monkeypatch.setattr(ratio, "Fraction", refuse)
    d = 2**62
    got = run([0, 1], [1, 1], [0, 1], [d-1, d], [1, 1])
    assert list(got) == [float(Q(1, d-1)), float(Q(1, d))]


def test_bounded_common_factor_keeps_unreached_rows_zero(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("zero rows do not need an overflowing denominator product")
    monkeypatch.setattr(ratio, "Fraction", refuse)
    got = run([0], [1], [0], [2**62], [1, 7, 11])
    assert list(got) == [float(Q(1, 2**62)), 0, 0]


@pytest.mark.parametrize("mode, expected", [(ratio.SUM, 0), (ratio.ABS, 0), (ratio.COVERAGE, 1)])
def test_mixed_denominator_cancellation_is_zero_at_every_precision_hint(mode, expected):
    # The old grid emitted 2.5849394142282115e-26 for ABS in the scale benchmark.
    arrays = [np.asarray(v, dtype=np.int64) for v in
              ([0]*5, [-3, 2, 1, 2, 2], list(range(5)), [3, 9, 3, 9, 9], [2])]
    for hint in (0, 32, 64, 96, 126):
        for exact in (False, True):
            got = ratio.axis_ratio(*arrays, 1, hint, mode=mode, exact=exact)
            assert got[0] == expected
