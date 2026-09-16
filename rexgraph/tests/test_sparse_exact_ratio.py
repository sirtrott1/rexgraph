"""Sparse ratio coordinates and exact cancellation beyond fixed precision."""
from fractions import Fraction as Q
import numpy as np
import pytest
from rexgraph.core import _exact_ratio as ratio


def run(items, values, seeds, degrees, denominators, **kwargs):
    arrays = [np.asarray(v, dtype=np.int64) for v in (items, values, seeds, degrees, denominators)]
    return ratio.axis_ratio(*arrays, len(denominators), 0, **kwargs)


@pytest.mark.parametrize("exact", [False, True])
def test_nonzero_cancellation_cannot_fall_below_a_fixed_precision(exact):
    d = 2**62
    expected = Q(1, d-1) - Q(1, d)
    got = run([0, 0], [1, -1], [0, 1], [d-1, d], [1], exact=exact)
    assert got[0] == (expected if exact else float(expected)) and got[0] != 0


def test_sparse_coordinates_do_not_allocate_the_cartesian_axis():
    # The old product needs 6.4 GB for these axes despite just two incidences.
    n = 20000
    got = run([0, n-1], [1, 3], [n-1, 0], [1]*n, [1]*n, exact=True)
    assert got[0] == 1 and got[-1] == 3 and sum(got) == 4


@pytest.mark.parametrize("mode", [ratio.SUM, ratio.ABS, ratio.COVERAGE])
def test_repeated_coordinates_and_grouped_cancellation_are_exact(mode):
    values = [2**63-1, -(2**63), 3, -4]
    got = run([0, 0, 1, 1], values, [0, 0, 0, 1], [3, 7], [2, 5],
              group=np.array([0, 0], dtype=np.int64), n_groups=1, mode=mode, exact=True)[0]
    a, b = -Q(1, 3), Q(1)-Q(4, 7)
    expected = (a/2+b/5 if mode == ratio.SUM else abs(a)/2+abs(b)/5 if mode == ratio.ABS
                else (Q(2**64-1, 3)-abs(a))/2 + (1+Q(4, 7)-abs(b))/5)
    assert got == expected


@pytest.mark.parametrize("args", [([1], [1], [0], [1], [1]),
    ([0], [1], [-1], [1], [1]), ([0], [1], [1], [1], [1]),
    ([0], [1], [0], [0], [1]), ([0], [1], [0], [1], [0]),
    ([0], [], [0], [1], [1])])
def test_bad_axes_and_denominators_are_refused(args):
    with pytest.raises(ValueError):
        run(*args)
