"""The exact character must agree with the float one, at every arity.

They are two readings of the same operators, so a disagreement is a defect in one of
them by definition. The exact tower was reading the float channels back and rounding each
diagonal to an integer, which is a no-op on a pairwise complex (T = 2, F and C integers)
and destructive on any branching one:

    T[e,e] = 1 + 1/(k-1)   is in (1, 3/2] for every k >= 3, so rounding maps EVERY
                           branching arity to 1 and erases the arity signal entirely
    F[e,e]                 built from shares like 1/(k-1); a 1/2 rounds to 0 and the
                           whole orientation channel can vanish

The second one showed as a rendering collapse: with F identically zero the second channel
parameter is zero for every cell, so every vertex lands on y = 0 and the picture is a
line. These hold the agreement, the non-integrality, and the consequence.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.rational_trig import (
    exact_channel_diagonals,
    exact_character,
    exact_star_character,
)

CASES = {
    "pairwise triangle": ([0, 2, 4, 6], [0, 1, 1, 2, 2, 0]),
    "pairwise path": ([0, 2, 4, 6], [0, 1, 1, 2, 2, 3]),
    "3-ary + legs": ([0, 3, 5, 7], [0, 1, 2, 0, 1, 1, 3]),
    "4-ary + legs": ([0, 4, 6, 8, 10], [0, 1, 2, 3, 0, 1, 1, 2, 2, 4]),
    "5-ary + legs": ([0, 5, 7, 9], [0, 1, 2, 3, 4, 0, 1, 4, 5]),
    "two wide relations": ([0, 4, 8, 10], [0, 1, 2, 3, 0, 1, 4, 3, 4, 5]),
}


def _weighted(offsets, vertices, weights):
    """The primary factory carries its metric on the declared C1 basis."""
    rex = RexGraph.from_hypergraph(
        np.array(offsets, dtype=np.int32), np.array(vertices, dtype=np.int32),
        w_E=np.array(weights, dtype=float),
    )
    rex._ensure_clean()
    return rex


def _rex(offsets, vertices):
    rex = RexGraph.from_hypergraph(np.array(offsets, dtype=np.int32),
                                   np.array(vertices, dtype=np.int32))
    rex._ensure_clean()
    return rex


@pytest.mark.parametrize("tag", list(CASES))
def test_the_exact_character_agrees_with_the_float_one(tag):
    rex = _rex(*CASES[tag])
    rows, _names = exact_character(rex)
    exact = np.array([[float(x) for x in row] for row in rows])
    assert exact == pytest.approx(np.asarray(rex.structural_character), abs=1e-12)


@pytest.mark.parametrize("tag", list(CASES))
def test_the_exact_star_character_agrees_too(tag):
    rex = _rex(*CASES[tag])
    rows, _names = exact_star_character(rex)
    exact = np.array([[float(x) for x in row] for row in rows])
    assert exact == pytest.approx(np.asarray(rex.star_character), abs=1e-12)


def test_the_diagonals_are_not_integers_once_a_relation_branches():
    """The premise the rounding rested on. True at k=2 and false at every k above it."""
    diagonals, names = exact_channel_diagonals(_rex(*CASES["5-ary + legs"]))
    assert diagonals["L1_down"][0] == Fraction(5, 4), "T = 1 + 1/(k-1) at k=5"
    assert any(d.denominator != 1 for n in names for d in diagonals[n])


def test_the_pairwise_diagonals_are_integers():
    """Which is why rounding was invisible: on a 2-ary complex it changes nothing."""
    diagonals, names = exact_channel_diagonals(_rex(*CASES["pairwise triangle"]))
    assert all(d.denominator == 1 for n in names for d in diagonals[n])


@pytest.mark.parametrize("k,expected", [(2, "2"), (3, "3/2"), (4, "4/3"),
                                        (5, "5/4"), (9, "9/8")])
def test_arity_survives_in_the_boundary_norm(k, expected):
    """T = 1 + 1/(k-1) is how arity reaches the character. Every value for k >= 3 sits in
    (1, 3/2], so rounding collapsed them all onto 1 and made a 4-ary relation
    indistinguishable from a 100-ary one."""
    rex = _rex([0, k], list(range(k)))
    diagonals, _names = exact_channel_diagonals(rex)
    assert diagonals["L1_down"][0] == Fraction(expected)


def test_the_orientation_channel_is_not_zeroed():
    """F on the 5-ary case is [1/2, 0, 1/2]: the middle leg genuinely carries nothing and
    the two ends genuinely do. Rounding flattened all three to zero while leaving the
    trace nonzero, so every cell read 0/1 and the channel silently died."""
    diagonals, _names = exact_channel_diagonals(_rex(*CASES["5-ary + legs"]))
    assert diagonals["L_SG"] == [Fraction(1, 2), Fraction(0), Fraction(1, 2)]


def test_the_layout_does_not_collapse_onto_an_axis():
    """The consequence, at the far end. With F identically zero the second channel
    parameter vanishes for every cell and the whole complex draws on y = 0."""
    from rexgraph.projection import project_complex

    cells = project_complex(_rex(*CASES["5-ary + legs"]), grade="vertex")["cells"]
    assert len({c["y"] for c in cells}) > 1, "every vertex landed on one horizontal"
    assert any(Fraction(c["y"]) != 0 for c in cells)


def test_a_complex_that_cannot_be_rational_says_so():
    """The normalized G channel takes a square root, so there is no rational character
    for it. That is reported as an absence rather than approximated."""
    rex = _rex(*CASES["4-ary + legs"])
    assert rex.g_channel == "raw", "the rational path expects the raw Gramian"
    rex._g_channel = "normalized"
    diagonals, names = exact_channel_diagonals(rex)
    assert diagonals is None and names == []


#### the metric


@pytest.mark.parametrize("tag,weights", [
    ("pairwise triangle", [5.0, 1.0, 1.0]),
    ("4-ary + legs", [2.0, 3.0, 1.0, 5.0]),
    ("5-ary + legs", [7.0, 1.0, 2.0]),
])
def test_the_exact_character_agrees_with_the_float_one_under_weighting(tag, weights):
    offsets, vertices = CASES[tag]
    rex = _weighted(offsets, vertices, weights)
    rows, _names = exact_character(rex)
    exact = np.array([[float(x) for x in row] for row in rows])
    assert exact == pytest.approx(np.asarray(rex.structural_character), abs=1e-12)


def test_the_two_diagonals_coincide_under_weighting():
    """diag(T) = diag(G) is the identity F is defined by: squaring kills the sign, so all
    of B1's sign content lives off-diagonal. G is T's unsigned twin and carries the same
    per-relation metric. Leaving G unweighted broke this at every w != 1."""
    rex = _weighted(*CASES["pairwise triangle"], [5.0, 1.0, 1.0])
    diagonals, _names = exact_channel_diagonals(rex)
    assert diagonals["L1_down"] == diagonals["L_O"]
    assert diagonals["L1_down"] == [Fraction(50), Fraction(2), Fraction(2)]


def test_coparticipation_stays_unweighted():
    """C is deliberately not scaled: co-participation is a topological fact about which
    relations meet, not a geometric one about how far apart they are."""
    plain = exact_channel_diagonals(_rex(*CASES["pairwise triangle"]))[0]
    heavy = exact_channel_diagonals(
        _weighted(*CASES["pairwise triangle"], [5.0, 1.0, 1.0]))[0]
    assert plain["L_C"] == heavy["L_C"]
