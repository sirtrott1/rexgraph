"""Lengths and angles of a complex, without a square root or an arccosine.

Rendering needs lengths and angles. Taken the usual way those are sqrt and arccos, both
transcendental and neither a reading of the boundary tensor. Quadrance and spread are
the same geometry one step earlier, where it is still rational: Q = <v,v> is the squared
length and s = 1 - <u,v>^2/(Q_u Q_v) is the squared sine, so cos^2 = 1 - s exactly.

The part worth pinning hardest is the SOURCE. The share 1/(k-1) is not binary-exact for
most arities, so reading the assembled float B1 and calling the result exact returns the
exact value of a double instead of the value. These columns are rebuilt from the
boundary structure.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.geometry import (
    cos_squared,
    geometry_of,
    relation_quadrance,
    relation_spread,
    spreads_at,
)
from rexgraph.graph import RexGraph


def _kary(k: int) -> RexGraph:
    return RexGraph.from_hypergraph(np.array([0, k], dtype=np.int32),
                                    np.arange(k, dtype=np.int32))


def _triangle() -> RexGraph:
    return RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                    targets=np.array([1, 2, 0], dtype=np.int32))


#### length carries arity


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7, 9, 12])
def test_the_quadrance_of_a_relation_is_one_plus_one_over_k_minus_one(k):
    """A relation's squared length IS its boundary concentration, so arity is legible
    from the geometry rather than being a separate attribute."""
    assert relation_quadrance(_kary(k), 0) == Fraction(1) + Fraction(1, k - 1)


@pytest.mark.parametrize("k", [4, 6, 7, 9, 12])
def test_exactness_survives_an_arity_whose_share_is_not_binary(k):
    """The failure this guards: 1/3 and 1/5 have no exact double, so converting the
    STORED column to a Fraction gives the exact value of a double. At k=4 that route
    returns 432691404877902290367942354447019/324518553658426726783156020576256."""
    q = relation_quadrance(_kary(k), 0)
    assert isinstance(q, Fraction)
    assert q.denominator == k - 1
    assert q.numerator == k


def test_a_pairwise_relation_is_the_k_equals_two_case():
    assert relation_quadrance(_kary(2), 0) == 2


def test_quadrance_falls_toward_one_as_a_relation_widens():
    """Concentration: a 2-ary relation is maximally concentrated, a wide one diffuse."""
    values = [relation_quadrance(_kary(k), 0) for k in (2, 3, 4, 8, 16)]
    assert values == sorted(values, reverse=True)
    assert values[-1] > 1


#### angle, as a squared sine


def test_the_spread_of_a_triangles_relations_is_three_quarters():
    """Two boundary columns meeting at a vertex sit at 120 degrees, so sin^2 = 3/4 and
    cos^2 = 1/4, both rational."""
    t = _triangle()
    s = relation_spread(t, 0, 2)
    assert s == Fraction(3, 4)
    assert cos_squared(s) == Fraction(1, 4)


def test_spread_and_cos_squared_are_complementary():
    t = _triangle()
    s = relation_spread(t, 0, 2)
    assert s + cos_squared(s) == 1


def test_a_relation_has_no_spread_against_itself():
    """Parallel, so the squared sine is zero."""
    assert relation_spread(_triangle(), 1, 1) == 0


def test_disjoint_relations_are_perpendicular():
    """Sharing no vertex, the inner product vanishes and the spread is one."""
    g = RexGraph(sources=np.array([0, 2], dtype=np.int32),
                 targets=np.array([1, 3], dtype=np.int32))
    assert relation_spread(g, 0, 1) == 1


def test_a_branching_relation_has_a_rational_spread_against_a_pairwise_one():
    g = RexGraph.from_hypergraph(np.array([0, 4, 6], dtype=np.int32),
                                 np.array([0, 1, 2, 3, 0, 1], dtype=np.int32))
    s = relation_spread(g, 0, 1)
    assert isinstance(s, Fraction)
    assert 0 <= s <= 1


def test_nothing_here_returns_an_irrational():
    """The whole point: every value is a Fraction, so no sqrt, sin, cos or atan2 was
    involved in producing it."""
    g = RexGraph.from_hypergraph(np.array([0, 4, 6, 8], dtype=np.int32),
                                 np.array([0, 1, 2, 3, 0, 1, 1, 2], dtype=np.int32))
    for e in range(g.nE):
        assert isinstance(relation_quadrance(g, e), Fraction)
    for a in range(g.nE):
        for b in range(a + 1, g.nE):
            s = relation_spread(g, a, b)
            assert s is None or isinstance(s, Fraction)


#### the star, which is what a renderer places


def test_the_spreads_at_a_vertex_cover_its_incident_relations():
    t = _triangle()
    at_two = spreads_at(t, 2)
    assert len(at_two) == 1                      # two relations meet there: one pair
    assert set(at_two[0]["relations"]) == {1, 2}


def test_a_vertex_with_one_relation_has_no_angle():
    g = RexGraph(sources=np.array([0], dtype=np.int32),
                 targets=np.array([1], dtype=np.int32))
    assert spreads_at(g, 0) == []


#### the assembled reading


def test_geometry_of_reports_only_relations_that_meet():
    g = RexGraph(sources=np.array([0, 2], dtype=np.int32),
                 targets=np.array([1, 3], dtype=np.int32))
    assert geometry_of(g)["meeting"] == []


def test_geometry_of_can_hand_a_renderer_floats():
    """A float is a final rounding of a rational, not an accumulated approximation."""
    out = geometry_of(_triangle(), exact=False)
    assert out["exact"] is False
    assert all(isinstance(q, float) for q in out["quadrance"])
    assert out["meeting"][0]["spread"] == pytest.approx(0.75)


def test_the_exact_form_serialises_as_the_fraction_it_is():
    out = geometry_of(_triangle(), exact=True)
    assert out["quadrance"][0] == "2"
    assert out["meeting"][0]["spread"] == "3/4"
