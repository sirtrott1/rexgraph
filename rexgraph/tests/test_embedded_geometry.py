"""Geometry emerges from an EMBEDDING, so a file that carries one carries the geometry.

`relation_quadrance` is the intrinsic length, `1 + 1/(k-1)`, a function of arity alone.
It is the same for benzene flat and benzene puckered, because it is not about where the
atoms are. `embed` applies the boundary column to an embedding instead, which gives one
vector per relation at any arity:

    k = 2   c = (-1, +1)             ->  p_b - p_a, the edge vector
    k > 2   c = (-1, 1/(k-1), ...)   ->  (mean of the others) - p_distinguished

and the quadrance and spread of those are the real lengths and angles. Exact whenever the
coordinates are: an SDF writes four decimal places, so every one is a Fraction over 10^4
and a bond angle comes back as a rational rather than an arccosine.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.geometry import embed, embedded_geometry_of, geometry_of
from rexgraph.graph import RexGraph


def _corner():
    """Two relations meeting at the origin, along the axes."""
    rex = RexGraph(sources=np.array([0, 0], dtype=np.int32),
                   targets=np.array([1, 2], dtype=np.int32))
    rex._ensure_clean()
    points = [[Fraction(0), Fraction(0)], [Fraction(1), Fraction(0)],
              [Fraction(0), Fraction(1)]]
    return rex, points


def test_a_two_ary_relation_embeds_as_its_edge_vector():
    rex, points = _corner()
    assert embed(rex, points) == [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]


def test_perpendicular_relations_have_full_spread():
    rex, points = _corner()
    out = embedded_geometry_of(rex, points)
    assert out["exact"] is True
    assert Fraction(out["meeting"][0]["spread"]) == 1
    assert Fraction(out["meeting"][0]["cos_squared"]) == 0


def test_the_embedded_length_is_the_real_one():
    rex, points = _corner()
    assert [Fraction(q) for q in embedded_geometry_of(rex, points)["quadrance"]] == [1, 1]


def test_it_is_a_different_reading_from_the_intrinsic_one():
    """The intrinsic quadrance of a 2-ary relation is 2 whatever the embedding, because
    it is `1 + 1/(k-1)` and says nothing about where the atoms sit."""
    rex, points = _corner()
    assert geometry_of(rex, exact=True)["quadrance"] == ["2", "2"]
    assert embedded_geometry_of(rex, points)["quadrance"] == ["1", "1"]


def test_moving_a_vertex_moves_the_embedded_reading_only():
    rex, points = _corner()
    stretched = [points[0], [Fraction(3), Fraction(0)], points[2]]
    assert (geometry_of(rex, exact=True)["quadrance"]
            == geometry_of(rex, exact=True)["quadrance"])
    assert (embedded_geometry_of(rex, points)["quadrance"]
            != embedded_geometry_of(rex, stretched)["quadrance"])


@pytest.mark.parametrize("k", [3, 4, 6])
def test_a_wide_relation_gets_one_vector_not_k(k):
    """Which is what lets it have a length at all without being split into pairs."""
    rex = RexGraph.from_hypergraph(np.array([0, k], dtype=np.int32),
                                   np.array(list(range(k)), dtype=np.int32))
    rex._ensure_clean()
    points = [[Fraction(i), Fraction(0)] for i in range(k)]
    vectors = embed(rex, points)
    assert len(vectors) == 1
    assert len(vectors[0]) == 2


def test_a_wide_relation_embeds_as_the_offset_from_its_distinguished_vertex():
    """(-1, 1/(k-1), ...) applied to the points is (mean of the others) minus the
    distinguished one, which reduces to the edge vector at k=2."""
    rex = RexGraph.from_hypergraph(np.array([0, 3], dtype=np.int32),
                                   np.array([0, 1, 2], dtype=np.int32))
    rex._ensure_clean()
    points = [[Fraction(0)], [Fraction(2)], [Fraction(4)]]
    assert embed(rex, points) == [[Fraction(3)]]


def test_the_angle_is_rational_not_an_arccosine():
    """A 3-4-5 corner: cos^2 = 9/25 exactly, and no sqrt was taken to get it."""
    rex = RexGraph(sources=np.array([0, 0], dtype=np.int32),
                   targets=np.array([1, 2], dtype=np.int32))
    rex._ensure_clean()
    points = [[Fraction(0), Fraction(0)], [Fraction(5), Fraction(0)],
              [Fraction(3), Fraction(4)]]
    out = embedded_geometry_of(rex, points)
    assert Fraction(out["meeting"][0]["cos_squared"]) == Fraction(9, 25)


def test_relations_that_do_not_meet_have_no_angle():
    rex = RexGraph(sources=np.array([0, 2], dtype=np.int32),
                   targets=np.array([1, 3], dtype=np.int32))
    rex._ensure_clean()
    points = [[Fraction(i), Fraction(0)] for i in range(4)]
    assert embedded_geometry_of(rex, points)["meeting"] == []


def test_floats_are_carried_as_the_doubles_they_are():
    rex, points = _corner()
    out = embedded_geometry_of(rex, [[float(x) for x in p] for p in points], exact=False)
    assert out["exact"] is False
    assert out["quadrance"] == pytest.approx([1.0, 1.0])
