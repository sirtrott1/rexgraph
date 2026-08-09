"""The complex projected into the plane with every angle an exact rational.

A picture normally costs a square root for the length and an arccosine for the angle,
which discards the exactness at the last step after the whole tower was built to keep it.
A rational point on the unit circle removes that: cos = (1-t^2)/(1+t^2) and
sin = 2t/(1+t^2) are both rational and satisfy cos^2 + sin^2 = 1 exactly, so a direction
from a rational parameter is rational in both components.

The construction is the slot-wise difference of squares: (1 + X)(1 - X) = 1 - X^2, and
with X a cosine that is 1 - cos^2 = sin^2, which IS the spread. These pin that identity,
the exactness of the coordinates, and that the angle between two rendered cells is a
rational number rather than a float.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.projection import (
    channel_parameters,
    direction_spread,
    plane_spread,
    project,
    project_complex,
    rational_direction,
)

#### rational directions


@pytest.mark.parametrize("t", [0, Fraction(1, 3), 1, 2, Fraction(3, 4),
                               Fraction(5, 12), Fraction(-2, 7)])
def test_a_rational_parameter_gives_a_rational_point_on_the_circle(t):
    c, s = rational_direction(t)
    assert isinstance(c, Fraction) and isinstance(s, Fraction)
    assert c * c + s * s == 1, "the point is not exactly on the unit circle"


def test_zero_is_the_positive_axis_and_one_is_a_quarter_turn():
    assert rational_direction(0) == (1, 0)
    assert rational_direction(1) == (0, 1)


#### the slot identity


@pytest.mark.parametrize("t1,t2", [(0, 1), (Fraction(1, 3), Fraction(3, 4)),
                                   (Fraction(1, 2), Fraction(1, 2)), (2, Fraction(1, 5))])
def test_one_plus_x_times_one_minus_x_is_the_spread(t1, t2):
    """(1 + X)(1 - X) = 1 - X^2, and with X the cosine of the difference that is
    sin^2: the slot returns the spread directly."""
    c1, s1 = rational_direction(t1)
    c2, s2 = rational_direction(t2)
    ip = c1 * c2 + s1 * s2
    assert (1 + ip) * (1 - ip) == direction_spread(t1, t2)


def test_a_direction_has_no_spread_against_itself():
    assert direction_spread(Fraction(2, 5), Fraction(2, 5)) == 0


def test_a_quarter_turn_is_full_spread():
    assert direction_spread(0, 1) == 1


@pytest.mark.parametrize("t1,t2", [(0, Fraction(1, 3)), (1, 2), (Fraction(3, 7), 4)])
def test_every_spread_is_rational_and_in_range(t1, t2):
    s = direction_spread(t1, t2)
    assert isinstance(s, Fraction)
    assert 0 <= s <= 1


#### projecting a complex


def _path():
    return RexGraph(sources=np.array([0, 1, 2, 3, 4], dtype=np.int32),
                    targets=np.array([1, 2, 3, 4, 5], dtype=np.int32))


def _branching():
    return RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2], dtype=np.int32))


@pytest.mark.parametrize("build", [_path, _branching], ids=["path", "branching"])
def test_every_coordinate_is_an_exact_rational(build):
    """Nothing here called sqrt, sin, cos or atan2, so every coordinate is a Fraction
    and the float beside it is a final rounding rather than an accumulated error."""
    out = project_complex(build(), grade="vertex")
    assert out["exact"] is True
    for cell in out["cells"]:
        assert isinstance(Fraction(cell["x"]), Fraction)
        assert isinstance(Fraction(cell["y"]), Fraction)
        assert float(Fraction(cell["x"])) == pytest.approx(cell["at"][0])


def test_the_angle_between_two_rendered_cells_is_rational():
    out = project_complex(_path(), grade="vertex")
    a = (Fraction(out["cells"][1]["x"]), Fraction(out["cells"][1]["y"]))
    b = (Fraction(out["cells"][2]["x"]), Fraction(out["cells"][2]["y"]))
    s = plane_spread(a, b)
    assert isinstance(s, Fraction)
    assert 0 <= s <= 1


def test_relations_can_be_projected_too():
    out = project_complex(_path(), grade="edge")
    assert len(out["cells"]) == _path().nE


def test_an_unknown_grade_is_refused():
    with pytest.raises(ValueError, match="grade must be"):
        project_complex(_path(), grade="faces")


def test_a_point_at_the_origin_has_no_direction():
    """An absence, which must not be read as a zero angle."""
    assert plane_spread((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))) is None


#### the projection is not degenerate


def test_a_branching_complex_leaves_the_unit_circle():
    """The two channel pairs give independent parameters, so a cell whose channels are
    asymmetric is placed off the circle rather than only around it."""
    out = project_complex(_branching(), grade="vertex")
    off = [c for c in out["cells"]
           if Fraction(c["x"]) ** 2 + Fraction(c["y"]) ** 2 != 1]
    assert off, "every cell landed on the circle: the second parameter is not doing work"


def test_structurally_identical_cells_land_together():
    """Two vertices with the same star character are the same cell to this projection,
    which is what makes position mean something."""
    out = project_complex(_path(), grade="vertex")
    at = [(c["x"], c["y"]) for c in out["cells"]]
    assert len(set(at)) < len(at)


def test_the_parameters_come_from_the_channel_pairs():
    u, v = channel_parameters([Fraction(1, 4)] * 4)
    assert u == 1 and v == 1                     # equal shares: both ratios are one


def test_a_missing_channel_gives_the_positive_axis():
    """A pair whose denominator vanishes contributes the zero parameter. That is a real
    position (the channel carrying the direction is absent), not a failure."""
    u, v = channel_parameters([Fraction(0), Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)])
    assert u == 0
    assert project([Fraction(0), Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)])[0] == 1
