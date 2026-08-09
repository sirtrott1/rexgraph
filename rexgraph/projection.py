"""Projecting a complex into the plane, with every angle an exact rational.

The character is already a vector: `chi(e)` and `phi(v)` are points in the simplex over
the channel hats. Getting from there to a picture normally means a square root for the
length and an arccosine for the angle, which throws the exactness away at the last step,
after the whole tower was built to keep it.

It does not have to. A rational point on the unit circle is

    cos = (1 - t^2)/(1 + t^2),      sin = 2t/(1 + t^2)

for rational `t`, with `cos^2 + sin^2 = 1` exactly. So a direction taken from a rational
parameter is rational in both components, and the angle between two such directions is a
rational spread. Nothing here calls sqrt, sin, cos or atan2.

The construction, slot-wise over the channels:

    (1 + X) (1 - X) = 1 - X^2

with `X` the channel cosines, so each slot returns `1 - cos^2 = sin^2`, which IS the
spread. Four channels give four spreads, and the pairs `(0,3)` and `(1,2)` supply the two
independent parameters the plane needs. Both axes read their parameter through the same
cosine: a sine on one of them puts that axis at its flat maximum exactly where ordinary
characters cluster, and the picture loses a dimension to the map rather than to the
complex.

What this is NOT: a refinement. A Hilbert-space projection into the plane converges on a
picture by successive approximation. Here the tensor field emits the coordinates directly
and the arithmetic stays rational the whole way, so there is nothing to converge and no
tolerance to choose.

What it does not settle: WHICH faces are present. B_1 is topology and is exact, but B_2
is a selection (exponentially many satisfy the chain condition), so a rendering declares
its face rule rather than deriving one.
"""

from __future__ import annotations

from fractions import Fraction

__all__ = ["rational_direction", "direction_spread", "channel_parameters",
           "project", "plane_spread"]


def rational_direction(t) -> tuple:
    """`(cos, sin)` on the unit circle for a rational parameter, exactly.

    The half-angle (Weierstrass) parametrisation. `t = 0` is the positive x axis and
    `t = 1` is a quarter turn; every rational `t` gives a rational point, and every
    rational point on the circle arises from one.
    """
    t = t if isinstance(t, Fraction) else Fraction(t)
    d = 1 + t * t
    return (1 - t * t) / d, 2 * t / d


def direction_spread(t1, t2) -> Fraction:
    """`sin^2` of the angle between two rational directions. Exact.

    This is the `(1 + X)(1 - X) = 1 - X^2` slot applied to the cosine of the difference:
    both directions are unit, so their inner product IS that cosine and the spread is one
    subtraction away.
    """
    c1, s1 = rational_direction(t1)
    c2, s2 = rational_direction(t2)
    ip = c1 * c2 + s1 * s2
    return Fraction(1) - ip * ip


def channel_parameters(shares) -> tuple:
    """Two rational parameters from one cell's channel shares.

    The four channels pair as `(0, 3)` and `(1, 2)`: each pair contributes one direction,
    and the parameter is the ratio inside the pair. A ratio of shares is rational because
    the shares are, so no rounding enters here either.

    A pair whose denominator vanishes contributes the zero parameter, which is the
    positive axis. That is a real position rather than a failure: the channel carrying
    the direction is simply absent.
    """
    vals = [s if isinstance(s, Fraction) else Fraction(s) for s in shares]
    while len(vals) < 4:
        vals.append(Fraction(0))
    u = vals[3] / vals[0] if vals[0] != 0 else Fraction(0)
    v = vals[2] / vals[1] if vals[1] != 0 else Fraction(0)
    return u, v


def project(shares) -> tuple:
    """One cell's plane coordinates, both exact rationals.

    Each channel pair contributes one axis through the cosine of its parameter, so the two
    axes are the same function of their own parameter and respond alike.

    Taking the SINE on the second axis is what this replaced, and it collapsed the picture.
    `sin` is at its maximum at `t = 1`, so its derivative there is zero, and `t = 1` is
    exactly where the parameters sit for an ordinary complex: `v = chi_2 / chi_1` with
    `chi_0 = chi_1` identically, so a cell with no strong channel preference has `v` near
    one and the axis annihilates its variation. Measured on a 4-ary relation with its
    spanning cycle, four vertices with genuinely different characters spread 0.136 in x and
    0.005 in y, an aspect of 1:26 that is an artifact of the map rather than a fact about
    the complex. Through the cosine the same four spread 0.136 by 0.175.
    """
    u, v = channel_parameters(shares)
    cu, _su = rational_direction(u)
    cv, _sv = rational_direction(v)
    return cu, cv


def plane_spread(a, b) -> Fraction | None:
    """The spread between two projected points, exactly.

    The angle a renderer would draw between two cells, as a rational number. None when
    either point is at the origin, where no direction is defined; that is an absence and
    must not be read as zero.
    """
    ax, ay = (Fraction(a[0]), Fraction(a[1]))
    bx, by = (Fraction(b[0]), Fraction(b[1]))
    qa = ax * ax + ay * ay
    qb = bx * bx + by * by
    if qa == 0 or qb == 0:
        return None
    ip = ax * bx + ay * by
    return Fraction(1) - (ip * ip) / (qa * qb)


def project_complex(rex, *, grade: str = "vertex") -> dict:
    """Every cell of one grade placed in the plane, exactly.

    Runs on the character carried rationally from the boundary operators
    (`rational_trig.exact_character`), not on the stored float64: reading the stored
    value would give the exact projection of a double rather than the projection, which
    is the same distinction `exact_spread` turns on.

    Returns rational coordinates as strings alongside floats, because a renderer has to
    put a number in a path and the float should be visibly a final rounding of an exact
    value rather than an accumulated approximation.
    """
    from rexgraph.rational_trig import exact_character, exact_star_character

    if grade not in ("vertex", "edge"):
        raise ValueError(f"grade must be 'vertex' or 'edge', got {grade!r}")
    rows, names = (exact_star_character(rex) if grade == "vertex"
                   else exact_character(rex))
    if rows is None:
        return {"grade": grade, "channels": [], "cells": [], "exact": False,
                "note": "no exact character is available for this complex"}

    cells = []
    for i, shares in enumerate(rows):
        u, v = channel_parameters(shares)
        x, y = project(shares)
        cells.append({
            "index": i,
            "u": str(u), "v": str(v),
            "x": str(x), "y": str(y),
            "at": [float(x), float(y)],
        })
    return {"grade": grade, "channels": list(names), "cells": cells, "exact": True,
            "note": ("coordinates are rational; the floats in `at` are a final rounding "
                     "of the exact values in `x` and `y`")}
