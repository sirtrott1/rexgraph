"""Exact geometry of a relational complex: lengths and angles without irrationals.

Rendering a complex needs lengths and angles. Taken the usual way those are a square
root and an arccosine, both transcendental, both approximate, and neither a reading of
the boundary tensor. Rational trigonometry gives the same geometry one step earlier,
where it is still rational:

    quadrance   Q(v) = <v, v>              the squared length
    spread      s(u, v) = 1 - <u,v>^2 / (Q(u) Q(v))   the squared SINE

so `cos^2 = 1 - s` exactly, and nothing here calls sqrt, sin, cos or atan2.

Both are read off the boundary COLUMNS, which makes them geometry the complex already
has rather than a layout decision. A relation's quadrance is

    Q = 1 + 1/(k-1)

for arity k: 2 for a pairwise relation, falling toward 1 as a relation widens. That is
the T channel, so a relation's length IS its boundary concentration and arity is legible
from it. Verified exact at k = 2..7.

The columns are rebuilt from the boundary structure, NOT read off the assembled float
`B1`. The share 1/(k-1) is not binary-exact for most arities, so converting the stored
double to a Fraction gives the exact value of the double instead of the value: at k=4 it
returns 432691404877902290367942354447019/324518553658426726783156020576256 where the
answer is 4/3. Exactness needs an exact source, not an exact reading of an inexact one.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

__all__ = ["relation_quadrance", "relation_spread", "spreads_at",
           "embed", "embedded_geometry_of",
           "cos_squared", "geometry_of"]


def _columns(rex, edge_ids):
    from rexgraph.faces import _exact_b1_block
    return _exact_b1_block(rex, np.asarray(edge_ids, dtype=np.int64))


def _quadrance(col) -> Fraction:
    return sum((v * v for v in col.values()), Fraction(0))


def _inner(a, b) -> Fraction:
    shared = set(a) & set(b)
    return sum((a[v] * b[v] for v in shared), Fraction(0))


def relation_quadrance(rex, edge: int, *, exact: bool = True):
    """`Q = <c, c>` for one relation's boundary column: its squared length.

    `1 + 1/(k-1)` at arity k, so this is where arity shows up as geometry.
    """
    q = _quadrance(_columns(rex, [int(edge)])[0])
    return q if exact else float(q)


def relation_spread(rex, a: int, b: int, *, exact: bool = True):
    """`s = 1 - <c_a, c_b>^2 / (Q_a Q_b)`: the squared sine of the angle between two
    relations.

    0 when they are parallel, 1 when perpendicular. Returns None when either relation
    has zero quadrance, where no angle is defined; that is an absence and must not be
    read as 0.
    """
    ca, cb = _columns(rex, [int(a), int(b)])
    qa, qb = _quadrance(ca), _quadrance(cb)
    if qa == 0 or qb == 0:
        return None
    ip = _inner(ca, cb)
    s = Fraction(1) - (ip * ip) / (qa * qb)
    return s if exact else float(s)


def cos_squared(s):
    """`cos^2 = 1 - s`. The companion of spread, and the one a renderer wants for a
    projection. Still rational, still no arccosine."""
    if s is None:
        return None
    return (Fraction(1) - s) if isinstance(s, Fraction) else (1.0 - float(s))


def spreads_at(rex, vertex: int, *, exact: bool = True) -> list:
    """Every pairwise spread between the relations meeting at one vertex.

    This is the angular structure of a star, which is what a renderer needs to place
    the relations around a cell. Reported rather than chosen: the complex already fixes
    these angles, so a layout that ignores them is drawing something else.
    """
    rex._ensure_clean()
    bp, bi = rex._boundary_ptr, rex._boundary_idx
    v = int(vertex)
    if bp is None:
        src, tgt = rex._ensure_src_tgt()
        incident = [e for e in range(int(rex.nE))
                    if int(src[e]) == v or int(tgt[e]) == v]
    else:
        bp, bi = np.asarray(bp), np.asarray(bi)
        incident = [e for e in range(int(rex.nE))
                    if v in {int(x) for x in bi[bp[e]:bp[e + 1]]}]
    out = []
    for i, a in enumerate(incident):
        for b in incident[i + 1:]:
            s = relation_spread(rex, a, b, exact=exact)
            out.append({"relations": [int(a), int(b)], "spread": s,
                        "cos_squared": cos_squared(s)})
    return out


def embed(rex, positions, *, exact: bool = True) -> list:
    """Each relation's vector in space: its boundary column applied to the embedding.

    `Q = <c, c>` above is the INTRINSIC length, a function of arity alone. This is the
    extrinsic one, and it needs an embedding because geometry emerges from an embedding
    rather than from the complex: the complex fixes which cells there are and how they
    meet, and where they sit is a further fact a file can carry.

    The construction is `B_1^T P`, one row per relation::

        k = 2   c = (-1, +1)                 ->  p_b - p_a, the edge vector
        k > 2   c = (-1, 1/(k-1), ...)       ->  (mean of the others) - p_distinguished

    so it reduces to the edge vector at arity two and stays one vector per relation above
    it, which is what lets a k-ary relation have a length at all without being split.

    Exact when the positions are rational, which a coordinate file gives for free: an SDF
    writes `1.2124`, four decimal places, exactly a Fraction over 10^4. Passing floats
    gives the exact value of the double, which is the usual distinction.
    """
    rex._ensure_clean()
    P = positions
    rows = []
    for e in range(int(rex.nE)):
        col = _columns(rex, [e])[0]
        vec = None
        for v, coefficient in col.items():
            point = [Fraction(x) if exact else float(x) for x in P[int(v)]]
            scaled = [coefficient * x if exact else float(coefficient) * x
                      for x in point]
            vec = scaled if vec is None else [a + b for a, b in zip(vec, scaled,
                                                                    strict=True)]
        rows.append(vec if vec is not None else [])
    return rows


def embedded_geometry_of(rex, positions, *, limit: int = 0, exact: bool = True) -> dict:
    """Lengths and angles of the complex AS EMBEDDED, exactly where the source allows.

    The same readings `geometry_of` gives from the boundary columns alone, taken instead
    on their images under the embedding, through `rational_trig.quadrance` and `spread`.
    Nothing here calls sqrt, sin, cos or atan2: the quadrance is the squared length and
    the spread the squared sine, so a bond angle comes back as an exact rational.

    Reported beside the intrinsic reading rather than replacing it. They answer different
    questions: the intrinsic quadrance of a benzene ring's delocalised relation is
    `1 + 1/5` whatever the molecule's conformation, and the embedded one moves when the
    ring puckers.
    """
    from rexgraph.rational_trig import quadrance as _q
    from rexgraph.rational_trig import spread as _s

    vectors = embed(rex, positions, exact=exact)
    nE = int(rex.nE)
    n = nE if not limit else min(nE, int(limit))
    quad = [_q(vectors[e], exact=exact) if vectors[e] else (Fraction(0) if exact else 0.0)
            for e in range(n)]

    supports = _supports(rex)
    meeting = []
    for a in range(n):
        for b in range(a + 1, n):
            if not (supports[a] & supports[b]):
                continue
            if not vectors[a] or not vectors[b]:
                continue
            spread = _s(vectors[a], vectors[b], exact=exact)
            if spread is None:
                continue
            meeting.append({
                "relations": [a, b],
                "spread": str(spread) if exact else float(spread),
                "cos_squared": (str(cos_squared(spread)) if exact
                                else float(cos_squared(spread))),
            })
    return {
        "embedded": True,
        "exact": bool(exact),
        "quadrance": [str(q) if exact else float(q) for q in quad],
        "meeting": meeting,
        "reading": ("quadrance is the squared length of B_1^T P and spread the squared "
                    "sine between two of them, so a bond angle is an exact rational when "
                    "the coordinates are; the intrinsic reading is a different question "
                    "and is reported beside this one"),
    }


def _supports(rex) -> list:
    """Each relation's vertex set, arity-general."""
    rex._ensure_clean()
    bp, bi = np.asarray(rex._boundary_ptr), np.asarray(rex._boundary_idx)
    return [{int(v) for v in bi[bp[e]:bp[e + 1]]} for e in range(int(rex.nE))]


def geometry_of(rex, *, limit: int = 0, exact: bool = False) -> dict:
    """The complex's own lengths, and the spreads between relations that meet.

    `exact=False` returns floats for a renderer that has to put a number in a path;
    the rational value is what it was computed FROM either way, so the float is a final
    rounding rather than an accumulated approximation.
    """
    nE = int(rex.nE)
    n = nE if not limit else min(nE, int(limit))
    quad = [relation_quadrance(rex, e, exact=exact) for e in range(n)]

    rex._ensure_clean()
    bp, bi = rex._boundary_ptr, rex._boundary_idx
    if bp is None:
        src, tgt = rex._ensure_src_tgt()
        supports = [{int(src[e]), int(tgt[e])} for e in range(nE)]
    else:
        bp, bi = np.asarray(bp), np.asarray(bi)
        supports = [{int(x) for x in bi[bp[e]:bp[e + 1]]} for e in range(nE)]

    meeting = []
    for a in range(n):
        for b in range(a + 1, n):
            if supports[a] & supports[b]:
                s = relation_spread(rex, a, b, exact=exact)
                meeting.append({"relations": [a, b], "spread": s,
                                "cos_squared": cos_squared(s)})
    return {
        "quadrance": [str(q) if isinstance(q, Fraction) else q for q in quad],
        "meeting": [{**m, "spread": (str(m["spread"]) if isinstance(m["spread"], Fraction)
                                     else m["spread"]),
                     "cos_squared": (str(m["cos_squared"])
                                     if isinstance(m["cos_squared"], Fraction)
                                     else m["cos_squared"])}
                    for m in meeting],
        "exact": bool(exact),
        "reading": ("quadrance is the squared length (1 + 1/(k-1) at arity k) and "
                    "spread is the squared sine; cos^2 = 1 - spread"),
    }
