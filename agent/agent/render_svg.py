"""Drawing a relational complex, from the payload and nothing else.

This module computes no mathematics. Every number it draws with comes from the library:
positions from `projection`, lengths from `geometry.relation_quadrance`, angles from
`relation_spread`, the 2-cells from `faces.solve_face_basis`, their vertex order from
`graded_boundary._order_face_ccw`, the field from `rational_trig.exact_character`. What is
here is the mapping from those to a path string, which is presentation and nothing else.

Three decisions are worth stating, because each is the picture disagreeing with the usual
one for a reason.

**A k-ary relation is ONE cell.** It is drawn as a single closed shape over its whole
support, not as C(k,2) lines between its vertices and not with an invented hub. A clique
expansion invents edges and dissolves the relation's identity; a star expansion invents a
vertex that is not in the complex. The boundary column says which vertices the relation
touches, and the shape is that set.

**Length carries arity.** A relation's quadrance is `1 + 1/(k-1)`, so the stroke width
comes off it rather than off a legend. A 2-ary relation is maximally concentrated and a
wide one diffuse, and the drawing says so without being told which is which.

**Colour is derived, not chosen.** `rexgraph.color` mixes the character into K7's channel
operators, reads its spectrum as wavelengths against the Balmer limit and integrates those
through the CIE colour-matching functions. So a colour is a physical consequence of a
character rather than a lookup, and structurally identical cells come out identical
without being told to.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

__all__ = ["render_svg", "channel_colour", "colour_scheme", "UNSELECTED"]

#: hues for a categorical colouring. Distinct rather than meaningful: an attribute's
#: values have no order unless the attribute says so, and a ramp over them would invent
#: one.
_CATEGORIES = ("#3373d9", "#e68c26", "#2fa36b", "#b8437f", "#7a5cc4",
               "#c0392b", "#0f8b8d", "#8a6d1f", "#5d6d7e", "#a04000")


def colour_scheme(payload, colour_by: str):
    """A per-cell colour function, and what it is a picture of.

    Three kinds, and the difference matters enough to report:

        character    the k7 spectral colour. DERIVED: a physical consequence of the
                     cell's character, with no scale to choose and no legend that can
                     lie. The default, and the only one that is not a decision.
        an attribute a categorical colouring over the values a source recorded. The hues
                     carry no order, because the values have none unless the attribute
                     says so.
        a quantity   a ramp over a scalar the complex computes, which is what a heat map
                     is. It NORMALISES, so the domain is reported: a ramp that hides its
                     endpoints is a picture that changes meaning when the data does.

    Returns `(colour_of, legend)` where `colour_of(grade, index, shares)` gives the hex.
    """
    if colour_by in ("character", "", None):
        return (lambda grade, index, shares: None), {"kind": "character"}

    cells = (payload.get("attributes") or {}).get("cells") or {}
    quantities = payload.get("quantities") or {}

    if colour_by in quantities:
        values = quantities[colour_by]
        numeric = [v for v in values.values() if isinstance(v, (int, float))]
        lo, hi = (min(numeric), max(numeric)) if numeric else (0.0, 0.0)
        span = (hi - lo) or 1.0

        def ramp(grade, index, shares):
            if grade != 1 or index not in values:
                return None
            t = (float(values[index]) - lo) / span
            # a single cold-to-hot ramp, stated rather than tuned
            r = int(255 * min(1.0, max(0.0, 1.5 * t)))
            b = int(255 * min(1.0, max(0.0, 1.5 * (1.0 - t))))
            g = int(255 * (1.0 - abs(2 * t - 1)) * 0.55)
            return "#%02x%02x%02x" % (r, g, b)

        return ramp, {"kind": "quantity", "quantity": colour_by,
                      "domain": [lo, hi],
                      "reading": "a ramp normalises, so its endpoints are the picture"}

    seen, assigned = {}, {}
    for grade, indexed in cells.items():
        for index, values in indexed.items():
            if colour_by in values:
                key = str(values[colour_by])
                if key not in seen:
                    seen[key] = _CATEGORIES[len(seen) % len(_CATEGORIES)]
                assigned[(int(grade), int(index))] = seen[key]

    def categorical(grade, index, shares):
        return assigned.get((grade, index))

    return categorical, {"kind": "attribute", "attribute": colour_by,
                         "legend": seen,
                         "reading": "hues carry no order; the values have none"}




def channel_colour(shares, *, dLT: float = 1.0, eps: float = 1.0) -> str:
    """A cell's character as a colour, through `rexgraph.color.spectral_color`.

    Not a palette. The character is mixed into K7's channel operators, its spectrum is
    read as wavelengths against the Balmer limit and integrated through the CIE
    colour-matching functions, so the colour is a physical consequence of the character
    and two cells the same colour have the same character. `dLT` positions the picture on
    the spectrum and `eps` scales the intensity; both are the caller's, as in spore.
    """
    from rexgraph.color import hex_color, spectral_color

    return hex_color(spectral_color(shares, dLT=dLT, eps=eps))


#: how far back an unselected cell is drawn. Not zero: it stays visible, because the
#: selection is a reading OF this complex and hiding the rest would make the picture a
#: different complex that happens to agree on the part you asked about.
UNSELECTED = 0.12


def _boundary_ids(relation):
    """A relation's boundary vertices, by INDEX.

    The payload carries labels and indices both. Recovering indices from the labels by
    sorting them assumes labels sort into index order, and a molecule's do not: "H10"
    sorts before "H7", so a benzene drew bonds between atoms that have none.
    """
    return [int(v) for v in relation.get("boundary_index",
                                         range(len(relation.get("boundary", []))))]


def _selection_dimming(payload):
    """A per-cell opacity factor from the payload's selection, or 1 everywhere.

    Dimming rather than deleting. Removing the unselected cells would change the
    character of every cell that remained, recolour cells the filter never mentioned, and
    move the positions, so the filtered picture would disagree with the unfiltered one
    about cells neither of them selected.
    """
    selection = payload.get("selection")
    if not selection:
        return lambda grade, index: 1.0
    grade, mask = int(selection["grade"]), selection["mask"]

    def factor(cell_grade, index):
        if cell_grade != grade:
            return 1.0
        return 1.0 if index < len(mask) and mask[index] else UNSELECTED

    return factor


#: how far apart to fan cells that land on one point, as a fraction of the drawing.
FAN_RADIUS = 0.018


def _fan_coincident(placed, width, height, pad=0):
    """Separate cells that landed on exactly one point, and count how many.

    Not a fix for the collapse and not pretending to be. Cells at the same point are
    there because they are structurally identical, which is a true and useful statement,
    and no layout can separate what the structure does not. But drawing eight of them on
    one pixel means the picture shows one cell and says nothing about the other seven, so
    they go on a small circle around the shared point, in index order so it is
    deterministic, and the caption says how many were fanned.
    """
    groups = {}
    for index, point in placed.items():
        groups.setdefault((round(point[0], 6), round(point[1], 6)), []).append(index)
    base = FAN_RADIUS * min(width, height)
    fanned = {k: v for k, v in groups.items() if len(v) > 1}
    out = dict(placed)
    for (cx, cy), members in fanned.items():
        n = len(members)
        # phyllotaxis rather than one ring: a fixed-radius ring of 58 identical cells has
        # them closer together than the markers are wide, so it draws as a blob and says
        # less than the single dot did. Radius by sqrt(rank) keeps the area per cell
        # constant however many there are, and the golden angle keeps them from lining up
        # into spokes that would read as structure.
        for k, index in enumerate(sorted(members)):
            radius = base * np.sqrt((k + 0.5) / n) * np.sqrt(n)
            angle = k * 2.399963229728653
            out[index] = (cx + radius * np.cos(angle), cy + radius * np.sin(angle))

    # the viewport was fitted to the points BEFORE the fan, so a group sitting near an
    # edge can now be outside it. Clamping would pile them back onto the border and undo
    # the fan; scaling everything about the centre keeps every relative position and the
    # whole picture in frame.
    if out and pad:
        xs = [x for x, _ in out.values()]
        ys = [y for _, y in out.values()]
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        half_w, half_h = max(xs) - cx, max(ys) - cy
        room_w, room_h = width / 2.0 - pad, height / 2.0 - pad
        k = min(room_w / half_w if half_w > room_w else 1.0,
                room_h / half_h if half_h > room_h else 1.0)
        if k < 1.0:
            out = {i: (cx + (x - cx) * k, cy + (y - cy) * k)
                   for i, (x, y) in out.items()}
    return out, len(fanned), sum(len(v) for v in fanned.values())


def _spread_of(points):
    """How close the placed points are to a straight line, and how many are distinct.

    Reported rather than acted on: a layout that has collapsed should say so instead of
    letting a reader take a line for a shape.
    """
    P = np.asarray([[float(x), float(y)] for x, y in points], dtype=float)
    distinct = len({(round(x, 9), round(y, 9)) for x, y in P})
    if P.shape[0] < 3:
        return 1.0, distinct
    Q = P - P.mean(axis=0)
    sv = np.linalg.svd(Q, full_matrices=False)[1]
    return (float(sv[1] / sv[0]) if sv[0] > 0 else 0.0), distinct


def _viewport(points, width, height, pad):
    """Map exact rational coordinates into the drawing box. Presentation, not geometry."""
    if not points:
        return lambda p: (width / 2, height / 2)
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    sx = (width - 2 * pad) / (x1 - x0) if x1 > x0 else 0.0
    sy = (height - 2 * pad) / (y1 - y0) if y1 > y0 else 0.0
    scale = min(s for s in (sx, sy) if s > 0) if (sx > 0 or sy > 0) else 1.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def place(p):
        x = width / 2 + (float(p[0]) - cx) * scale
        y = height / 2 - (float(p[1]) - cy) * scale        # SVG y grows downward
        return x, y

    return place


def _ring(payload, place, vertices, points):
    """A face's vertices in boundary order, through the library's own ordering."""
    from rexgraph.graded_boundary import _order_face_ccw

    if len(vertices) < 3:
        return [place(v) for v in vertices]
    flat = np.array([[place(v)[0], place(v)[1], 0.0] for v in vertices])
    # passing the face's own centroid as the centre degenerates the outward normal to z,
    # so the 3D ordering reduces exactly to polar angle in the plane
    order = _order_face_ccw(flat, list(range(len(vertices))), flat.mean(axis=0))
    return [place(vertices[i]) for i in order]


def _camera(azimuth, elevation):
    """An orthographic camera, exactly.

    The camera is the observer, not the data: where you stand does not change what is
    there. But an observation should not invent detail either, and a rotation by an
    arbitrary angle does: `cos(0.6)` is irrational, so a float camera puts the whole
    picture at coordinates that have no exact value at all, and every pan accumulates its
    own error on top of the last.

    So the parameters are half-angle parameters rather than angles. Every rational `t`
    gives a rational point on the circle through `(1-t^2)/(1+t^2)`, `2t/(1+t^2)`, which is
    `projection.rational_direction`, already here for the plane. The basis built from
    those is exactly orthonormal, so `right . up = 0` and each has quadrance 1 as
    integers, and composing two of them stays rational: navigating around the complex
    drifts by nothing, however far you go.

    `t = 0` looks down the axis and `t = 1` is a quarter turn, so the parameter is a
    position on the orbit rather than a number of radians.

    Orthographic, still: a perspective divide scales lengths by depth, and the lengths
    here are quadrances the drawing is supposed to agree with.
    """
    from rexgraph.projection import rational_direction

    ca, sa = rational_direction(azimuth)
    ce, se = rational_direction(elevation)
    right = (ca, Fraction(0), -sa)
    up = (-sa * se, ce, -ca * se)
    forward = (right[1] * up[2] - right[2] * up[1],
               right[2] * up[0] - right[0] * up[2],
               right[0] * up[1] - right[1] * up[0])

    def look(p):
        v = [x if isinstance(x, Fraction) else Fraction(x) for x in p]
        return (sum(a * b for a, b in zip(v, right, strict=True)),
                sum(a * b for a, b in zip(v, up, strict=True)),
                sum(a * b for a, b in zip(v, forward, strict=True)))

    return look


def _hull_ink(n_relations, base=0.18):
    """Fill opacity for one branching relation's hull, given how many are in the picture.

    n translucent hulls stacked compound towards opaque, so a drawing with thirty
    branching relations fills in and stops showing any of them. Dividing by sqrt(n) holds
    the total ink about constant as the count grows, and leaves a picture with four or
    fewer relations drawn exactly as it always was.

    The OUTLINE is deliberately not scaled with it. The outline is what says where the
    relation ends, and it has to survive the crowd that the fill is getting out of the way
    of.
    """
    return base / max(1.0, np.sqrt(max(1, int(n_relations)) / 4.0))


def _label_room(placed, texts=None, font_size=11.0):
    """Which cells the drawing has room to label, and how many that is.

    Not a vertex-count cutoff. A label is a box: `font_size` tall, about `0.6 * font_size`
    per character wide in this font, starting 9px right of its vertex. It collides with
    the next cell exactly when that cell is nearer than the box reaches, which is a fact
    about the picture rather than a number to pick, and it scales with the canvas for
    free: the same complex drawn twice as wide gets its labels back.

    The width is what matters and the height is what a first pass wrongly used. A label
    is roughly three times wider than tall, so measuring against the height let 251 labels
    through on a drawing with room for a fraction of them, and the picture came out
    correct and unreadable.

    Cells that have no room keep their marker and their `<title>`, so nothing is lost: the
    name is one hover away, and the alternative is a drawing where every name is present
    and none is legible.
    """
    items = sorted(placed.items())
    if len(items) < 2:
        return set(placed), len(placed)
    P = np.array([[x, y] for _, xy in items for x, y in [xy]])
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    labels = texts or {i: f"v{i}" for i, _ in items}
    reach = np.array([9.0 + 0.6 * font_size * len(str(labels.get(i, f"v{i}")))
                      for i, _ in items])
    room = nearest >= reach
    return {items[k][0] for k in np.flatnonzero(room)}, int(room.sum())


def _resolve_exposure(payload, dLT, eps):
    """Settle `dLT` and `eps` for this payload, and say what was settled and why.

    The K7 colour is a physical consequence of the character, and it is also a photograph:
    it has an exposure, and at spore's fixed `dLT = 1` most complexes fall off the end of
    the visible band and come back black. Measured on a real binding panel, all eight
    relations did. So the default here is `"auto"`, which asks `rexgraph.color.exposure`
    to solve for the setting rather than pick one, and reports it in the caption. A caller
    who passes a number gets that number, unchanged, including 1.
    """
    if dLT != "auto" and eps != "auto":
        return float(dLT), float(eps), ""
    cells = (payload.get("field") or {}).get("cells") or []
    rows = [c["at"] for c in cells if c.get("at")]
    if not rows:
        return (1.0 if dLT == "auto" else float(dLT),
                1.0 if eps == "auto" else float(eps), "")
    from rexgraph.color import exposure

    solved = exposure(rows)
    out_dLT = solved["dLT"] if dLT == "auto" else float(dLT)
    out_eps = solved["eps"] if eps == "auto" else float(eps)
    dark = solved["of"] - solved["visible"]
    note = f"exposure dLT {out_dLT:.3g}"
    if dark:
        note += f", {dark} of {solved['of']} outside the visible band"
    return out_dLT, out_eps, note


#: the views this renderer draws. `structural` is a drawing; the rest are readings.
_VIEWS = ("plane", "structural", "character", "embedded", "flow")


def render_svg(payload, *, width: int = 900, height: int = 700, pad: int = 60,
               labels: bool = True, dLT: float | str = "auto",
               eps: float | str = "auto",
               view: str = "structural", azimuth=Fraction(3, 5), elevation=Fraction(9, 20),
               colour_by: str = "character") -> str:
    """The payload as an SVG document.

    Takes `agent.graph_view.render_payload` output. Draws, in order: the solved 2-cells as
    filled polygons, then the relations, then the vertices, so a cell never hides the
    boundary that defines it.

    `view="flow"` places cells by where a SIGNAL puts them, for data that has no geometry
    of its own. Semantics and measurements do not come with an embedding and are often
    structurally degenerate besides: every ligand in a binding panel has one binding and
    one panel membership, so a layout reading structure collapses them onto one point and
    says so truthfully and uselessly. Flow separates them, because the measurement does.

    `view="embedded"` uses the coordinates the SOURCE carried, which is the picture to
    draw when there is one: geometry emerges from an embedding, so a molecule's own atom
    block is its geometry and the character layout is a different statement. Benzene makes
    the difference plain, since its six carbons are structurally identical and stack in
    character space while sitting on a hexagon in the file.

    `view="structural"` is the one to reach for when the question is what the graph LOOKS
    like. The other views place a cell by what it IS, and two cells that are structurally
    identical then land on the same point, correctly and unhelpfully: a 9-vertex star puts
    all 9 on one, because all 9 have star character (1/3, 1/3, 1/3). This view places a
    cell by what it is NEAR, off L0's low eigenvectors with force refinement, which is the
    layout already sitting in `rexgraph.core._spectral`. Measured against the plane view,
    spread being distance from collinear: star of 8 goes 0.0000 to 0.9949, path of 6
    0.0000 to 0.4862. It is float and iterative, and the caption says so.

    `view="plane"` uses the exact rational coordinates, where a position is the cell's own
    star and nothing else, so it is exact and local. `view="character"` uses the 3D
    character embedding through an orthographic camera, which is where a height field
    lives: the third axis is a real channel rather than an added dimension, so a ridge in
    the picture is a ridge in the character. Depth-sorted rather than z-buffered, which is
    enough because the cells are flat.

    The camera is orthographic on purpose. A perspective divide would scale lengths by
    depth, and the lengths here are quadrances the drawing is supposed to agree with.

    `azimuth` and `elevation` are HALF-ANGLE PARAMETERS, not radians: every rational one
    gives a rational point on the circle, so the camera basis is exactly orthonormal and
    panning composes without drift. 0 looks down the axis, 1 is a quarter turn.
    """
    if view not in _VIEWS:
        raise ValueError(f"view must be one of {_VIEWS}, got {view!r}")

    dLT, eps, exposure_note = _resolve_exposure(payload, dLT, eps)


    colour_of, _legend = colour_scheme(payload, colour_by)

    if view == "flow":
        flow = payload.get("positions", {}).get("flow")
        if not flow or flow.get("available") is False:
            return _document(width, height, [
                '<text x="20" y="30" fill="#888">this payload carries no signal to '
                'flow</text>'])
        return _render_flow(payload, flow, width, height, pad, labels, dLT, eps,
                            colour_of, exposure_note)

    if view == "embedded":
        embedded = payload.get("positions", {}).get("embedded")
        if not embedded:
            return _document(width, height, ['<text x="20" y="30" fill="#888">'
                                             'this source carried no coordinates</text>'])
        return _render_embedded(payload, embedded, width, height, pad, labels,
                                dLT, eps, azimuth, elevation, colour_of,
                                exposure_note)

    if view == "character":
        return _render_character(payload, width, height, pad, labels, dLT, eps,
                                 azimuth, elevation, colour_of, exposure_note)

    drawn_view = view
    structural = payload.get("positions", {}).get("structural", {})
    # The adjacency layout is undefined for a branching or witness relation: it seeds from
    # the pairwise component kernel, which does not span H0 once a relation carries more
    # than two participants. Rather than return a blank document, fall through to the
    # exact placement, which is defined at every arity and is exact rather than iterative.
    # The caption names which view was drawn, so the substitution is stated rather than
    # silent; a drawing that quietly answered a different question under the requested
    # view's name would be worse than an empty one.
    if view == "structural" and structural.get("available") is False:
        drawn_view = "exact"

    if drawn_view == "structural":
        source = structural
        cells = source.get("cells", [])
        # Fraction of a float is that double exactly, so the viewport arithmetic below is
        # unchanged. It does not make the coordinate exact and the caption does not claim
        # it does: the force refinement is iterative and the seed is an eigensolve.
        points = {c["index"]: (Fraction(c["at"][0]), Fraction(c["at"][1]))
                  for c in cells}
    else:
        source = payload.get("positions", {}).get("exact", {})
        cells = source.get("cells", [])
        points = {c["index"]: (Fraction(c["x"]), Fraction(c["y"])) for c in cells}
    if not cells:
        return _document(width, height, ['<text x="20" y="30" fill="#888">'
                                         'no coordinates for this complex</text>'])

    to_box = _viewport(list(points.values()), width, height, pad)
    ratio, distinct = _spread_of(points.values())
    placed, n_groups, n_fanned = _fan_coincident(
        {i: to_box(p) for i, p in points.items()}, width, height, pad)
    place = placed.__getitem__

    field = {c["index"]: c["at"] for c in payload.get("field", {}).get("cells", [])}
    relations = payload.get("relations", [])
    dim_of = _selection_dimming(payload)
    # BY INDEX, not by position: `limit` truncates the relation list while a face still
    # names the relations it bounds by their index in the complex, so indexing
    # positionally raised as soon as a picture was bounded.
    by_index = {r["index"]: r for r in relations}
    body = []

    # 2-cells first, so the relations bounding them stay visible on top
    for face in payload.get("faces", []):
        bounding = [by_index.get(e) for e in face["relations"]]
        if any(r is None for r in bounding):
            continue                 # a face whose relations were not drawn is not drawn
        ring = sorted({v for r in bounding for v in _boundary_ids(r)} & set(points))
        if len(ring) < 3:
            continue
        pts = _ring(payload, place, ring, points)
        d = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        # a frustrated face is hatched rather than recoloured: orientation is a different
        # reading from character and must not be confused with it
        fill = "#c8ccd4" if face.get("parity", 1) >= 0 else "#d8b4b4"
        body.append(f'<polygon points="{d}" fill="{fill}" fill-opacity="0.35" '
                    f'stroke="none"><title>face {face["index"]}: {face["gon"]}-gon, '
                    f'parity {face.get("parity")}</title></polygon>')

    for relation in relations:
        ids = [v for v in _boundary_ids(relation) if v in points]
        if not ids:
            continue
        shares = field.get(relation["index"], [1, 0, 0, 0])
        colour = (colour_of(1, relation["index"], shares)
                  or channel_colour(shares, dLT=dLT, eps=eps))
        # quadrance is 1 + 1/(k-1): concentrated at arity 2, diffuse as the relation
        # widens, so the stroke reads the arity off the geometry rather than a legend
        stroke = 1.0 + 3.0 * (float(relation["quadrance_float"]) - 1.0)
        # curvature is a SECOND reading and gets its own encoding: the stroke already
        # carries arity, so bending goes to opacity. Zero curvature is a flat relation and
        # draws faint; nothing here rescales it, so a complex with no faces draws faint
        # throughout, which is the true statement about it.
        bend = abs(float(relation.get("curvature", 0.0)))
        opacity = (0.55 + 0.45 * min(bend, 1.0)) * dim_of(1, relation["index"])
        title = (f'relation {relation["index"]}: arity {relation["arity"]}, '
                 f'quadrance {relation["quadrance"]}, '
                 f'curvature {relation.get("curvature", 0.0):.4f}')
        if len(ids) == 2:
            (x1, y1), (x2, y2) = place(ids[0]), place(ids[1])
            body.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                        f'stroke="{colour}" stroke-width="{stroke:.2f}" '
                        f'stroke-opacity="{opacity:.2f}" '
                        f'stroke-linecap="round"><title>{title}</title></line>')
        else:
            # ONE cell over the whole support: no clique, no invented hub
            pts = _ring(payload, place, ids, points)
            d = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            body.append(f'<polygon points="{d}" fill="{colour}" '
                        f'fill-opacity="{_hull_ink(len(relations)):.3f}" '
                        f'stroke="{colour}" stroke-width="{stroke:.2f}" '
                        f'stroke-opacity="{opacity:.2f}" '
                        f'stroke-linejoin="round"><title>{title}</title></polygon>')

    roomy, n_roomy = _label_room(placed) if labels else (set(), 0)
    for index, point in sorted(points.items()):
        x, y = place(index)
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#1b1d21" '
                    f'fill-opacity="{dim_of(0, index):.2f}" '
                    f'stroke="#f4f2ee" stroke-width="1.5"><title>vertex {index} at '
                    f'({point[0]}, {point[1]})</title></circle>')
        if labels and index in roomy:
            body.append(f'<text x="{x + 9:.2f}" y="{y - 8:.2f}" font-size="11" '
                        f'fill="#5a5f68">v{index}</text>')

    state = payload.get("state", {})
    orientation = payload.get("orientation") or {}
    note = ""
    if distinct < len(points):
        note += f' | {distinct}/{len(points)} distinct'
    if n_groups:
        note += f', {n_fanned} fanned apart'
    if ratio < 0.05 and len(points) > 2:
        note += ' | COLLINEAR, the character does not separate these'
    if labels and n_roomy < len(points):
        note += f' | {n_roomy}/{len(points)} labelled, the rest have no room'
    if drawn_view != view:
        note += (f' | {view} view undefined for a branching or witness relation, '
                 f'drawn with the exact placement')
    if exposure_note:
        note += f' | {exposure_note}'
    caption = (f'{len(points)} vertices, {len(relations)} relations, '
               f'{len(payload.get("faces", []))} faces{note}'
               f' | {state.get("state", "")}'
               + (f' | {"orientable" if orientation.get("orientable") else "not orientable"}'
                  if orientation else ""))
    body.append(f'<text x="{pad}" y="{height - 22}" font-size="12" fill="#5a5f68">'
                f'{caption}</text>')
    return _document(width, height, body)


def _render_flow(payload, flow, width, height, pad, labels, dLT, eps, colour_of=None,
                 exposure_note=""):
    """Cells at (potential, divergence): the flow's own ordering across, source strength up.

    Flat on purpose. This is not a projection of anything three-dimensional; both axes are
    readings of one signal, so a camera would suggest a space that is not there.
    """
    raw = np.asarray(flow["positions"], dtype=float)
    if raw.size == 0:
        return _document(width, height, ['<text x="20" y="30" fill="#888">'
                                         'nothing to place</text>'])
    place = _viewport([(x, y) for x, y in raw], width, height, pad)
    flat = {i: place(tuple(p)) for i, p in enumerate(raw)}
    depth = dict.fromkeys(flat, 0.0)
    caption = "potential across, divergence up"
    if exposure_note:
        caption += f" | {exposure_note}"
    decomposition = flow.get("decomposition")
    if decomposition:
        caption += (f" | gradient {decomposition['gradient']:.0%}"
                    f" curl {decomposition['curl']:.0%}"
                    f" harmonic {decomposition['harmonic']:.0%}")
    return _scene(payload, raw, width, height, pad, labels, dLT, eps,
                  Fraction(0), Fraction(0), colour_of=colour_of, caption=caption,
                  placed=(flat, depth))


def _render_embedded(payload, embedded, width, height, pad, labels, dLT, eps,
                     azimuth, elevation, colour_of=None, exposure_note=""):
    """The source's own coordinates, through the same orthographic camera.

    Three dimensions if the file gave three, which a molecule does. The camera is the one
    the character view uses, so the two pictures are comparable.
    """
    # the EXACT coordinates, not the float copies beside them. A coordinate file records
    # its positions exactly (an SDF writes four decimals, a PDB three), the camera is
    # rational, so this view is exact from the file to the last step before the path
    # string. Reading `at` would have thrown that away at the first hop for nothing.
    rows = []
    for c in embedded["cells"]:
        point = [Fraction(x) for x in c.get("exact", [])] or \
                [Fraction(x) for x in c["at"]]
        point = (point + [Fraction(0)] * 3)[:3]
        rows.append(point)
    return _scene(payload, rows, width, height, pad, labels, dLT, eps, azimuth, elevation,
                  colour_of=colour_of,
                  caption="source coordinates | orthographic, exact, depth sorted"
                          + (f" | {exposure_note}" if exposure_note else ""))


def _render_character(payload, width, height, pad, labels, dLT, eps, azimuth, elevation,
                      colour_of=None, exposure_note=""):
    """The 3D character embedding, orthographic and depth-sorted."""
    character = payload.get("positions", {}).get("character", {})
    raw = np.asarray(character.get("positions", []), dtype=float)
    if raw.size == 0:
        return _document(width, height, ['<text x="20" y="30" fill="#888">'
                                         'no character positions for this complex</text>'])
    if raw.shape[1] < 3:
        raw = np.hstack([raw, np.zeros((raw.shape[0], 3 - raw.shape[1]))])

    channels = ", ".join(character.get("channels", []))
    return _scene(payload, raw, width, height, pad, labels, dLT, eps, azimuth, elevation,
                  colour_of=colour_of,
                  caption=f"character space over {channels} | orthographic, "
                          f"depth sorted"
                          + (f" | {exposure_note}" if exposure_note else ""))


def _order_projected(ids, flat):
    """Projected vertices in boundary order, through the library's own ordering."""
    from rexgraph.graded_boundary import _order_face_ccw

    if len(ids) < 3:
        return list(ids)
    pts = np.array([[flat[v][0], flat[v][1], 0.0] for v in ids])
    order = _order_face_ccw(pts, list(range(len(ids))), pts.mean(axis=0))
    return [ids[i] for i in order]


def _scene(payload, raw, width, height, pad, labels, dLT, eps, azimuth, elevation,
           *, caption, colour_of=None, placed=None):
    """One scene builder: the character view, the source coordinates, and the flow.

    `placed` short-circuits the camera for a view that is already flat, so the flow does
    not get rotated through a third dimension it does not have.
    """
    if placed is not None:
        flat, depth = placed
    else:
        look = _camera(azimuth, elevation)
        seen = [look(p) for p in raw]
        place = _viewport([(x, y) for x, y, _ in seen], width, height, pad)
        flat = {i: place((x, y)) for i, (x, y, _) in enumerate(seen)}
        depth = {i: z for i, (_x, _y, z) in enumerate(seen)}

    field = {c["index"]: c["at"] for c in payload.get("field", {}).get("cells", [])}
    relations = payload.get("relations", [])
    dim_of = _selection_dimming(payload)
    by_index = {r["index"]: r for r in relations}
    drawable = []

    # the solved 2-cells, depth-sorted with everything else. They were missing here
    # entirely: the plane view drew the faces the sign solver produces and the 3D views
    # drew none, so the same complex had two different contents depending on the camera.
    for face in payload.get("faces", []):
        bounding = [by_index.get(e) for e in face["relations"]]
        if any(r is None for r in bounding):
            continue
        ring = sorted({v for r in bounding for v in _boundary_ids(r)} & set(flat))
        if len(ring) < 3:
            continue
        ordered = _order_projected(ring, flat)
        pts = " ".join(f"{flat[v][0]:.2f},{flat[v][1]:.2f}" for v in ordered)
        fill = "#c8ccd4" if face.get("parity", 1) >= 0 else "#d8b4b4"
        drawable.append((min(depth[v] for v in ring) - 1,
                         f'<polygon points="{pts}" fill="{fill}" fill-opacity="0.35" '
                         f'stroke="none"><title>face {face["index"]}: {face["gon"]}-gon, '
                         f'parity {face.get("parity")}</title></polygon>'))

    for relation in relations:
        ids = [v for v in _boundary_ids(relation) if v in flat]
        if len(ids) < 2:
            continue
        shares = field.get(relation["index"], [1, 0, 0, 0])
        colour = ((colour_of(1, relation["index"], shares) if colour_of else None)
                  or channel_colour(shares, dLT=dLT, eps=eps))
        stroke = 1.0 + 3.0 * (float(relation["quadrance_float"]) - 1.0)
        bend = abs(float(relation.get("curvature", 0.0)))
        opacity = (0.55 + 0.45 * min(bend, 1.0)) * dim_of(1, relation["index"])
        title = (f'relation {relation["index"]}: arity {relation["arity"]}, '
                 f'curvature {relation.get("curvature", 0.0):.4f}')
        mean_depth = sum(depth[v] for v in ids) / len(ids)
        if len(ids) == 2:
            (x1, y1), (x2, y2) = flat[ids[0]], flat[ids[1]]
            drawable.append((mean_depth,
                             f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
                             f'y2="{y2:.2f}" stroke="{colour}" '
                             f'stroke-width="{stroke:.2f}" stroke-opacity="{opacity:.2f}" '
                             f'stroke-linecap="round"><title>{title}</title></line>'))
        else:
            # same ordering as the plane view: in file order a polygon can self-cross,
            # and a crossed outline is a different shape from the cell it stands for
            ordered = _order_projected(ids, flat)
            pts = " ".join(f"{flat[v][0]:.2f},{flat[v][1]:.2f}" for v in ordered)
            drawable.append((mean_depth,
                             f'<polygon points="{pts}" fill="{colour}" '
                             f'fill-opacity="{_hull_ink(len(relations)):.3f}" '
                             f'stroke="{colour}" '
                             f'stroke-width="{stroke:.2f}" stroke-opacity="{opacity:.2f}" '
                             f'stroke-linejoin="round"><title>{title}</title></polygon>'))

    # the same rule the plane view uses. A camera can crowd cells the layout had spread,
    # so the room has to be measured on the PROJECTED points rather than inherited.
    roomy, n_roomy = _label_room(flat) if labels else (set(), 0)
    for index, (x, y) in flat.items():
        drawable.append((depth[index],
                         f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#1b1d21" '
                         f'fill-opacity="{dim_of(0, index):.2f}" '
                         f'stroke="#f4f2ee" stroke-width="1.5"><title>vertex {index}'
                         f'</title></circle>'))
        if labels and index in roomy:
            drawable.append((depth[index],
                             f'<text x="{x + 9:.2f}" y="{y - 8:.2f}" font-size="11" '
                             f'fill="#5a5f68">v{index}</text>'))

    # painter's algorithm: farthest first. Enough here because every cell is flat.
    body = [markup for _d, markup in sorted(drawable, key=lambda item: item[0])]
    # the same reading the plane view captions with, so the two do not describe one
    # complex differently depending on which way you are looking at it
    state = payload.get("state", {}) or {}
    orientation = payload.get("orientation") or {}
    tail = f' | {state.get("state")}' if state.get("state") else ""
    if orientation:
        tail += f' | {"orientable" if orientation.get("orientable") else "not orientable"}'
    body.append(f'<text x="{pad}" y="{height - 22}" font-size="12" fill="#5a5f68">'
                f'{caption}{tail}</text>')
    return _document(width, height, body)


def _document(width, height, body) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            f'<rect width="{width}" height="{height}" fill="#faf8f5"/>\n'
            + "\n".join(body) + "\n</svg>\n")
