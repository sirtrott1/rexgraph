"""
agent.graph_view: structural coordinates and reachability, without a spectral slice.

The old dashboard positioned vertices by the eigenvectors of L0. That is a linear
grouping: it cuts the complex along a direction that happens to minimise a quadratic
form, and the position it gives a cell says where that cut fell, not what the cell IS.
It also needs a dense eigendecomposition to say it.

The character is already a position. `phi(v)` lives in the simplex over the channel
hats, so a vertex's coordinates ARE its shares of topology, geometry, frustration and
co-participation, and two cells near each other are alike in what they participate in
rather than merely close in a cut. It is exact, needs no eigensolve, and is available at
every scale through the sparse character path.

    character    phi(v) or chi(e) embedded in the regular simplex over the channels.
                 Canonical and lossless whenever the target dimension is nhats - 1;
                 below that it is a projection and says so.
    propagator   where heat from a seed actually goes, e^{-tL} applied by Chebyshev
                 matvec. Coordinates are reach from chosen anchors, so "far" means the
                 complex does not carry signal there, not that a cut separated them.

Reachability is the same operator rather than a hop count: a diffusion says how much
arrives, which is the question a graph database answers with a hop limit and a guess.

Those are the ANALYTIC coordinates and none of them is a graph drawing. Asked to draw a
star, the character puts all nine cells on one point, correctly: they have the same star
character and it is not hiding anything. So `structural_positions` is here too, and it is
the spectral layout the paragraph above rejects, kept apart rather than mixed in. The
rejection stands where it was aimed. A cut is the wrong answer to "what IS this cell",
which is what `positions` is for; it is the right answer to "who is this cell NEAR",
which is what a picture asks. Two questions, two functions, and each says which it
answers.
"""

from __future__ import annotations

import numpy as np

__all__ = ["character_positions", "propagator_positions", "positions",
           "reach", "neighbors", "exact_positions", "structural_positions",
           "flow_positions", "render_payload", "MODES"]

#: the coordinate modes this module offers, in the order a caller should prefer them
MODES = ("character", "propagator")


def character_positions(rex, *, grade: str = "vertex", dim: int = 3) -> dict:
    """Cells placed by their channel character.

    `grade="vertex"` uses phi(v), `grade="edge"` uses chi(e). Both are points in the
    simplex over the channel hats, so the embedding is a change of coordinates rather
    than a fit: nothing is being minimised and there is no iteration to converge.

    The embedding itself is `core._fiber.signal_sphere_proj`, the library's own, rather
    than a second frame built here: equal shares land at the centroid and a cell dominated
    by one channel lands at that channel's corner, so a coordinate reads back as "this
    cell is mostly frustration" without a legend.

    Exact when `dim >= nhats - 1`. Below that the extra channels are dropped and
    `exact` comes back False, because a lower-dimensional picture of a 3-simplex has
    lost something and saying otherwise would make a projection look like a reading.
    """
    if grade not in ("vertex", "edge"):
        raise ValueError(f"grade must be 'vertex' or 'edge', got {grade!r}")
    values = np.asarray(
        rex.vertex_character if grade == "vertex" else rex.structural_character,
        dtype=float)
    if values.ndim != 2 or values.size == 0:
        n = int(rex.nV if grade == "vertex" else rex.nE)
        return {"positions": np.zeros((n, dim)), "mode": "character", "grade": grade,
                "channels": [], "exact": False,
                "note": "no character is available for this complex"}

    channels = list(getattr(rex, "hat_names", None) or
                    [f"channel_{i}" for i in range(values.shape[1])])
    # the library's own simplex embedding, not a second one. There are exactly four
    # channels, and an inactive one is dropped rather than added to, so nhats <= 4 always
    # and the kernel is barycentric across that whole range.
    from rexgraph.core import _fiber

    nhats = int(values.shape[1])
    embedded = np.asarray(_fiber.signal_sphere_proj(
        np.ascontiguousarray(values, dtype=float), values.shape[0], nhats))
    embedded = embedded[:, :max(nhats - 1, 1)]
    full = embedded.shape[1]
    if dim <= full:
        out = embedded[:, :dim]
    else:
        out = np.zeros((embedded.shape[0], dim), dtype=float)
        out[:, :full] = embedded
    return {"positions": np.ascontiguousarray(out), "mode": "character",
            "grade": grade, "channels": channels, "exact": bool(dim >= full),
            "note": "" if dim >= full else
                    f"{full} channel dimensions projected into {dim}"}


def propagator_positions(rex, *, anchors=None, dim: int = 3, t: float = 1.0) -> dict:
    """Vertices placed by how much heat reaches them from chosen anchors.

    One coordinate per anchor: the value of `e^{-tL0}` applied to a unit source there.
    Applied by Chebyshev matvec, so no spectrum is formed and the cost is O(nnz * order).

    Anchors default to the highest-degree vertices, which is a choice and is reported as
    one. Distance here means the complex does not carry signal between two cells, which
    is a statement about transport rather than about where a cut fell.
    """
    from rexgraph.scale_propagator import heat_apply

    nV = int(rex.nV)
    if nV == 0:
        return {"positions": np.zeros((0, dim)), "mode": "propagator",
                "anchors": [], "t": float(t), "exact": False, "note": "empty complex"}

    L0 = rex.L0_sparse if hasattr(rex, "L0_sparse") else None
    if L0 is None:
        import scipy.sparse as sp
        from rexgraph.core._sparse import to_scipy_csr
        B1 = to_scipy_csr(rex.B1_sparse).tocsr()
        L0 = sp.csr_matrix(B1 @ B1.T)

    if anchors is None:
        degree = np.asarray(L0.diagonal()).ravel()
        anchors = [int(i) for i in np.argsort(-degree)[:dim]]
    anchors = [int(a) for a in anchors][:dim]

    cols = []
    for a in anchors:
        seed = np.zeros(nV, dtype=float)
        seed[a] = 1.0
        cols.append(np.asarray(heat_apply(L0, seed, float(t))).ravel())
    out = np.zeros((nV, dim), dtype=float)
    for j, c in enumerate(cols):
        out[:, j] = c
    return {"positions": np.ascontiguousarray(out), "mode": "propagator",
            "anchors": anchors, "t": float(t), "exact": True,
            "note": "coordinates are heat reach from the listed anchors"}


def positions(rex, *, mode: str = "character", dim: int = 3, **kw) -> dict:
    """Structural coordinates for the cells of a complex.

    Deliberately no spectral-embedding mode. A layout off the eigenvectors of L0 is a
    linear grouping whose coordinates describe a cut rather than the cells, and it costs
    a dense eigendecomposition to produce.

    `structural_positions` is that layout, and is not reachable from here on purpose. It
    is a DRAWING, where these are readings, and the separation is the point: a caller
    asking for a cell's coordinates should not be able to get a cut back by passing a
    string.
    """
    if mode == "character":
        return character_positions(rex, dim=dim, **kw)
    if mode == "propagator":
        return propagator_positions(rex, dim=dim, **kw)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def neighbors(rex, vertex: int) -> dict:
    """The star of a vertex: every cell incident to it, as a closed subcomplex.

    The graph-database answer to this is an adjacency lookup, which returns cells and
    leaves the caller to decide what their boundary means. A star is already a
    subcomplex, so what comes back can be analysed as a complex without repair.
    """
    v = int(vertex)
    if not 0 <= v < int(rex.nV):
        raise IndexError(f"vertex {v} is outside 0..{int(rex.nV) - 1}")
    v_mask, e_mask, f_mask = rex.star_of_vertex(v)
    return {"vertex": v,
            "vertices": [int(i) for i in np.flatnonzero(np.asarray(v_mask))],
            "edges": [int(i) for i in np.flatnonzero(np.asarray(e_mask))],
            "faces": [int(i) for i in np.flatnonzero(np.asarray(f_mask))]}


def reach(rex, seeds, *, t: float = 1.0, limit: int = 25) -> dict:
    """What a seed actually reaches, and how strongly.

    A hop limit answers "within k steps", which is a property of the query rather than
    of the structure: it counts every path the same and cannot say whether a cell is
    reached through one thread or a hundred. Diffusing the seed answers how much
    arrives, so the ranking is the complex's own and there is no depth to pick.
    """
    from rexgraph.scale_propagator import heat_apply

    nV = int(rex.nV)
    seeds = [int(s) for s in (seeds if isinstance(seeds, (list, tuple)) else [seeds])]
    bad = [s for s in seeds if not 0 <= s < nV]
    if bad:
        raise IndexError(f"seed vertices outside 0..{nV - 1}: {bad}")

    import scipy.sparse as sp
    from rexgraph.core._sparse import to_scipy_csr
    B1 = to_scipy_csr(rex.B1_sparse).tocsr()
    L0 = sp.csr_matrix(B1 @ B1.T)

    x = np.zeros(nV, dtype=float)
    x[seeds] = 1.0
    field = np.asarray(heat_apply(L0, x, float(t))).ravel()
    order = np.argsort(-np.abs(field))
    order = [int(i) for i in order if int(i) not in set(seeds)][: int(limit)]
    return {"seeds": seeds, "t": float(t),
            "reached": [{"vertex": int(i), "value": float(field[int(i)])}
                        for i in order]}


def exact_positions(rex, *, grade: str = "vertex") -> dict:
    """Cells placed by the rational projection: every coordinate a Fraction.

    The other two modes return floats because they go through a propagator or a simplex
    frame. This one carries the character rationally from the boundary operators and
    lands on rational plane coordinates, so the angle between any two cells is a rational
    spread and nothing called sqrt, sin, cos or atan2 on the way.
    """
    from rexgraph.projection import project_complex
    return project_complex(rex, grade=grade)


def structural_positions(rex, *, dim: int = 2) -> dict:
    """Cells placed so ADJACENCY is readable: the graph drawing, not the analytic view.

    The character views answer "which cells are structurally alike", and they answer it
    exactly. They are a poor drawing of a graph, and on a lot of graphs they are a line.
    That is not a defect in them. The character has `nhats` shares summing to one and
    `chi_T = chi_G` identically, since the diagonal squares each incidence entry and
    squaring kills the sign, so a complex where F is inactive has exactly one free
    parameter and its honest character picture IS one-dimensional. Measured: a 9-vertex
    star puts all 9 cells on 1 point, because all 9 have star character `(1/3, 1/3, 1/3)`
    and are genuinely indistinguishable to it.

    A drawing wants the other question. `rex.layout` answers it and was already in core:
    the low eigenvectors of `L0` as a seed, then force-directed refinement, Barnes-Hut
    above the threshold. Measured against the character projection on the same complexes,
    spread being how far the points are from collinear and distinct being how many
    separate positions they occupy:

        complex               character            rex.layout
        star of 8             0.0000   1 of 9      0.9949  9 of 9
        path of 6             0.0000   3 of 6      0.4862  6 of 6
        6-cycle + 2 chords    0.7112   4 of 6      0.4578  6 of 6
        branching, k <= 5     0.7225  11 of 12     0.7872  12 of 12
        BindingDB panel       0.7507  98 of 1554   0.6877  26 of 1554

    Two things that reading shows. The layout separates what the character cannot, which
    is the point. And on the real panel NEITHER separates much, because 1554 ligands that
    each bind one target are structurally identical and no layout can distinguish what the
    structure does not: that is what the renderer's fan is for, and what a signal view is
    for when there is a measurement to separate them by.

    The force step is pairwise by construction, and that is enforced above rather than
    tolerated: this function declines a complex carrying any relation of arity other than
    two, so the springs never receive a support they cannot represent. `_ensure_src_tgt`
    is exact on what remains, and raises rather than truncating if it is reached with a
    branching C1 at all. Kept from when this WAS a live limitation: feeding the force step
    every pair inside each support measured WORSE on relation cohesion (0.676 against
    0.607, lower being tighter), so pair expansion was not the answer for branching
    either, and declining is.

    Not exact, and says so. Floats the whole way, and the force refinement is iterative.

    Undefined for a branching or witness relation, and it now says that rather than
    drawing something. The seed is the low L0 eigenvectors taken through the pairwise
    component kernel, and that kernel is not the kernel of a complex whose relations carry
    more than two participants: one k-ary relation is one component but rank one, so H0 has
    k-1 dimensions and the component indicators span only one of them. The core declines
    the mode for that reason.

    The absence is reported rather than filled. Substituting the character or exact
    embedding here and still calling the result structural would be the same substitution
    in a different place: those answer who is LIKE whom and where a cell exactly IS, not
    who is NEAR whom, and the payload carries them separately under their own names.
    """
    rex._ensure_clean()
    if not all(len(support) == 2 for support in rex.relation_supports()):
        return {
            "mode": "structural", "grade": "vertex", "exact": False,
            "positions": np.zeros((0, int(dim))), "cells": [],
            "available": False,
            "note": ("undefined for a branching or witness relation: the adjacency layout "
                     "seeds from the pairwise component kernel, which does not span H0 "
                     "once a relation has more than two participants. The exact and "
                     "character embeddings in this payload are defined for any arity."),
        }
    P = np.asarray(rex.layout_3d if int(dim) >= 3 else rex.layout, dtype=float)
    return {
        "mode": "structural", "grade": "vertex", "exact": False,
        "available": True,
        "positions": P,
        "cells": [{"index": i, "at": [float(x) for x in row]}
                  for i, row in enumerate(P)],
        "note": ("adjacency layout: L0's low eigenvectors seeded then force-refined. "
                 "Reads who is NEAR whom, where the character reads who is LIKE whom. "
                 "Float, and iterative in the refinement."),
    }


def flow_positions(rex, signal, *, grade: int = 1) -> dict:
    """Cells placed by where a SIGNAL puts them, for a complex with no geometry to show.

    Semantics and measurements do not come with an embedding. Worse, they are often
    structurally degenerate: in a binding panel every ligand has one binding and one panel
    membership, so their stars are identical, and a layout that reads structure has
    nothing to separate them by. Measured on a real BindingDB panel, 37 vertices occupied
    2 distinct positions, because they have 2 distinct star characters. That picture is
    true and useless.

    What such data does have is FLOW. A signal on the relations decomposes into gradient,
    curl and harmonic parts, and the gradient part descends a potential::

        div = B1 g            how much the signal accumulates at each vertex
        phi = L0^+ div        the potential it descends, by the library's own LSQR
                              solve, which deflates L0's per-component constant kernel
                              exactly rather than drifting into it

    `phi` is a coordinate DERIVED from the data rather than invented for it, and `div`
    says whether a vertex is a source or a sink. So a vertex sits at `(phi, div)`: the
    flow's own ordering across, its source strength up. Two ligands with the same
    structure and different measurements separate, which is the whole point, and they
    separate by the amount the measurement differs.

    The decomposition comes back with the positions, because it is the caption this
    picture needs: `pct_grad = 1` says there is no cycle content here at all, so the
    layout is the whole story, and a complex with curl in it is one where this view is
    only part of it.
    """
    from rexgraph.core._sparse import to_scipy_csr
    from rexgraph.sparse_interfacing import _l0_pinv_matvec

    rex._ensure_clean()
    values = np.asarray(signal, dtype=float).ravel()
    expected = int(rex.nE if grade == 1 else rex.nV)
    if values.shape[0] != expected:
        raise ValueError(
            f"a grade-{grade} signal needs {expected} values, got {values.shape[0]}")

    hodge = rex.hodge_full(values) if grade == 1 else None
    B1 = to_scipy_csr(rex.B1_sparse).tocsr()
    divergence = np.asarray(B1 @ values).ravel()
    potential = np.asarray(_l0_pinv_matvec((B1 @ B1.T).tocsr(), divergence)).ravel()

    return {
        "mode": "flow", "grade": grade,
        "positions": np.column_stack([potential, divergence]),
        "potential": [float(x) for x in potential],
        "divergence": [float(x) for x in divergence],
        "decomposition": ({"gradient": float(hodge["pct_grad"]),
                           "curl": float(hodge["pct_curl"]),
                           "harmonic": float(hodge["pct_harm"])} if hodge else None),
        "exact": False,
        "note": ("x is the potential the gradient part descends, y is the divergence at "
                 "that vertex. Derived from the signal, not fitted to it."),
    }


def render_payload(rex, *, labels=None, dim: int = 3, limit: int = 0,
                   select: dict | None = None, select_dim: int = 1,
                   signal=None) -> dict:
    """Everything a renderer needs, with the geometry exact.

    Assembled here rather than left to the caller because the pieces have to agree: the
    positions, the lengths and the angles are three readings of the same boundary
    columns, and a renderer that took them from different places could draw a relation
    at a length its own quadrance contradicts.

    What is in it:

        positions     rational plane coordinates, plus the character embedding
        quadrance     per relation, `1 + 1/(k-1)` at arity k, so LENGTH CARRIES ARITY
        spreads       between relations that meet, as squared sines; `cos^2 = 1 - s`
        state         latent / filled / closed, which is what decides the face rule
        boundaries    each relation's whole boundary, so a k-ary one draws as one cell
        attributes    what the source said about each cell, per grade
        selection     which cells a criteria dict picks out, as masks. A SELECTION, not
                      a subcomplex: deleting the rest would change every remaining
                      cell's character and recolour cells the filter never touched, so
                      the filtered picture would disagree with the unfiltered one about
                      cells neither of them selected
        curvature     RCFE curvature per relation, which needs faces to be non-zero
        faces         the solved 2-cells: their relations, coefficients, gon and ring
        orientation   whether the faces can be coherently oriented, gauge-invariantly
        field         chi per cell, the channel shares, which is the tensor field
        embedded      the source's own coordinates, when it carried any, with the
                      lengths and angles taken against them, exactly

    `floats` beside every exact value is a final rounding for the path string, not an
    accumulated approximation.
    """
    from rexgraph.faces import face_support, orientation_holonomy
    from rexgraph.geometry import geometry_of
    from rexgraph.rational_trig import exact_character
    from rexgraph.tower import apd, closure_at, manifold_state

    rex._ensure_clean()
    geom = geometry_of(rex, limit=limit, exact=True)
    approx = geometry_of(rex, limit=limit, exact=False)

    # an embedding the SOURCE carried, if there is one. Geometry emerges from an
    # embedding, so where a file records coordinates those are the geometry and the
    # character positions are a different picture: one shows where the cells are, the
    # other shows which cells are structurally alike. Benzene is the clean case, since
    # its six carbons are structurally identical and stack in character space while
    # sitting on a hexagon in the file.
    embedding = getattr(rex, "_embedding", None)
    embedded = None
    if embedding:
        from rexgraph.geometry import embedded_geometry_of
        embedded = embedded_geometry_of(rex, embedding, limit=limit, exact=True)
    bp, bi = rex.boundary_ptr, rex.boundary_idx

    def _label(v):
        return str(labels[v]) if labels is not None and v < len(labels) else f"v{v}"

    if bp is None:
        src, tgt = rex._ensure_src_tgt()
        supports = [[int(s), int(t)] for s, t in zip(src, tgt, strict=True)]
    else:
        bp_a, bi_a = np.asarray(bp), np.asarray(bi)
        supports = [[int(v) for v in bi_a[bp_a[e]:bp_a[e + 1]]]
                    for e in range(int(rex.nE))]

    n = int(rex.nE) if not limit else min(int(rex.nE), int(limit))
    # RCFE curvature is per RELATION and reads B2, so it is zero on a face-free complex
    # and says how much the 2-cells bend around each relation once there are any. Arity is
    # already carried by the quadrance, so this is a second and independent reading.
    curvature = np.asarray(rex.rcfe_curvature, dtype=float) if int(rex.nF_hodge) else None
    relations = []
    for e in range(n):
        relations.append({
            "index": e,
            "boundary": [_label(v) for v in supports[e]],
            # the INDICES as well as the labels. A consumer that recovers indices by
            # sorting the label strings gets a different map whenever the labels do not
            # sort into index order, which "C1..C6, H7..H12" does not: H10 sorts before
            # H7. `cell_view` already carries both for the same reason.
            "boundary_index": [int(v) for v in supports[e]],
            "arity": len(supports[e]),
            "quadrance": geom["quadrance"][e],
            "quadrance_float": approx["quadrance"][e],
            "curvature": (float(curvature[e]) if curvature is not None
                          and e < curvature.shape[0] else 0.0),
        })

    # the channel field, per cell. chi is already a point in the simplex over the four
    # channels and its entries are shares summing to one, so it maps to colour with no
    # scale to choose and no legend that can lie. Exact where the complex allows it, with
    # the float beside it for the fill string.
    rows, channel_names = exact_character(rex)
    chi = np.asarray(rex.structural_character, dtype=float)
    field = {"grade": "relation", "channels": list(channel_names), "exact": rows is not None,
             "cells": [{"index": e,
                        "exact": [str(x) for x in rows[e]] if rows is not None else None,
                        "at": [round(float(x), 6) for x in chi[e]]}
                       for e in range(n)] if chi.size else []}

    # the 2-cells, which nothing was carrying: a solved face column is the drawable
    # polygon, and its support is the gon (a stored zero is not a side, which is the same
    # rule `face_support` and `surface_identity` use)
    faces = []
    nF = int(rex.nF_hodge)
    if nF:

        from rexgraph.core._sparse import to_scipy_csr

        B2 = to_scipy_csr(rex.B2_hodge_sparse).tocsc()
        cells = apd(rex, 2)["cells"]
        for f in range(nF if not limit else min(nF, int(limit))):
            lo, hi = B2.indptr[f], B2.indptr[f + 1]
            column = {int(B2.indices[j]): float(B2.data[j]) for j in range(lo, hi)}
            used = {e: c for e, c in column.items() if c != 0.0}
            ring = sorted({v for e in used for v in supports[e]})
            faces.append({
                "index": f,
                "relations": sorted(used),
                "coefficients": [used[e] for e in sorted(used)],
                "gon": face_support(list(column.values())),
                "vertices": [_label(v) for v in ring],
                "parity": cells[f]["parity"] if f < len(cells) else None,
            })

    # what the source said, per grade, and which cells a criteria dict picks out
    store = getattr(rex, "_cell_metadata", None) or {}
    attributes = {str(grade): {str(index): dict(values)
                               for index, values in cells.items()}
                  for grade, cells in store.items() if cells}
    keys = sorted({k for cells in store.values() for values in cells.values()
                   for k in values})
    selection = None
    if select:
        chosen = rex.select_by_attribute(select, dim=int(select_dim))
        selection = {"criteria": dict(select), "grade": int(select_dim),
                     "mask": [int(x) for x in chosen],
                     "n_selected": int(sum(int(x) for x in chosen)),
                     "reading": ("a selection, not a subcomplex: the cells stay and the "
                                 "unselected ones are drawn back, so the picture still "
                                 "reads the same complex")}

    # where a SIGNAL puts the cells, for data that has no geometry of its own
    flow = None
    if signal is not None:
        try:
            flow = flow_positions(rex, signal)
        except Exception as exc:
            flow = {"mode": "flow", "available": False,
                    "reason": f"{type(exc).__name__}: {exc}"}

    coordinates = None
    if embedding:
        coordinates = {"grade": "vertex", "exact": True, "source": "file",
                       "cells": [{"index": i,
                                  "at": [float(x) for x in point],
                                  "exact": [str(x) for x in point]}
                                 for i, point in enumerate(embedding)]}

    return {
        "positions": {
            "exact": exact_positions(rex, grade="vertex"),
            "embedded": coordinates,
            "relations": exact_positions(rex, grade="edge"),
            "character": character_positions(rex, grade="vertex", dim=dim),
            "structural": structural_positions(rex, dim=dim),
            "flow": flow,
        },
        "relations": relations,
        "faces": faces,
        "orientation": orientation_holonomy(rex, grade=2) if nF else None,
        "field": field,
        "attributes": {"keys": keys, "cells": attributes},
        # the scalars a picture can be coloured BY, per relation. A heat map is a ramp
        # over one of these, and it normalises, so whatever draws it reports the domain.
        "quantities": {
            "curvature": {r["index"]: r.get("curvature", 0.0) for r in relations},
            "arity": {r["index"]: r["arity"] for r in relations},
            "quadrance": {r["index"]: float(r["quadrance_float"]) for r in relations},
        },
        "selection": selection,
        "spreads": geom["meeting"],
        "spreads_float": approx["meeting"],
        "embedded_geometry": embedded,
        "state": manifold_state(rex),
        "closure": closure_at(rex, 2),
        "reading": ("quadrance is the squared length and carries arity; spread is the "
                    "squared sine with cos^2 = 1 - s; state decides the face rule; the "
                    "field is chi, already shares over the channels, so colour needs no "
                    "scale; per-face parity is the REPRESENTATIVE's sign and orientation "
                    "carries the gauge-invariant reading"),
    }
