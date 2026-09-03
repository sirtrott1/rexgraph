"""Graph routes: draw a complex, at any step, in any view, and take the file away.

A session already holds its complexes, one per step, with ids and history. What it could
not do is show one: every reading was reachable and the picture was not, so the app could
describe a complex it could not draw.

Everything here goes through `agent.graph_view.render_payload` and `agent.render_svg`, the
same path the pipeline's `drawing` stage and the `rexgraph_render` tool take, so what the
app draws and what an agent draws cannot differ.

Nothing here computes mathematics. The positions are exact rationals from the cells' own
stars, the lengths are quadrances so they carry arity, the angles are spreads, the colour
is the character through K7's spectrum, and the faces are solved. This module chooses
which complex and hands back a document.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException
from starlette.responses import Response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/graph")

#: the layouts, and what each one is a picture OF
VIEWS = {
    "structural": "the graph drawing: cells placed by what they are NEAR, off L0's low "
                  "eigenvectors with force refinement. Float and iterative, and the one "
                  "to read when the question is what the graph looks like",
    "plane": "exact rational coordinates from each cell's own star: local and exact",
    "character": "the 3D channel embedding: which cells are structurally alike",
    "embedded": "the coordinates the source carried, when it carried any",
    "flow": "where a SIGNAL puts the cells, for data with no geometry of its own; "
            "needs `signal`, one value per relation",
}


def _session(session_id: str):
    from agent.server.app import get_store

    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(404, f"no session {session_id}")
    return session


def _complex_at(session_id: str, step: int | None):
    """The complex a session holds, at a step or as it stands."""
    session = _session(session_id)
    if step is None:
        rex = session.current()
    else:
        try:
            rex = session.at(int(step))
        except Exception as exc:
            raise HTTPException(404, f"no step {step} in {session_id}") from exc
    if rex is None:
        raise HTTPException(404, f"session {session_id} holds no complex")
    labels = (getattr(rex, "_agent_meta", {}) or {}).get("vertex_labels")
    return rex, labels


def _payload(rex, labels, body: dict):
    from agent.graph_view import render_payload

    return render_payload(
        rex, labels=labels,
        limit=int(body.get("limit") or 0),
        select=body.get("select") or None,
        select_dim=int(body.get("select_dim") or 1),
        # a signal makes the flow view possible: for data with no geometry of its own,
        # where a cell sits is where the measurement puts it. Named `signal` rather than
        # inferred, because which reading to lay a complex out by is the caller's claim.
        signal=body.get("signal"),
    )


@router.get("/views")
async def views():
    """The layouts, and what each is a picture of.

    Named rather than assumed, because they answer different questions and a reader
    choosing between them should be told which: benzene's six carbons are structurally
    identical and stack in the character view while sitting on a hexagon in the embedded
    one, and neither picture is wrong.
    """
    return {"views": VIEWS, "default": "structural"}


@router.post("/{session_id}/render")
async def render(session_id: str, body: dict = Body(default={})):
    """The readings and the drawing for a session's complex.

    `step` picks a point in the session's history; omitted, it is the state as it stands.
    `view` picks the layout, `select` dims the cells that do not match rather than
    removing them, and `limit` bounds the document, which is REPORTED rather than
    decided: a truncated picture says so.
    """
    view = str(body.get("view") or "structural")
    if view not in VIEWS:
        raise HTTPException(400, f"view must be one of {', '.join(VIEWS)}")
    rex, labels = _complex_at(session_id, body.get("step"))
    payload = _payload(rex, labels, body)

    from agent.render_svg import colour_scheme, render_svg
    from agent.server.artifacts import plain

    colour_by = str(body.get("colour_by") or "character")
    _fn, legend = colour_scheme(payload, colour_by)
    drawn = {r["index"] for r in payload.get("relations", [])}
    return plain({
        "session_id": session_id,
        "step": body.get("step"),
        "view": view,
        "svg": render_svg(payload, view=view, colour_by=colour_by),
        "cells_drawn": len(drawn),
        "cells_total": int(rex.nE),
        "truncated": len(drawn) < int(rex.nE),
        "faces_drawn": sum(1 for f in payload.get("faces", [])
                           if set(f["relations"]) <= drawn),
        "faces_total": int(rex.nF_hodge),
        "colour": legend,
        "quantities": sorted(payload.get("quantities") or {}),
        "state": payload.get("state"),
        "orientation": payload.get("orientation"),
        "attributes": payload.get("attributes"),
        "selection": payload.get("selection"),
        "field": payload.get("field"),
        "relations": payload.get("relations"),
        "faces": payload.get("faces"),
    })


@router.post("/{session_id}/image")
async def image(session_id: str, body: dict = Body(default={})):
    """The drawing as a file to keep.

    An SVG, because the document is what was computed: the coordinates in it are the
    exact rationals rounded once for the path string, and rasterising here would throw
    that away and fix a resolution besides. A caller wanting a raster has the file.
    """
    view = str(body.get("view") or "structural")
    if view not in VIEWS:
        raise HTTPException(400, f"view must be one of {', '.join(VIEWS)}")
    rex, labels = _complex_at(session_id, body.get("step"))

    from agent.render_svg import render_svg

    svg = render_svg(_payload(rex, labels, body), view=view,
                     colour_by=str(body.get("colour_by") or "character"))
    step = body.get("step")
    name = f"{session_id}-{view}" + (f"-step{int(step)}" if step is not None else "")
    return Response(
        content=svg.encode("utf-8"),
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{name}.svg"',
                 "X-Content-Type-Options": "nosniff"},
    )


@router.get("/{session_id}/history")
async def history(session_id: str):
    """Every step of a session, with the shape of the complex at each.

    The session's own history, read for what a picture needs: which steps exist, what
    happened at each, and how big the complex was, so a reader can pick one to draw
    without loading them all.
    """
    session = _session(session_id)
    out = []
    for entry in session.history():
        out.append({k: entry.get(k) for k in
                    ("step", "action", "summary", "timestamp", "nV", "nE", "nF")
                    if k in entry})
    return {"session_id": session_id, "steps": out, "n_steps": len(out)}


@router.get("/lineages")
async def lineages():
    """The edit lineages this workspace has recorded, newest first.

    A lineage is one object with two coordinates: the RCDB version chain and the step
    inside its TemporalRex. Recording an edit appends to both, so an edit history IS a
    temporal complex rather than a log kept beside one.
    """
    from agent import work_recorder as wr

    try:
        return {"lineages": wr.recorded()}
    except Exception as exc:                       # a store that is not there is not an error
        logger.debug("no lineage store: %s", exc)
        return {"lineages": [], "reason": str(exc)}


@router.get("/lineage/{lineage_id}")
async def lineage(lineage_id: str):
    """Every recorded state of one lineage, with the shape of the complex at each.

    Read off the temporal store, so the steps are the edits and the shapes are what the
    complex actually was, not a description written alongside.
    """
    from agent import work_recorder as wr

    steps = wr.history(lineage_id)
    if not steps:
        raise HTTPException(404, f"no recorded lineage {lineage_id}")
    return {"lineage_id": lineage_id, "steps": steps, "n_steps": len(steps)}


@router.post("/lineage/{lineage_id}/render")
async def render_lineage(lineage_id: str, body: dict = Body(default={})):
    """Draw a lineage at a moment in its history.

    The state reconstructs as a RexGraph, so it draws through exactly the path a live
    complex does: a past edit is not a different kind of thing.
    """
    from agent import work_recorder as wr
    from agent.render_svg import render_svg
    from agent.server.artifacts import plain

    view = str(body.get("view") or "structural")
    if view not in VIEWS:
        raise HTTPException(400, f"view must be one of {', '.join(VIEWS)}")
    when = body.get("at")
    if when is None:
        raise HTTPException(400, "give `at`, a timestamp within the lineage")
    step, rex = wr.state_at(lineage_id, float(when))
    if rex is None:
        raise HTTPException(404, f"no recorded state of {lineage_id} at {when}")
    labels = (getattr(rex, "_agent_meta", {}) or {}).get("vertex_labels")
    payload = _payload(rex, labels, body)
    drawn = {r["index"] for r in payload.get("relations", [])}
    return plain({
        "lineage_id": lineage_id, "step": int(step), "at": float(when), "view": view,
        "svg": render_svg(payload, view=view,
                          colour_by=str(body.get("colour_by") or "character")),
        "cells_drawn": len(drawn), "cells_total": int(rex.nE),
        "state": payload.get("state"), "orientation": payload.get("orientation"),
    })


@router.post("/lineage/{lineage_id}/record")
async def record_lineage(lineage_id: str, body: dict = Body(default={})):
    """Record a session's current complex as the next state of a lineage.

    The edit is the complex, not a description of it, so what gets stored reconstructs
    into the same thing that was analysed and drawn.
    """
    from agent import work_recorder as wr

    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(400, "give `session_id`, the session holding the complex")
    rex, _labels = _complex_at(str(session_id), body.get("step"))
    info = wr.record_complex(str(body.get("kind") or "edit"), rex,
                             lineage_id=lineage_id, force=True,
                             meta={"session_id": session_id},
                             when=body.get("at"))
    if info is None:
        raise HTTPException(400, "nothing was recorded")
    return info


@router.post("/{session_id}/cell")
async def cell(session_id: str, body: dict = Body(default={})):
    """Everything about one cell, for a click on the drawing.

    Which is what makes the picture an interface rather than an image: a relation carries
    its boundary, its arity, its quadrance, its character and its curvature, and a click
    should return those rather than a tooltip's worth of them.
    """
    grade, index = int(body.get("grade", 1)), int(body.get("index", 0))
    if grade not in (0, 1, 2):
        raise HTTPException(400, "grade must be 0, 1 or 2")
    rex, labels = _complex_at(session_id, body.get("step"))

    from agent.server.artifacts import plain

    try:
        shape = rex.cell_shape(grade, index)
    except Exception as exc:
        raise HTTPException(404, f"no cell {index} at grade {grade}") from exc
    out = {"session_id": session_id, "grade": grade, "index": index, "shape": shape,
           "attributes": rex.get_metadata(grade, index)}
    if grade == 1:
        from rexgraph.geometry import relation_quadrance, spreads_at

        out["quadrance"] = str(relation_quadrance(rex, index))
        out["boundary"] = [int(v) for v in rex.relation_supports()[index]]
        if labels:
            out["boundary_labels"] = [str(labels[v]) for v in out["boundary"]
                                      if v < len(labels)]
    if grade == 0:
        from rexgraph.geometry import spreads_at

        # exact rationals, so they leave as strings rather than as floats: the whole
        # point of the spread tower is that an angle is not a float
        out["angles_at"] = [{"relations": a["relations"],
                             "spread": str(a["spread"]),
                             "cos_squared": str(a["cos_squared"])}
                            for a in spreads_at(rex, index)[:20]]
    if grade == 2:
        # the relations a face bounds come from B2's column, not from cell_shape, which
        # carries `below` only at grade 1. Passing an empty list read "open", which is
        # the right answer to the wrong question.
        from rexgraph.core._sparse import to_scipy_csr

        B2 = to_scipy_csr(rex.B2_hodge_sparse).tocsc()
        bounding = [int(B2.indices[j])
                    for j in range(B2.indptr[index], B2.indptr[index + 1])
                    if B2.data[j] != 0]
        out["bounds"] = bounding
        out["reading"] = rex.face_reading(bounding)
    return plain(out)


@router.post("/{session_id}/select")
async def select(session_id: str, body: dict = Body(default={})):
    """Which cells a criteria dict picks out, as a mask.

    A selection, not a subcomplex. Deleting the rest would change the character of every
    cell that remained and recolour cells the filter never mentioned, so the filtered
    picture would disagree with the unfiltered one about cells neither of them selected.
    """
    rex, _labels = _complex_at(session_id, body.get("step"))
    criteria = body.get("criteria") or {}
    grade = int(body.get("grade", 1))
    try:
        mask = rex.select_by_attribute(criteria, dim=grade)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    from agent.server.artifacts import plain

    return plain({"session_id": session_id, "grade": grade, "criteria": criteria,
                  "mask": [int(x) for x in mask],
                  "n_selected": int(sum(int(x) for x in mask)),
                  "n_cells": int(len(mask))})


@router.get("/{session_id}/attributes")
async def attributes(session_id: str):
    """What the source said about the cells, per grade, and the keys available.

    The keys are what a filter box offers, so they come from the complex rather than from
    a list someone maintains.
    """
    rex, _labels = _complex_at(session_id, None)
    store = getattr(rex, "_cell_metadata", None) or {}
    keys = {}
    for grade, cells in store.items():
        found = sorted({k for values in cells.values() for k in values})
        if found:
            keys[str(grade)] = found
    from agent.server.artifacts import plain

    return plain({"session_id": session_id, "keys": keys,
                  "n_annotated": {str(g): len(c) for g, c in store.items() if c}})
