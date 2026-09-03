"""
agent.mcp_tools: the stack's capabilities as tools a model can call.

A tool definition is a promise that a name resolves to something that runs. Definitions
kept apart from their handlers drift: a name gets renamed on one side, the definition
still advertises it, and the failure only appears when a model tries to call it.

So a tool here is one object carrying its schema AND its handler, `definitions()`
derives the MCP payload from that object rather than from a parallel list, and
`call()` dispatches through the same registry. A name that resolves to nothing cannot
be advertised, because there is nowhere to write it down.

Every handler takes JSON-shaped arguments and returns a JSON-shaped result, so the
same registry serves an MCP server, an HTTP route and a Python caller unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

#: files accepted by the tools that read them. Named by handle rather than by path:
#: a path is a name in the SERVER's namespace, so accepting one from the network is
#: accepting a request to read any file the process can. See `server.handles`.
_FILES = {"type": "array", "items": {"type": "string"},
          "description": "Handles of the files to read, in order, as returned by "
                         "upload. A local operator may pass paths instead."}


@dataclass
class Context:
    """Who is calling and what they may reach.

    Absent for a direct Python caller, which is the operator on their own machine and
    is not being restricted from it. Present for every call that arrived over a socket.
    """

    workspace: str = "default"
    identity: str = "local"
    is_admin: bool = True
    auth_enabled: bool = False


@dataclass
class Tool:
    """One callable capability: its schema, who may run it, and what answers it."""

    name: str
    description: str
    properties: dict
    handler: Callable
    required: list[str] = field(default_factory=list)
    #: "user" or "admin". Anything that reaches beyond the caller's own request needs
    #: admin, so the level is declared next to the handler rather than in a table
    #: somewhere else that can fall out of step with it.
    requires: str = "user"
    #: whether this tool names content it will read, and so has to be resolved against
    #: the caller's workspace before the handler sees it
    reads_files: bool = True
    #: WHICH arguments carry that content. A tool taking two complexes names them
    #: something other than `files`, and resolving only the one canonical name would
    #: let the others through as raw paths, which is the hole this exists to close.
    file_args: tuple = ("files",)
    #: whether the handler is given the calling context, for tools that reach stored
    #: state and so have to check it belongs to the caller
    wants_context: bool = False

    def definition(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {"type": "object", "properties": self.properties,
                             "required": list(self.required)},
            "requires": self.requires,
        }


def _join(files, face_selection: str = "") -> dict:
    from agent.knowledge import join
    k = join(*files)
    return {
        "n_entities": k.nV, "n_relations": k.nE,
        "report": k.report,
        "relations": [list(t) for t in k.triples(with_origin=True)[:200]],
    }


def _reason(files, terms=None) -> dict:
    from agent import ontology_reasoning as R
    from agent.knowledge import join
    k = join(*files)
    return R.reason(k.triples(), terms=terms or None)


def _enrich(files, study, universe=None, min_term_size: int = 1) -> dict:
    from agent.enrichment import enrich
    from agent.knowledge import join
    k = join(*files)
    out = enrich(k, study, universe=universe or None,
                 min_term_size=int(min_term_size))
    out["terms"] = out["terms"][:50]
    return out


def _releases(files, labels=None) -> dict:
    from agent.ontology_releases import load_releases, navigate, summary
    releases = load_releases(files, labels=labels or None)
    out = summary(releases)
    out["navigation"] = navigate(releases)
    return out


def _similarity(files, term_a: str, term_b: str) -> dict:
    from agent.knowledge import join
    from agent.term_similarity import (
        ancestor_overlap,
        discrimination,
        hierarchy_from_knowledge,
    )
    h = hierarchy_from_knowledge(join(*files))
    overlap = ancestor_overlap(h, term_a, term_b)
    lost = discrimination(h, term_a, term_b)
    return {
        "terms": [term_a, term_b],
        "overlap": float(overlap),
        "overlap_exact": str(overlap),
        "shared_ancestors": sorted(h.shared_ancestors(term_a, term_b)),
        "n_shared": lost["n_shared"],
    }


def _homology(files) -> dict:
    from agent.knowledge import join
    rex = join(*files).rex(face_selection="none")
    tower = rex.rank_tower()
    return {
        "betti": [int(b) for b in rex.betti],
        "ranks": tower["ranks"],
        "grades": tower["grades"],
        "euler": tower["euler"],
    }


def _health(files) -> dict:
    from agent.knowledge import join
    k = join(*files)
    return {"health": k.health(), "n_entities": k.nV, "n_relations": k.nE}


def _propagate(files, seed, t: float = 1.0, limit: int = 25) -> dict:
    import numpy as np

    from agent.knowledge import join
    k = join(*files)
    field = k.propagate(list(seed), t=float(t))
    triples = k.triples()
    order = np.argsort(-np.abs(field))[: int(limit)]
    return {"seed": list(seed),
            "reached": [{"relation": list(triples[int(i)]),
                         "value": float(field[int(i)])} for i in order]}


def _stored(record_id: str, quantity: str, op: str, threshold: float,
            channel: int = 0, limit: int = 100, _ctx=None) -> dict:
    import numpy as np

    from agent.rcdb import default_store
    store = default_store()
    record = store.get_record(record_id)
    meta = (record.meta or {}) if record is not None else {}

    # A record id is a bare string, so guessing one is cheap. The store is shared
    # across workspaces, so belonging is checked here rather than assumed: a record
    # stamped with another workspace reads as absent, because saying "not yours"
    # confirms it exists.
    if _ctx is not None and _ctx.auth_enabled:
        owner = meta.get("workspace")
        if owner is not None and owner != _ctx.workspace:
            record = None

    rex = store.get(record_id) if record is not None else None
    if rex is None:
        return {"error": f"no record {record_id!r}"}
    mask = rex.select(quantity, op, float(threshold), channel=int(channel))
    return {
        "record": record_id,
        "n_selected": int(np.asarray(mask).sum()),
        "cells": rex.selected(mask, labels=meta.get("vertex_labels"),
                              limit=int(limit)),
    }


def _source_rex(files=None, record_id: str = "", face_selection: str = "", _ctx=None):
    """The complex a tool is being asked about, from files or from the store."""
    if record_id:
        from agent.rcdb import default_store
        store = default_store()
        record = store.get_record(record_id)
        meta = (record.meta or {}) if record is not None else {}
        if _ctx is not None and _ctx.auth_enabled and record is not None:
            owner = meta.get("workspace")
            if owner is not None and owner != _ctx.workspace:
                record = None
        rex = store.get(record_id) if record is not None else None
        if rex is None:
            raise KeyError(f"no record {record_id!r}")
        return rex, meta
    from agent.knowledge import join
    k = join(*(files or []))
    return k.rex(face_selection=face_selection or "none"), {}


def _edges_of(rex, mask):
    """The edge mask a selection induces, and which grade the selection was on.

    `select` returns a mask over whichever grade the quantity lives on: kappa and phi
    are per vertex, chi and curvature per relation. Feeding a vertex mask in as an edge
    mask is not a type error anywhere, it just silently restricts to the wrong cells, so
    the grade is decided here from the mask's own length.

    A vertex selection induces the relations whose boundary lies ENTIRELY inside it.
    Read off the whole boundary rather than a pair of endpoints, so a branching relation
    is included only when every vertex it touches was selected.
    """
    import numpy as np

    mask = np.asarray(mask).astype(bool).ravel()
    nV, nE = int(rex.nV), int(rex.nE)
    if mask.shape[0] == nE:
        return mask.astype(np.uint8), "edge"
    if mask.shape[0] != nV:
        raise ValueError(
            f"a selection of {mask.shape[0]} cells matches neither nV={nV} nor nE={nE}")
    rex._ensure_clean()
    bp, bi = rex.boundary_ptr, rex.boundary_idx
    out = np.zeros(nE, dtype=np.uint8)
    if bp is None:
        src, tgt = rex._ensure_src_tgt()
        out[:] = (mask[np.asarray(src)] & mask[np.asarray(tgt)]).astype(np.uint8)
        return out, "vertex"
    bp = np.asarray(bp)
    bi = np.asarray(bi)
    for e in range(nE):
        span = bi[bp[e]:bp[e + 1]]
        out[e] = 1 if span.size and bool(mask[span].all()) else 0
    return out, "vertex"


def _restrict(quantity: str, op: str, threshold: float, files=None, record_id: str = "",
              channel: int = 0, limit: int = 50, _ctx=None) -> dict:
    import numpy as np

    rex, meta = _source_rex(files, record_id, _ctx=_ctx)
    mask = np.asarray(rex.select(quantity, op, float(threshold), channel=int(channel)))
    e_mask, grade = _edges_of(rex, mask)

    # Closing the selection is what makes this a restriction rather than a filter: the
    # kernel pulls in every cell the chosen relations bound, so what comes back is a
    # complex in its own right and its boundary operators do not reference anything
    # outside it. An operation handed this cannot reach past it.
    v_mask, e_closed, f_mask = rex.subcomplex(e_mask=e_mask)
    quot = rex.quotient(v_mask, e_closed, f_mask)

    labels = meta.get("vertex_labels")
    return {
        "selected_on": grade,
        "selected": int(np.asarray(mask).astype(bool).sum()),
        "closed_to": {"nV": int(np.asarray(v_mask).sum()),
                      "nE": int(np.asarray(e_closed).sum()),
                      "nF": int(np.asarray(f_mask).sum())},
        "whole": {"nV": int(rex.nV), "nE": int(rex.nE),
                  "betti": [int(b) for b in rex.betti]},
        # the homology of what is left when the restriction is collapsed: what the
        # restriction was HIDING, which a filtered row set cannot answer at all
        "relative_betti": [int(b) for b in (quot.get("betti_rel") or [])],
        "cells": rex.selected(mask, labels=labels, limit=int(limit)),
    }


def _curvature(files=None, record_id: str = "", limit: int = 25, _ctx=None) -> dict:
    import numpy as np

    rex, _meta = _source_rex(files, record_id, face_selection="auto", _ctx=_ctx)
    ac = rex.attributed_curvature()
    kappa = np.asarray(ac["kappa_f"], dtype=float)
    order = np.argsort(-kappa)[: int(limit)]
    eq = rex.strain_equilibrium() if int(rex.nF_hodge) else {}
    return {
        "nF": int(rex.nF_hodge),
        "total_curvature": float(kappa.sum()),
        "strained_faces": [{"face": int(f), "kappa": float(kappa[int(f)])}
                           for f in order if kappa[int(f)] > 0.0],
        # the conservation law over the same field: an identity, so a residual above
        # zero is a disagreement rather than a tolerance being exceeded
        "bianchi_ok": bool(eq.get("bianchi_ok", True)),
        "bianchi_residual": float(eq.get("bianchi_residual", 0.0)),
        "strain_norm": float(eq.get("strain_norm", 0.0)),
    }


def _positions(files=None, record_id: str = "", mode: str = "character",
               dim: int = 3, grade: str = "vertex", limit: int = 500,
               _ctx=None) -> dict:
    from agent.graph_view import positions
    rex, meta = _source_rex(files, record_id, _ctx=_ctx)
    kw = {"grade": grade} if mode == "character" else {}
    out = positions(rex, mode=mode, dim=int(dim), **kw)
    pos = out.pop("positions")
    labels = meta.get("vertex_labels") if out.get("grade", "vertex") == "vertex" else None
    rows = []
    for i, p in enumerate(pos[: int(limit)]):
        row = {"index": int(i), "at": [round(float(x), 6) for x in p]}
        if labels and i < len(labels):
            row["label"] = str(labels[i])
        rows.append(row)
    return {**out, "n": int(pos.shape[0]), "cells": rows}


def _neighbors(vertex: int, files=None, record_id: str = "", _ctx=None) -> dict:
    from agent.graph_view import neighbors
    rex, _meta = _source_rex(files, record_id, _ctx=_ctx)
    return neighbors(rex, int(vertex))


def _reach(seeds, files=None, record_id: str = "", t: float = 1.0,
           limit: int = 25, _ctx=None) -> dict:
    from agent.graph_view import reach
    rex, meta = _source_rex(files, record_id, _ctx=_ctx)
    out = reach(rex, seeds, t=float(t), limit=int(limit))
    labels = meta.get("vertex_labels")
    if labels:
        for row in out["reached"]:
            if row["vertex"] < len(labels):
                row["label"] = str(labels[row["vertex"]])
    return out


def _join_complexes(files_a, files_b, how: str = "inner", limit: int = 50,
                    _ctx=None) -> dict:
    from agent.knowledge import join as _knowledge_join
    from rexgraph.joins import join as _join

    ka = _knowledge_join(*(files_a or []))
    kb = _knowledge_join(*(files_b or []))
    rex_a, rex_b = ka.rex(face_selection="none"), kb.rex(face_selection="none")
    labels_a = (ka.labels if hasattr(ka, "labels") else None) or \
        (getattr(rex_a, "_agent_meta", {}) or {}).get("vertex_labels") or []
    labels_b = (kb.labels if hasattr(kb, "labels") else None) or \
        (getattr(rex_b, "_agent_meta", {}) or {}).get("vertex_labels") or []

    joined, report = _join(rex_a, rex_b, how=how,
                           labels_r=labels_a, labels_s=labels_b)
    report["betti"] = [int(b) for b in joined.betti]
    report["chain_valid"] = not joined.self_loop_face_indices
    report["shared_labels"] = sorted(set(map(str, labels_a)) &
                                     set(map(str, labels_b)))[: int(limit)]
    return report


def _cells(files=None, record_id: str = "", grade: str = "both", limit: int = 100,
           positions: bool = True, _ctx=None) -> dict:
    from agent.cell_view import cells
    rex, meta = _source_rex(files, record_id, face_selection="auto", _ctx=_ctx)
    return cells(rex, grade=grade, labels=meta.get("vertex_labels"),
                 limit=int(limit), positions=bool(positions))


def _overview(files=None, record_id: str = "", cells: bool = True,
              limit: int = 100, positions: bool = True, _ctx=None) -> dict:
    from agent.overview import overview
    rex, meta = _source_rex(files, record_id, face_selection="auto", _ctx=_ctx)
    return overview(rex, labels=meta.get("vertex_labels"), cells=bool(cells),
                    limit=int(limit), positions=bool(positions))


def _tower(files=None, record_id: str = "", grade: int = 2, _ctx=None) -> dict:
    from rexgraph.tower import closure_at, tower_law
    rex, _meta = _source_rex(files, record_id, face_selection="auto", _ctx=_ctx)
    law = tower_law(rex)
    return {"mass": law["mass"], "trace": law["trace"], "moments": law["moments"],
            "law_holds": law["holds"], "law_residual": law["residual"],
            "closure": closure_at(rex, int(grade))}


def _apd(files=None, record_id: str = "", grade: int = 1, view: str = "local",
         limit: int = 200, _ctx=None) -> dict:
    from rexgraph.tower import apd
    rex, _meta = _source_rex(files, record_id, face_selection="auto", _ctx=_ctx)
    out = apd(rex, int(grade), view=view)
    if out.get("cells") and limit:
        out["shown"] = min(int(limit), len(out["cells"]))
        out["cells"] = out["cells"][:int(limit)]
    return out


def _face(files=None, record_id: str = "", relations=None, column=None, _ctx=None) -> dict:
    from rexgraph.faces import face_reading
    rex, _meta = _source_rex(files, record_id, face_selection="auto", _ctx=_ctx)
    if not relations:
        raise ValueError("relations is required: the relation indices to test")
    return face_reading(rex, relations, column)


def _render(files=None, record_id: str = "", dim: int = 3, limit: int = 200,
            fmt: str = "data", view: str = "structural", select=None, select_dim: int = 1,
            colour_by: str = "character", _ctx=None) -> dict:
    from agent.graph_view import render_payload
    rex, meta = _source_rex(files, record_id, face_selection="auto", _ctx=_ctx)
    payload = render_payload(rex, labels=meta.get("vertex_labels"),
                             dim=int(dim), limit=int(limit),
                             select=select or None, select_dim=int(select_dim))
    if fmt not in ("data", "svg", "both"):
        raise ValueError(f"fmt must be 'data', 'svg' or 'both', got {fmt!r}")
    if fmt == "data":
        return payload
    from agent.render_svg import render_svg

    drawn = {r["index"] for r in payload.get("relations", [])}
    drawing = {"svg": render_svg(payload, view=view, colour_by=colour_by),
               "view": view,
               "cells_drawn": len(drawn),
               "cells_total": int(rex.nE),
               "truncated": len(drawn) < int(rex.nE),
               "faces_drawn": sum(1 for f in payload.get("faces", [])
                                  if set(f["relations"]) <= drawn),
               "faces_total": int(rex.nF_hodge)}
    return drawing if fmt == "svg" else {**payload, "drawing": drawing}


#: every capability a model can call, each with the handler that answers it
def _courier_survey(hive: str, tags=None, limit: int = 100) -> dict:
    from agent.courier import CarrySpec, get_courier
    c = get_courier()
    spec = CarrySpec(tags=list(tags or []), limit=int(limit))
    return {"hive": hive, "records": c.survey(hive, carry=spec)}


def _courier_deliver(source: str, dest: str, tags=None, ids=None,
                     limit: int = 100) -> dict:
    """A trip between places the operator already bound.

    The destination is looked up, never built from the argument. A tool that took a url
    would let anything that can call a tool name a machine to send records to, which is
    the one thing a carrier must not decide for itself.
    """
    from agent.courier import CarrySpec, get_courier
    c = get_courier()
    if dest not in c.destinations():
        raise ValueError(
            f"no destination {dest!r}; this courier routes for "
            f"{', '.join(c.destinations()) or 'nothing yet'}. Register it first.")
    spec = CarrySpec(tags=list(tags or []), ids=list(ids or []), limit=int(limit))
    return c.deliver(source, dest, carry=spec)


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("rexgraph_join_sources",
         "Join ontology, annotation, structure and schema files into one relational "
         "complex on the cross-references the files themselves declare. Reports which "
         "entities were reached by more than one file and which identifiers collided.",
         {"files": _FILES,
          "face_selection": {"type": "string",
                             "description": "Face rule, or omit for the default."}},
         _join, required=["files"]),
    Tool("rexgraph_reason_ontology",
         "Check an ontology for classes that cannot have an instance, and report its "
         "equivalence classes and homology relative to its own hierarchy. A class is "
         "unsatisfiable when it descends from two classes asserted disjoint; the "
         "answer names both ancestor chains.",
         {"files": _FILES,
          "terms": {"type": "array", "items": {"type": "string"},
                    "description": "Extract the module for these terms."}},
         _reason, required=["files"]),
    Tool("rexgraph_enrich",
         "Which ontology terms a set of entities is concentrated in. Returns the "
         "exact hypergeometric p, the Benjamini-Hochberg q, and a persistence reading "
         "of the same complex.",
         {"files": _FILES,
          "study": {"type": "array", "items": {"type": "string"},
                    "description": "The entities to test."},
          "universe": {"type": "array", "items": {"type": "string"},
                       "description": "Background set; defaults to everything "
                                      "annotated."},
          "min_term_size": {"type": "integer"}},
         _enrich, required=["files", "study"]),
    Tool("rexgraph_release_series",
         "Read ontology releases in order and report what changed: terms introduced, "
         "terms merged into a survivor as against deleted outright, and which release "
         "was a structural surprise rather than ordinary growth.",
         {"files": _FILES,
          "labels": {"type": "array", "items": {"type": "string"},
                     "description": "A label per release, in upload order."}},
         _releases, required=["files"]),
    Tool("rexgraph_term_similarity",
         "How much two ontology terms share, as the overlap over ALL their common "
         "ancestors rather than the single most informative one.",
         {"files": _FILES,
          "term_a": {"type": "string"}, "term_b": {"type": "string"}},
         _similarity, required=["files", "term_a", "term_b"]),
    Tool("rexgraph_homology",
         "The homology of the joined complex: Betti numbers, the rank tower, and the "
         "gradient/curl/harmonic dimension at every grade.",
         {"files": _FILES},
         _homology, required=["files"]),
    Tool("rexgraph_structure_health",
         "Whether load drains through a joined structure or gets trapped circulating. "
         "Reports the Hodge split, the entities every path runs through, and any "
         "cycle holding harmonic content.",
         {"files": _FILES}, _health, required=["files"]),
    Tool("rexgraph_propagate",
         "Diffuse a seed of entities across the complex and report what it reaches, "
         "through the coupled field operator rather than by counting hops.",
         {"files": _FILES,
          "seed": {"type": "array", "items": {"type": "string"}},
          "t": {"type": "number", "description": "Diffusion time."},
          "limit": {"type": "integer"}},
         _propagate, required=["files", "seed"]),
    Tool("rexgraph_query_stored",
         "Select cells inside a stored complex by a structural invariant (kappa, a "
         "chi or phi channel, RCFE curvature) rather than by a recorded attribute.",
         {"record_id": {"type": "string"},
          "quantity": {"type": "string",
                       "description": "kappa, chi, phi, curvature, coherence_local"},
          "op": {"type": "string", "description": ">, >=, <, <=, ==, !=, between"},
          "threshold": {"type": "number"},
          "channel": {"type": "integer"}, "limit": {"type": "integer"}},
         _stored, required=["record_id", "quantity", "op", "threshold"],
         reads_files=False, wants_context=True),
    Tool("rexgraph_restrict",
         "Restrict a complex to the cells a structural invariant selects, and report "
         "what the restriction contains AND what it hides. The selection is CLOSED "
         "into a subcomplex, so the result is a complex whose boundary operators do "
         "not reference anything outside it: an operation handed it cannot reach past "
         "it, which a row filter cannot promise. relative_betti is the homology of "
         "what was excluded, a question a WHERE clause has no answer to.",
         {"files": _FILES,
          "record_id": {"type": "string",
                        "description": "A stored record, instead of files."},
          "quantity": {"type": "string",
                       "description": "kappa, chi, phi, curvature, coherence_local"},
          "op": {"type": "string", "description": ">, >=, <, <=, ==, !=, between"},
          "threshold": {"type": "number"},
          "channel": {"type": "integer"}, "limit": {"type": "integer"}},
         _restrict, required=["quantity", "op", "threshold"], wants_context=True),
    Tool("rexgraph_curvature",
         "Where a complex is strained: the per-face boundary curvature "
         "kappa_f = ||B1 B2[:, f]||, ranked, plus the Bianchi conservation residual "
         "over the same field. Reports which face carries the strain rather than a "
         "single number for the whole object.",
         {"files": _FILES,
          "record_id": {"type": "string",
                        "description": "A stored record, instead of files."},
          "limit": {"type": "integer"}},
         _curvature, wants_context=True),
    Tool("rexgraph_positions",
         "Structural coordinates for the cells of a complex. 'character' places each "
         "cell by its shares of the channel decomposition, so position says what a "
         "cell participates in and nearness means structurally alike. 'propagator' "
         "places it by heat reach from anchors, so distance means the complex does not "
         "carry signal there. Neither is a spectral embedding: a layout off the "
         "eigenvectors of L0 describes where a cut fell rather than the cells.",
         {"files": _FILES,
          "record_id": {"type": "string",
                        "description": "A stored record, instead of files."},
          "mode": {"type": "string", "description": "character or propagator"},
          "dim": {"type": "integer"},
          "grade": {"type": "string",
                    "description": "vertex or edge, for the character mode."},
          "limit": {"type": "integer"}},
         _positions, wants_context=True),
    Tool("rexgraph_neighbors",
         "The star of a vertex: every cell incident to it, returned as a CLOSED "
         "subcomplex rather than an adjacency list, so the answer can be analysed as a "
         "complex without repair.",
         {"vertex": {"type": "integer"},
          "files": _FILES,
          "record_id": {"type": "string",
                        "description": "A stored record, instead of files."}},
         _neighbors, required=["vertex"], wants_context=True),
    Tool("rexgraph_reach",
         "What a seed reaches and how strongly, by diffusing it through the complex. "
         "A hop limit answers 'within k steps', which is a property of the query; this "
         "answers how much arrives, which is a property of the structure, so there is "
         "no depth to choose and a cell reached by many paths outranks one reached by "
         "a thread.",
         {"seeds": {"type": "array", "items": {"type": "integer"}},
          "files": _FILES,
          "record_id": {"type": "string",
                        "description": "A stored record, instead of files."},
          "t": {"type": "number", "description": "Diffusion time."},
          "limit": {"type": "integer"}},
         _reach, required=["seeds"], wants_context=True),
    Tool("rexgraph_join",
         "Join two complexes through a vertex identification by label. What is joined "
         "is the RELATIONS, since they are primitive: two are the same relation when "
         "they distinguish the same vertex and reach the same others, read off the "
         "boundary so a branching relation is matched as one relation of arity k rather "
         "than as a set of pairs. The result is a complex, not a row set: faces are "
         "carried only where their whole boundary survived, so B1 B2 = 0 still holds.",
         {"files_a": {"type": "array", "items": {"type": "string"},
                      "description": "Handles for the left complex."},
          "files_b": {"type": "array", "items": {"type": "string"},
                      "description": "Handles for the right complex."},
          "how": {"type": "string", "description": "inner, left or outer"},
          "limit": {"type": "integer"}},
         _join_complexes, required=["files_a", "files_b"],
         file_args=("files_a", "files_b"), wants_context=True),
    Tool("rexgraph_cells",
         "One row per cell, carrying what that cell is: its share of each channel by "
         "NAME, its coherence against both the global and the local tower, its "
         "structural position, and for a relation its WHOLE boundary and arity rather "
         "than a source and a target. A branching relation is one row over k cells, not "
         "a pair with the rest missing.",
         {"files": _FILES,
          "record_id": {"type": "string",
                        "description": "A stored record, instead of files."},
          "grade": {"type": "string", "description": "vertex, edge or both"},
          "limit": {"type": "integer"},
          "positions": {"type": "boolean"}},
         _cells, wants_context=True),
    Tool("rexgraph_overview",
         "Everything worth saying about a complex, in one call: counts and the ARITY "
         "distribution, Betti and the rank tower, what it is made of per channel, the "
         "Hodge dimensions, where the curvature sits, and the cross-checks. Euler from "
         "the counts against Euler from Betti, and the harmonic dimensions against "
         "Betti, are identities over the integers, so a disagreement is a defect rather "
         "than a number to interpret.",
         {"files": _FILES,
          "record_id": {"type": "string",
                        "description": "A stored record, instead of files."},
          "cells": {"type": "boolean",
                    "description": "Include the per-cell rows."},
          "limit": {"type": "integer"},
          "positions": {"type": "boolean"}},
         _overview, wants_context=True),
    Tool("rexgraph_tower",
         "The mass tower: ||B_k||^2 per grade, the trace tower it determines, and the "
         "moments between grades. tr(L_k) = ||B_k||^2 + ||B_k+1||^2 is an identity, so "
         "one sequence fixes everything and a mismatch is a defect. The mass is "
         "EXTENSIVE (additive over disjoint components) where the normalised character "
         "is not, which is why it carries structure without the global coupling. Also "
         "reports closure at a grade, by the cheap necessary test (mass equality) and "
         "the actual one (every cell bounding exactly two).",
         {"files": _FILES,
          "record_id": {"type": "string",
                        "description": "A stored record, instead of files."},
          "grade": {"type": "integer", "description": "Grade to test closure at."}},
         _tower, wants_context=True),
    Tool("rex_apd",
         "Arity, parity and degree per cell: the three directions of a graded complex. "
         "Arity looks DOWN a grade (how many cells this one bounds), degree looks UP "
         "(how many contain it), parity reads the ORIENTATION as the sign product of "
         "the boundary. None is derived from another, so a cell can be wide and lonely "
         "or narrow and busy independently. Parity is only informative from grade 2 up: "
         "a B_1 column is canonically (-1, +share, ...), exactly one negative at every "
         "arity, so reversing a relation does not move it, while a face reads [1,1,1] "
         "balanced against [1,-1,-1] reversed. view='global' returns the means instead, "
         "and those means ARE the terms of the surface identity, so local and global "
         "here are one operator read at two scopes.",
         {"files": _FILES,
          "record_id": {"type": "string",
                        "description": "A stored record, instead of files."},
          "grade": {"type": "integer", "description": "Grade to read. 1 is relations."},
          "view": {"type": "string", "enum": ["local", "global"],
                   "description": "Per cell, or the means over the grade."},
          "limit": {"type": "integer"}},
         _apd, wants_context=True),
    Tool("rex_face",
         "Whether a set of relations bounds a face, with what signs, and why not. "
         "'bounds' when they carry exactly one cycle, so the column is determined up to "
         "an overall sign; 'open' when they are independent and enclose nothing; "
         "'degenerate' when they carry several cycles and are therefore not one face but "
         "a space of them. Reports the exact solved column, which relations run against "
         "their stored orientation, and the holonomy around the cycle. Pass `column` to "
         "check signs you already have: a wrong orientation is otherwise invisible, "
         "because the chain filter drops an invalid face silently, and this returns "
         "chain_valid with the exact residual and the column that would have worked.",
         {"files": _FILES,
          "record_id": {"type": "string",
                        "description": "A stored record, instead of files."},
          "relations": {"type": "array", "items": {"type": "integer"},
                        "description": "Relation indices to test as one face."},
          "column": {"type": "array", "items": {"type": "number"},
                     "description": "Signs to check, instead of solving for them."}},
         _face, wants_context=True),
    Tool("rexgraph_render",
         "Everything a renderer needs, with the geometry exact: rational plane "
         "coordinates, per-relation quadrance (1 + 1/(k-1) at arity k, so length "
         "carries arity), the spreads between relations that meet (squared sines, "
         "cos^2 = 1 - s), each relation's whole boundary, and the latent/filled/closed "
         "state that decides the face rule. Assembled together because positions, "
         "lengths and angles are three readings of the same boundary columns and must "
         "agree.",
         {"files": _FILES,
          "record_id": {"type": "string",
                        "description": "A stored record, instead of files."},
          "dim": {"type": "integer"}, "limit": {"type": "integer"},
          "fmt": {"type": "string", "enum": ["data", "svg", "both"],
                  "description": "'data' is the readings, 'svg' the drawing itself, "
                                 "'both' the readings with the drawing beside them."},
          "view": {"type": "string",
                   "enum": ["structural", "plane", "character", "embedded"],
                   "description": "'structural' is the graph drawing, cells placed by "
                                  "what they are near; 'plane' the exact rational layout "
                                  "from each cell's own star; 'character' the 3D channel "
                                  "embedding, which shows which cells are alike; "
                                  "'embedded' the coordinates the source carried, when "
                                  "it carried any. The last three place a cell by what it "
                                  "IS, so structurally identical cells share a point."},
          "select": {"type": "object",
                     "description": "Criteria on stored attributes. The matching cells "
                                    "are drawn forward and the rest back, so the picture "
                                    "still reads the same complex."},
          "select_dim": {"type": "integer",
                         "description": "Grade the selection applies to. 0 vertices, "
                                        "1 relations, 2 faces."},
          "colour_by": {"type": "string",
                        "description": "'character' is the derived k7 spectral colour, "
                                       "the only one that is not a decision. An "
                                       "attribute name colours categorically. A quantity "
                                       "name (curvature, arity, quadrance) ramps, which "
                                       "is a heat map, and reports its domain."}},
         _render, wants_context=True),
    Tool("rexgraph_courier_survey",
         "What a trip out of one of this courier's bound stores would consider, "
         "carrying nothing. Reports each record's id, version, tags and structure, so "
         "a caller can decide whether a trip is worth making before asking for one.",
         {"hive": {"type": "string",
                   "description": "A store this courier routes for, by the name it was "
                                  "bound under."},
          "tags": {"type": "array", "items": {"type": "string"},
                   "description": "Only records carrying any of these tags."},
          "limit": {"type": "integer"}},
         _courier_survey, required=["hive"], requires="admin", reads_files=False),
    Tool("rexgraph_courier_deliver",
         "Carry catalogued complexes from one bound store to another, or to a "
         "registered remote peer. Records the destination already holds are skipped by "
         "structural signature, so a repeat trip writes nothing. Both ends must already "
         "be registered on this courier; a destination cannot be named by url here.",
         {"source": {"type": "string", "description": "The store to carry from."},
          "dest": {"type": "string",
                   "description": "A bound store or a registered peer to carry to."},
          "tags": {"type": "array", "items": {"type": "string"},
                   "description": "Only records carrying any of these tags."},
          "ids": {"type": "array", "items": {"type": "string"},
                  "description": "Named records, instead of a tag match."},
          "limit": {"type": "integer"}},
         _courier_deliver, required=["source", "dest"], requires="admin",
         reads_files=False),
]}


def definitions(context: Context | None = None) -> list[dict]:
    """The tool definitions, derived from the registry that also dispatches.

    Narrowed to what this caller may actually run. Advertising a tool that would be
    refused invites a model to keep trying it, and tells anyone reading the list what
    exists at levels above them.
    """
    tools = TOOLS.values()
    if context is not None and not context.is_admin:
        tools = [t for t in tools if t.requires != "admin"]
    return [t.definition() for t in tools]


def call(name: str, context: Context | None = None, **arguments):
    """Run a tool by name, under one caller's context.

    Every call from outside the process passes a context, and the three things that
    have to hold before a handler runs are checked here rather than in nine handlers:
    the caller's level, the arguments the schema requires, and that any file named is
    one this workspace holds. A handler receives paths it is allowed to read, so it
    does not carry the question.

    `context=None` is the direct Python caller: the operator on their own machine,
    who is not being restricted from it.

    Raises rather than returning an error shape for an unknown name, because a model
    calling a tool that does not exist is a caller error and silently answering it
    would hide the drift this module exists to prevent.
    """
    tool = TOOLS.get(name)
    if tool is None:
        raise KeyError(
            f"no tool named {name!r}. Available: {', '.join(sorted(TOOLS))}")

    if context is not None and tool.requires == "admin" and not context.is_admin:
        raise PermissionError(f"{name} requires admin of workspace "
                              f"{context.workspace!r}")

    missing = [k for k in tool.required if k not in arguments]
    if missing:
        raise TypeError(f"{name} needs {', '.join(missing)}")

    unknown = [k for k in arguments if k not in tool.properties]
    if unknown:
        raise TypeError(f"{name} does not take {', '.join(sorted(unknown))}")

    if context is not None and tool.reads_files:
        from agent.server.handles import resolve_inputs
        arguments = dict(arguments)
        for name in tool.file_args:
            if arguments.get(name):
                arguments[name] = resolve_inputs(
                    context.workspace, arguments[name],
                    auth_enabled=context.auth_enabled)

    if tool.wants_context:
        arguments = dict(arguments, _ctx=context)
    return tool.handler(**arguments)
