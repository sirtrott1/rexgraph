"""
agent.server.routes.rcdb: the Relational Complex Database over HTTP.

Stores every analysed complex as a first-class record and lets you query
the database *by structure* (Betti, coherence, voids), not just by id.
The backend is chosen per-deployment via the ``REXGRAPH_RCDB_URI`` env
var (default: a file store under the config dir), so the same API runs on
SQLite locally and Postgres in production.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Body, HTTPException

from agent.server.artifacts import complex_file, plain

router = APIRouter(prefix="/v1/db")

def _store():
    """The process-wide default store. Resolution lives in `agent.rcdb.default_store`
    so HTTP and non-HTTP callers share one store instead of two resolvers."""
    from agent.rcdb import default_store
    return default_store()


def _rex_from_body(body: dict):
    sid = body.get("session_id")
    if sid:
        try:
            from agent.server.app import get_store
            s = get_store().get(sid)
            if s is not None and s.current() is not None:
                return s.current(), f"session:{sid}"
        except Exception:
            pass
    text = body.get("text")
    if text and text.strip():
        from agent.auto import auto_rex
        return auto_rex(text), "text"
    return None, "no session_id or text"


@router.get("/info")
async def db_info():
    """Backend + aggregate stats + a type/tag breakdown for the overview."""
    st = _store()
    try:
        stats = st.stats()
        recs = st.list(limit=10 ** 9)
        tag_counts: dict = {}
        sources: dict = {}
        for r in recs:
            for t in (r.signature.get("tags") or []) or ["untagged"]:
                tag_counts[t] = tag_counts.get(t, 0) + 1
            src = r.signature.get("source") or "unknown"
            sources[src] = sources.get(src, 0) + 1
        return {"uri_scheme": st.backend, **stats,
                "by_tag": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
                "by_source": dict(sorted(sources.items(), key=lambda x: -x[1]))}
    except Exception as e:
        raise HTTPException(500, f"DB error: {e}") from e


@router.post("/put")
async def db_put(body: dict = Body(...)):
    """Store a complex (from a session or text) as a record.

    Body: {id?, session_id?|text?, tags?[]}. Returns the record signature.
    """
    rex, source = _rex_from_body(body)
    if rex is None:
        raise HTTPException(400, f"No complex: {source}")
    rec_id = body.get("id") or f"rc_{int(time.time()*1000)}"
    meta = getattr(rex, "_agent_meta", {}) or {}
    try:
        rec = _store().put(rec_id, rex, meta=meta, tags=body.get("tags") or [])
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Store failed: {e}") from e
    return {"stored": True, "source": source, **rec.to_dict()}


@router.get("/list")
async def db_list(limit: int = 100, offset: int = 0):
    return {"records": [r.to_dict() for r in _store().list(limit=limit, offset=offset)]}


@router.post("/query")
async def db_query(body: dict = Body(...)):
    """Structural query. Body accepts any of: min_nV/max_nV, min_nE/max_nE,
    min_nF, min_betti1/max_betti1, min_kappa/max_kappa, chain_valid,
    has_voids, tags_any[], tags_all[], source, labels_any[], labels_all[], limit.

    The body is splatted into the predicate, so an unsupported key is a malformed
    request rather than a server fault: the store raises TypeError naming the keys it
    accepts, and that comes back as a 400 carrying the same list.
    """
    limit = int(body.pop("limit", 100)) if isinstance(body, dict) else 100
    try:
        recs = _store().query(limit=limit, **(body or {}))
    except TypeError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}") from e
    return {"count": len(recs), "records": [r.to_dict() for r in recs]}


@router.get("/get/{rec_id}")
async def db_get(rec_id: str):
    rec = _store().get_record(rec_id)
    if rec is None:
        raise HTTPException(404, "Record not found")
    return rec.to_dict()


@router.get("/export/{rec_id}")
async def db_export(rec_id: str, format: str = "safetensors"):
    """Download the stored complex in any container the library writes.

    safetensors (the storage form), .rex, hdf5 or zarr. The complex itself, not a
    summary of it.
    """
    rex = _store().get(rec_id)
    if rex is None:
        raise HTTPException(404, "Record not found")
    return complex_file(rex, rec_id, format)


@router.post("/similar")
async def db_similar(body: dict = Body(...)):
    """Find stored complexes structurally similar to a query.

    Body: {session_id?|text?|id?, top_k?}. Returns matches with a 0-1
    similarity score (show as a percentage) and how many concepts/tables
    they share.
    """
    from agent.rcdb import find_similar
    st = _store()
    exclude = None
    if body.get("id"):
        rex = st.get(body["id"])
        rec = st.get_record(body["id"])
        from agent.rcdb import _labels_of
        labels = _labels_of(rec, rex) if rec else []
        exclude = body["id"]
    else:
        rex, source = _rex_from_body(body)
        labels = (getattr(rex, "_agent_meta", {}) or {}).get("vertex_labels", []) if rex else []
    if rex is None:
        raise HTTPException(400, "Provide id, session_id, or text")
    matches = find_similar(st, rex, labels, top_k=int(body.get("top_k", 10)),
                           exclude_id=exclude)
    return {"matches": matches}


@router.post("/record-work")
async def db_record_work(body: dict = Body(...)):
    """Record one state of the platform's own work.

    Body: {kind: 'pipeline-run'|'conversation', labels, edges?, lineage_id?, tags?,
    when?}. The lineage is one record holding a TemporalRex: each call appends a
    step and stores it as the next version, so the state has both a version and a
    position in time.

    Calling this route records. The `record_work` workspace setting governs what the
    platform records BY ITSELF as you use it; a client posting here is asking, and
    an explicit request does not need the automatic switch to be on.
    """
    from agent import work_recorder as wr
    from agent.server.scope import effective_workspace
    labels = body.get("labels") or []
    if not labels:
        raise HTTPException(400, "Provide 'labels' (stages or turns)")
    kind = body.get("kind", "pipeline-run")
    if kind not in wr.KINDS:
        raise HTTPException(400, f"kind must be one of: {', '.join(wr.KINDS)}")
    lineage_id = body.get("lineage_id") or body.get("id")
    if not lineage_id:
        raise HTTPException(400, "Provide 'lineage_id' (the run or session this belongs to)")
    info = wr.record(kind, labels, lineage_id=lineage_id,
                     workspace=effective_workspace(body.get("workspace", "")),
                     edges=body.get("edges"), tags=body.get("tags"),
                     when=body.get("when"), force=True)
    if info is None:
        return {"recorded": False, "reason": "need at least two states to relate"}
    return {"recorded": not info.get("unchanged"), **info}


@router.get("/recorded")
async def db_recorded(workspace: str = "default", kind: str = None):
    """The lineages recorded from this platform's own work, newest first."""
    from agent import work_recorder as wr
    return {"lineages": wr.recorded(workspace=workspace, kind=kind)}


@router.get("/recorded/{lineage_id}/at")
async def db_recorded_at(lineage_id: str, when: float):
    """The recorded state current at a moment: its position in the temporal rex and
    the complex reconstructed there."""
    from agent import work_recorder as wr
    step, rex = wr.state_at(lineage_id, when)
    if step is None:
        raise HTTPException(404, "No recorded state at or before that moment")
    from agent.rcdb import structural_signature
    return {"lineage_id": lineage_id, "step": step, "when": when,
            "signature": structural_signature(rex)}


@router.get("/lineage/{lineage_id}")
async def db_lineage(lineage_id: str):
    """Version history + drift trajectory for a lineage."""
    from agent.rcdb import drift
    return drift(_store(), lineage_id)


@router.post("/cluster")
async def db_cluster(body: dict = Body(...)):
    """Group stored complexes into structural families by coherence.
    Body: {tags_any?, threshold?}."""
    from agent.rcdb import cluster_complexes
    return cluster_complexes(_store(), tags_any=body.get("tags_any"),
                             threshold=float(body.get("threshold", 0.7)))


@router.post("/compare")
async def db_compare(body: dict = Body(...)):
    """Compare two stored complexes (e.g. schema v1 vs v2). Body: {a, b}."""
    from agent.rcdb import compare
    if not body.get("a") or not body.get("b"):
        raise HTTPException(400, "Provide 'a' and 'b' record ids")
    result = compare(_store(), body["a"], body["b"])
    if result is None:
        raise HTTPException(404, "One or both records not found")
    return result


@router.post("/cells/{rec_id}")
async def query_cells(rec_id: str, body: dict = Body(...)):
    """Select cells inside a stored complex by structural invariant.

    `/query` filters WHICH records match; this filters WHICH CELLS inside one. The
    predicates run on quantities the complex computes (kappa, a chi or phi channel,
    RCFE curvature) rather than on stored attributes, so a selection like "vertices
    whose coherence is below 0.6" needs nothing to have been recorded in advance.

    Body: {"where": [{"quantity", "op", "threshold", "threshold_high"?, "channel"?},
    ...], "combine": "and" | "or", "limit"?}. Several clauses compose with `combine`.
    """
    from rexgraph.graph import RexGraph

    store = _store()
    rex = store.get(rec_id)
    if rex is None:
        raise HTTPException(404, f"no record {rec_id!r}")
    clauses = body.get("where") or []
    if not clauses:
        raise HTTPException(400, "Provide 'where' as a list of predicate clauses")

    combine = str(body.get("combine", "and")).lower()
    if combine not in ("and", "or"):
        raise HTTPException(400, "combine must be 'and' or 'or'")

    mask = None
    grades = set()
    for c in clauses:
        try:
            m = rex.select(c["quantity"], c["op"], float(c["threshold"]),
                           float(c.get("threshold_high", 0.0)),
                           channel=int(c.get("channel", 0)))
        except KeyError as e:
            raise HTTPException(400, f"clause is missing {e}") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        grades.add(len(m))
        if len(grades) > 1:
            raise HTTPException(
                400, "these predicates select on different grades (vertices and "
                     "edges), so they cannot be combined into one mask")
        mask = m if mask is None else (RexGraph.select_and(mask, m)
                                       if combine == "and"
                                       else RexGraph.select_or(mask, m))

    import numpy as _np
    n_selected = int(_np.asarray(mask).sum())
    meta = store.get_record(rec_id).meta or {}
    return {
        "record": rec_id,
        "n_selected": n_selected,
        "n_cells": int(len(mask)),
        "grade": "vertex" if len(mask) == int(rex.nV) else "edge",
        "cells": plain(rex.selected(mask, labels=meta.get("vertex_labels"),
                                     limit=int(body.get("limit", 200)))),
        "combine": combine,
    }


@router.get("/explain/{rec_id}")
async def explain_cell(rec_id: str, dim: int = 0, idx: int = 0):
    """Everything the complex knows about one cell.

    EXPLAIN, in the query sense: not a plan, but the cell's position in the field.
    A vertex reports its coherence, star character, dominant and discrepant channel
    and its neighbours; an edge reports what lies below, above and lateral to it, its
    character and its effective resistance.
    """
    store = _store()
    rex = store.get(rec_id)
    if rex is None:
        raise HTTPException(404, f"no record {rec_id!r}")
    limit = int(rex.nV) if int(dim) == 0 else int(rex.nE)
    if not 0 <= int(idx) < limit:
        raise HTTPException(
            400, f"index {idx} is out of range for grade {dim} ({limit} cells)")
    out = plain(rex.explain(int(dim), int(idx)))
    meta = store.get_record(rec_id).meta or {}
    labels = meta.get("vertex_labels") or []
    if int(dim) == 0 and int(idx) < len(labels):
        out["label"] = labels[int(idx)]
    return out


@router.delete("/{rec_id}")
async def db_delete(rec_id: str):
    return {"deleted": _store().delete(rec_id)}
