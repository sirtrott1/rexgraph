"""
agent.server.routes.rcdb - the Relational Complex Database over HTTP.

Stores every analysed complex as a first-class record and lets you query
the database *by structure* (Betti, coherence, voids), not just by id.
The backend is chosen per-deployment via the ``REXGRAPH_RCDB_URI`` env
var (default: a file store under the config dir), so the same API runs on
SQLite locally and Postgres in production.
"""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response

from agent.rcdb import open_store

router = APIRouter(prefix="/v1/db")

_STORE = None


def _store():
    global _STORE
    if _STORE is None:
        uri = os.environ.get("REXGRAPH_RCDB_URI")
        if not uri:
            root = os.path.expanduser("~/.config/rexgraph/rcdb")
            uri = "file://" + root
        _STORE = open_store(uri)
    return _STORE


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
        raise HTTPException(500, f"DB error: {e}")


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
    except Exception as e:
        raise HTTPException(500, f"Store failed: {e}")
    return {"stored": True, "source": source, **rec.to_dict()}


@router.get("/list")
async def db_list(limit: int = 100, offset: int = 0):
    return {"records": [r.to_dict() for r in _store().list(limit=limit, offset=offset)]}


@router.post("/query")
async def db_query(body: dict = Body(...)):
    """Structural query. Body accepts any of: min_nV/max_nV, min_nE/max_nE,
    min_nF, min_betti1/max_betti1, min_kappa/max_kappa, chain_valid,
    has_voids, tags_any[], tags_all[], source, limit."""
    limit = int(body.pop("limit", 100)) if isinstance(body, dict) else 100
    try:
        recs = _store().query(limit=limit, **(body or {}))
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")
    return {"count": len(recs), "records": [r.to_dict() for r in recs]}


@router.get("/get/{rec_id}")
async def db_get(rec_id: str):
    rec = _store().get_record(rec_id)
    if rec is None:
        raise HTTPException(404, "Record not found")
    return rec.to_dict()


@router.get("/export/{rec_id}")
async def db_export(rec_id: str):
    """Download the stored complex as a .safetensors file."""
    from agent.rcdb import serialize_complex
    rex = _store().get(rec_id)
    if rex is None:
        raise HTTPException(404, "Record not found")
    data = serialize_complex(rex)
    return Response(content=data, media_type="application/octet-stream",
                    headers={"Content-Disposition":
                             f'attachment; filename="{rec_id}.safetensors"'})


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
    """Store a pipeline run or conversation as a complex in the RCDB, so the
    platform's own work is queryable structure. Body:
    {kind: 'pipeline-run'|'conversation', labels, edges?, id?, lineage_id?, tags?}.
    """
    from agent import lineage_adapters as la
    kind = body.get("kind", "pipeline-run")
    labels = body.get("labels") or []
    if not labels:
        raise HTTPException(400, "Provide 'labels' (stages or turns)")
    build = la.conversation_to_rex if kind == "conversation" else la.run_to_rex
    rex, meta = build(labels, edges=body.get("edges"))
    if rex is None:
        raise HTTPException(400, "Need at least two connected nodes to form a complex")
    tags = (body.get("tags") or []) + [kind]
    if body.get("lineage_id"):
        from agent.rcdb import put_version
        info = put_version(_store(), body["lineage_id"], rex, meta=meta, tags=tags)
        return {"stored": info["id"], "version": info}
    rid = body.get("id") or f"{kind}-{int(__import__('time').time())}"
    _store().put(rid, rex, meta=meta, tags=tags)
    return {"stored": rid}


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


@router.delete("/{rec_id}")
async def db_delete(rec_id: str):
    return {"deleted": _store().delete(rec_id)}
