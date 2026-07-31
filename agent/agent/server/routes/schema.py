"""agent.server.routes.schema: topological diagnosis of database schemas."""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from agent import schema_complex as sc

router = APIRouter(prefix="/v1/schema")


def _parse_schema(body: dict):
    """Parse a schema model from the request body (shared by analyze/strain)."""
    if body.get("ddl"):
        return sc.parse_schema_ddl(body["ddl"], dialect=body.get("dialect"))
    if body.get("spec"):
        return sc.parse_schema_json(body["spec"])
    if body.get("mongo"):
        return sc.infer_mongo_schema(body["mongo"])
    if body.get("mongo_connection"):
        return sc.reflect_mongo(body["mongo_connection"], body.get("db") or "test",
                                sample=int(body.get("sample", 100)))
    if body.get("connection"):
        return sc.reflect_schema(body["connection"])
    raise HTTPException(
        400, "Provide 'ddl' (+optional 'dialect'), 'spec', 'mongo', "
             "'connection', or 'mongo_connection'")


@router.post("/analyze")
async def analyze_schema(body: dict = Body(...)):
    """Diagnose a schema's actual topology.

    Provide one of:
      * ``ddl``        - CREATE TABLE / FOREIGN KEY SQL text
      * ``spec``       - JSON schema {tables:[{name,columns,primary_key,foreign_keys}]}
      * ``connection`` - a SQLAlchemy URL to reflect a live database

    Optional: ``store_id`` (persist the schema complex in the RCDB) and
    ``tags``. Returns circular-dependency chains, the Hodge hierarchy/
    tension split, implied-missing relations (voids), and hub tables.
    """
    try:
        model = _parse_schema(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read schema: {e}")

    report = sc.diagnose_schema(model)

    if body.get("lineage_id"):
        try:
            rex, meta = sc.schema_to_rex(model)
            if rex is not None:
                from agent.rcdb import default_store as _store
                from agent.rcdb import version_if_changed
                info = version_if_changed(_store(), body["lineage_id"], rex, meta=meta,
                                          tags=(body.get("tags") or []) + ["schema"])
                report["version"] = info
        except Exception as e:
            report["store_error"] = str(e)
    elif body.get("store_id"):
        try:
            rex, meta = sc.schema_to_rex(model)
            if rex is not None:
                from agent.rcdb import default_store as _store
                _store().put(body["store_id"], rex, meta=meta,
                             tags=(body.get("tags") or []) + ["schema"])
                report["stored_as"] = body["store_id"]
        except Exception as e:
            report["store_error"] = str(e)

    return report


@router.post("/lint")
async def lint_schema(body: dict = Body(...)):
    """Data-model lint: per-relation character + anomalies + conflict tables."""
    try:
        model = _parse_schema(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read schema: {e}")
    return sc.relation_lint(model)


@router.post("/faces")
async def schema_faces_route(body: dict = Body(...)):
    """Explore how the face-selection algorithm changes the schema's geometry.

    Same tables & foreign keys, different definition of "what counts as a
    co-participation" -> different curl/harmonic reading. Returns, per mode
    ('coparticipation', 'autoface', 'promote', 'none'), the face count, Betti
    numbers, and the Hodge split - a side-by-side of the schema's topological
    options (e.g. an FK triangle is persistent-harmonic under 'coparticipation'
    but bounded-curl under 'autoface').
    """
    try:
        model = _parse_schema(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read schema: {e}")
    return {"face_options": sc.explore_schema_faces(model),
            "selections": list(sc.SCHEMA_FACE_SELECTIONS)}


@router.post("/strain")
async def schema_strain_route(body: dict = Body(...)):
    """Data-forced strain: weight the schema by real data magnitudes and
    measure the geometric strain the data imposes.

    Body: a schema (ddl/spec/mongo/connection) plus either explicit
    ``weights`` {"from->to": cardinality} or a ``connection`` to pull live
    cardinality from. Returns the heat map (how much / where), per-relation
    attribution (who / what), and effective root causes (how many / coupled).
    """
    try:
        model = _parse_schema(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read schema: {e}")

    weights = body.get("weights")
    row_counts = None
    if not weights and body.get("connection"):
        try:
            weights, row_counts = sc.pull_cardinality_stats(body["connection"], model, approximate=bool(body.get("approximate", False)))
        except Exception as e:
            raise HTTPException(400, f"Could not pull statistics: {e}")

    result = sc.schema_strain(model, weights=weights)
    if row_counts is not None:
        result["row_counts"] = row_counts
    return result
