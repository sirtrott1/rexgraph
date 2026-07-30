"""
agent.server.routes.ontology - diagnose RDFS/OWL ontologies as complexes.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

router = APIRouter(prefix="/v1/ontology")


@router.post("/analyze")
async def analyze_ontology(body: dict = Body(...)):
    """Diagnose an ontology from triples.

    Body: {triples: [[s, p, o], ...], store_id?, lineage_id?, tags?}.
    Returns the descriptive readout (subsumption hierarchy, bounded
    definitions, inconsistencies) and optionally stores the complex.
    """
    from agent import ontology_complex as oc
    triples = body.get("triples")
    if not triples:
        raise HTTPException(400, "Provide 'triples' as [[subject, predicate, object], ...]")
    try:
        model = oc.parse_rdf([tuple(t) for t in triples])
    except Exception as e:
        raise HTTPException(400, f"Could not parse triples: {e}")
    report = oc.diagnose_ontology(model)

    if body.get("lineage_id") or body.get("store_id"):
        try:
            rex, meta = oc.ontology_to_rex(model)
            if rex is not None:
                from agent.rcdb import default_store as _store
                tags = (body.get("tags") or []) + ["ontology"]
                if body.get("lineage_id"):
                    from agent.rcdb import put_version
                    report["version"] = put_version(_store(), body["lineage_id"],
                                                     rex, meta=meta, tags=tags)
                else:
                    _store().put(body["store_id"], rex, meta=meta, tags=tags)
                    report["stored_as"] = body["store_id"]
        except Exception as e:
            report["store_error"] = str(e)
    return report
