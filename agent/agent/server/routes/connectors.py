"""
agent.server.routes.connectors: the Connector seam over HTTP.

One route group over :mod:`agent.connectors.service`: list what can be connected
(with per-scheme driver status for the "not configured" UX), validate an
integration, read a source read-only, and ingest its structure into the RCDB.
SQL, warehouses, Mongo, ontologies, graphs, and streams all flow through the
same path, the one the ``rexgraph-connect`` CLI uses.

A request identifies its source by a saved-connection ``name`` (resolved via the
SecretStore, credentials never returned), an inline ``uri``, or a bare
``scheme`` (for in-memory shapes, whose structure rides in ``source``).
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from agent.connectors import service as svc

router = APIRouter(prefix="/v1/connectors")


def _resolve(body: dict) -> str:
    from agent.server.dbguard import check_db_uri
    if body.get("name") and not body.get("uri"):
        from agent.server.scope import secret_store
        try:
            uri = secret_store().get(body["name"])
        except KeyError as exc:
            raise HTTPException(404, f"No saved connection '{body['name']}'") from exc
        check_db_uri(uri)
        return uri
    uri = body.get("uri") or body.get("scheme")
    if not uri:
        raise HTTPException(400, "Provide a saved connection 'name', a 'uri', or a 'scheme'")
    check_db_uri(uri)   # no-op for bare in-memory scheme names (no '://')
    return uri


@router.get("")
async def list_connectors():
    """List connectors, their capabilities, and per-scheme driver availability."""
    return {"connectors": svc.list_connectors()}


@router.post("/read")
async def read_source(body: dict = Body(...)):
    """Build the complex read-only and return a structural summary (no storage)."""
    uri = _resolve(body)
    try:
        return svc.read(uri, source=body.get("source"),
                        **svc.weight_kwargs(uri, bool(body.get("weights"))))
    except Exception as e:                       # noqa: BLE001
        raise HTTPException(400, f"read failed: {e}") from e


@router.post("/validate")
async def validate_source(body: dict = Body(...)):
    """Run the validation harness and return the pass/fail report."""
    uri = _resolve(body)
    try:
        report = svc.validate(uri, source=body.get("source"),
                              **svc.weight_kwargs(uri, bool(body.get("weights"))))
    except Exception as e:                       # noqa: BLE001
        raise HTTPException(400, f"validate failed: {e}") from e
    return {
        "connector": report.connector,
        "ok": report.ok,
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                   for c in report.checks],
    }


@router.post("/ingest")
async def ingest_source(body: dict = Body(...)):
    """Build + persist the source's structure into the app's RCDB."""
    uri = _resolve(body)
    record_id = body.get("id") or body.get("store_id")
    if not record_id:
        raise HTTPException(400, "Provide an 'id' to store as")
    from agent.rcdb import default_store as _store
    try:
        return svc.ingest(uri, record_id, store=_store(), source=body.get("source"),
                          tags=body.get("tags") or [],
                          **svc.weight_kwargs(uri, bool(body.get("weights"))))
    except Exception as e:                       # noqa: BLE001
        raise HTTPException(400, f"ingest failed: {e}") from e
