"""
agent.server.routes.dbmanager - enterprise database manager.

Save and test connections to any SQL database (or MongoDB), browse a live
schema, and import it into the RCDB as a diagnosed relational complex.

Security note: connection URIs may contain credentials. They are stored
server-side under the config dir and are ALWAYS masked in responses. A
production deployment should back this with a real secrets manager
(Vault/KMS); the storage here is behind a single interface so that swap
is a drop-in.
"""

from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse, urlunparse

from fastapi import APIRouter, Body, HTTPException

router = APIRouter(prefix="/v1/dbmanager")

# Connection secrets live behind the SecretStore interface (file by default,
# env-reference / Vault as drop-ins via REXGRAPH_SECRETS_URI).
_SECRETS = None


def _secrets():
    global _SECRETS
    if _SECRETS is None:
        from agent.secrets import open_secret_store
        _SECRETS = open_secret_store()
    return _SECRETS


def _mask(uri: str) -> str:
    from agent.secrets import mask_uri
    return mask_uri(uri)


def _resolve(body: dict) -> str:
    """Get a concrete URI from a saved connection name or an inline uri."""
    from agent.server.dbguard import check_db_uri
    if body.get("name") and not body.get("uri"):
        try:
            uri = _secrets().get(body["name"])
        except KeyError:
            raise HTTPException(404, f"No saved connection '{body['name']}'")
        check_db_uri(uri)
        return uri
    if body.get("uri"):
        check_db_uri(body["uri"])
        return body["uri"]
    raise HTTPException(400, "Provide a saved connection 'name' or a 'uri'")


@router.get("/connections")
async def list_connections():
    return {"connections": _secrets().list()}


@router.post("/connections")
async def save_connection(body: dict = Body(...)):
    name = (body.get("name") or "").strip()
    uri = (body.get("uri") or "").strip()
    if not name or not uri:
        raise HTTPException(400, "Provide 'name' and 'uri'")
    from agent.server.dbguard import check_db_uri
    check_db_uri(uri)
    _secrets().put(name, uri, body.get("kind", "sql"))
    return {"saved": name, "uri": _mask(uri)}


@router.delete("/connections/{name}")
async def delete_connection(name: str):
    return {"deleted": _secrets().delete(name)}


@router.post("/test")
async def test_connection(body: dict = Body(...)):
    uri = _resolve(body)
    kind = body.get("kind", "sql")
    try:
        if kind == "mongo":
            from pymongo import MongoClient
            MongoClient(uri, serverSelectionTimeoutMS=4000).server_info()
        else:
            from sqlalchemy import create_engine
            create_engine(uri).connect().close()
        return {"ok": True, "uri": _mask(uri)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "uri": _mask(uri)}


@router.post("/tables")
async def list_tables(body: dict = Body(...)):
    uri = _resolve(body)
    try:
        from agent import schema_complex as sc
        if body.get("kind") == "mongo":
            m = sc.reflect_mongo(uri, body.get("db") or "test",
                                 sample=int(body.get("sample", 50)))
            return {"tables": [{"table": t.name, "columns": len(t.columns),
                                "foreign_keys": sum(1 for fk in m.foreign_keys
                                                    if fk.from_table == t.name),
                                "primary_key": t.primary_key} for t in m.tables]}
        return {"tables": sc.list_tables(uri, with_counts=bool(body.get("counts", True)))}
    except Exception as e:
        raise HTTPException(400, f"Could not list tables: {e}")


@router.post("/import")
async def import_schema(body: dict = Body(...)):
    """Reflect a live schema, diagnose it, and store it in the RCDB."""
    uri = _resolve(body)
    try:
        from agent import schema_complex as sc
        if body.get("kind") == "mongo":
            model = sc.reflect_mongo(uri, body.get("db") or "test",
                                     sample=int(body.get("sample", 100)))
        else:
            model = sc.reflect_schema(uri)
        report = sc.diagnose_schema(model)
        store_id = body.get("store_id") or (body.get("name") or "imported") + "-schema"
        rex, meta = sc.schema_to_rex(model)
        if rex is not None:
            from agent.rcdb import default_store as _store
            _store().put(store_id, rex, meta=meta,
                         tags=(body.get("tags") or []) + ["schema", "imported"])
            report["stored_as"] = store_id
        return report
    except Exception as e:
        raise HTTPException(400, f"Import failed: {e}")


@router.post("/strain")
async def connection_strain(body: dict = Body(...)):
    """Measure data-forced strain on a saved connection (reflect + pull live
    cardinality + compute strain), keeping credentials server-side."""
    uri = _resolve(body)
    try:
        from agent import schema_complex as sc
        model = sc.reflect_schema(uri)
        weights, counts = sc.pull_cardinality_stats(uri, model, approximate=bool(body.get("approximate", False)))
        result = sc.schema_strain(model, weights=weights)
        result["row_counts"] = counts
        return result
    except Exception as e:
        raise HTTPException(400, f"Strain analysis failed: {e}")


@router.post("/ddl")
async def generate_ddl(body: dict = Body(...)):
    """Generate cycle-safe CREATE TABLE DDL from a JSON schema spec."""
    from agent import schema_complex as sc
    spec = body.get("spec")
    if not spec:
        raise HTTPException(400, "Provide a 'spec'")
    model = sc.parse_schema_json(spec)
    return {"ddl": sc.export_schema_ddl(model, dialect=body.get("dialect", "generic"))}
