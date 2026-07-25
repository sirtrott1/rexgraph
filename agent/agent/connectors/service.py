"""
agent.connectors.service - the one place a source becomes a (validated,
optionally stored) complex.

The CLI (`rexgraph-connect`), the HTTP route (`/api/v1/connectors`), and any
future desktop/exe build all call *these* functions - never the registry or the
harness directly - so there is exactly one implementation of "onboard a source"
to build on. Pure Python: no argparse, no FastAPI, no printing.

    list_connectors()                      -> what can I connect, and is the driver here?
    driver_status(scheme)                  -> is scheme's driver importable? hint if not
    read(uri, source=None)                 -> build the complex read-only, return a summary
    validate(uri, source=None)             -> run the harness, return the report
    ingest(uri, store_uri, record_id, …)   -> build + persist structure into an RCStore

For URI-addressed live sources (SQL, warehouses, Mongo) the connection URI *is*
the source, so ``source`` defaults to ``uri``. For in-memory shapes (ontologies,
edge lists, graph/stream stand-ins) pass a scheme as ``uri`` and the structure as
``source``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import _SCHEME_MAP, open_connector, to_rexgraph
from .validate import validate_connector, ValidationReport

# Per-scheme driver probe: (module to import, pip hint). None => no driver needed
# (the in-memory/structure path). This powers the app's "not configured" UX so a
# missing warehouse driver shows an install hint instead of a dead button.
_WAREHOUSE_HINT = "pip install 'rexgraph-agent[warehouse]'"

# Schemes for which a cardinality-weight pull is meaningful (SQL + warehouses).
# Shared by the CLI and the HTTP route so "weights" means one thing everywhere.
WEIGHTABLE_SCHEMES = frozenset({
    "sqlite", "postgresql", "postgres", "mysql", "mariadb", "oracle", "mssql",
    "snowflake", "bigquery", "redshift", "databricks",
})


def weight_kwargs(uri: str, want_weights: bool) -> Dict[str, Any]:
    """`{'with_weights': True}` only when weights are requested *and* the scheme
    supports them; otherwise empty (so non-SQL connectors aren't handed an
    argument their constructor doesn't accept)."""
    return ({"with_weights": True}
            if want_weights and _scheme_of(uri) in WEIGHTABLE_SCHEMES else {})
_DRIVER_PROBE: Dict[str, Tuple[Optional[str], str]] = {
    "sqlite": ("sqlalchemy", "pip install sqlalchemy"),
    "postgresql": ("sqlalchemy", "pip install sqlalchemy psycopg2-binary"),
    "postgres": ("sqlalchemy", "pip install sqlalchemy psycopg2-binary"),
    "mysql": ("sqlalchemy", "pip install sqlalchemy pymysql"),
    "mariadb": ("sqlalchemy", "pip install sqlalchemy pymysql"),
    "oracle": ("sqlalchemy", "pip install sqlalchemy oracledb"),
    "mssql": ("sqlalchemy", "pip install sqlalchemy pyodbc"),
    "snowflake": ("snowflake.sqlalchemy", _WAREHOUSE_HINT),
    "bigquery": ("sqlalchemy_bigquery", _WAREHOUSE_HINT),
    "redshift": ("sqlalchemy_redshift", _WAREHOUSE_HINT),
    "databricks": ("databricks.sqlalchemy", _WAREHOUSE_HINT),
    "mongodb": ("pymongo", "pip install pymongo"),
    "neo4j": ("neo4j", "pip install neo4j  (or pass an in-memory structure)"),
    "bolt": ("neo4j", "pip install neo4j  (or pass an in-memory structure)"),
    "kafka": ("confluent_kafka", "pip install confluent-kafka  (or pass a structure)"),
    "pulsar": ("pulsar", "pip install pulsar-client  (or pass a structure)"),
    "ontology": (None, ""),
    "rdf": (None, ""),
    "owl": (None, ""),
    "edges": (None, ""),
    "table": (None, ""),
}


def _scheme_of(uri: str) -> str:
    scheme = uri.split("://", 1)[0] if "://" in uri else uri
    return scheme.split("+", 1)[0].lower()


def driver_status(scheme: str) -> Dict[str, Any]:
    """Whether the driver for ``scheme`` is importable, plus a pip hint if not.
    In-memory shapes report available=True (no driver needed)."""
    scheme = scheme.lower()
    probe, hint = _DRIVER_PROBE.get(scheme, ("", ""))
    if probe is None:                       # in-memory shape, no driver
        return {"available": True, "hint": ""}
    if not probe:                           # unknown scheme
        return {"available": False, "hint": ""}
    try:
        __import__(probe)
        return {"available": True, "hint": ""}
    except Exception:
        return {"available": False, "hint": hint}


def list_connectors() -> List[Dict[str, Any]]:
    """The registry, grouped by connector: schemes served, advertised
    capabilities, and driver availability for each scheme."""
    by_conn: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for scheme, (mod, cls) in sorted(_SCHEME_MAP.items()):
        key = (mod, cls)
        if key not in by_conn:
            inst = open_connector(scheme)     # scheme alone selects the connector
            caps = inst.capabilities()
            by_conn[key] = {
                "connector": cls,
                "module": f"agent.connectors.{mod}",
                "schemes": [],
                "capabilities": caps.summary(),
                "supports": {
                    "topology": True, "weights": caps.weights,
                    "modality": caps.modality, "faces": caps.faces,
                },
            }
        st = driver_status(scheme)
        by_conn[key]["schemes"].append({
            "scheme": scheme, "driver_available": st["available"],
            "driver_hint": st["hint"],
        })
    return list(by_conn.values())


def _summary(g: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": meta.get("source"),
        "nV": g.nV, "nE": g.nE, "nF": g.nF,
        "betti": list(g.betti),
        "chain_valid": bool(g.chain_valid),
        "weighted": meta.get("weights") is not None,
        "modality": meta.get("modality") is not None,
    }


def read(uri: str, source: Any = None, **kwargs) -> Dict[str, Any]:
    """Build the complex read-only and return a summary (no storage). For live
    URI sources, ``source`` defaults to the URI itself."""
    connector = open_connector(uri, **kwargs)
    rex, meta = connector.read(uri if source is None else source)
    g = to_rexgraph(rex, meta)
    return _summary(g, meta)


def validate(uri: str, source: Any = None, store_uri: str = "memory://",
             **kwargs) -> ValidationReport:
    """Run the validation harness for ``uri``'s connector against the source."""
    connector = open_connector(uri, **kwargs)
    return validate_connector(connector, uri if source is None else source,
                              store_uri=store_uri)


def ingest(uri: str, record_id: str, *, store: Any = None,
           store_uri: Optional[str] = None, source: Any = None,
           tags: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
    """Build the complex and persist its *structure* into an RCStore. The only
    writing operation, and it writes solely to the host's own store - either a
    pre-opened ``store`` (e.g. the app's singleton) or one opened from
    ``store_uri``."""
    from agent.rcdb import open_store

    if store is None:
        if not store_uri:
            raise ValueError("ingest needs a 'store' object or a 'store_uri'")
        store = open_store(store_uri)

    connector = open_connector(uri, **kwargs)
    rex, meta = connector.read(uri if source is None else source)
    g = to_rexgraph(rex, meta)
    store.put(record_id, g, meta=getattr(g, "_agent_meta", None),
              tags=list(tags or []))
    out = _summary(g, meta)
    out.update({"stored_as": record_id, "store": store_uri or type(store).__name__})
    return out
