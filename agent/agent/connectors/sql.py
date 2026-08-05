"""
agent.connectors.sql: the SQL connector (the "out of the box" flagship).

One adapter for the whole SQL *shape*: it wraps the engine's existing
SQLAlchemy reflection, so a single connector covers Postgres, MySQL/MariaDB,
Oracle, SQL Server, and SQLite via SQLAlchemy's dialects: the driver is the
only per-vendor delta, not the code.

    read(conn_str) -> (rex, meta)

Vertices are tables, edges are foreign keys (child -> parent), faces are genuine
co-participations (junction/associative entities). FK modality
(nullable / identifying / on-delete) rides along as ``meta['modality']``.
Cardinality weights (which enable data-forced strain) are opt-in via
``SQLConnector(with_weights=True)`` because they require reading live row
counts: approximate, catalog-based counts, never row data.

Read-only throughout: schema metadata + aggregate counts only.
"""

from __future__ import annotations

from typing import Any

from . import BaseConnector, Capabilities, ConnectorError

# SQLAlchemy URL schemes this one adapter covers (dialect = per-vendor delta).
_SQL_SCHEMES = (
    "sqlite", "postgresql", "postgres", "mysql", "mariadb",
    "oracle", "mssql",
)


def _redact(conn_str: str) -> str:
    """Strip credentials from a connection string so it can go in an error message.

    A SQLAlchemy URL carries them as `scheme://user:password@host/db`. Everything
    between `//` and the last `@` is replaced, which covers passwords containing `@`
    or `:`. Strings with no credentials are returned unchanged.
    """
    scheme, sep, rest = conn_str.partition("://")
    if not sep or "@" not in rest:
        return conn_str
    _, _, tail = rest.rpartition("@")
    return f"{scheme}://***@{tail}"


class SQLConnector(BaseConnector):
    """Reflect a live SQL database into a relational complex."""

    def __init__(self, with_weights: bool = False):
        # Weights need live row counts; off by default keeps the connector
        # pure-structure. Advertised capabilities track this instance's config
        # so what's advertised is exactly what's emitted.
        self.with_weights = with_weights

    def capabilities(self) -> Capabilities:
        return Capabilities(
            weights=self.with_weights,
            modality=True,     # always derivable from the FK catalog
            faces=True,        # co-participation faces where junctions exist
            schemes=_SQL_SCHEMES,
        )

    def read(self, source: Any) -> tuple[Any, dict[str, Any]]:
        conn_str = str(source)
        from ..schema_complex import reflect_schema, schema_to_rex

        model = reflect_schema(conn_str)
        rex, sm_meta = schema_to_rex(model)
        if rex is None:
            raise ConnectorError(
                "no foreign-key edges reflected - nothing to form a complex "
                f"from (source={_redact(conn_str)})")

        labels: list[str] = list(sm_meta["vertex_labels"])
        edges: list[tuple[str, str]] = [tuple(e) for e in sm_meta["edges"]]
        modality = self._modality(model, edges)
        weights = self._weights(conn_str, edges) if self.with_weights else None

        from ..secrets import mask_uri
        return self.result(
            rex,                       # already a faced RexGraph
            vertex_labels=labels,
            edges=edges,
            source=mask_uri(conn_str),
            modality=modality,
            weights=weights,
            extra={"dialect": conn_str.split(":", 1)[0]},
        )

    #### modality: align per-edge with the emitted edge order
    @staticmethod
    def _modality(model, edges: list[tuple[str, str]]) -> list[dict[str, Any]]:
        by_pair: dict[tuple[str, str], Any] = {}
        for fk in model.foreign_keys:
            by_pair.setdefault((fk.from_table, fk.to_table), fk)
        out: list[dict[str, Any]] = []
        for a, b in edges:
            fk = by_pair.get((a, b))
            out.append({
                "nullable": bool(getattr(fk, "nullable", True)) if fk else True,
                "identifying": bool(getattr(fk, "identifying", False)) if fk else False,
                "on_delete": getattr(fk, "on_delete", "") if fk else "",
            })
        return out

    #### weights: catalog row counts as a cardinality proxy
    @staticmethod
    def _weights(conn_str: str, edges: list[tuple[str, str]]) -> list[float] | None:
        from ..schema_complex import list_tables
        try:
            counts = {t["table"]: float(t.get("rows") or 0)
                      for t in list_tables(conn_str, with_counts=True)}
        except Exception:
            return None
        # weight each FK edge by the child (many-side) row count, the
        # cardinality pressure the data puts on that relationship.
        return [max(1.0, counts.get(a, 1.0)) for a, _ in edges]
