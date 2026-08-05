"""
agent.connectors: the "adapt to any system" layer.

A **connector** turns a source (a live DB, a dump, a stream, an in-memory graph,
an ontology) into a relational complex: the signed incidence ``B₁`` topology,
optional ``B₂`` faces, and a ``meta`` dict of labels/edges/weights/modality. The
contract is defined by :class:`agent.interfaces.Connector` and is deliberately
tiny, stable, and **read-only** - it is the one thing a customer or the services
team implements to teach the engine a new system.

This package ships:

  * :class:`BaseConnector` - a fill-in-the-blanks base that supplies a default
    :meth:`~agent.interfaces.Connector.capabilities` and a ``result`` helper that
    assembles and length-checks the standard ``meta`` dict, so every connector
    emits the same shape.
  * the standards-family adapters (SQL / document / semantic / generic …), each
    covering a *shape* of system rather than a single vendor.

Customer/proprietary connectors live *outside* the core, depending only on the
seam - never editing the engine. That isolation is what keeps every adapter
auditable on its own and every paid integration a known, testable quantity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

from ..interfaces import Capabilities, Connector
from .build import faces_to_csc, to_rexgraph

__all__ = ["Capabilities", "Connector", "BaseConnector", "ConnectorError",
           "faces_to_csc", "to_rexgraph", "open_connector"]

# URI scheme -> (module, class). Mirrors rcdb.open_store: one entry point that
# routes a source URI to the connector for its shape. Each scheme resolves to
# exactly one connector; vendor drivers are the per-source delta, not new code.
_SCHEME_MAP = {
    "sqlite": ("sql", "SQLConnector"),
    "postgresql": ("sql", "SQLConnector"),
    "postgres": ("sql", "SQLConnector"),
    "mysql": ("sql", "SQLConnector"),
    "mariadb": ("sql", "SQLConnector"),
    "oracle": ("sql", "SQLConnector"),
    "mssql": ("sql", "SQLConnector"),
    "snowflake": ("warehouse", "WarehouseConnector"),
    "bigquery": ("warehouse", "WarehouseConnector"),
    "redshift": ("warehouse", "WarehouseConnector"),
    "databricks": ("warehouse", "WarehouseConnector"),
    "mongodb": ("document", "DocumentConnector"),
    "ontology": ("semantic", "SemanticConnector"),
    "rdf": ("semantic", "SemanticConnector"),
    "owl": ("semantic", "SemanticConnector"),
    "neo4j": ("graph", "GraphConnector"),
    "bolt": ("graph", "GraphConnector"),
    "kafka": ("stream", "StreamConnector"),
    "pulsar": ("stream", "StreamConnector"),
    "edges": ("generic", "GenericConnector"),
    "table": ("generic", "GenericConnector"),
}


def open_connector(uri: str, **kwargs):
    """Return a connector instance for ``uri``'s scheme (e.g. ``postgresql://…``
    -> :class:`~agent.connectors.sql.SQLConnector`). Extra kwargs pass to the
    connector (e.g. ``open_connector("sqlite:///x", with_weights=True)``).
    Unknown schemes raise :class:`ConnectorError`. Modules are imported lazily
    to avoid an import cycle with this package."""
    import importlib
    scheme = uri.split("://", 1)[0] if "://" in uri else uri
    scheme = scheme.split("+", 1)[0].lower()          # strip SQLAlchemy driver
    entry = _SCHEME_MAP.get(scheme)
    if entry is None:
        known = ", ".join(sorted(_SCHEME_MAP))
        raise ConnectorError(f"no connector for scheme {scheme!r}; known: {known}")
    mod_name, cls_name = entry
    mod = importlib.import_module(f".{mod_name}", __name__)
    return getattr(mod, cls_name)(**kwargs)


class ConnectorError(ValueError):
    """Raised when a connector produces output inconsistent with the contract
    (e.g. label/edge counts that disagree with the topology)."""


class BaseConnector:
    """Convenience base for connectors. Subclasses implement :meth:`read` and,
    if they can supply more than topology, override :meth:`capabilities`.

    ``read`` must return ``(rex, meta)`` where ``rex`` is either a built
    ``RexGraph`` or the ``(sources, targets)`` edge arrays, and ``meta`` is the
    dict described on :class:`agent.interfaces.Connector`. Use :meth:`result`
    to build ``meta`` so the required keys and length invariants are enforced
    in one place.

    ``BaseConnector`` writes nothing, reads nothing on its own, and holds no
    state - it only shapes and checks a connector's output.
    """

    #: Subclasses override to advertise weights/modality/faces and URI schemes.
    CAPABILITIES: Capabilities = Capabilities()

    def capabilities(self) -> Capabilities:
        return self.CAPABILITIES

    def read(self, source: Any) -> tuple[Any, dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError(
            "A connector must implement read(source) -> (rex, meta). "
            "See agent.connectors.template for a worked skeleton."
        )

    #### meta assembly
    @staticmethod
    def result(
        rex: Any,
        *,
        vertex_labels: Sequence[str],
        edges: Sequence[tuple[str, str]],
        source: str,
        weights: Sequence[float] | None = None,
        modality: Sequence[dict[str, Any]] | None = None,
        faces: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Assemble the standard ``(rex, meta)`` return value, enforcing the
        contract's length invariants so a malformed connector fails loudly at
        the source rather than deep in the engine.

        Invariants checked here:
          * ``vertex_labels`` is present and non-empty (``nV``).
          * ``edges`` count is ``nE``; each edge is a ``(src, dst)`` pair.
          * ``weights`` / ``modality``, when given, have length ``nE``.
        """
        labels = list(vertex_labels)
        edge_pairs = [tuple(e) for e in edges]
        nV, nE = len(labels), len(edge_pairs)
        if nV == 0:
            raise ConnectorError("vertex_labels is empty; a complex needs ≥1 vertex")
        for e in edge_pairs:
            if len(e) != 2:
                raise ConnectorError(f"edge {e!r} is not a (src, dst) pair")
        meta: dict[str, Any] = {
            "vertex_labels": labels,
            "edges": edge_pairs,
            "source": str(source),
            "nV": nV,
            "nE": nE,
        }
        if weights is not None:
            w = list(weights)
            if len(w) != nE:
                raise ConnectorError(
                    f"weights length {len(w)} != nE {nE}")
            meta["weights"] = w
        if modality is not None:
            mod = list(modality)
            if len(mod) != nE:
                raise ConnectorError(
                    f"modality length {len(mod)} != nE {nE}")
            meta["modality"] = mod
        if faces is not None:
            meta["faces"] = faces
        if extra:
            meta.update(extra)
        return rex, meta
