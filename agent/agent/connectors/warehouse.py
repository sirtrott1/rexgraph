"""agent.connectors.warehouse: cloud data warehouses (Snowflake / BigQuery /
Redshift / Databricks).

Same *shape* as SQL: it is the SQL connector with the warehouse dialects added.
The reflection code is unchanged - installing the vendor's SQLAlchemy driver is
the only delta. Validated against SQLite as a structural stand-in here; point it
at a live warehouse URI in the host environment (where the driver is installed).
"""
from __future__ import annotations
from .sql import SQLConnector
from . import Capabilities

_WAREHOUSE_SCHEMES = ("snowflake", "bigquery", "redshift", "databricks",
                      "postgresql", "sqlite")


class WarehouseConnector(SQLConnector):
    def capabilities(self) -> Capabilities:
        base = super().capabilities()
        return Capabilities(weights=base.weights, modality=base.modality,
                            faces=base.faces, schemes=_WAREHOUSE_SCHEMES)
