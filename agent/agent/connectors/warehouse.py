"""SQL reflection for Snowflake, BigQuery, Redshift and Databricks.

WarehouseConnector extends SQLConnector's supported schemes. Install the
corresponding SQLAlchemy driver before opening a warehouse URI.
"""
from __future__ import annotations

from . import Capabilities
from .sql import SQLConnector

_WAREHOUSE_SCHEMES = ("snowflake", "bigquery", "redshift", "databricks",
                      "postgresql", "sqlite")


class WarehouseConnector(SQLConnector):
    def capabilities(self) -> Capabilities:
        base = super().capabilities()
        return Capabilities(weights=base.weights, modality=base.modality,
                            faces=base.faces, schemes=_WAREHOUSE_SCHEMES)
