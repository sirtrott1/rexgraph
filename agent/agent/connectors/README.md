# Connectors: the "adapt to any system" layer

A **connector** teaches the engine one system. It reads a source *read-only* and
returns a relational complex: the signed-incidence `B₁` topology, optional `B₂`
faces, and a `meta` dict of labels/edges/weights/modality. Everything else (the
math, the store, the analyses) follows from that. The contract is small, stable,
and the one thing a customer or the services team implements to onboard a new
system. Proprietary connectors live *outside* the core and depend only on this
seam, which is what keeps each integration auditable on its own and every paid
onboarding a **known, testable quantity**.

## The contract

```python
def read(self, source) -> tuple[rex, meta]
```

* `rex`: a built `RexGraph`, or the `(sources, targets)` edge-endpoint arrays.
* `meta`: required: `vertex_labels` (len `nV`), `edges` (len `nE`, `(src,dst)`
  pairs), `source`. Optional: `weights` (len `nE`), `modality` (len `nE`),
  `faces` (dense `B₂`). Build it with `BaseConnector.result(...)`, which enforces
  the length invariants so a malformed connector fails at the source, not deep in
  the engine.

`capabilities()` advertises what the connector can supply (`weights`, `modality`,
`faces`, and the URI `schemes` it serves). Invariants every connector must keep:
read-only, structure-only (never row/cell values), and `∂²=0` when it emits faces.

## Write one

Copy the template and fill the two TODOs:

```bash
cp agent/connectors/template.py agent/connectors/my_system.py
python -m agent.connectors.template          # see the worked example run
```

## Validate it

The harness turns an integration into a pass/fail report: contract shape, builds
in the engine, chain condition `∂²=0`, Betti/signature, RCDB round-trip, a
read-only probe, and capability consistency:

```python
from agent.connectors.validate import validate_connector
assert validate_connector(MyConnector(), my_source).ok
```

```bash
python -m agent.connectors.validate my.module:MyConnector [source]
```

## Open one by URI

```python
from agent.connectors import open_connector
conn = open_connector("postgresql://host/db")          # -> SQLConnector
conn = open_connector("sqlite:///x", with_weights=True)
```

## What ships

| Connector | Shape | Schemes | In-sandbox |
|-----------|-------|---------|------------|
| `SQLConnector` | relational DBs | sqlite, postgresql, mysql, mariadb, oracle, mssql | yes (SQLite) |
| `WarehouseConnector` | cloud warehouses | snowflake, bigquery, redshift, databricks | yes (SQLite stand-in) |
| `DocumentConnector` | document/NoSQL | mongodb | yes (in-memory) |
| `SemanticConnector` | RDFS/OWL | ontology, rdf, owl | yes (in-memory triples) |
| `GenericConnector` | edge/table long-tail | edges, table | yes (in-memory / CSV) |
| `GraphConnector` | property graphs | neo4j, bolt | yes, shape only (live needs driver) |
| `StreamConnector` | streaming | kafka, pulsar | yes, shape only (live needs client) |

The warehouse/graph/stream live paths need a driver or broker from the host
environment; their *shape* is validated here against in-memory stand-ins, so
implementing the live read is a wiring task against an already-passing contract.
