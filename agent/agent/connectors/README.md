# Connectors

Connectors read source structure and return a relational complex with metadata.
They expose relationships, labels, optional weights and faces through one
interface. A connector reads its source without modifying it.

## Installation

From the repository root, `make install` includes the standard connector
dependencies. Use `sh install.sh --extras warehouse` for cloud warehouse drivers.
Other live sources may need their own database driver or broker client.

```sh
.venv/bin/rexgraph-connect list
.venv/bin/python -I -m agent.connectors.template
```

The template runs an example with local data. See the
[installation guide](../../../README.md#installation) for environment setup.

## Contract

```python
def read(self, source) -> tuple[rex, meta]:
    ...
```

`rex` is a `RexGraph` or a pair of source and target index arrays. Required
metadata consists of `vertex_labels`, `edges` and `source`. Labels match the
vertex count and edges match the relation count. Optional `weights` and
`modality` entries match the relation count; `faces` supplies an upper boundary.
`BaseConnector.result` checks these metadata lengths.

`capabilities()` declares weights, modality, faces and URI schemes. Connectors
that emit faces must preserve `B1 B2 = 0`. Source values are excluded from the
structural connector contract.

## Custom connectors

Implement `BaseConnector.read` and declare `CAPABILITIES`. The source template
is [template.py](template.py). From the repository root:

```sh
cp agent/agent/connectors/template.py agent/agent/connectors/my_system.py
```

Validate the implementation against a representative source:

```python
from agent.connectors.validate import validate_connector

report = validate_connector(MyConnector(), my_source)
assert report.ok
```

The report checks metadata, complex construction, boundary closure, structural
signature, RCDB persistence, source write behavior and capability declarations.
A local fixture establishes behavior for that fixture; live access still needs
validation against the configured service.

## Available connectors

| Connector | Sources | Live dependencies |
| --- | --- | --- |
| `SQLConnector` | SQLite, PostgreSQL, MySQL, MariaDB, Oracle, SQL Server | SQLAlchemy and the selected database driver |
| `WarehouseConnector` | Snowflake, BigQuery, Redshift, Databricks | Agent `warehouse` extra |
| `DocumentConnector` | MongoDB | pymongo |
| `SemanticConnector` | RDF and OWL declarations | Source specific |
| `GenericConnector` | Edge lists and tables | Source specific |
| `GraphConnector` | Neo4j | neo4j |
| `StreamConnector` | Kafka and Pulsar | Broker client |

`open_connector(uri, **options)` selects a connector by URI scheme. Connection
options are passed to the selected implementation.
