# RCDB

RCDB stores native relational complexes and typed records with version history.
It supports memory, local, SQLite and object storage, retains exact payload
values, and provides explicit publication, copy and migration operations.
Applications can query metadata and structural signatures without reconstructing
every saved complex.

RCDB depends on RexGraph, NumPy and safetensors. Agent, RCQL and System can use
its stores independently. Base native record operations do not require SciPy;
older matrix index and document search APIs use the `scipy` extra.

## Installation

From the repository root, `make install` includes RCDB with SQL, object storage
and cryptography dependencies. Use `make python` to open the installed
interpreter. If Core is already installed, run `python -m pip install ./rcdb`
for the base package. See the [installation guide](../README.md#installation)
and [dependency guide](../DEPENDENCIES.md).

## Saving and reading a version

```python
from contextlib import closing
from rexgraph import RexGraph
from rcdb import LocalStore

network = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
with closing(LocalStore("records")) as store:
    saved = store.put("network", network, analytics=False)
    restored = store.get_version("network", saved.version)
    assert tuple(restored.betti) == (1, 1, 0)
```

`put` appends a version, and native stores retain previous versions and deletion
markers. Recreating a deleted identifier therefore allocates the next version.
The example disables derived analytics and closes the store when it finishes.

`as_of` selects by transaction time, and `valid_at` selects by the record's
valid time. Point in time queries select one visible version per identifier
before applying predicates. History collections retain their own explicit
version contract.

## Backends

`open_store(uri)` selects a backend by URI scheme:

| URI or interface | Purpose |
| --- | --- |
| `memory://` | Native records in process memory |
| `local:///path` | Checked local records and history |
| `sqlite:///path` | Native records and transactional SQL queries |
| Other SQLAlchemy URLs | Compatibility SQL storage |
| `open_object_store(file_or_memory_uri)` | Native object storage for a new prefix |
| `s3://`, `gs://`, `az://` | Provider adapters with an explicit native or legacy contract |
| `file:///path`, `rex:///path` | Read only legacy layouts |
| `auto://path` | Detect an existing layout or create a local store |

Cloud providers need their own driver and credentials. A generic remote
filesystem does not automatically provide native atomic publication. Local
stores share a directory lock between handles and POSIX processes; SQLite uses
transactions. Other SQL dialects and legacy writers retain their compatibility
behavior. See the [backend reference](REFERENCE.md#backends) for consistency
and publication limits.

## Records, queries and analysis

Native record codecs support complexes, temporal objects, exact values,
declarations, provenance and selected artifact types. RCQL result storage is
registered explicitly when required. Metadata and tags can be queried across
record kinds; graph predicates apply to records containing complexes.

Structural signatures summarize selected graph measurements for indexing.
Derived analytics and optional search features have their own dependencies;
the stored native payload remains the record's state identity. Protected
search can restrict exposed fields and use configured keyed tokens.

| Capability | Interface |
| --- | --- |
| Versioned records | `put`, `get_version`, `read_record`, `history` |
| Metadata queries | `query`, `stats`, tags and structural predicates |
| Governed history | `configure_security`, `commit_mutation`, `verify_commits` |
| Record transfer | `record_packet`, `copy_record`, checked copy receipts |
| Migration | `plan_migration`, `migrate_batch`, resumable receipts |
| Corpus reads | `corpus_snapshot`, exact record response readings |
| Maintenance | Replay checkpoints, staging retention plans and checked application |

The [API reference](REFERENCE.md) documents selectors, codec inventories,
search restrictions and maintenance policies.

## Governed history

A store can require mutations with checked commit history:

```python
from contextlib import closing
from rcdb import open_store

with closing(open_store("memory://")) as store:
    store.configure_security(require_commits=True)
    committed = store.commit_mutation("sample", network, expected_version=0)
    selected = store.read_record("sample", version=committed.version)
    assert store.verify_commits("sample")
```

Expected version guards allow a caller to refuse a stale write. A failed native
publication preserves the preceding state; an unknown publication outcome
requires reopening and verification before continuing. Record history, change
cursors and payload digests describe different aspects of the store and have
separate contracts. See [governed history](REFERENCE.md#governed-history).

## Portable record versions

`record_packet(store, record_id, version=...)` captures a selected record version,
including its payload, metadata, validity and source identity. `copy_record`
is the shared transfer operation and returns a checked receipt. The destination
applies its own publication and integrity policy; it does not inherit the
source's encryption keys or runtime hooks.

Agent's `RexClient` and courier use the same selected record contract for remote
transfer. See the [portable packet reference](REFERENCE.md#portable-record-versions),
[cross store copy contract](REFERENCE.md#cross-store-copies-and-identity-receipts)
and [Agent transfer guide](../agent/RECORD_TRANSFER.md).

## Testing and license

From the repository root, run `make test-rcdb TEST_ARGS='-rs'` after installing
a development profile. RCDB is distributed under the [Apache License 2.0](LICENSE).
