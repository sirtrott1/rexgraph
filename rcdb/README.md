# RCDB

A versioned store for native relational complexes and typed records. RCDB depends on RexGraph,
not on RCQL, Agent or System. Those packages use the same store contracts.

## Install and run

From the repository root, install the core before the store:

```sh
python -m pip install .
python -m pip install './rcdb[sql,objectstore,crypto]'
```

The extras are optional. The base package provides native memory and local stores,
and read only file and rex stores.
These record operations do not require SciPy. The older index matrix and
document search APIs require `rexgraph-rcdb[scipy]`. See the
[dependency profiles](../DEPENDENCIES.md) for the Core and Agent distinction.

The index scalar accession reading also runs without SciPy.
`index.record_response_exact(index, terms, reading="share")` returns nonzero
record scores as Fractions. `index.record_response(index, terms)` returns
the same scores rounded once per record, together with record IDs. Both read
integer incidence through Core. Share divides each contribution by its
relation width minus one; existence omits that denominator. Both divide by
the seed's incidence degree. Pass a collection of complete vocabulary terms.
The channel profile and multiple step readings still require the SciPy extra.

`store.corpus_snapshot(as_of=None, valid_at=None, signature_fields=None)`
captures one visible version per record, including RexStore append log entries.
Its `response(terms, reading="existence", exact=True)` returns aligned IDs,
versions and scores with an accession snapshot digest. Restricted signature
fields exclude metadata labels before degree calculation.

TemporalRex payloads retain optional vertex labels per snapshot in their
sealed state metadata. Conversation captures therefore retain term names and
stable primary turn IDs through the same record readers. A temporal record does not persist a live conversation gate.

```python
from rcdb import open_store
from fractions import Fraction
from rexgraph.graph import RexGraph

rex = RexGraph.from_hypergraph(
    [0, 3, 5], [0, 1, 2, 1, 2],
    w_E=[Fraction(2, 3), Fraction(5, 7)],
)
store = open_store("memory://").configure_security(require_commits=True)
first = store.commit_mutation("sample", rex, expected_version=0)
snapshot = store.read_record("sample", version=first.version)
assert snapshot.value.w_E[0] == Fraction(2, 3)
assert store.verify_commits("sample")
assert len(store.commit_history("sample")) == 1
store.close()
```



## Backends

`open_store(uri)` selects the backend by scheme. Each backend exposes `RCStore`.

| uri | backend | for |
|---|---|---|
| `memory://` | `MemoryStore` | native records in process memory |
| `local:///path` | `LocalStore` | checked local history, tombstones and cursors |
| `file:///path` | `FileStore` | read only legacy file layout |
| `rex:///path` | `RexStore` | read only legacy packed layout |
| `sqlite:///path` | `SQLStore` | native records and transactional SQL queries |
| other SQLAlchemy URL | `SQLStore` | compatibility SQL storage |
| `s3://`, `gs://`, `az://` | object adapter | existing legacy readers or an explicitly configured native provider |
| `open_object_store(file_or_memory_uri)` | `NativeObjectStore` for new prefixes | native object storage |

`register_backend(scheme, opener)` adds one from outside. `available_backends()`
lists what is registered. `auto://path` detects an existing layout and uses
LocalStore for a new directory. It refuses conflicting layout markers.
`REXGRAPH_RCDB_URI` overrides the process default, `auto://<config>/rcdb`.

Records are bitemporal: `put` appends a version rather than replacing one, a
version is closed by the arrival of its successor, and `as_of` and `valid_at`
select against transaction time and valid time separately. Native stores retain
deleted versions and tombstones; recreating an ID allocates the next version.
History collections refuse time selectors. Point in time queries select one
version per ID before applying predicates.

`query` filters metadata, tags, record types and structural signatures. Native
memory, local and object stores build requested metadata indexes lazily; SQLite
uses checked SQL projections. `limit=0` validates selectors and returns no records.
Generic records participate in metadata queries; graph predicates select complexes.
`stats()` reports `count`/`n_records`, `n_versions`, graph cell totals and mean
curvature. Graph totals and averages exclude generic records.

LocalStore serializes complete writes and `read_transaction()` through a directory
lock. Independent threads and POSIX processes share that lock; platforms without
`fcntl` provide thread exclusion. SQLite writes use one transaction and
`BEGIN IMMEDIATE`; reads pin one snapshot. Other SQL dialects retain their
compatibility contracts. MemoryStore history lasts for that store instance.
Close a store when finished; closed native handles refuse further record operations.

A failed native write preserves the preceding publication. If the provider cannot
determine whether publication succeeded, the handle refuses reuse until reopened
and verified. Activity hooks run after publication and are not durable change cursors.

## Interoperating with an application

RCDB does not import the agent, or anything else built on it. What an application
adds arrives through four callables:

- `configure_hooks(activity=None, scope=None, privacy=None, similarity=None)`

  `activity` records a change into the application's own feed, `scope` narrows the
  default store to what a request may see, `privacy` projects metadata before it is
  stored, and `similarity` replaces the default scoring. Each is a plain callable,
  so nothing here depends on where it came from.

With none of them set the store still stores.



## The signature

Every `put` computes a structural signature and stores it beside the payload:
shape, betti numbers, chain validity, coherence, and a label sample. That is what
`query` filters on without deserializing a complex, and what `find_similar` and
`cluster_complexes` read.

The measurements live in `rcdb.analytics`.



## Protected search

The canonical snapshot reconstructs records, so it is not a security boundary: it
writes labels in the clear. Sealing records and shipping a plaintext term index
beside them protects nothing.

`rcdb.protected_index` builds a separate, disposable relation whose vocabulary is
fixed width tokens. A persisted index carries neither a plaintext term nor a
plaintext record id, and exact lookup still answers.

- `IndexPolicy(modes, key_id)` with a mode per accession kind:
  `public` and `structural` are domain separated SHA-256, so a token cannot collide
  across kinds but a guessed term can still be recomputed; `keyed` is HMAC under a
  key the caller must hold, which is the mode that resists enumeration; `none` is
  not indexed.
- `build_search_relation(records, policy, keys)` / `build_search_relation_from_tokens(...)`
- `save_search_relation(path, relation)` / `load_search_relation(path)`

Identities live only in an in memory resolver, never in the file, so a persisted
relation returns tokens and refuses to name records. The key arrives through an
`IndexKeyProvider` rather than as bytes, which is what lets an application scope a
key identifier per tenant.

**A search index is derived, disposable state.** Rebuild it rather than migrate it.



## Sealing records

- `configure_security(key_id=None, keys=None, mutation_policy=None, verifiers=None, transition_signer=None, lineage_signer=None, signature_mode="public", metadata_fields=None, require_commits=False)`

Optional: stores without a key write plaintext record envelopes. Introducing a
key preserves reads of earlier plaintext and legacy records.

With `key_id` and a `KeyProvider`, payloads are sealed with AES GCM. Opening
decides by the ENVELOPE rather than by configuration, so records written before a
key was introduced stay readable beside records written after it, a sealed blob in
an unconfigured store is a refusal rather than a plaintext read of ciphertext, and
a wrong key is a refusal rather than a library exception.

Sealing the payload is not enough on its own. A signature is a description of the
data, so a store that seals its records and writes the full signature beside them
has described what it sealed. `signature_mode` keeps the shape and the invariants
under `structural`, or only what is needed to address a record under `minimal`, and
`metadata_fields` is an allow list over what metadata persists.

**This is not database at rest confidentiality.** Record identifiers, the activity
log, SQL identity columns, file and directory names, and backend metadata are all
outside the envelope. Those are a separate identity and storage problem, and a
deployment that needs them covered needs disk or filesystem encryption underneath.

## Record ownership and compatibility

Every new payload carries a `RecordEnvelope` binding its store ID, literal record
ID, version, object kind, codec, semantic digest and payload bytes. Record reads
check this binding even with `verify=False`. A valid payload from another address
cannot replace the selected record. `copy_record` creates a new destination
envelope while preserving the value.

Content digests detect corruption; encryption and signed mutation policies supply
authentication. Public metadata remains subject to the configured signature and
metadata policy. Store identity is independent of graph identity.

`put_prepared` verifies native state and ownership without reconstructing a graph.
Prepared static graphs require native v10 state; decode and save older payloads
again before using this API. Supplied analytics remain the caller's responsibility.
Existing unbound records remain readable, but cannot replace an enveloped publication.

Missing ownership declarations, unreadable snapshots and damaged authoritative
journals refuse opening or reading. They do not create a new identity or return
an empty store. Compaction preserves record bindings; recompression skips bound
payloads whose bytes would require a new metadata publication.

## Checked local journals

Ordinary opens refuse incomplete journal tails and complete corruption. For an
existing checked FileStore or RexStore journal, inspect and repair explicitly:

Use the store identity recorded when the store was created. FileStore uses
`index.rexlog`; RexStore uses `records.log`.

```python
from rcdb.journal import LocalJournal

journal = LocalJournal("./legacy/index.rexlog", store_id=known_store_id)
status = journal.inspect()
if status.torn:
    journal.repair_torn_tail()
```

Repair removes only an incomplete final block after validating its prefix. It
refuses complete corruption, reordered frames and foreign ownership. Reopen the
store after repair. `frames(allow_torn_tail=True)` reads a verified prefix without
changing bytes. Checksums provide integrity, not authentication.

Legacy object readers validate their snapshot and journal segments and leave
source bytes untouched. Missing published payloads are errors even with
`verify=False`. Compatibility layouts are not silently converted to native stores.

## Native object stores

```python
from contextlib import closing
from fractions import Fraction
from tempfile import TemporaryDirectory
from rcdb import open_object_store

with TemporaryDirectory() as folder:
    uri = "file://"+folder+"/objects"
    with closing(open_object_store(uri)) as store:
        store.put_record("exact", {"q": Fraction(1, 7)}, expected_version=0)
        cursor = store.change_cursor
    with closing(open_object_store(uri, read_only=True)) as source:
        assert source.change_cursor == cursor
        assert source.get("exact")["q"] == Fraction(1, 7)
```

`open_object_store(uri, native=None)` detects the layout: existing legacy
manifests open read only; new prefixes and native markers select NativeObjectStore.
`native=True` or `False` requests an adapter and refuses incompatible existing data.
`file://` and `memory://` in `open_store` retain their FileStore and MemoryStore
meanings; use the object factory for these object layouts.

NativeObjectStore shares the native version, codec, query, tombstone and cursor
contracts. `read_transaction()` pins one published image. The file provider
arbitrates threads and POSIX processes; the memory provider is atomic within one
process and has no disk durability. Remote native storage requires an explicit
`publication_provider(fs, root)` with bounded reads and atomic compare and swap;
a generic fsspec driver alone does not provide that capability.

A negative publication precondition raises `VersionConflictError`. An unknown
outcome raises `PublicationUncertainError` and blocks handle reuse. Reopen and
inspect the published state before retrying. `compact()` retains history;
staging cleanup uses the explicit retention API below.

## Cross store copies and identity receipts

`copy_record` reads a selected publication and carries its value, metadata, tags
and valid time. It returns the destination record. Pass `return_receipt=True` for
a `CopyReceipt` naming both store IDs, literal record IDs, versions and object
digests. Destination versions and transaction clocks are allocated normally;
source mutation packages are not transplanted.

`receipt.as_record()`, `receipt.to_bytes()` and `CopyReceipt.from_bytes(...)`
preserve exact integers, including versions above `2**53`. A receipt provides
integrity, not signature authentication. Required destinations create their own
mutation commit; pass `governed=True` to request one where commits are optional.

## Native record history

```python
from rcdb import LocalStore, BlobCodecSpec

store = LocalStore("./records", compression=BlobCodecSpec("zlib", 6))
try:
    initial = store.change_cursor
    store.put("literal/id@1", rex, analytics=False)
    changes = store.changes(initial, limit=100)
    resumed = store.cursor_for(changes[-1])
finally:
    store.close()
```

Creation defaults to no compression. Reopening reads the stored codec; an
explicitly different codec refuses. Installing an optional compressor does not
change an existing store's format. MemoryStore, LocalStore, native SQLite SQLStore
and NativeObjectStore share these declarations and native record history.

`history` and exact version reads retain deleted payloads. `get` and ordinary
collections hide deleted current records. Recreation allocates a new version,
and optional mutation commits start a new lineage after deletion.

`ChangeCursor` identifies a checked position in one store's publication history.
`changes(cursor)` returns puts and deletes and rejects foreign or invalid cursors.
Use `cursor_for(change)` to resume. Cursors are logical positions, not byte offsets
or live record counts. Checkpointing and staging retention preserve them.

## Governed history

A version and the signed artifact that attests to it.

- `commit_mutation(id, rex, meta=None, tags=None, actor="", ...)` returns the record.
- `commit_history(id)` returns the packages held for one record.
- `verify_commits(id)` checks those packages against the actual preceding versions.

`require_commits=True` refuses ordinary writes, prepared writes and raw deletion.
The current Core mutation contract attests static RexGraph transitions; it does
not attest generic values, TemporalRex results or deletion. An optional write
does not claim a mutation artifact. A missing artifact claimed by a publication
is an integrity error even when commits are optional.

An uncertain publication retains its artifact and blocks writes until the store
is reopened and verified. Native providers retain audit history across deletion
and start a new commit lineage on recreation.

`state_manifest()` describes logical history independently of backend layout
and analytics. `state_digest()` hashes that manifest with the native value codec,
preserving rational metadata, large integers, tuple/list distinctions and array
dtype/shape. The logical state format is version 2; recompute persisted version 1
logical state hashes. Core object identities and record envelopes are unchanged.
Reading the logical manifest reads all historical payloads. It does not replace
`verify_commits`.

## Checked native migration

`plan_migration(source, destination, actor="")` pins complete native source history
and a different, empty native destination. `migrate_batch(..., limit=100)` preserves
record IDs, exact versions, metadata, signatures, valid times, transaction clocks,
tombstones and delete/recreate history. MemoryStore, LocalStore, native SQLite
SQLStore and NativeObjectStore support this API. Source growth after planning is
excluded. Destination envelopes use the destination's ownership and codec.

```python
from contextlib import closing
from tempfile import TemporaryDirectory
from rexgraph import RexGraph
from rcdb import LocalStore, MemoryStore, MigrationPlan, plan_migration, migrate_batch

with TemporaryDirectory() as folder, closing(MemoryStore()) as source:
    source.put("example", RexGraph.from_graph([0], [1]), analytics=False)
    source.delete("example")
    with closing(LocalStore(folder + "/destination")) as destination:
        plan = plan_migration(source, destination)
        saved_plan = plan.to_bytes()
        first = migrate_batch(source, destination, plan, limit=1)
    with closing(LocalStore(folder + "/destination")) as destination:
        result = migrate_batch(source, destination, MigrationPlan.from_bytes(saved_plan))
        assert result.complete and destination.get("example") is None
        assert destination.get_version("example", 1).nE == 1
```

Save `plan.to_bytes()` and reload with `MigrationPlan.from_bytes(...)` to resume.
Resume checks the actual destination prefix, payloads and required commits. Unrelated
destination writes, a changed policy or damaged payload refuse advancement. Each
publication is durable independently; a failed batch can leave a completed prefix.
Resume checks the full migrated prefix, so very small batches can be costly.

Required commit destinations issue fresh packages under their own actor and policy.
Plans containing deletions or TemporalRex results refuse these destinations before
writing. Policies that would change source declarations also refuse. Reconfigure
signing and encryption capabilities after reopening; plans contain no key material.

`MigrationProgress` and `MigrationStepReceipt` provide native records and checked
byte formats. Keep audit records in a separate store: writing them into the migration
destination changes its prefix and prevents resume.

## Legacy readers and available history migration

FileStore, RexStore and legacy ObjectStore open read only by default. Reads and
close leave source bytes untouched. Existing bound ownership is checked; an
unowned layout receives a content derived source snapshot identity. `auto://`
keeps existing legacy data visible without migrating or replacing it.

Migrate the history that the source still contains into a new native store:

```python
from contextlib import closing
from rcdb import open_store, plan_legacy_migration, migrate_legacy_batch

with closing(open_store("file://./legacy")) as source, closing(open_store("local://./native")) as target:
    plan = plan_legacy_migration(source, target)
    saved_plan = plan.to_bytes()
    print(plan.limitations)
    batch = migrate_legacy_batch(source, target, plan, accept_loss=True, limit=100)
    while not batch.complete:
        batch = migrate_legacy_batch(source, target, plan, accept_loss=True, limit=100)
    mappings = batch.receipts
```

Execution requires `accept_loss=True`. Deleted or compacted history and original
transaction order cannot be reconstructed. Destination clocks and versions are
reallocated; receipts map each available source version to its destination.
Source mutation packages are not transplanted. Resume with the saved plan checks
the actual destination prefix. Keep the source quiescent during migration; a
changed source requires reopening its reader.

The older `migrate` copies selected current IDs and their available histories into
an occupied destination, with new transaction clocks. It reports its copy scope
and receipts but does not replay native tombstones or original clocks.

Explicit `read_only=False` retains the historical compatibility writers. They do
not gain native publication guarantees. New stores and Agent defaults use the
native engine; no implicit migration retires an existing writer.

## Typed native records

Native stores accept `put_record(id, value, codec=VALUE_CODEC, meta=None, tags=None,
valid_from=None, valid_to=None, tx_time=None, expected_version=None)`. Typed values
share version allocation, time selectors, cursors and tombstones with complexes.
`get`, `get_version` and `read_record` reconstruct the declared value.

Use `read_record` to distinguish a stored `None` from a missing record: the former
returns a snapshot whose value is None; the latter returns no snapshot. False,
zero, explicit absence and empty containers remain present values.

```python
from contextlib import closing
from fractions import Fraction
from rcdb import MemoryStore, PROVENANCE_CODEC

with closing(MemoryStore()) as store:
    record = store.put_record("measurements", {"ratio": Fraction(1, 7), "keys": (1, 2)},
                              tx_time=10.0, expected_version=0)
    store.put_record("provenance", {"input": record.envelope.object_digest, "method": "exact"},
                     codec=PROVENANCE_CODEC, tx_time=20.0)
    assert store.read_record("measurements").value["ratio"] == Fraction(1, 7)
    assert store.query(record_type="ProvenanceRecord")[0].id == "provenance"
```

`VALUE_CODEC` preserves Core's native exact values and containers. `PROVENANCE_CODEC`
accepts string keyed records. `DECLARATION_CODEC`, `COPY_RECEIPT_CODEC` and the
`MIGRATION_*_CODEC` references preserve their corresponding declarations.
Register RCQL's portable result codec explicitly:

```python
from contextlib import closing
from rcdb import MemoryStore
from rcql import Result, register_result_storage_codec

codec = register_result_storage_codec()
with closing(MemoryStore()) as store:
    store.put_record("query/result", Result(values=(1,)), codec=codec)
    assert store.get("query/result").to_bytes() == Result(values=(1,)).to_bytes()
```

Codec names select installed capabilities and never import a provider. Install
optional codecs on every reader. Metadata remains readable when a payload's
codec is unavailable. `record_codecs=(CodecRef(...), ...)` declares the inventory
at store creation; changing an existing inventory requires migration.

`copy_record` and checked migration carry typed values through the same APIs.
Destinations requiring graph mutation commits refuse generic values. Metadata
and tag queries include typed records; `record_type` selects their declared kind.
Graph family, similarity and structural predicates select complexes only.

## Portable record versions

`record_packet(store, id, version=..., as_of=..., valid_at=...)` captures one checked
publication. Missing records return `None`; a stored `None` has a real packet.
`RecordPacket.from_bytes(packet.to_bytes())` checks the frame and envelope.
`packet.snapshot()` resolves the installed codec and verifies the value's identity.

Packets carry exact metadata, tags, valid time, selected source version and source
transaction interval. They do not export encryption keys, compression, store
journals or mutation packages. Install optional codecs on both peers.

```python
from contextlib import closing
from fractions import Fraction
from rcdb import MemoryStore, copy_record, record_packet

with closing(MemoryStore()) as source, closing(MemoryStore()) as destination:
    source.put_record("reading", Fraction(1, 7))
    packet = record_packet(source, "reading", version=1)
    receipt = copy_record(packet.source(), destination, packet.record,
                          destination_id="received/reading", return_receipt=True)
    assert destination.get(receipt.destination_record_id) == Fraction(1, 7)
```

`destination_id` defaults to the source ID. Destination versions and clocks are
allocated normally, subject to destination policy. Packets provide integrity;
the transport authenticates the sender. Agent's `RexClient` exposes record fetch
and publication over HTTP with binary copy receipts. See
[stored record transfer](../agent/README.md#stored-record-transfer).

## Extras

Base install provides native memory/local backends, legacy file/rex readers, the
signature and the protected search index. safetensors is a base dependency.

| extra | brings | for |
|---|---|---|
| `sql` | sqlalchemy | `SQLStore` |
| `objectstore` | fsspec | `ObjectStore` |
| `crypto` | cryptography | payload encryption and signed commits |

```bash
pip install "rexgraph-rcdb[sql,objectstore,crypto]"
```

## Replay and staging maintenance

LocalStore and NativeObjectStore provide `checkpoint()` to reduce physical replay
reads. Checkpoints preserve every version, tombstone and logical cursor; reopening
still validates retained transitions. Local checkpoints do not truncate the journal.

```python
from rcdb import LocalStore, RetentionPolicy

store = LocalStore("./records")
checkpoint = store.checkpoint(max_frames=256, target_bytes=8 * 1024 * 1024)
plan = store.plan_retention(RetentionPolicy(grace_seconds=86400,
                                          max_objects=256,
                                          max_bytes=256 * 1024 * 1024))
# Review the plan before applying it.
result = store.apply_retention(plan)
store.close()
```

Retention removes only recognized unpublished staging objects, within the selected
age, count and byte limits. It preserves every published version and claimed artifact.
MemoryStore and native SQLite artifacts have no staging timestamp; their cleanup
requires `grace_seconds=0`. Changed state or changed candidates refuse a stale plan.
A cleanup I/O failure can leave partial staging removal; inspect and make a fresh plan.

Maintenance requires an unscoped store handle. Native object maintenance upgrades
the head format from v1 to v2; upgrade all clients before invoking it. A corrupt
local checkpoint refuses reopening; an absent optional cache falls back to the
primary journal. Checkpointing does not prune published history.
