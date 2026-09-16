# RCDB

A versioned store for native relational complexes. RCDB depends on RexGraph,
not on RCQL, Agent or System. Those packages use the same store contracts.

## Install and run

From the repository root, install the core before the store:

```sh
python -m pip install .
python -m pip install './rcdb[sql,objectstore,crypto]'
```

The extras are optional. The base package provides memory, file and rex stores.
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

The installed lifecycle passes local roundtrips, navigation, native relation edits, exact rational commits,
all 20 cross backend copy directions and fresh process reopening. Both original
serialization defects are fixed, including HDF5/Zarr temporal timestamps.
The 927 check gate covers five local backends and five file formats; it does not
claim remote provider, concurrent process writer, crash or scale acceptance.

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

`open_store(uri)` selects the backend by scheme. All five expose `RCStore`.
Publication and durability guarantees still depend on the backend.

| uri | backend | for |
|---|---|---|
| `memory://` | `MemoryStore` | tests, ephemeral work |
| `file:///path` | `FileStore` | one directory, one file per version |
| `rex:///path` | `RexStore` | append log and tensor index |
| `sqlite:///path`, any SQLAlchemy url | `SQLStore` | shared, queryable, transactional |
| `s3://`, `gs://`, `az://`, any fsspec url | `ObjectStore` | object storage |

`register_backend(scheme, opener)` adds one from outside. `available_backends()`
lists what is registered.

Records are bitemporal: `put` appends a version rather than replacing one, a
version is closed by the arrival of its successor, and `as_of` and `valid_at`
select against transaction time and valid time separately.

Use one writer handle per store. Locks serialize publication within that handle;
they are not a lock shared by independent processes. Expected version checks
prevent stale updates through the supported publication path. Distributed writes,
remote object providers and recovery after a process crash need separate acceptance.

`read_record(id, version=None, as_of=None, valid_at=None)` returns one detached
native payload, copied `ComplexRecord` and canonical state digest together as a
`RecordSnapshot`. Selection and decoding share the handle's publication lock.
Use this contract for version aware consumers such as RCQL; it avoids pairing
metadata from one read with a payload from another.



## Interoperating with an application

RCDB does not import the agent, or anything else built on it. What an application
adds arrives through four callables:

- `configure_hooks(activity=None, scope=None, privacy=None, similarity=None)`

  `activity` records a change into the application's own feed, `scope` narrows the
  default store to what a request may see, `privacy` projects metadata before it is
  stored, and `similarity` replaces the default scoring. Each is a plain callable,
  so nothing here depends on where it came from.

With none of them set the store still stores. That is the property that makes this
installable alone, and an architecture test enforces the direction rather than
leaving it to convention.



## The signature

Every `put` computes a structural signature and stores it beside the payload:
shape, betti numbers, chain validity, coherence, and a label sample. That is what
`query` filters on without deserializing a complex, and what `find_similar` and
`cluster_complexes` read.

The measurements live in `rcdb.analytics` rather than in the application, because
a store that had to import the application to describe what it is storing would not
be standalone in any useful sense.



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

Additive: a store that never calls it reads and writes exactly what it always did.

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



## Governed history

A prepared payload is not a substitute for a required mutation commit.

A version and the signed artifact that attests to it.

- `commit_mutation(id, rex, meta=None, tags=None, actor="", ...)` -> the record
- `commit_history(id)` -> the packages held for one record
- `verify_commits(id)` -> whether this record's history is the one its commits attest to

`verify_commits` walks forward and checks each package against the version that
actually precedes it in the store, rather than against whatever the package claims.
That is the chain property: a package proves a transition only when the endpoint it
started from is the one on disk.

The artifact is staged BEFORE the record and rolled back on a definite publication
failure. An uncertain publication deliberately retains its artifact: deleting it
could leave a version without its attestation. The handle then refuses writes until
reopened and verified. A rollback failure can also leave an unreferenced artifact;
unreferenced artifacts are not exposed as published versions.

`require_commits=True` makes it mandatory. An ordinary `put` is then refused,
because it would create a version with nothing attesting to it, and so is a raw
`delete`: deletion is the one operation a chain cannot describe, since no artifact
says a record was meant to stop existing. Deleting a record reclaims its artifacts,
so a later record reusing that id does not inherit an attestation it never earned.

An audit journal proves a log LINE was not edited. This is a different object: it
proves that record `X@v3` is the state the commit after `v2` says follows it.

`state_manifest()` describes the visible published logical history independently of
backend layout and optional analytics; `state_digest()` hashes it with the framework
manifest codec. This reads all historical payloads, not just the index. Hashing commit
identities is not commit verification. `copy_record` pins the selected source version
and automatically uses a new destination mutation commit when that store requires
commits; `governed=True` requests one even on an optional commit destination.



## Extras

Base install is the memory, file and rex backends, the signature and the protected
search index. safetensors is a BASE dependency rather than an extra, because every
backend's put writes safetensors bytes: a store without it would import and then fail
on its first write.

| extra | brings | for |
|---|---|---|
| `sql` | sqlalchemy | `SQLStore` |
| `objectstore` | fsspec | `ObjectStore` |
| `crypto` | cryptography | payload encryption and signed commits |

```bash
pip install "rexgraph-rcdb[sql,objectstore,crypto]"
```
