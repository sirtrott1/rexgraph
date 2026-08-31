# RCDB

A database of relational complexes. Sits beside `rexgraph` rather than inside an
application, so a store can be installed, tested and reasoned about on its own.

```python
from rcdb import open_store

store = open_store("rex:///data/complexes")
store.put("run-1", rex, meta={"vertex_labels": labels}, tags=["ingest"])
store.query(labels_any=["oncology"], limit=20)
```

---

## Backends

`open_store(uri)` dispatches on the scheme. All five present the same `RCStore`
surface, so nothing above them cares which is underneath.

| uri | backend | for |
|---|---|---|
| `memory://` | `MemoryStore` | tests, ephemeral work |
| `file:///path` | `FileStore` | one directory, one file per version |
| `rex:///path` | `RexStore` | append-only log plus a tensor index, O(1) put |
| `sqlite:///path`, any SQLAlchemy url | `SQLStore` | shared, queryable, transactional |
| `s3://`, `gs://`, `az://`, any fsspec url | `ObjectStore` | object storage |

`register_backend(scheme, opener)` adds one from outside. `available_backends()`
lists what is registered.

Records are bitemporal: `put` appends a version rather than replacing one, a
version is closed by the arrival of its successor, and `as_of` and `valid_at`
select against transaction time and valid time separately.

---

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

---

## The signature

Every `put` computes a structural signature and stores it beside the payload:
shape, betti numbers, chain validity, coherence, and a label sample. That is what
`query` filters on without deserializing a complex, and what `find_similar` and
`cluster_complexes` read.

The measurements live in `rcdb.analytics` rather than in the application, because
a store that had to import the application to describe what it is storing would not
be standalone in any useful sense.

---

## Protected search

The canonical snapshot reconstructs records, so it is not a security boundary: it
writes labels in the clear. Sealing records and shipping a plaintext term index
beside them protects nothing.

`rcdb.protected_index` builds a separate, disposable relation whose vocabulary is
fixed-width tokens. A persisted index carries neither a plaintext term nor a
plaintext record id, and exact lookup still answers.

- `IndexPolicy(modes, key_id)` with a mode per accession kind:
  `public` and `structural` are domain-separated SHA-256, so a token cannot collide
  across kinds but a guessed term can still be recomputed; `keyed` is HMAC under a
  key the caller must hold, which is the mode that resists enumeration; `none` is
  not indexed.
- `build_search_relation(records, policy, keys)` / `build_search_relation_from_tokens(...)`
- `save_search_relation(path, relation)` / `load_search_relation(path)`

Identities live only in an in-memory resolver, never in the file, so a persisted
relation returns tokens and refuses to name records. The key arrives through an
`IndexKeyProvider` rather than as bytes, which is what lets an application scope a
key identifier per tenant.

**A search index is derived, disposable state.** Rebuild it rather than migrate it.

---

## Sealing records

- `configure_security(key_id=None, keys=None, mutation_policy=None, verifiers=None, transition_signer=None, lineage_signer=None, signature_mode="public", metadata_fields=None, require_commits=False)`

Additive: a store that never calls it reads and writes exactly what it always did.

With `key_id` and a `KeyProvider`, payloads are sealed with AES-GCM. Opening
decides by the ENVELOPE rather than by configuration, so records written before a
key was introduced stay readable beside records written after it, a sealed blob in
an unconfigured store is a refusal rather than a plaintext read of ciphertext, and
a wrong key is a refusal rather than a library exception.

Sealing the payload is not enough on its own. A signature is a description of the
data, so a store that seals its records and writes the full signature beside them
has described what it sealed. `signature_mode` keeps the shape and the invariants
under `structural`, or only what is needed to address a record under `minimal`, and
`metadata_fields` is an allow-list over what metadata persists.

**This is not database-at-rest confidentiality.** Record identifiers, the activity
log, SQL identity columns, file and directory names, and backend metadata are all
outside the envelope. Those are a separate identity and storage problem, and a
deployment that needs them covered needs disk or filesystem encryption underneath.

---

## Governed history

A version and the signed artifact that attests to it.

- `commit_mutation(id, rex, meta=None, tags=None, actor="", ...)` -> the record
- `commit_history(id)` -> the packages held for one record
- `verify_commits(id)` -> whether this record's history is the one its commits attest to

`verify_commits` walks forward and checks each package against the version that
actually precedes it in the store, rather than against whatever the package claims.
That is the chain property: a package proves a transition only when the endpoint it
started from is the one on disk.

The artifact is staged BEFORE the record and rolled back if the record write fails, so
a version is never published without one and a normal failure leaves neither. Only a
failure of the rollback itself can leave an artifact behind, and that is the safe
direction: an unreferenced artifact is inert, a version with no commit is a hole.

`require_commits=True` makes it mandatory. An ordinary `put` is then refused,
because it would create a version with nothing attesting to it, and so is a raw
`delete`: deletion is the one operation a chain cannot describe, since no artifact
says a record was meant to stop existing. Deleting a record reclaims its artifacts,
so a later record reusing that id does not inherit an attestation it never earned.

An audit journal proves a log LINE was not edited. This is a different object: it
proves that record `X@v3` is the state the commit after `v2` says follows it.

---

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
