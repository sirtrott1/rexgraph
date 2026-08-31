# RCQL

RCQL is the typed query and mutation layer for relational complexes.

Human query text and programmatic callers lower to the same typed AST. System and Agent can construct that AST directly without generating executable strings.

## Local artifacts

RCQL reads registered `FileCatalog` sources. It does not expose a general filesystem primitive. File names are catalog relative names and the catalog resolves and rechecks them before each access.

```text
FROM CATALOG("files") RETURN FILES(), SEARCH("document")
FROM FILE("files", "root0/document.rex") RETURN DESCRIBE(), STATE_HASH()
```

RCDB stores use the same source model. Read operators include `RCDB_LIST`, `RCDB_SEARCH`, `RCDB_GET`, `RCDB_HISTORY`, `RCDB_COMMITS`, `RCDB_VERIFY`, `RCDB_HASH`, `RCDB_STATE_HASH`, `RCDB_STATS`, and `RCDB_SECURITY`.

## Capabilities

A `BoundSource` carries a `SourcePolicy`. Permissions are checked before an operator receives the underlying source. Derived `FILE` sources retain the catalog policy.

Exact record operations require `identity`. Mutation requires both `mutate` and `identity`. Commit history requires both `history` and `identity`. `RCDB_SECURITY` requires `admin`.

A source policy can also restrict which structural signature fields appear in record lists and searches.

## Mutations

Typed mutation requests use `MutationQuery`. They do not share the ordinary read operator registry.

A mutation lowers to the RCDB `commit_mutation` path, which constructs a TemporalRex mutation package and applies the store integrity policy before publication. A store configured with `require_commits=True` rejects ordinary `put` updates and raw deletion.
