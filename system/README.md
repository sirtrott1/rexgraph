# RexGraph System

System is the RexGraph observatory. It reads live Rex values through RCQL and does not reimplement the mathematics.

Run it with no initial source:

```bash
rexgraph-system
```

Load a Rex source at startup:

```bash
rexgraph-system --source main=graph.rex
```

Applications can also register live values with `system.register_source`.

## File catalogs

System can register explicit local roots without exposing absolute paths to RCQL or the frontend:

```bash
rexgraph-system --catalog files=/data/rex
```

Catalogs index `.rex` bundles, `.safetensors` files, and RCDB stores. Full content hashes are computed on demand so large model files do not slow catalog startup. Catalog names are relative to an opaque root label. Symlinks are not traversed. Search uses literal terms, not regular expressions or shell patterns.

```text
FROM CATALOG("files") RETURN FILES()
FROM CATALOG("files") RETURN SEARCH("corpus")
FROM CATALOG("files") RETURN FILE_HASH("root0/corpus.rex")
FROM FILE("files", "root0/corpus.rex") RETURN BETTI(1), STATE_HASH()
```
