# RexGraph System

System is the RexGraph observatory. It reads live Rex values through RCQL and does not reimplement the mathematics.

Run it with no initial source:

```bash
rexgraph-system
```

Load a Rex source at startup:

```bash
rexgraph-system --source main=graph.rcbd
```

Applications can also register live values with `system.register_source`.

The wheel includes the frontend under `system/frontend`, including its React
runtime. Launching does not download code or write into site packages. A missing
asset is an incomplete installation error, not a request to fetch a CDN dependency.
The assets are the same React 18.2 distribution used by this checkout's Agent UI;
their [MIT license](https://github.com/facebook/react/blob/v18.2.0/LICENSE) ships
as `system/frontend/REACT-LICENSE.txt`. No Agent install is required by System.

Runtime file SHA-256 values:

- `react.production.min.js`: `4b4969fa4ef3594324da2c6d78ce8766fbbc2fd121fff395aedf997db0a99a06`
- `react-dom.production.min.js`: `21758ed084cd0e37e735722ee4f3957ea960628a29dfa6c3ce1a1d47a2d6e4f7`

The ReactDOM distribution internally reports `18.2.0-next-9e3b772b8-20220608`;
the packaged bytes are preserved, not relabeled. Frontend HTTP tests check every
dependency referenced by the index, and run against installed wheels as well as
source. These are asset/API acceptance checks, not physical desktop/browser UX
acceptance. RCQL remains the only evaluator; the frontend renders its results.

## File catalogs

System can register explicit local roots without exposing absolute paths to RCQL or the frontend:

```bash
rexgraph-system --catalog files=/data/rex
```

Catalogs index `.rcbd` bundles (including legacy `.rex` ones, which are identified by their manifest rather than their name), `.safetensors` files, and RCDB stores. Full content hashes are computed on demand so large model files do not slow catalog startup. Catalog names are relative to an opaque root label. Symlinks are not traversed. Search uses literal terms, not regular expressions or shell patterns.

```text
FROM CATALOG("files") RETURN FILES()
FROM CATALOG("files") RETURN SEARCH("corpus")
FROM CATALOG("files") RETURN FILE_HASH("root0/corpus.rcbd")
FROM FILE("files", "root0/corpus.rcbd") RETURN BETTI(1), STATE_HASH()
```
