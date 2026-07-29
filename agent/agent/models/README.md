# models: the model-builder framework (on rexgraph.nn)

Pick an archetype, override its parameters, point it at data, and train it as a single run, a staged
multistep run, or a multi-model fusion. These are the assembled example models. They live outside
the `rexgraph` repo, which ships only `rexgraph.nn` (the parts to build them).

## Archetypes (the selector)

| name | use-case | data kind | key params |
|---|---|---|---|
| `mlp`  | tabular / vector: classification or regression | vector | `d_hid`, `n_layers`, `task` |
| `cnn`  | image classification (`norm=False` exercises HodgeAdam's conditioning) | image | `depth`, `width`, `norm` |
| `lm`   | sequence / language modeling (next-token) | sequence | `d`, `n_head`, `n_layer`, `attention` (`relational`/`standard`) |
| `hgnn` | node classification on hypergraphs / higher-order relational data (advection+diffusion, uses signed orientation) | hypergraph | `d_hid`, `n_layers`, `flow`, `oriented` |

Every archetype is built from `rexgraph.nn` components (HodgeAdam optimizer, PropagatorAttention,
`build_attention`). Register a new one with `register_archetype(...)`.

## Use it (Python)

```python
from agent.models import list_archetypes, run, build

list_archetypes()                          # names, use-cases, and each archetype's params

# single run on synthetic data (default), your optimizer + params
run("cnn", params={"norm": False}, optimizer="hodge", steps=300)

# your data
run("mlp", data="mydata.csv", optimizer="hodge")     # csv/jsonl/npz table
run("lm",  data="corpus.txt", params={"attention": "standard"})

# just build the model (no training)
model, cfg, bundle = build("hgnn", params={"n_layers": 3})

# multistep: stage training (curriculum / optimizer schedule / warmup to refine)
run("mlp", mode="multistep", stages=[
    {"optimizer": "adam",  "steps": 100},          # warm up
    {"optimizer": "hodge", "steps": 300, "lr": 5e-4},  # refine with the chosen optimizer
])

# multi-model fusion: ensemble / data-split specialists / stacking
run("mlp", mode="fusion", fusion="ensemble",
    specs=[("mlp", {}), ("mlp", {"d_hid": 64})])   # average predictions
run("cnn", mode="fusion", fusion="split", specs=[("cnn", {}), ("cnn", {"norm": False})])  # data-parallel specialists
run("mlp", mode="fusion", fusion="stack", specs=[("mlp", {}), ("mlp", {"n_layers": 3})])  # meta-head over base logits
```

## Use it (CLI)

```
python -m models list
python -m models build     --archetype cnn --set norm=false --optimizer hodge --steps 300
python -m models build     --archetype mlp --data mydata.csv --optimizer hodge
python -m models multistep --archetype mlp --stage optimizer=adam,steps=100 --stage optimizer=hodge,steps=300
python -m models fusion    --spec mlp --spec mlp:d_hid=64 --fusion ensemble
```

## rexgraph IO: data in, models + complexes out

Everything persists through `rexgraph.io` (and RCDB for complexes), so a trained model is portable
from a laptop file store to Postgres by changing a URI. See `store.py`.


```python
from agent.models import run, load_bundle, save_checkpoint, load_checkpoint, save_complex_rex, to_rcdb

# data in: any rexgraph.io source to a DataBundle
load_bundle("train.parquet")            # parquet table (feature cols + label)
load_bundle("vecs.safetensors")         # a save_vectors / embedding corpus
load_bundle("graph.rex")                # a .rex bundle to hypergraph (signed complex)
load_bundle("postgresql://...", table="samples")   # a database table
# run() takes any of these directly:
run("mlp", data="train.parquet", save_to="ckpt")

# model out (a checkpoint on the IO stack): weights.safetensors, config.json, trajectory.safetensors
#   (the trajectory is a rexgraph.io vector corpus, same format as embeddings / hodge trajectories,
#    so it lands in the RCDB vector store and is queryable alongside them)
save_checkpoint("ckpt", model, "mlp", cfg, bundle=bundle, result=r)
model, conf = load_checkpoint("ckpt")

# complex: a hypergraph's relational complex to .rex, or catalogued in the RCDB
save_complex_rex(bundle, "hg.rex")
to_rcdb(bundle, "sqlite:///rcdb.sqlite", name="my_hg", tags=["hgnn"])   # stored by Betti/coherence signature
```

The flow: data (parquet / vectors / .rex / SQL) to DataBundle to model (weights to safetensors,
config to json, training trajectory to `save_vectors`). For `hgnn` the complex goes to a `.rex`
bundle or the RCDB, where it is queryable by its topology, not just id. The optimizer's own
coordinated-vs-rotational trajectory (`rexgraph.nn.save_hodge_trajectory`) uses the same vector path.

## Notes

- **Device**: defaults to `cpu` (runs everywhere). Pass `device="cuda"` for `mlp`/`lm`/`hgnn`; `cnn`
  stays on cpu because this box's ROCm build has no working conv kernel (matmul/LoRA do run on GPU).
- **Data**: `vector` (csv/jsonl/npz) and `sequence` (text) load from files; for `image`/`hypergraph`,
  pass a `DataBundle` (see `data.py`) or use the synthetic generators.
- **Optimizer**: `auto` (default; routes per model type: GreensCochain for cochain-native models, else Adam), or any `rexgraph.nn` optimizer by name: `greens`, `adam`, `adamw`, `sgd`, `hodge`/`hodge-arch` (deprecated, back-compat).
