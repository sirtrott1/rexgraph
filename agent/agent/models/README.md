# models: the model builder framework (on rexgraph.nn)

Pick an archetype, override its parameters, point it at data, and train it as a single run, a staged
multistep run, or a multi model fusion. These assembled example models live in the Agent
distribution. Core supplies `rexgraph.nn` components and retained relational models in
`rexgraph.flow`; those cochain models use the complex as their parameter space.

## Archetypes (the selector)

| name | use case | data kind | key params |
|---|---|---|---|
| `mlp`  | tabular / vector: classification or regression | vector | `d_hid`, `n_layers`, `task` |
| `cnn`  | image classification | image | `depth`, `width`, `norm` |
| `lm`   | sequence / language modeling (next token) | sequence | `d`, `n_head`, `n_layer`, `attention` (`relational`/`standard`) |
| `hgnn` | node classification on hypergraphs / higher order relational data (advection+diffusion, uses signed orientation) | hypergraph | `d_hid`, `n_layers`, `flow`, `oriented` |

Every archetype is built from `rexgraph.nn` components (PropagatorAttention, `build_attention`, the
rcf_torch propagators). No archetype builds an optimizer: training routes through
`make_optimizer("auto")`, and since all four are feature space models that resolves to plain Adam.
Register a new one with `register_archetype(...)`.

## Use it (Python)

```python
from agent.models import list_archetypes, run, build

list_archetypes()                          # names, use-cases, and each archetype's params

# single run on synthetic data (default). optimizer defaults to "auto" (the router)
run("cnn", params={"norm": False}, steps=300)

# your data
run("mlp", data="mydata.csv")                        # csv/jsonl/npz table
run("lm",  data="corpus.txt", params={"attention": "standard"})

# any optimizer by name when you want a specific one (an A/B arm, say)
run("mlp", data="mydata.csv", optimizer="adamw")

# just build the model (no training)
model, cfg, bundle = build("hgnn", params={"n_layers": 3})

# multistep: stage training (curriculum / optimizer schedule / warmup to refine)
run("mlp", mode="multistep", stages=[
    {"steps": 100},                       # warm up on the routed optimizer
    {"steps": 300, "lr": 5e-4},           # refine at a lower lr
])

# multi-model fusion: ensemble / data-split specialists / stacking
run("mlp", mode="fusion", fusion="ensemble",
    specs=[("mlp", {}), ("mlp", {"d_hid": 64})])   # average predictions
run("cnn", mode="fusion", fusion="split", specs=[("cnn", {}), ("cnn", {"norm": False})])  # data-parallel specialists
run("mlp", mode="fusion", fusion="stack", specs=[("mlp", {}), ("mlp", {"n_layers": 3})])  # meta-head over base logits
```

## Use it (CLI)

```
python -m agent.models list
python -m agent.models build     --archetype cnn --set norm=false --steps 300
python -m agent.models build     --archetype mlp --data mydata.csv --optimizer adamw
python -m agent.models multistep --archetype mlp --stage steps=100 --stage steps=300,lr=5e-4
python -m agent.models fusion    --spec mlp --spec mlp:d_hid=64 --fusion ensemble
```

## rexgraph IO: data in, models + complexes out

Data and complex I/O use `rexgraph.io`, with RCDB cataloguing for complexes. Agent feature model
checkpoints are local directories containing weights and configuration; their path is not a SQL
URI. Table sources, including SQL, are loaded into memory. Use the mapped
numeric table loader below when the complete table should remain on disk.


```python
from agent.models import run, load_bundle, save_checkpoint, load_checkpoint, save_complex_rex, to_rcdb

# data in: any rexgraph.io source to a DataBundle
load_bundle("train.parquet")            # parquet table (feature cols + label)
load_bundle("vecs.safetensors")         # a save_vectors / embedding corpus
load_bundle("graph.rcbd")               # a .rcbd bundle to hypergraph (signed complex)
load_bundle("postgresql://...", table="samples")   # a database table
# run() takes any of these directly:
run("mlp", data="train.parquet", save_to="ckpt")

# model out (a checkpoint on the IO stack): weights.safetensors, config.json, trajectory.safetensors
#   (the trajectory is a rexgraph.io vector corpus, same format as embeddings / hodge trajectories,
#    so it lands in the RCDB vector store and is queryable alongside them)
save_checkpoint("ckpt", model, "mlp", cfg, bundle=bundle, result=r)
model, conf = load_checkpoint("ckpt")

# complex: a hypergraph's relational complex to .rcbd, or catalogued in the RCDB
save_complex_rex(bundle, "hg.rcbd")
to_rcdb(bundle, "sqlite:///rcdb.sqlite", name="my_hg", tags=["hgnn"])   # stored by Betti/coherence signature
```

The flow: data (parquet / vectors / .rcbd / SQL) to DataBundle to model (weights to safetensors,
config to json, training trajectory to `save_vectors`). For `hgnn` the complex goes to a `.rcbd`
bundle or the RCDB, where it is queryable by its topology, not just id. The optimizer's own
coordinated vs rotational trajectory (`rexgraph.nn.save_hodge_trajectory`) uses the same vector path.

## Notes

- **Device**: defaults to `cpu`. Pass `device="auto"` to use the compute recommendation,
  or request a Torch device explicitly. GPU training requires working kernels for that archetype;
  availability depends on the installed Torch runtime and device.
- **Data**: `vector` (csv/jsonl/npz) and `sequence` (text) load from files; for `image`/`hypergraph`,
  pass a `DataBundle` (see `data.py`) or use the synthetic generators.
- **Regression**: `params={"task": "regression"}` selects real targets, MSE training, real predictions,
  and a `-test MSE` metric, including fusion. Direct table/vector loading accepts
  `load_bundle(source, task="regression")`. Classification rejects fractional or negative class
  indices. Unlabelled vector corpora retain `y=None` and yield no accuracy metric.
- **HGNN flow**: `flow=False` runs only the vertex heat branch. The default includes the signed
  cross grade flow. Model construction, `.rcbd` export and RCDB publication retain the declared
  node count, including isolates, and stored participant order.
- **Checkpoint semantics**: prediction without explicit data uses the saved archetype configuration
  for synthetic data. `resume` loads feature model weights before starting a new optimizer; it is
  a warm start. Exact optimizer/RNG continuation and source bound model records use Core's retained
  model lifecycle (`rexgraph.nn.create_checkpoint`, `train_checkpoint`, and the RCQL model operators).
  Those records check the recorded implementation and environment when resuming.
- **Optimizer**: `auto` (default; routes per model type: GreensCochain for cochain native models, else Adam), or any `rexgraph.nn` optimizer by name: `greens`, `adam`, `adamw`, `sgd`, `hodge`/`hodge-arch` (deprecated, back compat).

## Larger numeric datasets

```python
from agent.models import load_mapped_table, run
bundle = load_mapped_table("features.npy", "class_ids.npy", n_classes=4,
                           eval_batch_size=1024)
result = run("mlp", data=bundle, device="cpu", steps=200)
```

The feature matrix and optional scalar targets are read only memory maps. Loading
checks headers; each fetched batch checks finite features and target validity.
Classification requires integer storage and an explicit class count. Regression
uses `task="regression"`; set the archetype task to regression as well. Unlabelled
bundles work for prediction.

Splits are ordered ranges using the supplied train/validation/test ratios. Shuffle
the files beforehand when you need a random split; chronological data can retain
its order. Training samples within the training range. Split fusion partitions
that range by strides, without a dataset sized permutation.

Training moves only the sampled batch. Evaluation, prediction and fusion bound
feature reads by `eval_batch_size`. Returning all predictions still requires host
memory for that result. Mapped file pages can remain in the OS cache independently of tensor batches.
Stacking recomputes base features rather than retaining a complete feature cache.

For `device="cuda"` on CUDA or ROCm, batches use pinned host memory and nonblocking
copies on Torch's current stream. The loader fetches one batch at a time.

`rexgraph.nn.PackedTernaryLinear` composes the Core packed CPU/HIP operator with
learnable layers. Its ternary map is fixed; input gradients use the actual
transpose, including double backward. Float32/float64 HIP tensors stay on their
device and use the active Torch stream. Move the layer and input together. As with
other Torch operations, a caller using another stream must establish readiness
with `wait_stream`; allocator lifetime is handled by the layer. Explicit HIP use
raises when the native library or compatible ROCm device is absent.

Run `python -m rexgraph.gpu_preflight --json` on the target machine before relying
on GPU propagation, Green gradients or the native packed training bridge. A skipped
optional native check is reported separately from checks that executed.
