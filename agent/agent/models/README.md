# Agent models

The model builder assembles MLP, CNN, language and hypergraph models from
`rexgraph.nn` components. It supports single runs, staged training and model
fusion. These feature models use Adam through `make_optimizer("auto")`.
Core's [cochain models](../../../rexgraph/flow/README.md) use cell parameters
and their own Green optimizer contract.

## Installation

From the repository root:

```sh
make install-ml
.venv/bin/rexgraph-models list
```

This installs PyTorch, Core neural components and Agent model dependencies.
See the [installation guide](../../../README.md#optional-runtimes) for GPU builds.

## Archetypes

| Archetype | Purpose | Data kind | Main parameters |
| --- | --- | --- | --- |
| `mlp` | Classification or regression on tabular features | `vector` | `d_hid`, `n_layers`, `task` |
| `cnn` | Image classification | `image` | `depth`, `width`, `norm` |
| `lm` | Next token prediction | `sequence` | `d`, `n_head`, `n_layer`, `attention` |
| `hgnn` | Node classification with oriented flow and diffusion | `hypergraph` | `d_hid`, `n_layers`, `flow`, `oriented` |

Register an additional archetype with `register_archetype(...)`.

## Usage

The following run uses the archetype's synthetic data:

```python
from agent.models import list_archetypes, run

print(list_archetypes())
result = run("mlp", steps=20, device="cpu")
```

Supply a table, a text file or a `DataBundle` to train on application data.
`build` creates a model without training it. Staged runs accept a list of
step counts and learning rates. Fusion supports `ensemble`, `split` and `stack`.

From the repository root:

```sh
.venv/bin/rexgraph-models build --archetype mlp --data mydata.csv --optimizer adamw
.venv/bin/rexgraph-models multistep --archetype mlp --stage steps=100 --stage steps=300,lr=5e-4
.venv/bin/rexgraph-models fusion --spec mlp --spec mlp:d_hid=64 --fusion ensemble
```

## Data and model storage

Data loaders use rexgraph.io, and source complexes can be catalogued in RCDB.
Feature model checkpoints are local directories containing weights and
configuration. Table sources, including SQL, load into memory; use the mapped
numeric loader below to keep the complete table on disk.

```python
from agent.models import build, load_checkpoint, save_checkpoint

model, configuration, bundle = build("mlp")
save_checkpoint("checkpoint", model, "mlp", configuration, bundle=bundle)
restored, saved_configuration = load_checkpoint("checkpoint")
```

Loaders produce a `DataBundle` for model construction and training. Model weights
are saved as safetensors and configuration as JSON. Training trajectories use
`save_vectors`; `rexgraph.nn.save_hodge_trajectory` uses the same vector storage
for Hodge components of optimizer updates. An `hgnn` source complex can be saved
as a native `.rcbd` bundle or an RCDB record for structural queries.

## Configuration

The device defaults to `cpu`. Pass `device="auto"` to use the compute
recommendation or specify a Torch device. The selected runtime must support the
archetype's operations.

Vector inputs load from CSV, JSONL or NPZ files, and sequence inputs from text.
Image and hypergraph inputs use a `DataBundle` or generated inputs;
[data.py](data.py) defines the bundle fields.

Set `params={"task": "regression"}` for real targets, mean squared error
training and real predictions. The reported score is negative test MSE,
including fusion. `load_bundle(source, task="regression")` selects the same
target interpretation. Classification requires nonnegative integer class
indices. Unlabelled vector data retains `y=None` and supplies no accuracy metric.

For HGNN, `flow=False` selects vertex heat propagation only. The default also
includes signed flow across grades. Model construction and native storage
retain the declared node count, isolated vertices and participant order.

The automatic optimizer uses `GreensCochain` for models with complex parameter
groups and Adam for ordinary groups. Named choices are `greens`, `adam`, `adamw` and
`sgd`.

## Checkpoint semantics

Prediction without explicit data generates inputs using the saved archetype
configuration. `resume` performs a warm start by restoring weights and creating a
new optimizer. To continue with saved optimizer and random state, use
`rexgraph.nn.create_checkpoint`, `train_checkpoint` or the RCQL model lifecycle
operators. Retained records check the implementation and environment when
resuming.

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

Training transfers the sampled batch, while evaluation, prediction and fusion
bound feature reads by `eval_batch_size`. Returning all predictions requires host
memory for that result. The operating system can retain mapped pages in its
cache. Stacking recomputes base features as needed.

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
