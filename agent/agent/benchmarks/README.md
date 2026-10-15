# Agent benchmarks

The benchmark registry runs supported optimizers on generated matrix tasks
and public image datasets. It returns training and evaluation metrics for
the supplied task configuration. The packed kernel check compares Core CPU
operations with the selected device implementation.

## Installation

Run `make install-ml` from the repository root to install PyTorch, datasets
and Pillow for the benchmark tasks. See the
[installation guide](../../../README.md#optional-runtimes) for device runtimes.

## Optimizer benchmarks

```python
from agent.benchmarks import benchmark_ab, run_benchmark

result = run_benchmark("ill-cond", optimizer="adam", steps=40, seed=0)
comparison = benchmark_ab("ill-cond", optimizers=("adam", "adamw"), steps=40)
```

The matrix regression and completion tasks generate their inputs from the
specified seed. MNIST, Fashion MNIST and CIFAR10 tasks load their named public
datasets when requested. Comparison runs use the same task inputs and select
a learning rate from the supplied grid. Returned measurements describe those
runs and are not stored in the source tree.

## Packed kernel checks

```sh
.venv/bin/python -I -m agent.benchmarks.bench_hip_ternary --device cpu --rows 128 --cols 257 \
  --threads 1 --iterations 3 --output /tmp/rexgraph-kernel-check.json
```

Use `--device hip` when a compatible GPU and the current native kernels are
available. An explicit unavailable device returns a failure. The output records
the selected backend, inputs, correctness comparisons and timings. Keep generated
outputs outside the checkout.

Device diagnosis is available through `python -m rexgraph.gpu_preflight`.
