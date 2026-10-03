The local model harness measures training quality and speed for the existing
cochain, edge/face, Green, Lagrangian, and attention architectures:

```sh
python -m agent.benchmarks.bench_local_models --task cochain --models adam,greens \
  --steps 100 --threads 1 --output runs/cochain.json
python -m agent.benchmarks.bench_local_models --task relational --dim 32 --batch 32 \
  --steps 500 --output runs/relational.json
python -m agent.benchmarks.bench_local_models --task recall --steps 1000 \
  --output runs/recall.json
python -m agent.benchmarks.bench_local_models --task causal --steps 1000 \
  --output runs/causal.json
```

Install the appropriate PyTorch build and the Core/Agent optional model
dependencies in your own environment. These commands do not download datasets.
Each output path must be new; JSON is updated after every completed run, and
independently verified model checkpoints are saved beside it. Checkpoints contain
model weights and configuration, without optimizer continuation. The native
cochain artifact also contains its source complex. Validation and test use
separate seeds/edge masks; reported test quality belongs to the final checkpoint.
Use validation for configuration selection. Training data streams are identical
across compared architectures for a given seed and batch size.

Tune CPU threads for the actual shape before training. More threads can be
slower on small models; the harness defaults to one thread and one interop
thread, without changing library or host defaults:

```sh
python -m agent.benchmarks.bench_local_models --mode speed --task recall \
  --models sdpa,propagator --threads 1,2,4,8,16,32 --iterations 20 \
  --output runs/cpu-threads.json
```

Speed mode measures forward/backward/optimizer updates on a static batch after
warmup. Training mode also includes synthetic data generation and device transfer,
and excludes held out evaluation and checkpoint writing from its training time.
CPU memory is process high water RSS, including imported libraries; it is not
the model's isolated allocation. GPU memory is Torch peak allocated memory.
Results include interpreter, package, compiled module, backend and thread paths.
The `sdpa` attention baseline uses PyTorch's fused operator with the same causal
or bidirectional semantics as its comparison. Parameter counts are reported;
architectures are not claimed to be parameter matched.

ROCm uses `--device cuda` too. Run the same commands on a host that exposes
its GPU device nodes, then tune GPU batch size separately:

```sh
python -m agent.benchmarks.bench_local_models --task causal --device cuda \
  --batch 256 --steps 1000 --output runs/gpu-causal.json
```

An explicit unavailable GPU fails without falling back to CPU. `--precision
bfloat16` is an optional numerical tradeoff for feature models; qualify learning
quality on your backend before selecting it. Native sparse cochains instead
support `--dtype float32` or `float64`. Shortening their `--green-iters` or attention
`--order` also needs a solve/polynomial accuracy check on the intended complex.

These synthetic tasks establish local behavior, not production model quality.
Hub labels intentionally match coparticipation (transductive homophily).
Triangle counting intentionally supplies triangle faces, and can be solved
exactly by counting those faces without a neural network. The regular graph
pairwise control has identical constant node features and fixed degree;
differences reflect the information supplied to each architecture. `--target
4cycles` tests a count not supplied directly as the face count. Recall tests key
binding, not language model perplexity; causal recall is not a text generation
throughput benchmark. Use a representative held out dataset for product claims.

The custom HIP qualification harness compares packed integer/float64 products
and canonical/declared channel towers with the compiled Core CPU kernels:

```sh
python -m agent.benchmarks.bench_hip_ternary --device hip --rows 4096 --cols 4097 \
  --threads 4 --iterations 50 --contention --output runs/hip-kernels.json
python -m agent.benchmarks.bench_hip_ternary --device cpu --rows 4096 --cols 4097 \
  --threads 4 --output runs/cpu-kernels.json
```

HIP requires a visible GPU and the optional `lib_ternary_hip.so` built from the
current Core source for its gfx architecture. Core's Meson build discovers
`hipcc`; `-Csetup-args=-Dhip_arch=gfx1151` explicitly selects that architecture
on a Radeon 8060S build host. Build/install in a separate environment.
`REXGRAPH_TERNARY_HIP` can name an independently built library. A HIP request
without working kernels records failure and exits 1. CPU only results explicitly
leave GPU qualification false. Rebuild native HIP after kernel source changes;
editing the `.hip` file does not update an existing binary.

Resident timing uploads the operator once; one shot timing includes its upload
and allocation on every product. Both include vector transfers, host output and
synchronization. Those NumPy APIs do not expose Torch autograd. For fixed ternary
operators, `rexgraph.nn.PackedTernaryLinear` provides resident HIP float32/float64
forward and input gradient kernels on the active Torch stream. Its packed weights
are fixed; it does not train ternary weights through a straight through estimator.
The Vulkan launcher serves llama.cpp inference. Passing NumPy output from HIP into
a training model does not make that operation differentiable.

Check device visibility independently of training qualification:

```bash
python -m rexgraph.gpu_preflight --diagnose --json
python -m rexgraph.gpu_preflight --json
```

The first command enumerates HIP and Vulkan and reports Linux device open errors,
visibility filters, and the process's `/dev` mount. It does not qualify arithmetic.
HIP on Linux requires access to `/dev/kfd` and the GPU's `/dev/dri/renderD*` node.
A sandbox can expose both runtime libraries while hiding those devices and sysfs;
an empty device list in that process does not establish that the host lacks a GPU.
Vulkan discovery uses the loader directly and does not require `vulkaninfo`.

The contention measurement runs independent CPU and HIP products in parallel
and compares equal work with sequential execution and the fastest device alone.
It measures a fixed equal split, without determining the optimal partition.
Sweep thread counts and representative shapes on the target host. The existing
`agent.device_routing` planner accepts measured worker rates for independent
tasks; it does not partition one optimizer update across CPU and GPU. Unified
memory shares bandwidth, so using both devices does not guarantee a speedup.

At a fixed model shape and batch size, a larger dataset mostly increases the
number of steps per epoch. Larger models and contexts increase each step's
cost. Current dataset bundles commonly materialize their arrays/operators, so
dataset size also matters for RAM; this harness does not qualify an out of core
training pipeline. Sparse cochain models use a fixed complex and have parameters
per cell. Their coparticipation adjacency can grow with squared hub degree.
Dense attention grows quadratically with context length; the banded path reduces
the interaction count but still needs benchmarking against optimized attention.
