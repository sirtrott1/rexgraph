"""Reproducible local training and timing for RexGraph's existing model architectures.

The synthetic tasks qualify mechanisms and performance at the recorded shapes; they
are not language model perplexity or evidence of superiority on production datasets.
Run ``python -m agent.benchmarks.bench_local_models --help`` for the bounded CLI.
ROCm uses PyTorch's ``cuda`` device name. Explicit unavailable devices fail.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from agent.benchmarks import bench_associative_recall as recall
from agent.benchmarks import bench_relational_model as relational
from rexgraph.flow.cochain import CoParticipationCochain
from rexgraph.graph import RexGraph
from rexgraph.nn import make_optimizer
from rexgraph.nn.relational_attention import CausalPropagatorAttention, PropagatorAttention


@dataclass(frozen=True)
class Config:
    task: str = "recall"
    model: str = "propagator"
    dim: int = 64
    layers: int = 2
    batch: int = 128
    order: int = 16
    hops: int = 4
    window: int = 8
    green_iters: int = 20
    groups: int = 64
    arity: int = 32
    dtype: str = "float32"
    precision: str = "float32"
    graphs: str = "regular"
    target: str = "triangles"
    pairs: int = 6


class SDPAttention(recall.StandardAttention):
    """Same projections as the existing baseline, using PyTorch's SDPA kernel."""

    def __init__(self, d, n_head, *, causal=False):
        super().__init__(d, n_head)
        self.causal = causal

    def forward(self, x, return_diag=False):
        b, t, d = x.shape
        q, k, v = (z.reshape(b, t, self.h, self.dk).transpose(1, 2)
                   for z in self.qkv(x).chunk(3, -1))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.proj(y.transpose(1, 2).reshape(b, t, d)), None


def resolve_device(name):
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("this benchmark qualifies CPU and CUDA/ROCm devices")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm requested but no GPU is visible; CPU fallback is disabled")
        # Availability alone does not prove the installed wheel has working kernels.
        torch.ones(8, device=device).square().sum().item()
    return device


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def provenance(device):
    import agent
    import rexgraph
    from rexgraph.core import _laplacians
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    return {
        "interpreter": sys.executable, "python": platform.python_version(),
        "torch": torch.__version__, "torch_path": torch.__file__, "hip": torch.version.hip,
        "rexgraph_path": rexgraph.__file__, "compiled_path": _laplacians.__file__,
        "agent_path": agent.__file__,
        "package_versions": {n: metadata.version(n) for n in ("rexgraph", "rexgraph-agent")},
        "device": str(device), "gpu_visible": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu": cpu, "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "mkldnn_available": torch.backends.mkldnn.is_available(),
    }


class Experiment:
    """Owns model, data streams and optimizer. Validation and test seeds are disjoint."""

    def __init__(self, config, seed, device):
        self.c, self.seed, self.device = config, seed, device
        c = config
        if c.dim < 4 or c.dim % 4 or c.layers < 1 or c.batch < 1:
            raise ValueError("dim must be a positive multiple of 4; layers and batch must be positive")
        if c.precision not in {"float32", "bfloat16"} or c.dtype not in {"float32", "float64"}:
            raise ValueError("unsupported precision or dtype")
        if c.task != "cochain" and c.dtype != "float32":
            raise ValueError("feature models use float32 parameters; precision selects autocast")
        if c.task == "cochain" and c.precision != "float32":
            raise ValueError("cochain sparse solves require float32 or float64, without autocast")
        self.rng = np.random.default_rng(seed + 10000)
        torch.manual_seed(seed)
        if c.task in {"recall", "causal"}:
            if not 1 <= c.pairs <= 8:
                raise ValueError("associative recall requires 1..8 distinct keys")
            if c.task == "recall":
                factories = {
                    "sdpa": lambda: SDPAttention(c.dim, 4),
                    "standard": lambda: recall.StandardAttention(c.dim, 4),
                    "propagator": lambda: PropagatorAttention(c.dim, 4, cheb_order=c.order),
                    "importance": lambda: PropagatorAttention(c.dim, 4, cheb_order=c.order, importance=True),
                }
            else:
                factories = {
                    "sdpa": lambda: SDPAttention(c.dim, 4, causal=True),
                    "propagator": lambda: CausalPropagatorAttention(c.dim, 4, hops=c.hops,
                                                                     window=c.window),
                    "dense": lambda: CausalPropagatorAttention(c.dim, 4, hops=c.hops,
                                                                window=c.window, sparse=False),
                }
            self.model = recall.Encoder(25, 2 * c.pairs + 2, c.dim, 4, c.layers, factories[c.model])
            lr = 3e-3
        elif c.task == "relational":
            factories = {
                "pairwise": lambda: relational.PairwiseGNN(c.dim, c.layers),
                "complex": lambda: relational.ComplexNet(c.dim, c.layers, True),
                "faces_off": lambda: relational.ComplexNet(c.dim, c.layers, False),
                "green": lambda: relational.GreenComplexNet(c.dim, c.layers, True),
                "lagrangian": lambda: relational.LagrangianGreenNet(c.dim, c.layers, True),
            }
            self.model = factories[c.model]()
            lr = 3e-3
        elif c.task == "cochain":
            if c.arity < 4 or c.groups < 4 or c.green_iters < 1:
                raise ValueError("cochain requires groups >= 4, arity >= 4, green_iters >= 1")
            src = np.repeat(np.arange(c.groups, dtype=np.int32), c.arity)
            tgt = c.groups + np.arange(len(src), dtype=np.int32)
            rex = RexGraph(sources=src, targets=tgt)
            self.labels = torch.tensor(np.repeat(np.arange(c.groups) % 4, c.arity),
                                       device=device, dtype=torch.long)
            splits = [[], [], []]
            split_rng = np.random.default_rng(seed + 70000)
            for g in range(c.groups):
                ids = split_rng.permutation(c.arity) + g * c.arity
                n = max(1, c.arity // 4)
                for dest, portion in zip(splits, (ids[:n], ids[n:2*n], ids[2*n:]), strict=True):
                    dest.extend(portion.tolist())
            self.indices = [torch.tensor(ids, device=device, dtype=torch.long) for ids in splits]
            self.model = CoParticipationCochain(rex, 4, green_iters=c.green_iters,
                                               dtype=getattr(torch, c.dtype))
            if c.model not in {"greens", "adam"}:
                raise ValueError("cochain model must be greens or adam")
            lr = 0.3
        else:
            raise ValueError(f"unknown task {c.task!r}")
        self.model.to(device)
        if c.task == "cochain" and c.model == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.optimizer_name = "Adam (structure blind cochain control)"
        else:
            self.optimizer, self.optimizer_name = make_optimizer("auto", self.model,
                                                                  self.model.parameters(), lr=lr)
        # These models have no stochastic forward operations. Reset AFTER initialization
        # so every architecture receives the same training sequences for this seed.
        torch.manual_seed(seed + 10000)

    def autocast(self):
        if self.c.precision == "bfloat16":
            return torch.autocast(self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def batch(self, size=None, rng=None):
        c = self.c
        size = c.batch if size is None else size
        if c.task in {"recall", "causal"}:
            # Generate on CPU for identical data across devices. Include transfer in
            # end to end timing; static batch timing isolates device training cost.
            seq, tgt, _, _ = recall.make_batch(size, c.pairs, 8, 16, "cpu")
            return seq.to(self.device), tgt.to(self.device)
        if c.task == "relational":
            return relational.make_batch(size, self.rng if rng is None else rng, self.device,
                                          target=c.target, graphs=c.graphs)
        return None

    def prediction(self, batch):
        if self.c.task == "cochain":
            return self.model()
        x, _ = batch
        y = self.model(x)
        return y[:, -1, :] if self.c.task in {"recall", "causal"} else y

    def loss(self, batch):
        out = self.prediction(batch)
        if self.c.task == "cochain":
            ids = self.indices[0]
            return F.cross_entropy(out[ids], self.labels[ids])
        _, y = batch
        if self.c.task == "relational":
            return F.mse_loss(out, y)
        return F.cross_entropy(out, y)

    def step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            loss = self.loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def evaluate(self, split, count=1024):
        if split not in {"validation", "test"} or count < 1:
            raise ValueError("evaluation requires a named held-out split and positive count")
        was = self.model.training
        self.model.eval()
        try:
            if self.c.task == "cochain":
                ids = self.indices[1 if split == "validation" else 2]
                out = self.model()[ids]
                y = self.labels[ids]
                return {"accuracy": float((out.argmax(-1) == y).float().mean()),
                        "loss": float(F.cross_entropy(out, y)), "examples": len(ids)}
            seed = self.seed + (20000 if split == "validation" else 30000)
            rng = np.random.default_rng(seed)
            outputs, targets = [], []
            # Evaluation must not consume the training RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                for start in range(0, count, self.c.batch):
                    batch = self.batch(min(self.c.batch, count - start), rng)
                    with self.autocast():
                        outputs.append(self.prediction(batch).float().cpu())
                    targets.append(batch[1].cpu())
            out, y = torch.cat(outputs), torch.cat(targets)
            if self.c.task == "relational":
                mse = float(F.mse_loss(out, y))
                variance = float(((y - y.mean()) ** 2).mean())
                return {"mse": mse, "r2": 1.0 - mse / variance if variance > 0 else None,
                        "target_variance": variance, "examples": count}
            return {"accuracy": float((out.argmax(-1) == y).float().mean()),
                    "loss": float(F.cross_entropy(out, y)), "examples": count}
        finally:
            self.model.train(was)


def timed_steps(experiment, *, warmup=3, iterations=10):
    """Isolate static batch forward/backward/update time, never model initialization."""
    batch = experiment.batch()
    for _ in range(warmup):
        experiment.step(batch)
    sync(experiment.device)
    if experiment.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(experiment.device)
    elapsed = []
    for _ in range(iterations):
        sync(experiment.device)
        t0 = time.perf_counter()
        loss = experiment.step(batch)
        sync(experiment.device)
        elapsed.append((time.perf_counter() - t0) * 1000)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training objective")
    ms = statistics.median(elapsed)
    samples = len(experiment.indices[0]) if experiment.c.task == "cochain" else experiment.c.batch
    return {"step_ms_median": ms, "step_ms_min": min(elapsed), "step_ms_max": max(elapsed),
            "examples_per_second": samples * 1000 / ms, "timed_steps": iterations,
            "warmup_steps": warmup, "parameter_count": sum(p.numel() for p in experiment.model.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in experiment.model.parameters()),
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "gpu_peak_allocated_mib": (torch.cuda.max_memory_allocated(experiment.device) / 2**20
                                       if experiment.device.type == "cuda" else None)}


def train(config, *, seed, device, steps, eval_count=1024, checkpoint_dir=None, log_every=100):
    experiment = Experiment(config, seed, device)
    start_metrics = experiment.evaluate("validation", eval_count)
    trajectory = []
    train_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, steps, log_every):
        sync(device)
        t0 = time.perf_counter()
        for _ in range(start, min(steps, start + log_every)):
            loss = experiment.step(experiment.batch())
        sync(device)
        train_seconds += time.perf_counter() - t0
        if not torch.isfinite(loss) or not all(torch.isfinite(p).all() for p in experiment.model.parameters()):
            raise FloatingPointError("nonfinite training objective or parameters")
        row = {"step": min(steps, start + log_every), "train_seconds": train_seconds,
               "train_loss": float(loss), "validation": experiment.evaluate("validation", eval_count)}
        trajectory.append(row)
        print(json.dumps({"progress": config.model, "task": config.task, "seed": seed, **row}), flush=True)
    validation = experiment.evaluate("validation", eval_count)
    # This is final checkpoint test quality; no test based checkpoint selection.
    test = experiment.evaluate("test", eval_count)
    result = {"config": asdict(config), "seed": seed, "steps": steps,
              "optimizer": experiment.optimizer_name, "validation_start": start_metrics,
              "validation": validation, "test": test, "trajectory": trajectory,
              "train_seconds": train_seconds,
              "end_to_end_step_ms": 1000 * train_seconds / steps,
              "examples_per_second_end_to_end": (len(experiment.indices[0]) if config.task == "cochain"
                                                 else config.batch) * steps / train_seconds,
              "parameter_count": sum(p.numel() for p in experiment.model.parameters()),
              "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "gpu_peak_allocated_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                                         if device.type == "cuda" else None)}
    if checkpoint_dir is not None:
        from safetensors.torch import load_file, save_file
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        path = checkpoint_dir / "model.safetensors"
        if config.task == "cochain":
            experiment.model.save_safetensors(str(path))
            restored = CoParticipationCochain.load_safetensors(str(path)).to(device)
            assert torch.equal(experiment.model.Z, restored.Z)
            original_adj, restored_adj = experiment.model._adj.coalesce(), restored._adj.coalesce()
            assert original_adj.shape == restored_adj.shape
            assert torch.equal(original_adj.indices(), restored_adj.indices())
            assert torch.equal(original_adj.values(), restored_adj.values())
        else:
            save_file({n: x.detach().cpu().contiguous() for n, x in experiment.model.state_dict().items()}, str(path))
            # Construct an independent instance and actually evaluate the loaded weights.
            restored = Experiment(config, seed, device)
            restored.model.load_state_dict(load_file(str(path), device=str(device)))
            assert restored.evaluate("test", eval_count) == test
        (checkpoint_dir / "config.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        result["checkpoint"] = str(path)
        result["checkpoint_verified"] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "speed"), default="train")
    parser.add_argument("--task", choices=("cochain", "recall", "causal", "relational"), default="recall")
    parser.add_argument("--models", default=None, help="comma-separated architectures")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (also ROCm); no fallback")
    parser.add_argument("--threads", default="1", help="comma-separated CPU counts (speed mode only)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--eval-count", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--hops", type=int, default=4)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--green-iters", type=int, default=20)
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--arity", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--graphs", choices=("er", "regular"), default="regular")
    parser.add_argument("--target", choices=("triangles", "4cycles"), default="triangles")
    parser.add_argument("--output", type=Path, required=True, help="new JSON result path; checkpoints alongside")
    args = parser.parse_args(argv)
    threads = [int(n) for n in args.threads.split(",")]
    seeds = [int(n) for n in args.seeds.split(",")]
    if min(threads) < 1 or args.steps < 1 or args.iterations < 1 or args.eval_count < 1:
        parser.error("threads, steps, iterations and eval count must be positive")
    if args.mode == "train" and len(threads) != 1:
        parser.error("training requires one thread setting; use speed mode to tune threads")
    if args.output.exists() or args.output.with_suffix(".checkpoints").exists():
        parser.error("output or checkpoint directory already exists; choose a new path")
    device = resolve_device(args.device)
    torch.set_num_interop_threads(1)
    if args.task == "relational":
        relational._verify_ops()
    defaults = {"cochain": "adam,greens", "recall": "sdpa,propagator,importance",
                "causal": "sdpa,propagator", "relational": "pairwise,faces_off,complex,green,lagrangian"}
    models = (args.models or defaults[args.task]).split(",")
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "measurements": results,
                "scope": "Synthetic mechanism qualification; final test is never used for tuning."
                         " CPU RSS is process high-water memory including imported libraries."}
    for nthreads in threads:
        torch.set_num_threads(nthreads)
        for name in models:
            config = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__ if k != "model"}, model=name)
            for seed in (seeds if args.mode == "train" else seeds[:1]):
                if args.mode == "speed":
                    experiment = Experiment(config, seed, device)
                    result = {"config": asdict(config), "seed": seed,
                              **timed_steps(experiment, iterations=args.iterations)}
                else:
                    ckpt = args.output.with_suffix(".checkpoints") / f"{name}-seed-{seed}"
                    result = train(config, seed=seed, device=device, steps=args.steps,
                                   eval_count=args.eval_count, checkpoint_dir=ckpt)
                result["environment"] = provenance(device)
                results.append(result)
                print(json.dumps({"result": result}, allow_nan=False), flush=True)
                # Write after each run so completed measurements survive interruption.
                temporary = args.output.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
                temporary.replace(args.output)
    return document


if __name__ == "__main__":
    main()
