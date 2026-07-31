"""agent.benchmarks: optimizer and model benchmarks for the rexgraph.nn substrate.

The Track-2 credibility harness: HodgeAdam vs Adam/AdamW/SGD on recognized tasks, with a fair,
lr-tuned, verdict-producing A/B. Registered benchmarks: ill-cond (controlled ill-conditioning,
the diagnostic that tests the per-Hodge-component preconditioning claim), matrix-completion,
bilinear-game (a purely rotational field), and mnist/fashion-mnist/cifar10 (HuggingFace datasets).
"""
from .optimizers import (  # noqa: F401
    register_benchmark, benchmarks, run_benchmark, benchmark_ab,
)
