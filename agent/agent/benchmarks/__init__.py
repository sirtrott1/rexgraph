"""Benchmark registry and execution for supported training optimizers.

Registered tasks include generated matrix regression and completion,
plus MNIST, Fashion MNIST and CIFAR10 classification. run_benchmark
returns training and evaluation metrics for a named task. benchmark_ab
compares optimizers using the same task inputs; its defaults are Adam
and AdamW. External image datasets are loaded only for those tasks.
"""
from .optimizers import (  # noqa: F401
    benchmark_ab,
    benchmarks,
    register_benchmark,
    run_benchmark,
)
