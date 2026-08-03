"""agent.benchmarks: optimizer and model benchmarks for the rexgraph.nn substrate.

The Track-2 credibility harness: any two registered optimizers against each other on recognized
tasks, with a fair, lr-tuned, verdict-producing A/B. `benchmark_ab` defaults to the ("hodge",
"adam") pair because those are the two arms the harness was built to settle; a single
`run_benchmark` with no optimizer named takes the routing default. Registered benchmarks: ill-cond
(controlled ill-conditioning, the diagnostic that tests the per-Hodge-component preconditioning
claim), matrix-completion, bilinear-game (a purely rotational field), and
mnist/fashion-mnist/cifar10 (HuggingFace datasets).
"""
from .optimizers import (  # noqa: F401
    benchmark_ab,
    benchmarks,
    register_benchmark,
    run_benchmark,
)
