"""The optimizer benchmark harness (agent.benchmarks) + the bench lifecycle phase."""
import pytest

pytest.importorskip("torch")

from agent import benchmarks, lifecycle


def test_benchmark_registry_and_run():
    names = {b["name"] for b in benchmarks.benchmarks()}
    assert {"ill-cond", "mnist", "cifar10", "matrix-completion"} <= names
    r = benchmarks.run_benchmark("ill-cond", optimizer="hodge", steps=40)
    assert r["optimizer_class"] == "HodgeAdam" and r["eval_final"] < r["eval_start"]   # it optimizes


def test_benchmark_ab_is_lr_tuned_and_verdicts():
    ab = benchmarks.benchmark_ab("ill-cond", optimizers=("hodge", "adam"), steps=40)
    assert "verdict" in ab and ab["best"] in ("hodge", "adam", "tie")


def test_bench_lifecycle_phase():
    assert "bench" in {p["name"] for p in lifecycle.phases()}
    rl = lifecycle.run("bench", benchmark="ill-cond", steps=30)
    assert rl.status == "ok"


def test_relational_model_gate_and_differentiable():
    """The edge-primary ComplexNet: the ∂²=0/sparse-vs-dense gate holds, it's differentiable end to
    end, and the faces-ablation + pairwise baseline build (the Track-1 comparison harness)."""
    import numpy as np
    import torch
    from agent.benchmarks import bench_relational_model as B
    assert B._verify_ops()                                   # sparse B1/B2 == dense, B1@B2 == 0
    m = B.ComplexNet(24, 2, use_faces=True)
    b, y = B.make_batch(16, np.random.default_rng(0), "cpu", target="triangles")
    torch.nn.functional.mse_loss(m(b), y).backward()
    assert any(p.grad is not None for p in m.parameters())   # differentiable through B1/B2 scatter
    assert B.ComplexNet(24, 2, use_faces=False) is not None  # faces-off ablation
    assert B.PairwiseGNN(24, 2) is not None                  # matched pairwise baseline


def test_intrinsic_2x2_and_associative_recall_run():
    """The 2x2 organ ablation and the associative-recall bench run end to end (tiny)."""
    from agent.benchmarks import bench_associative_recall as A
    from agent.benchmarks import bench_intrinsic_model as I
    acc, dt = I.run("standard", "adam", seed=0, steps=2, device="cpu")   # one cell of the 2x2
    assert 0.0 <= acc <= 1.0 and dt > 0
    a2, _ = A.run(lambda: A.StandardAttention(64, 4), seed=0, steps=2, device="cpu")
    assert 0.0 <= a2 <= 1.0
