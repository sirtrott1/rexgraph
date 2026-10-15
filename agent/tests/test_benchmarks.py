"""The optimizer benchmark harness (agent.benchmarks) + the bench lifecycle phase."""
import pytest

pytest.importorskip("torch")

from agent import benchmarks, lifecycle


def test_benchmark_registry_and_run():
    names = {b["name"] for b in benchmarks.benchmarks()}
    assert {"ill-cond", "mnist", "cifar10", "matrix-completion"} <= names
    r = benchmarks.run_benchmark("ill-cond", optimizer="adam", steps=40)
    assert r["optimizer_class"] == "Adam" and r["eval_final"] < r["eval_start"]   # it optimizes


def test_benchmark_ab_is_lr_tuned_and_verdicts():
    ab = benchmarks.benchmark_ab("ill-cond", optimizers=("adamw", "adam"), steps=40)
    assert "verdict" in ab and ab["best"] in ("adamw", "adam", "tie")


def test_bench_lifecycle_phase():
    assert "bench" in {p["name"] for p in lifecycle.phases()}
    rl = lifecycle.run("bench", benchmark="ill-cond", steps=30)
    assert rl.status == "ok"
