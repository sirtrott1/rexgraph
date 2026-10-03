"""Fair data streams, real device failures and independently reloadable benchmark models."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from agent.benchmarks.bench_local_models import (  # noqa: E402
    Config, Experiment, SDPAttention, resolve_device, train,
)
from agent.benchmarks.bench_associative_recall import StandardAttention  # noqa: E402


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_fused_bidirectional_baseline_matches_existing_operator():
    standard = StandardAttention(16, 4)
    fused = SDPAttention(16, 4)
    fused.load_state_dict(standard.state_dict())
    x = torch.randn(3, 9, 16)
    a, b = standard(x)[0], fused(x)[0]
    torch.testing.assert_close(a, b)
    # Verify parameter gradients too: a speed baseline must preserve training semantics.
    a.sum().backward()
    b.sum().backward()
    for p, q in zip(standard.parameters(), fused.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)


def test_fused_causal_baseline_does_not_read_future_tokens():
    fused = SDPAttention(16, 4, causal=True)
    x = torch.randn(2, 8, 16)
    changed = x.clone()
    changed[:, 5:] += 10
    torch.testing.assert_close(fused(x)[0][:, :5], fused(changed)[0][:, :5])


def test_explicit_gpu_request_cannot_silently_run_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        resolve_device("cuda")


def test_data_stream_is_shared_across_architectures_and_evaluation_isolated():
    c = Config(dim=16, layers=1, batch=8)
    a = Experiment(replace(c, model="sdpa"), 3, torch.device("cpu"))
    first = a.batch()
    rng_before = torch.random.get_rng_state()
    validation = a.evaluate("validation", 32)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert a.evaluate("validation", 32) == validation
    next_a = a.batch()
    b = Experiment(c, 3, torch.device("cpu"))
    torch.testing.assert_close(b.batch()[0], first[0])
    b.evaluate("test", 32)
    next_b = b.batch()
    for x, y in zip(next_a, next_b, strict=True):
        torch.testing.assert_close(x, y)
    # Observe actual split data, not just nominal names or metric differences.
    from agent.benchmarks.bench_associative_recall import make_batch
    torch.manual_seed(20003)
    val_data = make_batch(32, 6, 8, 16, "cpu")[0]
    torch.manual_seed(30003)
    test_data = make_batch(32, 6, 8, 16, "cpu")[0]
    assert not torch.equal(val_data, test_data)


@pytest.mark.parametrize("task,model", [("cochain", "greens"), ("recall", "propagator"),
                                       ("causal", "propagator"), ("relational", "green")])
def test_training_checkpoints_reload_independently(tmp_path, task, model):
    c = Config(task=task, model=model, dim=16, layers=1, batch=4,
               groups=4, arity=8, green_iters=4, order=4)
    r = train(c, seed=0, device=torch.device("cpu"), steps=2, eval_count=8,
              checkpoint_dir=tmp_path / task, log_every=1)
    assert r["steps"] == 2 and r["train_seconds"] > 0
    assert r["checkpoint_verified"] is True
    assert r["test"]["examples"] >= 8


def test_cochain_training_indices_are_disjoint_from_validation_and_test():
    c = Config(task="cochain", model="greens", groups=4, arity=8)
    experiment = Experiment(c, 0, torch.device("cpu"))
    train_ids, val_ids, test_ids = [set(x.tolist()) for x in experiment.indices]
    assert not train_ids & val_ids and not train_ids & test_ids and not val_ids & test_ids
    assert train_ids | val_ids | test_ids == set(range(32))
