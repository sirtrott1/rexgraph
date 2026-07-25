"""agent.training_monitor: structural diagnosis of a loss trajectory + live watched training."""
import pytest

from agent import hive as hivemod, agent_complex
from agent.training_monitor import diagnose, TrainingMonitor


def test_diagnose_healthy_descent():
    assert diagnose([1.0, 0.8, 0.6, 0.4, 0.2])["status"] == "healthy"


def test_diagnose_not_learning_flat():
    d = diagnose([0.5] * 8)
    assert d["status"] == "not_learning" and d["fix"] == "raise_lr_then_switch_archetype"


def test_diagnose_diverging_nan_and_increasing():
    assert diagnose([1.0, 2.0, float("nan"), 5.0])["status"] == "diverging"
    assert diagnose([0.2, 0.4, 0.6, 0.9])["status"] == "diverging"     # loss climbing


def test_diagnose_converged():
    assert diagnose([1.0, 0.5, 0.3, 0.3, 0.3, 0.3])["status"] == "converged"


def test_diagnose_overfitting():
    d = diagnose([1.0, 0.6, 0.3, 0.1], val=[0.9, 0.6, 0.7, 0.85])   # train down, val up
    assert d["status"] == "overfitting" and d["fix"] == "reduce_capacity"


def test_train_watched_runs_and_registers(tmp_path):
    pytest.importorskip("torch")
    hivemod.reset_hive(); agent_complex.reset_live()
    h = hivemod.get_hive()
    r = TrainingMonitor(h).train_watched("net", "mlp", steps=20, device="cpu")
    assert r.get("registered") is True and h.get("net") is not None
    assert r["final"]["diagnosis"]["status"] in ("healthy", "converged", "not_learning", "unknown")
