"""Optimizer routing: every training path goes through make_optimizer("auto"), nothing names
HodgeAdam. The router returns GreensCochain for a cochain on a relational complex and plain Adam
for a feature-space model; those two arms are pinned here so a regression is a test failure, not a
silently demoted optimizer."""
import inspect
from pathlib import Path

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

import rexgraph.nn as R
from rexgraph.nn.optim import GreensCochain


def test_relational_bench_builds_through_make_optimizer(monkeypatch):
    """bench_relational_model.train builds its optimizer with make_optimizer("auto"), and
    ComplexNet (feature-space: no greens_groups) routes to plain Adam."""
    from agent.benchmarks import bench_relational_model as B
    seen = {}
    real = B.make_optimizer

    def spy(name, model, params, **kw):
        opt, label = real(name, model, params, **kw)
        seen.update(name=name, label=label, lr=kw.get("lr"))
        return opt, label

    monkeypatch.setattr(B, "make_optimizer", spy)
    B.train(lambda: B.ComplexNet(8, 1, use_faces=False), 0, 1, "cpu")
    assert seen["name"] == "auto"
    assert seen["label"] == "Adam(auto)"
    assert seen["lr"] == 3e-3                                # the benchmark's lr is preserved


def test_relational_bench_does_not_name_hodge():
    from agent.benchmarks import bench_relational_model as B
    src = inspect.getsource(B)
    assert "HodgeAdam" not in src and "HodgeSGD" not in src


def test_auto_routes_feature_space_model_to_adam():
    model = torch.nn.Linear(4, 2)
    opt, label = R.make_optimizer("auto", model, model.parameters(), lr=1e-3)
    assert type(opt) is torch.optim.Adam
    assert label == "Adam(auto)"


def test_auto_routes_cochain_to_greens():
    """A CoParticipationCochain exposes greens_groups(), so "auto" hands it GreensCochain."""
    from rexgraph.flow.cochain import CoParticipationCochain
    from rexgraph.graph import RexGraph
    rex = RexGraph(sources=np.array([0, 1, 2, 0], np.int32),
                   targets=np.array([1, 2, 3, 3], np.int32))
    model = CoParticipationCochain(rex, 2)
    opt, label = R.make_optimizer("auto", model, model.parameters(), lr=0.3)
    assert isinstance(opt, GreensCochain)
    assert label.startswith("GreensCochain(auto:")


def test_flow_package_never_names_hodge():
    """rexgraph.flow is the relational-native surface: HodgeAdam has no business in it."""
    import rexgraph.flow
    root = Path(rexgraph.flow.__path__[0])
    hits = [p.name for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".md")
            and ("HodgeAdam" in p.read_text() or "HodgeSGD" in p.read_text())]
    assert hits == []
