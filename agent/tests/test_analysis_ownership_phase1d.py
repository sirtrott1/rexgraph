"""The live Agent owns analysis orchestration; legacy RexGraph analysis is not a dependency."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import agent
from agent.analysis import analyze
from agent.auto import auto_analyze
from agent.adapters import EdgeConstruction
from rexgraph import RexGraph


def test_agent_has_no_executable_rexgraph_analysis_imports():
    root = Path(agent.__file__).resolve().parent
    bad = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "rexgraph.analysis" or n.startswith("rexgraph.analysis.") for n in names):
                bad.append((path.name, node.lineno))
    assert bad == []


def test_agent_analysis_runs_native_pipeline_and_preserves_labels():
    rex = RexGraph(
        sources=np.array([0, 1], dtype=np.int32),
        targets=np.array([1, 2], dtype=np.int32),
    )
    events = []
    out = analyze(
        rex,
        depth="quick",
        labels=["alpha", "beta", "gamma"],
        draw=False,
        on_stage=lambda name, result: events.append((name, result)),
    )
    assert out["construction"] == {
        "nV": 3,
        "nE": 2,
        "nF": 0,
        "dimension": 1,
        "chain_valid": True,
    }
    assert {"construction", "topology", "spectral", "drawing"} <= out.keys()
    assert events and events[0][0] == "construction"


def test_agent_analysis_rejects_unknown_depth_before_dispatch():
    rex = RexGraph(
        sources=np.array([], dtype=np.int32),
        targets=np.array([], dtype=np.int32),
    )
    try:
        analyze(rex, depth="legacy", draw=False)
    except ValueError as exc:
        assert "unknown analysis depth" in str(exc)
    else:
        raise AssertionError("invalid analysis depth was accepted")


def test_auto_analyze_does_not_delegate_to_legacy_dashboard(monkeypatch):
    import rexgraph.analysis as legacy

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy rexgraph.analysis was called")

    monkeypatch.setattr(legacy, "analyze", forbidden)
    edges = EdgeConstruction(
        sources=np.array([0, 1], dtype=np.int32),
        targets=np.array([1, 2], dtype=np.int32),
        weights=np.ones(2, dtype=float),
        signs=np.ones(2, dtype=float),
        type_labels=np.zeros(2, dtype=np.int32),
        vertex_labels=["a", "b", "c"],
        n_types=1,
        type_names=["edge"],
    )
    out = auto_analyze(edges, depth="quick")
    assert out["construction"]["nV"] == 3
    assert "topology" in out and "spectral" in out
