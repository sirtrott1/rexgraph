"""Coherence has one entry point, and the O(nV*solve) read is not it.

The global Green's coherence costs one block-CG solve per vertex, and 22 agent call
sites reached for it to compute a scalar. These pin the contract that replaced them:
`coherence_kappa`/`coherence_mean` answer at any scale, `coherence_greens` is gated
and returns None rather than a substitute, and the two are different quantities
rather than an exact one and an approximation.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
from agent.metrics import (
    coherence_greens,
    coherence_greens_mean,
    coherence_kappa,
    coherence_mean,
    greens_budget,
)

from rexgraph.graph import RexGraph

AGENT_SRC = pathlib.Path(__file__).resolve().parents[1] / "agent"

# Deliberate users of the global read, each for a stated reason.
ALLOWED_GREENS_CALLERS = {
    "metrics.py",        # the primitive itself
    "diagnostics.py",    # a probe whose job is to exercise the real property
    "cell_view.py",      # reports both readings side by side, and says so
}


def _path(n=40):
    return RexGraph(sources=np.arange(n - 1, dtype=np.int32),
                    targets=np.arange(1, n, dtype=np.int32))


def _star(n=40):
    return RexGraph(sources=np.zeros(n - 1, dtype=np.int32),
                    targets=np.arange(1, n, dtype=np.int32))


def test_kappa_is_per_vertex_and_finite():
    for rex in (_path(), _star()):
        k = coherence_kappa(rex)
        assert k.shape == (int(rex.nV),)
        assert np.all(np.isfinite(k))


def test_mean_matches_the_vector():
    rex = _path()
    assert coherence_mean(rex) == pytest.approx(float(coherence_kappa(rex).mean()))


def test_empty_complex_does_not_raise():
    rex = RexGraph(sources=np.zeros(0, dtype=np.int32), targets=np.zeros(0, dtype=np.int32))
    assert coherence_kappa(rex).size == int(rex.nV)
    assert coherence_mean(rex, default=-1.0) in (0.0, -1.0)


def test_greens_returns_none_over_budget_rather_than_substituting():
    rex = _path(40)
    assert coherence_greens(rex, budget=10) is None
    assert coherence_greens_mean(rex, budget=10) is None
    # under budget it is computed, and it is NOT the local vector
    g = coherence_greens(rex, budget=0)
    assert g is not None and g.shape == (int(rex.nV),)


def test_greens_and_local_are_different_quantities():
    """Not an exactness gap: a caller that swaps one for the other changes the
    number it reports. The star is the clean witness (local is identically 1)."""
    rex = _star(40)
    local = coherence_kappa(rex)
    greens = coherence_greens(rex, budget=0)
    assert greens is not None
    assert not np.allclose(local, greens)


def test_budget_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("REXGRAPH_VERTEX_CHARACTER_MAX_NODES", "7")
    assert greens_budget() == 7
    assert coherence_greens(_path(40)) is None
    monkeypatch.setenv("REXGRAPH_VERTEX_CHARACTER_MAX_NODES", "not-a-number")
    assert greens_budget() == 1500


def test_signature_kappa_mean_is_the_scale_free_read():
    from agent.rcdb import structural_signature
    rex = _star(40)
    sig = structural_signature(rex)
    assert sig["coherence_method"] == "local"
    assert sig["kappa_mean"] == pytest.approx(round(coherence_mean(rex), 6))


def test_no_agent_module_reaches_past_the_primitive():
    """The wiring itself, so the next caller cannot re-introduce the hot path: no
    agent module outside ALLOWED_GREENS_CALLERS may touch `.coherence`."""
    offenders = []
    for f in AGENT_SRC.rglob("*.py"):
        if f.name in ALLOWED_GREENS_CALLERS:
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "coherence":
                offenders.append(f"{f.relative_to(AGENT_SRC)}:{node.lineno}")
    assert not offenders, (
        "these reach the O(nV*solve) read directly; use agent.metrics.coherence_kappa "
        f"/ coherence_mean / coherence_greens instead: {offenders}"
    )
