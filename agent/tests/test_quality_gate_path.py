"""Check that the context quality gate reads Core's interfacing bundle."""
from __future__ import annotations

import numpy as np
import pytest

from agent.pipeline_runner import GATE_OK, GATE_REFUSE, _context_quality_gate
from rexgraph.graph import RexGraph


def _complex(n_edges=40, n_vertices=18, seed=5):
    rng = np.random.RandomState(seed)
    src = rng.randint(0, n_vertices, n_edges).astype(np.int32)
    tgt = ((src + 1 + rng.randint(0, 4, n_edges)) % n_vertices).astype(np.int32)
    return RexGraph(sources=src, targets=tgt)


LABELS = [f"term{i}" for i in range(18)]


def test_the_gate_scores_a_query_that_names_vertices():
    rex = _complex()
    out = _context_quality_gate(rex, LABELS, "term0 term3 term7 explain")
    assert out["verdict"] in (GATE_OK, "warn"), out
    assert out["score"] is not None, f"no score produced: {out}"
    assert out["n_shared"] == 3, out


def test_the_gate_refuses_a_query_that_names_nothing():
    rex = _complex()
    out = _context_quality_gate(rex, LABELS, "kangaroo helicopter mango")
    assert out["verdict"] == GATE_REFUSE and out["n_shared"] == 0, out


def test_the_score_is_the_librarys_own_interfacing_score():
    """Not a lookalike computed from different operators."""
    from rexgraph.core._interfacing import quality_gate
    rex = _complex()
    out = _context_quality_gate(rex, LABELS, "term0 term3 term7")
    bundle = rex.interfacing_vector(np.array([0, 3, 7], np.int32), np.ones(3), None)
    want = float(np.asarray(quality_gate(
        np.asarray(bundle["scores"], np.float64).reshape(1, -1))).mean())
    assert out["score"] == pytest.approx(want), (out["score"], want)


def test_the_gate_still_scores_when_no_dense_eigenbasis_exists():
    """Sparse dispatch returns a quality score without a full dense L0 eigenbasis."""
    rex = _complex(n_edges=900, n_vertices=400, seed=9)
    out = _context_quality_gate(rex, [f"term{i}" for i in range(400)],
                                "term0 term11 term250")
    assert out["score"] is not None, f"gate gave up: {out}"
    assert "unavailable" not in " ".join(out["reasons"]), out
