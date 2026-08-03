"""The context gate warns on thin coverage and refuses on none.

Three states, and which one applies is decided by what was measured:

    REFUSE  the query supplies terms and none name a vertex. Retrieval returned
            nothing, so an answer comes from the model alone.
    WARN    coverage is measurable and weak.
    OK      coverage is fine, or it could not be measured at all.

The last distinction is the one that matters: an unevaluated gate is not evidence
of a bad answer, so it must not refuse.
"""

import numpy as np
from agent.pipeline_runner import (
    GATE_OK,
    GATE_REFUSE,
    GATE_WARN,
    _context_quality_gate,
)

from rexgraph.graph import RexGraph

LABELS = ["alpha", "beta", "gamma", "delta"]


def _rex():
    return RexGraph(sources=np.array([0, 1, 2], np.int32),
                    targets=np.array([1, 2, 3], np.int32))


def test_no_shared_vocabulary_refuses():
    g = _context_quality_gate(_rex(), LABELS, "xxxx yyyy zzzz")
    assert g["verdict"] == GATE_REFUSE
    assert g["n_shared"] == 0
    assert g["reasons"]


def test_the_refusal_says_how_many_terms_missed():
    g = _context_quality_gate(_rex(), LABELS, "xxxx yyyy zzzz")
    assert "3 query terms" in g["reasons"][0]


def test_shared_vocabulary_does_not_refuse():
    g = _context_quality_gate(_rex(), LABELS, "alpha beta")
    assert g["verdict"] in (GATE_OK, GATE_WARN)
    assert g["n_shared"] == 2


def test_a_short_token_that_names_a_vertex_counts():
    """Identifiers are routinely one or two characters. Judging coverage by token
    length would report zero for a query that names its subject exactly."""
    short = ["p53", "AR", "ER"]
    rex = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32))
    g = _context_quality_gate(rex, short, "AR binding")
    assert g["verdict"] != GATE_REFUSE
    assert g["n_shared"] == 1


def test_punctuation_does_not_hide_a_match():
    g = _context_quality_gate(_rex(), LABELS, "what about alpha, and beta?")
    assert g["n_shared"] == 2


def test_a_query_with_no_usable_terms_is_not_a_refusal():
    """Nothing was measured, so there is no finding to refuse on."""
    g = _context_quality_gate(_rex(), LABELS, "a an of")
    assert g["verdict"] == GATE_OK
    assert g["n_shared"] is None


def test_an_empty_complex_is_not_a_refusal():
    g = _context_quality_gate(None, LABELS, "alpha")
    assert g["verdict"] == GATE_OK


def test_a_complex_with_no_relations_is_not_a_refusal():
    empty = RexGraph(sources=np.array([], np.int32), targets=np.array([], np.int32))
    assert _context_quality_gate(empty, LABELS, "alpha")["verdict"] == GATE_OK


def test_a_raising_gate_stays_permissive():
    """Refusing because the gate broke would turn an internal fault into a refusal."""
    class Exploding:
        nE = 5
        @property
        def spectral_bundle(self):
            raise RuntimeError("boom")
    g = _context_quality_gate(Exploding(), LABELS, "alpha beta")
    assert g["verdict"] == GATE_OK
    assert "could not be evaluated" in g["reasons"][0]


def test_the_verdict_shape_is_stable():
    for q in ("alpha beta", "xxxx yyyy", "a an"):
        g = _context_quality_gate(_rex(), LABELS, q)
        assert set(g) == {"verdict", "reasons", "score", "n_shared"}
        assert g["verdict"] in (GATE_OK, GATE_WARN, GATE_REFUSE)
