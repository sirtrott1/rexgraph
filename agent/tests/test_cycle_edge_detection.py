"""Cycle membership is read from the harmonic basis, not from a threshold on one flow.

Which edges carry the independent cycles is structural. The harmonic part of a chosen
flow is not: it is that flow's content in the harmonic space, its magnitude depends on the
frame, and it vanishes entirely when the flow is orthogonal to that space.
"""

from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph


def _orthogonal_case():
    """A 4-cycle with two edges reversed: beta_1 is 1, the all-ones harmonic part is 0."""
    return RexGraph.from_graph(sources=[0, 1, 3, 0], targets=[1, 2, 2, 3])


def test_the_old_threshold_would_report_no_edges_on_a_real_cycle():
    """The failure this fix removes, pinned so it cannot be reintroduced quietly."""
    rex = _orthogonal_case()
    assert int(rex.betti[1]) == 1

    _, _, harm = rex.hodge(np.ones(int(rex.nE), dtype=np.float64))
    assert np.abs(harm).max() < 1e-9, "the all-ones flow has no harmonic content here"
    assert np.where(np.abs(harm) > 1e-6)[0].size == 0, (
        "a threshold on this flow finds nothing, while beta_1 says there is a cycle"
    )


def test_detect_cycles_returns_the_edges_of_the_independent_cycle():
    pytest.importorskip("agent.integrations.langgraph_rex")

    rex = _orthogonal_case()
    from rexgraph.harmonic_sparse import harmonic_basis

    support = np.unique(np.asarray(harmonic_basis(rex).tocoo().row, dtype=np.int64))
    assert support.tolist() == [0, 1, 2, 3], (
        "every edge of this 4-cycle carries the independent cycle"
    )


def test_a_filled_cycle_is_not_a_hole():
    """The harmonic basis is the right one because it accounts for faces.

    The cycle basis spans ker(B1) alone, so it still reports a column for a triangle whose
    face has been filled. beta_1 is 0 there, and the harmonic basis agrees.
    """
    from rexgraph.harmonic_sparse import cycle_basis, harmonic_basis

    filled = RexGraph.from_simplicial(sources=[0, 0, 1], targets=[1, 2, 2],
                                      triangles=[[0, 1, 2]])
    assert int(filled.betti[1]) == 0
    assert harmonic_basis(filled).shape[1] == 0
    assert cycle_basis(filled).shape[1] == 1, (
        "the cycle basis would wrongly report a hole that a face has filled"
    )
