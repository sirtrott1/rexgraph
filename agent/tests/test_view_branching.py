"""The agent views on a branching complex.

The views read the library's neighborhood operators, so they inherited the pairwise
`_v2e`: a vertex past the second in a k-ary relation reported no relations and no
neighbours, and its exact rational coordinate was the degenerate default rather than a
position derived from its star. That matters most for the renderer, where the whole point
is that position is structure.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from agent.graph_view import exact_positions, neighbors, render_payload
from rexgraph.graph import RexGraph

LOST = 3


@pytest.fixture
def rex():
    """A 4-ary relation over {0,1,2,3} with 2-ary legs. Vertex 3 is only in the wide one."""
    r = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 4], dtype=np.int32))
    r._ensure_clean()
    return r


def test_the_vertex_has_neighbours(rex):
    """It reported `{"edges": [], "vertices": [3]}`: incident to nothing, adjacent to
    itself."""
    out = neighbors(rex, LOST)
    assert out["edges"] == [0]
    assert out["vertices"] == [0, 1, 2, 3]


def test_it_is_not_parked_at_the_degenerate_coordinate(rex):
    """With an empty star both channel parameters fell to 1, and both axes read their
    parameter through the cosine, which is 0 there: it sat at exactly (0, 0) whatever the
    complex around it looked like."""
    cell = exact_positions(rex)["cells"][LOST]
    assert (Fraction(cell["x"]), Fraction(cell["y"])) != (Fraction(0), Fraction(0))
    assert Fraction(cell["x"]) == Fraction(-3225, 8633)
    assert Fraction(cell["y"]) == Fraction(135, 377)


def test_its_coordinate_is_still_exact(rex):
    """The fix moves the value, not the arithmetic: no sqrt entered on the way."""
    cell = exact_positions(rex)["cells"][LOST]
    assert float(Fraction(cell["x"])) == pytest.approx(cell["at"][0])
    assert float(Fraction(cell["y"])) == pytest.approx(cell["at"][1])


def test_structurally_different_vertices_do_not_stack_on_the_default():
    """The symptom as it would show in a picture, and the reason it was easy to miss:
    stacking is CORRECT for vertices with equal stars, so the wrong kind has to be
    separated from the right kind to see it.

    Two relations {0,1,2,3} and {0,1,4,3} plus a leg {4,5}. Vertex 2 is in one relation
    and vertex 3 in two, so they are genuinely different cells. Both appear only past the
    second position, so the pairwise path made both isolated and drew them on the same
    (0, 1). Vertices 0, 1 and 3 DO share a position here, and should: their stars are
    equal.
    """
    rex = RexGraph.from_hypergraph(
        np.array([0, 4, 8, 10], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 4, 3, 4, 5], dtype=np.int32))
    rex._ensure_clean()
    at = [(Fraction(c["x"]), Fraction(c["y"])) for c in exact_positions(rex)["cells"]]
    assert at[2] != at[3], "two different cells drew on one point"
    assert (Fraction(0), Fraction(0)) not in at
    assert at[0] == at[1] == at[3], "equal stars should still land together"


def test_the_render_payload_carries_the_corrected_positions(rex):
    payload = render_payload(rex, limit=50)
    cell = payload["positions"]["exact"]["cells"][LOST]
    assert Fraction(cell["x"]) == Fraction(-3225, 8633)
