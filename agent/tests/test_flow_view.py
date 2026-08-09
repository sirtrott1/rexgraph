"""Placing cells by a signal, for data that has no geometry of its own.

Semantics and measurements do not come with an embedding, and they are often structurally
degenerate besides. In a binding panel every ligand has one binding and one panel
membership, so their stars are identical and a layout that reads structure collapses them
onto one point. Measured on a real BindingDB panel: 37 vertices, 2 distinct positions,
because there are 2 distinct star characters. That picture is true and useless.

What such data does have is flow. The gradient part of a signal descends a potential,
`phi = L0^+ B1 g`, solved by the library's own LSQR seam, which deflates L0's
per-component constant kernel exactly. `phi` is a coordinate derived FROM the data rather
than invented for it, so two cells with the same structure and different measurements
separate, and they separate by how much the measurement differs.
"""
from __future__ import annotations

import numpy as np
import pytest

from agent.graph_view import exact_positions, flow_positions, render_payload
from agent.render_svg import render_svg
from rexgraph.graph import RexGraph


@pytest.fixture
def star():
    """Four leaves with four different measurements: no structure to tell them apart."""
    rex = RexGraph(sources=np.array([0, 0, 0, 0], dtype=np.int32),
                   targets=np.array([1, 2, 3, 4], dtype=np.int32))
    rex._ensure_clean()
    return rex


#### it separates what structure cannot


def test_structure_cannot_separate_the_leaves(star):
    """The premise. Identical stars, so identical positions, and the layout is right to
    say so."""
    cells = exact_positions(star)["cells"]
    assert len({(c["x"], c["y"]) for c in cells}) == 1


def test_the_flow_separates_them(star):
    flow = flow_positions(star, [1.0, 5.0, 2.0, 9.0])
    assert len({tuple(np.round(p, 9)) for p in flow["positions"]}) == 5


def test_they_separate_by_how_much_the_measurement_differs(star):
    """Not merely apart: in the right order and by the right amount. On a star the
    potential recovers the signal up to a constant."""
    signal = [1.0, 5.0, 2.0, 9.0]
    potential = np.asarray(flow_positions(star, signal)["potential"])[1:]
    assert np.corrcoef(potential, signal)[0, 1] == pytest.approx(1.0)


def test_equal_measurements_still_coincide(star):
    """Which is correct: two cells the data does not distinguish are not distinguished."""
    flow = flow_positions(star, [3.0, 3.0, 7.0, 1.0])
    positions = [tuple(np.round(p, 9)) for p in flow["positions"]]
    assert positions[1] == positions[2]


#### the decomposition is the caption


def test_a_star_reports_pure_gradient(star):
    """No cycles, so no curl and no harmonic: the potential is the whole story, and the
    picture should say that rather than imply there is more."""
    decomposition = flow_positions(star, [1.0, 5.0, 2.0, 9.0])["decomposition"]
    assert decomposition["gradient"] == pytest.approx(1.0)
    assert decomposition["curl"] == pytest.approx(0.0)


def test_a_complex_with_cycles_reports_the_rest():
    rex = RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                   targets=np.array([1, 2, 0], dtype=np.int32))
    rex._ensure_clean()
    decomposition = flow_positions(rex, [1.0, 1.0, 1.0])["decomposition"]
    assert decomposition["gradient"] < 1.0, "a cycle carries content the potential misses"


#### the view


def test_the_payload_carries_the_flow_only_when_given_a_signal(star):
    assert render_payload(star)["positions"]["flow"] is None
    assert render_payload(star, signal=[1.0, 2.0, 3.0, 4.0])["positions"]["flow"]


def test_the_view_renders(star):
    svg = render_svg(render_payload(star, signal=[1.0, 5.0, 2.0, 9.0]), view="flow")
    assert svg.count("<title>vertex") == star.nV
    assert "potential across, divergence up" in svg


def test_the_view_says_what_the_decomposition_was(star):
    svg = render_svg(render_payload(star, signal=[1.0, 5.0, 2.0, 9.0]), view="flow")
    assert "gradient 100%" in svg


def test_without_a_signal_it_says_so_rather_than_drawing_nothing(star):
    assert "no signal" in render_svg(render_payload(star), view="flow")


def test_a_signal_of_the_wrong_length_is_refused(star):
    with pytest.raises(ValueError, match="needs 4 values"):
        flow_positions(star, [1.0, 2.0])


def test_a_bad_signal_does_not_lose_the_rest_of_the_payload(star):
    payload = render_payload(star, signal=[1.0])
    assert payload["positions"]["flow"]["available"] is False
    assert payload["relations"], "the readings went with it"


#### what it does NOT fix


def test_the_divergence_axis_is_degree_dominated(star):
    """Worth pinning as a limitation rather than discovering it again. A hub's divergence
    is the sum of its leaves', so in any star the second axis is dominated by one vertex
    and the picture is a fan rather than a scatter. That is the true statement about a
    star, not a fault in the layout, but it is not a readable diagram either."""
    flow = flow_positions(star, [1.0, 5.0, 2.0, 9.0])
    divergence = np.abs(np.asarray(flow["divergence"]))
    assert divergence[0] >= divergence[1:].sum() - 1e-9
