"""A drawing that has collapsed must separate what it can and say what it cannot."""

import re

import numpy as np
import pytest

from agent.graph_view import render_payload, structural_positions
from agent.render_svg import _label_room, render_svg
from rexgraph.graph import RexGraph


def _star(n=8):
    rex = RexGraph(sources=np.zeros(n, np.int32),
                   targets=np.arange(1, n + 1, dtype=np.int32))
    rex._ensure_clean()
    return rex


def _points(svg):
    return [(float(x), float(y)) for x, y in
            re.findall(r'<circle cx="([-\d.]+)" cy="([-\d.]+)"', svg)]


def _spread(points):
    P = np.asarray(points, dtype=float)
    Q = P - P.mean(axis=0)
    sv = np.linalg.svd(Q, full_matrices=False)[1]
    return float(sv[1] / sv[0]) if sv[0] > 1e-12 else 0.0


#### the collapse is real, and is the character's honest answer


def test_the_character_puts_a_whole_star_on_one_point():
    """Not a bug. Every cell of a star has star character (1/3, 1/3, 1/3)."""
    from rexgraph.projection import project
    from rexgraph.rational_trig import exact_star_character

    rows, _ = exact_star_character(_star())
    assert len({project(row) for row in rows}) == 1


def test_the_structural_layout_separates_what_the_character_cannot():
    placed = structural_positions(_star())["positions"]
    distinct = len({tuple(np.round(row, 6)) for row in placed})
    # the character puts all 9 on ONE point; the layout has to do far better than that.
    # Not exactly 9: the force step is iterative and its arithmetic differs by BLAS and
    # by platform, and a macOS runner separated 7. The claim is the separation, not a
    # count that happens to hold on one machine.
    assert distinct >= 6, f"only {distinct} of 9 separated"
    assert _spread(placed[:, :2]) > 0.5


def test_the_structural_view_says_it_is_not_exact():
    assert structural_positions(_star())["exact"] is False


#### and the drawing separates the cells anyway, and reports that it had to


def test_coincident_cells_are_drawn_apart_rather_than_on_top_of_each_other():
    svg = render_svg(render_payload(_star()), view="plane")
    points = _points(svg)
    assert len(points) == 9
    assert len(set(points)) == 9


def test_the_caption_names_the_collapse_it_drew_around():
    svg = render_svg(render_payload(_star()), view="plane")
    assert "1/9 distinct" in svg
    assert "9 fanned apart" in svg
    assert "COLLINEAR" in svg


def test_a_large_coincident_group_stays_inside_the_frame():
    """The viewport is fitted before the fan, so the fan has to fit itself back in."""
    svg = render_svg(render_payload(_star(60)), view="plane", width=900, height=700)
    points = _points(svg)
    assert len(set(points)) == 61
    assert all(0 <= x <= 900 and 0 <= y <= 700 for x, y in points)


#### labels


def test_labels_are_dropped_only_where_they_would_not_fit():
    roomy, count = _label_room({0: (0.0, 0.0), 1: (500.0, 0.0)})
    assert count == 2
    crowded, count = _label_room({i: (float(i), 0.0) for i in range(20)})
    assert count == 0
    assert crowded == set()


def test_a_crowded_drawing_reports_how_many_it_could_label():
    svg = render_svg(render_payload(_star(60)), view="plane")
    assert "labelled, the rest have no room" in svg


def test_a_roomy_drawing_labels_everything_and_says_nothing_about_it():
    svg = render_svg(render_payload(_star(3)), view="structural")
    assert "have no room" not in svg


#### exposure reaches the drawing


def test_the_drawing_reports_the_exposure_it_solved_for():
    groups = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9], [0, 3, 10]]
    ptr = np.cumsum([0] + [len(g) for g in groups]).astype(np.int32)
    idx = np.array([v for g in groups for v in g], np.int32)
    rex = RexGraph.from_hypergraph(ptr, idx)
    rex._ensure_clean()
    assert "exposure dLT" in render_svg(render_payload(rex), view="structural")


def test_an_explicit_exposure_is_used_and_not_announced():
    svg = render_svg(render_payload(_star()), view="structural", dLT=1.0, eps=1.0)
    assert "exposure dLT" not in svg


#### the view is real everywhere it is offered


@pytest.mark.parametrize("view", ["structural", "plane", "character"])
def test_every_offered_view_draws(view):
    svg = render_svg(render_payload(_star()), view=view)
    assert svg.startswith("<svg")
    assert len(_points(svg)) > 0
