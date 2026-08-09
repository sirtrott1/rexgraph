"""The pipeline draws what it analysed, and a tool can return the drawing.

An analysis that reports numbers about a complex and cannot show it makes the reader
reconstruct the picture from the numbers. `drawing` is a stage in every depth, first after
construction, because a picture of what was built is the cheapest thing the pipeline can
say and the one wanted before any number.

No threshold decides whether to draw. `draw_limit` bounds the document and the result
REPORTS what was drawn against what exists, so a truncated picture says so instead of a
rule quietly deciding this complex is too big to look at.
"""
from __future__ import annotations

import numpy as np
import pytest

from agent.pipeline import AnalysisPipeline
from rexgraph.faces import auto_hyperface, autoface
from rexgraph.graph import RexGraph


@pytest.fixture
def rex():
    r = RexGraph(sources=np.array([0, 1, 2, 0, 3], dtype=np.int32),
                 targets=np.array([1, 2, 0, 3, 4], dtype=np.int32))
    autoface(r)
    r._ensure_clean()
    return r


#### the stage


def test_drawing_is_a_stage_at_every_depth():
    assert "drawing" in AnalysisPipeline.STAGES_QUICK
    assert "drawing" in AnalysisPipeline.STAGES_STANDARD
    assert "drawing" in AnalysisPipeline.STAGES_FULL


def test_it_runs_and_produces_a_document(rex):
    drawing = AnalysisPipeline(rex).run(depth="quick")["drawing"]
    assert drawing["drawn"] is True
    assert drawing["svg"].startswith("<svg")


def test_it_draws_the_complex_that_was_analysed(rex):
    result = AnalysisPipeline(rex).run(depth="quick")
    assert result["drawing"]["cells_total"] == result["construction"]["nE"]


def test_it_reaches_a_progressive_callback(rex):
    seen = []
    pipe = AnalysisPipeline(rex)
    pipe.on_stage(lambda name, data: seen.append(name))
    pipe.run(depth="quick")
    assert "drawing" in seen


#### truncation is reported, not decided


def test_an_unbounded_picture_says_it_is_complete(rex):
    drawing = AnalysisPipeline(rex).run(depth="quick")["drawing"]
    assert drawing["truncated"] is False
    assert drawing["cells_drawn"] == drawing["cells_total"]


def test_a_bounded_picture_says_what_it_left_out(rex):
    drawing = AnalysisPipeline(rex, draw_limit=2).run(depth="quick")["drawing"]
    assert drawing["truncated"] is True
    assert drawing["cells_drawn"] == 2
    assert drawing["cells_total"] == 5


def test_a_face_whose_relations_were_not_drawn_is_not_counted(rex):
    """It is skipped in the document, so counting the payload's faces would over-report
    exactly where the report matters."""
    assert AnalysisPipeline(rex, draw_limit=2).run(depth="quick")["drawing"]["faces_drawn"] == 0
    assert AnalysisPipeline(rex).run(depth="quick")["drawing"]["faces_drawn"] == 1


def test_it_can_be_switched_off(rex):
    drawing = AnalysisPipeline(rex, draw=False).run(depth="quick")["drawing"]
    assert drawing["drawn"] is False
    assert "off" in drawing["reason"]


def test_a_failure_is_reported_not_raised(rex):
    """A pipeline that cannot draw has still analysed the complex."""
    pipe = AnalysisPipeline(rex)
    pipe.rex = object()
    result = pipe._stage_drawing()
    assert result["drawn"] is False and "reason" in result


#### the tool returns the drawing itself


def _tool(rex, **kw):
    import agent.mcp_tools as tools

    original = tools._source_rex
    tools._source_rex = lambda *a, **k: (rex, {})
    try:
        return tools._render(**kw)
    finally:
        tools._source_rex = original


def test_the_tool_still_returns_the_readings_by_default(rex):
    assert "relations" in _tool(rex)


def test_it_returns_a_drawing_when_asked(rex):
    drawing = _tool(rex, fmt="svg")
    assert drawing["svg"].startswith("<svg")
    assert drawing["cells_total"] == rex.nE


def test_both_gives_the_readings_with_the_drawing_beside_them(rex):
    both = _tool(rex, fmt="both")
    assert "relations" in both and "drawing" in both


def test_the_views_are_reachable(rex):
    assert _tool(rex, fmt="svg", view="character")["view"] == "character"


def test_an_unknown_format_is_refused(rex):
    with pytest.raises(ValueError, match="fmt must be"):
        _tool(rex, fmt="png")


def test_a_selection_reaches_the_drawing():
    """So an agent can draw one part forward without the picture becoming a different
    complex."""
    rex = RexGraph.from_hypergraph(np.array([0, 4, 6, 8], dtype=np.int32),
                                   np.array([0, 1, 2, 3, 0, 1, 1, 2], dtype=np.int32))
    auto_hyperface(rex)
    rex._ensure_clean()
    rex.attach_metadata(1, 0, "kind", "wide")
    both = _tool(rex, fmt="both", select={"kind": "wide"}, select_dim=1)
    assert both["selection"]["n_selected"] == 1
    assert both["drawing"]["cells_total"] == rex.nE
