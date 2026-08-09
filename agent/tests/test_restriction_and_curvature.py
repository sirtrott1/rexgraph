"""Restriction as a first-class query, and curvature as a located reading.

A row filter returns rows. A restriction returns a COMPLEX: the selection is closed, so
every relation kept has its whole boundary kept with it, and the boundary operators of
the result do not reference anything outside it. An operation handed one cannot reach
past it, which is a property of the object rather than a check the caller remembers.
That closure is what these tests hold it to.

The other half is what a filter cannot answer at all: relative homology, the shape of
what the restriction EXCLUDED. A WHERE clause has no version of that question.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    yield tmp_path


@pytest.fixture
def rex():
    """Two triangles joined at a vertex, so a partial selection is meaningful."""
    src = np.array([0, 1, 2, 3, 4, 5, 2], dtype=np.int32)
    dst = np.array([1, 2, 0, 4, 5, 3, 3], dtype=np.int32)
    return RexGraph(sources=src, targets=dst)


def _closed(rex, v_mask, e_mask) -> bool:
    """Every relation kept has its entire boundary kept: the closure property."""
    rex._ensure_clean()
    bp, bi = np.asarray(rex._boundary_ptr), np.asarray(rex._boundary_idx)
    v = np.asarray(v_mask).astype(bool)
    for e in np.flatnonzero(np.asarray(e_mask).astype(bool)):
        if not v[bi[bp[e]:bp[e + 1]]].all():
            return False
    return True


#### the closure property


def test_a_restriction_is_closed(rex):
    """The whole claim. A relation whose endpoint was excluded would leave the result
    referencing a cell that is not there, which is what a filtered row set does."""
    e_mask = np.zeros(rex.nE, dtype=np.uint8)
    e_mask[:3] = 1                                # the first triangle only
    v_mask, e_closed, _f = rex.subcomplex(e_mask=e_mask)
    assert _closed(rex, v_mask, e_closed)


def test_a_partial_restriction_is_strictly_smaller(rex):
    e_mask = np.zeros(rex.nE, dtype=np.uint8)
    e_mask[:3] = 1
    v_mask, e_closed, _f = rex.subcomplex(e_mask=e_mask)
    assert int(np.asarray(e_closed).sum()) < rex.nE
    assert int(np.asarray(v_mask).sum()) < rex.nV


def test_the_restriction_carries_its_own_homology(rex):
    """One triangle taken alone is a cycle: beta_1 = 1 on the restriction, whatever the
    whole complex reads."""
    e_mask = np.zeros(rex.nE, dtype=np.uint8)
    e_mask[:3] = 1
    v_mask, e_closed, f_mask = rex.subcomplex(e_mask=e_mask)
    quot = rex.quotient(v_mask, e_closed, f_mask)
    assert quot.get("betti_rel") is not None


#### the tool


def _obo(tmp_path, n=12):
    p = tmp_path / "terms.obo"
    p.write_text("".join(
        f"[Term]\nid: GO:{i:07d}\nname: t{i}\n"
        + (f"is_a: GO:{max(1, i - 1):07d}\n" if i > 1 else "") + "\n"
        for i in range(1, n)))
    return str(p)


@pytest.mark.parametrize("quantity,grade", [("kappa", "vertex"), ("chi", "edge")])
def test_the_tool_selects_on_the_grade_the_quantity_lives_on(
        tmp_path, quantity, grade):
    """kappa and phi are per vertex, chi per relation. A vertex mask used as an edge
    mask is not a type error anywhere: it silently restricts to the wrong cells."""
    from agent.mcp_tools import call
    out = call("rexgraph_restrict", quantity=quantity, op=">", threshold=0.0,
               files=[_obo(tmp_path)], limit=2)
    assert out["selected_on"] == grade


def test_the_tool_never_reports_more_cells_than_the_complex_has(tmp_path):
    """The bug this caught: a vertex mask fed in as an edge mask closed to more
    relations than existed."""
    from agent.mcp_tools import call
    out = call("rexgraph_restrict", quantity="kappa", op=">", threshold=0.0,
               files=[_obo(tmp_path)], limit=2)
    assert out["closed_to"]["nE"] <= out["whole"]["nE"]
    assert out["closed_to"]["nV"] <= out["whole"]["nV"]


def test_a_selection_matching_neither_grade_is_refused(rex):
    from agent.mcp_tools import _edges_of
    with pytest.raises(ValueError, match="matches neither"):
        _edges_of(rex, np.ones(rex.nV + rex.nE + 5, dtype=bool))


def test_a_branching_relation_needs_its_whole_boundary(tmp_path):
    """Arity-general: a k-ary relation joins the restriction only when every vertex it
    touches was selected, read off the boundary rather than a pair of endpoints."""
    from agent.mcp_tools import _edges_of
    g = RexGraph.from_hypergraph(np.array([0, 4, 6], dtype=np.int32),
                                 np.array([0, 1, 2, 3, 0, 1], dtype=np.int32))
    partial = np.zeros(g.nV, dtype=bool)
    partial[[0, 1]] = True                       # covers the 2-ary, not the 4-ary
    e_mask, grade = _edges_of(g, partial)
    assert grade == "vertex"
    assert int(e_mask.sum()) == 1, "the branching relation was included on a partial boundary"


#### curvature


def test_curvature_reports_zero_on_a_complex_that_bounds(tmp_path):
    from agent.mcp_tools import call
    out = call("rexgraph_curvature", files=[_obo(tmp_path)])
    assert out["total_curvature"] == 0.0
    assert out["bianchi_ok"] is True


def test_curvature_is_reported_per_face_not_per_complex():
    """A number for the whole object says something is wrong; a per-face field says
    where, which is the difference between a flag and a diagnosis."""
    g = RexGraph(sources=np.array([0, 1, 2, 3], dtype=np.int32),
                 targets=np.array([1, 2, 0, 0], dtype=np.int32))
    g.add_faces([[0, 1, 2]], [[1.0, 1.0, 1.0]])
    kappa = np.asarray(g.attributed_curvature()["kappa_f"])
    assert kappa.shape[0] == g.nF_hodge, "curvature is not one reading per face"
