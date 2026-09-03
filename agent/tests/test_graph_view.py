"""Structural coordinates, and reach that is not a hop count.

The old dashboard positioned vertices by the eigenvectors of L0. That is a linear
grouping: the coordinate says where a cut fell, not what the cell is, and producing it
costs a dense eigendecomposition. Nothing here has a spectral-embedding mode, and that
is the point rather than an omission.

The character already IS a position. phi(v) lives in the simplex over the channel hats,
so the coordinates are the cell's shares of topology, geometry, frustration and
co-participation. These tests hold the embedding to being a change of coordinates: a
cell that is purely one channel lands on that channel's corner, and equal shares land
at the centre. Nothing is fitted, so there is nothing to converge.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from agent.graph_view import (
    character_positions,
    neighbors,
    positions,
    propagator_positions,
    reach,
)

from rexgraph.graph import RexGraph


@pytest.fixture
def rex():
    """Two triangles sharing a vertex."""
    return RexGraph(sources=np.array([0, 1, 2, 2, 3, 4], dtype=np.int32),
                    targets=np.array([1, 2, 0, 3, 4, 2], dtype=np.int32))


#### the frame is a change of coordinates, not a fit


def _embed(chi):
    """The library's own simplex embedding, which is what character_positions uses."""
    from rexgraph.core import _fiber

    chi = np.ascontiguousarray(np.atleast_2d(chi), dtype=float)
    return np.asarray(_fiber.signal_sphere_proj(chi, chi.shape[0], chi.shape[1]))


@pytest.mark.parametrize("k", [3, 4])
def test_the_embedding_is_a_regular_simplex(k):
    """Every channel the same distance from every other, so the embedding privileges
    none of them. `core._fiber.signal_sphere_proj` is the library's, not a second frame
    built in the view layer; 3 and 4 are the counts CHANNEL_ORDER can produce, since
    inactive channels lower it and nothing raises it."""
    corners = _embed(np.eye(k))
    d = [np.linalg.norm(corners[i] - corners[j])
         for i in range(k) for j in range(i + 1, k)]
    assert np.allclose(d, d[0]), f"k={k}: channels are not equidistant"


@pytest.mark.parametrize("k", [3, 4])
def test_a_cell_of_one_channel_lands_on_that_corner(k):
    """What makes a coordinate readable without a legend: a cell that is entirely
    frustration sits at the frustration corner."""
    corners = _embed(np.eye(k))
    for i in range(k):
        p = np.zeros(k)
        p[i] = 1.0
        assert np.allclose(_embed(p)[0], corners[i])


@pytest.mark.parametrize("k", [3, 4])
def test_equal_shares_land_at_the_centre(k):
    """The centre of the simplex, which is where a cell carrying no preference belongs.
    It is the corners' centroid rather than the origin, because the library places the
    simplex with a vertex at the origin."""
    corners = _embed(np.eye(k))
    assert np.allclose(_embed(np.ones(k) / k)[0], corners.mean(axis=0))


#### the modes


def test_character_positions_are_exact_at_full_dimension(rex):
    out = character_positions(rex, dim=3)
    assert out["positions"].shape == (rex.nV, 3)
    assert out["exact"] is True
    assert out["channels"], "the channels the coordinates mean are not reported"


def test_a_lower_dimension_says_it_is_a_projection(rex):
    """A 2D picture of a 3-simplex has lost something, and reporting it as exact would
    make a projection look like a reading."""
    out = character_positions(rex, dim=2)
    assert out["exact"] is False
    assert "projected" in out["note"]


def test_edge_character_positions_are_per_relation(rex):
    out = character_positions(rex, grade="edge", dim=3)
    assert out["positions"].shape[0] == rex.nE


def test_positions_are_deterministic(rex):
    """No iteration, no seed, no force-directed refinement to converge."""
    a = character_positions(rex, dim=3)["positions"]
    b = character_positions(rex, dim=3)["positions"]
    assert np.array_equal(a, b)


def test_propagator_positions_are_reach_from_anchors(rex):
    out = propagator_positions(rex, dim=3, t=1.0)
    assert out["positions"].shape == (rex.nV, 3)
    assert len(out["anchors"]) == 3
    # an anchor is the strongest source of its own coordinate
    for j, a in enumerate(out["anchors"]):
        assert out["positions"][a, j] == pytest.approx(
            out["positions"][:, j].max(), rel=1e-6)


def test_there_is_no_spectral_mode(rex):
    from agent.graph_view import MODES
    assert "spectral" not in MODES and "fiedler" not in MODES
    with pytest.raises(ValueError, match="mode must be"):
        positions(rex, mode="spectral")


#### neighbourhood and reach


def test_the_star_is_a_closed_subcomplex(rex):
    """An adjacency list returns cells and leaves their boundary to the caller."""
    out = neighbors(rex, 2)
    rex._ensure_clean()
    bp, bi = np.asarray(rex.boundary_ptr), np.asarray(rex.boundary_idx)
    verts = set(out["vertices"])
    for e in out["edges"]:
        assert set(int(v) for v in bi[bp[e]:bp[e + 1]]) <= verts


def test_a_vertex_outside_the_complex_is_refused(rex):
    with pytest.raises(IndexError):
        neighbors(rex, rex.nV + 5)


def test_reach_ranks_by_how_much_arrives(rex):
    """Not by hop count: a cell reached through many relations outranks one reached
    through a thread, and there is no depth to choose."""
    out = reach(rex, [0], limit=4)
    values = [r["value"] for r in out["reached"]]
    assert values == sorted(values, key=abs, reverse=True)
    assert 0 not in [r["vertex"] for r in out["reached"]], "the seed is not a result"


def test_reach_refuses_a_seed_outside_the_complex(rex):
    with pytest.raises(IndexError):
        reach(rex, [rex.nV + 3])


def test_a_disconnected_component_is_not_reached(rex):
    """Transport, not topology-by-cut: heat does not cross where there is no relation."""
    g = RexGraph(sources=np.array([0, 2], dtype=np.int32),
                 targets=np.array([1, 3], dtype=np.int32))
    reached = {r["vertex"]: r["value"] for r in reach(g, [0], limit=4)["reached"]}
    assert reached.get(2, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert reached.get(3, 0.0) == pytest.approx(0.0, abs=1e-12)


#### the render payload


def test_the_render_payload_agrees_with_itself(rex):
    """Positions, lengths and angles are three readings of the same boundary columns.
    Assembled together so a renderer cannot draw a relation at a length its own
    quadrance contradicts."""
    from agent.graph_view import render_payload

    from rexgraph.geometry import relation_quadrance
    p = render_payload(rex)
    for row in p["relations"]:
        assert row["quadrance"] == str(relation_quadrance(rex, row["index"]))
        assert float(row["quadrance_float"]) == pytest.approx(
            float(Fraction(row["quadrance"])))


def test_every_rendered_quantity_is_exact(rex):
    from agent.graph_view import render_payload
    p = render_payload(rex)
    for row in p["relations"]:
        Fraction(row["quadrance"])               # parses or raises
    for m in p["spreads"]:
        Fraction(m["spread"])
        Fraction(m["cos_squared"])
    for c in p["positions"]["exact"]["cells"]:
        Fraction(c["x"]), Fraction(c["y"])


def test_the_payload_carries_the_face_rule(rex):
    """latent / filled / closed is what decides whether a surface can be drawn without
    a boundary edge."""
    from agent.graph_view import render_payload
    p = render_payload(rex)
    assert p["state"]["state"] in ("latent", "partially filled", "filled",
                                   "closed", "acyclic")
    assert "closed" in p["closure"]


def test_a_branching_relation_renders_as_one_cell():
    """arity is in the row, and the boundary lists every vertex, so a k-ary relation is
    one thing to draw rather than a pair with the rest missing."""
    from agent.graph_view import render_payload
    g = RexGraph.from_hypergraph(np.array([0, 4, 6], dtype=np.int32),
                                 np.array([0, 1, 2, 3, 0, 1], dtype=np.int32))
    rows = render_payload(g, labels=list("abcd"))["relations"]
    wide = [r for r in rows if r["arity"] == 4]
    assert len(wide) == 1
    assert wide[0]["boundary"] == ["a", "b", "c", "d"]
    assert wide[0]["quadrance"] == "4/3"          # length carries arity


def test_length_carries_arity_across_the_payload():
    from agent.graph_view import render_payload
    for k in (2, 3, 4, 5):
        g = RexGraph.from_hypergraph(np.array([0, k], dtype=np.int32),
                                     np.arange(k, dtype=np.int32))
        row = render_payload(g)["relations"][0]
        assert Fraction(row["quadrance"]) == 1 + Fraction(1, k - 1)


#### the payload carries the whole drawable state


def _closed():
    """A 4-ary relation with the 4-cycle spanning it, hyperfaces attached."""
    from rexgraph.faces import auto_hyperface

    rex = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10, 12], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32))
    auto_hyperface(rex)
    rex._ensure_clean()
    return rex


def test_the_solved_two_cells_are_in_the_payload():
    """They were not, so a renderer could not draw the faces the sign solver produces."""
    from agent.graph_view import render_payload

    rex = _closed()
    faces = render_payload(rex)["faces"]
    assert len(faces) == rex.nF_hodge > 0
    assert all(f["relations"] and f["coefficients"] for f in faces)


def test_a_face_reports_its_gon_not_the_relations_offered():
    """A solved column carries an explicit zero for a relation it does not use, and that
    is not a side. Same rule as `faces.face_support` and `surface_identity`."""
    from agent.graph_view import render_payload

    for face in render_payload(_closed())["faces"]:
        assert face["gon"] == len(face["relations"])
        assert 0.0 not in face["coefficients"]


def test_the_orientation_reading_is_the_gauge_invariant_one():
    """Per-face parity is the representative's sign; orientability is not."""
    from agent.graph_view import render_payload

    payload = render_payload(_closed())
    assert payload["orientation"]["orientable"] in (True, False)
    assert any(f["parity"] in (1, -1) for f in payload["faces"])


def test_the_channel_field_is_carried_exactly():
    """chi is already shares over the channels, so colour needs no scale and no legend
    that can lie. The exact rational is beside the float that goes in the fill string."""
    from agent.graph_view import render_payload

    field = render_payload(_closed())["field"]
    assert field["exact"] is True
    assert field["channels"] == ["L1_down", "L_O", "L_SG", "L_C"]
    cell = field["cells"][0]
    assert sum(Fraction(x) for x in cell["exact"]) == 1
    assert float(Fraction(cell["exact"][0])) == pytest.approx(cell["at"][0])


def test_relations_are_placed_as_well_as_vertices():
    """A renderer drawing a k-ary relation as one cell needs its position too."""
    from agent.graph_view import render_payload

    payload = render_payload(_closed())
    assert len(payload["positions"]["relations"]["cells"]) == _closed().nE


def test_an_open_complex_carries_no_faces_rather_than_empty_ones():
    from agent.graph_view import render_payload

    rex = RexGraph(sources=np.array([0, 1], dtype=np.int32),
                   targets=np.array([1, 2], dtype=np.int32))
    payload = render_payload(rex)
    assert payload["faces"] == []
    assert payload["orientation"] is None
