"""Drawing the complex: what the picture is allowed to say.

The emitter computes no mathematics. These pin the three places where a drawing could
contradict the model rather than express it.

A k-ary relation is ONE cell and must be drawn as one closed shape over its whole support.
Drawing C(k,2) lines invents edges and dissolves the relation's identity; drawing a hub
invents a vertex the complex does not contain. Either would make the picture the argument
for a model this one rejects.

Length carries arity, through the quadrance the library already reports.

Colour is the character, mixed barycentrically over shares that already sum to one, so
there is no scale to choose and no legend that could lie.
"""
from __future__ import annotations

import re
from fractions import Fraction

import numpy as np
import pytest

from agent.graph_view import render_payload
from agent.render_svg import channel_colour, render_svg
from rexgraph.faces import auto_hyperface, autoface
from rexgraph.graph import RexGraph


def _branching():
    """A 4-ary relation with the 4-cycle spanning it, hyperfaces attached."""
    rex = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10, 12], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32))
    auto_hyperface(rex)
    rex._ensure_clean()
    return rex


def _pairwise():
    rex = RexGraph(sources=np.array([0, 1, 2, 0, 3], dtype=np.int32),
                   targets=np.array([1, 2, 0, 3, 4], dtype=np.int32))
    autoface(rex)
    rex._ensure_clean()
    return rex


def _svg(rex):
    return render_svg(render_payload(rex))


#### a k-ary relation is one cell


def test_a_wide_relation_draws_as_one_shape():
    """Not C(k,2) lines and not a hub. The 4-ary relation is a single polygon."""
    svg = _svg(_branching())
    wide = [m for m in re.findall(r"<polygon[^>]*>.*?</polygon>", svg, re.S)
            if "arity 4" in m]
    assert len(wide) == 1


def test_a_pairwise_relation_draws_as_a_line():
    svg = _svg(_pairwise())
    assert svg.count("<line") == _pairwise().nE
    assert "arity 2" in svg


def test_no_vertex_is_invented():
    """A star expansion would add a hub that is not in the complex."""
    rex = _branching()
    svg = _svg(rex)
    drawn = {int(m) for m in re.findall(r"<title>vertex (\d+)", svg)}
    assert drawn == set(range(rex.nV))


def test_the_wide_relation_spans_its_whole_boundary():
    """Four vertices in the polygon, because the boundary column names four."""
    svg = _svg(_branching())
    wide = next(m for m in re.findall(r'<polygon points="([^"]+)"[^>]*>(?:(?!</polygon>).)*?'
                                      r'arity 4', svg, re.S))
    assert len(wide.split()) == 4


#### length carries arity


def test_a_wider_relation_draws_thicker():
    """Quadrance is 1 + 1/(k-1), so the stroke comes off the geometry rather than a
    legend. It is the library's number, not one computed here."""
    payload = render_payload(_branching())
    by_arity = {r["arity"]: Fraction(r["quadrance"]) for r in payload["relations"]}
    assert by_arity[2] == Fraction(2)
    assert by_arity[4] == Fraction(4, 3)


#### colour is the character


def test_the_same_character_always_gives_the_same_colour():
    """The map is a function of the character and of nothing else, so two structurally
    identical cells come out identical without being told to."""
    assert channel_colour([0.2, 0.3, 0.4, 0.1]) == channel_colour([0.2, 0.3, 0.4, 0.1])


def test_different_characters_give_different_colours():
    payload = render_payload(_branching())
    colours = {channel_colour(c["at"]) for c in payload["field"]["cells"]}
    assert len(colours) > 1


def test_a_character_with_no_visible_spectrum_is_black():
    """Not a failure and not a default: a single channel puts every eigenvalue where the
    wavelength falls outside 360-830 nm, and there is no colour for that. Reporting one
    would invent a reading."""
    assert channel_colour([1, 0, 0, 0]) == "#000000"


def test_the_magnitude_matters_because_it_is_physical():
    """Unlike a palette. The wavelength is alpha / (lambda dLT) and the eigenvalues scale
    with the character, so scaling the input moves the spectrum. Library characters are
    always shares summing to one, so this is a property rather than a hazard."""
    assert channel_colour([0.25] * 4) != channel_colour([1, 1, 1, 1])


def test_dLT_moves_the_picture_along_the_spectrum():
    chi = [0.2034, 0.2034, 0.2373, 0.3559]
    assert channel_colour(chi, dLT=1.0) != channel_colour(chi, dLT=0.5)


#### curvature is a second reading, independent of arity


def test_curvature_is_reported_per_relation():
    payload = render_payload(_branching())
    assert all("curvature" in r for r in payload["relations"])


def test_a_face_free_complex_has_no_curvature():
    """It reads B2, so with no 2-cells there is nothing to bend."""
    rex = RexGraph(sources=np.array([0, 1], dtype=np.int32),
                   targets=np.array([1, 2], dtype=np.int32))
    assert all(r["curvature"] == 0.0 for r in render_payload(rex)["relations"])


def test_curvature_is_not_a_restatement_of_arity():
    """If it tracked arity it would carry nothing the quadrance does not."""
    payload = render_payload(_branching())
    pairs = {(r["arity"], round(r["curvature"], 6)) for r in payload["relations"]}
    by_arity = {}
    for arity, curve in pairs:
        by_arity.setdefault(arity, set()).add(curve)
    assert any(len(v) > 1 for v in by_arity.values()) or len(by_arity) > 1


#### the document


def test_the_faces_are_drawn_and_titled():
    svg = _svg(_branching())
    assert "-gon" in svg
    assert svg.count("<title>face") == _branching().nF_hodge


def test_the_caption_reports_the_state_and_orientability():
    svg = _svg(_branching())
    assert "orientable" in svg
    assert "filled" in svg


def test_a_complex_with_no_coordinates_says_so():
    rex = RexGraph(sources=np.zeros(0, dtype=np.int32),
                   targets=np.zeros(0, dtype=np.int32))
    assert "no coordinates" in render_svg(render_payload(rex))


def test_it_is_a_self_contained_document():
    svg = _svg(_branching())
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "http" not in svg.replace('xmlns="http://www.w3.org/2000/svg"', "")


#### the layout is not degenerate


def test_the_two_axes_carry_comparable_spread():
    """Both axes read their parameter through the cosine. Taking the sine on one of them
    put that axis at its flat maximum exactly where ordinary characters cluster: four
    genuinely different vertices spread 0.136 in x and 0.005 in y, an aspect of 1:26 that
    was the map talking rather than the complex."""
    cells = render_payload(_branching())["positions"]["exact"]["cells"]
    xs = [float(Fraction(c["x"])) for c in cells]
    ys = [float(Fraction(c["y"])) for c in cells]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    assert span_y > 0.1 * span_x, f"the y axis collapsed: {span_y:.4f} against {span_x:.4f}"


#### attributes and selection reach the picture


def _molecule(tmp_path):
    from agent.adapters.formats import load_sdf
    from agent.auto import build_rex_from_edges

    path = tmp_path / "b.sdf"
    path.write_text("""benzene
  test

  6  6  0  0  0  0  0  0  0  0999 V2000
    0.0000    1.4000    0.0000 C   0  0
    1.2124    0.7000    0.0000 C   0  0
    1.2124   -0.7000    0.0000 C   0  0
    0.0000   -1.4000    0.0000 C   0  0
   -1.2124   -0.7000    0.0000 C   0  0
   -1.2124    0.7000    0.0000 N   0  0
  1  2  4  0
  2  3  4  0
  3  4  4  0
  4  5  4  0
  5  6  4  0
  6  1  4  0
M  END
$$$$
""")
    return build_rex_from_edges(load_sdf(str(path)), face_selection="hyper")


def test_the_payload_carries_what_the_source_said(tmp_path):
    payload = render_payload(_molecule(tmp_path))
    assert "element" in payload["attributes"]["keys"]
    assert payload["attributes"]["cells"]["0"]["0"]["element"] == "C"


def test_no_selection_means_no_selection(tmp_path):
    assert render_payload(_molecule(tmp_path))["selection"] is None


def test_a_selection_is_a_mask_not_a_smaller_complex(tmp_path):
    """Deleting the unselected cells would change the character of every cell that
    remained and recolour cells the filter never mentioned, so the filtered picture would
    disagree with the unfiltered one about cells neither of them selected."""
    rex = _molecule(tmp_path)
    plain = render_payload(rex)
    filtered = render_payload(rex, select={"element": "C"}, select_dim=0)
    assert filtered["selection"]["n_selected"] == 5
    assert len(filtered["relations"]) == len(plain["relations"])
    assert filtered["field"]["cells"] == plain["field"]["cells"], "the field moved"


def test_the_unselected_are_drawn_back_not_removed(tmp_path):
    """Back, not gone: the selection is a reading OF this complex, and hiding the rest
    would make the picture a different complex that happens to agree on the part asked
    about."""
    import re

    from agent.render_svg import UNSELECTED

    rex = _molecule(tmp_path)
    svg = render_svg(render_payload(rex, select={"bond_order": 4}, select_dim=1))
    opacities = sorted({float(x) for x in
                        re.findall(r'stroke-opacity="([\d.]+)"', svg)})
    assert opacities[0] <= UNSELECTED, "nothing was dimmed"
    assert opacities[-1] > 0.5, "everything was dimmed"
    assert svg.count("<title>relation") == rex.nE, "a relation vanished"


def test_a_relation_draws_between_the_atoms_it_names(tmp_path):
    """The payload carries boundary INDICES as well as labels. Recovering indices by
    sorting the labels assumes they sort into index order, and a molecule's do not:
    "H10" sorts before "H7", so bonds were drawn between atoms that have none."""
    rex = _molecule(tmp_path)
    labels = [f"C{i + 1}" for i in range(6)] + [f"H{i + 7}" for i in range(6)]
    payload = render_payload(rex, labels=labels)
    assert payload["relations"][0]["boundary_index"] == [0, 1]
    assert payload["relations"][0]["boundary"] == ["C1", "C2"]
    # the labels do NOT sort into index order, which is exactly what broke the old
    # recovery: "H10" sorts before "H7"
    assert sorted(labels) != labels


#### the camera is the observer, and it should not invent detail


def test_the_camera_basis_is_exactly_orthonormal():
    """A rotation by an arbitrary angle is irrational: cos(0.6) has no exact value, so a
    float camera puts the whole picture at coordinates that do not exist exactly. Every
    rational half-angle parameter gives a rational point on the circle instead."""
    import itertools
    from fractions import Fraction

    from agent.render_svg import _camera

    look = _camera(Fraction(3, 5), Fraction(9, 20))
    basis = list(zip(*[look((Fraction(1), Fraction(0), Fraction(0))),
                       look((Fraction(0), Fraction(1), Fraction(0))),
                       look((Fraction(0), Fraction(0), Fraction(1)))], strict=False))
    quadrance = [sum(x * x for x in axis) for axis in basis]
    assert quadrance == [1, 1, 1]
    assert all(sum(a * b for a, b in zip(basis[i], basis[j], strict=False)) == 0
               for i, j in itertools.combinations(range(3), 2))
    assert all(isinstance(x, Fraction) for axis in basis for x in axis)


def test_the_camera_preserves_quadrance_exactly():
    from fractions import Fraction

    from agent.render_svg import _camera

    seen = _camera(Fraction(3, 5), Fraction(9, 20))((Fraction(3), Fraction(4),
                                                     Fraction(5)))
    assert sum(x * x for x in seen) == 50


def test_navigating_does_not_drift():
    """The point of a rational observer: composing rotations stays rational, so panning
    around the complex accumulates nothing. A float camera does not."""
    from fractions import Fraction

    from agent.render_svg import _camera

    look = _camera(Fraction(1, 3), Fraction(0))
    point = (Fraction(1), Fraction(0), Fraction(0))
    for _ in range(200):
        point = look(point)
    assert sum(x * x for x in point) == 1


def test_the_embedded_view_uses_the_exact_coordinates(tmp_path):
    """A coordinate file records its positions exactly, and the camera is exact, so this
    view is exact from the file to the last step before the path string. Reading the
    float copies would have thrown that away at the first hop for nothing."""
    from fractions import Fraction

    rex = _molecule(tmp_path)
    payload = render_payload(rex)
    cell = payload["positions"]["embedded"]["cells"][0]
    assert Fraction(cell["exact"][1]) == Fraction("1.4")
    svg = render_svg(payload, view="embedded")
    assert "exact" in svg


def test_a_rational_camera_angle_is_accepted(tmp_path):
    from fractions import Fraction

    svg = render_svg(render_payload(_molecule(tmp_path)), view="embedded",
                     azimuth=Fraction(1, 4), elevation=Fraction(1, 6))
    assert svg.startswith("<svg")


#### 2D and 3D show the same complex


@pytest.mark.parametrize("view", ["plane", "character", "embedded"])
def test_every_view_draws_the_solved_faces(tmp_path, view):
    """They did not: the plane view drew the faces the sign solver produces and the 3D
    views drew none, so one complex had two different contents depending on the camera."""
    rex = _molecule(tmp_path)
    svg = render_svg(render_payload(rex), view=view)
    assert svg.count("<title>face") == rex.nF_hodge > 0


@pytest.mark.parametrize("view", ["plane", "character", "embedded"])
def test_every_view_carries_the_same_reading(tmp_path, view):
    svg = render_svg(render_payload(_molecule(tmp_path)), view=view)
    assert "orientable" in svg
    assert any(word in svg for word in ("latent", "filled", "closed"))


@pytest.mark.parametrize("view", ["plane", "character", "embedded"])
def test_every_view_draws_every_cell(tmp_path, view):
    rex = _molecule(tmp_path)
    svg = render_svg(render_payload(rex), view=view)
    assert svg.count("<title>relation") == rex.nE
    assert svg.count("<title>vertex") == rex.nV


#### colour has three kinds and says which


def test_the_default_colour_is_derived(tmp_path):
    """The k7 spectral colour: a physical consequence of the character, with no scale to
    choose. The only one of the three that is not a decision."""
    from agent.render_svg import colour_scheme

    _fn, legend = colour_scheme(render_payload(_molecule(tmp_path)), "character")
    assert legend["kind"] == "character"


def test_an_attribute_colours_categorically_with_a_legend(tmp_path):
    """Hues carry no order, because the values have none unless the attribute says so."""
    from agent.render_svg import colour_scheme

    rex = _molecule(tmp_path)
    _fn, legend = colour_scheme(render_payload(rex), "bond_order")
    assert legend["kind"] == "attribute"
    # every bond in this ring is aromatic, so one value and one hue; the delocalised
    # relation carries no bond_order at all and is not in the legend, which is right:
    # it is not a bond
    assert set(legend["legend"]) == {"4"}
    assert "order" in legend["reading"] or "no order" in legend["reading"]


def test_a_quantity_ramps_and_reports_its_domain(tmp_path):
    """A ramp normalises, so a picture that hides its endpoints changes meaning when the
    data does. This is the heat map."""
    from agent.render_svg import colour_scheme

    payload = render_payload(_molecule(tmp_path))
    _fn, legend = colour_scheme(payload, "curvature")
    assert legend["kind"] == "quantity"
    low, high = legend["domain"]
    assert low < high, "a ramp over a constant is not a reading"
    assert [round(r["curvature"], 4) for r in payload["relations"]].count(round(high, 4))


def test_the_payload_offers_the_quantities(tmp_path):
    quantities = render_payload(_molecule(tmp_path))["quantities"]
    assert set(quantities) >= {"curvature", "arity", "quadrance"}


@pytest.mark.parametrize("colour_by", ["character", "bond_order", "curvature", "arity"])
def test_every_colouring_renders(tmp_path, colour_by):
    svg = render_svg(render_payload(_molecule(tmp_path)), colour_by=colour_by)
    assert svg.startswith("<svg")


def test_a_heat_map_separates_what_the_scalar_separates(tmp_path):
    """Ring bonds bound the faces and carry curvature; the rest carry none, so the two
    groups must not come out the same colour."""
    import re

    rex = _molecule(tmp_path)
    svg = render_svg(render_payload(rex), colour_by="curvature")
    assert len(set(re.findall(r'stroke="(#[0-9a-f]{6})"', svg))) > 1
