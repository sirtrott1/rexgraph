"""A delocalised ring is one relation, not k bonds.

MDL bond order 4 says the electrons are shared across the ring rather than held in
alternating pairs. Reading that as k separate 2-ary bonds is the same loss clique
expansion makes: it invents bonds the chemistry does not have and dissolves the system's
identity as one object.

Both grades are carried. The sigma framework stays 2-ary and the delocalised system is
added over the same atoms, which is the chemistry (a ring has a bonded framework AND a
shared pi system) and also what makes the ring closable: a wide relation alone bounds
nothing, so a complex built from the systems alone is a forest of stars with no face.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from agent.adapters.formats import load_sdf
from agent.auto import build_rex_from_edges
from rexgraph.graph import RexGraph

_BENZENE = """benzene
  test

 12 12  0  0  0  0  0  0  0  0999 V2000
    0.0000    1.4000    0.0000 C   0  0
    1.2124    0.7000    0.0000 C   0  0
    1.2124   -0.7000    0.0000 C   0  0
    0.0000   -1.4000    0.0000 C   0  0
   -1.2124   -0.7000    0.0000 C   0  0
   -1.2124    0.7000    0.0000 C   0  0
    0.0000    2.4900    0.0000 H   0  0
    2.1564    1.2450    0.0000 H   0  0
    2.1564   -1.2450    0.0000 H   0  0
    0.0000   -2.4900    0.0000 H   0  0
   -2.1564   -1.2450    0.0000 H   0  0
   -2.1564    1.2450    0.0000 H   0  0
  1  2  4  0
  2  3  4  0
  3  4  4  0
  4  5  4  0
  5  6  4  0
  6  1  4  0
  1  7  1  0
  2  8  1  0
  3  9  1  0
  4 10  1  0
  5 11  1  0
  6 12  1  0
M  END
$$$$
"""


@pytest.fixture
def benzene(tmp_path):
    path = tmp_path / "benzene.sdf"
    path.write_text(_BENZENE)
    return str(path)


def _arities(rex):
    return np.diff(np.asarray(rex.boundary_ptr)).tolist()


def test_the_ring_becomes_one_relation(benzene):
    construction = load_sdf(benzene)
    assert len(construction.branching) == 1
    assert len(construction.branching[0]) == 6, "six carbons share the pi system"


def test_the_complex_carries_it_as_one_cell(benzene):
    rex = build_rex_from_edges(load_sdf(benzene), face_selection="hyper")
    rex._ensure_clean()
    assert sorted(_arities(rex)) == [2] * 12 + [6]


def test_both_grades_are_present(benzene):
    """The sigma framework AND the delocalised system, which is the chemistry and also
    what lets the ring close."""
    rex = build_rex_from_edges(load_sdf(benzene), face_selection="hyper")
    rex._ensure_clean()
    assert rex.nE == 13
    assert rex.nF_hodge == 2, "the ring did not close"


def test_the_molecule_is_one_component(benzene):
    rex = build_rex_from_edges(load_sdf(benzene), face_selection="hyper")
    assert rex.betti[1] == 0, "the filled ring should leave no hole"


def test_the_wide_column_sums_to_zero(benzene):
    """(-1, 1/(k-1), ...) at k=6, so the constant vector stays in the kernel."""
    rex = build_rex_from_edges(load_sdf(benzene), face_selection="none")
    rex._ensure_clean()
    B1 = np.asarray(rex.B1)
    wide = next(e for e in range(rex.nE) if (np.abs(B1[:, e]) > 1e-12).sum() == 6)
    assert B1[:, wide].sum() == pytest.approx(0.0)


def test_pairwise_is_still_available(benzene):
    """What every reader did before, for a caller comparing against one."""
    rex = build_rex_from_edges(load_sdf(benzene, aromatic="pairwise"),
                               face_selection="auto")
    rex._ensure_clean()
    assert _arities(rex) == [2] * 12
    assert not load_sdf(benzene, aromatic="pairwise").branching


def test_an_unknown_mode_is_refused(benzene):
    with pytest.raises(ValueError, match="aromatic must be"):
        load_sdf(benzene, aromatic="delocalised")


def test_a_molecule_with_no_aromatic_bonds_is_unchanged(tmp_path):
    """Ethane: single bonds only, so nothing branches and the reading is what it was."""
    path = tmp_path / "ethane.sdf"
    path.write_text("""ethane
  test

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0
    1.5000    0.0000    0.0000 C   0  0
  1  2  1  0
M  END
$$$$
""")
    assert load_sdf(str(path)).branching == []


def test_a_fused_system_is_one_relation_not_two_rings(tmp_path):
    """Naphthalene's ten carbons share ONE delocalised system. Two rings sharing an edge
    is a decomposition the file does not assert."""
    path = tmp_path / "fused.sdf"
    lines = ["fused", "  test", "", " 10 11  0  0  0  0  0  0  0  0999 V2000"]
    lines += ["    0.0000    0.0000    0.0000 C   0  0"] * 10
    ring = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
            (5, 7), (7, 8), (8, 9), (9, 10), (10, 4)]
    lines += [f"{a:3d}{b:3d}  4  0" for a, b in ring]
    lines += ["M  END", "$$$$", ""]
    path.write_text("\n".join(lines))
    branching = load_sdf(str(path)).branching
    assert len(branching) == 1
    assert len(branching[0]) == 10


#### the file's own coordinates are the geometry


def test_the_atom_block_is_carried_exactly(benzene):
    """An SDF writes four decimal places, so every coordinate is a Fraction over 10^4 and
    the geometry taken against it stays on the exact tower. This was discarded."""
    construction = load_sdf(benzene)
    assert len(construction.embedding) == 12
    assert all(isinstance(x, Fraction) for p in construction.embedding for x in p)


def test_the_bond_length_is_the_real_one(benzene):
    """1.4 angstrom carbons, so quadrance 49/25 exactly, against an intrinsic 2 that is a
    function of arity and says nothing about the conformation."""
    from rexgraph.geometry import embedded_geometry_of, geometry_of

    rex = build_rex_from_edges(load_sdf(benzene), face_selection="hyper")
    embedding = load_sdf(benzene).embedding
    embedded = embedded_geometry_of(rex, embedding)
    assert Fraction(embedded["quadrance"][1]) == Fraction(49, 25)
    assert geometry_of(rex, exact=True)["quadrance"][1] == "2"


def test_the_bond_angle_is_an_exact_rational(benzene):
    """Benzene is 120 degrees, so cos^2 = 1/4. It comes back as a rational to the
    precision the file was written at, with no arccosine anywhere."""
    from rexgraph.geometry import embedded_geometry_of

    rex = build_rex_from_edges(load_sdf(benzene), face_selection="hyper")
    out = embedded_geometry_of(rex, load_sdf(benzene).embedding)
    adjacent = next(m for m in out["meeting"] if m["relations"] == [0, 1])
    cos2 = Fraction(adjacent["cos_squared"])
    assert cos2.denominator > 1, "an exact rational, not a rounded float"
    assert float(cos2) == pytest.approx(0.25, abs=1e-4)


def test_the_payload_carries_both_pictures(benzene):
    """Where the cells are, and which cells are alike. Benzene's six carbons are
    structurally identical, so they stack in character space while sitting on a hexagon
    in the file, and neither picture is wrong."""
    from agent.graph_view import render_payload

    payload = render_payload(build_rex_from_edges(load_sdf(benzene),
                                                  face_selection="hyper"))
    assert payload["positions"]["embedded"] is not None
    assert payload["positions"]["exact"] is not None
    assert payload["embedded_geometry"]["exact"] is True


def test_a_source_with_no_coordinates_says_so():
    from agent.graph_view import render_payload
    from agent.render_svg import render_svg

    rex = RexGraph(sources=np.array([0, 1], dtype=np.int32),
                   targets=np.array([1, 2], dtype=np.int32))
    payload = render_payload(rex)
    assert payload["positions"]["embedded"] is None
    assert "carried no coordinates" in render_svg(payload, view="embedded")
