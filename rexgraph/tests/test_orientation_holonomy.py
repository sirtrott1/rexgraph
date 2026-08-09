"""Grade-2 orientation, measured so that a per-cell sign flip cannot move it.

A solved face column is determined only up to an overall sign: a face and its reverse are
the same cell, and `solve_face_column` returns the leading-positive representative because
something has to be returned. So a raw sign product at grade 2 describes the
REPRESENTATIVE. Negate one column of a two-triangle complex and it moves from +1 to -1
without a single cell changing.

The invariant is the holonomy, exactly as at grade 1 where frustration is the product of
signs around a cycle rather than a count of negative edges. One grade up the loop runs
through the cells: two faces meeting on a relation agree when they induce OPPOSITE
coefficients on it, which is what coherent orientation means, so the pairwise agreement is
-sign(c_a[e] c_b[e]) and the holonomy is its product around a closed loop. Each face
appears exactly twice in a closed loop, so its sign cancels.

Balanced everywhere is exactly coherent orientability, so this measures what orienting
face-by-face attempts and does not need the attempt to succeed to report the obstruction.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from rexgraph.faces import orientation_holonomy, solve_face_column
from rexgraph.graph import RexGraph
from rexgraph.tower import apd

#: tetrahedron boundary: closed and orientable
TETRA = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
#: the five-triangle Moebius band: non-orientable
MOEBIUS = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1)]


def _build(triangles, flip=()):
    edges = sorted({tuple(sorted((t[i], t[(i + 1) % 3]))) for t in triangles
                    for i in range(3)})
    rex = RexGraph(sources=np.array([e[0] for e in edges], dtype=np.int32),
                   targets=np.array([e[1] for e in edges], dtype=np.int32))
    index = {frozenset(e): i for i, e in enumerate(edges)}
    face_edges, face_signs = [], []
    for f, t in enumerate(triangles):
        ids = [index[frozenset((t[i], t[(i + 1) % 3]))] for i in range(3)]
        column = solve_face_column(rex, np.array(ids, dtype=np.int32))
        sign = -1.0 if f in flip else 1.0
        face_edges.append(np.array(ids, dtype=np.int32))
        face_signs.append(np.array([sign * float(x) for x in column]))
    rex.add_faces(face_edges, face_signs)
    rex._ensure_clean()
    return rex


#### the reading


def test_a_closed_orientable_surface_is_balanced():
    out = orientation_holonomy(_build(TETRA))
    assert out["orientable"] is True
    assert out["frustrated"] == 0
    assert out["n_loops"] == 3, "6 adjacencies over 4 cells leaves 3 independent loops"


def test_a_non_orientable_surface_is_not():
    out = orientation_holonomy(_build(MOEBIUS))
    assert out["orientable"] is False
    assert out["frustrated"] >= 1


#### and neither answer moves under the gauge


@pytest.mark.parametrize("flip", [(), (1,), (0, 2), (0, 1, 2, 3)])
def test_the_tetrahedron_reads_balanced_under_any_flip(flip):
    assert orientation_holonomy(_build(TETRA, flip))["orientable"] is True


@pytest.mark.parametrize("flip", [(), (2,), (0, 3), (1, 2, 4)])
def test_the_moebius_band_reads_frustrated_under_any_flip(flip):
    assert orientation_holonomy(_build(MOEBIUS, flip))["orientable"] is False


def test_every_subset_of_flips_gives_the_same_rate():
    """The property in full: 2^4 gauges, one answer."""
    rates = {orientation_holonomy(_build(TETRA, flip))["rate"]
             for r in range(5) for flip in itertools.combinations(range(4), r)}
    assert rates == {0}


#### which is exactly what the raw sign product fails to do


def test_the_raw_per_cell_sign_product_is_not_invariant():
    """The measure this replaces. Same complex, one column negated, parity moves."""
    plain = apd(_build(TETRA), 2)["cells"]
    flipped = apd(_build(TETRA, (0,)), 2)["cells"]
    assert plain[0]["parity"] != flipped[0]["parity"]
    assert plain[0]["n_negative"] != flipped[0]["n_negative"]


def test_apd_says_so_rather_than_presenting_it_as_a_reading():
    out = apd(_build(TETRA), 2)
    assert out["parity_is_gauge"] is True
    assert "REPRESENTATIVE" in out["parity_note"]


def test_the_global_view_carries_the_invariant_instead():
    """balanced/n_frustrated come from the holonomy, so they do not move."""
    assert apd(_build(TETRA), 2, view="global")["balanced"] is True
    assert apd(_build(TETRA, (1,)), 2, view="global")["balanced"] is True
    assert apd(_build(MOEBIUS), 2, view="global")["balanced"] is False
    assert apd(_build(MOEBIUS, (2,)), 2, view="global")["balanced"] is False


#### edges


def test_grade_one_is_refused_with_a_reason():
    """Orientation is a relation BETWEEN cells, and a B1 column is canonical: there is no
    freedom to gauge away, so the grade-1 holonomy is a different object."""
    out = orientation_holonomy(_build(TETRA), grade=1)
    assert out["orientable"] is None
    assert "canonical" in out["reason"]


def test_two_faces_sharing_a_relation_close_no_loop():
    """The case that started this: a face-adjacency graph with no cycle has no holonomy to
    read, so both representatives must agree, and they do."""
    rex = RexGraph(sources=np.array([0, 1, 2, 1, 3], dtype=np.int32),
                   targets=np.array([1, 2, 0, 3, 2], dtype=np.int32))
    rex.add_faces([np.array([0, 1, 2], dtype=np.int32),
                   np.array([1, 3, 4], dtype=np.int32)], signs=None)
    rex._ensure_clean()
    out = orientation_holonomy(rex)
    assert out["n_loops"] == 0
    assert out["orientable"] is True
