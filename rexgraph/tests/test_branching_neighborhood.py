"""What a vertex past the second in a branching relation used to look like.

Every neighborhood reading went through `_v2e`, which was built from
`_ensure_src_tgt`: two vertices per relation whatever the arity. So in a k-ary relation
the vertices past the second had no incident relations at all, and each of these readings
returned the value for an isolated vertex rather than a wrong-ish one: zero energy, a
never entry time in the filtration, an empty star.

These are the values that changed, held to the correct ones. The pairwise cases are in
`test_apd_hyperslice.py`, which pins that none of this moved for a 2-ary complex.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph

#: a 4-ary relation over {0,1,2,3} plus 2-ary legs 0-1, 1-2, 2-4.
#: Vertex 3 sits ONLY in the 4-ary relation, so it is the one the pairwise path lost.
_OFFSETS = np.array([0, 4, 6, 8, 10], dtype=np.int32)
_VERTICES = np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 4], dtype=np.int32)

LOST = 3


@pytest.fixture
def rex():
    r = RexGraph.from_hypergraph(_OFFSETS.copy(), _VERTICES.copy())
    r._ensure_clean()
    return r


def test_the_vertex_is_incident_to_the_relation_that_contains_it(rex):
    ptr, idx = rex._v2e
    assert sorted(int(e) for e in idx[ptr[LOST]:ptr[LOST + 1]]) == [0]


def test_it_has_energy(rex):
    """`vertex_energy_character` was exactly 0.0 there: no incident relation, no energy.
    Exactly zero rather than small, which is the signature of an absent reading rather
    than a lossy one."""
    energy = np.asarray(rex.vertex_energy_character)
    assert energy[LOST] > 0
    assert energy[LOST] == pytest.approx(0.630921, abs=1e-5)


def test_its_star_character_is_not_the_uniform_default(rex):
    """The four channels over its star. With an empty star it read exactly
    [1/4, 1/4, 1/4, 1/4]: the simplex centre, maximum entropy, no information at all, and
    identical for every such vertex in every complex. It now carries the profile of the
    relation it is actually in, which is C-dominant because the wide relation is where
    co-participation lives."""
    chi = np.asarray(rex.star_character)[LOST]
    assert chi.sum() == pytest.approx(1.0)
    assert not np.allclose(chi, 0.25), "still the no-information default"
    assert chi == pytest.approx([0.239908, 0.239908, 0.164937, 0.355248], abs=1e-5)
    assert int(np.argmax(chi)) == 3, "co-participation should dominate a wide relation"


def test_it_enters_the_filtration(rex):
    """`edge_sublevel` gave it 1e308, the never-enters sentinel, so persistent homology
    ran on a complex missing a vertex. The other filtration kinds read the boundary CSR
    directly and were always right, which is why only this one moved."""
    entry = np.asarray(rex.filtration(
        "edge_sublevel", signal=np.arange(rex.nE, dtype=float))[0])
    assert np.all(np.isfinite(entry))
    assert entry[LOST] < 1e300
    assert entry[LOST] == pytest.approx(0.0), "it should enter with its own relation"


def test_its_star_contains_the_relation(rex):
    v_mask, e_mask, _f_mask = rex.star_of_vertex(LOST)
    assert e_mask[0] == 1, "the 4-ary relation is not in the star"
    assert v_mask[LOST] == 1


def test_its_hyperslice_quotient_is_not_empty(rex):
    _v_mask, e_mask, _f_mask = rex.hyperslice_quotient(0, LOST)
    assert e_mask.sum() > 0


def test_the_relation_reports_all_four_vertices_everywhere_it_is_described(rex):
    """below, cell_shape and the telescope are three descriptions of one boundary, so a
    disagreement between them is the defect showing in only some of them."""
    below, _above, _lateral = rex.hyperslice(1, 0)
    assert sorted(int(v) for v in below) == [0, 1, 2, 3]
    assert sorted(int(v) for v in np.asarray(rex.cell_shape(1, 0)["below"])) == [0, 1, 2, 3]
    telescope = rex.hyperslice_telescope(1, 0, depth=1)
    assert sorted(v for _d, v in telescope["below_1"]) == [0, 1, 2, 3]


def test_a_vertex_sees_every_coparticipant(rex):
    """Vertex 0 is in the 4-ary relation and in the 0-1 leg. Through the pairwise path it
    saw one other endpoint per relation, so it reported a single neighbour."""
    _above, lateral = rex.hyperslice(0, 0)
    assert sorted(int(v) for v in lateral) == [1, 2, 3]


def test_the_energy_of_the_whole_complex_moved_with_it(rex):
    """Not only the lost vertex: vertex 2 is in the 4-ary relation and in two legs, and
    the pairwise path dropped its membership in the wide one too."""
    assert np.asarray(rex.vertex_energy_character)[2] == pytest.approx(1.161876, abs=1e-5)
    assert np.asarray(rex.local_coherence)[2] == pytest.approx(0.907864, abs=1e-5)
