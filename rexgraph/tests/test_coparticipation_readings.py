"""Co-participation has two readings, and they were never in conflict.

Two relations co-participate where their boundaries meet, and "how much" has two honest
answers::

    count   |supp(i) INTERSECT supp(j)|      how MANY vertices they meet at
    share   sum_v |c_i(v)| |c_j(v)|          how MUCH of each one meets

They coincide exactly at arity 2, where every share is 1, which is why RexGraph and spore
agreed to the last digit on every pairwise complex and only ever diverged on branching
ones.

The primaries differ because the systems do. RexGraph defaults to the SHARE because it
propagates: a relation spread over k vertices must carry proportionally less at each, or
signal through a branching vertex multiplies instead of dividing. spore defaults to the
COUNT because a language reasons about the structure as declared, where the question is
how crowded a neighbourhood is.

Numbers below are spore's, so agreement is a real cross-implementation check.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.sparse_character import build_sparse_channels

#: h over {a,b,c,d} with legs a-b and a-c, the brief's fixture
_H = (np.array([0, 4, 6, 8], dtype=np.int32),
      np.array([0, 1, 2, 3, 0, 1, 0, 2], dtype=np.int32))


def _rex(c_channel="share", offsets=None, vertices=None):
    rex = RexGraph(boundary_ptr=(offsets if offsets is not None else _H[0]).copy(),
                   boundary_idx=(vertices if vertices is not None else _H[1]).copy(),
                   c_channel=c_channel)
    rex._ensure_clean()
    return rex


#### the two readings, and that they are independent


def test_the_share_is_how_much_of_each_relation_meets():
    """h and a leg share the distinguished vertex at 1 and one other at 1/(k-1)."""
    overlap = np.asarray(_rex().overlap_share_sparse.todense())
    assert overlap[0, 1] == pytest.approx(4 / 3)      # 1 + 1/3
    assert overlap[1, 2] == pytest.approx(1.0)        # two legs at the same vertex


def test_the_count_is_how_many_vertices_they_meet_at():
    overlap = np.asarray(_rex().overlap_count_sparse.todense())
    assert overlap[0, 1] == 2
    assert overlap[1, 2] == 1


@pytest.mark.parametrize("a,b,count,share", [
    ((0, 1), (0, 2), 1, Fraction(1)),                    # two pairwise at one vertex
    ((0, 1, 2), (1, 2), 2, Fraction(1)),                 # leg meets 3-ary at two vertices
    ((0, 1, 2), (0, 1), 2, Fraction(3, 2)),              # leg at the distinguished vertex
    ((0, 1, 2, 3), (0, 1), 2, Fraction(4, 3)),           # the same at a 4-ary
])
def test_the_witnesses_from_the_brief(a, b, count, share):
    offsets = np.array([0, len(a), len(a) + len(b)], dtype=np.int32)
    vertices = np.array(list(a) + list(b), dtype=np.int32)
    rex = _rex(offsets=offsets, vertices=vertices)
    assert np.asarray(rex.overlap_count_sparse.todense())[0, 1] == count
    assert np.asarray(rex.overlap_share_sparse.todense())[0, 1] == pytest.approx(
        float(share))


def test_they_are_independent_in_both_directions():
    """Neither is a rescaling of the other. Two relations can agree on one and differ on
    the other, in either direction."""
    def readings(a, b):
        offsets = np.array([0, len(a), len(a) + len(b)], dtype=np.int32)
        rex = _rex(offsets=offsets,
                   vertices=np.array(list(a) + list(b), dtype=np.int32))
        return (np.asarray(rex.overlap_count_sparse.todense())[0, 1],
                np.asarray(rex.overlap_share_sparse.todense())[0, 1])

    same_share = readings((0, 1), (0, 2)), readings((0, 1, 2), (1, 2))
    assert same_share[0][1] == pytest.approx(same_share[1][1])
    assert same_share[0][0] != same_share[1][0]

    same_count = readings((0, 1, 2), (0, 1)), readings((0, 1, 2, 3), (0, 1))
    assert same_count[0][0] == same_count[1][0]
    assert same_count[0][1] != pytest.approx(same_count[1][1])


def test_they_coincide_on_a_pairwise_complex():
    """Every share is 1 at arity 2, which is why the two implementations only ever
    diverged on branching complexes."""
    rex = _rex(offsets=np.array([0, 2, 4, 6], dtype=np.int32),
               vertices=np.array([0, 1, 1, 2, 2, 0], dtype=np.int32))
    assert np.allclose(np.asarray(rex.overlap_share_sparse.todense()),
                       np.asarray(rex.overlap_count_sparse.todense()))


#### the selector, against spore's numbers


@pytest.mark.parametrize("c_channel,trace", [("share", 22 / 3), ("count", 10.0)])
def test_the_character_answers_the_selected_reading(c_channel, trace):
    """spore's numbers on the same fixture, so this is a cross-implementation check."""
    channels = dict(build_sparse_channels(_rex(c_channel)))
    assert float(channels["L_C"].diagonal().sum()) == pytest.approx(trace)


def test_the_default_is_the_share():
    """RexGraph propagates, so it wants the reading that conserves."""
    assert _rex().c_channel == "share"


def test_an_unknown_reading_is_refused():
    with pytest.raises(ValueError, match="c_channel must be"):
        RexGraph(sources=np.array([0], dtype=np.int32),
                 targets=np.array([1], dtype=np.int32), c_channel="tally")


#### one construction, so the paths cannot disagree


@pytest.mark.parametrize("c_channel,trace", [("share", 22 / 3), ("count", 10.0)])
def test_every_path_answers_the_same_reading(c_channel, trace):
    """The trap: spore had three constructions and after changing two, trC answered the
    old channel while RL4 answered the new one from one process. The dense kernel derives
    L_C from K1, which is the share, so it would always have answered the share."""
    rex = _rex(c_channel)
    assert float(rex.L_coparticipation.diagonal().sum()) == pytest.approx(trace)
    assert float(dict(build_sparse_channels(rex))["L_C"].diagonal().sum()
                 ) == pytest.approx(trace)
    dense = rex._dense_rcf_bundle.get("L_C")
    if dense is not None:
        assert float(np.trace(np.asarray(dense))) == pytest.approx(trace, abs=1e-6)


def test_it_is_a_proper_laplacian_in_both_readings():
    for c_channel in ("share", "count"):
        rows = np.asarray(_rex(c_channel).L_coparticipation.sum(axis=1)).ravel()
        assert np.allclose(rows, 0.0)


#### the flow does not follow the character's choice


def test_the_flow_operator_does_not_move():
    """Propagation needs the conserving reading whatever the character is describing
    with. A selector on the overlap PROPERTY would have dragged the flow along."""
    from rexgraph.flow import coparticipation_neighbors

    share_ptr, share_idx = coparticipation_neighbors(_rex("share"))
    count_ptr, count_idx = coparticipation_neighbors(_rex("count"))
    assert share_ptr.tolist() == count_ptr.tolist()
    assert share_idx.tolist() == count_idx.tolist()


def test_the_flow_reads_the_share_explicitly():
    from rexgraph.flow import coparticipation_neighbors

    rex = _rex("count")
    canonical = rex.overlap_share_sparse.tocsr().copy()
    canonical.setdiag(0)
    canonical.eliminate_zeros()
    ptr, idx = coparticipation_neighbors(rex)
    assert ptr.tolist() == canonical.indptr.tolist()
    assert idx.tolist() == canonical.indices.tolist()


#### the name


def test_the_old_name_still_resolves():
    """It returned shares while being called counts, which is what misled the reading in
    the first place. Renamed, with the old spelling kept."""
    rex = _rex()
    assert np.allclose(np.asarray(rex.overlap_counts_sparse.todense()),
                       np.asarray(rex.overlap_share_sparse.todense()))
