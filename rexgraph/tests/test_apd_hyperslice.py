"""APD and hyperslice are one operator read at two levels.

`hyperslice(1, e)` returns three SETS around a relation: below (its boundary), above (the
faces containing it), lateral (the relations it shares a vertex with). `apd` returns the
MEASURES of the same neighborhood: arity is |below|, degree is |above|, and the third set
is the C channel, `sum(deg(v) - 1)` over the support, which is the line-graph degree.

So hyperslice answers WHICH and apd answers HOW MANY, and the composition is the obvious
one: apd in the global view finds the cells worth looking at, hyperslice says what is
around them, apd in the local view reads the neighborhood. Parity is the one reading with
no hyperslice counterpart, because a set has no sign.

These also pin the arity generality both directions inherit from the boundary CSR. The
pairwise `(sources, targets)` path holds two vertices per relation, so a branching
relation used to report its first two and every vertex past them read as isolated.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.faces import autoface
from rexgraph.graph import RexGraph
from rexgraph.tower import apd


def _pairwise():
    """Triangle with a two-relation tail, filled, so degree varies."""
    rex = RexGraph(sources=np.array([0, 1, 2, 0, 3], dtype=np.int32),
                   targets=np.array([1, 2, 0, 3, 4], dtype=np.int32))
    autoface(rex)
    return rex


def _branching():
    """A 4-ary relation with two 2-ary legs."""
    return RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2], dtype=np.int32))


#### the two are the same neighborhood, counted or named


@pytest.mark.parametrize("build", [_pairwise, _branching], ids=["pairwise", "branching"])
def test_arity_is_the_size_of_the_hyperslice_below(build):
    rex = build()
    cells = apd(rex, 1)["cells"]
    for e in range(rex.nE):
        below, _above, _lateral = rex.hyperslice(1, e)
        assert cells[e]["arity"] == len(below)


@pytest.mark.parametrize("build", [_pairwise, _branching], ids=["pairwise", "branching"])
def test_degree_is_the_size_of_the_hyperslice_above(build):
    rex = build()
    cells = apd(rex, 1)["cells"]
    for e in range(rex.nE):
        _below, above, _lateral = rex.hyperslice(1, e)
        assert cells[e]["degree"] == len(above)


def test_the_lateral_set_is_the_coparticipation_channel():
    """The third hyperslice set has no APD component, and it is not missing: its count is
    the C channel, sum(deg(v) - 1) over the support, which is the line-graph degree.

    They agree as a COUNT only where no two relations share more than one vertex, since C
    carries multiplicity and lateral is a set. Asserted on a complex where they do agree,
    so the identification is exact rather than approximate.
    """
    rex = _pairwise()
    B1 = np.asarray(rex.B1)
    incident = (np.abs(B1) > 1e-12)
    deg = incident.sum(axis=1)
    for e in range(rex.nE):
        support = np.nonzero(incident[:, e])[0]
        c_channel = sum(int(deg[v]) - 1 for v in support)
        assert len(rex.hyperslice(1, e)[2]) == c_channel


def test_parity_has_no_hyperslice_counterpart():
    """A set has no sign, so the orientation reading exists only on the APD side. This is
    the one place the two are not interchangeable."""
    rex = _pairwise()
    reversed_rex = RexGraph(sources=np.array([1, 1, 2, 0, 3], dtype=np.int32),
                            targets=np.array([0, 2, 0, 3, 4], dtype=np.int32))
    autoface(reversed_rex)
    assert rex.hyperslice(2, 0)[0].tolist() == reversed_rex.hyperslice(2, 0)[0].tolist()
    assert (apd(rex, 2)["cells"][0]["n_negative"]
            != apd(reversed_rex, 2)["cells"][0]["n_negative"])


#### both are arity-general, because both read the boundary column


def test_a_branching_relation_reports_its_whole_boundary():
    """(sources, targets) holds two vertices per relation whatever its arity. Reading the
    boundary CSR instead returns all four here."""
    rex = _branching()
    below, _above, _lateral = rex.hyperslice(1, 0)
    assert sorted(int(v) for v in below) == [0, 1, 2, 3]


def test_a_vertex_past_the_second_is_not_isolated():
    """Vertex 3 is only in the 4-ary relation. Through the pairwise path it had no
    incident relations at all."""
    rex = _branching()
    above, lateral = rex.hyperslice(0, 3)
    assert sorted(int(e) for e in above) == [0]
    assert sorted(int(v) for v in lateral) == [0, 1, 2], "co-participants are missing"


def test_a_vertex_sees_every_coparticipant_not_just_one():
    """The pairwise form takes the ONE other endpoint of each incident relation. A k-ary
    relation has k-1 others."""
    rex = _branching()
    _above, lateral = rex.hyperslice(0, 0)
    assert len(lateral) == 3


def test_the_vertex_to_relation_map_is_arity_general():
    rex = _branching()
    ptr, idx = rex._v2e
    B1 = np.asarray(rex.B1)
    for v in range(rex.nV):
        through_map = sorted(int(e) for e in idx[ptr[v]:ptr[v + 1]])
        through_boundary = sorted(int(e) for e in np.nonzero(np.abs(B1[v]) > 1e-12)[0])
        assert through_map == through_boundary


def test_the_pairwise_case_is_untouched():
    """The boundary CSR of a pairwise complex IS (sources, targets), so nothing about the
    simple case moved: same sets, same order."""
    rex = _pairwise()
    src, tgt = rex.sources, rex.targets
    for e in range(rex.nE):
        below, _above, _lateral = rex.hyperslice(1, e)
        assert below.tolist() == [int(src[e]), int(tgt[e])]


#### composing them


def test_global_apd_selects_and_hyperslice_expands():
    """The intended loop: read the means, find the cell that departs from them, ask
    hyperslice what is around it, read that neighborhood locally."""
    rex = _branching()
    mean_arity = float(np.mean([c["arity"] for c in apd(rex, 1)["cells"]]))
    outlier = max(apd(rex, 1)["cells"], key=lambda c: abs(c["arity"] - mean_arity))
    assert outlier["index"] == 0

    below, _above, lateral = rex.hyperslice(1, outlier["index"])
    neighborhood = {int(e) for e in lateral} | {outlier["index"]}
    assert len(below) == outlier["arity"]
    assert neighborhood == {0, 1, 2}, "the wide relation does not reach the whole complex"
