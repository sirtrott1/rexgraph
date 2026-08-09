"""Joining complexes at any arity, and the result still being a complex.

What is joined is the RELATIONS, since they are primitive and the vertices are their
boundary. A relation's identity is its oriented support: two are the same relation when
they distinguish the same vertex and reach the same others. That is read off the
boundary structure, so a branching relation is matched as one relation of arity k.

`core._joins` is the pairwise dense oracle and cannot do this. It finds a relation's
endpoints with `abs(B1[v, e]) > 0.5`, and a branching column carries `1/(k-1)`, which is
exactly 0.5 at k=3 and smaller above, so every share vertex is invisible to it. That
measurement is pinned here too, because it is the reason there is a second
implementation rather than a preference.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.joins import HOW, join, relation_key, vertex_correspondence


def _triangle():
    return RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                    targets=np.array([1, 2, 0], dtype=np.int32))


def _kary(k: int, order=None):
    idx = np.asarray(order if order is not None else range(k), dtype=np.int32)
    return RexGraph.from_hypergraph(np.array([0, k], dtype=np.int32), idx)


#### why this exists at all


@pytest.mark.parametrize("k,seen", [(2, 2), (3, 1), (4, 1), (5, 1)])
def test_the_dense_kernels_endpoint_test_misses_branching_shares(k, seen):
    """The measurement behind the second implementation: at arity 3 and above the
    0.5 cutoff sees only the distinguished vertex."""
    col = np.asarray(_kary(k).B1)[:, 0]
    assert sum(1 for v in range(k) if abs(col[v]) > 0.5) == seen


#### identity of a relation


def test_orientation_is_part_of_a_relations_identity():
    """At arity k the distinguished vertex is a k-way choice, so two relations over the
    same vertices that distinguish different ones are different relations."""
    a = relation_key(0, (1, 2, 3))
    b = relation_key(1, (0, 2, 3))
    assert a != b


def test_the_rest_of_the_support_is_unordered():
    """The share is uniform across it, so ordering carries nothing."""
    assert relation_key(0, (1, 2, 3)) == relation_key(0, (3, 1, 2))


def test_a_relation_that_does_not_translate_has_no_key():
    assert relation_key(0, (1, 9), remap={0: 0, 1: 1}) is None


#### correspondence


def test_vertices_are_identified_by_label():
    assert vertex_correspondence(["a", "b", "c"], ["b", "c", "d"]) == {1: 0, 2: 1}


def test_an_ambiguous_label_is_declined():
    """Identifying two distinct cells because they share a name is how a join silently
    merges things that were never the same."""
    assert vertex_correspondence(["a", "a"], ["a", "b"]) == {}


#### the joins


@pytest.mark.parametrize("how", HOW)
def test_the_result_is_a_valid_complex(how):
    r, s = _triangle(), _triangle()
    joined, _rep = join(r, s, how=how,
                        labels_r=["a", "b", "c"], labels_s=["b", "c", "d"])
    assert joined.self_loop_face_indices == [], "the join broke the chain condition"


def test_inner_keeps_only_what_both_carry():
    r, s = _triangle(), _triangle()
    _j, rep = join(r, s, how="inner",
                   labels_r=["a", "b", "c"], labels_s=["b", "c", "d"])
    assert rep["shared_relations"] == 1
    assert rep["kept_from_r"] == 1


def test_left_keeps_all_of_the_left_complex():
    r, s = _triangle(), _triangle()
    _j, rep = join(r, s, how="left",
                   labels_r=["a", "b", "c"], labels_s=["b", "c", "d"])
    assert rep["kept_from_r"] == r.nE


def test_outer_merges_identified_vertices_and_keeps_the_rest():
    r, s = _triangle(), _triangle()
    joined, rep = join(r, s, how="outer",
                       labels_r=["a", "b", "c"], labels_s=["b", "c", "d"])
    # a, b, c from R plus d from S: the two shared vertices are not duplicated
    assert rep["nV"] == 4
    assert joined.nE == r.nE + (s.nE - rep["shared_relations"])


def test_an_unknown_join_kind_is_refused():
    r = _triangle()
    with pytest.raises(ValueError, match="how must be"):
        join(r, r, how="cross", labels_r=["a", "b", "c"], labels_s=["a", "b", "c"])


#### branching, which is the point


def test_identical_branching_relations_match():
    labels = ["a", "b", "c", "d"]
    _j, rep = join(_kary(4), _kary(4), how="inner",
                   labels_r=labels, labels_s=labels)
    assert rep["shared_relations"] == 1


def test_a_reoriented_branching_relation_does_not_match():
    """Same four vertices, different distinguished one: a different relation."""
    labels = ["a", "b", "c", "d"]
    _j, rep = join(_kary(4), _kary(4, order=[1, 0, 2, 3]), how="inner",
                   labels_r=labels, labels_s=labels)
    assert rep["shared_relations"] == 0


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_a_branching_relation_survives_the_join_whole(k):
    """Matched as ONE relation of arity k, not expanded into pairs."""
    labels = [chr(ord("a") + i) for i in range(k)]
    joined, rep = join(_kary(k), _kary(k), how="inner",
                       labels_r=labels, labels_s=labels)
    assert rep["nE"] == 1
    joined._ensure_clean()
    bp = np.asarray(joined._boundary_ptr)
    assert int(bp[1] - bp[0]) == k, "the relation lost or gained boundary vertices"


#### faces


def test_a_face_is_carried_only_when_its_whole_boundary_survives():
    """Carrying a face over a relation that was dropped would break B1 B2 = 0."""
    from rexgraph.faces import autoface
    r = _triangle()
    autoface(r, 3)
    assert r.nF == 1
    # inner join keeps one relation, so the face cannot come with it
    joined, rep = join(r, _triangle(), how="inner",
                       labels_r=["a", "b", "c"], labels_s=["b", "c", "d"])
    assert rep["faces_carried"] == 0
    assert joined.self_loop_face_indices == []


def test_a_face_survives_a_left_join_that_keeps_its_boundary():
    from rexgraph.faces import autoface
    r = _triangle()
    autoface(r, 3)
    joined, rep = join(r, _triangle(), how="left",
                       labels_r=["a", "b", "c"], labels_s=["b", "c", "d"])
    assert rep["faces_carried"] == 1
    assert tuple(joined.betti) == tuple(r.betti)
