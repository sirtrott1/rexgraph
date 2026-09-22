"""The resolvent rank is PageRank's fixed point as a Green response, exact at every grade."""
import itertools
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.coordinate_map import CoordinateMetric
from rexgraph.native_field import NativeFieldCalculus
from rexgraph.resolvent_ranking import POLICIES, rank_calculus, resolvent_rank


def _graph(pairs, **kw):
    return RexGraph(sources=np.array([a for a, _ in pairs], np.int32),
                    targets=np.array([b for _, b in pairs], np.int32), **kw)


def _tetrahedron(faces):
    rex = _graph(list(itertools.combinations(range(4), 2)))
    rex.add_faces(faces)
    rex._ensure_clean()
    return rex


PAIRWISE = {
    "path of 3": [(0, 1), (1, 2)],
    "path of 5": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "star": [(0, 1), (0, 2), (0, 3), (0, 4)],
    "ring of 5": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
    "K4": list(itertools.combinations(range(4), 2)),
    "two components": [(0, 1), (1, 2), (3, 4)],
}


def test_the_paper_values():
    unit = _graph([(0, 1), (1, 2)])
    assert list(resolvent_rank(unit, 0, [1, 0, 0], Q(1, 2), "walk")) == [Q(7, 12), Q(1, 3), Q(1, 12)]
    weighted = _graph([(0, 1), (1, 2)], w_E=[Q(1), Q(1, 2)])
    assert list(resolvent_rank(weighted, 0, [1, 0, 0], Q(1, 2), "walk")) == [Q(5, 9), Q(1, 3), Q(1, 9)]
    five = _graph(PAIRWISE["path of 5"])
    assert list(resolvent_rank(five, 0, [1, 0, 0, 0, 0], Q(17, 20), "walk")) == [
        Q(438721, 1512560), Q(12461, 37814), Q(289, 1480), Q(4913, 37814), Q(83521, 1512560)]
    star = _graph(PAIRWISE["star"])
    assert list(resolvent_rank(star, 0, None, Q(17, 20), "walk")) == [Q(88, 185)] + [Q(97, 740)] * 4


@pytest.mark.parametrize("name", sorted(PAIRWISE))
@pytest.mark.parametrize("damping", [Q(1, 2), Q(17, 20), Q(0)])
def test_the_walk_metric_is_personalized_pagerank_on_pair_relations(name, damping):
    """Declared metric m is resistance: the adjacency walk reads the conductances 1/m."""
    from rexgraph.markov_oracle import PairwiseMarkovOracle
    from rexgraph.ranking_response import pagerank_solve
    pairs = PAIRWISE[name]
    metric = [Q(k % 3 + 1, k % 2 + 1) for k in range(len(pairs))]
    n = 1 + max(v for p in pairs for v in p)
    seed = [Q(k + 1) for k in range(n)]
    expected = pagerank_solve(PairwiseMarkovOracle(_graph(pairs, w_E=[1 / m for m in metric])), damping, seed)
    assert list(resolvent_rank(_graph(pairs, w_E=metric), 0, seed, damping, "walk")) == list(expected)


def test_faces_leave_the_walk_rank_unchanged_and_change_the_relation_rank():
    """The tetrahedron of the paper: three faces and all four have beta_1 = 0 and different responses."""
    faces = [[0, 1, 3], [0, 2, 4], [1, 2, 5], [3, 4, 5]]
    three, four = _tetrahedron(faces[:3]), _tetrahedron(faces)
    open_ = _graph(list(itertools.combinations(range(4), 2)))
    walk = [list(resolvent_rank(r, 0, [1, 0, 0, 0], Q(17, 20), "walk")) for r in (open_, three, four)]
    assert walk[0] == walk[1] == walk[2]
    seed = [0, 0, 0, 1, 0, 0]
    assert list(resolvent_rank(three, 1, seed, Q(1, 2))) == [0, 0, 0, Q(3, 10), Q(-1, 10), Q(1, 10)]
    assert list(resolvent_rank(four, 1, seed, Q(1, 2))) == [0, 0, 0, Q(1, 5), 0, 0]


def test_a_hole_is_never_damped_and_a_branching_relation_keeps_its_tail_difference():
    ring = _graph(PAIRWISE["ring of 5"])
    far = [resolvent_rank(ring, 1, [1, 0, 0, 0, 0], d)[2] for d in (Q(1, 2), Q(99, 100), Q(9999, 10000))]
    assert far[0] < far[1] < far[2] < Q(1, 5)
    relation = RexGraph.from_hypergraph(np.array([0, 3], np.int32), np.array([0, 1, 2], np.int32))
    result, info = resolvent_rank(relation, 0, [0, 1, 0], Q(1, 2), "walk", report=True)
    assert list(result) == [Q(1, 2), Q(3, 4), Q(-1, 4)]
    assert info["mass"] == 1 and info["signed"]


@pytest.mark.parametrize("name", sorted(PAIRWISE))
def test_the_walk_rank_conserves_mass_without_witnesses(name):
    rex = _graph(PAIRWISE[name])
    n = rex.nV
    result = resolvent_rank(rex, 0, [Q(k * k + 1) for k in range(n)], Q(3, 4), "walk")
    assert sum(result) == 1 and all(v >= 0 for v in result)


def test_degree_equals_walk_without_faces_and_derives_every_grade_with_them():
    open_ = _graph(PAIRWISE["K4"])
    assert rank_calculus(open_, 0, "degree").metrics == rank_calculus(open_, 0, "walk").metrics
    filled = _graph(list(itertools.combinations(range(4), 2)), w_E=[Q(k + 2) for k in range(6)])
    filled.add_faces([[0, 1, 3], [0, 2, 4]])
    filled._ensure_clean()
    walk, degree = rank_calculus(filled, 0, "walk"), rank_calculus(filled, 0, "degree")
    assert walk.metrics[1] == NativeFieldCalculus.from_rex(filled).metrics[1]
    edge_metric = degree.metrics[1].as_sparse().entries
    # Edge 0 lies on both faces, edges 1 to 4 on one each; edge 5 has no coface and keeps its declared 7.
    assert [edge_metric[e, e] for e in range(6)] == [Q(1, 2), Q(1), Q(1), Q(1), Q(1), Q(7)]


def test_the_completed_metric_is_the_induced_form_of_the_complete_coordinates():
    rex = _tetrahedron([[0, 1, 3]])
    calculus = rank_calculus(rex, 1, "completed")
    base = NativeFieldCalculus.from_rex(rex)
    eye = np.eye(6, dtype=int)
    form = base.hodge(1).apply(eye) + base.sector(1, "harmonic").apply(eye)
    assert calculus.metrics[1].as_sparse().entries == {
        (i, j): form[i, j] for i in range(6) for j in range(6) if form[i, j]}
    result = resolvent_rank(rex, 1, None, Q(1, 2), "completed")
    assert list(result) == list(calculus.green(1, 1).apply(np.full(6, Q(1, 6), object)))


def test_explicit_metrics_replace_the_policy_so_a_field_can_be_the_metric():
    rex = _graph(PAIRWISE["star"])
    walk_metric = rank_calculus(rex, 0, "walk").metrics[0].as_sparse().entries
    weights = [walk_metric[v, v] for v in range(5)]
    seed = [1, 2, 0, 0, 1]
    assert list(resolvent_rank(rex, 0, seed, Q(2, 3), metrics=[weights])) == list(
        resolvent_rank(rex, 0, seed, Q(2, 3), "walk"))


def test_a_field_calculus_source_uses_its_declared_metrics():
    rex = _tetrahedron([[0, 1, 3]])
    base = NativeFieldCalculus.from_rex(rex)
    assert list(resolvent_rank(base, 1, None, Q(1, 2))) == list(resolvent_rank(rex, 1, None, Q(1, 2)))
    declared = NativeFieldCalculus(base.complex, (
        CoordinateMetric.diagonal(base.complex.spaces[0], [Q(1), Q(2), Q(3), Q(4)]),
        CoordinateMetric.diagonal(base.complex.spaces[1], [Q(k + 1, 2) for k in range(6)]),
        CoordinateMetric.diagonal(base.complex.spaces[2], [Q(5, 3)])))
    result = resolvent_rank(declared, 1, None, Q(1, 2))
    assert list(result) == list(declared.green(1, 1).apply(np.full(6, Q(1, 6), object)))
    assert list(result) != list(resolvent_rank(rex, 1, None, Q(1, 2)))


@pytest.mark.parametrize("kwargs, error", [
    ({"damping": Q(1)}, ValueError), ({"damping": Q(-1, 2)}, ValueError), ({"damping": 0.85}, TypeError),
    ({"metric": "median"}, ValueError), ({"grade": 3}, ValueError), ({"grade": True}, TypeError),
    ({"seed": [0, 0, 0]}, ValueError), ({"seed": [1, -1, 1]}, ValueError), ({"seed": [1, 1]}, ValueError),
])
def test_invalid_requests_are_refused(kwargs, error):
    arguments = {"grade": 0, "seed": None, "damping": Q(1, 2), "metric": "declared", **kwargs}
    with pytest.raises(error):
        resolvent_rank(_graph([(0, 1), (1, 2)]), arguments.pop("grade"), **arguments)


def test_every_policy_is_named():
    assert POLICIES == ("declared", "walk", "degree", "completed")


def test_the_former_names_are_aliases():
    from rexgraph import markov, ranking_response
    from rexgraph.core import _standard
    assert markov.MarkovView is markov.ParticipationWalk
    assert markov.pagerank is markov.pagerank_iteration
    assert ranking_response.exact_pagerank is ranking_response.pagerank_solve
    assert ranking_response.pagerank_delta is ranking_response.pagerank_solve_delta
    assert _standard.pagerank is _standard.pagerank_iteration
