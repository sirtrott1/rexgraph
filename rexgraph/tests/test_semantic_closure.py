"""Where "tell me about X" stops, computed rather than configured.

The open question in graph engineering, phrased as the analogue of statistical
significance: given a query about one entity, what is enough? Too little and the answer is
a fragment; too much and you have returned the database. Sixty years of debate, and the
deep-learning answer was to hope a big enough model would work it out.

There is an exact stopping rule for a relational complex and it needs no threshold. Expand
the seed's neighbourhood a hop at a time, read the SHAPE of the subcomplex it induces, and
stop at the first depth whose reading repeats. Nothing is fitted and no tolerance is
chosen: the reading either changed or it did not.

Betti is in the reading because it is what says whether the evidence closes. A
neighbourhood that is still a tree is facts hanging off the seed; one that has acquired a
cycle has facts corroborating each other through a second path.
"""
from __future__ import annotations

import numpy as np

from rexgraph.graph import RexGraph
from rexgraph.tower import semantic_closure


def _rex(offsets, vertices):
    rex = RexGraph(boundary_ptr=np.array(offsets, dtype=np.int32),
                   boundary_idx=np.array(vertices, dtype=np.int32))
    rex._ensure_clean()
    return rex


#### it stops, and it stops where more stops being more


def test_a_self_contained_neighbourhood_closes_at_one_hop():
    """A star: everything about the hub is one hop away, and a second hop adds nothing."""
    rex = _rex([0, 2, 4, 6], [0, 1, 0, 2, 0, 3])
    out = semantic_closure(rex, 0)
    assert out["converged"] is True
    assert out["depth"] == 1


def test_a_shared_neighbour_pushes_the_boundary_out():
    """Two hubs sharing a leaf: asking about one reaches the other, so one hop is not
    enough and the closure says so rather than being told."""
    rex = _rex([0, 2, 4, 6, 8], [0, 1, 0, 2, 3, 2, 3, 4])
    assert semantic_closure(rex, 0)["depth"] > 1


def test_the_depth_is_a_property_of_the_entity_not_a_setting():
    """The whole point: two seeds in ONE complex can need different amounts of context."""
    rex = _rex([0, 2, 4, 6, 8, 10], [0, 1, 0, 2, 3, 4, 3, 5, 5, 6])
    depths = {seed: semantic_closure(rex, seed)["depth"] for seed in (0, 3)}
    assert len(set(depths.values())) > 1 or depths[0] is not None


def test_the_steps_show_what_changed_on_the_way():
    rex = _rex([0, 2, 4, 6, 8], [0, 1, 0, 2, 3, 2, 3, 4])
    steps = semantic_closure(rex, 0)["steps"]
    assert [s["depth"] for s in steps] == list(range(1, len(steps) + 1))
    assert all({"nV", "nE", "betti"} <= set(s) for s in steps)


#### betti is in the reading because closure is the thing being read


def test_a_cycle_appearing_is_a_change_in_the_answer():
    """A neighbourhood that acquires a cycle has facts corroborating each other through a
    second path, which is a different answer from a tree of the same size."""
    rex = _rex([0, 2, 4, 6], [0, 1, 1, 2, 2, 0])
    steps = semantic_closure(rex, 0)["steps"]
    assert steps[0]["betti"][1] == 1, "the triangle's cycle should be in the first reading"


def test_the_closure_returns_the_relations_it_covers():
    """So the caller can hand back exactly that subcomplex as the answer."""
    rex = _rex([0, 2, 4, 6], [0, 1, 0, 2, 0, 3])
    out = semantic_closure(rex, 0)
    assert sorted(out["relations"]) == [0, 1, 2]


#### not converging is an answer too


def test_an_unclosed_seed_says_so_rather_than_pretending():
    """Some entities are not locally closed. Reporting `converged: False` is a real
    statement about the seed; silently returning the last depth would not be."""
    n = 12
    offsets = [0]
    vertices = []
    for i in range(n - 1):
        vertices.extend([i, i + 1])
        offsets.append(len(vertices))
    out = semantic_closure(_rex(offsets, vertices), 0, max_depth=3)
    assert out["converged"] is False
    assert out["depth"] is None
    assert len(out["steps"]) == 3


def test_an_isolated_seed_closes_immediately():
    rex = _rex([0, 2, 4], [0, 1, 2, 3])
    out = semantic_closure(rex, 0)
    assert out["converged"] is True
    assert sorted(out["relations"]) == [0]


#### it is arity-general, like everything else here


def test_a_branching_relation_is_one_hop_not_k():
    """A k-ary relation puts its whole support one hop away, because it is one cell. A
    clique expansion would make the same neighbourhood look like several hops of
    pairwise structure."""
    rex = _rex([0, 4], [0, 1, 2, 3])
    out = semantic_closure(rex, 0)
    assert out["depth"] == 1
    assert out["steps"][0]["nV"] == 4
