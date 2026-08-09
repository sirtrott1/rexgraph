"""The clique expansion is a lossy shadow of the kernel, and must not be a live path.

`rank(B_1) = n_0 - c` is a GRAPH identity. An arity-k relation touches k vertices while
contributing rank one, so reaching a new vertex stops meaning reaching a new direction,
and a cycle is the absence of the second. Replacing the relation with C(k,2) pairwise
edges therefore does not approximate the kernel, it answers a different question.

Concretely: a lone k-ary relation encloses nothing, so its cycle space is empty. The
expansion turns it into a k-clique, which has C(k-1,2) independent cycles. Every one of
them is manufactured by the encoding.

The expansion stays in the library because it is what DEMONSTRATES the loss. These tests
hold it to that role: available to compare against, never the route an answer comes
through.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.graded_boundary import _sparse_rank
from rexgraph.graph import RexGraph


def _dim_ker_B1(rex) -> int:
    return int(rex.nE) - int(_sparse_rank(sp.csr_matrix(rex.B1)))


def _lone(k: int) -> RexGraph:
    """One relation of arity k and nothing else. It bounds nothing."""
    return RexGraph.from_hypergraph(np.array([0, k], dtype=np.int32),
                                    np.arange(k, dtype=np.int32))


@pytest.mark.parametrize("k,invented", [(3, 1), (4, 3), (5, 6), (6, 10)])
def test_the_expansion_invents_c_k_minus_1_choose_2_cycles(k, invented):
    """The measurement, not the assertion. A lone k-ary relation has no cycle; its
    clique expansion reports C(k-1, 2) of them."""
    g = _lone(k)
    assert _dim_ker_B1(g) == 0, "a lone relation should enclose nothing"
    assert _dim_ker_B1(g.clique_expansion) == invented


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_the_expansion_also_invents_relations(k):
    g = _lone(k)
    assert g.nE == 1
    assert g.clique_expansion.nE == k * (k - 1) // 2


#### the live path does not go through it


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_the_cycle_basis_of_a_lone_relation_is_empty(k):
    """It used to raise IndexError here: the expansion's cycles came back indexed
    against C(k,2) edges and were written into an array of length nE."""
    assert _lone(k).cycle_basis == []


def test_a_branching_complex_gets_the_exact_cycle_dimension():
    g = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10, 12], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32))
    assert g.has_branching
    assert len(g.cycle_basis) == _dim_ker_B1(g)


@pytest.mark.parametrize("build,expected", [
    (lambda: RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10, 12], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32)), 2),
    (lambda: RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                      targets=np.array([1, 2, 0], dtype=np.int32)), 1),
    (lambda: RexGraph(sources=np.array([0, 0, 1], dtype=np.int32),
                      targets=np.array([1, 2, 3], dtype=np.int32)), 0),
], ids=["branching", "triangle", "tree"])
def test_every_basis_vector_is_actually_a_cycle(build, expected):
    """B_1 c = 0 exactly, not nearly: the branching route solves over the rationals."""
    g = build()
    basis = g.cycle_basis
    assert len(basis) == expected
    for c in basis:
        assert float(np.abs(g.B1 @ c).max()) == 0.0


def test_the_pairwise_route_is_unchanged():
    """Only branching was rerouted; a pairwise complex still takes the compiled
    tree-cotree traversal."""
    g = RexGraph(sources=np.array([0, 1, 2, 3], dtype=np.int32),
                 targets=np.array([1, 2, 3, 0], dtype=np.int32))
    assert not g.has_branching
    assert len(g.cycle_basis) == 1


def test_the_expansion_is_still_available_to_compare_against():
    """Keeping it is the point: it is the evidence, so removing it would remove the
    demonstration along with the defect."""
    g = _lone(4)
    assert g.clique_expansion.nE == 6
