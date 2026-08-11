"""Significance as a field: the quadrance is the score, the spread is the distance."""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.semantic import (
    relation_field,
    semantic_gram,
    semantic_spread,
    significance,
)


def _g(s, t):
    rex = RexGraph(sources=np.asarray(s, np.int32), targets=np.asarray(t, np.int32))
    rex._ensure_clean()
    return rex


def _hyper(groups):
    ptr, idx = [0], []
    for grp in groups:
        idx.extend(grp)
        ptr.append(len(idx))
    rex = RexGraph.from_hypergraph(np.array(ptr, np.int32), np.array(idx, np.int32))
    rex._ensure_clean()
    return rex


def test_the_quadrance_is_the_significance():
    for rex in (_g([0, 1, 2, 0, 3], [1, 2, 0, 3, 4]),
                _g([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]),
                _hyper([[0, 1, 2], [2, 3, 4], [0, 4]])):
        _V, q = relation_field(rex)
        reff = np.asarray(rex._effective_resistance_batch(np.arange(int(rex.nE))))
        assert np.allclose(q, reff, atol=1e-9)
        assert np.allclose(significance(rex), reff, atol=1e-9)


def test_the_field_sums_to_the_rank():
    rex = _g([0, 1, 2, 0, 3], [1, 2, 0, 3, 4])
    _V, q = relation_field(rex)
    assert float(q.sum()) == pytest.approx(int(rex.nV) - int(rex.betti[0]), abs=1e-9)


def test_the_embedding_is_a_vector_per_relation_in_vertex_space():
    rex = _g([0, 1, 2, 0, 3], [1, 2, 0, 3, 4])
    V, q = relation_field(rex)
    assert V.shape == (int(rex.nV), int(rex.nE))
    assert q.shape == (int(rex.nE),)


def test_the_gram_is_psd_and_its_diagonal_is_the_significance():
    rex = _g([0, 1, 2, 0, 3], [1, 2, 0, 3, 4])
    G = semantic_gram(rex)
    assert np.allclose(G, G.T, atol=1e-9)
    assert np.min(np.linalg.eigvalsh(G)) > -1e-9          # PSD, so it is a kernel
    assert np.allclose(np.diag(G), significance(rex), atol=1e-9)


def test_the_spread_is_a_distance_on_the_field():
    rex = _g([0, 1, 2, 0, 3], [1, 2, 0, 3, 4])
    S = semantic_spread(rex)
    assert np.allclose(S, S.T, atol=1e-12)
    assert np.allclose(np.diag(S), 0.0, atol=1e-12)
    assert bool(((S >= -1e-9) & (S <= 1 + 1e-9)).all())


def test_parallel_relations_are_at_distance_zero():
    """Two relations over the same pair move the complex identically."""
    rex = _g([0, 0, 1], [1, 1, 2])
    S = semantic_spread(rex)
    assert S[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert S[0, 2] > 0.0


def test_disjoint_relations_are_orthogonal():
    """Nothing shared means nothing moved together: spread 1."""
    rex = _g([0, 2], [1, 3])
    S = semantic_spread(rex)
    assert S[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_the_spread_is_the_gram_block_over_its_diagonal():
    """Section 1, with the field as its vectors."""
    rex = _g([0, 1, 2, 0, 3], [1, 2, 0, 3, 4])
    G = semantic_gram(rex)
    S = semantic_spread(rex)
    q = np.diag(G)
    for i in range(len(q)):
        for j in range(len(q)):
            if i != j and q[i] > 1e-12 and q[j] > 1e-12:
                assert S[i, j] == pytest.approx(
                    1.0 - G[i, j] ** 2 / (q[i] * q[j]), abs=1e-9)


def test_it_works_at_arity():
    rex = _hyper([[0, 1, 2, 3], [2, 3, 4], [4, 5]])
    V, q = relation_field(rex)
    assert V.shape[1] == int(rex.nE)
    assert float(q.sum()) == pytest.approx(int(rex.nV) - int(rex.betti[0]), abs=1e-9)
    S = semantic_spread(rex)
    assert bool(((S >= -1e-9) & (S <= 1 + 1e-9)).all())
