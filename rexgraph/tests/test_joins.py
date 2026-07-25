"""
Tests for rexgraph.core._joins -- chain complex join operations.

Verifies:
    - Shared vertex map: correct matching by label
    - Inner join: chain condition, dimensions <= min(R, S)
    - Outer join: chain condition, dimensions >= max(R, S)
    - Left join: chain condition, keeps all R edges, adds new S edges
    - Attribute merge: blending formula correct
    - Integration through RexGraph: inner_join, outer_join, left_join
"""
import numpy as np
import pytest

from rexgraph.core import _joins
from rexgraph.graph import RexGraph


# Fixtures

@pytest.fixture
def triangle_a():
    """Triangle on vertices 0,1,2."""
    return RexGraph.from_simplicial(
        sources=np.array([0, 1, 0], dtype=np.int32),
        targets=np.array([1, 2, 2], dtype=np.int32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )


@pytest.fixture
def triangle_b():
    """Triangle on vertices 1,2,3 (shares vertices 1,2 with triangle_a)."""
    return RexGraph.from_simplicial(
        sources=np.array([1, 2, 1], dtype=np.int32),
        targets=np.array([2, 3, 3], dtype=np.int32),
        triangles=np.array([[1, 2, 3]], dtype=np.int32),
    )


@pytest.fixture
def shared_verts():
    """shared_vertices[v_A] = v_B or -1. Vertices 1,2 are shared."""
    # triangle_a has V={0,1,2}, triangle_b has V={1,2,3}
    # A's vertex 0 -> not shared
    # A's vertex 1 -> B's vertex 0 (which is label 1)
    # A's vertex 2 -> B's vertex 1 (which is label 2)
    return np.array([-1, 0, 1], dtype=np.int32)


# Shared Vertex Map

class TestSharedVertexMap:

    def test_basic_matching(self):
        labels_R = ['a', 'b', 'c']
        labels_S = ['b', 'c', 'd']
        sv = _joins.build_shared_vertex_map(labels_R, labels_S)
        assert sv[0] == -1  # 'a' not in S
        assert sv[1] == 0   # 'b' -> index 0 in S
        assert sv[2] == 1   # 'c' -> index 1 in S

    def test_no_overlap(self):
        labels_R = ['a', 'b']
        labels_S = ['c', 'd']
        sv = _joins.build_shared_vertex_map(labels_R, labels_S)
        assert np.all(sv == -1)

    def test_full_overlap(self):
        labels = ['x', 'y', 'z']
        sv = _joins.build_shared_vertex_map(labels, labels)
        assert list(sv) == [0, 1, 2]


# Inner Join

class TestInnerJoin:

    def test_chain_condition(self, triangle_a, triangle_b, shared_verts):
        result = _joins.inner_join(
            triangle_a.B1, triangle_a.B2, triangle_a.nV, triangle_a.nE, triangle_a.nF,
            triangle_b.B1, triangle_b.B2, triangle_b.nV, triangle_b.nE, triangle_b.nF,
            shared_verts)
        assert result['chain_residual'] < 1e-10

    def test_dimensions(self, triangle_a, triangle_b, shared_verts):
        result = _joins.inner_join(
            triangle_a.B1, triangle_a.B2, triangle_a.nV, triangle_a.nE, triangle_a.nF,
            triangle_b.B1, triangle_b.B2, triangle_b.nV, triangle_b.nE, triangle_b.nF,
            shared_verts)
        # Inner join has at most min(nV_R, nV_S) shared vertices
        assert result['nVj'] <= min(triangle_a.nV, triangle_b.nV)
        assert result['nEj'] <= min(triangle_a.nE, triangle_b.nE)

    def test_returns_keys(self, triangle_a, triangle_b, shared_verts):
        result = _joins.inner_join(
            triangle_a.B1, triangle_a.B2, triangle_a.nV, triangle_a.nE, triangle_a.nF,
            triangle_b.B1, triangle_b.B2, triangle_b.nV, triangle_b.nE, triangle_b.nF,
            shared_verts)
        for key in ['B1j', 'B2j', 'nVj', 'nEj', 'nFj', 'beta', 'chain_residual']:
            assert key in result


# Outer Join

class TestOuterJoin:

    def test_chain_condition(self, triangle_a, triangle_b, shared_verts):
        result = _joins.outer_join(
            triangle_a.B1, triangle_a.B2, triangle_a.nV, triangle_a.nE, triangle_a.nF,
            triangle_b.B1, triangle_b.B2, triangle_b.nV, triangle_b.nE, triangle_b.nF,
            shared_verts)
        assert result['chain_residual'] < 1e-10

    def test_dimensions_grow(self, triangle_a, triangle_b, shared_verts):
        result = _joins.outer_join(
            triangle_a.B1, triangle_a.B2, triangle_a.nV, triangle_a.nE, triangle_a.nF,
            triangle_b.B1, triangle_b.B2, triangle_b.nV, triangle_b.nE, triangle_b.nF,
            shared_verts)
        # Outer join has all edges from both
        assert result['nEj'] == triangle_a.nE + triangle_b.nE
        assert result['nFj'] == triangle_a.nF + triangle_b.nF
        # Shared vertices are merged, so nVj < nV_R + nV_S
        assert result['nVj'] <= triangle_a.nV + triangle_b.nV

    def test_betti(self, triangle_a, triangle_b, shared_verts):
        result = _joins.outer_join(
            triangle_a.B1, triangle_a.B2, triangle_a.nV, triangle_a.nE, triangle_a.nF,
            triangle_b.B1, triangle_b.B2, triangle_b.nV, triangle_b.nE, triangle_b.nF,
            shared_verts)
        b0, b1, b2 = result['beta']
        assert b0 >= 1  # at least one connected component


# Left Join

class TestLeftJoin:

    def test_chain_condition(self, triangle_a, triangle_b, shared_verts):
        result = _joins.left_join(
            triangle_a.B1, triangle_a.B2, triangle_a.nV, triangle_a.nE, triangle_a.nF,
            triangle_b.B1, triangle_b.B2, triangle_b.nV, triangle_b.nE, triangle_b.nF,
            shared_verts)
        assert result['chain_residual'] < 1e-10

    def test_keeps_r_vertices(self, triangle_a, triangle_b, shared_verts):
        result = _joins.left_join(
            triangle_a.B1, triangle_a.B2, triangle_a.nV, triangle_a.nE, triangle_a.nF,
            triangle_b.B1, triangle_b.B2, triangle_b.nV, triangle_b.nE, triangle_b.nF,
            shared_verts)
        assert result['nVj'] == triangle_a.nV

    def test_edges_at_least_r(self, triangle_a, triangle_b, shared_verts):
        result = _joins.left_join(
            triangle_a.B1, triangle_a.B2, triangle_a.nV, triangle_a.nE, triangle_a.nF,
            triangle_b.B1, triangle_b.B2, triangle_b.nV, triangle_b.nE, triangle_b.nF,
            shared_verts)
        assert result['nEj'] >= triangle_a.nE


# Attribute Merge

class TestAttributeMerge:

    def test_blend_formula(self):
        ew_R = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        amps_R = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ew_S = np.array([1.0, 1.0], dtype=np.float64)
        amps_S = np.array([10.0, 20.0], dtype=np.float64)
        sv = np.array([-1, 0, 1], dtype=np.int32)  # v1->s0, v2->s1
        result = _joins.attribute_merge(3, 3, ew_R, amps_R, ew_S, amps_S, sv, alpha=0.5)
        # v0: no blend
        assert result['merged_amps'][0] == 1.0
        # v1: 0.5*2.0 + 0.5*10.0 = 6.0
        assert abs(result['merged_amps'][1] - 6.0) < 1e-10
        # v2: 0.5*3.0 + 0.5*20.0 = 11.5
        assert abs(result['merged_amps'][2] - 11.5) < 1e-10
        assert result['n_enriched'] == 2


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_inner_join(self, triangle_a, triangle_b, shared_verts):
        result = triangle_a.inner_join(triangle_b, shared_verts)
        assert isinstance(result, dict)
        assert 'B1j' in result
        assert result['chain_residual'] < 1e-10

    def test_outer_join(self, triangle_a, triangle_b, shared_verts):
        result = triangle_a.outer_join(triangle_b, shared_verts)
        assert isinstance(result, dict)
        assert result['chain_residual'] < 1e-10

    def test_left_join(self, triangle_a, triangle_b, shared_verts):
        result = triangle_a.left_join(triangle_b, shared_verts)
        assert isinstance(result, dict)
        assert result['nVj'] == triangle_a.nV
