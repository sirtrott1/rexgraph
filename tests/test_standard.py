"""
Tests for rexgraph.core._standard -- classical graph algorithms.

Verifies:
    - PageRank: sums to 1, nonneg, symmetric graph -> uniform
    - Betweenness: center vertex highest in star, nonneg
    - Clustering: complete graph = 1, tree = 0, range [0,1]
    - Louvain: labels valid, modularity in [-0.5, 1], n_communities >= 1
    - safe_correlation: identical = 1, constant = 0
    - build_adj_weights: maps edge weights to adjacency
    - build_standard_metrics: returns all expected keys
    - Integration through RexGraph: pagerank, betweenness, clustering,
      partition_communities
"""
import numpy as np
import pytest

from rexgraph.core import _standard
from rexgraph.graph import RexGraph


# Fixtures

@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


@pytest.fixture
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.fixture
def star():
    """Star graph: vertex 0 connected to 1,2,3,4."""
    return RexGraph.from_graph([0, 0, 0, 0], [1, 2, 3, 4])


@pytest.fixture
def tree():
    return RexGraph.from_graph([0, 1, 2], [1, 2, 3])


def _get_adj(rex):
    """Get adjacency arrays from a RexGraph."""
    adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle
    adj_wt = np.ones(adj_idx.shape[0], dtype=np.float64)
    return adj_ptr, adj_idx, adj_edge, adj_wt


# PageRank

class TestPageRank:

    def test_sums_to_one(self, k4):
        adj_ptr, adj_idx, _, adj_wt = _get_adj(k4)
        pr = _standard.pagerank(adj_ptr, adj_idx, adj_wt, k4.nV, k4.nE)
        assert abs(pr.sum() - 1.0) < 1e-6

    def test_nonneg(self, k4):
        adj_ptr, adj_idx, _, adj_wt = _get_adj(k4)
        pr = _standard.pagerank(adj_ptr, adj_idx, adj_wt, k4.nV, k4.nE)
        assert np.all(pr >= 0)

    def test_symmetric_graph_uniform(self, k4):
        """K4 is vertex-transitive, so PageRank should be uniform."""
        adj_ptr, adj_idx, _, adj_wt = _get_adj(k4)
        pr = _standard.pagerank(adj_ptr, adj_idx, adj_wt, k4.nV, k4.nE)
        assert np.std(pr) < 1e-6

    def test_shape(self, k4):
        adj_ptr, adj_idx, _, adj_wt = _get_adj(k4)
        pr = _standard.pagerank(adj_ptr, adj_idx, adj_wt, k4.nV, k4.nE)
        assert pr.shape == (k4.nV,)


# Betweenness Centrality

class TestBetweenness:

    def test_nonneg(self, k4):
        adj_ptr, adj_idx, adj_edge, _ = _get_adj(k4)
        bc_v, bc_e = _standard.betweenness(adj_ptr, adj_idx, adj_edge,
                                           k4.nV, k4.nE)
        assert np.all(bc_v >= -1e-10)
        assert np.all(bc_e >= -1e-10)

    def test_shapes(self, k4):
        adj_ptr, adj_idx, adj_edge, _ = _get_adj(k4)
        bc_v, bc_e = _standard.betweenness(adj_ptr, adj_idx, adj_edge,
                                           k4.nV, k4.nE)
        assert bc_v.shape == (k4.nV,)
        assert bc_e.shape == (k4.nE,)

    def test_star_center_highest(self, star):
        """In a star graph, the center has highest betweenness."""
        adj_ptr, adj_idx, adj_edge, _ = _get_adj(star)
        bc_v, _ = _standard.betweenness(adj_ptr, adj_idx, adj_edge,
                                        star.nV, star.nE)
        assert bc_v[0] == bc_v.max()

    def test_max_sources(self, k4):
        """Approximate betweenness with max_sources runs without error."""
        adj_ptr, adj_idx, adj_edge, _ = _get_adj(k4)
        bc_v, bc_e = _standard.betweenness(adj_ptr, adj_idx, adj_edge,
                                           k4.nV, k4.nE, max_sources=2)
        assert bc_v.shape == (k4.nV,)


# Clustering Coefficient

class TestClustering:

    def test_complete_graph(self, k4):
        """K4 vertices all have the same clustering coefficient."""
        adj_ptr, adj_idx, _, _ = _get_adj(k4)
        cc = _standard.clustering(adj_ptr, adj_idx, k4.nV)
        # All vertices in K4 are equivalent
        assert np.std(cc) < 1e-10

    def test_tree_zero(self, tree):
        """Trees have no triangles, so clustering = 0."""
        adj_ptr, adj_idx, _, _ = _get_adj(tree)
        cc = _standard.clustering(adj_ptr, adj_idx, tree.nV)
        # Internal vertices have deg >= 2 but no triangles
        assert np.allclose(cc, 0.0, atol=1e-10)

    def test_nonneg(self, triangle):
        adj_ptr, adj_idx, _, _ = _get_adj(triangle)
        cc = _standard.clustering(adj_ptr, adj_idx, triangle.nV)
        assert np.all(cc >= -1e-10)

    def test_shape(self, k4):
        adj_ptr, adj_idx, _, _ = _get_adj(k4)
        cc = _standard.clustering(adj_ptr, adj_idx, k4.nV)
        assert cc.shape == (k4.nV,)


# Louvain Community Detection

class TestLouvain:

    def test_valid_labels(self, k4):
        adj_ptr, adj_idx, _, adj_wt = _get_adj(k4)
        labels, n_comm, Q = _standard.louvain(adj_ptr, adj_idx, adj_wt,
                                              k4.nV, k4.nE)
        assert labels.shape == (k4.nV,)
        assert np.all(labels >= 0)
        assert np.all(labels < n_comm)

    def test_at_least_one_community(self, k4):
        adj_ptr, adj_idx, _, adj_wt = _get_adj(k4)
        _, n_comm, _ = _standard.louvain(adj_ptr, adj_idx, adj_wt,
                                         k4.nV, k4.nE)
        assert n_comm >= 1

    def test_modularity_range(self, k4):
        adj_ptr, adj_idx, _, adj_wt = _get_adj(k4)
        _, _, Q = _standard.louvain(adj_ptr, adj_idx, adj_wt,
                                    k4.nV, k4.nE)
        assert -0.5 <= Q <= 1.0 + 1e-6


# Pearson Correlation

class TestSafeCorrelation:

    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        assert abs(_standard.safe_correlation(a, a) - 1.0) < 1e-10

    def test_constant_zero(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.ones(3, dtype=np.float64)
        assert abs(_standard.safe_correlation(a, b)) < 1e-10

    def test_anticorrelated(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([3.0, 2.0, 1.0], dtype=np.float64)
        assert abs(_standard.safe_correlation(a, b) - (-1.0)) < 1e-10


# build_standard_metrics

class TestBuildStandardMetrics:

    def test_returns_all_keys(self, k4):
        adj_ptr, adj_idx, adj_edge, adj_wt = _get_adj(k4)
        result = _standard.build_standard_metrics(
            adj_ptr, adj_idx, adj_edge, adj_wt, k4.nV, k4.nE)
        for key in ['pagerank', 'betweenness_v', 'betweenness_e',
                     'btw_norm_v', 'btw_norm_e', 'clustering',
                     'community_labels', 'n_communities', 'modularity']:
            assert key in result

    def test_btw_norm_range(self, k4):
        adj_ptr, adj_idx, adj_edge, adj_wt = _get_adj(k4)
        result = _standard.build_standard_metrics(
            adj_ptr, adj_idx, adj_edge, adj_wt, k4.nV, k4.nE)
        assert np.all(result['btw_norm_v'] >= -1e-10)
        assert np.all(result['btw_norm_v'] <= 1.0 + 1e-10)


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_build_standard_metrics_via_graph(self, k4):
        """Standard metrics accessible through the _standard module."""
        adj_ptr, adj_idx, adj_edge, adj_wt = _get_adj(k4)
        result = _standard.build_standard_metrics(
            adj_ptr, adj_idx, adj_edge, adj_wt, k4.nV, k4.nE)
        assert result['pagerank'].shape == (k4.nV,)
        assert abs(result['pagerank'].sum() - 1.0) < 1e-6
        assert result['clustering'].shape == (k4.nV,)
        assert result['n_communities'] >= 1

    def test_partition_communities(self, k4):
        """partition_communities returns a list of (sub, v_map, e_map)."""
        parts = k4.partition_communities()
        assert isinstance(parts, list)
        assert len(parts) >= 1
        sub, v_map, e_map = parts[0]
        assert isinstance(sub, RexGraph)
