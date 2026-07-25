"""
Tests for rexgraph.core._cycles - deterministic fundamental cycle basis.

Verifies:
    - Symmetric adjacency has correct structure
    - BFS spanning forest is valid (tree edges = nV - beta_0)
    - Cycle count equals beta_1 = nE - nV + beta_0
    - Every cycle lies in ker(B1)
    - Deterministic: same graph always produces same cycles
    - Trees have no cycles
    - Disconnected graphs have correct component count
    - Cycle space dimension matches union-find computation
"""
import numpy as np
import pytest

from rexgraph.core import _cycles


# Helpers

def _triangle():
    """Triangle: 3V, 3E, beta_1 = 1."""
    return 3, 3, np.array([0, 1, 0], dtype=np.int32), np.array([1, 2, 2], dtype=np.int32)


def _k4():
    """K4: 4V, 6E, beta_1 = 3."""
    return 4, 6, np.array([0, 0, 0, 1, 1, 2], dtype=np.int32), np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)


def _tree():
    """Path 0-1-2-3: 4V, 3E, beta_1 = 0."""
    return 4, 3, np.array([0, 1, 2], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32)


def _two_components():
    """Two disconnected triangles: 6V, 6E, beta_0 = 2, beta_1 = 2."""
    src = np.array([0, 1, 0, 3, 4, 3], dtype=np.int32)
    tgt = np.array([1, 2, 2, 4, 5, 5], dtype=np.int32)
    return 6, 6, src, tgt


def _square():
    """Square (4-cycle): 4V, 4E, beta_1 = 1."""
    return 4, 4, np.array([0, 1, 2, 0], dtype=np.int32), np.array([1, 2, 3, 3], dtype=np.int32)


# Symmetric Adjacency

class TestSymmetricAdjacency:

    def test_shape(self):
        nV, nE, s, t = _triangle()
        ptr, idx, edge = _cycles.build_symmetric_adjacency(nV, nE, s, t)
        assert ptr.shape == (nV + 1,)
        assert idx.shape == (2 * nE,)
        assert edge.shape == (2 * nE,)

    def test_degree_sum(self):
        """Sum of row lengths = 2 * nE."""
        nV, nE, s, t = _triangle()
        ptr, idx, edge = _cycles.build_symmetric_adjacency(nV, nE, s, t)
        assert ptr[-1] == 2 * nE

    def test_rows_sorted(self):
        """Neighbors within each row are sorted by vertex index."""
        nV, nE, s, t = _k4()
        ptr, idx, edge = _cycles.build_symmetric_adjacency(nV, nE, s, t)
        for v in range(nV):
            row = idx[ptr[v]:ptr[v+1]]
            assert np.all(np.diff(row) >= 0)

    def test_edge_index_valid(self):
        """Every adj_edge entry is a valid edge index."""
        nV, nE, s, t = _k4()
        ptr, idx, edge = _cycles.build_symmetric_adjacency(nV, nE, s, t)
        assert np.all(edge >= 0)
        assert np.all(edge < nE)


# BFS Spanning Forest

class TestBFSForest:

    def test_tree_edges_count(self):
        """Number of tree edges = nV - beta_0."""
        nV, nE, s, t = _triangle()
        ptr, idx, edge = _cycles.build_symmetric_adjacency(nV, nE, s, t)
        par, par_e, dep, itree, nc = _cycles.bfs_spanning_forest(ptr, idx, edge, nV, nE)
        assert itree.sum() == nV - nc

    def test_single_component(self):
        nV, nE, s, t = _triangle()
        ptr, idx, edge = _cycles.build_symmetric_adjacency(nV, nE, s, t)
        _, _, _, _, nc = _cycles.bfs_spanning_forest(ptr, idx, edge, nV, nE)
        assert nc == 1

    def test_two_components(self):
        nV, nE, s, t = _two_components()
        ptr, idx, edge = _cycles.build_symmetric_adjacency(nV, nE, s, t)
        _, _, _, _, nc = _cycles.bfs_spanning_forest(ptr, idx, edge, nV, nE)
        assert nc == 2

    def test_depth_nonnegative(self):
        nV, nE, s, t = _k4()
        ptr, idx, edge = _cycles.build_symmetric_adjacency(nV, nE, s, t)
        _, _, dep, _, _ = _cycles.bfs_spanning_forest(ptr, idx, edge, nV, nE)
        assert np.all(dep >= 0)

    def test_root_depth_zero(self):
        nV, nE, s, t = _triangle()
        ptr, idx, edge = _cycles.build_symmetric_adjacency(nV, nE, s, t)
        par, _, dep, _, _ = _cycles.bfs_spanning_forest(ptr, idx, edge, nV, nE)
        # Root vertices have parent[v] == v and depth 0
        roots = np.where(np.arange(nV, dtype=np.int32) == par)[0]
        assert all(dep[r] == 0 for r in roots)


# Fundamental Cycles

class TestFundamentalCycles:

    def test_triangle_one_cycle(self):
        nV, nE, s, t = _triangle()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        assert nF == 1
        assert nc == 1

    def test_k4_three_cycles(self):
        nV, nE, s, t = _k4()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        assert nF == 3  # beta_1 = 6 - 4 + 1 = 3

    def test_tree_no_cycles(self):
        nV, nE, s, t = _tree()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        assert nF == 0
        assert len(ce) == 0

    def test_two_components_two_cycles(self):
        nV, nE, s, t = _two_components()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        assert nF == 2  # beta_1 = 6 - 6 + 2 = 2
        assert nc == 2

    def test_square_one_cycle(self):
        nV, nE, s, t = _square()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        assert nF == 1

    def test_cycle_lengths_positive(self):
        nV, nE, s, t = _k4()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        assert np.all(cl > 0)

    def test_total_length_matches(self):
        """Sum of cycle_lengths = len(cycle_edges)."""
        nV, nE, s, t = _k4()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        assert cl.sum() == len(ce)

    def test_signs_are_pm1(self):
        nV, nE, s, t = _k4()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        assert np.all(np.abs(cs) == 1.0)


# Cycles in ker(B1)

class TestCyclesInKernel:

    def test_triangle_in_kernel(self):
        nV, nE, s, t = _triangle()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        ok, err = _cycles.verify_cycles_in_kernel(nV, nE, s, t, ce, cs, cl)
        assert ok
        assert err < 1e-10

    def test_k4_in_kernel(self):
        nV, nE, s, t = _k4()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        ok, err = _cycles.verify_cycles_in_kernel(nV, nE, s, t, ce, cs, cl)
        assert ok
        assert err < 1e-10

    def test_two_components_in_kernel(self):
        nV, nE, s, t = _two_components()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        ok, err = _cycles.verify_cycles_in_kernel(nV, nE, s, t, ce, cs, cl)
        assert ok

    def test_square_in_kernel(self):
        nV, nE, s, t = _square()
        ce, cs, cl, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
        ok, err = _cycles.verify_cycles_in_kernel(nV, nE, s, t, ce, cs, cl)
        assert ok


# Determinism

class TestDeterminism:

    def test_same_result_twice(self):
        """Same graph produces identical cycle basis on repeated calls."""
        nV, nE, s, t = _k4()
        r1 = _cycles.find_fundamental_cycles(nV, nE, s, t)
        r2 = _cycles.find_fundamental_cycles(nV, nE, s, t)
        assert np.array_equal(r1[0], r2[0])  # cycle_edges
        assert np.array_equal(r1[1], r2[1])  # cycle_signs
        assert np.array_equal(r1[2], r2[2])  # cycle_lengths


# Cycle Space Dimension

class TestCycleSpaceDimension:

    def test_triangle(self):
        nV, nE, s, t = _triangle()
        assert _cycles.cycle_space_dimension(nV, nE, s, t) == 1

    def test_tree(self):
        nV, nE, s, t = _tree()
        assert _cycles.cycle_space_dimension(nV, nE, s, t) == 0

    def test_k4(self):
        nV, nE, s, t = _k4()
        assert _cycles.cycle_space_dimension(nV, nE, s, t) == 3

    def test_matches_find_cycles(self):
        """cycle_space_dimension matches nF from find_fundamental_cycles."""
        nV, nE, s, t = _k4()
        dim = _cycles.cycle_space_dimension(nV, nE, s, t)
        _, _, _, nF, _ = _cycles.find_fundamental_cycles(nV, nE, s, t)
        assert dim == nF

    def test_formula(self):
        """beta_1 = nE - nV + beta_0."""
        for data in [_triangle(), _k4(), _tree(), _two_components(), _square()]:
            nV, nE, s, t = data
            _, _, _, nF, nc = _cycles.find_fundamental_cycles(nV, nE, s, t)
            assert nF == nE - nV + nc
