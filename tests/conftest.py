"""Shared test fixtures for rexgraph."""
import pytest
import numpy as np

@pytest.fixture
def small_graph():
    """4-vertex, 5-edge, 1-face test graph."""
    edges = [(0,1),(1,2),(0,2),(0,3),(1,3)]
    sources = np.array([e[0] for e in edges], dtype=np.int32)
    targets = np.array([e[1] for e in edges], dtype=np.int32)
    return {'sources': sources, 'targets': targets, 'nV': 4, 'nE': 5}

@pytest.fixture
def k4_graph():
    """Complete graph K4: 4 vertices, 6 edges, 4 faces."""
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    sources = np.array([e[0] for e in edges], dtype=np.int32)
    targets = np.array([e[1] for e in edges], dtype=np.int32)
    return {'sources': sources, 'targets': targets, 'nV': 4, 'nE': 6}
