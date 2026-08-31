"""Shared test fixtures and configuration for rexgraph."""
import pathlib
import sys

import numpy as np
import pytest

collect_ignore_glob = ["**/*.pyx"]


def _ensure_sibling(name: str) -> None:
    """Reach a sibling distribution when it is not installed.

    A few tests here exercise integration with rexgraph-rcdb through agent.rcdb. The
    package under test must never import those, which its own architecture test enforces,
    but a TEST may. The bare import is dropped first because the repository root shadows
    each as a namespace package, and an empty one imports fine and then fails on the first
    real attribute.
    """
    try:
        module = __import__(name)
        if getattr(module, "__file__", None):
            return
    except ImportError:
        pass
    sys.modules.pop(name, None)
    root = pathlib.Path(__file__).resolve().parents[2] / name
    if root.is_dir():
        sys.path.insert(0, str(root))


_ensure_sibling("rcdb")


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
