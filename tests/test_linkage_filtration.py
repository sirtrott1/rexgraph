"""
Tests for linkage complex construction and character-based
quotient filtration.
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.core import _fiber, _quotient


@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32),
    )


# linkage_complex

class TestLinkageComplex:

    def test_empty_at_high_threshold(self, k4):
        sfb = k4.fiber_similarity
        result = _fiber.linkage_complex(sfb, 2.0, k4.nV)
        assert result['n_edges'] == 0

    def test_complete_at_zero_threshold(self):
        sim = np.ones((4, 4), dtype=np.float64)
        np.fill_diagonal(sim, 0)
        result = _fiber.linkage_complex(sim, -0.1, 4)
        assert result['n_edges'] == 6  # C(4,2)

    def test_chain_condition(self, k4):
        sfb = k4.fiber_similarity
        result = _fiber.linkage_complex(sfb, 0.1, k4.nV)
        if result['nF'] > 0 and result['B1'] is not None and result['B2'] is not None:
            B1 = np.asarray(result['B1'], dtype=np.float64)
            B2 = np.asarray(result['B2'], dtype=np.float64)
            product = B1 @ B2
            assert np.max(np.abs(product)) < 1e-10

    def test_betti_euler(self, k4):
        sfb = k4.fiber_similarity
        result = _fiber.linkage_complex(sfb, 0.1, k4.nV)
        b0, b1, b2 = result['beta']
        euler = result['nV'] - result['n_edges'] + result['nF']
        assert b0 - b1 + b2 == euler

    def test_triangles_shape(self, k4):
        sfb = k4.fiber_similarity
        result = _fiber.linkage_complex(sfb, 0.1, k4.nV)
        tri = result['triangles']
        assert tri.shape[1] == 3
        assert tri.shape[0] == result['nF']

    def test_triangle_vertices_distinct(self, k4):
        sfb = k4.fiber_similarity
        result = _fiber.linkage_complex(sfb, 0.1, k4.nV)
        for row in result['triangles']:
            assert len(set(row)) == 3

    def test_graph_method(self, k4):
        rex = k4.linkage_complex(sfb_threshold=0.1)
        assert isinstance(rex, RexGraph)


# quotient_filtration_by_character

class TestQuotientFiltration:

    def test_shape(self, k4):
        n_steps = 10
        result = _quotient.quotient_filtration_by_character(
            k4.structural_character, 0, n_steps,
            k4.B1, k4.B2_hodge, k4.nV, k4.nE, k4.nF_hodge)
        assert result['thresholds'].shape == (n_steps,)
        assert result['beta0'].shape == (n_steps,)
        assert result['beta1'].shape == (n_steps,)
        assert result['beta2'].shape == (n_steps,)
        assert result['n_edges_remaining'].shape == (n_steps,)

    def test_edges_non_increasing(self, k4):
        result = _quotient.quotient_filtration_by_character(
            k4.structural_character, 0, 10,
            k4.B1, k4.B2_hodge, k4.nV, k4.nE, k4.nF_hodge)
        remaining = result['n_edges_remaining']
        assert np.all(np.diff(remaining) <= 0)

    def test_transition_valid(self, k4):
        result = _quotient.quotient_filtration_by_character(
            k4.structural_character, 0, 10,
            k4.B1, k4.B2_hodge, k4.nV, k4.nE, k4.nF_hodge)
        ti = result['transition_index']
        assert ti == -1 or (1 <= ti < 10)

    def test_order_matches_chi(self, k4):
        chi = k4.structural_character
        result = _quotient.quotient_filtration_by_character(
            chi, 0, 10, k4.B1, k4.B2_hodge, k4.nV, k4.nE, k4.nF_hodge)
        order = result['edges_removed_order']
        chi_vals = chi[order, 0]
        assert np.all(np.diff(chi_vals) <= 1e-12)

    def test_graph_method(self, k4):
        result = k4.quotient_filtration(channel=0, n_steps=5)
        assert 'beta1' in result
        assert 'transition_index' in result
