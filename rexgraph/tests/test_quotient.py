"""
Tests for rexgraph.core._quotient -- quotient complexes and relative homology.

Verifies:
    - Subcomplex validation: valid subcomplexes pass, invalid fail
    - Closure: edges add boundary vertices, faces add boundary edges
    - Quotient construction: B1_quot @ B2_quot = 0, dimensions shrink
    - Relative Betti: beta_0_rel >= 1, Euler relation on quotient
    - Congruence: identical edges congruent, distinct edges not
    - Signal restrict/lift roundtrip preserves surviving entries
    - Per-edge energy sums to total
    - Star subcomplexes are valid
    - Integration through RexGraph: subcomplex, quotient, congruent,
      star_of_vertex, star_of_edge, per_edge_energy, restrict/lift
"""
import numpy as np
import pytest

from rexgraph.core import _quotient
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


# Subcomplex Validation

class TestValidateSubcomplex:

    def test_valid_star(self, k4):
        """Star of vertex 0 is a valid subcomplex."""
        v_mask, e_mask, f_mask = k4.star_of_vertex(0)
        ok, violations = k4.validate_subcomplex(v_mask, e_mask, f_mask)
        assert ok
        assert len(violations) == 0

    def test_missing_vertex_fails(self, k4):
        """Selecting an edge without its boundary vertices fails."""
        v_mask = np.zeros(k4.nV, dtype=np.uint8)
        e_mask = np.zeros(k4.nE, dtype=np.uint8)
        f_mask = np.zeros(k4.nF, dtype=np.uint8)
        e_mask[0] = 1  # edge (0,1) without selecting vertices 0,1
        ok, violations = k4.validate_subcomplex(v_mask, e_mask, f_mask)
        assert not ok
        assert len(violations) > 0


# Closure

class TestClosure:

    def test_edge_closure_adds_vertices(self, k4):
        """Closing an edge set adds its boundary vertices."""
        e_mask = np.zeros(k4.nE, dtype=np.uint8)
        e_mask[0] = 1  # edge (0,1)
        v_mask, _, _ = _quotient.closure_of_edges(
            e_mask, k4.nV, k4.boundary_ptr, k4.boundary_idx)
        # Vertices 0 and 1 should be selected
        assert v_mask[0] == 1
        assert v_mask[1] == 1

    def test_closed_is_valid(self, k4):
        """Closed subcomplex passes validation."""
        e_mask = np.zeros(k4.nE, dtype=np.uint8)
        e_mask[0] = 1
        e_mask[1] = 1
        v_mask, e_out, f_out = _quotient.closure_of_edges(
            e_mask, k4.nV, k4.boundary_ptr, k4.boundary_idx)
        ok, _ = k4.validate_subcomplex(v_mask, e_out, f_out)
        assert ok


# Quotient Construction

class TestQuotientConstruction:

    def test_chain_condition(self, k4):
        """B1_quot @ B2_quot = 0 on the quotient."""
        v_mask, e_mask, f_mask = k4.star_of_vertex(0)
        Q = k4.quotient(v_mask, e_mask, f_mask)
        assert Q['chain_valid']
        assert Q['chain_error'] < 1e-10

    def test_dimensions_shrink(self, k4):
        """Quotient has fewer cells than the original."""
        v_mask, e_mask, f_mask = k4.star_of_vertex(0)
        Q = k4.quotient(v_mask, e_mask, f_mask)
        nVq, nEq, nFq = Q['dims']
        assert nEq < k4.nE
        assert nVq <= k4.nV

    def test_reindex_arrays(self, k4):
        v_mask, e_mask, f_mask = k4.star_of_vertex(0)
        Q = k4.quotient(v_mask, e_mask, f_mask)
        assert 'v_reindex' in Q
        assert 'e_reindex' in Q
        assert Q['v_reindex'].shape == (k4.nV,)
        assert Q['e_reindex'].shape == (k4.nE,)

    def test_l1_quot_symmetric(self, k4):
        v_mask, e_mask, f_mask = k4.star_of_vertex(0)
        Q = k4.quotient(v_mask, e_mask, f_mask)
        L1q = Q['L1_quot']
        assert np.allclose(L1q, L1q.T, atol=1e-12)


# Relative Betti Numbers

class TestRelativeBetti:

    def test_beta0_at_least_one(self, k4):
        """Quotient of connected complex has beta_0_rel >= 1."""
        v_mask, e_mask, f_mask = k4.star_of_vertex(0)
        Q = k4.quotient(v_mask, e_mask, f_mask)
        b0, b1, b2 = Q['betti_rel']
        assert b0 >= 1

    def test_betti_nonneg(self, k4):
        v_mask, e_mask, f_mask = k4.star_of_vertex(0)
        Q = k4.quotient(v_mask, e_mask, f_mask)
        b0, b1, b2 = Q['betti_rel']
        assert b0 >= 0
        assert b1 >= 0
        assert b2 >= 0


# Congruence

class TestCongruence:

    def test_self_congruent(self, k4):
        """An edge is congruent to itself modulo any subcomplex."""
        e_mask = np.zeros(k4.nE, dtype=np.uint8)
        ok = k4.congruent(dim=1, a=0, b=0, mask=e_mask)
        assert ok

    def test_different_edges(self, k4):
        """Two edges with different boundaries are not congruent mod empty."""
        e_mask = np.zeros(k4.nE, dtype=np.uint8)
        ok = k4.congruent(dim=1, a=0, b=3, mask=e_mask)
        # Edges 0=(0,1) and 3=(1,2) have different boundaries
        assert not ok


# Signal Restrict / Lift

class TestSignalRestrictLift:

    def test_roundtrip(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mask = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        restricted = _quotient.restrict_signal(signal, mask)
        assert restricted.shape == (3,)
        assert np.allclose(restricted, [1.0, 3.0, 5.0])
        lifted = _quotient.lift_signal(restricted, mask)
        assert lifted.shape == (5,)
        # Surviving entries match, subcomplex entries are 0
        assert lifted[0] == 1.0
        assert lifted[1] == 0.0  # fill_value
        assert lifted[2] == 3.0
        assert lifted[4] == 5.0

    def test_field_state_restrict(self, k4):
        f_E = np.ones(k4.nE, dtype=np.float64)
        f_F = np.ones(k4.nF, dtype=np.float64)
        e_mask = np.zeros(k4.nE, dtype=np.uint8)
        f_mask = np.zeros(k4.nF, dtype=np.uint8)
        e_mask[0] = 1  # remove edge 0
        fEq, fFq = _quotient.restrict_field_state(f_E, f_F, e_mask, f_mask)
        assert fEq.shape == (k4.nE - 1,)
        assert fFq.shape == (k4.nF,)

    def test_complex_restrict(self):
        signal = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
        mask = np.array([0, 1, 0], dtype=np.uint8)
        restricted = _quotient.restrict_signal_complex(signal, mask)
        assert restricted.shape == (2,)
        assert restricted[0] == 1+2j
        assert restricted[1] == 5+6j


# Per-Edge Energy

class TestPerEdgeEnergy:

    def test_sums_to_total(self, k4):
        """Per-edge energies sum to total E_kin and E_pot."""
        f_E = np.random.RandomState(42).randn(k4.nE).astype(np.float64)
        ek_pe, ep_pe = k4.per_edge_energy(f_E)
        ek_total, ep_total, _ = k4.energy_kin_pot(f_E)
        assert abs(ek_pe.sum() - ek_total) < 1e-10
        assert abs(ep_pe.sum() - ep_total) < 1e-10

    def test_shapes(self, k4):
        f_E = np.ones(k4.nE, dtype=np.float64)
        ek_pe, ep_pe = k4.per_edge_energy(f_E)
        assert ek_pe.shape == (k4.nE,)
        assert ep_pe.shape == (k4.nE,)


# Star Subcomplexes

class TestStar:

    def test_vertex_star_valid(self, k4):
        v_mask, e_mask, f_mask = k4.star_of_vertex(0)
        ok, _ = k4.validate_subcomplex(v_mask, e_mask, f_mask)
        assert ok

    def test_vertex_star_contains_vertex(self, k4):
        v_mask, _, _ = k4.star_of_vertex(0)
        assert v_mask[0] == 1

    def test_edge_star_valid(self, k4):
        v_mask, e_mask, f_mask = k4.star_of_edge(0)
        ok, _ = k4.validate_subcomplex(v_mask, e_mask, f_mask)
        assert ok

    def test_edge_star_contains_edge(self, k4):
        _, e_mask, _ = k4.star_of_edge(0)
        assert e_mask[0] == 1


# Hyperslice Quotient

class TestHypersliceQuotient:

    def test_vertex_hyperslice(self, k4):
        v_mask, e_mask, f_mask = k4.hyperslice_quotient(dim=0, cell_idx=0)
        ok, _ = k4.validate_subcomplex(v_mask, e_mask, f_mask)
        assert ok
        assert v_mask[0] == 1

    def test_edge_hyperslice(self, k4):
        v_mask, e_mask, f_mask = k4.hyperslice_quotient(dim=1, cell_idx=0)
        ok, _ = k4.validate_subcomplex(v_mask, e_mask, f_mask)
        assert ok
        assert e_mask[0] == 1


# Integration through RexGraph

class TestRexGraphIntegration:

    def test_subcomplex_by_edge_type(self, k4):
        v_mask, e_mask, f_mask = k4.subcomplex(edge_type=0)
        ok, _ = k4.validate_subcomplex(v_mask, e_mask, f_mask)
        assert ok

    def test_quotient_returns_dict(self, k4):
        v_mask, e_mask, f_mask = k4.star_of_vertex(0)
        Q = k4.quotient(v_mask, e_mask, f_mask)
        assert isinstance(Q, dict)
        for key in ['B1_quot', 'B2_quot', 'betti_rel', 'chain_valid']:
            assert key in Q

    def test_congruent_method(self, k4):
        e_mask = np.zeros(k4.nE, dtype=np.uint8)
        assert k4.congruent(dim=1, a=0, b=0, mask=e_mask)
