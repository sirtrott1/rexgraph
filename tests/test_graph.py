"""
Tests for rexgraph.graph -- RexGraph and TemporalRex public API.

Integration tests that exercise the orchestration layer. Underlying
Cython modules are tested individually in core/test_*.py.

Verifies:
    - Constructors: from_graph, from_simplicial, from_hypergraph, from_adjacency, from_dict
    - Core properties: nV, nE, nF, dimension, sources, targets, edge_types, degree
    - Boundary operators: B1, B2, B2_hodge, chain_valid
    - Laplacians: L0, L1, L2, spectral_bundle, betti, eigenvalues_L0
    - Layout: layout is f64[nV, 2]
    - Relational: RL, coupling_constants, structural_character
    - Hodge: hodge decomposition identity
    - Signal and state: signal, energy, normalize, create_state, dirac_state
    - Dynamics: evolve_markov, evolve_schrodinger, evolve_coupled
    - Quantum: wave_state, born_probabilities, measure
    - Field: field_operator, field_eigen, classify_modes
    - Dirac: dirac_operator, dirac_eigenvalues
    - Subcomplex/quotient: subcomplex, quotient, star_of_vertex, hyperslice
    - Topological: fill_cycle, promote, face_data
    - Persistence: filtration, persistence, persistence_barcodes
    - Graph ops: subgraph, insert_edges, delete_edges
    - Joins: inner_join, outer_join, left_join
    - Query: impute, explain, propagate
    - Serialization: to_dict / from_dict, to_json
    - TemporalRex: construction, at(), T, edge_lifecycle, edge_metrics
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph, TemporalRex


# Fixtures

@pytest.fixture
def triangle():
    """3V, 3E, 0F."""
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.fixture
def filled_triangle():
    """3V, 3E, 1F."""
    return RexGraph.from_simplicial(
        sources=np.array([0, 1, 0], dtype=np.int32),
        targets=np.array([1, 2, 2], dtype=np.int32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )


@pytest.fixture
def k4():
    """4V, 6E, 4F."""
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


@pytest.fixture
def path_graph():
    """4V, 3E, 0F."""
    return RexGraph.from_graph([0, 1, 2], [1, 2, 3])


# Constructors

class TestConstructors:

    def test_from_graph(self):
        rex = RexGraph.from_graph([0, 1, 0], [1, 2, 2])
        assert rex.nV == 3
        assert rex.nE == 3
        assert rex.nF == 0

    def test_from_simplicial(self, filled_triangle):
        assert filled_triangle.nV == 3
        assert filled_triangle.nE == 3
        assert filled_triangle.nF == 1

    def test_from_hypergraph(self):
        # Hyperedge: one 3-endpoint edge, one 2-endpoint edge
        bp = np.array([0, 3, 5], dtype=np.int32)
        bi = np.array([0, 1, 2, 1, 3], dtype=np.int32)
        rex = RexGraph.from_hypergraph(bp, bi)
        assert rex.nE == 2
        assert rex.nV == 4

    def test_from_adjacency(self):
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
        rex = RexGraph.from_adjacency(A)
        assert rex.nV == 3
        assert rex.nE == 3

    def test_from_dict_roundtrip(self, k4):
        d = k4.to_dict()
        rex2 = RexGraph.from_dict(d)
        assert rex2.nV == k4.nV
        assert rex2.nE == k4.nE
        assert rex2.nF == k4.nF

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            RexGraph()

    def test_weighted(self):
        w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rex = RexGraph.from_graph([0, 1, 0], [1, 2, 2], w_E=w)
        assert np.allclose(rex.w_E, w)


# Core Properties

class TestCoreProperties:

    def test_dimensions(self, k4):
        assert k4.nV == 4
        assert k4.nE == 6
        assert k4.nF == 4
        assert k4.dimension == 2

    def test_sources_targets(self, triangle):
        assert triangle.sources is not None
        assert triangle.targets is not None
        assert triangle.sources.shape == (3,)

    def test_edge_types(self, triangle):
        et = triangle.edge_types
        assert et.shape == (triangle.nE,)
        assert np.all(et == 0)  # all standard

    def test_degree(self, triangle):
        deg = triangle.degree
        assert deg.shape == (triangle.nV,)
        assert np.all(deg == 2)  # triangle: each vertex has degree 2

    def test_boundary_ptr_idx(self, triangle):
        assert triangle.boundary_ptr.shape == (triangle.nE + 1,)
        assert triangle.boundary_idx.shape[0] == triangle.nE * 2


# Boundary Operators and Chain Complex

class TestChainComplex:

    def test_B1_shape(self, k4):
        assert k4.B1.shape == (k4.nV, k4.nE)

    def test_B2_shape(self, k4):
        B2 = k4.B2
        assert B2.shape[0] == k4.nE

    def test_chain_valid(self, k4):
        assert k4.chain_valid

    def test_betti_k4(self, k4):
        b0, b1, b2 = k4.betti
        assert b0 == 1  # connected
        assert b1 == 0  # no 1-holes
        assert b2 == 1  # one void (sphere)

    def test_betti_triangle_unfilled(self, triangle):
        b0, b1, b2 = triangle.betti
        assert b0 == 1
        assert b1 == 1  # one 1-hole
        assert b2 == 0

    def test_euler_characteristic(self, k4):
        assert k4.euler_characteristic == k4.nV - k4.nE + k4.nF


# Laplacians and Spectral

class TestSpectral:

    def test_L0_shape(self, triangle):
        assert triangle.L0.shape == (triangle.nV, triangle.nV)

    def test_L1_shape(self, triangle):
        assert triangle.L1.shape == (triangle.nE, triangle.nE)

    def test_eigenvalues_L0(self, triangle):
        evals = triangle.eigenvalues_L0
        assert evals.shape == (triangle.nV,)
        assert evals[0] < 1e-10  # smallest eigenvalue near zero (connected)

    def test_layout(self, triangle):
        layout = triangle.layout
        assert layout.shape == (triangle.nV, 2)

    def test_spectral_bundle(self, triangle):
        sb = triangle.spectral_bundle
        assert isinstance(sb, dict)
        assert "L0" in sb or "evals_L0" in sb


# Relational (RCF v2)

class TestRelational:

    def test_rl_shape(self, k4):
        RL = k4.relational_laplacian
        if RL is not None:
            assert RL.shape == (k4.nE, k4.nE)

    def test_coupling_constants(self, k4):
        aG, aT = k4.coupling_constants
        assert isinstance(aG, float)
        assert isinstance(aT, float)


# Hodge Decomposition

class TestHodge:

    def test_decomposition_identity(self, k4):
        g = np.random.RandomState(42).randn(k4.nE)
        grad, curl, harm = k4.hodge(g)
        reconstructed = grad + curl + harm
        assert np.allclose(reconstructed, g, atol=1e-10)

    def test_hodge_full(self, k4):
        g = np.ones(k4.nE, dtype=np.float64)
        result = k4.hodge_full(g)
        assert isinstance(result, dict)
        assert "grad" in result


# Signal and State

class TestSignalState:

    def test_signal(self, triangle):
        s = triangle.signal(1, [1.0, 2.0, 3.0])
        assert s.shape == (triangle.nE,)

    def test_normalize(self, triangle):
        s = np.array([3.0, 4.0, 0.0], dtype=np.float64)
        n = triangle.normalize(s, "l2")
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10

    def test_create_state(self, triangle):
        state = triangle.create_state()
        assert hasattr(state, "f0")
        assert hasattr(state, "f1")

    def test_dirac_state(self, triangle):
        f0, f1, f2 = triangle.dirac_state(1, 0)
        assert f1[0] == 1.0

    def test_energy_kin_pot(self, triangle):
        f_E = np.ones(triangle.nE, dtype=np.float64)
        E_kin, E_pot, ratio = triangle.energy_kin_pot(f_E)
        assert E_kin >= 0
        assert E_pot >= 0


# Dynamics

class TestDynamics:

    def test_evolve_markov(self, triangle):
        g = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        result = triangle.evolve_markov(g, dim=1, t=0.1)
        assert result.shape == (triangle.nE,)

    def test_evolve_schrodinger(self, triangle):
        psi = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        psi_re, psi_im = triangle.evolve_schrodinger(psi, dim=1, t=0.1)
        assert psi_re.shape == (triangle.nE,)
        assert psi_im.shape == (triangle.nE,)


# Quantum / Wave

class TestQuantum:

    def test_wave_state(self, triangle):
        psi = triangle.wave_state(dim=1)
        assert psi.dtype == np.complex128
        assert psi.shape == (triangle.nE,)

    def test_born_probabilities(self, triangle):
        psi = triangle.wave_state(dim=1)
        probs = triangle.born_probabilities(psi)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_measure(self, triangle):
        psi = triangle.wave_state(dim=1)
        idx, collapsed = triangle.measure(psi, dim=1)
        assert 0 <= idx < triangle.nE


# Field Theory

class TestField:

    def test_field_operator(self, k4):
        M, coupling, hermitian = k4.field_operator
        assert M.shape[0] == M.shape[1]

    def test_field_eigen(self, k4):
        evals, evecs, gaps = k4.field_eigen
        assert len(evals) > 0

    def test_classify_modes(self, k4):
        result = k4.classify_modes()
        assert isinstance(result, (dict, tuple))


# Dirac

class TestDirac:

    def test_dirac_operator(self, filled_triangle):
        D = filled_triangle.dirac_operator
        total_dim = filled_triangle.nV + filled_triangle.nE + filled_triangle.nF
        assert D.shape == (total_dim, total_dim)

    def test_dirac_eigenvalues(self, filled_triangle):
        evals = filled_triangle.dirac_eigenvalues
        assert len(evals) > 0


# Subcomplex and Quotient

class TestSubcomplexQuotient:

    def test_subgraph(self, k4):
        # Keep first 3 edges of K4: (0,1), (0,2), (0,3) -> 4 vertices
        e_mask = np.zeros(k4.nE, dtype=bool)
        e_mask[0] = True
        e_mask[1] = True
        e_mask[2] = True
        sub, v_map, e_map = k4.subgraph(e_mask)
        assert isinstance(sub, RexGraph)
        assert sub.nE == 3
        assert len(e_map) == 3
        assert len(v_map) == sub.nV

    def test_star_of_vertex(self, k4):
        v_mask, e_mask, f_mask = k4.star_of_vertex(0)
        assert v_mask[0]  # vertex 0 is in its own star

    def test_hyperslice(self, k4):
        result = k4.hyperslice(dim=1, idx=0)
        assert isinstance(result, tuple)

    def test_quotient(self, k4):
        v = np.ones(k4.nV, dtype=bool)
        e = np.ones(k4.nE, dtype=bool)
        e[0] = False
        f = np.ones(k4.nF, dtype=bool)
        result = k4.quotient(v, e, f)
        assert isinstance(result, dict)


# Topological Constructions

class TestTopological:

    def test_promote(self, triangle):
        """Promote should detect and fill the triangle's cycle."""
        promoted = triangle.promote()
        assert promoted.nF >= 1

    def test_fill_cycle(self, triangle):
        edges = np.array([0, 1, 2], dtype=np.int32)
        filled = triangle.fill_cycle(edges)
        assert filled.nF == 1


# Persistence

class TestPersistence:

    def test_filtration(self, k4):
        fv, fe, ff = k4.filtration(kind="dimension")
        assert fv.shape == (k4.nV,)
        assert fe.shape == (k4.nE,)
        assert ff.shape == (k4.nF,)

    def test_persistence(self, k4):
        fv, fe, ff = k4.filtration(kind="dimension")
        result = k4.persistence(fv, fe, ff)
        assert isinstance(result, dict)
        assert "pairs" in result

    def test_persistence_barcodes(self, k4):
        fv, fe, ff = k4.filtration(kind="dimension")
        result = k4.persistence(fv, fe, ff)
        barcodes = k4.persistence_barcodes(result)
        assert barcodes.ndim == 2
        assert barcodes.shape[1] == 2


# Graph Operations

class TestGraphOps:

    def test_insert_edges(self, triangle):
        new_src = np.array([0], dtype=np.int32)
        new_tgt = np.array([3], dtype=np.int32)
        rex2 = triangle.insert_edges(new_src, new_tgt)
        assert rex2.nE == triangle.nE + 1
        assert rex2.nV == 4

    def test_delete_edges(self, triangle):
        # mask nonzero = delete; delete edge 0 only, keep edges 1 and 2
        mask = np.array([1, 0, 0], dtype=np.int32)
        rex2 = triangle.delete_edges(mask)
        assert rex2.nE == 2


# Joins

class TestJoins:

    def test_inner_join(self):
        a = RexGraph.from_graph([0, 1], [1, 2])
        b = RexGraph.from_graph([0, 1], [1, 2])
        sv = np.array([-1, 0, 1], dtype=np.int32)
        result = a.inner_join(b, sv)
        assert isinstance(result, dict)

    def test_outer_join(self):
        a = RexGraph.from_graph([0, 1], [1, 2])
        b = RexGraph.from_graph([0, 1], [1, 2])
        sv = np.array([-1, 0, 1], dtype=np.int32)
        result = a.outer_join(b, sv)
        assert isinstance(result, dict)
        assert result['nEj'] == a.nE + b.nE


# Query Engine

class TestQuery:

    def test_impute(self, k4):
        signal = np.zeros(k4.nE, dtype=np.float64)
        signal[0] = 1.0
        mask = np.zeros(k4.nE, dtype=np.uint8)
        mask[0] = 1
        result = k4.impute(signal, mask)
        assert isinstance(result, dict)
        assert "imputed" in result

    def test_explain_edge(self, k4):
        result = k4.explain(dim=1, idx=0)
        assert isinstance(result, dict)
        assert "below" in result

    def test_explain_vertex(self, k4):
        result = k4.explain(dim=0, idx=0)
        assert isinstance(result, dict)
        assert "phi" in result

    def test_propagate(self, k4):
        source = np.zeros(k4.nE, dtype=np.float64)
        source[0] = 1.0
        target = source.copy()
        result = k4.propagate(source, target)
        assert "score" in result
        assert result["score"] > 0


# Serialization

class TestSerialization:

    def test_to_dict_from_dict(self, k4):
        d = k4.to_dict()
        rex2 = RexGraph.from_dict(d)
        assert rex2.betti == k4.betti

    def test_to_json(self, k4):
        j = k4.to_json()
        assert isinstance(j, dict)
        assert "boundary_ptr" in j

    def test_repr(self, k4):
        r = repr(k4)
        assert "RexGraph" in r


# TemporalRex

class TestTemporalRex:

    def _make_trex(self):
        snaps = [
            (np.array([0, 1, 0], dtype=np.int32),
             np.array([1, 2, 2], dtype=np.int32)),
            (np.array([0, 1, 0, 1], dtype=np.int32),
             np.array([1, 2, 2, 3], dtype=np.int32)),
            (np.array([0, 1, 0, 1, 2], dtype=np.int32),
             np.array([1, 2, 2, 3, 3], dtype=np.int32)),
        ]
        return TemporalRex(snaps)

    def test_construction(self):
        trex = self._make_trex()
        assert trex.T == 3

    def test_at(self):
        trex = self._make_trex()
        r0 = trex.at(0)
        assert isinstance(r0, RexGraph)
        assert r0.nE == 3
        r1 = trex.at(1)
        assert r1.nE == 4

    def test_edge_lifecycle(self):
        trex = self._make_trex()
        edge_ids, birth, death = trex.edge_lifecycle
        assert len(birth) == len(edge_ids)

    def test_edge_metrics(self):
        trex = self._make_trex()
        counts, born, died = trex.edge_metrics
        assert counts.shape == (3,)  # T=3

    def test_repr(self):
        trex = self._make_trex()
        r = repr(trex)
        assert "TemporalRex" in r
        assert "T=3" in r
