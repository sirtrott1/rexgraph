"""
Scale tests for rexgraph -- correctness and stability from 10 edges
to 500K edges.

Two graph generators:
    _random_graph     - dense random graph, nV ~ sqrt(2*nE)
    _sparse_graph     - sparse random graph, nV ~ nE/avg_degree

Tiers:
    tiny    -       10 edges: full pipeline including analysis
    small   -      100 edges: spectral, Hodge, persistence, query
    medium  -     1000 edges: Betti, Markov, perturbation, standard metrics
    large   -    10000 edges: construction, degree, standard metrics, subgraph
    xlarge  -   100000 edges: construction, boundary, standard metrics, serialization
    huge    -   500000 edges: construction, degree, adjacency, standard metrics

The xlarge and huge tiers use sparse graphs (avg degree ~10) so nV
is large enough that the tests are realistic but nV x nV dense
operations (L0, eigendecomposition) are skipped.

Skip slow tiers: pytest -m "not slow"
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph


# Graph generators

def _random_graph(nE, seed=42):
    """Dense random graph. nV ~ sqrt(2*nE), high average degree."""
    rng = np.random.RandomState(seed)
    nV = max(int(np.sqrt(2 * nE)), 4)
    src = rng.randint(0, nV, size=nE).astype(np.int32)
    tgt = rng.randint(0, nV, size=nE).astype(np.int32)
    mask = src == tgt
    tgt[mask] = (tgt[mask] + 1) % nV
    return RexGraph.from_graph(src, tgt)


def _sparse_graph(nE, avg_degree=10, seed=42):
    """Sparse random graph. nV ~ nE / avg_degree."""
    rng = np.random.RandomState(seed)
    nV = max(nE // avg_degree, 4)
    src = rng.randint(0, nV, size=nE).astype(np.int32)
    tgt = rng.randint(0, nV, size=nE).astype(np.int32)
    mask = src == tgt
    tgt[mask] = (tgt[mask] + 1) % nV
    return RexGraph.from_graph(src, tgt)


def _random_simplicial(nE, seed=42):
    """Random simplicial complex with triangles."""
    rng = np.random.RandomState(seed)
    nV = max(int(np.sqrt(2 * nE)), 6)
    src = rng.randint(0, nV, size=nE).astype(np.int32)
    tgt = rng.randint(0, nV, size=nE).astype(np.int32)
    mask = src == tgt
    tgt[mask] = (tgt[mask] + 1) % nV
    pairs = set()
    clean_src, clean_tgt = [], []
    for s, t in zip(src, tgt):
        key = (min(s, t), max(s, t))
        if key not in pairs and s != t:
            pairs.add(key)
            clean_src.append(min(s, t))
            clean_tgt.append(max(s, t))
    src = np.array(clean_src, dtype=np.int32)
    tgt = np.array(clean_tgt, dtype=np.int32)
    nE_actual = len(src)

    triangles = []
    for (a, b) in list(pairs)[:min(len(pairs), 500)]:
        for c in range(nV):
            if c <= b:
                continue
            if (min(a, c), max(a, c)) in pairs and (min(b, c), max(b, c)) in pairs:
                triangles.append([a, b, c])
                if len(triangles) >= nE_actual // 3:
                    break
        if len(triangles) >= nE_actual // 3:
            break

    if triangles:
        tri = np.array(triangles, dtype=np.int32)
        return RexGraph.from_simplicial(src, tgt, tri)
    return RexGraph.from_graph(src, tgt)


# Tiny: 10 edges, full pipeline

class TestTiny:

    @pytest.fixture
    def rex(self):
        return _random_graph(10, seed=1)

    @pytest.fixture
    def rex_simplicial(self):
        return _random_simplicial(10, seed=1)

    def test_construction(self, rex):
        assert rex.nE == 10
        assert rex.nV > 0

    def test_betti(self, rex):
        b0, b1, b2 = rex.betti
        assert b0 >= 1
        assert b1 >= 0
        assert b2 == 0

    def test_spectral(self, rex):
        evals = rex.eigenvalues_L0
        assert evals.shape[0] == rex.nV
        assert evals[0] < 1e-8

    def test_layout(self, rex):
        layout = rex.layout
        assert layout.shape == (rex.nV, 2)

    def test_hodge(self, rex):
        g = np.ones(rex.nE, dtype=np.float64)
        grad, curl, harm = rex.hodge(g)
        assert np.allclose(grad + curl + harm, g, atol=1e-10)

    def test_signal(self, rex):
        s = rex.signal(1, np.ones(rex.nE))
        assert s.shape == (rex.nE,)

    def test_markov(self, rex):
        g = np.zeros(rex.nE, dtype=np.float64)
        g[0] = 1.0
        result = rex.evolve_markov(g, dim=1, t=0.1)
        assert result.shape == (rex.nE,)

    def test_energy(self, rex):
        f = np.ones(rex.nE, dtype=np.float64)
        E_kin, E_pot, ratio = rex.energy_kin_pot(f)
        assert E_kin >= 0
        assert E_pot >= 0

    def test_analysis(self, rex):
        from rexgraph.analysis import analyze
        data = analyze(rex, run_perturbation=False)
        assert data["meta"]["nE"] == 10

    def test_serialization(self, rex):
        d = rex.to_dict()
        rex2 = RexGraph.from_dict(d)
        assert rex2.nE == rex.nE
        assert rex2.betti == rex.betti

    def test_simplicial_chain(self, rex_simplicial):
        if rex_simplicial.nF > 0:
            assert rex_simplicial.chain_valid


# Small: 100 edges

class TestSmall:

    @pytest.fixture
    def rex(self):
        return _random_graph(100, seed=2)

    @pytest.fixture
    def rex_simplicial(self):
        return _random_simplicial(100, seed=2)

    def test_construction(self, rex):
        assert rex.nE == 100

    def test_betti(self, rex):
        b0, b1, b2 = rex.betti
        assert b0 + b1 + b2 >= 1

    def test_hodge_identity(self, rex):
        g = np.random.RandomState(2).randn(rex.nE)
        grad, curl, harm = rex.hodge(g)
        assert np.allclose(grad + curl + harm, g, atol=1e-9)

    def test_coupling(self, rex):
        aG, aT = rex.coupling_constants
        assert isinstance(aG, float)
        assert isinstance(aT, float)

    def test_edge_types(self, rex):
        et = rex.edge_types
        assert et.shape == (100,)
        assert np.all(et >= 0)
        assert np.all(et <= 3)

    def test_persistence(self, rex):
        fv, fe, ff = rex.filtration(kind="dimension")
        result = rex.persistence(fv, fe, ff)
        assert "pairs" in result

    def test_standard_metrics(self, rex):
        from rexgraph.core import _standard
        adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle
        adj_wt = np.ones(adj_idx.shape[0], dtype=np.float64)
        sm = _standard.build_standard_metrics(
            adj_ptr, adj_idx, adj_edge, adj_wt, rex.nV, rex.nE)
        assert "pagerank" in sm
        assert sm["pagerank"].shape == (rex.nV,)

    def test_subgraph(self, rex):
        mask = np.zeros(rex.nE, dtype=bool)
        mask[:50] = True
        sub, v_map, e_map = rex.subgraph(mask)
        assert sub.nE == 50

    def test_delete_edges(self, rex):
        mask = np.zeros(rex.nE, dtype=np.int32)
        mask[0] = 1
        rex2 = rex.delete_edges(mask)
        assert rex2.nE == 99

    def test_wave_state(self, rex):
        psi = rex.wave_state(dim=1)
        assert psi.dtype == np.complex128
        probs = rex.born_probabilities(psi)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_simplicial_faces(self, rex_simplicial):
        if rex_simplicial.nF > 0:
            assert rex_simplicial.chain_valid
            b0, b1, b2 = rex_simplicial.betti
            euler = rex_simplicial.nV - rex_simplicial.nE + rex_simplicial.nF
            assert b0 - b1 + b2 == euler


# Medium: 1000 edges

class TestMedium:

    @pytest.fixture
    def rex(self):
        return _random_graph(1000, seed=3)

    def test_construction(self, rex):
        assert rex.nE == 1000
        assert rex.nV > 0

    def test_betti(self, rex):
        b0, b1, b2 = rex.betti
        euler = rex.nV - rex.nE
        assert b0 - b1 == euler

    def test_degree(self, rex):
        deg = rex.degree
        assert deg.shape == (rex.nV,)
        assert np.all(deg >= 1)

    def test_perturbation(self, rex):
        f_E = np.zeros(rex.nE, dtype=np.float64)
        f_E[0] = 1.0
        f_F = np.zeros(0, dtype=np.float64)
        times = np.linspace(0, 1.0, 10, dtype=np.float64)
        result = rex.analyze_perturbation(f_E, f_F, times=times)
        assert "trajectory" in result or "E_kin" in result

    def test_markov_diffusion(self, rex):
        g = np.zeros(rex.nE, dtype=np.float64)
        g[0] = 1.0
        result = rex.evolve_markov(g, dim=1, t=1.0)
        assert result.shape == (rex.nE,)
        assert np.max(result) < 1.0

    def test_normalize(self, rex):
        g = np.random.RandomState(3).randn(rex.nE)
        n = rex.normalize(g, "l2")
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10


# Large: 10000 edges (dense generator)

class TestLarge:

    @pytest.fixture
    def rex(self):
        return _random_graph(10000, seed=4)

    def test_construction(self, rex):
        assert rex.nE == 10000

    def test_degree(self, rex):
        deg = rex.degree
        assert deg.shape == (rex.nV,)

    def test_edge_types(self, rex):
        et = rex.edge_types
        assert et.shape == (10000,)

    def test_boundary_access(self, rex):
        bp = rex.boundary_ptr
        bi = rex.boundary_idx
        assert bp.shape == (10001,)
        assert bi.shape[0] == 20000

    def test_subgraph(self, rex):
        mask = np.zeros(rex.nE, dtype=bool)
        mask[:1000] = True
        sub, v_map, e_map = rex.subgraph(mask)
        assert sub.nE == 1000

    def test_to_dict_roundtrip(self, rex):
        d = rex.to_dict()
        rex2 = RexGraph.from_dict(d)
        assert rex2.nE == rex.nE
        assert rex2.nV == rex.nV


# Large sparse: 10000 edges, ~1000 vertices (avg degree 10)

class TestLargeSparse:

    @pytest.fixture
    def rex(self):
        return _sparse_graph(10000, avg_degree=10, seed=40)

    def test_construction(self, rex):
        assert rex.nE == 10000
        assert rex.nV >= 500

    def test_degree_distribution(self, rex):
        deg = rex.degree
        # Sparse graph: mean degree should be around 2*avg_degree
        # (each edge contributes to 2 vertex degrees)
        mean_deg = float(np.mean(deg))
        assert mean_deg < 100  # not a complete graph

    def test_standard_metrics(self, rex):
        from rexgraph.core import _standard
        adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle
        adj_wt = np.ones(adj_idx.shape[0], dtype=np.float64)
        sm = _standard.build_standard_metrics(
            adj_ptr, adj_idx, adj_edge, adj_wt, rex.nV, rex.nE)
        pr = sm["pagerank"]
        assert pr.shape == (rex.nV,)
        assert abs(float(np.sum(pr)) - 1.0) < 1e-4
        assert sm["n_communities"] >= 1

    def test_adjacency_structure(self, rex):
        adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle
        # Symmetric adjacency: each directed edge appears twice
        assert adj_idx.shape[0] == 2 * rex.nE
        # adj_ptr has nV+1 entries
        assert adj_ptr.shape[0] == rex.nV + 1

    def test_subgraph_sparse(self, rex):
        mask = np.zeros(rex.nE, dtype=bool)
        mask[:500] = True
        sub, v_map, e_map = rex.subgraph(mask)
        assert sub.nE == 500
        assert sub.nV <= rex.nV

    def test_sparse_betti(self, rex):
        """Betti via sparse rank arithmetic (no dense L1)."""
        from rexgraph.core._laplacians import _sparse_betti
        b0, b1, b2, r1, r2 = _sparse_betti(rex._B1_dual, None, rex.nV, rex.nE, 0)
        assert b0 >= 1
        assert b1 >= 0
        assert b2 == 0  # no faces
        euler = rex.nV - rex.nE
        assert b0 - b1 == euler

    def test_sparse_fiedler(self, rex):
        """Fiedler value via sparse eigsh."""
        from rexgraph.core._laplacians import _sparse_fiedler_L0
        fv, fvec, evals, evecs = _sparse_fiedler_L0(rex._B1_dual, rex.nV, rex.nE)
        assert isinstance(fv, float)
        assert fv >= 0
        assert fvec.shape == (rex.nV,)


# XLarge: 100000 edges (dense)

@pytest.mark.slow
class TestXLarge:

    @pytest.fixture
    def rex(self):
        return _random_graph(100000, seed=5)

    def test_construction(self, rex):
        assert rex.nE == 100000
        assert rex.nV > 0

    def test_boundary_access(self, rex):
        bp = rex.boundary_ptr
        assert bp[-1] == 200000

    def test_signal(self, rex):
        s = rex.signal(1, np.ones(rex.nE))
        assert s.shape == (100000,)

    def test_degree(self, rex):
        deg = rex.degree
        assert deg.shape == (rex.nV,)
        assert np.all(deg >= 1)

    def test_edge_types(self, rex):
        et = rex.edge_types
        assert et.shape == (100000,)

    def test_to_dict_roundtrip(self, rex):
        d = rex.to_dict()
        rex2 = RexGraph.from_dict(d)
        assert rex2.nE == 100000


# XLarge sparse: 100000 edges, ~10000 vertices

@pytest.mark.slow
class TestXLargeSparse:

    @pytest.fixture
    def rex(self):
        return _sparse_graph(100000, avg_degree=10, seed=50)

    def test_construction(self, rex):
        assert rex.nE == 100000
        assert rex.nV >= 5000

    def test_degree_distribution(self, rex):
        deg = rex.degree
        mean_deg = float(np.mean(deg))
        assert 5 < mean_deg < 100

    def test_standard_metrics(self, rex):
        from rexgraph.core import _standard
        adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle
        adj_wt = np.ones(adj_idx.shape[0], dtype=np.float64)
        sm = _standard.build_standard_metrics(
            adj_ptr, adj_idx, adj_edge, adj_wt, rex.nV, rex.nE)
        pr = sm["pagerank"]
        assert pr.shape == (rex.nV,)
        assert abs(float(np.sum(pr)) - 1.0) < 1e-3
        bc_v = sm["betweenness_v"]
        assert bc_v.shape == (rex.nV,)
        assert sm["n_communities"] >= 1

    def test_adjacency_symmetric(self, rex):
        adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle
        assert adj_idx.shape[0] == 2 * rex.nE

    def test_subgraph(self, rex):
        mask = np.zeros(rex.nE, dtype=bool)
        mask[:10000] = True
        sub, v_map, e_map = rex.subgraph(mask)
        assert sub.nE == 10000

    def test_delete_edges(self, rex):
        mask = np.zeros(rex.nE, dtype=np.int32)
        mask[:100] = 1
        rex2 = rex.delete_edges(mask)
        assert rex2.nE == rex.nE - 100

    def test_serialization(self, rex):
        d = rex.to_dict()
        rex2 = RexGraph.from_dict(d)
        assert rex2.nE == rex.nE
        assert rex2.nV == rex.nV

    def test_insert_edges(self, rex):
        new_src = np.array([0, 1, 2], dtype=np.int32)
        new_tgt = np.array([3, 4, 5], dtype=np.int32)
        rex2 = rex.insert_edges(new_src, new_tgt)
        assert rex2.nE == rex.nE + 3

    def test_sparse_betti(self, rex):
        """Betti via sparse rank arithmetic at 100K edges."""
        from rexgraph.core._laplacians import _sparse_betti
        b0, b1, b2, r1, r2 = _sparse_betti(rex._B1_dual, None, rex.nV, rex.nE, 0)
        assert b0 >= 1
        assert b1 >= 0
        euler = rex.nV - rex.nE
        assert b0 - b1 == euler

    def test_sparse_fiedler(self, rex):
        """Fiedler via sparse eigsh at 100K edges."""
        from rexgraph.core._laplacians import _sparse_fiedler_L0
        fv, fvec, evals, evecs = _sparse_fiedler_L0(rex._B1_dual, rex.nV, rex.nE)
        assert fv >= 0
        assert fvec.shape == (rex.nV,)
        # Should have a spectral gap (random graph is connected)
        assert fv > 0

    def test_sparse_bundle(self, rex):
        """Full sparse spectral bundle at 100K edges."""
        from rexgraph.core._laplacians import build_all_laplacians_sparse
        sb = build_all_laplacians_sparse(
            rex._B1_dual, rex._B2_hodge_dual,
            rex.nV, rex.nE, rex.nF)
        assert sb['_sparse_mode'] is True
        assert sb['beta0'] >= 1
        assert sb['beta1'] >= 0
        assert sb['fiedler_val_L0'] >= 0
        # Edge-space operators are None (too large for dense)
        assert sb['L1_full'] is None
        assert sb['RL'] is None

    def test_subgraph_dense_analysis(self, rex):
        """Extract small subgraph, run full dense spectral on it."""
        mask = np.zeros(rex.nE, dtype=bool)
        mask[:500] = True
        sub, v_map, e_map = rex.subgraph(mask)
        assert sub.nE == 500
        # The subgraph is small enough for full dense analysis
        b0, b1, b2 = sub.betti
        assert b0 >= 1
        euler = sub.nV - sub.nE + sub.nF
        assert b0 - b1 + b2 == euler


# Huge: 500000 edges (dense)

@pytest.mark.slow
class TestHuge:

    @pytest.fixture
    def rex(self):
        return _random_graph(500000, seed=6)

    def test_construction(self, rex):
        assert rex.nE == 500000

    def test_nV(self, rex):
        assert rex.nV > 0

    def test_boundary_ptr(self, rex):
        assert rex.boundary_ptr.shape == (500001,)

    def test_degree(self, rex):
        deg = rex.degree
        assert deg.shape == (rex.nV,)


# Huge sparse: 500000 edges, ~50000 vertices

@pytest.mark.slow
class TestHugeSparse:

    @pytest.fixture
    def rex(self):
        return _sparse_graph(500000, avg_degree=10, seed=60)

    def test_construction(self, rex):
        assert rex.nE == 500000
        assert rex.nV >= 20000

    def test_degree(self, rex):
        deg = rex.degree
        assert deg.shape == (rex.nV,)
        mean_deg = float(np.mean(deg))
        assert 5 < mean_deg < 100

    def test_boundary_access(self, rex):
        bp = rex.boundary_ptr
        bi = rex.boundary_idx
        assert bp.shape == (rex.nE + 1,)
        assert bi.shape[0] == 2 * rex.nE

    def test_edge_types(self, rex):
        et = rex.edge_types
        assert et.shape == (rex.nE,)
        assert np.all(et >= 0)

    def test_adjacency(self, rex):
        adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle
        assert adj_ptr.shape[0] == rex.nV + 1
        assert adj_idx.shape[0] == 2 * rex.nE

    def test_standard_metrics(self, rex):
        from rexgraph.core import _standard
        adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle
        adj_wt = np.ones(adj_idx.shape[0], dtype=np.float64)
        # Use approximate betweenness for speed (sample 50 sources)
        sm = _standard.build_standard_metrics(
            adj_ptr, adj_idx, adj_edge, adj_wt,
            rex.nV, rex.nE, btw_max_sources=50)
        pr = sm["pagerank"]
        assert pr.shape == (rex.nV,)
        assert abs(float(np.sum(pr)) - 1.0) < 1e-2
        assert sm["n_communities"] >= 1
        assert sm["modularity"] >= 0.0

    def test_subgraph(self, rex):
        mask = np.zeros(rex.nE, dtype=bool)
        mask[:10000] = True
        sub, v_map, e_map = rex.subgraph(mask)
        assert sub.nE == 10000

    def test_serialization_huge(self, rex):
        d = rex.to_dict()
        rex2 = RexGraph.from_dict(d)
        assert rex2.nE == rex.nE
        assert rex2.nV == rex.nV

    def test_sparse_betti_huge(self, rex):
        """Betti via sparse rank at 500K edges."""
        from rexgraph.core._laplacians import _sparse_betti
        b0, b1, b2, r1, r2 = _sparse_betti(rex._B1_dual, None, rex.nV, rex.nE, 0)
        assert b0 >= 1
        assert b1 >= 0
        euler = rex.nV - rex.nE
        assert b0 - b1 == euler

    def test_sparse_fiedler_huge(self, rex):
        """Fiedler via sparse eigsh at 500K edges."""
        from rexgraph.core._laplacians import _sparse_fiedler_L0
        fv, fvec, evals, evecs = _sparse_fiedler_L0(rex._B1_dual, rex.nV, rex.nE)
        assert fv >= 0
        assert fvec.shape == (rex.nV,)

    def test_sparse_bundle_huge(self, rex):
        """Full sparse spectral bundle at 500K edges."""
        from rexgraph.core._laplacians import build_all_laplacians_sparse
        sb = build_all_laplacians_sparse(
            rex._B1_dual, rex._B2_hodge_dual,
            rex.nV, rex.nE, rex.nF)
        assert sb['_sparse_mode'] is True
        assert sb['beta0'] >= 1
        assert sb['L1_full'] is None

    def test_subgraph_then_dense_huge(self, rex):
        """Extract 200-edge subgraph from 500K, verify full spectral."""
        mask = np.zeros(rex.nE, dtype=bool)
        mask[:200] = True
        sub, v_map, e_map = rex.subgraph(mask)
        assert sub.nE == 200
        b0, b1, b2 = sub.betti
        assert b0 >= 1
        euler = sub.nV - sub.nE + sub.nF
        assert b0 - b1 + b2 == euler

    def test_quotient_from_sparse_huge(self, rex):
        """Build quotient from sparse at 500K edges."""
        from rexgraph.core._quotient import build_quotient_from_sparse
        

        # Keep only 1000 edges in the quotient
        n_sub = rex.nE - 1000
        e_mask = np.zeros(rex.nE, dtype=np.uint8)
        e_mask[:n_sub] = 1

        src, tgt = rex._ensure_src_tgt()
        v_mask = np.zeros(rex.nV, dtype=np.uint8)
        for i in range(n_sub):
            v_mask[src[i]] = 1
            v_mask[tgt[i]] = 1

        f_mask = np.zeros(max(rex.nF, 0), dtype=np.uint8)

        info = build_quotient_from_sparse(
            rex._B1_dual, None,
            v_mask, e_mask, f_mask,
            rex.nV, rex.nE, rex.nF)

        nEq = info["dims"][1]
        assert nEq <= 1500  # ~1000 surviving edges
        assert info["chain_valid"]

        # Quotient is small: full dense spectral should be present
        if nEq > 0 and "spectral_bundle_quot" in info:
            sb = info["spectral_bundle_quot"]
            assert sb["beta0"] >= 1


# Cross-scale invariants

class TestInvariants:

    @pytest.mark.parametrize("nE", [10, 50, 200, 1000])
    def test_euler_relation(self, nE):
        rex = _random_graph(nE, seed=nE)
        b0, b1, b2 = rex.betti
        euler = rex.nV - rex.nE + rex.nF
        assert b0 - b1 + b2 == euler

    @pytest.mark.parametrize("nE", [10, 50, 200])
    def test_hodge_identity(self, nE):
        rex = _random_graph(nE, seed=nE)
        g = np.random.RandomState(nE).randn(rex.nE)
        grad, curl, harm = rex.hodge(g)
        assert np.allclose(grad + curl + harm, g, atol=1e-8)

    @pytest.mark.parametrize("nE", [10, 50, 200])
    def test_born_sum(self, nE):
        rex = _random_graph(nE, seed=nE)
        psi = rex.wave_state(dim=1)
        probs = rex.born_probabilities(psi)
        assert abs(probs.sum() - 1.0) < 1e-10

    @pytest.mark.parametrize("nE", [10, 50, 200, 1000])
    def test_serialization_roundtrip(self, nE):
        rex = _random_graph(nE, seed=nE)
        d = rex.to_dict()
        rex2 = RexGraph.from_dict(d)
        assert rex2.nE == rex.nE
        assert rex2.nV == rex.nV
        assert rex2.betti == rex.betti


# Sparse-specific invariants

class TestSparseInvariants:

    @pytest.mark.parametrize("nE,avg_deg", [(1000, 10), (5000, 8), (10000, 12)])
    def test_adjacency_symmetric(self, nE, avg_deg):
        rex = _sparse_graph(nE, avg_degree=avg_deg, seed=nE)
        adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle
        assert adj_idx.shape[0] == 2 * rex.nE

    @pytest.mark.parametrize("nE,avg_deg", [(1000, 10), (5000, 8)])
    def test_pagerank_sums_to_one(self, nE, avg_deg):
        from rexgraph.core import _standard
        rex = _sparse_graph(nE, avg_degree=avg_deg, seed=nE)
        adj_ptr, adj_idx, adj_edge = rex._adjacency_bundle
        adj_wt = np.ones(adj_idx.shape[0], dtype=np.float64)
        sm = _standard.build_standard_metrics(
            adj_ptr, adj_idx, adj_edge, adj_wt, rex.nV, rex.nE)
        assert abs(float(np.sum(sm["pagerank"])) - 1.0) < 1e-3

    @pytest.mark.parametrize("nE,avg_deg", [(1000, 10), (5000, 8)])
    def test_serialization_roundtrip(self, nE, avg_deg):
        rex = _sparse_graph(nE, avg_degree=avg_deg, seed=nE)
        d = rex.to_dict()
        rex2 = RexGraph.from_dict(d)
        assert rex2.nE == rex.nE
        assert rex2.nV == rex.nV

    @pytest.mark.parametrize("nE,avg_deg", [(1000, 10), (5000, 8), (10000, 12)])
    def test_sparse_betti_euler(self, nE, avg_deg):
        """Sparse Betti satisfies Euler relation at multiple scales."""
        from rexgraph.core._laplacians import _sparse_betti
        rex = _sparse_graph(nE, avg_degree=avg_deg, seed=nE)
        b0, b1, b2, r1, r2 = _sparse_betti(rex._B1_dual, None, rex.nV, rex.nE, 0)
        euler = rex.nV - rex.nE
        assert b0 - b1 == euler
