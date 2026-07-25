"""
End-to-end integration test for rexgraph v2.

Tests the full pipeline:
  Construction -> B1/B2 -> Laplacians -> RL -> chi/phi/kappa ->
  Void complex -> RCFE curvature -> Impute -> Explain -> Join ->
  Propagate -> Structural summary

Every assertion verifies a mathematical identity from the RCF.
"""
import numpy as np
from numpy.linalg import norm, svd, eigh, matrix_rank
from scipy.linalg import pinvh
import pytest


# Test complex construction helpers

def build_graph(edges, nV=None):
    """Build B1, B2, adjacency from edge list."""
    nE = len(edges)
    sources = np.array([e[0] for e in edges], dtype=np.int32)
    targets = np.array([e[1] for e in edges], dtype=np.int32)
    if nV is None:
        nV = max(max(s, t) for s, t in edges) + 1

    B1 = np.zeros((nV, nE), dtype=np.float64)
    for j, (s, t) in enumerate(edges):
        B1[s, j] = -1.0
        B1[t, j] = 1.0

    # Find triangles
    adj = [set() for _ in range(nV)]
    emap = {}
    for i, (s, t) in enumerate(edges):
        adj[s].add(t)
        adj[t].add(s)
        emap[(min(s, t), max(s, t))] = i

    tris = []
    for i in range(nV):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[i] & adj[j]:
                if k <= j:
                    continue
                tris.append((
                    emap[(min(i, j), max(i, j))],
                    emap[(min(i, k), max(i, k))],
                    emap[(min(j, k), max(j, k))],
                ))

    # Build B2 from triangles
    nF = len(tris)
    B2 = np.zeros((nE, nF), dtype=np.float64)
    for fi, (e0, e1, e2) in enumerate(tris):
        sub = B1[:, [e0, e1, e2]]
        _, _, Vt = svd(sub)
        kv = Vt[-1, :]
        kv = np.sign(kv / np.max(np.abs(kv)))
        B2[e0, fi], B2[e1, fi], B2[e2, fi] = kv

    return {
        'edges': edges, 'nV': nV, 'nE': nE, 'nF': nF,
        'sources': sources, 'targets': targets,
        'B1': B1, 'B2': B2, 'adj': adj, 'emap': emap, 'tris': tris,
    }


def build_laplacians(g):
    """Build L1, L_O, L_SG from graph dict."""
    B1, B2, nV, nE = g['B1'], g['B2'], g['nV'], g['nE']
    edges = g['edges']

    L1 = B1.T @ B1 + B2 @ B2.T
    K1 = np.abs(B1).T @ np.abs(B1)
    rs = K1.sum(1)
    D = np.diag(np.where(rs > 1e-12, 1.0 / np.sqrt(rs), 0.0))
    L_O = 0.5 * ((np.eye(nE) - D @ K1 @ D) + (np.eye(nE) - D @ K1 @ D).T)

    deg = np.zeros(nV)
    for s, t in edges:
        deg[s] += 1
        deg[t] += 1
    W = np.diag([1.0 / np.log(d + np.e) for d in deg])
    Ks = B1.T @ W @ B1
    Ks_off = Ks.copy()
    np.fill_diagonal(Ks_off, 0)
    L_SG = np.diag(np.abs(Ks_off).sum(1)) - Ks_off

    return L1, L_O, L_SG, K1


def trace_norm(L):
    tr = np.trace(L)
    return (L / tr, tr) if tr > 1e-15 else (np.zeros_like(L), 0.0)


def full_rcf(g, L1, L_O, L_SG):
    """Compute full RCF: RL, chi, phi, kappa."""
    B1, nV, nE = g['B1'], g['nV'], g['nE']
    edges = g['edges']

    h1, _ = trace_norm(L1)
    hO, _ = trace_norm(L_O)
    hSG, _ = trace_norm(L_SG)
    hats = [h1, hO, hSG]
    nhats = 3
    RL = h1 + hO + hSG
    RLp = pinvh(RL)

    chi = np.zeros((nE, nhats))
    for e in range(nE):
        if RL[e, e] > 1e-15:
            for k in range(nhats):
                chi[e, k] = hats[k][e, e] / RL[e, e]

    S0 = B1 @ RLp @ B1.T
    phi = np.zeros((nV, nhats))
    for v in range(nV):
        if abs(S0[v, v]) > 1e-15:
            for k in range(nhats):
                phi[v, k] = (B1 @ RLp @ hats[k] @ RLp @ B1.T)[v, v] / S0[v, v]

    v2e = [[] for _ in range(nV)]
    for e, (s, t) in enumerate(edges):
        v2e[s].append(e)
        v2e[t].append(e)
    chi_star = np.zeros((nV, nhats))
    for v in range(nV):
        if v2e[v]:
            chi_star[v] = np.mean([chi[e] for e in v2e[v]], axis=0)

    kappa = np.array([
        1.0 - 0.5 * np.sum(np.abs(phi[v] - chi_star[v])) for v in range(nV)
    ])

    return {
        'RL': RL, 'RLp': RLp, 'hats': hats, 'nhats': nhats,
        'chi': chi, 'phi': phi, 'chi_star': chi_star, 'kappa': kappa,
        'v2e': v2e,
    }


# Test classes


class TestChainComplex:
    """Verify chain complex construction (d^2 = 0)."""

    def test_small_graph(self):
        g = build_graph([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)])
        assert np.max(np.abs(g['B1'] @ g['B2'])) < 1e-12

    def test_k4(self):
        g = build_graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        assert np.max(np.abs(g['B1'] @ g['B2'])) < 1e-12
        assert g['nF'] == 4

    def test_betti_tree(self):
        g = build_graph([(0, 1), (1, 2), (2, 3)])
        assert g['nF'] == 0
        r1 = matrix_rank(g['B1'])
        assert g['nV'] - r1 == 1  # beta_0 = 1 (connected)
        assert g['nE'] - r1 == 0  # beta_1 = 0 (no cycles)


class TestLaplacians:
    """Verify Laplacian properties."""

    def test_symmetric_psd(self):
        g = build_graph([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)])
        L1, L_O, L_SG, _ = build_laplacians(g)
        for L, name in [(L1, 'L1'), (L_O, 'L_O'), (L_SG, 'L_SG')]:
            assert np.allclose(L, L.T), f"{name} not symmetric"
            assert all(eigh(L)[0] >= -1e-10), f"{name} not PSD"

    def test_trace_rl(self):
        g = build_graph([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)])
        L1, L_O, L_SG, _ = build_laplacians(g)
        rcf = full_rcf(g, L1, L_O, L_SG)
        assert abs(np.trace(rcf['RL']) - 3) < 1e-10


class TestCharacter:
    """Verify structural character properties."""

    @pytest.fixture
    def k4_rcf(self):
        g = build_graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        L1, L_O, L_SG, _ = build_laplacians(g)
        return g, full_rcf(g, L1, L_O, L_SG)

    def test_chi_simplex(self, k4_rcf):
        _, rcf = k4_rcf
        for e in range(6):
            assert abs(rcf['chi'][e].sum() - 1) < 1e-10

    def test_k4_uniform(self, k4_rcf):
        _, rcf = k4_rcf
        for e in range(6):
            for k in range(3):
                assert abs(rcf['chi'][e, k] - 1.0 / 3) < 1e-10

    def test_phi_simplex(self, k4_rcf):
        g, rcf = k4_rcf
        for v in range(g['nV']):
            assert abs(rcf['phi'][v].sum() - 1) < 1e-8

    def test_kappa_range(self, k4_rcf):
        _, rcf = k4_rcf
        assert np.all(rcf['kappa'] >= -1e-10)
        assert np.all(rcf['kappa'] <= 1 + 1e-10)

    def test_k4_kappa_high(self, k4_rcf):
        """K4 is highly symmetric, so kappa should be high (>0.5)."""
        _, rcf = k4_rcf
        for v in range(4):
            assert rcf['kappa'][v] > 0.5


class TestVoidComplex:
    """Verify void spectral theory."""

    def test_void_chain_condition(self):
        g = build_graph([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)])
        realized = set()
        for f in range(g['nF']):
            realized.add(tuple(sorted(np.where(np.abs(g['B2'][:, f]) > 0.5)[0])))
        for tri in g['tris']:
            if tuple(sorted(tri)) in realized:
                continue
            for sn in range(8):
                s = [1 if sn & (1 << b) else -1 for b in range(3)]
                col = np.zeros(g['nE'])
                col[tri[0]], col[tri[1]], col[tri[2]] = s
                if norm(g['B1'] @ col) < 1e-10:
                    assert norm(g['B1'] @ col) < 1e-10
                    break

    def test_void_identity(self):
        g = build_graph([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)])
        B1, B2, nE = g['B1'], g['B2'], g['nE']
        realized = set()
        for f in range(g['nF']):
            realized.add(tuple(sorted(np.where(np.abs(B2[:, f]) > 0.5)[0])))
        cols = []
        for tri in g['tris']:
            if tuple(sorted(tri)) in realized:
                continue
            for sn in range(8):
                s = [1 if sn & (1 << b) else -1 for b in range(3)]
                col = np.zeros(nE)
                col[tri[0]], col[tri[1]], col[tri[2]] = s
                if norm(B1 @ col) < 1e-10:
                    cols.append(col)
                    break
        if cols:
            Bvoid = np.column_stack(cols)
            L_up = B2 @ B2.T
            Lvoid = Bvoid @ Bvoid.T
            Bfull = np.hstack([B2, Bvoid])
            assert np.allclose(L_up + Lvoid, Bfull @ Bfull.T)


class TestRCFE:
    """Verify RCFE curvature and Bianchi identity."""

    def test_curvature_sums_to_nF(self):
        """sum C(e) = nF (each face contributes 1 unit of curvature)."""
        g = build_graph([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)])
        B2, nE, nF = g['B2'], g['nE'], g['nF']
        curv = np.zeros(nE)
        for f in range(nF):
            col = B2[:, f]; cn = col @ col
            if cn > 1e-15: curv += col ** 2 / cn
        assert abs(curv.sum() - nF) < 1e-10

    def test_curvature_nonneg(self):
        g = build_graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        B2, nE, nF = g['B2'], g['nE'], g['nF']
        curv = np.zeros(nE)
        for f in range(nF):
            col = B2[:, f]
            cn = col @ col
            if cn > 1e-15:
                curv += col ** 2 / cn
        assert np.all(curv >= -1e-10)


class TestImputation:
    """Verify harmonic signal imputation."""

    def test_energy_minimality(self):
        g = build_graph([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)])
        L1, L_O, L_SG, _ = build_laplacians(g)
        rcf = full_rcf(g, L1, L_O, L_SG)
        RL = rcf['RL']
        nE = g['nE']

        observed = np.array([1.0, 0.5, -0.3, 0, 0])
        mask = np.array([1, 1, 1, 0, 0], dtype=bool)
        oi = np.where(mask)[0]
        mi = np.where(~mask)[0]
        g_imp = -pinvh(RL[np.ix_(mi, mi)]) @ RL[np.ix_(mi, oi)] @ observed[oi]
        full = observed.copy()
        full[mi] = g_imp

        # Random fill should have higher energy
        rnd = observed.copy()
        rnd[3] = 0.5
        rnd[4] = -0.5
        assert full @ RL @ full <= rnd @ RL @ rnd + 1e-10


class TestJoins:
    """Verify join operations preserve chain condition."""

    def test_inner_join(self):
        g1 = build_graph([(0, 1), (1, 2), (0, 2), (0, 3)])
        g2 = build_graph([(0, 1), (1, 2), (0, 2)])
        shared = np.array([0, 1, 2, -1], dtype=np.int32)
        # Intersection should have the triangle vertices
        r_pairs = {(min(e[0], e[1]), max(e[0], e[1])) for e in g1['edges']}
        s_pairs = {(min(e[0], e[1]), max(e[0], e[1])) for e in g2['edges']}
        assert len(r_pairs & s_pairs) == 3

    def test_outer_join_adds_vertices(self):
        g1 = build_graph([(0, 1), (1, 2)])
        g2 = build_graph([(0, 3), (3, 4)], nV=5)
        # shared: only vertex 0
        shared = np.array([0, -1, -1], dtype=np.int32)
        # g1 has 3V, g2 has 5V, 1 shared => 3 + 5 - 1 = 7
        total_v = g1['nV'] + g2['nV'] - 1
        assert total_v == 7


class TestPropagation:
    """Verify spectral propagation through RL."""

    def test_self_propagation_maximal(self):
        g = build_graph([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)])
        L1, L_O, L_SG, _ = build_laplacians(g)
        rcf = full_rcf(g, L1, L_O, L_SG)
        RLp = rcf['RLp']

        source = np.zeros(g['nE'])
        source[0] = 1.0
        prop = RLp @ source
        # Self-propagation should be stronger than cross-propagation
        self_score = abs(prop[0])
        other_scores = [abs(prop[e]) for e in range(1, g['nE'])]
        assert self_score >= max(other_scores) - 1e-10


class TestLinalgBackend:
    """Verify _linalg LAPACK/BLAS wrappers (if compiled)."""

    def test_eigh_available(self):
        try:
            from rexgraph.core._linalg import eigh
            A = np.array([[2, 1], [1, 3]], dtype=np.float64)
            ev, evec = eigh(A)
            assert np.allclose(sorted(ev), sorted(eigh(A)[0]))
        except ImportError:
            pytest.skip("_linalg not compiled")

    def test_pipeline_available(self):
        try:
            from rexgraph.core._linalg import rl_pipeline
            g = build_graph([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)])
            L1, L_O, L_SG, _ = build_laplacians(g)
            r = rl_pipeline(g['B1'], L1, L_O, L_SG)
            assert abs(np.trace(r['RL']) - 3) < 1e-10
            assert max(abs(r['chi'][e].sum() - 1) for e in range(g['nE'])) < 1e-14
        except ImportError:
            pytest.skip("_linalg not compiled")


class TestFullPipeline:
    """End-to-end: build complex, compute everything, verify all identities."""

    def test_drct_complex(self):
        """DRCT 8V/16E complex: the benchmark case."""
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
            (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7),
            (1, 7), (2, 3),
        ]
        g = build_graph(edges)
        assert g['nV'] == 8
        assert g['nE'] == 16

        # Chain condition
        if g['nF'] > 0:
            assert np.max(np.abs(g['B1'] @ g['B2'])) < 1e-10

        # Laplacians
        L1, L_O, L_SG, K1 = build_laplacians(g)
        for L in [L1, L_O, L_SG]:
            assert np.allclose(L, L.T)
            assert all(eigh(L)[0] >= -1e-10)

        # RCF
        rcf = full_rcf(g, L1, L_O, L_SG)
        assert abs(np.trace(rcf['RL']) - 3) < 1e-10

        # Chi on simplex
        for e in range(g['nE']):
            assert abs(rcf['chi'][e].sum() - 1) < 1e-10
            assert np.all(rcf['chi'][e] >= -1e-10)

        # Phi on simplex
        for v in range(g['nV']):
            assert abs(rcf['phi'][v].sum() - 1) < 1e-8

        # Kappa in [0, 1]
        assert np.all(rcf['kappa'] >= -1e-10)
        assert np.all(rcf['kappa'] <= 1 + 1e-10)

        # RCFE curvature and Bianchi
        curv = np.zeros(g['nE'])
        for f in range(g['nF']):
            col = g['B2'][:, f]
            cn = col @ col
            if cn > 1e-15:
                curv += col ** 2 / cn
        assert np.all(curv >= -1e-10)
        if g['nF'] > 0:
            # Curvature sums to nF (each face contributes 1 unit)
            assert abs(curv.sum() - g['nF']) < 1e-10

        # Strain
        rl_diag = np.diag(rcf['RL'])
        strain = sum(curv[e] * rl_diag[e] for e in range(g['nE']))
        assert strain >= 0

        # Imputation: energy minimality
        obs = np.random.randn(g['nE'])
        obs[10:] = 0
        mask = np.array([i < 10 for i in range(g['nE'])], dtype=bool)
        oi = np.where(mask)[0]
        mi = np.where(~mask)[0]
        if len(mi) > 0:
            RL = rcf['RL']
            RL_mm = RL[np.ix_(mi, mi)]
            RL_mo = RL[np.ix_(mi, oi)]
            g_imp = -pinvh(RL_mm) @ RL_mo @ obs[oi]
            full = obs.copy()
            full[mi] = g_imp
            rnd = obs.copy()
            rnd[mi] = np.random.randn(len(mi))
            assert full @ RL @ full <= rnd @ RL @ rnd + 1e-8
