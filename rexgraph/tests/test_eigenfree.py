"""Eigen-free / sparse kernel correctness - the layer the sparse-math refactor added.

Each test pins an in-repo eigen-free result against its dense oracle, so the
matrix-free path cannot silently drift from the exact linear algebra it replaces.
These are the in-repo home of the numeric checks the math verification scripts run.
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph


def _rex(edges):
    s = np.array([e[0] for e in edges], dtype=np.int32)
    t = np.array([e[1] for e in edges], dtype=np.int32)
    return RexGraph.from_graph(s, t)


# harmonic content present (independent cycles), no faces
_COMPLEXES = [
    [(0, 1), (1, 2), (2, 3), (0, 3), (1, 4), (4, 5), (2, 5)],   # two 4-cycles sharing an edge
    [(0, 1), (1, 2), (0, 2)],                                    # bare triangle
]


@pytest.mark.parametrize("edges", _COMPLEXES)
def test_greens_diagonal_eigenfree_matches_dense_pinv(edges):
    """diag(RL4^-1) via block-CG equals diag(pinv(RL4)) to machine precision (script 11)."""
    rex = _rex(edges)
    got = rex.greens_diagonal_eigenfree
    want = np.diag(np.linalg.pinv(np.asarray(rex.RL, dtype=float)))
    assert got.shape == (rex.nE,)
    np.testing.assert_allclose(got, want, atol=1e-9)


@pytest.mark.parametrize("edges", _COMPLEXES)
def test_energy_character_is_rl4_row_energy(edges):
    """Per-edge energy character equals diag(RL4^2) = row energies of RL4 (script 14)."""
    rex = _rex(edges)
    RL = np.asarray(rex.RL, dtype=float)
    np.testing.assert_allclose(rex.energy_character, (RL @ RL).diagonal(), atol=1e-9)


def _Kk(k):
    import itertools
    edges = list(itertools.combinations(range(k), 2))
    tris = list(itertools.combinations(range(k), 3))
    s = np.array([e[0] for e in edges], np.int32); t = np.array([e[1] for e in edges], np.int32)
    return RexGraph.from_simplicial(s, t, np.array(tris, np.int32))


@pytest.mark.parametrize("weights", ["unit", "random", "extreme"])
def test_attributed_curvature_matches_from_scratch_weighted_boundary(weights):
    """The sparse attributed curvature equals the definition computed from scratch:
    R = B1^w @ B2^w with B1^w[v,e]=a_v B1[v,e]sqrt(w_e), B2^w[e,f]=sqrt(w_e)B2[e,f], and
    kappa_f=||R[:,f]|| - across unit, random, and extreme weights (script 14/Def 3.1-3.2)."""
    rng = np.random.default_rng(1)
    rex = _Kk(5)
    nV, nE = rex._nV, rex._nE
    if weights == "unit":
        w_e, a_v = np.ones(nE), np.ones(nV)
    elif weights == "random":
        w_e, a_v = rng.random(nE) + 0.05, rng.random(nV) + 0.05
    else:
        w_e, a_v = np.r_[1e3, rng.random(nE - 1) * 1e-3], rng.random(nV) + 0.05
    got = rex.attributed_curvature(w_e, a_v)
    sqw = np.sqrt(np.maximum(w_e, 0.0))
    R_true = (a_v[:, None] * rex.B1 * sqw[None, :]) @ (sqw[:, None] * rex.B2_hodge)
    np.testing.assert_allclose(got['R'], R_true, atol=1e-10)
    np.testing.assert_allclose(got['kappa_f'], np.linalg.norm(R_true, axis=0), atol=1e-10)


def test_trace_moments_share_one_power_walk_matches_per_order():
    """The moment engine [tr(X),..,tr(X^a)] from one incremental power walk equals per-order
    trace_power, and the Renyi curve reads off it identically (scripts 16/18/19)."""
    import scipy.sparse as sp

    from rexgraph import scale_propagator as spg
    rng = np.random.default_rng(0)
    A = sp.random(60, 60, density=0.08, random_state=0); A = (A + A.T).tocsr()
    tm = spg.trace_moments(A, 5)
    for a in (1, 2, 3, 4, 5):
        assert abs(tm[a - 1] - spg.trace_power(A, a)) <= 1e-9 * (abs(spg.trace_power(A, a)) + 1)
        assert abs(spg.renyi_from_moments(tm, a) - spg.renyi_entropy(A, a)) <= 1e-9 if a > 1 else True


# -- eigen-free heat propagation of signals (Chebyshev matvec) vs dense e^{-tL} ----

def _dense_heat(L, f, t):
    w, V = np.linalg.eigh(np.asarray(L, dtype=float))
    return V @ (np.exp(-t * w) * (V.T @ f))


@pytest.mark.parametrize("edges", _COMPLEXES)
@pytest.mark.parametrize("t", [0.1, 0.5, 2.0])
def test_heat_apply_matches_dense_expm(edges, t):
    """e^{-tL1} f via Chebyshev sparse mat-vecs (no eigendecomposition) equals the
    dense eigendecomposition apply to Chebyshev tolerance, for any t."""
    from rexgraph import scale_propagator as spg
    rex = _rex(edges)
    L = rex.L1_sparse.tocsr()
    Ld = np.asarray(L.todense())
    rng = np.random.default_rng(0)
    f = rng.standard_normal(L.shape[0])
    got = spg.heat_apply(L, f, t)
    want = _dense_heat(Ld, f, t)
    np.testing.assert_allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("edges", _COMPLEXES)
def test_heat_trajectory_shares_matvecs_and_matches_dense(edges):
    """The trajectory (one shared set of Chebyshev vectors, then per-t coefficient
    combos) equals the dense apply at every timestep."""
    from rexgraph import scale_propagator as spg
    rex = _rex(edges)
    L = rex.L1_sparse.tocsr()
    Ld = np.asarray(L.todense())
    rng = np.random.default_rng(1)
    f = rng.standard_normal(L.shape[0])
    times = np.array([0.05, 0.25, 1.0, 3.0])
    traj = spg.heat_trajectory(L, f, times)
    want = np.stack([_dense_heat(Ld, f, t) for t in times])
    assert traj.shape == (len(times), L.shape[0])
    np.testing.assert_allclose(traj, want, atol=1e-10)


@pytest.mark.parametrize("edges", _COMPLEXES)
def test_heat_apply_block_matches_columnwise(edges):
    """A block of signals (n, m) propagates as an spmm and equals column-by-column
    single-signal propagation - the shape the multi-core/GPU path batches over."""
    from rexgraph import scale_propagator as spg
    rex = _rex(edges)
    L = rex.L1_sparse.tocsr()
    rng = np.random.default_rng(2)
    F = rng.standard_normal((L.shape[0], 5))
    blk = spg.heat_apply(L, F, 0.7)
    cols = np.stack([spg.heat_apply(L, F[:, j], 0.7) for j in range(5)], axis=1)
    np.testing.assert_allclose(blk, cols, atol=1e-12)


def test_heat_t0_is_identity():
    """e^{-0*L} f = f exactly."""
    from rexgraph import scale_propagator as spg
    rex = _rex(_COMPLEXES[0])
    L = rex.L1_sparse.tocsr()
    f = np.arange(L.shape[0], dtype=float)
    np.testing.assert_allclose(spg.heat_apply(L, f, 0.0), f, atol=1e-12)


# -- eigen-free Betti (union-find + exact rational rank) == spectrum-derived Betti --

@pytest.mark.parametrize("edges", _COMPLEXES)
def test_betti_eigenfree_matches_spectral_bundle(edges):
    """rex.betti (beta_0 union-find, rank(B_k) exact rational column reduction - no
    SVD, no eigendecomposition) equals the spectral bundle's spectrum-derived betti."""
    rex = _rex(edges)
    sb = rex.spectral_bundle
    spectral = (int(sb['beta0']), int(sb['beta1']), int(sb['beta2']))
    assert rex.betti == spectral


def test_betti_eigenfree_on_faced_and_graded():
    """Eigen-free betti on complexes WITH faces / higher grades matches the exact
    topology: tetra shell (2-sphere) beta_2=1; filled disk beta_1=0."""
    tetra = RexGraph.from_simplicial(
        np.array([0, 0, 0, 1, 1, 2]), np.array([1, 2, 3, 2, 3, 3]),
        np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))
    assert tetra.betti == (1, 0, 1)                 # hollow tetrahedron = 2-sphere
    # a single filled triangle (disk): beta_0=1, beta_1=0, beta_2=0
    disk = RexGraph.from_simplicial(np.array([0, 1, 0]), np.array([1, 2, 2]),
                                    np.array([[0, 1, 2]]))
    assert disk.betti == (1, 0, 0)


# -- live perturbation engines routed off dense eigh onto matrix-free Chebyshev ----

def _tetra_rex():
    return RexGraph.from_simplicial(
        np.array([0, 0, 0, 1, 1, 2]), np.array([1, 2, 3, 2, 3, 3]),
        np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))


def test_analyze_perturbation_is_eigenfree_and_matches_dense():
    """rex.analyze_perturbation now propagates via Chebyshev heat (no eigh(L1)); its
    trajectory matches the dense spectral mode-sum, and the full pipeline result is
    intact."""
    from rexgraph.core import _signal
    g = _tetra_rex()
    f_E = np.arange(g.nE, dtype=float) + 1.0
    times = np.linspace(0, 5, 30)
    res = g.analyze_perturbation(f_E, times=times)
    op = g.relational_laplacian
    opd = np.asarray(op.todense()) if hasattr(op, "todense") else np.asarray(op)
    w, V = np.linalg.eigh(opd)
    dense = _signal.propagate_diffusion(f_E, None, w, V, times)
    np.testing.assert_allclose(res["trajectory"], dense, atol=1e-10)
    assert {"E_kin", "bioes_tags", "hodge_final"} <= set(res)


def test_analyze_perturbation_field_diffusion_is_eigenfree():
    """rex.analyze_perturbation_field(mode='diffusion') evolves the graded field via
    matrix-free Chebyshev on the SPARSE M (no dense (nE+nF)² eigendecomposition) and
    matches the dense field trajectory; wave mode still runs."""
    from rexgraph.core import _field
    g = _tetra_rex()
    f_E = np.arange(g.nE, dtype=float) + 1.0
    f_F = np.zeros(g.nF_hodge)
    times = np.linspace(0, 4, 25)
    res = g.analyze_perturbation_field(f_E, f_F, times=times, mode="diffusion")
    M, _, _ = g.field_operator
    Md = np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)
    w, V = np.linalg.eigh(Md)
    F0 = np.concatenate([f_E, f_F])
    dense = _field.field_diffusion_trajectory(F0, w, V, times)
    np.testing.assert_allclose(res["field_trajectory"], dense, atol=1e-10)
    resw = g.analyze_perturbation_field(f_E, f_F, times=times, mode="wave")
    assert "field_trajectory" in resw


def test_analyze_perturbation_field_wave_is_eigenfree():
    """Wave-mode field perturbation now evolves positions AND velocities via matrix-free
    Chebyshev (cos(t√M) / -√M sin(t√M)) on the SPARSE M - no dense eigendecomposition -
    matching the dense wave_evolve_trajectory."""
    from rexgraph.core import _field
    g = _tetra_rex()
    f_E = np.arange(g.nE, dtype=float) + 1.0
    f_F = np.zeros(g.nF_hodge)
    times = np.linspace(0, 4, 25)
    res = g.analyze_perturbation_field(f_E, f_F, times=times, mode="wave")
    M, _, _ = g.field_operator
    Md = np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)
    w, V = np.linalg.eigh(Md)
    F0 = np.concatenate([f_E, f_F])
    pos, _ = _field.wave_evolve_trajectory(F0, w, V, np.sqrt(np.maximum(w, 0)), times)
    np.testing.assert_allclose(res["field_trajectory"], pos, atol=1e-10)
    assert "wave_total" in res


# --- dynamics-evolver reroutes: methods now matrix-free, no eigh ------------------

@pytest.mark.parametrize("edges", _COMPLEXES)
@pytest.mark.parametrize("t", [0.15, 0.9])
def test_evolve_markov_matches_dense_expm(edges, t):
    """rex.evolve_markov now applies e^{-tL} by Chebyshev on the SPARSE L; equals the
    dense expm markov_continuous_expm (no O(n^3) expm)."""
    from rexgraph.core import _transition
    g = _rex(edges)
    vec = np.arange(g.nE, dtype=float) + 1.0
    got = g.evolve_markov(vec, 1, t)
    ref = _transition.markov_continuous_expm(
        np.ascontiguousarray(vec), np.ascontiguousarray(np.asarray(g.L1, dtype=float)), t)
    np.testing.assert_allclose(got, ref, atol=1e-10)


@pytest.mark.parametrize("edges", _COMPLEXES)
@pytest.mark.parametrize("t", [0.15, 0.9])
def test_evolve_schrodinger_matches_dense_spectral(edges, t):
    """rex.evolve_schrodinger now applies e^{-iLt}=cos(tL)-i sin(tL) from one shared
    Chebyshev vector set on the SPARSE L; equals the dense mode-sum (no eigh)."""
    from rexgraph.core import _transition
    g = _rex(edges)
    psi = np.arange(g.nE, dtype=float) - 2.0
    re, im = g.evolve_schrodinger(psi, 1, t)
    ev, evec = np.linalg.eigh(np.asarray(g.L1, dtype=float))
    rre, rim = _transition.schrodinger_evolve_spectral(np.ascontiguousarray(psi), ev, evec, t)
    np.testing.assert_allclose(re, rre, atol=1e-10)
    np.testing.assert_allclose(im, rim, atol=1e-10)


@pytest.mark.parametrize("t", [0.2, 1.1])
def test_schrodinger_apply_matches_dense_modesum(t):
    """scale_propagator.schrodinger_apply(L, psi, t) == V diag(e^{-i lam t}) Vᵀ psi for
    both real and complex psi (the reusable unitary primitive)."""
    from rexgraph import scale_propagator as _spg
    from rexgraph.core import _wave
    g = _tetra_rex()
    L = g.L1_sparse
    Ld = np.asarray(L.todense())
    w, V = np.linalg.eigh(Ld)
    for psi in (np.arange(g.nE, dtype=float) + 1.0,
                (np.arange(g.nE) + 1.0) + 1j * (np.arange(g.nE) - 3.0)):
        got = _spg.schrodinger_apply(L, psi, t)
        ref = _wave.schrodinger_spectral(np.ascontiguousarray(psi, dtype=np.complex128), w, V, t)
        np.testing.assert_allclose(got, ref, atol=1e-10)


def test_evolve_field_wave_edge_and_face_are_eigenfree():
    """rex.evolve_field_wave / evolve_field_trajectory now evolve BOTH tiers matrix-free
    (e^{-i RL1 t} psi_E, e^{-i L2 t} psi_F) against the exact eigh of the sparse
    operators - the face tier is genuinely propagated (no longer frozen when the dense
    L2 spectrum happens to be absent)."""
    import scipy.sparse as sp

    from rexgraph import scale_propagator as _spg
    from rexgraph.core import _wave
    g = _tetra_rex()
    nE, nF = g.nE, int(g.nF_hodge)
    rng = np.random.default_rng(0)
    psi_E = rng.standard_normal(nE) + 1j * rng.standard_normal(nE)
    psi_F = rng.standard_normal(nF) + 1j * rng.standard_normal(nF)
    t = 0.55
    eE, eF, eV = g.evolve_field_wave(psi_E.copy(), psi_F.copy(), t)
    # exact reference: eigh of the SAME sparse operators the reroute uses
    RL1 = g.relational_laplacian
    RL1 = sp.csr_matrix(np.asarray(RL1)) if RL1 is not None else g.L1_sparse
    wr, Vr = np.linalg.eigh(np.asarray(RL1.todense()))
    wl, Vl = np.linalg.eigh(np.asarray(g.L2_sparse.todense()))
    rE = _wave.schrodinger_spectral(psi_E.copy(), wr, Vr, t)
    rF = _wave.schrodinger_spectral(psi_F.copy(), wl, Vl, t)
    rV = g.B1.dot(rE.real) + 1j * g.B1.dot(rE.imag)
    np.testing.assert_allclose(eE, rE, atol=1e-10)
    np.testing.assert_allclose(eF, rF, atol=1e-10)
    np.testing.assert_allclose(eV, rV, atol=1e-10)
    # trajectory form agrees timepoint-by-timepoint
    times = np.array([0.0, 0.3, 0.9])
    tE, tF, tV = g.evolve_field_trajectory(psi_E.copy(), psi_F.copy(), times)
    for i, tt in enumerate(times):
        np.testing.assert_allclose(tE[i], _spg.schrodinger_apply(RL1, psi_E.copy(), tt), atol=1e-10)
        np.testing.assert_allclose(tF[i], _spg.schrodinger_apply(g.L2_sparse, psi_F.copy(), tt), atol=1e-10)


@pytest.mark.parametrize("edges", _COMPLEXES)
def test_spectral_channel_score_is_eigenfree(edges):
    """rex.spectral_channel_score = sourceᵀ RL4⁺ target now via one block-CG solve
    (RL4 full-rank SPD ⇒ RL4⁺=RL4⁻¹); equals the dense eigenmode sum, no eigh."""
    from rexgraph.core import _channels
    rex = _rex(edges)
    rng = np.random.default_rng(2)
    src = rng.standard_normal(rex.nE)
    tgt = rng.standard_normal(rex.nE)
    got = rex.spectral_channel_score(src, tgt)
    ev, evec = rex._rl_eigen
    ref = _channels.spectral_channel_score(
        np.ascontiguousarray(src), np.ascontiguousarray(tgt), ev, evec, rex.nE)
    assert abs(got - ref) < 1e-9


@pytest.mark.parametrize("edges", _COMPLEXES + [
    [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],   # K4-ish, richer channels
])
def test_primal_signal_character_is_eigenfree(edges):
    """rex.primal_signal_character = [psiᵀ hat_X⁺ psi] now via LSQR pseudoinverse
    quadratic forms on the sparse hats; equals the dense per-channel eigenmode
    pseudoinverse (no hat_eigen bundle)."""
    from rexgraph.core import _channels
    rex = _rex(edges)
    rng = np.random.default_rng(4)
    psi = rng.standard_normal(rex.nE)
    got = rex.primal_signal_character(psi)
    hat_eigen = rex._hat_eigen_bundle
    ref = _channels.primal_signal_character(
        np.ascontiguousarray(psi), [h[0] for h in hat_eigen],
        [h[1] for h in hat_eigen], rex.nhats, rex.nE)
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_pinv_quadratic_form_matches_dense_pseudoinverse():
    """sparse_character.pinv_quadratic_form(A, v) = vᵀ A⁺ v for a SINGULAR PSD A via
    LSQR (projects off ker(A) exactly) == the dense eigenmode pseudoinverse."""
    from rexgraph.sparse_character import build_sparse_character_cheap, pinv_quadratic_form
    rex = _rex(_COMPLEXES[0])
    cheap = build_sparse_character_cheap(rex)
    rng = np.random.default_rng(9)
    v = rng.standard_normal(rex.nE)
    for hat in cheap['hats']:
        H = np.asarray(hat.todense())
        w, V = np.linalg.eigh(H)
        c = V.T @ v
        ref = float(np.sum(c[w > 1e-10] ** 2 / w[w > 1e-10]))
        got = pinv_quadratic_form(hat, v)
        assert abs(got - ref) < 1e-8


def test_primal_signal_character_kernel_guards_arity():
    """Regression: the _channels kernel is compiled boundscheck=False; if nhats disagrees with the
    number of hat arrays it used to read past the list and segfault. It must raise instead."""
    from rexgraph.core import _channels
    with pytest.raises(ValueError):
        _channels.primal_signal_character(np.zeros(4), [], [], 4, 4)   # nhats=4, zero hats


def test_sparse_betti_is_arity_aware_for_branching_hyperedges():
    """Regression: the sparse bundle-slot Betti used a one-source/one-target column reader that
    assumed arity-2 and overcounted beta0 on branching hyperedges. It must equal exact .betti."""
    import scipy.sparse as sp

    from rexgraph.core._laplacians import _sparse_betti
    # {0,1,2} is a branching (arity-3) 1-cell; {2,3},{3,4},{4,2} form a triangle
    ptr = np.array([0, 3, 5, 7, 9]); idx = np.array([0, 1, 2, 2, 3, 3, 4, 4, 2])
    g = RexGraph.from_hypergraph(ptr, idx)
    exact = tuple(int(b) for b in g.betti[:3])
    B1 = sp.csr_matrix(np.asarray(g.B1, dtype=float))
    b0, b1, b2, rank_B1, rank_B2 = _sparse_betti(B1, None, g.nV, g.nE, g.nF)
    assert (b0, b1, b2) == exact          # arity-aware: matches the exact path exactly


def test_alpha_G_is_exact_c2_on_complete_graphs():
    """alpha_G = c^2 = G/T = tr((B2 B2^T)^2)/tr((B1^T B1)^2) is the CANONICAL geometry<->topology
    exchange rate, exact rational = (k-2)/2 on the autofaced complete graph K_k (CANONICAL_RESOLUTION
    section 2). Replaces the outdated fiedler(L1)/fiedler(L_O) float ratio. Also checks RL_1 = L1_down
    + alpha_G*L1_up and that E_kin + alpha_G*E_pot = <f|RL_1|f>."""
    import itertools
    for k in (4, 5, 6, 7, 8):
        E = list(itertools.combinations(range(k), 2))
        Tr = np.array([list(x) for x in itertools.combinations(range(k), 3)], dtype=np.int64)
        g = RexGraph.from_simplicial(np.array([e[0] for e in E], np.int32),
                                     np.array([e[1] for e in E], np.int32), Tr)
        b = g.spectral_bundle
        assert abs(b['alpha_G'] - (k - 2) / 2) < 1e-9, (k, b['alpha_G'])
        # edge-space operators are built on demand on the scale-free path (bundle keys
        # are None); the accessors give the same L1_down / L1_up / RL_1 = L1_down + c^2 L1_up.
        Ld, Lu = np.asarray(g.L1_down), np.asarray(g.L1_up)
        RL1 = np.asarray(g.relational_laplacian)
        assert np.allclose(RL1, Ld + b['alpha_G'] * Lu, atol=1e-9)
        f = np.random.default_rng(k).standard_normal(g.nE)
        ek, ep, _ = g.energy_kin_pot(f)
        assert abs((ek + b['alpha_G'] * ep) - float(f @ (RL1 @ f))) < 1e-8   # energy matches operator


def test_sparse_alpha_G_matches_dense_cheap_exact():
    """cheap+exact alpha_G = c^2 = G/T on the scale-free path equals the dense oracle (both are the
    exact integer-trace exchange rate, no eigensolve). Was NaN in sparse mode before."""
    import rexgraph.core._common as common
    saved = common.get_algorithm_config()['eigen_dense_limit']
    try:
        for edges in _COMPLEXES + [[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]]:
            common.configure_algorithms(eigen_dense_limit=2000)
            dense = _rex(edges).spectral_bundle.get('alpha_G')
            common.configure_algorithms(eigen_dense_limit=1)
            sparse = _rex(edges).spectral_bundle.get('alpha_G')   # cheap+exact fill
            if np.isnan(dense):
                assert np.isnan(sparse)
            else:
                assert abs(dense - sparse) < 1e-9, (edges, dense, sparse)
    finally:
        common.configure_algorithms(eigen_dense_limit=saved)


def test_sparse_edge_fiedler_eigenpair_matches_dense():
    """cheap+exact edge Fiedler eigenpair (value + vector, DEDICATED keys) on the scale-free path
    matches the dense oracle; the vector satisfies L1 v = lambda v."""
    import scipy.sparse as sp

    import rexgraph.core._common as common
    from rexgraph.core._sparse import to_scipy_csr
    saved = common.get_algorithm_config()['eigen_dense_limit']
    try:
        for edges in _COMPLEXES + [[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]]:
            # the L1 Fiedler eigenpair is a LAZY accessor (edge_fiedler), not eagerly in the bundle
            # (that ARPACK solve was the dominant cost of a large-hive monitor step). Verify it is a
            # true eigenpair of L1 and equals the smallest nonzero eigenvalue of the dense L1.
            gs = _rex(edges)
            lam, v = gs.edge_fiedler
            assert abs(gs.fiedler_val_L1 - lam) < 1e-12
            B1 = to_scipy_csr(gs._B1_dual).astype(float); L1 = B1.T @ B1
            if gs._nF > 0:
                B2 = to_scipy_csr(gs._B2_hodge_dual).astype(float); L1 = L1 + B2 @ B2.T
            v = np.asarray(v)
            assert np.linalg.norm(sp.csr_matrix(L1) @ v - lam * v) / (np.linalg.norm(v) + 1e-12) < 1e-6
            w = np.linalg.eigvalsh(np.asarray(L1.todense() if sp.issparse(L1) else L1))
            pos = w[w > 1e-9]
            assert abs(lam - (pos.min() if pos.size else 0.0)) < 1e-6
    finally:
        common.configure_algorithms(eigen_dense_limit=saved)


def test_scale_free_never_fills_full_spectrum_keys_with_partial():
    """Regression: the scale-free (sparse) fill must NOT put PARTIAL low modes under the
    FULL-spectrum keys. A full-eigenbasis consumer (measure_in_eigenbasis) reads those keys and
    would silently truncate. In sparse mode they stay None, and the consumer rebuilds the SAME full
    RL_1 = L1_down + c^2 * L1_up on demand, so the operator it measures over is identical to the
    dense path's.

    (measure_in_eigenbasis itself ends in np.random.choice, so its returned probability is a random
    draw and cannot be compared across two decompositions -- and RL_1 here is degenerate, so two
    eigenbases need not agree mode-for-mode. The deterministic, degeneracy-robust invariant is that
    the OPERATOR and its full spectrum are the same either way.)"""
    import itertools

    import rexgraph.core._common as common
    saved = common.get_algorithm_config()['eigen_dense_limit']
    try:
        E = list(itertools.combinations(range(6), 2))
        Tr = np.array([list(x) for x in itertools.combinations(range(6), 3)], dtype=np.int64)
        args = (np.array([e[0] for e in E], np.int32), np.array([e[1] for e in E], np.int32), Tr)
        common.configure_algorithms(eigen_dense_limit=2000)
        gd = RexGraph.from_simplicial(*args)
        common.configure_algorithms(eigen_dense_limit=1)
        gs = RexGraph.from_simplicial(*args); nE = gs.nE
        # (a) sparse mode never stores PARTIAL vectors under a full-spectrum key
        for key in ('evecs_RL_1', 'evecs_L1', 'evecs_L2'):
            v = gs.spectral_bundle.get(key)
            assert v is None or np.asarray(v).shape[1] == nE   # full or absent, never partial
        # (b) the operator measure_in_eigenbasis(dim=1) uses is the SAME full RL_1 either way:
        #     dense reads bundle['RL_1']; sparse rebuilds L1_down + c^2 * L1_up on demand.
        RL_dense = np.asarray(gd.relational_laplacian)
        RL_sparse = np.asarray(gs.relational_laplacian)
        assert RL_dense.shape == (nE, nE)
        assert np.allclose(RL_dense, RL_sparse, atol=1e-9)
        # full spectra agree (sorted eigenvalues are operator invariants, robust to degeneracy)
        wd = np.linalg.eigvalsh(RL_dense); ws = np.linalg.eigvalsh(RL_sparse)
        assert wd.shape == (nE,) and ws.shape == (nE,)
        assert np.allclose(wd, ws, atol=1e-9)
    finally:
        common.configure_algorithms(eigen_dense_limit=saved)


class TestSingularGreensDeflated:
    """diag(L+) for a SINGULAR edge operator via harmonic-projector deflation
    L+ = (L + P_H)^-1 - P_H (oracle 09). The plain greens_diagonal (SPD RL4) blows up
    on a kernel; the deflated form must match the dense pseudoinverse to ~1e-9 using
    only the combinatorial harmonic/cycle basis (no eigendecomposition)."""

    def _l1_down(self, g):
        import scipy.sparse as sp
        B1 = np.asarray(g.B1)
        return sp.csr_matrix(B1.T @ B1)

    def test_matches_dense_pinv_two_squares(self):
        # oracle-09 complex: two 4-cycles sharing an edge, no faces, beta1 = 2
        from rexgraph.harmonic_sparse import cycle_basis
        from rexgraph.scale_propagator import greens_diagonal_deflated
        E = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5), (5, 2)]
        g = RexGraph(sources=np.array([e[0] for e in E], np.int32),
                     targets=np.array([e[1] for e in E], np.int32))
        L1 = self._l1_down(g)
        H = cycle_basis(g)
        assert H.shape[1] == 2
        gd = greens_diagonal_deflated(L1, H)
        ref = np.diag(np.linalg.pinv(L1.toarray()))
        assert np.allclose(gd, ref, atol=1e-8)

    def test_triangle_beta1_one(self):
        from rexgraph.harmonic_sparse import cycle_basis
        from rexgraph.scale_propagator import greens_diagonal_deflated
        g = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32))
        L1 = self._l1_down(g)
        gd = greens_diagonal_deflated(L1, cycle_basis(g))
        assert np.allclose(gd, np.diag(np.linalg.pinv(L1.toarray())), atol=1e-8)

    def test_full_rank_reduces_to_inverse(self):
        # empty kernel basis -> diag(L^-1); equals greens_diagonal on the SPD RL4
        from rexgraph.scale_propagator import greens_diagonal, greens_diagonal_deflated
        from rexgraph.sparse_character import build_sparse_character_cheap
        g = RexGraph.from_simplicial(
            np.array([0, 0, 0, 1, 1, 2], np.int32), np.array([1, 2, 3, 2, 3, 3], np.int32),
            np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32))
        RL = build_sparse_character_cheap(g)['RL']
        assert np.allclose(greens_diagonal_deflated(RL, None), greens_diagonal(RL), atol=1e-9)

    def test_graph_greens_character_edge(self):
        # public accessor: diag(L1+) for the edge Laplacian, unfilled and filled
        g = RexGraph(sources=np.array([0, 1, 2, 3, 0, 1, 4, 5], np.int32),
                     targets=np.array([1, 2, 3, 0, 4, 4, 5, 2], np.int32))
        B1 = np.asarray(g.B1)
        assert np.allclose(g.greens_character_edge, np.diag(np.linalg.pinv(B1.T @ B1)), atol=1e-8)

    def test_branching_edge_greens_finite(self):
        g = RexGraph.from_hypergraph(np.array([0, 3, 6, 9]), np.array([0, 1, 2, 1, 2, 3, 3, 4, 0]))
        gc = g.greens_character_edge
        assert gc.shape == (g.nE,) and np.all(np.isfinite(gc)) and np.all(gc >= -1e-9)


class TestMalaughActionMoment:
    """The Malaugh action<->moment calculus (oracle 16): per-complex edge moments
    X(k) (all O(nnz) traces, no eigendecomposition), and the discrete calculus
    moment DX(k)=X(k+1)-X(k) / action S(k)=sum_{j<=k}X(j) conjugate by the scale-FTC."""

    def _Kk(self, k):
        import itertools
        E = list(itertools.combinations(range(k), 2))
        Tr = np.array([list(x) for x in itertools.combinations(range(k), 3)], np.int64)
        return RexGraph.from_simplicial(np.array([e[0] for e in E], np.int32),
                                        np.array([e[1] for e in E], np.int32), Tr)

    def test_tower_quantities_match_oracle_on_Kk(self):
        from rexgraph.scale_propagator import malaugh_quantities
        for k in range(3, 9):
            q = malaugh_quantities(self._Kk(k))
            assert abs(q['c2_E'] - (k - 2) / 2) < 1e-9       # energy coupling = (k-2)/2
            assert abs(q['H_T'] - np.log(k - 1)) < 1e-9      # harmonic-log = log(k-1)

    def test_action_moment_ftc_conjugacy(self):
        from rexgraph.scale_propagator import action_moment, malaugh_quantities
        HT = np.array([malaugh_quantities(self._Kk(k))['H_T'] for k in range(3, 10)])
        am = action_moment(HT)
        assert am['moment'].shape == (HT.shape[0] - 1,)
        assert am['action'].shape == HT.shape
        # FTC: differencing the action returns the tower; the moment telescopes back
        assert np.allclose(np.diff(am['action']), HT[1:])
        assert np.allclose(HT[0] + np.cumsum(np.r_[0.0, am['moment']]), HT)

    def test_entropy_action_converges(self):
        # the topological entropy cost |DH_T| shrinks across the tower (harmonic converges)
        from rexgraph.scale_propagator import action_moment, malaugh_quantities
        HT = np.array([malaugh_quantities(self._Kk(k))['H_T'] for k in range(3, 12)])
        m = np.abs(action_moment(HT)['moment'])
        assert np.all(np.diff(m) < 0)                        # strictly decreasing increments

    def test_action_moment_vector_tower(self):
        # per-step vector towers difference/accumulate along axis 0
        from rexgraph.scale_propagator import action_moment
        X = np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 9.0]])
        am = action_moment(X)
        assert np.allclose(am['moment'], np.array([[2., 3.], [3., 4.]]))
        assert np.allclose(am['action'], np.array([[1., 2.], [4., 7.], [10., 16.]]))


class TestChannelSpectralGaps:
    """The exact per-channel spectral-gap METRIC lambda_2. T/G use the transpose duality
    (exact vertex-dual Fiedler); it must match the dense hat eigen-gap, and drive an
    exact per_channel_mixing_times."""

    def _Kk(self, k):
        import itertools
        E = list(itertools.combinations(range(k), 2))
        Tr = np.array([list(x) for x in itertools.combinations(range(k), 3)], np.int64)
        return RexGraph.from_simplicial(np.array([e[0] for e in E], np.int32),
                                        np.array([e[1] for e in E], np.int32), Tr)

    def test_gaps_match_dense_hat_eigen(self):
        from rexgraph.core import _character
        for k in (4, 5, 6, 7):
            g = self._Kk(k)
            gaps = g.channel_spectral_gaps
            db = g._dense_rcf_bundle
            for i, nm in enumerate(db['hat_names']):
                evals, _ = _character.hat_eigen(db['hats'][i], g.nE)
                lam2_dense = float(evals[evals > 1e-10].min())
                assert abs(gaps[nm] - lam2_dense) < 1e-7, (k, nm)

    def test_mixing_times_are_ln_nE_over_gap(self):
        g = self._Kk(6)
        gaps = g.channel_spectral_gaps
        mt = g.per_channel_mixing_times
        for i, nm in enumerate(g.hat_names):
            assert abs(mt[i] - np.log(g.nE) / gaps[nm]) < 1e-6

    def test_transpose_duality_exact_T_G(self):
        # T/G gap == lambda_2 of the tiny vertex-dual Laplacian / trace (exact)
        g = self._Kk(6)
        B1 = np.asarray(g.B1, float)
        for nm, M in (('L1_down', B1 @ B1.T), ('L_O', np.abs(B1) @ np.abs(B1).T)):
            w = np.linalg.eigvalsh(M); lam2 = w[w > 1e-9].min()
            trX = np.trace((B1.T @ B1) if nm == 'L1_down' else (np.abs(B1).T @ np.abs(B1)))
            assert abs(g.channel_spectral_gaps[nm] - lam2 / trX) < 1e-9


class TestRelaxationMomentTower:
    """The edge-centric relaxation entry point = the moment tower (not the Fiedler
    metric). Bundled quantities are consistent with their standalone accessors and
    finite on branching hyperedges."""

    def test_relaxation_bundle_consistent(self):
        g = RexGraph.from_simplicial(
            np.array([0, 0, 0, 1, 1, 2], np.int32), np.array([1, 2, 3, 2, 3, 3], np.int32),
            np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32))
        r = g.relaxation
        assert abs(r['harmonic_log'] - g.harmonic_entropy) < 1e-12
        assert abs(r['effective_modes'] - np.exp(g.harmonic_entropy)) < 1e-9
        assert np.allclose(r['energy_character'], g.energy_character)
        assert np.allclose(r['greens_edge'], g.greens_character_edge)
        assert r['effective_modes'] >= 1.0 - 1e-9

    def test_relaxation_finite_on_branching(self):
        g = RexGraph.from_hypergraph(np.array([0, 3, 6, 9]), np.array([0, 1, 2, 0, 1, 3, 2, 3, 4]))
        r = g.relaxation
        assert np.isfinite(r['harmonic_log']) and np.isfinite(r['effective_modes'])
        assert np.all(np.isfinite(r['energy_character'])) and np.all(np.isfinite(r['greens_edge']))
