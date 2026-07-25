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
    from rexgraph.core import _wave
    from rexgraph import scale_propagator as _spg
    import scipy.sparse as sp
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
    from rexgraph.sparse_character import pinv_quadratic_form, build_sparse_character_cheap
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
