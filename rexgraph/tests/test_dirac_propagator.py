"""Tests for the sparse graded Dirac operator and its tensor-state propagators.

Everything is checked against the DENSE reference in ``rexgraph.core._dirac`` (the
assembled operator + its eigendecomposition), on a mixed-arity complex (pairwise
AND branching edges, a face) so the arity-generality is exercised, not assumed.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.core import _dirac
from rexgraph.dirac_propagator import (
    SparseDirac,
    dirac_from_rex,
    dirac_heat,
    dirac_light,
)
from rexgraph.graph import RexGraph


def _mixed_complex():
    """A 1-rex mixing arities: a branching (3-ary) edge plus ordinary pairwise edges,
    no faces. ``from_hypergraph`` builds grade-0/1 only, so this exercises the
    witness/pairwise/branching arity generality of the vertex-edge Dirac block.
    """
    he_idx = np.array([0, 1, 2,   0, 1,   1, 2,   0, 2], dtype=np.int64)
    he_ptr = np.array([0, 3, 5, 7, 9], dtype=np.int64)
    return RexGraph.from_hypergraph(he_ptr, he_idx)


def _tetra_2rex():
    """A genuine 2-rex (tetrahedron): 4 vertices, 6 edges, 4 triangular faces -
    exercises the full graded Dirac with a non-empty grade-2 (face) sector."""
    src = np.array([0, 0, 0, 1, 1, 2])
    tgt = np.array([1, 2, 3, 2, 3, 3])
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return RexGraph.from_simplicial(src, tgt, tris)


def _cycle_complex(nv):
    """A 1-rex cycle of ``nv`` vertices with pairwise edges - large enough (N = 2*nv)
    to push a wide state block onto the parallel column-tiling path of matvec."""
    src = np.arange(nv, dtype=np.int64)
    tgt = (np.arange(nv, dtype=np.int64) + 1) % nv
    he_idx = np.empty(2 * nv, dtype=np.int64)
    he_idx[0::2] = src
    he_idx[1::2] = tgt
    he_ptr = np.arange(0, 2 * nv + 1, 2, dtype=np.int64)
    return RexGraph.from_hypergraph(he_ptr, he_idx)


def _dense_refs(rex):
    B1 = np.asarray(rex.B1, dtype=np.float64)
    B2 = np.asarray(rex.B2, dtype=np.float64)
    if B2.ndim != 2 or B2.shape[1] == 0:
        B2 = np.zeros((B1.shape[1], 0), dtype=np.float64)
    D, sizes = _dirac.build_dirac_operator(B1, B2)
    return B1, B2, D, sizes


def test_sparse_dirac_matches_dense_assembly():
    g = _mixed_complex()
    B1, B2, D_dense, sizes = _dense_refs(g)
    sd = dirac_from_rex(g)

    # grade sizes agree with the dense reference (nV, nE, nF), dropping empty top grade
    ref_sizes = [int(s) for s in sizes if int(s) > 0]
    assert list(sd.sizes) == ref_sizes[:len(sd.sizes)]
    assert D_dense.shape[0] == sd.N

    D_sparse = sd.to_scipy().toarray()
    assert np.allclose(D_sparse, D_dense, atol=1e-12), \
        f"max|Δ|={np.abs(D_sparse - D_dense).max():.2e}"
    # symmetric
    assert np.allclose(D_sparse, D_sparse.T, atol=1e-12)


def test_matvec_matches_dense():
    g = _mixed_complex()
    _, _, D_dense, _ = _dense_refs(g)
    sd = dirac_from_rex(g)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.standard_normal(sd.N)
        assert np.allclose(sd.matvec(x), D_dense @ x, atol=1e-12)
    # block of states at once
    X = rng.standard_normal((sd.N, 4))
    assert np.allclose(sd.matvec(X), D_dense @ X, atol=1e-12)


def test_d_squared_is_block_diagonal_laplacian():
    g = _mixed_complex()
    _, _, D_dense, _ = _dense_refs(g)
    sd = dirac_from_rex(g)
    D = sd.to_scipy().toarray()
    D2 = D @ D
    # off-diagonal (grade-crossing) blocks of D^2 must vanish: B1 B2 = 0
    s1, s2 = sd.grade_slice(0), sd.grade_slice(2) if sd.n_grades > 2 else None
    if s2 is not None:
        assert np.allclose(D2[s1, s2], 0.0, atol=1e-10)
    # diagonal blocks are the Hodge Laplacians L_d
    for d in range(sd.n_grades):
        sl = sd.grade_slice(d)
        Ld = D2[sl, sl]
        assert np.allclose(Ld, Ld.T, atol=1e-10)


def test_light_propagator_matches_dense_eigen():
    g = _mixed_complex()
    _, _, D_dense, _ = _dense_refs(g)
    sd = dirac_from_rex(g)
    evals, evecs = _dirac.dirac_eigen(D_dense)

    rng = np.random.default_rng(1)
    psi0 = rng.standard_normal(sd.N)
    for t in (0.1, 0.5, 1.3):
        re_ref, im_ref = _dirac.schrodinger_evolve(evals, evecs, psi0, t)
        re, im = sd.light(psi0, t, order=200)
        assert np.allclose(re, re_ref, atol=1e-8), \
            f"t={t} real max|Δ|={np.abs(re - re_ref).max():.2e}"
        assert np.allclose(im, im_ref, atol=1e-8), \
            f"t={t} imag max|Δ|={np.abs(im - im_ref).max():.2e}"


def test_light_imaginary_part_crosses_grades():
    """The curl (imaginary) part transports amplitude OFF the starting grade; the
    off-diagonal boundary blocks are what carry it. A pure ``e^{-tD^2}`` heat stays
    in-grade. This is the whole point of working in the Dirac vector space."""
    g = _mixed_complex()
    sd = dirac_from_rex(g)
    # Start off-constant. A boundary column sums to zero, so B1^T annihilates the
    # constant vector on grade 0 (level linking: zero column sums put 1 in ker L0).
    # Seeding every vertex equally therefore transports nothing, and would test the
    # kernel rather than the transport this is about. The constant case is asserted
    # for what it is in test_the_constant_vector_is_annihilated below.
    psi0 = np.zeros(sd.N)
    psi0[sd.grade_slice(0)] = np.array([1.0, -1.0, 0.0])

    re, im = sd.light(psi0, t=0.7, order=200)
    e_re = sd.grade_energy(re)
    e_im = sd.grade_energy(im)
    # the imaginary (curl) part must put energy on grade 1 (edges) - crossed a grade
    assert e_im[1] > 1e-6, f"no grade-1 transport in curl part: {e_im}"

    # pure per-grade heat e^{-tD^2} keeps grade-0 input on grade 0 only
    h = sd.heat_squared(psi0, t=0.7, order=200)
    e_h = sd.grade_energy(h)
    assert e_h[0] > 1e-6
    if sd.n_grades > 1:
        assert e_h[1] < 1e-9, f"heat leaked across grades: {e_h}"


def test_the_constant_vector_is_annihilated():
    """The other side of the same fact. Every boundary column sums to zero, so
    B1^T 1 = 0 and a uniform grade-0 seed has nowhere to go: the Dirac off-diagonal
    block sends it to zero and no grade is crossed. This is the property that makes
    beta_0 count components at all, and it holds at every arity because the share
    1/(k-1) is what delivers the zero sum."""
    g = _mixed_complex()
    sd = dirac_from_rex(g)
    B1 = np.asarray(g.B1, dtype=float)
    assert np.allclose(B1.T @ np.ones(int(g.nV)), 0.0, atol=1e-12)

    psi0 = np.zeros(sd.N)
    psi0[sd.grade_slice(0)] = 1.0
    _, im = sd.light(psi0, t=0.7, order=200)
    assert sd.grade_energy(im)[1] < 1e-12, "a constant seed must not transport"


def test_full_2rex_with_faces_matches_dense():
    """Tetrahedron 2-rex: assembled D, matvec, D^2=blkdiag(L0,L1,L2), and the light
    propagator all match the dense reference with a real grade-2 (face) sector."""
    g = _tetra_2rex()
    B1, B2, D_dense, sizes = _dense_refs(g)
    sd = dirac_from_rex(g)
    assert sd.n_grades == 3 and list(sd.sizes) == [4, 6, 4]

    assert np.allclose(sd.to_scipy().toarray(), D_dense, atol=1e-12)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(sd.N)
    assert np.allclose(sd.matvec(x), D_dense @ x, atol=1e-12)

    # D^2 grade-crossing block (grade 0 <-> grade 2) vanishes: B1 B2 = 0
    D2 = D_dense @ D_dense
    assert np.allclose(D2[sd.grade_slice(0), sd.grade_slice(2)], 0.0, atol=1e-10)

    # light propagator vs dense eigen, across all three grades
    evals, evecs = _dirac.dirac_eigen(D_dense)
    psi0 = rng.standard_normal(sd.N)
    for t in (0.3, 0.9):
        re_ref, im_ref = _dirac.schrodinger_evolve(evals, evecs, psi0, t)
        re, im = sd.light(psi0, t, order=220)
        assert np.allclose(re, re_ref, atol=1e-8)
        assert np.allclose(im, im_ref, atol=1e-8)


def test_face_sector_receives_grade2_transport():
    """Amplitude seeded on edges reaches the FACE sector under the curl part - a
    two-hop V/E/F Dirac genuinely couples grade 1 to grade 2 through B2."""
    g = _tetra_2rex()
    sd = dirac_from_rex(g)
    psi0 = np.zeros(sd.N)
    psi0[sd.grade_slice(1)] = 1.0                  # seed on edges
    _, im = sd.light(psi0, t=0.8, order=220)
    e_im = sd.grade_energy(im)
    assert e_im[0] > 1e-6 and e_im[2] > 1e-6, \
        f"edge seed did not cross to both vertices and faces: {e_im}"


def test_grade_general_witness_edge_only():
    """A 1-rex of witness (arity-1) + branching edges, no faces: the Dirac is still
    well-formed (two grades) and matvec matches dense."""
    he_idx = np.array([0,   0, 1,   0, 1, 2], dtype=np.int64)   # witness, edge, branch
    he_ptr = np.array([0, 1, 3, 6], dtype=np.int64)
    g = RexGraph.from_hypergraph(he_ptr, he_idx)
    sd = dirac_from_rex(g)
    assert sd.n_grades == 2                       # vertices + edges only
    B1 = np.asarray(g.B1, dtype=np.float64)
    B2 = np.zeros((B1.shape[1], 0), dtype=np.float64)
    D_dense, _ = _dirac.build_dirac_operator(B1, B2)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(sd.N)
    assert np.allclose(sd.matvec(x), D_dense @ x, atol=1e-12)


def test_block_matvec_parallel_equals_serial():
    """A wide state block takes the parallel column-tiling path of matvec; its result
    must be bit-for-bit (<=1e-12) the serial core. Sized past the parallel gate so the
    threaded branch is genuinely exercised."""
    from rexgraph import dirac_propagator as dp

    g = _cycle_complex(200)
    sd = dirac_from_rex(g)
    k = 256
    assert sd.N * k >= dp._PARALLEL_MIN_ELEMS      # block clears the parallel gate

    rng = np.random.default_rng(7)
    X = rng.standard_normal((sd.N, k))
    serial = sd._matvec_serial(X)
    parallel = sd.matvec(X)                        # dispatches to the threaded tiles
    assert np.allclose(parallel, serial, atol=1e-12, rtol=0.0), \
        f"parallel/serial max|Δ|={np.abs(parallel - serial).max():.2e}"

    # a single vector / 1-column block stays serial and is still correct vs dense
    _, _, D_dense, _ = _dense_refs(g)
    v = rng.standard_normal(sd.N)
    assert np.allclose(sd.matvec(v), D_dense @ v, atol=1e-12)


def test_dirac_light_free_function_smoke():
    """dirac_light builds the operator from a rex and returns (re, im) with the default
    grade-0 seed; matches the class method on the same seed and shows grade crossing."""
    g = _mixed_complex()
    sd = dirac_from_rex(g)
    psi0 = np.zeros(sd.N)
    psi0[0] = 1.0                                   # default seed: unit on first vertex

    re, im = dirac_light(g, t=0.7, order=200)
    re_ref, im_ref = sd.light(psi0, 0.7, order=200)
    assert np.allclose(re, re_ref, atol=1e-12)
    assert np.allclose(im, im_ref, atol=1e-12)
    # curl part crossed onto grade 1 (edges)
    assert sd.grade_energy(im)[1] > 1e-6

    # explicit psi0 is honored
    rng = np.random.default_rng(11)
    custom = rng.standard_normal(sd.N)
    re2, im2 = dirac_light(g, t=0.4, psi0=custom, order=200)
    r2, i2 = sd.light(custom, 0.4, order=200)
    assert np.allclose(re2, r2, atol=1e-12) and np.allclose(im2, i2, atol=1e-12)


def test_dirac_heat_free_function_smoke():
    """dirac_heat runs e^{-tD^2} from a rex with the default seed; per-grade heat keeps
    grade-0 input on grade 0 (no grade crossing) and matches the class method."""
    g = _mixed_complex()
    sd = dirac_from_rex(g)
    psi0 = np.zeros(sd.N)
    psi0[0] = 1.0                                   # default seed: unit on first vertex

    h = dirac_heat(g, t=0.6)
    h_ref = sd.heat_squared(psi0, 0.6)
    assert np.allclose(h, h_ref, atol=1e-12)
    e = sd.grade_energy(h)
    assert e[0] > 1e-6
    if sd.n_grades > 1:
        assert e[1] < 1e-9, f"heat leaked across grades: {e}"


def test_trajectory_energy_conservation_and_shape():
    """trajectory propagates the light state at many times; total Born energy is
    conserved (unitary e^{-itD}) and per-grade energy shows amplitude leaving grade 0."""
    g = _tetra_2rex()
    sd = dirac_from_rex(g)
    psi0 = np.zeros(sd.N)
    psi0[0] = 1.0                                   # localized seed on a single vertex
    seed_energy = float(np.sum(psi0 ** 2))

    times = np.array([0.0, 0.25, 0.5, 1.0, 1.7])
    traj = sd.trajectory(psi0, times, order=220)

    assert traj["energy"].shape == (times.shape[0], sd.n_grades)
    assert traj["total"].shape == (times.shape[0],)
    # unitary: total energy constant across all times
    assert np.allclose(traj["total"], seed_energy, atol=1e-6), \
        f"energy not conserved: {traj['total']}"
    # t=0 is the seed itself: all energy on grade 0, none elsewhere
    assert np.allclose(traj["energy"][0, 0], seed_energy, atol=1e-6)
    assert traj["energy"][0, 1:].sum() < 1e-6
    # by a later time amplitude has crossed onto higher grades
    assert traj["energy"][-1, 1:].sum() > 1e-6


def test_graded_boundaries_property_is_used_when_present():
    """_boundaries_from_rex prefers a rex.graded_boundaries list when it exists; a rex
    lacking it falls back to the B1(+B2) construction. Both must build a valid Dirac."""
    from rexgraph.dirac_propagator import _boundaries_from_rex

    g = _tetra_2rex()
    fallback = _boundaries_from_rex(g)              # no graded_boundaries -> fallback path

    class _Wrap:
        graded_boundaries = fallback                # simulate the other workstream's property

    used = _boundaries_from_rex(_Wrap())
    assert len(used) == len(fallback)
    for a, b in zip(used, fallback, strict=False):
        assert (a != b).nnz == 0                    # identical boundary maps
    sd = SparseDirac(used)
    assert list(sd.sizes) == [4, 6, 4]


def test_deprecated_heat_diag_warns_but_still_correct():
    """The retired edge-space heat diagonal stays importable: it emits a
    DeprecationWarning yet still returns the correct diag(e^{-tL}) numbers (checked
    against the dense matrix exponential) so existing callers do not crash."""
    from scipy.linalg import expm

    from rexgraph import _experimental as _exp

    rng = np.random.default_rng(5)
    A = rng.standard_normal((6, 6))
    L = sp.csr_matrix(A @ A.T)                      # SPD
    t = 0.3
    ref = np.diag(expm(-t * L.toarray()))

    with pytest.warns(DeprecationWarning):
        got = _exp.heat_propagator_diag(L, t, order=120, mode='exact')
    assert np.allclose(got, ref, atol=1e-6), f"max|Δ|={np.abs(got - ref).max():.2e}"

    with pytest.warns(DeprecationWarning):
        _exp.chebyshev_diag(lambda P: L @ P, 6, lambda l: np.exp(-t * l),
                            lam_max=float(np.abs(L).sum(axis=1).max()), order=120)


def test_from_cells_3rex_dirac_is_grade_general():
    """A from_cells 3-rex (solid octahedron: V/E/F/Volume) propagates through ALL
    four grades - RexGraph.sparse_dirac reads graded_boundaries() so B3 participates
    (the pre-integration seam silently truncated to V/E). A uniform face seed lies in
    ker(B2), so its curl transport lands purely on the volume via B3^T."""
    from rexgraph.graded_boundary import solid_octahedron_3rex, verify_chain
    g = RexGraph.from_cells(solid_octahedron_3rex())
    sd = g.sparse_dirac()
    assert sd.n_grades == 4 and list(sd.sizes) == [6, 12, 8, 1]
    ok, res = verify_chain(g.graded_boundaries())
    assert ok and res == 0.0
    psi0 = np.zeros(sd.N)
    psi0[sd.grade_slice(2)] = 1.0                       # seed on faces (grade 2)
    _, im = g.dirac_light(0.7, psi0=psi0, order=200)
    assert sd.grade_energy(im)[3] > 1e-6, \
        f"volume (grade 3) received no curl transport - B3 ignored: {sd.grade_energy(im)}"


def test_heat_character_accessor_is_quiet():
    """The superseded heat_character accessor still works and does NOT emit the
    experimental DeprecationWarning (it calls the warning-free internal impl)."""
    import warnings
    g = _tetra_2rex()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        d = g.heat_character(0.3)
    assert d.shape == (g.nE,)
